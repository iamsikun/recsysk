from __future__ import annotations

import torch
from torch import nn
from recsys.algorithms.torch._mlp import build_mlp
from recsys.utils import ALGO_REGISTRY


class LocalActivationUnit(nn.Module):
    """
    Local Activation Unit used in Deep Interest Network (DIN).

    This unit calculates the attention weights between the query item (target item)
    and the user behavior history items. It captures the user's diverse interests
    with respect to the target item.

    Reference:
        Zhou, Guorui, et al. "Deep interest network for click-through rate prediction."
        KDD 2018.
    """

    def __init__(self, embed_dim: int, hidden_dims: list[int], dropout: float = 0.0):
        """
        Args:
            embed_dim: Dimension of the input embeddings.
            hidden_dims: List of hidden layer dimensions for the attention MLP.
            dropout: Dropout probability.
        """
        super().__init__()
        # Input to the MLP is [query, key, query-key, query*key], so 4 * embed_dim
        self.mlp = build_mlp(embed_dim * 4, hidden_dims, dropout)
        output_dim = hidden_dims[-1] if hidden_dims else embed_dim * 4
        self.out = nn.Linear(output_dim, 1)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Local Activation Unit.

        Args:
            query: Tensor of shape (B, E) representing the target item embedding.
            keys: Tensor of shape (B, T, E) representing the history item embeddings.
            mask: Optional boolean tensor of shape (B, T) indicating valid history items (True for valid).

        Returns:
            Tensor of shape (B, E) representing the weighted sum of history embeddings (interest vector).
        """
        # query: (B, E), keys: (B, T, E)
        batch_size, seq_len, _ = keys.shape

        # Expand query to match sequence length: (B, T, E)
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)

        # Construct attention input: [keys, query, keys-query, keys*query]
        # Shape: (B, T, 4*E)
        attn_input = torch.cat(
            [keys, query_expanded, keys - query_expanded, keys * query_expanded],
            dim=-1,
        )

        # Flatten for MLP: (B*T, 4*E)
        x = attn_input.view(batch_size * seq_len, -1)

        # Pass through MLP
        x = self.mlp(x)

        # Calculate scores: (B, T)
        scores = self.out(x).view(batch_size, seq_len)

        if mask is not None:
            # Mask padded items with a very small number before softmax
            scores = scores.masked_fill(~mask, -1e9)

        # Calculate attention weights
        weights = torch.softmax(scores, dim=-1)  # (B, T)

        # Weighted sum of keys (history embeddings): (B, 1, T) x (B, T, E) -> (B, 1, E)
        output = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)

        return output


@ALGO_REGISTRY.register("din")
class DeepInterestNetwork(nn.Module):
    """
    Deep Interest Network (DIN) for CTR prediction.

    DIN introduces a local activation unit to adaptively learn the representation
    of user interests from historical behaviors with respect to a candidate item.

    Reference:
        Zhou, Guorui, et al. "Deep interest network for click-through rate prediction."
        KDD 2018.
    """

    def __init__(
        self,
        feature_map: dict[str, int],
        embed_dim: int,
        mlp_dims: list[int],
        item_feature: str = "item_id",
        history_feature: str = "hist_item_id",
        sparse_feature_names: list[str] | None = None,
        dense_dim: int = 0,
        attention_mlp_dims: list[int] | None = None,
        dropout: float = 0.0,
        streams: list[dict] | None = None,
        feature_specs: list | None = None,
    ):
        """
        Args:
            feature_map: Mapping of feature names to their vocabulary sizes.
            embed_dim: Embedding dimension for sparse features.
            mlp_dims: Hidden dimensions for the final prediction MLP.
            item_feature: Name of the target item feature.
            history_feature: Name of the history feature (sequence of item
                ids). Ignored when ``streams`` is provided.
            sparse_feature_names: List of names of additional sparse features.
            dense_dim: Number of dense feature dimensions (if any).
            attention_mlp_dims: Hidden dimensions for the attention MLP.
            dropout: Dropout probability used in MLPs.
            streams: Optional list of parallel history stream configs. When
                present, DIN runs in multi-stream mode: one embedding table
                + one attention unit per stream, outputs concatenated. Each
                dict must have ``name``, ``history_feature``, and either
                ``vocab_size`` or a ``feature_map`` lookup name. The first
                stream is the "primary" one whose embedding also scores
                the target item. Single-stream backcompat is preserved
                when ``streams`` is None.
            feature_specs: Accepted but currently unused; injected by the
                runner uniformly across algorithms.
        """
        super().__init__()
        del feature_specs  # not consumed directly; streams carry what we need
        sparse_feature_names = sparse_feature_names or []
        attention_mlp_dims = attention_mlp_dims or [64, 32]

        if item_feature not in feature_map:
            raise ValueError(f"Missing item feature '{item_feature}' in feature_map")

        # Normalize into internal stream configs. Single-stream default
        # preserves the pre-multi-stream batch layout exactly.
        if streams is None:
            resolved_streams = [
                {
                    "name": "item",
                    "history_feature": history_feature,
                    "vocab_size": int(feature_map[item_feature]),
                }
            ]
        else:
            resolved_streams = []
            for i, s in enumerate(streams):
                if "name" not in s or "history_feature" not in s:
                    raise ValueError(
                        f"stream {i} needs 'name' and 'history_feature' keys"
                    )
                vocab = s.get("vocab_size")
                if vocab is None:
                    lookup_name = s.get("feature_map_key", s["history_feature"])
                    if lookup_name not in feature_map:
                        raise ValueError(
                            f"stream '{s['name']}' needs vocab_size or a "
                            f"feature_map entry under '{lookup_name}'"
                        )
                    vocab = int(feature_map[lookup_name])
                resolved_streams.append(
                    {
                        "name": s["name"],
                        "history_feature": s["history_feature"],
                        "vocab_size": int(vocab),
                    }
                )

        self.item_feature = item_feature
        self.history_feature = history_feature  # back-compat reference
        self.sparse_feature_names = sparse_feature_names
        self.dense_dim = dense_dim
        self._streams = resolved_streams
        # The target item is scored against the first stream's vocab
        # (that's the "canonical item" stream).
        self._primary_stream_name = resolved_streams[0]["name"]

        # One embedding + one attention unit per stream.
        self.stream_embeddings = nn.ModuleDict(
            {
                s["name"]: nn.Embedding(s["vocab_size"], embed_dim)
                for s in resolved_streams
            }
        )
        self.stream_attentions = nn.ModuleDict(
            {
                s["name"]: LocalActivationUnit(
                    embed_dim, attention_mlp_dims, dropout
                )
                for s in resolved_streams
            }
        )

        self.sparse_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(feature_map[name], embed_dim)
                for name in sparse_feature_names
            }
        )

        # Final MLP input = sparse (N*E) + target (E) + K interest vecs (K*E) + dense
        n_streams = len(resolved_streams)
        mlp_input_dim = (
            embed_dim * (len(sparse_feature_names) + 1 + n_streams) + dense_dim
        )
        self.mlp = build_mlp(mlp_input_dim, mlp_dims, dropout)
        output_dim = mlp_dims[-1] if mlp_dims else mlp_input_dim
        self.output = nn.Linear(output_dim, 1)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        In single-stream mode the batch dict is unchanged from the
        pre-refactor layout (``item_feature``, ``history_feature``,
        ``<history_feature>_mask``). In multi-stream mode each stream
        consumes its own ``history_feature`` key plus the matching
        ``_mask``.
        """
        target_item = x.get("target_item", x.get(self.item_feature))
        if target_item is None:
            raise ValueError("DIN requires a target item input")

        # Target is scored via the primary stream's embedding table.
        primary_embed = self.stream_embeddings[self._primary_stream_name]
        item_emb = primary_embed(target_item)  # (B, E)

        interest_vecs: list[torch.Tensor] = []
        for stream in self._streams:
            hist_key = stream["history_feature"]
            history_items = x.get(hist_key)
            if history_items is None and stream["name"] == "item":
                # Legacy alias.
                history_items = x.get("history_items")
            if history_items is None:
                raise ValueError(
                    f"DIN stream '{stream['name']}' needs history tensor "
                    f"at batch key '{hist_key}'"
                )
            mask_key = f"{hist_key}_mask"
            history_mask = x.get(mask_key)
            if history_mask is None and stream["name"] == "item":
                history_mask = x.get("history_mask")
            if history_mask is not None:
                history_mask = history_mask.bool()

            history_emb = self.stream_embeddings[stream["name"]](history_items)
            interest_vec = self.stream_attentions[stream["name"]](
                item_emb, history_emb, history_mask
            )
            interest_vecs.append(interest_vec)

        features: list[torch.Tensor] = []

        if self.sparse_feature_names:
            sparse_tensor = x.get("sparse_features")
            if sparse_tensor is None:
                raise ValueError(
                    "sparse_features input required for DIN but not provided in batch"
                )
            if sparse_tensor.shape[1] != len(self.sparse_feature_names):
                raise ValueError(
                    f"sparse_features column count ({sparse_tensor.shape[1]}) "
                    f"does not match config ({len(self.sparse_feature_names)})"
                )
            sparse_embs = [
                self.sparse_embeddings[name](sparse_tensor[:, idx])
                for idx, name in enumerate(self.sparse_feature_names)
            ]
            features.append(torch.cat(sparse_embs, dim=-1))

        features.append(item_emb)
        features.extend(interest_vecs)

        if self.dense_dim > 0:
            dense_tensor = x.get("dense_features")
            if dense_tensor is not None:
                features.append(dense_tensor)

        mlp_input = torch.cat(features, dim=-1)
        hidden = self.mlp(mlp_input)
        return self.output(hidden)
