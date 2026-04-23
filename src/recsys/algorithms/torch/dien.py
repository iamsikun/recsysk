"""Deep Interest Evolution Network (DIEN).

Reference
---------
Zhou, Guorui, et al. "Deep interest evolution network for click-through
rate prediction." AAAI 2019.

DIEN extends DIN with:

1. An **interest extractor** GRU over the history-item embeddings, whose
   hidden states are the per-step interest representations.
2. An **auxiliary loss** that supervises each hidden state ``h_t`` to
   distinguish the next clicked behavior ``e_{t+1}`` from a randomly
   sampled negative item (paper Eq. 6). The module exposes the scalar on
   ``self.last_aux_loss`` so :class:`recsys.engine.CTRTask` can pick it
   up via the opt-in hook; non-DIEN models leave the attribute unset.
3. An **interest evolving layer** — :class:`DynamicAUGRU` — whose per-
   step update gate is scaled by a DIN-style softmax attention score
   between each GRU hidden state and the target-item embedding. The
   final AUGRU state is concatenated with target / sparse / dense
   features and fed into a prediction MLP.

Batch shape
-----------
Same as DIN: dict batches of ``{item_id, hist_item_id, hist_item_id_mask,
user_id, ...}`` — produced by
:func:`recsys.data.transforms.sequence.build_sequence_dataset`. DIEN is
registered under the key ``"dien"``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from recsys.algorithms.torch._augru import DynamicAUGRU
from recsys.algorithms.torch._mlp import build_mlp
from recsys.utils import ALGO_REGISTRY


@ALGO_REGISTRY.register("dien")
class DeepInterestEvolutionNetwork(nn.Module):
    """DIEN (Zhou et al. 2019) for CTR prediction."""

    def __init__(
        self,
        feature_map: dict[str, int],
        embed_dim: int,
        mlp_dims: list[int],
        gru_hidden: int | None = None,
        item_feature: str = "item_id",
        history_feature: str = "hist_item_id",
        sparse_feature_names: list[str] | None = None,
        dense_dim: int = 0,
        attention_mlp_dims: list[int] | None = None,
        dropout: float = 0.0,
        feature_specs: list | None = None,
    ) -> None:
        """
        Args:
            feature_map: Mapping of feature names to vocabulary sizes
                (injected by the runner).
            embed_dim: Embedding dimension for sparse features.
            mlp_dims: Hidden dimensions for the final prediction MLP.
            gru_hidden: Hidden size shared by the interest-extractor GRU
                and the AUGRU. Defaults to ``embed_dim`` so the AUGRU's
                output lands in the same space as the target embedding.
            item_feature: Name of the target item feature (batch-dict key
                and ``feature_map`` key).
            history_feature: Batch-dict key holding the history tensor.
                The matching mask lives at ``f"{history_feature}_mask"``.
            sparse_feature_names: Extra scalar sparse features included
                in the prediction MLP (e.g. ``["user_id"]``).
            dense_dim: Number of dense-feature dims, if any.
            attention_mlp_dims: Hidden dims for the target-vs-hidden-
                state attention MLP. Defaults to ``[64, 32]`` matching
                DIN.
            dropout: Dropout probability used in MLPs.
            feature_specs: Accepted but unused — injected uniformly by
                the runner.
        """
        super().__init__()
        del feature_specs  # not consumed directly
        sparse_feature_names = sparse_feature_names or []
        attention_mlp_dims = attention_mlp_dims or [64, 32]
        gru_hidden = gru_hidden if gru_hidden is not None else embed_dim

        if item_feature not in feature_map:
            raise ValueError(
                f"Missing item feature '{item_feature}' in feature_map"
            )

        self.item_feature = item_feature
        self.history_feature = history_feature
        self.sparse_feature_names = list(sparse_feature_names)
        self.dense_dim = int(dense_dim)
        self.item_vocab_size = int(feature_map[item_feature])
        self.embed_dim = int(embed_dim)
        self.gru_hidden = int(gru_hidden)

        # Shared item-embedding table for target, history, aux-loss
        # positives, and in-batch aux-loss negatives. ``padding_idx=0``
        # keeps the pad slot at a fixed zero vector (the SequenceCtrBuilder
        # reserves 0 for pre-pad, see its +1 id shift).
        self.item_embedding = nn.Embedding(
            self.item_vocab_size, embed_dim, padding_idx=0
        )

        self.sparse_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(int(feature_map[name]), embed_dim)
                for name in sparse_feature_names
            }
        )

        # Interest extractor: a single-layer GRU over history embeddings.
        self.interest_gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=self.gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        # DIN-style attention between each GRU hidden state and the
        # target item embedding. The target is projected to gru_hidden
        # so the attention query/key share one space.
        self.target_to_hidden = (
            nn.Linear(embed_dim, self.gru_hidden)
            if embed_dim != self.gru_hidden
            else nn.Identity()
        )
        # DIN's LocalActivationUnit returns the attention-weighted sum;
        # we need the raw per-step softmax scores to gate AUGRU, so we
        # use a local scorer module with the same MLP shape.
        self._attn_scorer = _AttentionScorer(
            self.gru_hidden, attention_mlp_dims, dropout
        )

        # Interest evolving: AUGRU over the GRU's hidden states.
        self.augru = DynamicAUGRU(
            input_size=self.gru_hidden, hidden_size=self.gru_hidden
        )

        # Final MLP: concat [h'_T, target_emb, sparse_embs, (dense)].
        mlp_input_dim = (
            self.gru_hidden
            + embed_dim
            + embed_dim * len(sparse_feature_names)
            + self.dense_dim
        )
        self.mlp = build_mlp(mlp_input_dim, mlp_dims, dropout)
        output_dim = mlp_dims[-1] if mlp_dims else mlp_input_dim
        self.output = nn.Linear(output_dim, 1)

        # Opt-in aux-loss hook consumed by CTRTask.training_step. None
        # when the model is in eval mode or the batch has no valid
        # (t, t+1) pairs.
        self.last_aux_loss: torch.Tensor | None = None

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        target_item = x.get("target_item", x.get(self.item_feature))
        if target_item is None:
            raise ValueError("DIEN requires a target item input")

        history_items = x.get(self.history_feature)
        if history_items is None:
            # Legacy alias compat with the pre-multi-stream DIN layout.
            history_items = x.get("history_items")
        if history_items is None:
            raise ValueError(
                f"DIEN requires history tensor at batch key "
                f"'{self.history_feature}'"
            )
        mask_key = f"{self.history_feature}_mask"
        history_mask = x.get(mask_key)
        if history_mask is None:
            history_mask = x.get("history_mask")
        if history_mask is not None:
            history_mask = history_mask.bool()

        # --- Embeddings ---
        item_emb = self.item_embedding(target_item)  # (B, E)
        hist_emb = self.item_embedding(history_items)  # (B, T, E)

        # --- Interest extractor GRU over history ---
        interest_states, _ = self.interest_gru(hist_emb)  # (B, T, H)

        # --- Attention: target vs each hidden state ---
        # DIN-style attention computes a softmax over time steps.
        target_proj = self.target_to_hidden(item_emb)  # (B, H)
        att_scores = self._attn_scorer(
            target_proj, interest_states, history_mask
        )  # (B, T)

        # --- Interest evolving: AUGRU gated by attention scores ---
        final_interest = self.augru(
            interest_states, att_scores, history_mask
        )  # (B, H)

        # --- Auxiliary loss (train mode, history has a usable next-step) ---
        if self.training:
            self.last_aux_loss = self._auxiliary_loss(
                interest_states, history_items, history_mask
            )
        else:
            self.last_aux_loss = None

        # --- Final MLP ---
        features: list[torch.Tensor] = [final_interest, item_emb]

        if self.sparse_feature_names:
            sparse_tensor = x.get("sparse_features")
            if sparse_tensor is None:
                raise ValueError(
                    "sparse_features input required for DIEN but not "
                    "provided in batch"
                )
            if sparse_tensor.shape[1] != len(self.sparse_feature_names):
                raise ValueError(
                    f"sparse_features column count ({sparse_tensor.shape[1]}) "
                    f"does not match config ({len(self.sparse_feature_names)})"
                )
            for idx, name in enumerate(self.sparse_feature_names):
                features.append(self.sparse_embeddings[name](sparse_tensor[:, idx]))

        if self.dense_dim > 0:
            dense_tensor = x.get("dense_features")
            if dense_tensor is not None:
                features.append(dense_tensor)

        mlp_input = torch.cat(features, dim=-1)
        hidden = self.mlp(mlp_input)
        return self.output(hidden)

    def _auxiliary_loss(
        self,
        interest_states: torch.Tensor,
        history_items: torch.Tensor,
        history_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Binary NLL over (h_t, e_{t+1}_pos) vs (h_t, e_{t+1}_neg) pairs.

        Returns ``None`` when no valid (t, t+1) pair exists in the batch.
        """
        batch_size, seq_len, _ = interest_states.shape
        if seq_len < 2:
            return None

        # h_t for t in [0, T-2], predicts behavior at t+1.
        h_src = interest_states[:, :-1, :]  # (B, T-1, H)
        pos_ids = history_items[:, 1:]  # (B, T-1)

        if history_mask is not None:
            pair_mask = history_mask[:, :-1] & history_mask[:, 1:]
        else:
            pair_mask = torch.ones_like(pos_ids, dtype=torch.bool)

        if not pair_mask.any():
            return None

        # Sample one random negative per (b, t) position from [1, vocab).
        neg_ids = torch.randint(
            low=1,
            high=self.item_vocab_size,
            size=pos_ids.shape,
            device=pos_ids.device,
        )

        pos_emb = self.item_embedding(pos_ids)  # (B, T-1, E)
        neg_emb = self.item_embedding(neg_ids)  # (B, T-1, E)

        # If gru_hidden != embed_dim, interest states and item embeddings
        # live in different spaces — project the states to embed_dim for
        # the aux-loss dot product. Reusing target_to_hidden would invert
        # the direction; instead, use the transpose of its weight when
        # dims differ. Simplest: only scoring via dot product when dims
        # match is fine for the default (gru_hidden == embed_dim).
        if h_src.shape[-1] != pos_emb.shape[-1]:
            # Project items up to gru_hidden space.
            pos_emb = self.target_to_hidden(pos_emb)
            neg_emb = self.target_to_hidden(neg_emb)

        pos_logits = (h_src * pos_emb).sum(dim=-1)  # (B, T-1)
        neg_logits = (h_src * neg_emb).sum(dim=-1)

        pair_mask_f = pair_mask.float()
        n_valid = pair_mask_f.sum().clamp(min=1.0)

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits,
            torch.ones_like(pos_logits),
            reduction="none",
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits,
            torch.zeros_like(neg_logits),
            reduction="none",
        )
        total = (pos_loss + neg_loss) * pair_mask_f
        return total.sum() / n_valid


class _AttentionScorer(nn.Module):
    """Thin adapter over :class:`LocalActivationUnit` that returns the
    softmax attention scores (one per time step) rather than the
    attention-weighted sum vector.

    The arithmetic is identical to DIN's unit — we just need the scores
    so AUGRU can consume them. Sharing the unit is not ideal because
    :class:`LocalActivationUnit.forward` already consumes its own
    output via ``torch.bmm`` — pulling the intermediate out would
    require rewriting its forward. So we replicate the MLP here.
    """

    def __init__(self, embed_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        self.mlp = build_mlp(embed_dim * 4, hidden_dims, dropout)
        output_dim = hidden_dims[-1] if hidden_dims else embed_dim * 4
        self.out = nn.Linear(output_dim, 1)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return (B, T) softmax attention scores over time steps."""
        batch_size, seq_len, _ = keys.shape
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)
        attn_input = torch.cat(
            [keys, query_expanded, keys - query_expanded, keys * query_expanded],
            dim=-1,
        )
        x = self.mlp(attn_input.view(batch_size * seq_len, -1))
        scores = self.out(x).view(batch_size, seq_len)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        return torch.softmax(scores, dim=-1)
