from __future__ import annotations

import torch
from torch import nn
from recsys.utils import MODEL_REGISTRY
from recsys.models.utils import build_mlp


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


@MODEL_REGISTRY.register("din")
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
    ):
        """
        Args:
            feature_map: Mapping of feature names to their vocabulary sizes.
            embed_dim: Embedding dimension for sparse features.
            mlp_dims: Hidden dimensions for the final prediction MLP.
            item_feature: Name of the target item feature.
            history_feature: Name of the history feature (sequence of item ids).
            sparse_feature_names: List of names of additional sparse features (e.g., user_id, context).
            dense_dim: Number of dense feature dimensions (if any).
            attention_mlp_dims: Hidden dimensions for the attention MLP in LocalActivationUnit.
            dropout: Dropout probability used in MLPs.
        """
        super().__init__()
        sparse_feature_names = sparse_feature_names or []
        attention_mlp_dims = attention_mlp_dims or [64, 32]

        if item_feature not in feature_map:
            raise ValueError(f"Missing item feature '{item_feature}' in feature_map")

        self.item_feature = item_feature
        self.history_feature = history_feature
        self.sparse_feature_names = sparse_feature_names
        self.dense_dim = dense_dim

        # Target item embedding
        self.item_embedding = nn.Embedding(feature_map[item_feature], embed_dim)
        
        # Other sparse feature embeddings
        self.sparse_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(feature_map[name], embed_dim)
                for name in sparse_feature_names
            }
        )
        
        # Attention layer for user history
        self.attention = LocalActivationUnit(embed_dim, attention_mlp_dims, dropout)

        # Input dimension for the final MLP:
        # - Sparse features embeddings (len(sparse_feature_names) * embed_dim)
        # - Target item embedding (embed_dim)
        # - User interest vector (embed_dim) from attention
        # - Dense features (dense_dim)
        mlp_input_dim = embed_dim * (len(sparse_feature_names) + 2) + dense_dim
        
        self.mlp = build_mlp(mlp_input_dim, mlp_dims, dropout)
        
        output_dim = mlp_dims[-1] if mlp_dims else mlp_input_dim
        self.output = nn.Linear(output_dim, 1)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Dictionary containing input tensors:
                - x[item_feature] or x["target_item"]: (B,) Target item indices.
                - x[history_feature] or x["history_items"]: (B, T) User history item indices.
                - x[f"{history_feature}_mask"] or x["history_mask"]: (B, T) Boolean mask for history (optional).
                - x["sparse_features"]: (B, N) Tensor of other sparse feature indices (optional).
                - x["dense_features"]: (B, D) Tensor of dense features (optional).

        Returns:
            Logits of shape (B, 1).
        """
        target_item = x.get("target_item", x.get(self.item_feature))
        history_items = x.get("history_items", x.get(self.history_feature))
        history_mask = x.get(
            "history_mask", x.get(f"{self.history_feature}_mask")
        )

        if target_item is None or history_items is None:
            raise ValueError("DIN requires target item and history item inputs")

        if history_mask is not None:
            history_mask = history_mask.bool()

        # 1. Embed target item and history items
        item_emb = self.item_embedding(target_item)  # (B, E)
        history_emb = self.item_embedding(history_items)  # (B, T, E)

        # 2. Calculate user interest vector via attention
        interest_vec = self.attention(item_emb, history_emb, history_mask)  # (B, E)

        features: list[torch.Tensor] = []
        
        # 3. Process other sparse features
        if self.sparse_feature_names:
            sparse_tensor = x.get("sparse_features")
            if sparse_tensor is None:
                raise ValueError("sparse_features input required for DIN but not provided in batch")
            if sparse_tensor.shape[1] != len(self.sparse_feature_names):
                raise ValueError(
                    f"sparse_features column count ({sparse_tensor.shape[1]}) "
                    f"does not match config ({len(self.sparse_feature_names)})"
                )
            
            sparse_embs = []
            for idx, name in enumerate(self.sparse_feature_names):
                sparse_embs.append(self.sparse_embeddings[name](sparse_tensor[:, idx]))
            
            # Concatenate all sparse embeddings: (B, N*E)
            features.append(torch.cat(sparse_embs, dim=-1))

        # 4. Concatenate all features
        features.append(item_emb)
        features.append(interest_vec)

        if self.dense_dim > 0:
            dense_tensor = x.get("dense_features")
            if dense_tensor is not None:
                features.append(dense_tensor)

        mlp_input = torch.cat(features, dim=-1)
        
        # 5. Final MLP and prediction
        hidden = self.mlp(mlp_input)
        logits = self.output(hidden)
        
        return logits
