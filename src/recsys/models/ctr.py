from __future__ import annotations

import torch
from torch import nn
from recsys.utils import MODEL_REGISTRY


@MODEL_REGISTRY.register("deepfm")
class DeepFM(nn.Module):
    def __init__(
        self, feature_map: dict[str, int], embed_dim: int, mlp_dims: list[int]
    ):
        """
        Args:
            feature_map: dict[str, int] - mapping of feature names to their vocabulary sizes
            embed_dim: int - dimension of the embedding vectors
            mlp_dims: list[int] - dimensions of the MLP layers
        """
        super().__init__()
        # extract data dimensions
        n_fields = len(feature_map)
        total_dim = sum(feature_map.values())

        # Embedding layer
        self.cat_embed = nn.Embedding(total_dim, embed_dim)
        # Factorization machine layer
        self.o1_fc = nn.Embedding(total_dim, 1)
        # DNN layer
        self.dnn_mlp = nn.Sequential(
            nn.Linear(n_fields * embed_dim, mlp_dims[0]),
            *[
                nn.Sequential(
                    nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, mlp_dims[i + 1])
                )
                for i, dim in enumerate(mlp_dims[:-1])
            ],
            nn.Linear(mlp_dims[-1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_fields)
        Returns:
            logits: (B, 1)
        """
        # 1. embedding
        cat_embeddings = self.cat_embed(x)  # shape: (B, n_fields, embed_dim)
        # 2. factorization machine
        o1_output = torch.sum(self.o1_fc(x), dim=1)  # shape: (B, 1)
        square_of_sums = torch.sum(cat_embeddings, dim=1) ** 2  # (B, embed_dim)
        sum_of_squares = torch.sum(cat_embeddings**2, dim=1)  # (B, embed_dim)
        o2_output = 0.5 * torch.sum(
            square_of_sums - sum_of_squares, dim=1, keepdim=True
        )  # (B, 1)
        # 3. DNN
        dnn_output = self.dnn_mlp(cat_embeddings.view(x.shape[0], -1))  # (B, 1)
        # 4. combine all
        logits = o1_output + o2_output + dnn_output  # (B, 1)

        return logits
