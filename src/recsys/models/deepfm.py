from __future__ import annotations

import torch
from torch import nn
from recsys.utils import MODEL_REGISTRY


@MODEL_REGISTRY.register("deepfm")
class DeepFM(nn.Module):
    """
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction.
    
    It combines the power of factorization machines for recommendation and deep learning
    for feature learning in a new neural network architecture.
    
    Reference:
        Guo, Huifeng, et al. "DeepFM: a factorization-machine based neural network for CTR prediction."
        IJCAI 2017.
    """

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

        # Embedding layer for both FM (partially) and DNN
        self.cat_embed = nn.Embedding(total_dim, embed_dim)
        
        # Factorization machine layer: 1st order term linear weights
        self.o1_fc = nn.Embedding(total_dim, 1)
        
        # DNN layer (Deep part)
        # Note: DeepFM typically uses Batch Normalization
        layers = [nn.Linear(n_fields * embed_dim, mlp_dims[0])]
        for i, dim in enumerate(mlp_dims[:-1]):
            layers.append(nn.BatchNorm1d(dim)) # Apply BN before activation usually, or after. Original paper uses BN.
            layers.append(nn.ReLU())
            layers.append(nn.Linear(dim, mlp_dims[i + 1]))
        
        layers.append(nn.Linear(mlp_dims[-1], 1))
        
        self.dnn_mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, n_fields) containing feature indices.
            
        Returns:
            logits: Output tensor of shape (B, 1).
        """
        # 1. embedding lookup
        cat_embeddings = self.cat_embed(x)  # shape: (B, n_fields, embed_dim)
        
        # 2. Factorization Machine (FM) Component
        
        # First Order Term: sum(w_i * x_i)
        # Since x_i are 1 (categorical encoded), this is just sum(embedding_weights)
        o1_output = torch.sum(self.o1_fc(x), dim=1)  # shape: (B, 1)
        
        # Second Order Term: 0.5 * sum((sum(v_i x_i))^2 - sum((v_i x_i)^2))
        # Interaction between embeddings
        square_of_sums = torch.sum(cat_embeddings, dim=1) ** 2  # (B, embed_dim)
        sum_of_squares = torch.sum(cat_embeddings**2, dim=1)  # (B, embed_dim)
        o2_output = 0.5 * torch.sum(
            square_of_sums - sum_of_squares, dim=1, keepdim=True
        )  # (B, 1)
        
        # 3. Deep Component (DNN)
        # Flatten embeddings: (B, n_fields * embed_dim)
        dnn_input = cat_embeddings.view(x.shape[0], -1)
        dnn_output = self.dnn_mlp(dnn_input)  # (B, 1)
        
        # 4. Combine all components
        logits = o1_output + o2_output + dnn_output  # (B, 1)

        return logits
