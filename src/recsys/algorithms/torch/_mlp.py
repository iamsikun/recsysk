"""Shared MLP builder used by torch algorithms (DeepFM, DIN, ...)."""

from __future__ import annotations

from torch import nn


def build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    dropout: float = 0.0,
) -> nn.Sequential:
    """Build a stack of Linear -> ReLU (-> Dropout) layers.

    Args:
        input_dim: Dimension of the input features.
        hidden_dims: List of dimensions for the hidden layers.
        dropout: Dropout probability (0.0 disables).

    Returns:
        A ``nn.Sequential`` representing the MLP. The last layer is the
        final hidden layer — callers are responsible for adding the
        projection head.
    """
    layers: list[nn.Module] = []
    dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        dim = hidden_dim
    return nn.Sequential(*layers)
