from __future__ import annotations

import torch
from torch import nn


def build_mlp(input_dim: int, hidden_dims: list[int], dropout: float = 0.0) -> nn.Sequential:
    """
    Build a Multi-Layer Perceptron (MLP).

    Args:
        input_dim: Dimension of the input features.
        hidden_dims: List of dimensions for the hidden layers.
        dropout: Dropout probability.

    Returns:
        A Sequential module representing the MLP.
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
