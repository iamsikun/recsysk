from __future__ import annotations

import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """
    A generic dataset for tabular data where features and labels are pre-converted to tensors.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        """
        Args:
            X: Feature tensor.
            y: Label tensor.
        """
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SequenceDataset(Dataset):
    """Dataset that returns dict-style model inputs for sequence models."""

    def __init__(self, features: dict[str, torch.Tensor], labels: torch.Tensor):
        """
        Args:
            features: Dictionary of feature tensors. keys are feature names.
            labels: Tensor of labels.
        """
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.features.items()}, self.labels[
            idx
        ]


class TabularDictDataset(Dataset):
    """Dict-valued tabular dataset, shape-compatible with SequenceDataset.

    Used by :func:`recsys.data.transforms.tabular.build_tabular_dataset`
    when a feature set contains non-scalar columns (``dense_vector`` or
    ``multi_categorical``) that can't be packed into a single flat
    ``(B, n_fields)`` tensor. Rows are ``(dict[str, Tensor], label)``.
    """

    def __init__(self, features: dict[str, torch.Tensor], labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            {key: value[idx] for key, value in self.features.items()},
            self.labels[idx],
        )
