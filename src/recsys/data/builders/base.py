from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset


@dataclass
class DatasetBundle:
    """Container for datasets and metadata produced by a builder."""

    full: Dataset
    train: Dataset
    val: Dataset
    feature_map: dict[str, int]
    metadata: dict[str, Any] | None = None


class DatasetBuilder(ABC):
    """Abstract dataset builder interface."""

    @abstractmethod
    def build(self) -> DatasetBundle:
        raise NotImplementedError
