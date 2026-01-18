from __future__ import annotations

from enum import Enum
from dataclasses import dataclass


class FeatureType(Enum):
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"


@dataclass
class FeatureSpec:
    name: str  # Name of the feature in the dataset
    source_name: str  # Name of the feature in the source dataset
    type: FeatureType  # Type of the feature

    # Only used for categorical features
    vocab_size: int | None = None

    # Preprocessing params
    fill_value: float = 0  # Fill value for missing values
