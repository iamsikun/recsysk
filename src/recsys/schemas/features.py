from __future__ import annotations

from enum import Enum
from dataclasses import dataclass


class FeatureType(Enum):
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"


class FeatureRole(Enum):
    USER = "user"
    ITEM = "item"
    CONTEXT = "context"
    SEQUENCE = "sequence"
    GROUP = "group"
    LABEL = "label"


@dataclass
class FeatureSpec:
    name: str  # Name of the feature in the dataset
    source_name: str  # Name of the feature in the source dataset
    type: FeatureType  # Type of the feature
    role: FeatureRole  # Semantic role of the feature

    # Only used for categorical features
    vocab_size: int | None = None

    # Preprocessing params
    fill_value: float = 0  # Fill value for missing values

    # Only used when role == FeatureRole.GROUP
    group_id: str | None = None
