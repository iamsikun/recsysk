from __future__ import annotations

from enum import Enum
from dataclasses import dataclass


class FeatureType(Enum):
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    DENSE_VECTOR = "dense_vector"
    MULTI_CATEGORICAL = "multi_categorical"


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

    # Only used for categorical / multi_categorical features
    vocab_size: int | None = None

    # Preprocessing params
    fill_value: float = 0  # Fill value for missing values

    # Only used when role == FeatureRole.GROUP
    group_id: str | None = None

    # Only used when type == FeatureType.DENSE_VECTOR: fixed width of the
    # pretrained embedding column.
    vector_dim: int | None = None

    # Only used when type == FeatureType.MULTI_CATEGORICAL: pad/truncate
    # list columns to this length. 0 is reserved for padding.
    max_len: int | None = None

    # Only used when type == FeatureType.MULTI_CATEGORICAL: when True the
    # encoder also emits a parallel ``<name>_weight`` float column so the
    # downstream algo can do a weighted embedding-bag lookup.
    weighted: bool = False

    # Only used when role == FeatureRole.SEQUENCE: names the behavior
    # stream this history column belongs to (e.g. "action", "item",
    # "content"). Lets a model declare per-stream attention heads without
    # hard-coding feature names. None means "single-stream, legacy layout".
    stream: str | None = None
