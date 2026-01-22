from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import torch

from recsys.data.datasets import SequenceDataset


@dataclass(frozen=True)
class SequenceSpec:
    """Configuration for building sequence features from event data."""

    item_feature: str
    history_feature: str
    max_history_len: int
    sparse_feature_names: list[str]
    dense_feature_names: list[str]


def validate_sequence_spec(
    df: pl.DataFrame,
    feature_map: dict[str, int],
    spec: SequenceSpec,
) -> None:
    if spec.item_feature not in df.columns:
        raise ValueError(f"item_feature '{spec.item_feature}' not found in data")
    if spec.item_feature not in feature_map:
        raise ValueError(
            f"item_feature '{spec.item_feature}' must be included in features"
        )
    if "user_id" not in df.columns:
        raise ValueError("Sequence data requires a 'user_id' column")
    if spec.item_feature in spec.sparse_feature_names:
        raise ValueError("item_feature should not be included in sparse_feature_names")
    missing_sparse = [
        name for name in spec.sparse_feature_names if name not in df.columns
    ]
    if missing_sparse:
        raise ValueError(f"Missing sparse feature columns: {missing_sparse}")
    missing_dense = [name for name in spec.dense_feature_names if name not in df.columns]
    if missing_dense:
        raise ValueError(f"Missing dense feature columns: {missing_dense}")
    if spec.max_history_len <= 0:
        raise ValueError("max_history_len must be > 0")


def build_sequence_dataset(
    df: pl.DataFrame,
    feature_map: dict[str, int],
    spec: SequenceSpec,
) -> SequenceDataset:
    """Build sequence inputs from a ratings dataframe."""
    validate_sequence_spec(df, feature_map, spec)

    sort_cols: list[str] = ["user_id"]
    if "timestamp" in df.columns:
        sort_cols.append("timestamp")
    df = df.sort(sort_cols)

    user_ids = df["user_id"].to_numpy()
    item_ids = df[spec.item_feature].to_numpy()
    labels = df["label"].to_numpy().astype("float32")

    sparse_arrays = {
        name: df[name].to_numpy() for name in spec.sparse_feature_names
    }
    dense_arrays = {
        name: df[name].to_numpy().astype("float32")
        for name in spec.dense_feature_names
    }

    target_items: list[int] = []
    history_items: list[list[int]] = []
    history_masks: list[list[bool]] = []
    sparse_rows: list[list[int]] = []
    dense_rows: list[list[float]] = []
    labels_out: list[float] = []

    current_user = None
    user_hist: list[int] = []
    for idx in range(len(user_ids)):
        user_id = user_ids[idx]
        if current_user != user_id:
            current_user = user_id
            user_hist = []

        if not user_hist:
            user_hist.append(int(item_ids[idx]))
            continue

        hist = user_hist[-spec.max_history_len :]
        pad_len = spec.max_history_len - len(hist)
        padded_hist = ([0] * pad_len) + hist
        mask = ([False] * pad_len) + ([True] * len(hist))

        target_items.append(int(item_ids[idx]))
        history_items.append(padded_hist)
        history_masks.append(mask)
        labels_out.append(float(labels[idx]))

        if spec.sparse_feature_names:
            sparse_rows.append(
                [int(sparse_arrays[name][idx]) for name in spec.sparse_feature_names]
            )
        if spec.dense_feature_names:
            dense_rows.append(
                [float(dense_arrays[name][idx]) for name in spec.dense_feature_names]
            )

        user_hist.append(int(item_ids[idx]))

    features: dict[str, torch.Tensor] = {
        spec.item_feature: torch.tensor(target_items, dtype=torch.long),
        spec.history_feature: torch.tensor(history_items, dtype=torch.long),
        f"{spec.history_feature}_mask": torch.tensor(history_masks, dtype=torch.bool),
    }

    if spec.sparse_feature_names:
        features["sparse_features"] = torch.tensor(sparse_rows, dtype=torch.long)
    if spec.dense_feature_names:
        features["dense_features"] = torch.tensor(dense_rows, dtype=torch.float32)

    labels_tensor = torch.tensor(labels_out, dtype=torch.float32).unsqueeze(-1)
    return SequenceDataset(features, labels_tensor)
