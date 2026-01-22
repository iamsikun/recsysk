from __future__ import annotations

import polars as pl
import torch

from recsys.data.datasets import TabularDataset
from recsys.schemas.features import FeatureSpec, FeatureType


def apply_label_threshold(
    df: pl.DataFrame, rating_col: str, threshold: float
) -> pl.DataFrame:
    """Add a binary label column based on a rating threshold."""
    return df.with_columns(
        (pl.col(rating_col).cast(pl.Float32) >= threshold)
        .cast(pl.Int32)
        .alias("label")
    )


def encode_features(
    df: pl.DataFrame, features: list[FeatureSpec]
) -> tuple[pl.DataFrame, dict[str, int], list[str]]:
    """Encode features and return updated df, feature map, and ordered columns."""
    feature_map: dict[str, int] = {}
    processed_cols: list[str] = []

    for spec in features:
        col_name = spec.source_name
        if spec.type == FeatureType.CATEGORICAL:
            df = df.with_columns(
                pl.col(col_name)
                .cast(pl.Utf8)
                .cast(pl.Categorical)
                .to_physical()
                .cast(pl.Int64)
                .alias(spec.name)
            )
            spec.vocab_size = df[spec.name].max() + 1
            feature_map[spec.name] = spec.vocab_size
        elif spec.type == FeatureType.NUMERIC:
            df = df.with_columns(pl.col(col_name).cast(pl.Float32).alias(spec.name))
            feature_map[spec.name] = 1
        else:
            raise ValueError(f"Unsupported feature type: {spec.type}")

        processed_cols.append(spec.name)

    return df, feature_map, processed_cols


def build_tabular_dataset(
    df: pl.DataFrame, processed_cols: list[str]
) -> TabularDataset:
    """Build a TabularDataset from an encoded dataframe."""
    label_tensor = torch.from_numpy(
        df["label"].to_numpy().astype("float32")
    ).unsqueeze(-1)
    df_features = df.select(processed_cols)
    data_tensor = torch.from_numpy(df_features.to_numpy())
    return TabularDataset(data_tensor, label_tensor)
