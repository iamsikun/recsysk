from __future__ import annotations

import numpy as np
import polars as pl
import torch

from recsys.data.datasets import TabularDataset, TabularDictDataset
from recsys.schemas.features import FeatureSpec, FeatureType


_SCALAR_TYPES = {FeatureType.CATEGORICAL, FeatureType.NUMERIC}


def apply_label_threshold(
    df: pl.DataFrame, rating_col: str, threshold: float
) -> pl.DataFrame:
    """Add a binary label column based on a rating threshold."""
    return df.with_columns(
        (pl.col(rating_col).cast(pl.Float32) >= threshold)
        .cast(pl.Int32)
        .alias("label")
    )


def _encode_multi_categorical(
    df: pl.DataFrame, spec: FeatureSpec
) -> pl.DataFrame:
    """Python-level multi-categorical encode: build a vocab over all
    unique ids appearing in ``spec.source_name`` and replace each row's
    list with pad/truncate-to-``max_len`` int ids. 0 is reserved for
    padding. If ``spec.weighted``, a parallel ``<name>_weight`` Float32
    array column is produced in lock-step from ``<source_name>_weight``.
    """
    max_len = int(spec.max_len or 0)
    if max_len <= 0:
        raise ValueError(
            f"multi_categorical feature '{spec.name}' needs max_len > 0"
        )

    raw_lists = df[spec.source_name].to_list()

    if spec.weighted:
        weight_col_name = f"{spec.source_name}_weight"
        if weight_col_name not in df.columns:
            raise ValueError(
                f"weighted multi_categorical feature '{spec.name}' needs "
                f"a paired '{weight_col_name}' column in the source data"
            )
        raw_weights = df[weight_col_name].to_list()
    else:
        raw_weights = None

    unique: set = set()
    for row in raw_lists:
        if row is None:
            continue
        unique.update(row)
    # Sort deterministically; 0 reserved for pad.
    vocab = {v: i + 1 for i, v in enumerate(sorted(unique, key=repr))}

    encoded: list[list[int]] = []
    weights_out: list[list[float]] = []
    for i, row in enumerate(raw_lists):
        if row is None:
            row = []
        ids = [vocab.get(v, 0) for v in row][:max_len]
        padded = ids + [0] * (max_len - len(ids))
        encoded.append(padded)

        if raw_weights is not None:
            wrow = raw_weights[i] or []
            wtrim = [float(w) for w in wrow][:max_len]
            wpad = wtrim + [0.0] * (max_len - len(wtrim))
            weights_out.append(wpad)

    df = df.with_columns(
        pl.Series(
            name=spec.name,
            values=encoded,
            dtype=pl.Array(pl.Int64, max_len),
        )
    )
    if raw_weights is not None:
        df = df.with_columns(
            pl.Series(
                name=f"{spec.name}_weight",
                values=weights_out,
                dtype=pl.Array(pl.Float32, max_len),
            )
        )
    spec.vocab_size = len(vocab) + 1  # +1 for padding
    return df


def encode_features(
    df: pl.DataFrame, features: list[FeatureSpec]
) -> tuple[pl.DataFrame, dict[str, int], list[str]]:
    """Encode features and return updated df, feature map, and ordered columns.

    Supported types:

    * ``CATEGORICAL``: hash-encode via polars ``Categorical`` and cast to int64.
      ``feature_map[name] = vocab_size``.
    * ``NUMERIC``: cast to float32. ``feature_map[name] = 1``.
    * ``DENSE_VECTOR``: pass through as a ``pl.Array(Float32, vector_dim)``
      column (expects the source column to already be a list of floats).
      ``feature_map[name] = vector_dim`` — i.e. the *width*, not a vocab
      size. Algos tell dense from categorical via ``FeatureSpec.type``.
    * ``MULTI_CATEGORICAL``: encode each element of the list column and
      pad/truncate to ``spec.max_len`` (0 = pad).
      ``feature_map[name] = vocab_size`` (including the pad slot).
      When ``spec.weighted``, a paired ``<name>_weight`` float array is
      emitted from ``<source_name>_weight``.
    """
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
        elif spec.type == FeatureType.DENSE_VECTOR:
            vector_dim = int(spec.vector_dim or 0)
            if vector_dim <= 0:
                raise ValueError(
                    f"dense_vector feature '{spec.name}' needs vector_dim > 0"
                )
            df = df.with_columns(
                pl.col(col_name)
                .cast(pl.List(pl.Float32))
                .list.to_array(vector_dim)
                .alias(spec.name)
            )
            feature_map[spec.name] = vector_dim
        elif spec.type == FeatureType.MULTI_CATEGORICAL:
            df = _encode_multi_categorical(df, spec)
            feature_map[spec.name] = spec.vocab_size
        else:
            raise ValueError(f"Unsupported feature type: {spec.type}")

        processed_cols.append(spec.name)

    return df, feature_map, processed_cols


def build_tabular_dataset(
    df: pl.DataFrame,
    processed_cols: list[str],
    feature_specs: list[FeatureSpec] | None = None,
):
    """Build a tabular dataset from an encoded dataframe.

    When every feature in ``feature_specs`` is scalar (categorical or
    numeric), the legacy flat-tensor :class:`TabularDataset` is returned
    for backward compatibility with existing evaluators and algos. When
    any feature is ``DENSE_VECTOR`` or ``MULTI_CATEGORICAL``, a
    :class:`TabularDictDataset` is returned instead, with one tensor
    per feature keyed by ``spec.name``.
    """
    label_tensor = torch.from_numpy(
        df["label"].to_numpy().astype("float32")
    ).unsqueeze(-1)

    all_scalar = feature_specs is None or all(
        spec.type in _SCALAR_TYPES for spec in feature_specs
    )
    if all_scalar:
        df_features = df.select(processed_cols)
        data_tensor = torch.from_numpy(df_features.to_numpy())
        return TabularDataset(data_tensor, label_tensor)

    # Dict-valued path: one tensor per feature.
    feats: dict[str, torch.Tensor] = {}
    for spec in feature_specs or []:
        col = df[spec.name]
        if spec.type == FeatureType.CATEGORICAL:
            feats[spec.name] = torch.from_numpy(col.to_numpy().astype(np.int64))
        elif spec.type == FeatureType.NUMERIC:
            feats[spec.name] = torch.from_numpy(col.to_numpy().astype(np.float32))
        elif spec.type == FeatureType.DENSE_VECTOR:
            # Array(Float32, vector_dim) -> (N, vector_dim) ndarray.
            feats[spec.name] = torch.from_numpy(
                col.to_numpy().astype(np.float32)
            )
        elif spec.type == FeatureType.MULTI_CATEGORICAL:
            feats[spec.name] = torch.from_numpy(
                col.to_numpy().astype(np.int64)
            )
            if spec.weighted:
                w_col = df[f"{spec.name}_weight"]
                feats[f"{spec.name}_weight"] = torch.from_numpy(
                    w_col.to_numpy().astype(np.float32)
                )
    return TabularDictDataset(feats, label_tensor)
