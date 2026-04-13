"""Generic CTR builder for tabular CSV-style datasets.

Used by KuaiRec and KuaiRand benchmarks. The builder takes a function
that returns the raw interactions DataFrame, an optional label transform
(threshold-based binarisation), the encoded feature list, and the train
split fraction, and produces a :class:`DatasetBundle` ready for the
``movielens``-style datamodule wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import polars as pl
import torch
from torch.utils.data import random_split

from recsys.data.builders.base import DatasetBuilder, DatasetBundle
from recsys.data.transforms.tabular import (
    apply_label_threshold,
    build_tabular_dataset,
    encode_features,
)
from recsys.schemas.features import FeatureSpec


@dataclass
class CsvCtrConfig:
    """Configuration for :class:`CsvCtrBuilder`.

    Attributes
    ----------
    load_df:
        Zero-argument callable returning a polars DataFrame of raw rows.
    features:
        Encoded feature list (already normalised by ``build_feature_specs``).
    train_split:
        Fraction of rows assigned to the train split.
    seed:
        Optional seed for the random split.
    label_column:
        Column to read the raw label from.
    label_threshold:
        If set, the label is built as ``label_column >= threshold`` (and a
        new ``label`` column is appended). If ``None``, ``label_column`` is
        copied verbatim into the ``label`` column (cast to float32). Use
        ``None`` for already-binary clicks (KuaiRand ``is_click``).
    """

    load_df: Callable[[], pl.DataFrame]
    features: list[FeatureSpec]
    train_split: float
    seed: int | None
    label_column: str
    label_threshold: float | None


class CsvCtrBuilder(DatasetBuilder):
    """Tabular CTR builder driven by a polars-loading callback."""

    def __init__(self, config: CsvCtrConfig) -> None:
        self.config = config

    def _get_generator(self) -> torch.Generator | None:
        if self.config.seed is None:
            return None
        return torch.Generator().manual_seed(self.config.seed)

    def build(self) -> DatasetBundle:
        df = self.config.load_df()
        if self.config.label_threshold is not None:
            df = apply_label_threshold(
                df,
                rating_col=self.config.label_column,
                threshold=self.config.label_threshold,
            )
        else:
            df = df.with_columns(
                pl.col(self.config.label_column).cast(pl.Float32).alias("label")
            )
        df, feature_map, processed_cols = encode_features(df, self.config.features)
        full_dataset = build_tabular_dataset(
            df, processed_cols, feature_specs=self.config.features
        )
        train_size = int(len(full_dataset) * self.config.train_split)
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=self._get_generator(),
        )
        return DatasetBundle(
            full=full_dataset,
            train=train_dataset,
            val=val_dataset,
            feature_map=feature_map,
        )
