"""Generic sequential CTR builder.

Produces DIN-style dict-batch datasets from any CTR log with a
``(user, item, rating, timestamp)`` shape. Sibling of
:class:`recsys.data.builders.csv_ctr.CsvCtrBuilder`; where the CSV
builder emits flat tensors, this one emits dict batches containing
per-user-time-ordered history plus scalar target features, which DIN's
sequence-aware forward consumes via
:func:`recsys.data.transforms.sequence.build_sequence_dataset`.

Used by the Amazon Reviews sequential benchmark. The original MovieLens
sequence builder (``MovieLensSequenceBuilder``) pre-dates this
extraction and still carries its own copy of the same pipeline; a
follow-up can migrate MovieLens to this builder once the existing
``movielens_seq`` smoke gate is covered by a regression check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import polars as pl
import torch
from torch.utils.data import random_split

from recsys.data.builders.base import DatasetBuilder, DatasetBundle
from recsys.data.transforms.sequence import SequenceSpec, build_sequence_dataset
from recsys.data.transforms.tabular import (
    apply_label_threshold,
    encode_features,
)
from recsys.schemas.features import FeatureSpec


@dataclass
class SequenceCtrConfig:
    """Configuration for :class:`SequenceCtrBuilder`.

    Attributes
    ----------
    load_df:
        Zero-argument callable returning a polars DataFrame of raw
        (user, item, label-source, timestamp) rows.
    features:
        Encoded feature list (already normalised by ``build_feature_specs``).
    train_split:
        Fraction of rows assigned to the train split.
    seed:
        Optional seed for the random split.
    label_column:
        Column to read the raw label from (e.g. ``"rating"``).
    label_threshold:
        If set, label is ``label_column >= threshold``. If ``None``,
        ``label_column`` is copied verbatim into ``label`` as float32.
    item_feature:
        Encoded item column used both as the prediction target and the
        source of each user's history stream.
    history_feature:
        Batch-dict key the history tensor will land under (e.g.
        ``"hist_item_id"``).
    max_history_len:
        Pad/truncate length for the history stream.
    sparse_feature_names, dense_feature_names:
        Extra (non-history) feature names that should flow into each
        dict batch row alongside the target and history tensors.
    """

    load_df: Callable[[], pl.DataFrame]
    features: list[FeatureSpec]
    train_split: float
    seed: int | None
    label_column: str
    label_threshold: float | None
    item_feature: str
    history_feature: str
    max_history_len: int
    sparse_feature_names: list[str]
    dense_feature_names: list[str]


class SequenceCtrBuilder(DatasetBuilder):
    """Sequential CTR builder driven by a polars-loading callback."""

    def __init__(self, config: SequenceCtrConfig) -> None:
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
        df, feature_map, _ = encode_features(df, self.config.features)

        # Shift item ids by +1 so that the 0 slot is reserved for history
        # padding. Mirrors the MovieLensSequenceBuilder convention.
        if self.config.item_feature in df.columns:
            df = df.with_columns(
                (pl.col(self.config.item_feature) + 1).alias(self.config.item_feature)
            )
            if self.config.item_feature in feature_map:
                feature_map[self.config.item_feature] += 1

        spec = SequenceSpec.single_stream(
            item_feature=self.config.item_feature,
            history_feature=self.config.history_feature,
            max_history_len=self.config.max_history_len,
            sparse_feature_names=self.config.sparse_feature_names,
            dense_feature_names=self.config.dense_feature_names,
        )
        full_dataset = build_sequence_dataset(df, feature_map, spec)
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
