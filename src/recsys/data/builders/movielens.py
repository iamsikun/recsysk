from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import torch
from torch.utils.data import Dataset, random_split

from recsys.data.builders.base import DatasetBuilder, DatasetBundle
from recsys.data.loaders.movielens import MovieLens
from recsys.data.transforms.sequence import SequenceSpec, build_sequence_dataset
from recsys.data.transforms.tabular import (
    apply_label_threshold,
    build_tabular_dataset,
    encode_features,
)
from recsys.schemas.features import FeatureSpec


@dataclass
class MovieLensBaseConfig:
    data_base_path: Path
    dataset_size: str
    train_split: float
    rating_threshold: float
    seed: int | None
    features: list[FeatureSpec]


@dataclass
class MovieLensSequenceConfig:
    item_feature: str
    history_feature: str
    max_history_len: int
    sparse_feature_names: list[str]
    dense_feature_names: list[str]


class MovieLensTabularBuilder(DatasetBuilder):
    """Builds tabular datasets for MovieLens."""

    def __init__(self, config: MovieLensBaseConfig):
        self.config = config

    def _load_ratings(self) -> pl.DataFrame:
        loader = MovieLens(self.config.data_base_path)
        data: dict[str, pl.DataFrame] = loader.load(
            self.config.dataset_size, tables=["ratings"]
        )
        return data["ratings"]

    def _get_generator(self) -> torch.Generator | None:
        if self.config.seed is None:
            return None
        return torch.Generator().manual_seed(self.config.seed)

    def build(self) -> DatasetBundle:
        df = self._load_ratings()
        df = apply_label_threshold(df, rating_col="rating", threshold=self.config.rating_threshold)
        df, feature_map, processed_cols = encode_features(df, self.config.features)
        full_dataset = build_tabular_dataset(
            df, processed_cols, feature_specs=self.config.features
        )
        train_size = int(len(full_dataset) * self.config.train_split)
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=self._get_generator()
        )
        return DatasetBundle(
            full=full_dataset,
            train=train_dataset,
            val=val_dataset,
            feature_map=feature_map,
        )


class MovieLensSequenceBuilder(MovieLensTabularBuilder):
    """Builds sequence datasets for MovieLens."""

    def __init__(self, config: MovieLensBaseConfig, seq_config: MovieLensSequenceConfig):
        super().__init__(config)
        self.seq_config = seq_config

    def build(self) -> DatasetBundle:
        df = self._load_ratings()
        df = apply_label_threshold(df, rating_col="rating", threshold=self.config.rating_threshold)
        df, feature_map, processed_cols = encode_features(df, self.config.features)

        if self.seq_config.item_feature in df.columns:
            df = df.with_columns(
                (pl.col(self.seq_config.item_feature) + 1).alias(self.seq_config.item_feature)
            )
            if self.seq_config.item_feature in feature_map:
                feature_map[self.seq_config.item_feature] += 1

        spec = SequenceSpec.single_stream(
            item_feature=self.seq_config.item_feature,
            history_feature=self.seq_config.history_feature,
            max_history_len=self.seq_config.max_history_len,
            sparse_feature_names=self.seq_config.sparse_feature_names,
            dense_feature_names=self.seq_config.dense_feature_names,
        )
        full_dataset = build_sequence_dataset(df, feature_map, spec)
        train_size = int(len(full_dataset) * self.config.train_split)
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=self._get_generator()
        )
        return DatasetBundle(
            full=full_dataset,
            train=train_dataset,
            val=val_dataset,
            feature_map=feature_map,
        )
