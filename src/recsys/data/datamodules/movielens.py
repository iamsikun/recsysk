from __future__ import annotations

from pathlib import Path
from typing import Any

from recsys.data.builders.movielens import (
    MovieLensBaseConfig,
    MovieLensSequenceConfig,
    MovieLensTabularBuilder,
    MovieLensSequenceBuilder,
)
from recsys.data.datamodules.base import BuilderDataModule
from recsys.schemas.builder import build_feature_specs
from recsys.utils import DATASET_REGISTRY


def _default_features() -> list[dict]:
    return [
        {"name": "user_id", "source_name": "user_id", "type": "categorical"},
        {"name": "item_id", "source_name": "item_id", "type": "categorical"},
    ]


class MovieLensDataModule(BuilderDataModule):
    """MovieLens DataModule for tabular input."""

    def __init__(
        self,
        data_base_path: Path,
        batch_size: int,
        dataset_size: str = "20m",
        train_split: float = 0.8,
        rating_threshold: float = 3.5,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int | None = None,
        features: list | None = None,
    ):
        if features is None:
            features = _default_features()
        base_config = MovieLensBaseConfig(
            data_base_path=Path(data_base_path).resolve(),
            dataset_size=dataset_size,
            train_split=train_split,
            rating_threshold=rating_threshold,
            seed=seed,
            features=build_feature_specs(features),
        )
        builder = MovieLensTabularBuilder(base_config)
        super().__init__(
            builder=builder,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )


class MovieLensSequenceDataModule(BuilderDataModule):
    """MovieLens DataModule for sequence input."""

    def __init__(
        self,
        data_base_path: Path,
        batch_size: int,
        dataset_size: str = "20m",
        train_split: float = 0.8,
        rating_threshold: float = 3.5,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int | None = None,
        features: list | None = None,
        item_feature: str = "item_id",
        history_feature: str = "hist_item_id",
        max_history_len: int = 20,
        sparse_feature_names: list[str] | None = None,
        dense_feature_names: list[str] | None = None,
    ):
        if features is None:
            features = _default_features()
        base_config = MovieLensBaseConfig(
            data_base_path=Path(data_base_path).resolve(),
            dataset_size=dataset_size,
            train_split=train_split,
            rating_threshold=rating_threshold,
            seed=seed,
            features=build_feature_specs(features),
        )
        seq_config = MovieLensSequenceConfig(
            item_feature=item_feature,
            history_feature=history_feature,
            max_history_len=max_history_len,
            sparse_feature_names=sparse_feature_names or [],
            dense_feature_names=dense_feature_names or [],
        )
        builder = MovieLensSequenceBuilder(base_config, seq_config)
        super().__init__(
            builder=builder,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )


@DATASET_REGISTRY.register("movielens")
def build_movielens_datamodule(
    data_base_path: Path,
    batch_size: int,
    features: list | None = None,
    dataset_size: str = "20m",
    train_split: float = 0.8,
    rating_threshold: float = 3.5,
    num_workers: int = 0,
    pin_memory: bool = False,
    model_input: str = "tabular",
    item_feature: str = "item_id",
    history_feature: str = "hist_item_id",
    max_history_len: int = 20,
    sparse_feature_names: list[str] | None = None,
    dense_feature_names: list[str] | None = None,
    seed: int | None = None,
    **_: Any,
):
    """
    Build a MovieLens DataModule.

    Args:
        model_input: One of {"tabular", "sequence"}.
    """
    input_type = model_input.lower()
    valid_inputs = {"tabular", "sequence"}
    if input_type not in valid_inputs:
        raise ValueError(f"model_input must be one of {sorted(valid_inputs)}")

    if input_type == "sequence":
        return MovieLensSequenceDataModule(
            data_base_path=data_base_path,
            batch_size=batch_size,
            dataset_size=dataset_size,
            train_split=train_split,
            rating_threshold=rating_threshold,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
            features=features,
            item_feature=item_feature,
            history_feature=history_feature,
            max_history_len=max_history_len,
            sparse_feature_names=sparse_feature_names,
            dense_feature_names=dense_feature_names,
        )

    return MovieLensDataModule(
        data_base_path=data_base_path,
        batch_size=batch_size,
        dataset_size=dataset_size,
        train_split=train_split,
        rating_threshold=rating_threshold,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        features=features,
    )
