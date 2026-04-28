"""MicroVideo (THACIL) datamodule."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from recsys.data import microvideo
from recsys.data.builders.csv_ctr import CsvCtrBuilder, CsvCtrConfig
from recsys.data.datamodules.base import BuilderDataModule
from recsys.schemas.builder import build_feature_specs
from recsys.utils import DATASET_REGISTRY


def _default_features() -> list[dict]:
    return [
        {"name": "user_id", "source_name": "user_id", "type": "categorical", "role": "user"},
        {"name": "item_id", "source_name": "item_id", "type": "categorical", "role": "item"},
    ]


class MicroVideoDataModule(BuilderDataModule):
    """MicroVideo (THACIL) CTR datamodule."""

    def __init__(
        self,
        data_base_path: Path | None,
        batch_size: int,
        train_split: float = 0.8,
        label_column: str = "label",
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int | None = None,
        features: list | None = None,
    ) -> None:
        if features is None:
            features = _default_features()

        dataset_root = (
            Path(data_base_path).resolve() if data_base_path is not None else None
        )

        def _load_df() -> pl.DataFrame:
            return microvideo.load(dataset_root=dataset_root)

        config = CsvCtrConfig(
            load_df=_load_df,
            features=build_feature_specs(features),
            train_split=train_split,
            seed=seed,
            label_column=label_column,
            label_threshold=None,
        )
        builder = CsvCtrBuilder(config)
        super().__init__(
            builder=builder,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )


@DATASET_REGISTRY.register("microvideo")
def build_microvideo_datamodule(
    data_base_path: Path | None = None,
    batch_size: int = 1024,
    train_split: float = 0.8,
    label_column: str = "label",
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int | None = None,
    features: list | None = None,
    **_: Any,
) -> MicroVideoDataModule:
    return MicroVideoDataModule(
        data_base_path=data_base_path,
        batch_size=batch_size,
        train_split=train_split,
        label_column=label_column,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        features=features,
    )
