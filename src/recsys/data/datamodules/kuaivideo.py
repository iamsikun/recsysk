"""KuaiVideo datamodule (BARS / reczoo HF mirror)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from recsys.data import kuaivideo
from recsys.data.builders.csv_ctr import CsvCtrBuilder, CsvCtrConfig
from recsys.data.datamodules.base import BuilderDataModule
from recsys.schemas.builder import build_feature_specs
from recsys.utils import DATASET_REGISTRY


def _default_features() -> list[dict]:
    return [
        {"name": "user_id", "source_name": "user_id", "type": "categorical", "role": "user"},
        {"name": "item_id", "source_name": "item_id", "type": "categorical", "role": "item"},
    ]


class KuaiVideoDataModule(BuilderDataModule):
    """KuaiVideo CTR datamodule.

    The builder concatenates the BARS train/valid/test CSV splits and
    runs the harness's standard random train/val split on top — the
    BARS-supplied splits are not preserved by v1.
    """

    def __init__(
        self,
        data_base_path: Path | None,
        batch_size: int,
        train_split: float = 0.8,
        label_column: str = "is_click",
        max_rows: int | None = None,
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
            return kuaivideo.load(dataset_root=dataset_root, max_rows=max_rows)

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


@DATASET_REGISTRY.register("kuaivideo")
def build_kuaivideo_datamodule(
    data_base_path: Path | None = None,
    batch_size: int = 1024,
    train_split: float = 0.8,
    label_column: str = "is_click",
    max_rows: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int | None = None,
    features: list | None = None,
    **_: Any,
) -> KuaiVideoDataModule:
    return KuaiVideoDataModule(
        data_base_path=data_base_path,
        batch_size=batch_size,
        train_split=train_split,
        label_column=label_column,
        max_rows=max_rows,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        features=features,
    )
