"""Criteo Terabyte datamodule (24-day click logs, HF mirror)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from recsys.data import criteo_terabyte
from recsys.data.builders.csv_ctr import CsvCtrBuilder, CsvCtrConfig
from recsys.data.datamodules.base import BuilderDataModule
from recsys.schemas.builder import build_feature_specs
from recsys.utils import DATASET_REGISTRY


def _default_features() -> list[dict]:
    # Criteo Terabyte has no real user/item ids — the 26 categorical
    # hashes are anonymised. We tag cat_1 as user (it's typically the
    # publisher/site id) and cat_2 as item, both arbitrary placeholders
    # that map cleanly onto CTRTask's required_roles. Real users will
    # override via the benchmark YAML to expose more cat_* and int_*
    # fields once they want to push beyond the smoke gate.
    return [
        {"name": "user_id", "source_name": "cat_1", "type": "categorical", "role": "user"},
        {"name": "item_id", "source_name": "cat_2", "type": "categorical", "role": "item"},
    ]


class CriteoTerabyteDataModule(BuilderDataModule):
    """Criteo Terabyte CTR datamodule."""

    def __init__(
        self,
        data_base_path: Path | None,
        batch_size: int,
        train_split: float = 0.8,
        label_column: str = "label",
        days: list[str] | None = None,
        max_rows: int | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int | None = None,
        features: list | None = None,
    ) -> None:
        if features is None:
            features = _default_features()
        if days is None:
            days = ["day_0"]

        dataset_root = (
            Path(data_base_path).resolve() if data_base_path is not None else None
        )

        def _load_df() -> pl.DataFrame:
            return criteo_terabyte.load(
                dataset_root=dataset_root,
                days=days,
                max_rows=max_rows,
            )

        config = CsvCtrConfig(
            load_df=_load_df,
            features=build_feature_specs(features),
            train_split=train_split,
            seed=seed,
            label_column=label_column,
            label_threshold=None,  # already 0/1
            max_rows=max_rows,
        )
        builder = CsvCtrBuilder(config)
        super().__init__(
            builder=builder,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )


@DATASET_REGISTRY.register("criteo_terabyte")
def build_criteo_terabyte_datamodule(
    data_base_path: Path | None = None,
    batch_size: int = 1024,
    train_split: float = 0.8,
    label_column: str = "label",
    days: list[str] | None = None,
    max_rows: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int | None = None,
    features: list | None = None,
    **_: Any,
) -> CriteoTerabyteDataModule:
    return CriteoTerabyteDataModule(
        data_base_path=data_base_path,
        batch_size=batch_size,
        train_split=train_split,
        label_column=label_column,
        days=days,
        max_rows=max_rows,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        features=features,
    )
