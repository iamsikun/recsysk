"""Frappe datamodule.

Wires :func:`recsys.data.frappe.load_combined` through the standard
``CsvCtrBuilder`` so the Frappe CTR benchmark plugs into the runner /
Lightning path. The NFM mirror is the only variant in v1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from recsys.data import frappe
from recsys.data.builders.csv_ctr import CsvCtrBuilder, CsvCtrConfig
from recsys.data.datamodules.base import BuilderDataModule
from recsys.schemas.builder import build_feature_specs
from recsys.utils import DATASET_REGISTRY


def _default_features() -> list[dict]:
    # All ten libfm fields are categorical. user_id / item_id are tagged
    # with the standard roles; the remaining eight are context features.
    return [
        {"name": "user_id", "source_name": "user_id", "type": "categorical", "role": "user"},
        {"name": "item_id", "source_name": "item_id", "type": "categorical", "role": "item"},
        {"name": "daytime", "source_name": "daytime", "type": "categorical", "role": "context"},
        {"name": "weekday", "source_name": "weekday", "type": "categorical", "role": "context"},
        {"name": "isweekend", "source_name": "isweekend", "type": "categorical", "role": "context"},
        {"name": "homework", "source_name": "homework", "type": "categorical", "role": "context"},
        {"name": "cost", "source_name": "cost", "type": "categorical", "role": "context"},
        {"name": "weather", "source_name": "weather", "type": "categorical", "role": "context"},
        {"name": "country", "source_name": "country", "type": "categorical", "role": "context"},
        {"name": "city", "source_name": "city", "type": "categorical", "role": "context"},
    ]


class FrappeDataModule(BuilderDataModule):
    """Frappe CTR datamodule (NFM mirror, all 10 fields)."""

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
            return frappe.load_combined(dataset_root=dataset_root)

        config = CsvCtrConfig(
            load_df=_load_df,
            features=build_feature_specs(features),
            train_split=train_split,
            seed=seed,
            label_column=label_column,
            label_threshold=None,  # already binary 0/1
        )
        builder = CsvCtrBuilder(config)
        super().__init__(
            builder=builder,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )


@DATASET_REGISTRY.register("frappe")
def build_frappe_datamodule(
    data_base_path: Path | None = None,
    batch_size: int = 1024,
    train_split: float = 0.8,
    label_column: str = "label",
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int | None = None,
    features: list | None = None,
    **_: Any,
) -> FrappeDataModule:
    return FrappeDataModule(
        data_base_path=data_base_path,
        batch_size=batch_size,
        train_split=train_split,
        label_column=label_column,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        features=features,
    )
