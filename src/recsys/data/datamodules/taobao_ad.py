"""TaobaoAd datamodule.

Wires :func:`recsys.data.taobao_ad.load` (joined raw_sample × ad_feature
× user_profile) through the standard ``CsvCtrBuilder`` for tabular CTR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from recsys.data import taobao_ad
from recsys.data.builders.csv_ctr import CsvCtrBuilder, CsvCtrConfig
from recsys.data.datamodules.base import BuilderDataModule
from recsys.schemas.builder import build_feature_specs
from recsys.utils import DATASET_REGISTRY


def _default_features() -> list[dict]:
    # Defaults exercise the basic CTR signal (user × ad × click). Side
    # features (cate_id, brand, age_level, etc.) are available in the
    # joined frame and can be added in the benchmark YAML when desired.
    return [
        {"name": "user_id", "source_name": "user", "type": "categorical", "role": "user"},
        {"name": "item_id", "source_name": "adgroup_id", "type": "categorical", "role": "item"},
    ]


class TaobaoAdDataModule(BuilderDataModule):
    """TaobaoAd CTR datamodule (joined impression log + metadata)."""

    def __init__(
        self,
        data_base_path: Path | None,
        batch_size: int,
        train_split: float = 0.8,
        label_column: str = "clk",
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
            df = taobao_ad.load(dataset_root=dataset_root, join=True)
            assert isinstance(df, pl.DataFrame)
            return df

        config = CsvCtrConfig(
            load_df=_load_df,
            features=build_feature_specs(features),
            train_split=train_split,
            seed=seed,
            label_column=label_column,
            label_threshold=None,  # `clk` is already 0/1
        )
        builder = CsvCtrBuilder(config)
        super().__init__(
            builder=builder,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )


@DATASET_REGISTRY.register("taobao_ad")
def build_taobao_ad_datamodule(
    data_base_path: Path | None = None,
    batch_size: int = 1024,
    train_split: float = 0.8,
    label_column: str = "clk",
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int | None = None,
    features: list | None = None,
    **_: Any,
) -> TaobaoAdDataModule:
    return TaobaoAdDataModule(
        data_base_path=data_base_path,
        batch_size=batch_size,
        train_split=train_split,
        label_column=label_column,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        features=features,
    )
