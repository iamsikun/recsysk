"""KuaiRec datamodule.

Wraps :func:`recsys.data.kuairec.load` in a ``BuilderDataModule`` so the
KuaiRec CTR benchmark can plug into the standard runner / Lightning path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from recsys.data import kuairec
from recsys.data.builders.csv_ctr import CsvCtrBuilder, CsvCtrConfig
from recsys.data.datamodules.base import BuilderDataModule
from recsys.schemas.builder import build_feature_specs
from recsys.utils import DATASET_REGISTRY


def _default_features() -> list[dict]:
    return [
        {
            "name": "user_id",
            "source_name": "user_id",
            "type": "categorical",
            "role": "user",
        },
        {
            "name": "item_id",
            "source_name": "video_id",
            "type": "categorical",
            "role": "item",
        },
    ]


class KuaiRecDataModule(BuilderDataModule):
    """KuaiRec CTR datamodule (small_matrix or big_matrix)."""

    def __init__(
        self,
        data_base_path: Path | None,
        batch_size: int,
        variant: str = "small",
        train_split: float = 0.8,
        watch_ratio_threshold: float = 2.0,
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
            tables = kuairec.load(
                dataset_root=dataset_root,
                tables=[f"{variant}_matrix"],
                variant=variant,
            )
            return tables[f"{variant}_matrix"]

        config = CsvCtrConfig(
            load_df=_load_df,
            features=build_feature_specs(features),
            train_split=train_split,
            seed=seed,
            label_column="watch_ratio",
            label_threshold=watch_ratio_threshold,
        )
        builder = CsvCtrBuilder(config)
        super().__init__(
            builder=builder,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )


@DATASET_REGISTRY.register("kuairec")
def build_kuairec_datamodule(
    data_base_path: Path | None = None,
    batch_size: int = 1024,
    variant: str = "small",
    train_split: float = 0.8,
    watch_ratio_threshold: float = 2.0,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int | None = None,
    features: list | None = None,
    **_: Any,
) -> KuaiRecDataModule:
    """Build a :class:`KuaiRecDataModule`.

    ``variant`` is ``"small"`` (~1.4K x 3.3K dense) or ``"big"`` (~7K x
    10K full). The smoke gate uses ``small`` for speed.
    """
    return KuaiRecDataModule(
        data_base_path=data_base_path,
        batch_size=batch_size,
        variant=variant,
        train_split=train_split,
        watch_ratio_threshold=watch_ratio_threshold,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        features=features,
    )
