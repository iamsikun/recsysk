"""KuaiRand datamodule.

Wraps :func:`recsys.data.kuairand.load` in a ``BuilderDataModule`` so the
KuaiRand CTR benchmark can plug into the standard runner / Lightning
path. Defaults to the ``Pure`` variant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from recsys.data import kuairand
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


class KuaiRandDataModule(BuilderDataModule):
    """KuaiRand CTR datamodule (standard interaction logs)."""

    def __init__(
        self,
        data_base_path: Path | None,
        batch_size: int,
        variant: str = "pure",
        train_split: float = 0.8,
        label_column: str = "is_click",
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
            tables = kuairand.load(
                dataset_root=dataset_root,
                variant=variant,
                tables=["log_standard"],
            )
            return tables["log_standard"]

        config = CsvCtrConfig(
            load_df=_load_df,
            features=build_feature_specs(features),
            train_split=train_split,
            seed=seed,
            label_column=label_column,
            # is_click is already binary; no threshold needed.
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


@DATASET_REGISTRY.register("kuairand")
def build_kuairand_datamodule(
    data_base_path: Path | None = None,
    batch_size: int = 1024,
    variant: str = "pure",
    train_split: float = 0.8,
    label_column: str = "is_click",
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int | None = None,
    features: list | None = None,
    **_: Any,
) -> KuaiRandDataModule:
    """Build a :class:`KuaiRandDataModule`.

    ``variant`` is ``"pure"`` (default, smallest), ``"1k"``, or ``"27k"``.
    """
    return KuaiRandDataModule(
        data_base_path=data_base_path,
        batch_size=batch_size,
        variant=variant,
        train_split=train_split,
        label_column=label_column,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        features=features,
    )
