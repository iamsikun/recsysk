"""Amazon Reviews datamodule.

Wraps :func:`recsys.data.amazon.load` in a ``BuilderDataModule`` so the
Amazon CTR benchmark can plug into the standard runner / Lightning
path. Defaults to the ``All_Beauty`` category — the smallest 5-core
subset in the McAuley 2023 release and the only category size
consistent with the repo's sub-minute smoke gate. Books / Electronics
/ Sports_and_Outdoors are supported via the ``category`` config key.

Default features are three scalar categoricals: ``user_id``,
``item_id``, and ``store`` (item brand, joined in from the metadata
table and 90%+ populated in the 2023 ``All_Beauty`` metadata). Null
stores are replaced with ``"<unk>"`` in the loader so the encoder's
Utf8→Categorical cast sees a concrete string.

The 2023 ``All_Beauty`` metadata has an empty ``categories`` field for
every item (verified post-load), so this datamodule does **not**
include ``categories`` as a default ``MULTI_CATEGORICAL`` feature — it
would be a dead column for the pinned default. Users running on a
category with populated ``categories`` (or wanting to exercise the
``MULTI_CATEGORICAL`` encoder path regardless) can add it via their
benchmark YAML's ``data.features`` list; the loader already joins the
column in, so no datamodule changes are needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from recsys.data import amazon
from recsys.data.builders.csv_ctr import CsvCtrBuilder, CsvCtrConfig
from recsys.data.builders.sequence_ctr import SequenceCtrBuilder, SequenceCtrConfig
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
            "source_name": "item_id",
            "type": "categorical",
            "role": "item",
        },
        {
            "name": "store",
            "source_name": "store",
            "type": "categorical",
            "role": "item",
        },
    ]


class AmazonDataModule(BuilderDataModule):
    """Amazon Reviews CTR datamodule.

    The CTR label is derived from ``rating`` via the standard
    ``rating >= 4`` threshold used in the MovieLens benchmark. Swap
    ``label_threshold`` in the config to change it (e.g. ``>= 5`` for a
    stricter positive definition).
    """

    def __init__(
        self,
        data_base_path: Path | None,
        batch_size: int,
        category: str = "all_beauty",
        train_split: float = 0.8,
        label_column: str = "rating",
        label_threshold: float | None = 4.0,
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
            tables = amazon.load(
                dataset_root=dataset_root,
                category=category,
                tables=["reviews", "meta"],
                max_rows=max_rows,
            )
            return tables["reviews"]

        config = CsvCtrConfig(
            load_df=_load_df,
            features=build_feature_specs(features),
            train_split=train_split,
            seed=seed,
            label_column=label_column,
            label_threshold=label_threshold,
        )
        builder = CsvCtrBuilder(config)
        super().__init__(
            builder=builder,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )


class AmazonSequenceDataModule(BuilderDataModule):
    """Amazon Reviews sequential (history-aware) CTR datamodule.

    Emits dict-valued batches consumable by DIN and other sequence-aware
    algos. Labels are binarised from ``rating`` the same way as the
    tabular datamodule; history is built from the per-user time-ordered
    item_id stream.
    """

    def __init__(
        self,
        data_base_path: Path | None,
        batch_size: int,
        category: str = "all_beauty",
        train_split: float = 0.8,
        label_column: str = "rating",
        label_threshold: float | None = 4.0,
        max_rows: int | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int | None = None,
        features: list | None = None,
        item_feature: str = "item_id",
        history_feature: str = "hist_item_id",
        max_history_len: int = 20,
        sparse_feature_names: list[str] | None = None,
        dense_feature_names: list[str] | None = None,
    ) -> None:
        if features is None:
            features = _default_features()

        dataset_root = (
            Path(data_base_path).resolve() if data_base_path is not None else None
        )

        def _load_df() -> pl.DataFrame:
            tables = amazon.load(
                dataset_root=dataset_root,
                category=category,
                tables=["reviews", "meta"],
                max_rows=max_rows,
            )
            return tables["reviews"]

        config = SequenceCtrConfig(
            load_df=_load_df,
            features=build_feature_specs(features),
            train_split=train_split,
            seed=seed,
            label_column=label_column,
            label_threshold=label_threshold,
            item_feature=item_feature,
            history_feature=history_feature,
            max_history_len=max_history_len,
            sparse_feature_names=sparse_feature_names or [],
            dense_feature_names=dense_feature_names or [],
        )
        builder = SequenceCtrBuilder(config)
        super().__init__(
            builder=builder,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )


@DATASET_REGISTRY.register("amazon")
def build_amazon_datamodule(
    data_base_path: Path | None = None,
    batch_size: int = 1024,
    category: str = "all_beauty",
    train_split: float = 0.8,
    label_column: str = "rating",
    label_threshold: float | None = 4.0,
    max_rows: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int | None = None,
    features: list | None = None,
    model_input: str = "tabular",
    item_feature: str = "item_id",
    history_feature: str = "hist_item_id",
    max_history_len: int = 20,
    sparse_feature_names: list[str] | None = None,
    dense_feature_names: list[str] | None = None,
    **_: Any,
) -> AmazonDataModule | AmazonSequenceDataModule:
    """Build an :class:`AmazonDataModule` (``model_input="tabular"``) or
    :class:`AmazonSequenceDataModule` (``model_input="sequence"``).

    ``category`` is ``"all_beauty"`` (default, smallest), ``"books"``,
    ``"electronics"``, or ``"sports_and_outdoors"``. ``max_rows`` takes
    a deterministic time-ordered prefix of the joined reviews — useful
    for capping Books / Electronics to smoke-gate size.
    """
    input_type = model_input.lower()
    valid_inputs = {"tabular", "sequence"}
    if input_type not in valid_inputs:
        raise ValueError(f"model_input must be one of {sorted(valid_inputs)}")

    if input_type == "sequence":
        return AmazonSequenceDataModule(
            data_base_path=data_base_path,
            batch_size=batch_size,
            category=category,
            train_split=train_split,
            label_column=label_column,
            label_threshold=label_threshold,
            max_rows=max_rows,
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

    return AmazonDataModule(
        data_base_path=data_base_path,
        batch_size=batch_size,
        category=category,
        train_split=train_split,
        label_column=label_column,
        label_threshold=label_threshold,
        max_rows=max_rows,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        features=features,
    )
