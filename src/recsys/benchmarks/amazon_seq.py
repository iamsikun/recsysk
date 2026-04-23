"""Amazon Reviews sequential (history-aware) CTR benchmark.

Same dataset as :class:`recsys.benchmarks.amazon_ctr.AmazonCTRBenchmark`
— the only difference is ``model_input: sequence``, which routes the
datamodule through :class:`recsys.data.datamodules.amazon.AmazonSequenceDataModule`
and produces dict batches with ``{item_id, hist_item_id, user_id}``
shape for DIN-style attention-over-history models.

Default category is ``all_beauty`` (smallest 5-core, smoke-gate
friendly). The ``config_hash`` changes with ``category`` — runs
against Books / Electronics / Sports_and_Outdoors do not collapse into
the same parquet row as All_Beauty runs.
"""

from __future__ import annotations

import logging
from typing import Any

from recsys.benchmarks.base import Benchmark, BenchmarkData
from recsys.data.negatives.random_uniform import RandomUniform
from recsys.data.splits.random_split import RandomSplit
from recsys.tasks.ctr import CTRTask
from recsys.utils import BENCHMARK_REGISTRY, DATASET_REGISTRY

LOGGER = logging.getLogger(__name__)


@BENCHMARK_REGISTRY.register("amazon_seq")
class AmazonSeqBenchmark(Benchmark):
    """Amazon Reviews sequential CTR benchmark."""

    name = "amazon_seq"
    task = CTRTask()
    metric_names = [
        "auc",
        "logloss",
        "ndcg@10",
        "ndcg@50",
        "recall@10",
        "recall@50",
        "hr@10",
        "hr@50",
        "mrr",
    ]

    def __init__(
        self,
        data_cfg: dict[str, Any],
        eval_cfg: dict[str, Any] | None = None,
    ) -> None:
        self._data_cfg = dict(data_cfg)
        self._eval_cfg = dict(eval_cfg or {})
        self.splitter = RandomSplit(
            train_fraction=float(self._data_cfg.get("train_split", 0.8)),
        )
        self.negative_sampler = RandomUniform()

    def build(self) -> BenchmarkData:
        data_cfg = dict(self._data_cfg)
        data_cfg.setdefault("name", "amazon")
        data_cfg["model_input"] = "sequence"
        data_cfg.setdefault("item_feature", "item_id")
        data_cfg.setdefault("history_feature", "hist_item_id")
        data_cfg.setdefault("sparse_feature_names", ["user_id"])
        data_cfg.setdefault("max_history_len", 20)

        LOGGER.info(
            "AmazonSeqBenchmark.build: instantiating sequence datamodule category=%s max_rows=%s",
            data_cfg.get("category", "all_beauty"),
            data_cfg.get("max_rows"),
        )
        dm = DATASET_REGISTRY.build(data_cfg)
        dm.setup(stage="fit")

        feature_specs: list = []
        builder = getattr(dm, "builder", None)
        builder_config = getattr(builder, "config", None) if builder is not None else None
        if builder_config is not None:
            feature_specs = list(getattr(builder_config, "features", []) or [])

        return BenchmarkData(
            train=dm.train_dataset,
            val=dm.val_dataset,
            test=dm.val_dataset,
            feature_map=dict(dm.feature_map),
            feature_specs=feature_specs,
            datamodule=dm,
            metadata={
                "eval": dict(self._eval_cfg),
                "splitter": self.splitter,
                "negative_sampler": self.negative_sampler,
            },
        )
