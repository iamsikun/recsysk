"""KuaiVideo CTR benchmark (InterFormer's anchor video dataset)."""

from __future__ import annotations

import logging
from typing import Any

from recsys.benchmarks.base import Benchmark, BenchmarkData
from recsys.data.negatives.random_uniform import RandomUniform
from recsys.data.splits.random_split import RandomSplit
from recsys.tasks.ctr import CTRTask
from recsys.utils import BENCHMARK_REGISTRY, DATASET_REGISTRY

LOGGER = logging.getLogger(__name__)


@BENCHMARK_REGISTRY.register("kuaivideo_ctr")
class KuaiVideoCTRBenchmark(Benchmark):
    """KuaiVideo CTR benchmark on the BARS / reczoo HF mirror.

    InterFormer's anchor — 13.7M binary-click interactions across 10K
    users and 3.24M items. The benchmark exposes the gAUC metric in
    addition to the standard 9-metric set.
    """

    name = "kuaivideo_ctr"
    task = CTRTask()
    metric_names = [
        "auc",
        "logloss",
        "ne",
        "gauc",
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
        data_cfg.setdefault("name", "kuaivideo")

        LOGGER.info("KuaiVideoCTRBenchmark.build: instantiating datamodule")
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
