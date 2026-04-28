"""TaobaoAd CTR benchmark."""

from __future__ import annotations

import logging
from typing import Any

from recsys.benchmarks.base import Benchmark, BenchmarkData
from recsys.data.negatives.random_uniform import RandomUniform
from recsys.data.splits.random_split import RandomSplit
from recsys.tasks.ctr import CTRTask
from recsys.utils import BENCHMARK_REGISTRY, DATASET_REGISTRY

LOGGER = logging.getLogger(__name__)


@BENCHMARK_REGISTRY.register("taobao_ad_ctr")
class TaobaoAdCTRBenchmark(Benchmark):
    """TaobaoAd CTR benchmark on the Tianchi Ad Display dataset.

    InterFormer / Wukong / BST anchor — the joined impression log uses
    ``clk`` as the binary label. Side features (cate_id, brand,
    age_level, …) are available in the joined frame; benchmark YAMLs
    can opt them in.
    """

    name = "taobao_ad_ctr"
    task = CTRTask()
    metric_names = [
        "auc",
        "logloss",
        "ne",
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
        data_cfg.setdefault("name", "taobao_ad")

        LOGGER.info("TaobaoAdCTRBenchmark.build: instantiating datamodule")
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
