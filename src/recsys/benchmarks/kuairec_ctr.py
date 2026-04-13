"""KuaiRec CTR benchmark (small_matrix, watch-ratio threshold label)."""

from __future__ import annotations

import logging
from typing import Any

from recsys.benchmarks.base import Benchmark, BenchmarkData
from recsys.data.negatives.random_uniform import RandomUniform
from recsys.data.splits.random_split import RandomSplit
from recsys.tasks.ctr import CTRTask
from recsys.utils import BENCHMARK_REGISTRY, DATASET_REGISTRY

LOGGER = logging.getLogger(__name__)


@BENCHMARK_REGISTRY.register("kuairec_ctr")
class KuaiRecCTRBenchmark(Benchmark):
    """KuaiRec short-video CTR benchmark.

    Built on the KuaiRec dense ``small_matrix`` (~1.4K users x 3.3K
    videos). The pinned binary label is ``watch_ratio >= 2.0`` —
    "watched at least twice the video duration", a common engagement
    threshold for the dataset that matches strong play-completion. The
    threshold is part of the benchmark contract, not a config knob; a
    different threshold means a different benchmark.

    Same metric set as :class:`recsys.benchmarks.movielens_ctr.MovieLensCTRBenchmark`:
    AUC + LogLoss for point prediction, sampled-100 negative ranking
    metrics (NDCG/Recall/HR @10,50 and MRR).
    """

    name = "kuairec_ctr"
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
        data_cfg.setdefault("name", "kuairec")

        LOGGER.info(
            "KuaiRecCTRBenchmark.build: instantiating datamodule name=%s variant=%s",
            data_cfg.get("name"),
            data_cfg.get("variant", "small"),
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
