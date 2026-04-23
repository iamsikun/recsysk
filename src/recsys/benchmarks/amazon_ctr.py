"""Amazon Reviews CTR benchmark.

Default category is ``All_Beauty`` — the smallest per-category subset
in the Amazon Reviews 2023 release (~701K ratings, ~632K users, ~112K
items). Books / Electronics / Sports_and_Outdoors are supported via
the ``category`` data-config key but require multi-GB downloads on
first run and should normally be combined with ``max_rows`` for smoke
iteration.

Label is derived from the 1–5 ``rating`` column via a threshold
(``rating >= 4`` by default, matching the MovieLens CTR convention).

Because ``category`` flows through ``data_cfg``, it is mixed into the
run's ``config_hash`` — Books-AUC and Electronics-AUC do not collapse
into the same ``results/amazon_ctr.parquet`` row, which is the
correct cross-category behavior (they are not comparable numbers).

Same metric set as :class:`recsys.benchmarks.kuairand_ctr.KuaiRandCTRBenchmark`:
AUC + LogLoss for point prediction, sampled-100 negative ranking
metrics (NDCG/Recall/HR @10,50 and MRR).
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


@BENCHMARK_REGISTRY.register("amazon_ctr")
class AmazonCTRBenchmark(Benchmark):
    """Amazon Reviews CTR benchmark (per-category).

    Default category is ``all_beauty``. To run on a different category,
    set ``data.category`` in the benchmark YAML. The benchmark class
    itself does not pin a category — pinning happens at the YAML level
    so each category is its own ``config_hash``.
    """

    name = "amazon_ctr"
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

        LOGGER.info(
            "AmazonCTRBenchmark.build: instantiating datamodule name=%s category=%s max_rows=%s",
            data_cfg.get("name"),
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
