"""MovieLens CTR benchmark: tabular (user_id, item_id) with sampled-100 ranking."""

from __future__ import annotations

import logging
from typing import Any

from recsys.benchmarks.base import Benchmark, BenchmarkData
from recsys.data.negatives.random_uniform import RandomUniform
from recsys.data.splits.random_split import RandomSplit
from recsys.tasks.ctr import CTRTask
from recsys.utils import BENCHMARK_REGISTRY, DATASET_REGISTRY

LOGGER = logging.getLogger(__name__)


@BENCHMARK_REGISTRY.register("movielens_ctr")
class MovieLensCTRBenchmark(Benchmark):
    """MovieLens 20M tabular CTR benchmark.

    Pinned metric list covers point-prediction (AUC, LogLoss) plus
    sampled-100 negative ranking metrics (NDCG/Recall/HR @10,50 and MRR).
    Wave 3 note: train/val come from the existing MovieLens datamodule
    which does a random split; ``test`` is currently aliased to ``val``.
    TODO (Wave 4): wire a proper held-out test split once P6 lands.
    """

    name = "movielens_ctr"
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

    def __init__(self, data_cfg: dict[str, Any], eval_cfg: dict[str, Any] | None = None):
        self._data_cfg = dict(data_cfg)
        self._eval_cfg = dict(eval_cfg or {})
        # Wave 4 (P6): declarative record of split + negative-sampler
        # choices. The split is still performed inside the underlying
        # MovieLens builder because moving it would require touching
        # data/builders/, which is out of scope for this wave. Storing
        # the RandomSplit object here still lets Wave 5's result store
        # hash the benchmark config (split type + fraction) without a
        # builder rewrite. The negative sampler, by contrast, really is
        # plugged into the evaluator via BenchmarkData.metadata below.
        # TODO (Wave 5+): move the actual split call out of the builder
        # and drive it from benchmark.splitter.split(...).
        self.splitter = RandomSplit(
            train_fraction=float(self._data_cfg.get("train_split", 0.8)),
        )
        self.negative_sampler = RandomUniform()

    def build(self) -> BenchmarkData:
        data_cfg = dict(self._data_cfg)
        # Force tabular input regardless of what the caller passed.
        data_cfg.setdefault("name", "movielens")
        data_cfg["model_input"] = "tabular"

        LOGGER.info(
            "MovieLensCTRBenchmark.build: instantiating datamodule name=%s",
            data_cfg.get("name"),
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
            # TODO (Wave 4): replace with a held-out test split.
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
