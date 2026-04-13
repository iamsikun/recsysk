"""MovieLens sequential benchmark for DIN-style history-aware models."""

from __future__ import annotations

import logging
from typing import Any

from recsys.benchmarks.base import Benchmark, BenchmarkData
from recsys.data.negatives.random_uniform import RandomUniform
from recsys.data.splits.random_split import RandomSplit
from recsys.tasks.ctr import CTRTask
from recsys.utils import BENCHMARK_REGISTRY, DATASET_REGISTRY

LOGGER = logging.getLogger(__name__)


@BENCHMARK_REGISTRY.register("movielens_seq")
class MovieLensSeqBenchmark(Benchmark):
    """MovieLens sequential (history-aware) CTR benchmark.

    Same pinned metric list as :class:`MovieLensCTRBenchmark`, but the
    datamodule is built with ``model_input: sequence`` so that dict-valued
    batches (``{item_id, hist_item_id, ...}``) flow to sequence-aware
    algorithms like DIN.

    Ranking-metric protocol: sampled-K negatives per positive event.
    :class:`recsys.evaluation.evaluator.CTREvaluator.evaluate_full` takes
    the first positive dict-row per user, samples ``eval.n_negatives``
    items the user hasn't interacted with from the item vocabulary, and
    scores them against the positive target by stacking the row's
    history/user-context tensors and varying only the target ``item_id``
    column. ``user_id`` is read from ``sparse_features`` using the algo's
    ``sparse_feature_names`` list; the benchmark therefore requires
    ``user_id`` to appear in ``sparse_feature_names`` for the ranking
    path to group by user.
    """

    name = "movielens_seq"
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
        # choices. The MovieLensSeqBuilder still performs the split (and
        # its own train-time negative sampling) inline — that migration
        # is deferred because data/builders/ is out of scope for P6.
        # Storing the RandomSplit object here lets Wave 5's result store
        # hash the benchmark config without a builder rewrite. The
        # eval-time random-uniform negative sampler is actually consumed
        # by the evaluator via BenchmarkData.metadata below (dict-batch
        # limited — sequential ranking still returns NaN in Wave 4).
        # TODO (Wave 5+): move the split and the train-time negative
        # sampler out of the builder and drive them from these modules.
        self.splitter = RandomSplit(
            train_fraction=float(self._data_cfg.get("train_split", 0.8)),
        )
        self.negative_sampler = RandomUniform()

    def build(self) -> BenchmarkData:
        data_cfg = dict(self._data_cfg)
        data_cfg.setdefault("name", "movielens")
        data_cfg["model_input"] = "sequence"
        # Defaults for sequential DIN-style input. Callers can override via
        # the benchmark config.
        data_cfg.setdefault("item_feature", "item_id")
        data_cfg.setdefault("history_feature", "hist_item_id")
        data_cfg.setdefault("sparse_feature_names", ["user_id"])
        data_cfg.setdefault("max_history_len", 20)

        LOGGER.info(
            "MovieLensSeqBenchmark.build: instantiating sequence datamodule"
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
            # TODO (Wave 4): proper held-out test split.
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
