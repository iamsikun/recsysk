"""CTR task: AUC/LogLoss + sampled-negative ranking metrics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from recsys.algorithms.base import TaskType
from recsys.evaluation import CTREvaluator
from recsys.evaluation.evaluator import CTR_METRIC_NAMES, iter_predictions
from recsys.tasks.base import Task
from recsys.utils import TASK_REGISTRY

LOGGER = logging.getLogger(__name__)


@TASK_REGISTRY.register("ctr")
class CTRTask(Task):
    """Point-prediction task computing CTR and ranking metrics."""

    task_type = TaskType.CTR
    required_roles: set[str] = {"user", "item", "label"}

    def evaluate(
        self,
        algo: Any,
        benchmark_data: Any,
        metric_names: list[str],
        max_users_override: int | None = None,
    ) -> dict[str, float]:
        """Run :class:`CTREvaluator.evaluate_full` and filter the result.

        The ``algo`` argument may be a Lightning task wrapping the fitted
        torch module; :class:`CTREvaluator` pulls the actual forward-pass
        target out via ``task.model`` when available.
        """
        model = getattr(algo, "model", algo)
        datamodule = benchmark_data.datamodule
        eval_cfg = benchmark_data.metadata.get("eval", {})
        n_negatives = int(eval_cfg.get("n_negatives", 100))
        seed = int(eval_cfg.get("seed", 42))
        max_users = eval_cfg.get("max_users")
        if max_users_override is not None:
            max_users = max_users_override

        # Wave 4 (P6): forward the benchmark's declared negative sampler
        # into the evaluator so sampled-100 ranking negatives come from
        # the extracted ``recsys.data.negatives`` module rather than an
        # inline loop. Absent metadata.negative_sampler, the evaluator
        # falls back to a fresh ``RandomUniform`` (legacy compat).
        negative_sampler = benchmark_data.metadata.get("negative_sampler")

        ctr_metrics_requested = [m for m in metric_names if m in CTR_METRIC_NAMES]
        evaluator = CTREvaluator(metrics=ctr_metrics_requested or None)
        full = evaluator.evaluate_full(
            model,
            datamodule,
            n_negatives=n_negatives,
            seed=seed,
            max_users=max_users,
            negative_sampler=negative_sampler,
        )
        filtered = {name: full[name] for name in metric_names if name in full}
        missing = [name for name in metric_names if name not in full]
        if missing:
            LOGGER.warning(
                "CTRTask.evaluate: requested metrics not produced by "
                "CTREvaluator.evaluate_full: %s",
                missing,
            )
        return filtered

    def export_predictions(
        self,
        algo: Any,
        benchmark: Any,
        benchmark_data: Any,
        out_path: Path,
    ) -> None:
        """Score every row in ``benchmark_data.test`` and hand the
        results to ``benchmark.write_submission``.

        The benchmark owns the output shape — CSV row/column layout,
        header name, any id translation from dataset row-index to
        competition row-id. This keeps the harness dataset-agnostic.
        """
        model = getattr(algo, "model", algo)
        test_dataset = benchmark_data.test
        if test_dataset is None:
            raise ValueError(
                "CTRTask.export_predictions: benchmark_data.test is None"
            )
        predictions = iter_predictions(model, test_dataset)
        benchmark.write_submission(predictions, Path(out_path))
