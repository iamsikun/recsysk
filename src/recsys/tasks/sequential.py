"""Sequential task stub.

Wave 3 ships CTR only; the sequential evaluation path (next-item
prediction against a history buffer) lands in a later wave.
"""

from __future__ import annotations

from typing import Any

from recsys.algorithms.base import TaskType
from recsys.tasks.base import Task
from recsys.utils import TASK_REGISTRY


@TASK_REGISTRY.register("sequential")
class SequentialTask(Task):
    task_type = TaskType.SEQUENTIAL
    required_roles: set[str] = {"user", "item", "sequence"}

    def evaluate(
        self,
        algo: Any,
        benchmark_data: Any,
        metric_names: list[str],
    ) -> dict[str, float]:
        raise NotImplementedError(
            "SequentialTask.evaluate is not implemented yet; deferred to "
            "a later wave."
        )
