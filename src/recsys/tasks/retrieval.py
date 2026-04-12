"""Retrieval task stub.

Wave 3 ships CTR only; top-k retrieval lands in Wave 4 together with the
split/negative sampling extraction work. Registered so that benchmark
configs can already reference the name.
"""

from __future__ import annotations

from typing import Any

from recsys.algorithms.base import TaskType
from recsys.tasks.base import Task
from recsys.utils import TASK_REGISTRY


@TASK_REGISTRY.register("retrieval")
class RetrievalTask(Task):
    task_type = TaskType.RETRIEVAL
    required_roles: set[str] = {"user", "item"}

    def evaluate(
        self,
        algo: Any,
        benchmark_data: Any,
        metric_names: list[str],
    ) -> dict[str, float]:
        raise NotImplementedError(
            "RetrievalTask.evaluate is not implemented yet; deferred to Wave 4."
        )
