"""Abstract Task protocol.

A Task encapsulates the I/O contract between an Algorithm and an evaluator.
It owns the evaluation logic for a given task type (CTR, retrieval,
sequential, ...), taking an Algorithm-like object plus a ``BenchmarkData``
bundle and returning a metric dict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from recsys.algorithms.base import TaskType


class Task(ABC):
    """Abstract base for evaluation tasks."""

    #: The task type this task implements.
    task_type: TaskType

    #: Names of feature roles the task needs from the data module.
    required_roles: set[str] = set()

    @abstractmethod
    def evaluate(
        self,
        algo: Any,
        benchmark_data: Any,
        metric_names: list[str],
    ) -> dict[str, float]:
        """Compute the requested metrics on ``benchmark_data``.

        Args:
            algo: Fitted algorithm or Lightning task wrapping a fitted model.
            benchmark_data: A :class:`BenchmarkData` bundle providing the
                train/val/test splits and datamodule handle.
            metric_names: Names of metrics to compute. Unknown keys are
                filtered out.
        """
        raise NotImplementedError
