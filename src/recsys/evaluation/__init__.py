"""Evaluation module — CTR evaluator + parquet result store + reporting."""

from recsys.evaluation.evaluator import CTREvaluator
from recsys.evaluation.reporting import format_table, summary_table
from recsys.evaluation.store import ResultStore, RunResult

__all__ = [
    "CTREvaluator",
    "ResultStore",
    "RunResult",
    "evaluate_full",
    "format_table",
    "summary_table",
]


def evaluate_full(*args, **kwargs):  # pragma: no cover - convenience alias
    """Module-level alias for :meth:`CTREvaluator.evaluate_full`."""
    return CTREvaluator().evaluate_full(*args, **kwargs)

