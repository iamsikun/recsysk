"""Evaluation module — currently exposes the Phase 2 CTR evaluator."""

from recsys.evaluation.evaluator import CTREvaluator

__all__ = ["CTREvaluator", "evaluate_full"]


def evaluate_full(*args, **kwargs):  # pragma: no cover - convenience alias
    """Module-level alias for :meth:`CTREvaluator.evaluate_full`."""
    return CTREvaluator().evaluate_full(*args, **kwargs)

