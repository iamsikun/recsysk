"""Minimal CTR evaluator used during Phase 2 of the migration.

Iterates a validation/test dataloader, runs the model's forward pass to
obtain logits, applies sigmoid, and reports AUC + LogLoss on the collected
predictions. Intentionally small — this will be folded into the
``evaluation.Evaluator`` protocol in later phases.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import torch

from recsys.metrics.ctr import auc as _auc
from recsys.metrics.ctr import logloss as _logloss

LOGGER = logging.getLogger(__name__)

_METRIC_FNS = {
    "auc": _auc,
    "logloss": _logloss,
}


class CTREvaluator:
    """Compute CTR metrics on a dataloader.

    Parameters
    ----------
    metrics:
        Optional list of metric names. Defaults to ``["auc", "logloss"]``.
    """

    def __init__(self, metrics: list[str] | None = None) -> None:
        self.metrics = list(metrics) if metrics is not None else ["auc", "logloss"]
        unknown = [m for m in self.metrics if m not in _METRIC_FNS]
        if unknown:
            raise ValueError(
                f"Unknown CTR metrics: {unknown}. "
                f"Supported: {sorted(_METRIC_FNS)}"
            )

    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: Iterable,
        device: torch.device | str,
    ) -> dict[str, float]:
        device = torch.device(device) if not isinstance(device, torch.device) else device

        preds: list[np.ndarray] = []
        labels: list[np.ndarray] = []

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for batch in dataloader:
                    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                        raise ValueError(
                            "CTREvaluator expects each batch to be a (x, y) tuple; "
                            f"got {type(batch).__name__}"
                        )
                    x, y = batch[0], batch[1]
                    x = _to_device(x, device)
                    logits = model(x)
                    probs = torch.sigmoid(logits).detach().float().cpu().numpy()
                    y_np = y.detach().float().cpu().numpy()
                    preds.append(probs.reshape(-1))
                    labels.append(y_np.reshape(-1))
        finally:
            if was_training:
                model.train()

        if not preds:
            LOGGER.warning("CTREvaluator received an empty dataloader; returning {}")
            return {}

        y_score = np.concatenate(preds)
        y_true = np.concatenate(labels)

        results: dict[str, float] = {}
        for name in self.metrics:
            fn = _METRIC_FNS[name]
            try:
                results[name] = float(fn(y_true, y_score))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to compute metric %s: %s", name, exc)
                results[name] = float("nan")
        return results


def _to_device(x, device: torch.device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        moved = [_to_device(v, device) for v in x]
        return type(x)(moved) if isinstance(x, tuple) else moved
    return x
