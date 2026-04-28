"""CTR-style evaluation metrics (point-prediction)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid.
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC.

    Accepts either probabilities in [0, 1] or logits (AUC is scale/shift
    invariant, so no sigmoid is required).
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    return float(roc_auc_score(y_true, y_score))


def logloss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute binary log loss.

    Accepts logits or probabilities. If any value lies outside [0, 1], the
    inputs are assumed to be logits and a sigmoid is applied first.
    """
    y_true = np.asarray(y_true).reshape(-1).astype(np.float64)
    y_score = np.asarray(y_score).reshape(-1).astype(np.float64)
    if y_score.size == 0:
        return float("nan")
    if y_score.min() < 0.0 or y_score.max() > 1.0:
        y_score = _sigmoid(y_score)
    # Clip to avoid log(0) inside sklearn's log_loss.
    eps = 1e-7
    y_score = np.clip(y_score, eps, 1.0 - eps)
    return float(log_loss(y_true, y_score, labels=[0, 1]))


def ne(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute normalized entropy (logloss / entropy of the base rate).

    Standard CTR metric in the unified seq + non-seq line (Wukong, HSTU, etc.):
    logloss divided by the entropy of a model that always predicts the
    label mean. NE < 1 means the model beats the constant-base-rate predictor.
    Returns NaN if the labels are degenerate (all 0s or all 1s).
    """
    y_true = np.asarray(y_true).reshape(-1).astype(np.float64)
    if y_true.size == 0:
        return float("nan")
    p = float(y_true.mean())
    if p <= 0.0 or p >= 1.0:
        return float("nan")
    base_entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return float(logloss(y_true, y_score) / base_entropy)
