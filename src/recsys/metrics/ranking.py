"""Ranking / retrieval metrics.

Each metric takes per-user ground-truth relevant items and per-user ranked
predictions. ``ground_truth`` may be a list/set/array of relevant item IDs
for each user. ``predictions`` is an ordered sequence (best-first) of item
IDs for each user.

All metrics return the mean value across users. Users with an empty ground
truth set are skipped. Currently unused by Phase 2 but wired for Phase 5.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _to_set(items: Iterable) -> set:
    return set(int(i) for i in items)


def _iter_users(
    ground_truth: Sequence[Iterable],
    predictions: Sequence[Sequence],
):
    if len(ground_truth) != len(predictions):
        raise ValueError(
            f"ground_truth and predictions length mismatch: "
            f"{len(ground_truth)} vs {len(predictions)}"
        )
    for gt, pred in zip(ground_truth, predictions):
        gt_set = _to_set(gt)
        if not gt_set:
            continue
        yield gt_set, list(pred)


def ndcg_at_k(
    ground_truth: Sequence[Iterable],
    predictions: Sequence[Sequence],
    k: int,
) -> float:
    """Mean NDCG@k with binary relevance.

    DCG = sum_i rel_i / log2(i + 2), i in [0, k).
    IDCG is computed assuming as many relevant items as possible in the
    top-k positions.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    scores = []
    log2 = np.log2(np.arange(2, k + 2))  # log2(2)..log2(k+1)
    for gt_set, pred in _iter_users(ground_truth, predictions):
        topk = pred[:k]
        rels = np.array([1.0 if p in gt_set else 0.0 for p in topk])
        if rels.size < k:
            rels = np.concatenate([rels, np.zeros(k - rels.size)])
        dcg = float(np.sum(rels / log2))
        n_rel = min(len(gt_set), k)
        idcg = float(np.sum(1.0 / log2[:n_rel])) if n_rel > 0 else 0.0
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    if not scores:
        return 0.0
    return float(np.mean(scores))


def recall_at_k(
    ground_truth: Sequence[Iterable],
    predictions: Sequence[Sequence],
    k: int,
) -> float:
    """Mean Recall@k: |relevant ∩ top-k| / |relevant|."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    scores = []
    for gt_set, pred in _iter_users(ground_truth, predictions):
        topk = set(int(p) for p in pred[:k])
        hits = len(gt_set & topk)
        scores.append(hits / len(gt_set))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def hr_at_k(
    ground_truth: Sequence[Iterable],
    predictions: Sequence[Sequence],
    k: int,
) -> float:
    """Mean Hit Rate @k: 1 if any relevant item is in top-k, else 0."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    scores = []
    for gt_set, pred in _iter_users(ground_truth, predictions):
        topk = set(int(p) for p in pred[:k])
        scores.append(1.0 if gt_set & topk else 0.0)
    if not scores:
        return 0.0
    return float(np.mean(scores))


def mrr(
    ground_truth: Sequence[Iterable],
    predictions: Sequence[Sequence],
    k: int | None = None,
) -> float:
    """Mean Reciprocal Rank.

    For each user, 1 / rank of the first relevant item in the ranked list
    (1-indexed). If no relevant item appears (optionally within the top-k),
    contributes 0. Averaged across users with non-empty ground truth.
    """
    scores = []
    for gt_set, pred in _iter_users(ground_truth, predictions):
        limit = len(pred) if k is None else min(k, len(pred))
        rr = 0.0
        for i in range(limit):
            if int(pred[i]) in gt_set:
                rr = 1.0 / (i + 1)
                break
        scores.append(rr)
    if not scores:
        return 0.0
    return float(np.mean(scores))
