"""CTR evaluator.

Iterates a validation/test dataloader, runs the model's forward pass to
obtain logits, applies sigmoid, and reports AUC + LogLoss on the collected
predictions. Phase 5 adds :meth:`CTREvaluator.evaluate_full` which extends
the CTR set with sampled-100 negative ranking metrics.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import torch

from recsys.data.negatives.random_uniform import RandomUniform
from recsys.metrics.ctr import auc as _auc
from recsys.metrics.ctr import logloss as _logloss
from recsys.metrics.ranking import hr_at_k, mrr, ndcg_at_k, recall_at_k

LOGGER = logging.getLogger(__name__)

_METRIC_FNS = {
    "auc": _auc,
    "logloss": _logloss,
}

# Max users evaluated for ranking metrics in smoke mode.
_SMOKE_MAX_USERS = 200

# Emit the dict-batch ranking warning only once per process.
_DICT_BATCH_WARNED = False


class CTREvaluator:
    """Compute CTR + ranking metrics on a dataloader/datamodule.

    Parameters
    ----------
    metrics:
        Optional list of metric names for the legacy :meth:`evaluate`
        entrypoint. Defaults to ``["auc", "logloss"]``.
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

    def evaluate_full(
        self,
        model: torch.nn.Module,
        datamodule,
        n_negatives: int = 100,
        seed: int = 42,
        device: torch.device | str | None = None,
        max_users: int | None = None,
        negative_sampler=None,
    ) -> dict[str, float]:
        """Compute the full CTR + ranking metric set.

        Returns a dict with keys: ``auc``, ``logloss``, ``ndcg@10``,
        ``ndcg@50``, ``recall@10``, ``recall@50``, ``hr@10``, ``hr@50``,
        ``mrr``.

        Ranking metrics use the sampled-100 negatives protocol: for each
        evaluated positive interaction, sample ``n_negatives`` items the
        user hasn't seen in the val set and rank them against the held-out
        positive. The evaluation is capped at ``max_users`` users (default
        ``None`` == all) to keep the smoke gate cheap.

        For dict-batch algorithms (e.g. DIN sequential) the ranking step
        cannot easily synthesise a fake batch without access to the
        history column, so those metrics are returned as NaN and a warning
        is emitted once per process.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device) if not isinstance(device, torch.device) else device

        # ---- CTR metrics (AUC, LogLoss) via legacy path. ----
        val_loader = datamodule.val_dataloader()
        model.to(device)
        ctr_metrics = self.evaluate(model, val_loader, device)

        ranking_keys = [
            "ndcg@10",
            "ndcg@50",
            "recall@10",
            "recall@50",
            "hr@10",
            "hr@50",
            "mrr",
        ]

        # Pre-seed result with NaNs so we always return all keys.
        result: dict[str, float] = {
            "auc": float(ctr_metrics.get("auc", float("nan"))),
            "logloss": float(ctr_metrics.get("logloss", float("nan"))),
        }
        for k in ranking_keys:
            result[k] = float("nan")

        val_dataset = getattr(datamodule, "val_dataset", None)
        feature_map: dict[str, int] = getattr(datamodule, "feature_map", {}) or {}
        if val_dataset is None or not feature_map:
            LOGGER.warning(
                "evaluate_full: no val_dataset/feature_map; skipping ranking metrics"
            )
            return result

        # Column indices: rely on dict insertion order used everywhere in
        # the codebase (see Popularity.__init__ for the same assumption).
        feature_names = list(feature_map.keys())
        if "user_id" not in feature_names or "item_id" not in feature_names:
            LOGGER.warning(
                "evaluate_full: feature_map missing user_id or item_id; "
                "skipping ranking metrics. keys=%s",
                feature_names,
            )
            return result
        user_col = feature_names.index("user_id")
        item_col = feature_names.index("item_id")
        n_items = int(feature_map["item_id"])

        # ---- Probe one row to detect dict-batch limitation. ----
        try:
            first = val_dataset[0]
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("evaluate_full: val_dataset indexing failed: %s", exc)
            return result

        if not isinstance(first, (tuple, list)) or len(first) < 2:
            LOGGER.warning(
                "evaluate_full: val_dataset rows must be (x, y); got %s",
                type(first).__name__,
            )
            return result

        sample_x = first[0]
        if not isinstance(sample_x, torch.Tensor):
            # Dict-batch path (DIN sequential) — synthesising a fake batch
            # for sampled-100 ranking would require fabricating history
            # columns. Wave 3 leaves this NaN and documents the limitation.
            global _DICT_BATCH_WARNED
            if not _DICT_BATCH_WARNED:
                LOGGER.warning(
                    "evaluate_full: dict-batch inputs (e.g. DIN sequential) "
                    "not supported for ranking metrics; returning NaN. "
                    "This limitation is tracked as a follow-up."
                )
                _DICT_BATCH_WARNED = True
            return result

        # ---- Build user -> positive items, user -> all interacted items. ----
        user_positives: dict[int, list[tuple[int, torch.Tensor]]] = {}
        user_interacted: dict[int, set[int]] = {}

        for idx in range(len(val_dataset)):
            row = val_dataset[idx]
            if not isinstance(row, (tuple, list)) or len(row) < 2:
                continue
            x, y = row[0], row[1]
            if not isinstance(x, torch.Tensor):
                continue
            try:
                u = int(x[user_col].item())
                it = int(x[item_col].item())
            except Exception:  # noqa: BLE001
                continue
            label = float(y.item() if isinstance(y, torch.Tensor) else float(y))
            interacted = user_interacted.setdefault(u, set())
            interacted.add(it)
            if label > 0.0:
                positives = user_positives.setdefault(u, [])
                positives.append((it, x.detach().clone()))

        eligible_users = [u for u, plist in user_positives.items() if plist]
        if not eligible_users:
            LOGGER.warning("evaluate_full: no positive rows found in val_dataset")
            return result

        # Deterministic ordering, then cap.
        eligible_users.sort()
        if max_users is not None and len(eligible_users) > int(max_users):
            eligible_users = eligible_users[: int(max_users)]

        rng = np.random.default_rng(seed)

        # Wave 4 (P6): negative sampling extracted into
        # ``recsys.data.negatives.random_uniform.RandomUniform``. The
        # evaluator now receives the sampler via ``negative_sampler``
        # (forwarded by :class:`recsys.tasks.ctr.CTRTask.evaluate` from
        # ``benchmark_data.metadata["negative_sampler"]``). When no
        # sampler is supplied we fall back to a fresh ``RandomUniform``
        # instance so the legacy compat path keeps working byte-for-byte.
        sampler = negative_sampler if negative_sampler is not None else RandomUniform()

        per_user_gt: list[list[int]] = []
        per_user_preds: list[list[int]] = []

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for u in eligible_users:
                    pos_it, pos_row = user_positives[u][0]
                    interacted = user_interacted.get(u, set())
                    # Sampled-K negatives via the extracted
                    # RandomUniform sampler. ``exclude`` is the union of
                    # the user's interacted items and the held-out
                    # positive, matching the pre-Wave-4 inline loop's
                    # reject condition (cand in interacted or cand == pos_it).
                    exclude = interacted | {pos_it}
                    negatives_arr = sampler.sample(
                        n_negatives=n_negatives,
                        exclude=exclude,
                        vocab_size=n_items,
                        rng=rng,
                    )
                    negatives = [int(x) for x in negatives_arr.tolist()]
                    if not negatives:
                        continue
                    item_ids = [pos_it, *negatives]
                    batch = pos_row.unsqueeze(0).repeat(len(item_ids), 1).to(device)
                    batch[:, item_col] = torch.tensor(
                        item_ids, dtype=batch.dtype, device=device
                    )
                    try:
                        logits = model(batch)
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning(
                            "evaluate_full: model forward failed for user %s: %s",
                            u,
                            exc,
                        )
                        continue
                    scores = logits.detach().float().cpu().numpy().reshape(-1)
                    # Argsort descending.
                    order = np.argsort(-scores)
                    ranked = [int(item_ids[i]) for i in order]
                    per_user_gt.append([int(pos_it)])
                    per_user_preds.append(ranked)
        finally:
            if was_training:
                model.train()

        if not per_user_gt:
            LOGGER.warning("evaluate_full: ranking loop produced no users")
            return result

        try:
            result["ndcg@10"] = float(ndcg_at_k(per_user_gt, per_user_preds, 10))
            result["ndcg@50"] = float(ndcg_at_k(per_user_gt, per_user_preds, 50))
            result["recall@10"] = float(recall_at_k(per_user_gt, per_user_preds, 10))
            result["recall@50"] = float(recall_at_k(per_user_gt, per_user_preds, 50))
            result["hr@10"] = float(hr_at_k(per_user_gt, per_user_preds, 10))
            result["hr@50"] = float(hr_at_k(per_user_gt, per_user_preds, 50))
            result["mrr"] = float(mrr(per_user_gt, per_user_preds))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("evaluate_full: ranking metric computation failed: %s", exc)

        return result


def _to_device(x, device: torch.device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        moved = [_to_device(v, device) for v in x]
        return type(x)(moved) if isinstance(x, tuple) else moved
    return x
