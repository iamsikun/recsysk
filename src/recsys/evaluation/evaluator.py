"""CTR evaluator.

Iterates a validation/test dataloader, runs the model's forward pass to
obtain logits, applies sigmoid, and reports AUC + LogLoss on the collected
predictions. Phase 5 adds :meth:`CTREvaluator.evaluate_full` which extends
the CTR set with sampled-100 negative ranking metrics — both for tabular
CTR algos (DeepFM on movielens_ctr) and dict-batch sequential algos (DIN
on movielens_seq).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

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

        Ranking metrics use the sampled-K negatives protocol: for each
        evaluated positive interaction, sample ``n_negatives`` items the
        user hasn't seen in the val set and rank them against the held-out
        positive. The evaluation is capped at ``max_users`` users (default
        ``None`` == all) to keep the smoke gate cheap.

        Two input shapes are supported:

        * Tabular CTR (e.g. DeepFM on ``movielens_ctr``). Each val_dataset
          row is ``(tensor, label)`` where ``tensor`` is a 1-D feature
          vector indexed by ``feature_map`` insertion order. Candidates
          are scored by cloning the row and mutating the ``item_id``
          column.
        * Dict-batch sequential (e.g. DIN on ``movielens_seq``). Each row
          is ``({item_id, hist_item_id, hist_item_id_mask, sparse_features,
          dense_features?}, label)``. Candidates are scored by stacking
          the same history/user-context tensors and varying the target
          ``item_id`` tensor. The user identity is read from
          ``sparse_features[sparse_feature_names.index("user_id")]`` on
          the underlying algo module.
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

        result: dict[str, float] = {
            "auc": float(ctr_metrics.get("auc", float("nan"))),
            "logloss": float(ctr_metrics.get("logloss", float("nan"))),
        }
        # Pre-seed ranking keys; the branches below overwrite them.
        for k in ranking_keys:
            result[k] = float("nan")

        val_dataset = getattr(datamodule, "val_dataset", None)
        feature_map: dict[str, int] = getattr(datamodule, "feature_map", {}) or {}
        if val_dataset is None or not feature_map:
            LOGGER.warning(
                "evaluate_full: no val_dataset/feature_map; skipping ranking metrics"
            )
            return result

        if "item_id" not in feature_map:
            LOGGER.warning(
                "evaluate_full: feature_map missing item_id; skipping ranking metrics"
            )
            return result
        n_items = int(feature_map["item_id"])

        # ---- Probe one row to pick the right ranking path. ----
        first = val_dataset[0]
        if not isinstance(first, (tuple, list)) or len(first) < 2:
            LOGGER.warning(
                "evaluate_full: val_dataset rows must be (x, y); got %s",
                type(first).__name__,
            )
            return result

        sample_x = first[0]
        sampler = negative_sampler if negative_sampler is not None else RandomUniform()

        if isinstance(sample_x, dict):
            ranking_metrics = self._dict_batch_ranking(
                model=model,
                val_dataset=val_dataset,
                n_items=n_items,
                n_negatives=n_negatives,
                seed=seed,
                max_users=max_users,
                device=device,
                sampler=sampler,
            )
            result.update(ranking_metrics)
            return result

        # ---- Tabular ranking path. ----
        feature_names = list(feature_map.keys())
        if "user_id" not in feature_names:
            LOGGER.warning(
                "evaluate_full: feature_map missing user_id; "
                "skipping tabular ranking metrics. keys=%s",
                feature_names,
            )
            return result
        user_col = feature_names.index("user_id")
        item_col = feature_names.index("item_id")

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

        # Negative sampling is driven by the extracted
        # ``recsys.data.negatives.random_uniform.RandomUniform`` (or any
        # sampler the benchmark forwards via ``negative_sampler``). The
        # ``sampler`` variable was bound earlier in the common preamble.
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
                    ranked = [int(item_ids[k]) for k in order]
                    per_user_gt.append([int(pos_it)])
                    per_user_preds.append(ranked)
        finally:
            if was_training:
                model.train()

        if not per_user_gt:
            LOGGER.warning("evaluate_full: ranking loop produced no users")
            return result

        result.update(_compute_ranking_metrics(per_user_gt, per_user_preds))
        return result

    def _dict_batch_ranking(
        self,
        model: torch.nn.Module,
        val_dataset: Any,
        n_items: int,
        n_negatives: int,
        seed: int,
        max_users: int | None,
        device: torch.device,
        sampler: Any,
    ) -> dict[str, float]:
        """Sampled-K ranking evaluation for dict-batch sequential models.

        For each user with a positive row in ``val_dataset``, take the
        first such row (= the earliest positive event under that user's
        history-prefix schedule), sample ``n_negatives`` item IDs the user
        hasn't seen, then score the positive target against the negatives
        by stacking the row's history/user-context tensors ``K+1`` times
        and varying only the target ``item_id`` column. The ranked item
        list feeds the same ranking metric functions the tabular path
        uses. Returns a dict with keys ``ndcg@10``, ``ndcg@50``,
        ``recall@10``, ``recall@50``, ``hr@10``, ``hr@50``, ``mrr``.

        The user identity comes from
        ``sparse_features[sparse_feature_names.index("user_id")]`` on the
        wrapped algo module. If that layout is missing we fall back to
        giving each row its own singleton user, which yields a strictly
        per-event ranking (still a valid sampled-K protocol).
        """
        item_feature = getattr(model, "item_feature", "item_id")
        sparse_feature_names: list[str] = list(
            getattr(model, "sparse_feature_names", []) or []
        )
        user_sparse_idx = (
            sparse_feature_names.index("user_id")
            if "user_id" in sparse_feature_names
            else None
        )

        # ---- Bulk-read the dataset tensors. ----
        # SequenceDataset stores ``features: dict[str, Tensor]`` (each tensor
        # has shape (N, ...)) and ``labels: Tensor`` of shape (N, 1) or (N,).
        # The MovieLens builder wraps the SequenceDataset in
        # ``torch.utils.data.Subset`` via ``random_split``; unwrap to reach the
        # bulk tensors and remember the row indices into them.
        from torch.utils.data import Subset

        if isinstance(val_dataset, Subset):
            base_dataset = val_dataset.dataset
            row_indices = np.asarray(val_dataset.indices, dtype=np.int64)
        else:
            base_dataset = val_dataset
            row_indices = None

        features = getattr(base_dataset, "features", None)
        labels = getattr(base_dataset, "labels", None)
        if not isinstance(features, dict) or labels is None:
            LOGGER.warning(
                "evaluate_full: dict-batch val_dataset does not expose "
                "`features`/`labels`; cannot compute ranking metrics"
            )
            return {}
        if item_feature not in features:
            LOGGER.warning(
                "evaluate_full: dict-batch val_dataset missing target feature "
                "%s; available keys=%s",
                item_feature,
                sorted(features.keys()),
            )
            return {}

        item_col_full = features[item_feature].detach().cpu().numpy().reshape(-1)
        labels_full = labels.detach().cpu().numpy().reshape(-1)
        if row_indices is not None:
            item_col = item_col_full[row_indices]
            labels_arr = labels_full[row_indices]
        else:
            item_col = item_col_full
            labels_arr = labels_full
            row_indices = np.arange(len(item_col), dtype=np.int64)
        n_rows = len(item_col)

        if user_sparse_idx is not None and "sparse_features" in features:
            user_col_full = (
                features["sparse_features"][:, user_sparse_idx]
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            user_col = user_col_full[row_indices]
        else:
            # Degenerate fallback: each row is its own user. Valid
            # per-event ranking, just coarser aggregation.
            user_col = np.arange(n_rows, dtype=np.int64)

        # First positive row index per user, plus union of interacted
        # items per user (for negative-sampling exclusion).
        user_positive_idx: dict[int, int] = {}
        user_interacted: dict[int, set[int]] = {}
        for i in range(n_rows):
            u = int(user_col[i])
            it = int(item_col[i])
            bucket = user_interacted.setdefault(u, set())
            bucket.add(it)
            if labels_arr[i] > 0.0 and u not in user_positive_idx:
                user_positive_idx[u] = i

        eligible_users = sorted(user_positive_idx.keys())
        if not eligible_users:
            LOGGER.warning(
                "evaluate_full: no positive dict-batch rows found in val_dataset"
            )
            return {}
        if max_users is not None and len(eligible_users) > int(max_users):
            eligible_users = eligible_users[: int(max_users)]

        rng = np.random.default_rng(seed)

        per_user_gt: list[list[int]] = []
        per_user_preds: list[list[int]] = []

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for u in eligible_users:
                    subset_i = user_positive_idx[u]
                    base_row_idx = int(row_indices[subset_i])
                    pos_it = int(item_col[subset_i])
                    interacted = user_interacted.get(u, set())
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
                    candidate_batch_size = len(item_ids)

                    # Stack the positive row's tensors K+1 times, then
                    # overwrite the target item column with the candidate
                    # IDs. Everything else (history, sparse, dense) is
                    # shared across the candidates, which is exactly the
                    # standard sampled-ranking protocol for a
                    # history-aware CTR scorer.
                    batch: dict[str, torch.Tensor] = {}
                    for key, tensor in features.items():
                        row_tensor = tensor[base_row_idx]
                        stacked = (
                            row_tensor.unsqueeze(0)
                            .expand(candidate_batch_size, *row_tensor.shape)
                            .contiguous()
                            .to(device)
                        )
                        batch[key] = stacked

                    target_key = item_feature if item_feature in batch else "target_item"
                    target_tensor = torch.tensor(
                        item_ids,
                        dtype=batch[target_key].dtype,
                        device=device,
                    )
                    batch[target_key] = target_tensor

                    logits = model(batch)
                    scores = logits.detach().float().cpu().numpy().reshape(-1)
                    order = np.argsort(-scores)
                    ranked = [int(item_ids[k]) for k in order]
                    per_user_gt.append([int(pos_it)])
                    per_user_preds.append(ranked)
        finally:
            if was_training:
                model.train()

        if not per_user_gt:
            LOGGER.warning(
                "evaluate_full: dict-batch ranking loop produced no users"
            )
            return {}

        return _compute_ranking_metrics(per_user_gt, per_user_preds)


def _compute_ranking_metrics(
    per_user_gt: list[list[int]],
    per_user_preds: list[list[int]],
) -> dict[str, float]:
    """Shared metric computation used by both tabular and dict-batch paths."""
    return {
        "ndcg@10": float(ndcg_at_k(per_user_gt, per_user_preds, 10)),
        "ndcg@50": float(ndcg_at_k(per_user_gt, per_user_preds, 50)),
        "recall@10": float(recall_at_k(per_user_gt, per_user_preds, 10)),
        "recall@50": float(recall_at_k(per_user_gt, per_user_preds, 50)),
        "hr@10": float(hr_at_k(per_user_gt, per_user_preds, 10)),
        "hr@50": float(hr_at_k(per_user_gt, per_user_preds, 50)),
        "mrr": float(mrr(per_user_gt, per_user_preds)),
    }


def _to_device(x, device: torch.device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        moved = [_to_device(v, device) for v in x]
        return type(x)(moved) if isinstance(x, tuple) else moved
    return x


def iter_predictions(
    model: torch.nn.Module,
    dataset: Any,
    *,
    batch_size: int = 1024,
    device: torch.device | str | None = None,
) -> Iterable[tuple[int, float]]:
    """Yield ``(row_id, score)`` pairs for every row in ``dataset``.

    Shape-agnostic: handles both ``(x, y)`` tuple datasets (TabularDataset)
    and ``(dict, y)`` datasets (SequenceDataset / TabularDictDataset).
    ``row_id`` is the dataset row index — benchmarks that want external
    ids can override :meth:`recsys.benchmarks.base.Benchmark.write_submission`
    and translate.

    Labels are read but ignored so this function works against
    *unlabeled* test splits as well as labeled ones. Scores are
    sigmoid(logits); callers that want raw logits can wrap.
    """
    from torch.utils.data import DataLoader, Subset

    indices = (
        list(dataset.indices) if isinstance(dataset, Subset) else None
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device) if not isinstance(device, torch.device) else device

    def _collate(samples):
        # samples: list of (x, y) where x is either a Tensor or a dict.
        first_x = samples[0][0]
        labels = torch.stack(
            [torch.as_tensor(s[1]) for s in samples], dim=0
        )
        if isinstance(first_x, dict):
            batch = {
                key: torch.stack([s[0][key] for s in samples], dim=0)
                for key in first_x
            }
            return batch, labels
        xs = torch.stack([s[0] for s in samples], dim=0)
        return xs, labels

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            row_offset = 0
            for batch in loader:
                x, _y = batch
                x = _to_device(x, device)
                logits = model(x)
                probs = torch.sigmoid(logits).detach().float().cpu().numpy().reshape(-1)
                for i, score in enumerate(probs):
                    if indices is not None:
                        yield int(indices[row_offset + i]), float(score)
                    else:
                        yield row_offset + i, float(score)
                row_offset += len(probs)
    finally:
        if was_training:
            model.train()
