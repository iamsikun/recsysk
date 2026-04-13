"""Popularity baseline for CTR prediction.

Framework-agnostic item-popularity baseline: the score for an item is a
monotonic transform (log1p, max-normalised, mean-centred) of how many
times it appeared in the training set. Popularity is a classical
algorithm — it does *not* subclass ``nn.Module`` and does not import
``torch`` at module scope. The runner's classical branch calls
:meth:`Popularity.fit` directly, bypassing Lightning entirely.
"""

from __future__ import annotations

import logging

from recsys.algorithms.base import Algorithm, TaskType
from recsys.utils import ALGO_REGISTRY

LOGGER = logging.getLogger(__name__)


@ALGO_REGISTRY.register("popularity")
class Popularity(Algorithm):
    """Item-popularity CTR baseline.

    Parameters
    ----------
    feature_map:
        Mapping of feature names to their vocabulary sizes. The insertion
        order must match the column order used to build tabular batches,
        so that ``item_feature`` can be located by its index.
    item_feature:
        Name of the item-id feature to score on. Defaults to ``"item_id"``.
    """

    supported_tasks = {TaskType.CTR}
    required_roles = {"item", "label"}

    def __init__(
        self,
        feature_map: dict[str, int],
        item_feature: str = "item_id",
    ) -> None:
        if item_feature not in feature_map:
            raise ValueError(
                f"Popularity requires '{item_feature}' in feature_map; "
                f"got keys {list(feature_map)}"
            )
        self.item_feature = item_feature
        self.vocab_size = int(feature_map[item_feature])
        # Column index of the item feature inside tabular batches. Relies
        # on insertion-order semantics of ``dict`` (guaranteed since 3.7).
        self.item_column = list(feature_map.keys()).index(item_feature)

        # Frozen log-popularity scores, lazily materialised in ``fit``.
        # Kept as a plain torch tensor (no nn.Parameter) since this class
        # is not an nn.Module.
        self.popularity_scores = None  # type: ignore[assignment]

        # Duck-typed training flag + no-op ``.to`` / ``.eval`` / ``.train``
        # so the existing :class:`recsys.evaluation.evaluator.CTREvaluator`
        # — which was written against ``nn.Module`` — can swallow a raw
        # classical scorer unchanged. These are deliberately *not* an
        # ``nn.Module`` subclass: they exist so this file stays torch-free
        # at module scope.
        self.training = False

    def fit(self, train, val=None, batch_size: int = 8192) -> None:
        """Compute log(count+1) popularity scores from the training set.

        Iterates the dataset once, accumulates item-id occurrence counts,
        applies ``log(count + 1)``, normalizes to ``[0, 1]``, and centers
        the final logits on zero (mean-subtracted) so that ``sigmoid``
        produces values around 0.5 rather than saturating.
        """
        import torch
        from torch.utils.data import DataLoader

        LOGGER.info(
            "Popularity.fit: vocab_size=%d item_column=%d",
            self.vocab_size,
            self.item_column,
        )
        counts = torch.zeros(self.vocab_size, dtype=torch.float64)

        loader = DataLoader(
            train,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        n_seen = 0
        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise ValueError(
                        "Popularity.fit expects (x, y) batches; "
                        f"got {type(batch).__name__}"
                    )
                x = batch[0]
                if not isinstance(x, torch.Tensor):
                    raise TypeError(
                        "Popularity baseline currently only supports tabular "
                        "tensor batches (shape (B, n_fields)); "
                        f"got {type(x).__name__}"
                    )
                if x.dim() != 2:
                    raise ValueError(
                        f"Expected 2-D tabular batch, got shape {tuple(x.shape)}"
                    )
                item_ids = x[:, self.item_column].to(torch.long)
                item_ids = torch.clamp(item_ids, min=0, max=self.vocab_size - 1)
                counts.scatter_add_(
                    0, item_ids, torch.ones_like(item_ids, dtype=torch.float64)
                )
                n_seen += item_ids.numel()

        LOGGER.info(
            "Popularity.fit: saw %d interactions across %d items",
            n_seen,
            int((counts > 0).sum().item()),
        )

        log_counts = torch.log1p(counts).to(torch.float32)
        max_val = float(log_counts.max().item())
        if max_val > 0.0:
            log_counts = log_counts / max_val
        # Center so sigmoid isn't saturated at the high end.
        log_counts = log_counts - log_counts.mean()

        self.popularity_scores = log_counts

    def predict_scores(self, x):
        """Return per-sample logits based on item popularity.

        Accepts a 2-D tensor ``(B, n_fields)`` and returns a ``(B, 1)``
        logits tensor on the same device as ``x``.
        """
        import torch

        if self.popularity_scores is None:
            raise RuntimeError(
                "Popularity.predict_scores called before fit; "
                "popularity_scores have not been materialised."
            )
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "Popularity baseline only supports tabular tensor inputs; "
                f"got {type(x).__name__}"
            )
        if x.dim() != 2:
            raise ValueError(
                f"Expected 2-D tabular batch, got shape {tuple(x.shape)}"
            )
        scores = self.popularity_scores
        if scores.device != x.device:
            scores = scores.to(x.device)
            self.popularity_scores = scores
        item_ids = x[:, self.item_column].to(torch.long)
        item_ids = torch.clamp(item_ids, min=0, max=self.vocab_size - 1)
        logits = scores[item_ids].unsqueeze(-1)
        return logits

    # The CTR evaluator invokes scorers via ``model(x)``. Exposing
    # ``__call__`` (rather than forcing evaluator changes) keeps the
    # framework-agnostic classical algo compatible with the existing
    # evaluator without pulling torch / nn.Module into this module.
    def __call__(self, x):
        return self.predict_scores(x)

    # --- No-op nn.Module-style shims used by the evaluator. -----------

    def to(self, device):
        """Move cached popularity scores to ``device`` if already fit."""
        if self.popularity_scores is not None:
            self.popularity_scores = self.popularity_scores.to(device)
        return self

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = bool(mode)
        return self
