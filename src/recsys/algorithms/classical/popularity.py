"""Popularity baseline for CTR prediction.

This is an item-popularity baseline: the score for an item is a monotonic
transform of how many times it appeared in the training set. It is fit
once (no SGD) via :meth:`Popularity.fit_on_train_counts`, then serves as a
fixed-weight scorer during Lightning's (no-op) training and validation
passes.

Wave 2 caveat: the baseline lives inside an ``nn.Module`` subclass so it
can flow through the existing ``CTRTask``/Lightning/``CTREvaluator``
machinery unchanged. A "real" classical algorithm interface
(``Algorithm.fit`` + ``Algorithm.predict_scores`` with no torch) is
deferred to Wave 3 once the runner's Lightning assumptions are split out.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from recsys.utils import ALGO_REGISTRY

LOGGER = logging.getLogger(__name__)


@ALGO_REGISTRY.register("popularity")
class Popularity(nn.Module):
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

    def __init__(
        self,
        feature_map: dict[str, int],
        item_feature: str = "item_id",
    ) -> None:
        super().__init__()
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

        # Frozen "weights" — one log-popularity score per item id. We
        # register this as an nn.Parameter (requires_grad=False) so it
        # travels with the model state dict and is moved across devices.
        scores = torch.zeros(self.vocab_size, dtype=torch.float32)
        self.popularity_scores = nn.Parameter(scores, requires_grad=False)

        # Tiny trainable placeholder so Lightning-provided optimizers
        # don't blow up on an empty parameter list. It is never used in
        # forward and gets zero gradient because training is capped at
        # max_epochs=0 for this baseline.
        self._unused = nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def fit_on_train_counts(
        self,
        train_dataset: Dataset,
        batch_size: int = 8192,
    ) -> None:
        """Compute log(count+1) popularity scores from the training set.

        Iterates the dataset once, accumulates item-id occurrence counts,
        applies ``log(count + 1)``, normalizes to ``[0, 1]``, and centers
        the final logits on zero (mean-subtracted) so that ``sigmoid``
        produces values around 0.5 rather than saturating.
        """
        LOGGER.info(
            "Popularity.fit_on_train_counts: vocab_size=%d item_column=%d",
            self.vocab_size,
            self.item_column,
        )
        counts = torch.zeros(self.vocab_size, dtype=torch.float64)

        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        n_seen = 0
        for batch in loader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError(
                    "Popularity.fit_on_train_counts expects (x, y) batches; "
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
            # Guard against any out-of-range ids.
            item_ids = torch.clamp(item_ids, min=0, max=self.vocab_size - 1)
            counts.scatter_add_(
                0, item_ids, torch.ones_like(item_ids, dtype=torch.float64)
            )
            n_seen += item_ids.numel()

        LOGGER.info(
            "Popularity.fit_on_train_counts: saw %d interactions across %d items",
            n_seen,
            int((counts > 0).sum().item()),
        )

        log_counts = torch.log1p(counts).to(torch.float32)
        max_val = float(log_counts.max().item())
        if max_val > 0.0:
            log_counts = log_counts / max_val
        # Center so sigmoid isn't saturated at the high end.
        log_counts = log_counts - log_counts.mean()

        self.popularity_scores.data.copy_(log_counts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample logits based on item popularity."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "Popularity baseline only supports tabular tensor inputs; "
                f"got {type(x).__name__}"
            )
        if x.dim() != 2:
            raise ValueError(
                f"Expected 2-D tabular batch, got shape {tuple(x.shape)}"
            )
        item_ids = x[:, self.item_column].to(torch.long)
        item_ids = torch.clamp(item_ids, min=0, max=self.vocab_size - 1)
        logits = self.popularity_scores[item_ids].unsqueeze(-1)
        return logits
