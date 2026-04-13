"""Popularity-weighted negative sampler (stub — Wave 5+)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from recsys.data.negatives.base import NegativeSampler


@dataclass
class PopularityNegatives(NegativeSampler):
    """Sample negatives proportional to item popularity.

    Not yet implemented. Wave 5+ will accept a popularity vector
    (aligned with the item vocab) and draw weighted samples.
    """

    item_popularity: np.ndarray | None = field(default=None)
    alpha: float = 1.0

    def sample(
        self,
        *,
        n_negatives: int,
        exclude: set[int],
        vocab_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        raise NotImplementedError(
            "PopularityNegatives is stubbed for Wave 4. Wave 5+ will add "
            "the weighted-sampling implementation."
        )
