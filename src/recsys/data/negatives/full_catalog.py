"""Full-catalog negative enumerator (stub — Wave 5+)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recsys.data.negatives.base import NegativeSampler


@dataclass
class FullCatalogNegatives(NegativeSampler):
    """Return the full complement of ``exclude`` (no sampling).

    Not yet implemented. Wave 5+ will provide full-catalog ranking for
    benchmarks that can afford it (e.g. MovieLens 20M at eval time).
    """

    def sample(
        self,
        *,
        n_negatives: int,
        exclude: set[int],
        vocab_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        raise NotImplementedError(
            "FullCatalogNegatives is stubbed for Wave 4. Wave 5+ will "
            "add full-complement enumeration for ranking metrics."
        )
