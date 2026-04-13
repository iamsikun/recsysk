"""In-batch negative sampler (stub — Wave 5+)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recsys.data.negatives.base import NegativeSampler


@dataclass
class InBatchNegatives(NegativeSampler):
    """Use other positives in the same batch as negatives.

    Not yet implemented. Wave 5+ will plug this into two-tower and
    retrieval-style algorithms that prefer in-batch negatives over
    explicit sampling.
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
            "InBatchNegatives is stubbed for Wave 4. Wave 5+ will add "
            "the in-batch implementation."
        )
