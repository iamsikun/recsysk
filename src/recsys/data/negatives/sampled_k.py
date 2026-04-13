"""Train-time sampled-K negative sampler (stub — Wave 5+)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recsys.data.negatives.base import NegativeSampler


@dataclass
class SampledKNegatives(NegativeSampler):
    """Per-positive sampled-K negatives (train-time).

    Not yet implemented. Wave 5+ will plug this into the sequential
    builder to replace the inline negative-sampling currently baked
    into :class:`recsys.data.builders.movielens.MovieLensSeqBuilder`.
    """

    k: int = 1

    def sample(
        self,
        *,
        n_negatives: int,
        exclude: set[int],
        vocab_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        raise NotImplementedError(
            "SampledKNegatives is stubbed for Wave 4. Wave 5+ will add "
            "the train-time implementation."
        )
