"""Negative sampler protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class NegativeSampler(Protocol):
    """Protocol for eval-time / train-time negative samplers.

    Implementations return up to ``n_negatives`` item ids not in
    ``exclude`` (the user's already-interacted set, plus any held-out
    positive that must not collide). The returned array may be shorter
    than ``n_negatives`` if the sampler's internal budget is exhausted
    — callers are responsible for handling an empty result.
    """

    def sample(
        self,
        *,
        n_negatives: int,
        exclude: set[int],
        vocab_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return sampled negative item ids as an ``int64`` array."""
        ...
