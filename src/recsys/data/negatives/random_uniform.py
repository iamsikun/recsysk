"""Random-uniform negative sampler.

Extracted verbatim from the pre-Wave-4 inline loop in
:meth:`recsys.evaluation.evaluator.CTREvaluator.evaluate_full`. The RNG
call sequence must match the inline loop byte-for-byte so that eval
metrics are unchanged after the refactor.

Inline reference (pre-Wave-4)::

    negatives: list[int] = []
    attempts = 0
    while len(negatives) < n_negatives and attempts < n_negatives * 20:
        cand = int(rng.integers(0, n_items))
        attempts += 1
        if cand in interacted or cand == pos_it or cand in negatives:
            continue
        negatives.append(cand)

``interacted`` and ``{pos_it}`` are unioned into ``exclude`` by the
caller. The loop bails when ``n_negatives * 20`` attempts are spent,
which matches the Wave 3 behaviour exactly (no fallback to a full
complement scan — adding one here would consume extra RNG draws and
break byte-identity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recsys.data.negatives.base import NegativeSampler


@dataclass
class RandomUniform(NegativeSampler):
    """Uniform rejection sampler over the item vocabulary."""

    #: Multiplier on ``n_negatives`` that bounds the rejection budget.
    #: Default matches the pre-Wave-4 inline loop (``n_negatives * 20``).
    max_attempts_multiplier: int = 20

    def sample(
        self,
        *,
        n_negatives: int,
        exclude: set[int],
        vocab_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        negatives: list[int] = []
        attempts = 0
        max_attempts = n_negatives * self.max_attempts_multiplier
        while len(negatives) < n_negatives and attempts < max_attempts:
            cand = int(rng.integers(0, vocab_size))
            attempts += 1
            if cand in exclude or cand in negatives:
                continue
            negatives.append(cand)
        return np.array(negatives, dtype=np.int64)
