"""Negative samplers.

Wave 4 (P6) factors the eval-time sampled-K negative sampler out of
:class:`recsys.evaluation.evaluator.CTREvaluator.evaluate_full` into a
reusable :class:`RandomUniform` module. Popularity, full-catalog, sampled-K
(train-time), and in-batch samplers are stubbed for Wave 5+.
"""

from recsys.data.negatives.base import NegativeSampler
from recsys.data.negatives.random_uniform import RandomUniform

__all__ = ["NegativeSampler", "RandomUniform"]
