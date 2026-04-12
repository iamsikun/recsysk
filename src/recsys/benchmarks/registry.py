"""Benchmark registry re-export.

The underlying ``Registry`` instance lives in :mod:`recsys.utils` alongside
the others so the whole project shares a single import path. This module
simply re-exports it for benchmark-module consumers and ensures all
built-in benchmarks are imported for their decorator side effects.
"""

from __future__ import annotations

from recsys.utils import BENCHMARK_REGISTRY

# Side-effect imports register built-in benchmarks.
from recsys.benchmarks import movielens_ctr  # noqa: F401
from recsys.benchmarks import movielens_seq  # noqa: F401

__all__ = ["BENCHMARK_REGISTRY"]
