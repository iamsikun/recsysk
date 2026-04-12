"""Benchmark protocol and built-in benchmarks.

Importing this package registers the built-in benchmarks
(``movielens_ctr``, ``movielens_seq``) into
:data:`recsys.utils.BENCHMARK_REGISTRY`.
"""

from recsys.benchmarks.base import Benchmark, BenchmarkData
from recsys.benchmarks.movielens_ctr import MovieLensCTRBenchmark
from recsys.benchmarks.movielens_seq import MovieLensSeqBenchmark

__all__ = [
    "Benchmark",
    "BenchmarkData",
    "MovieLensCTRBenchmark",
    "MovieLensSeqBenchmark",
]
