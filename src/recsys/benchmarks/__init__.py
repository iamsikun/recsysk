"""Benchmark protocol and built-in benchmarks.

Importing this package registers the built-in benchmarks
(``movielens_ctr``, ``movielens_seq``, ``kuairec_ctr``, ``kuairand_ctr``)
into :data:`recsys.utils.BENCHMARK_REGISTRY`.
"""

from recsys.benchmarks.base import Benchmark, BenchmarkData
from recsys.benchmarks.kuairand_ctr import KuaiRandCTRBenchmark
from recsys.benchmarks.kuairec_ctr import KuaiRecCTRBenchmark
from recsys.benchmarks.movielens_ctr import MovieLensCTRBenchmark
from recsys.benchmarks.movielens_seq import MovieLensSeqBenchmark

__all__ = [
    "Benchmark",
    "BenchmarkData",
    "KuaiRandCTRBenchmark",
    "KuaiRecCTRBenchmark",
    "MovieLensCTRBenchmark",
    "MovieLensSeqBenchmark",
]
