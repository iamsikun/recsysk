"""Benchmark protocol and built-in benchmarks.

Importing this package registers the built-in benchmarks
(``movielens_ctr``, ``movielens_seq``, ``kuairec_ctr``, ``kuairand_ctr``,
``amazon_ctr``, ``amazon_seq``) into :data:`recsys.utils.BENCHMARK_REGISTRY`.
"""

from recsys.benchmarks.amazon_ctr import AmazonCTRBenchmark
from recsys.benchmarks.amazon_seq import AmazonSeqBenchmark
from recsys.benchmarks.base import Benchmark, BenchmarkData
from recsys.benchmarks.frappe_ctr import FrappeCTRBenchmark
from recsys.benchmarks.kuairand_ctr import KuaiRandCTRBenchmark
from recsys.benchmarks.kuairec_ctr import KuaiRecCTRBenchmark
from recsys.benchmarks.microvideo_ctr import MicroVideoCTRBenchmark
from recsys.benchmarks.movielens_ctr import MovieLensCTRBenchmark
from recsys.benchmarks.movielens_seq import MovieLensSeqBenchmark
from recsys.benchmarks.taobao_ad_ctr import TaobaoAdCTRBenchmark

__all__ = [
    "AmazonCTRBenchmark",
    "AmazonSeqBenchmark",
    "Benchmark",
    "BenchmarkData",
    "FrappeCTRBenchmark",
    "KuaiRandCTRBenchmark",
    "KuaiRecCTRBenchmark",
    "MicroVideoCTRBenchmark",
    "MovieLensCTRBenchmark",
    "MovieLensSeqBenchmark",
    "TaobaoAdCTRBenchmark",
]
