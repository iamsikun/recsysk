"""Thin argparse CLI for Phase 7: bench, report, list.

Usage examples::

    recsys list benchmarks
    recsys list algorithms
    recsys bench --experiment conf/experiments/deepfm_on_movielens_ctr.yaml
    recsys bench --experiment conf/experiments/deepfm_on_movielens_ctr.yaml \\
        --seeds 1,2,3 --results-dir results
    recsys report --benchmark movielens_ctr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Importing these packages triggers registry population.
import recsys.algorithms  # noqa: F401
import recsys.benchmarks  # noqa: F401
import recsys.data  # noqa: F401
import recsys.tasks  # noqa: F401
from recsys.evaluation.reporting import format_table, summary_table
from recsys.evaluation.store import ResultStore
from recsys.runner import configure_logging, load_config, run_experiment
from recsys.utils import ALGO_REGISTRY, BENCHMARK_REGISTRY


def _parse_seed_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _resolve_path(path: str, base: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute() or base is None:
        return p
    candidate = (base / p).resolve()
    if candidate.exists():
        return candidate
    return p


def cmd_bench(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)

    experiment_path = Path(args.experiment).resolve()
    experiment: dict[str, Any] = load_config(experiment_path)

    exp_dir = experiment_path.parent
    bench_path = _resolve_path(experiment["benchmark"], exp_dir)
    algo_path = _resolve_path(experiment["algo"], exp_dir)

    benchmark_cfg = load_config(bench_path)
    algo_cfg = load_config(algo_path)

    seeds: list[int] = args.seeds or experiment.get("seeds", [42])
    trainer_overrides: dict[str, Any] = dict(experiment.get("trainer", {}))

    results_dir = Path(args.results_dir)
    store = ResultStore(results_dir)

    exp_name = experiment.get("name", experiment_path.stem)
    for seed in seeds:
        print(f"[recsys bench] {exp_name} seed={seed}")
        metrics = run_experiment(
            algo_cfg=algo_cfg,
            benchmark_cfg=benchmark_cfg,
            seed=seed,
            trainer_overrides=trainer_overrides,
            results_dir=results_dir,
            store=store,
        )
        metric_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"  metrics: {metric_str}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    store = ResultStore(Path(args.results_dir))
    df = summary_table(store, args.benchmark)
    print(format_table(df))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    if args.what == "benchmarks":
        registry = BENCHMARK_REGISTRY
    else:
        registry = ALGO_REGISTRY
    for name in sorted(registry._registry):
        print(name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recsys",
        description="recsys benchmarking CLI",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    bench = sub.add_parser("bench", help="Run an experiment YAML")
    bench.add_argument("--experiment", required=True, help="Path to experiment YAML")
    bench.add_argument(
        "--seeds",
        type=_parse_seed_list,
        default=None,
        help="Comma-separated seed list; overrides experiment.seeds",
    )
    bench.add_argument(
        "--results-dir",
        default="results",
        help="Where parquet result files live (default: results/)",
    )
    bench.add_argument(
        "--log-level",
        default="INFO",
        help="Python log level (default: INFO)",
    )
    bench.set_defaults(func=cmd_bench)

    report = sub.add_parser("report", help="Print aggregated summary for a benchmark")
    report.add_argument("--benchmark", required=True, help="Benchmark name, e.g. movielens_ctr")
    report.add_argument(
        "--results-dir",
        default="results",
        help="Where parquet result files live (default: results/)",
    )
    report.set_defaults(func=cmd_report)

    lst = sub.add_parser("list", help="List registered benchmarks or algorithms")
    lst.add_argument("what", choices=["benchmarks", "algorithms"])
    lst.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
