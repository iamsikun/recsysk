"""Thin argparse CLI: bench, report, list, submit.

Usage examples::

    recsys list benchmarks
    recsys list algorithms
    recsys bench --experiment conf/experiments/deepfm_on_movielens_ctr.yaml
    recsys bench --experiment conf/experiments/deepfm_on_movielens_ctr.yaml \\
        --seeds 1,2,3 --results-dir results
    recsys report --benchmark movielens_ctr
    recsys submit --benchmark movielens_ctr --algo deepfm --seed 1 \\
        --out /tmp/submission.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Importing these packages triggers registry population.
import recsys.algorithms  # noqa: F401
import recsys.benchmarks  # noqa: F401
import recsys.data  # noqa: F401
import recsys.tasks  # noqa: F401
from recsys.algorithms.base import Algorithm
from recsys.engine import CTRTask as LightningCTRTask
from recsys.evaluation.reporting import format_table, summary_table
from recsys.evaluation.store import ResultStore
from recsys.runner import configure_logging, load_config, run_experiment
from recsys.utils import ALGO_REGISTRY, BENCHMARK_REGISTRY

LOGGER = logging.getLogger(__name__)


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


def cmd_submit(args: argparse.Namespace) -> int:
    """Export predictions for a previously-trained run to ``--out``.

    Flow:

    1. Query the result store for a run matching
       ``(benchmark, algo, [config_hash], [seed])`` — picks latest on tie.
    2. Read the sidecar JSON next to the stored checkpoint for the
       exact ``algo_cfg`` + ``benchmark_cfg`` used at fit time.
    3. Rebuild the benchmark via :func:`BENCHMARK_REGISTRY.build` and
       call ``benchmark.build()`` to materialise the test dataset.
    4. Load the checkpoint (Lightning ``.ckpt`` for torch algos, pickle
       ``.pkl`` for classical) and hand it to
       ``benchmark.task.export_predictions``.
    """
    configure_logging(args.log_level)

    store = ResultStore(Path(args.results_dir))
    run = store.get_run(
        benchmark=args.benchmark,
        algo=args.algo,
        algo_config_hash=args.config_hash,
        seed=args.seed,
    )
    if run is None:
        print(
            f"[recsys submit] no run found in {args.results_dir} matching "
            f"benchmark={args.benchmark} algo={args.algo} "
            f"config_hash={args.config_hash} seed={args.seed}",
            file=sys.stderr,
        )
        return 2
    if not run.model_checkpoint_path:
        print(
            f"[recsys submit] matched run has no model_checkpoint_path — "
            f"was it persisted? benchmark={run.benchmark} algo={run.algo} "
            f"hash={run.algo_config_hash} seed={run.seed}",
            file=sys.stderr,
        )
        return 3

    ckpt_path = Path(run.model_checkpoint_path)
    sidecar_path = ckpt_path.with_suffix(".json")
    if not sidecar_path.exists():
        print(
            f"[recsys submit] missing sidecar JSON at {sidecar_path}",
            file=sys.stderr,
        )
        return 4
    with sidecar_path.open() as fh:
        sidecar = json.load(fh)
    algo_cfg = sidecar["algo_cfg"]
    benchmark_cfg = sidecar["benchmark_cfg"]

    benchmark = BENCHMARK_REGISTRY.build(
        {
            "name": benchmark_cfg["name"],
            "data_cfg": benchmark_cfg.get("data", {}),
            "eval_cfg": benchmark_cfg.get("eval", {}),
        }
    )
    data = benchmark.build()

    algo_build_cfg = dict(algo_cfg)
    algo_build_cfg.pop("optimizer", None)
    algo_build_cfg.pop("loss", None)
    algo = ALGO_REGISTRY.build(
        algo_build_cfg,
        feature_map=data.feature_map,
        feature_specs=data.feature_specs,
    )

    # Load the checkpoint back into the constructed algo.
    if isinstance(algo, Algorithm):
        algo.load(ckpt_path)
        eval_target: Any = algo
    else:
        # LightningCTRTask does not use ``save_hyperparameters()`` — we
        # can't use ``load_from_checkpoint`` directly. Instead build an
        # empty task shell around the freshly-constructed algo and
        # copy the state dict from the ``.ckpt`` payload.
        import torch as _torch

        # Optimizer/loss are irrelevant for inference; pass stubs so
        # LightningCTRTask's __init__ is satisfied.
        lightning_task = LightningCTRTask(
            model=algo,
            optimizer_cls=_torch.optim.AdamW,
            optimizer_params={"lr": 1e-3},
            loss_fn=_torch.nn.functional.binary_cross_entropy_with_logits,
        )
        ckpt = _torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        lightning_task.load_state_dict(ckpt["state_dict"], strict=True)
        lightning_task.eval()
        eval_target = lightning_task

    out_path = Path(args.out)
    LOGGER.info(
        "cmd_submit: writing predictions for %s/%s -> %s",
        run.benchmark,
        run.algo,
        out_path,
    )
    benchmark.task.export_predictions(
        algo=eval_target,
        benchmark=benchmark,
        benchmark_data=data,
        out_path=out_path,
    )
    print(f"[recsys submit] wrote {out_path}")
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

    submit = sub.add_parser(
        "submit", help="Export predictions for a trained run"
    )
    submit.add_argument("--benchmark", required=True)
    submit.add_argument("--algo", required=True)
    submit.add_argument(
        "--config-hash",
        default=None,
        help="Optional algo_config_hash filter; picks latest matching row.",
    )
    submit.add_argument("--seed", type=int, default=None)
    submit.add_argument("--out", required=True, help="Output file path")
    submit.add_argument(
        "--results-dir",
        default="results",
        help="Where parquet result files + checkpoints live (default: results/)",
    )
    submit.add_argument(
        "--log-level",
        default="INFO",
        help="Python log level (default: INFO)",
    )
    submit.set_defaults(func=cmd_submit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
