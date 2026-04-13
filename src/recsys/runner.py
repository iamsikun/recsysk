from __future__ import annotations

import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightning as L
import torch
import yaml

from recsys.algorithms.base import Algorithm
from recsys.engine import CTRTask as LightningCTRTask
from recsys.evaluation import CTREvaluator
from recsys.evaluation.store import ResultStore, RunResult
from recsys.utils import (
    ALGO_REGISTRY,
    BENCHMARK_REGISTRY,
    DATASET_REGISTRY,
    LOSS_REGISTRY,
    OPTIMIZER_REGISTRY,
    config_hash,
)
import recsys.data  # Needed to register datasets
import recsys.algorithms  # Needed to register algorithms
import recsys.tasks  # Needed to register tasks
import recsys.benchmarks  # Needed to register benchmarks


LOGGER = logging.getLogger(__name__)


def _resolve_log_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    level_value = logging.getLevelName(str(level).upper())
    if isinstance(level_value, int):
        return level_value
    return logging.INFO


def configure_logging(level: str | int = "INFO") -> None:
    level_value = _resolve_log_level(level)
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level_value)
        return
    logging.basicConfig(
        level=level_value,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _resolve_num_workers(num_workers: int | str | None) -> int | None:
    if num_workers is None:
        return None
    if isinstance(num_workers, str):
        if num_workers.lower() != "auto":
            return int(num_workers)
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count - 1)
    return num_workers


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r") as handle:
        return yaml.safe_load(handle)


def _normalize_data_cfg(data_cfg: dict[str, Any], seed: int | None) -> dict[str, Any]:
    data_cfg = dict(data_cfg)
    if "seed" not in data_cfg and seed is not None:
        data_cfg["seed"] = seed
    if "num_workers" in data_cfg:
        resolved_workers = _resolve_num_workers(data_cfg["num_workers"])
        if resolved_workers is not None:
            data_cfg["num_workers"] = resolved_workers
    return data_cfg


def build_datamodule(cfg: dict[str, Any]):
    data_cfg = _normalize_data_cfg(cfg["data"], cfg.get("seed"))
    LOGGER.info("Building data module: %s", data_cfg.get("name"))
    dm = DATASET_REGISTRY.build(data_cfg)
    dm.setup(stage="fit")
    feature_map = getattr(dm, "feature_map", {})
    LOGGER.info("Data loaded. Feature map size: %s", len(feature_map))
    return dm


def build_model(cfg: dict[str, Any], feature_map: dict[str, int]):
    model_cfg = cfg["model"]
    LOGGER.info("Building model: %s", model_cfg.get("name"))
    return ALGO_REGISTRY.build(model_cfg, feature_map=feature_map)


def build_optimizer(cfg: dict[str, Any]):
    opt_cfg = dict(cfg["optimizer"])
    opt_name = opt_cfg.pop("name")
    opt_cls = OPTIMIZER_REGISTRY.get(opt_name)
    return opt_cls, opt_cfg


def build_loss(cfg: dict[str, Any]):
    return LOSS_REGISTRY.get(cfg["loss"]["name"])


def build_task(
    model,
    optimizer_cls,
    optimizer_params: dict[str, Any],
    loss_fn,
):
    task = LightningCTRTask(
        model=model,
        optimizer_cls=optimizer_cls,
        optimizer_params=optimizer_params,
        loss_fn=loss_fn,
    )
    LOGGER.info(
        "Task built. Model=%s Optimizer=%s Loss=%s",
        model.__class__.__name__,
        optimizer_cls.__name__,
        getattr(loss_fn, "__name__", repr(loss_fn)),
    )
    return task


def build_trainer(cfg: dict[str, Any]):
    trainer = L.Trainer(**cfg["trainer"])
    LOGGER.info("Trainer built: %s", trainer.__class__.__name__)
    return trainer


# Defaults used when a torch algorithm config doesn't carry its own
# optimizer/loss block. Classical algos (anything subclassing
# :class:`Algorithm` directly) never touch these — they go through the
# framework-agnostic bypass branch in :func:`run_experiment` and call
# ``algo.fit`` without any Lightning machinery.
_DEFAULT_OPTIMIZER = {"name": "adamw", "lr": 0.001}
_DEFAULT_LOSS = {"name": "binary_cross_entropy_with_logits"}


def _git_sha() -> str | None:
    """Best-effort ``git rev-parse HEAD``; return ``None`` on failure."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=3,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def run_experiment(
    algo_cfg: dict[str, Any],
    benchmark_cfg: dict[str, Any],
    seed: int = 42,
    trainer_overrides: dict[str, Any] | None = None,
    results_dir: Path | str | None = None,
    store: ResultStore | None = None,
) -> dict[str, float]:
    """Run an algorithm against a benchmark and return the metric dict.

    This is the single entrypoint the CLI and higher-level tooling hits.
    ``algo_cfg`` is what used to live under ``cfg["model"]`` plus optional
    ``optimizer``/``loss`` overrides. ``benchmark_cfg`` is what
    :func:`load_config` returns for a ``conf/benchmarks/*.yaml`` file.

    If ``store`` is provided (or ``results_dir`` is set, in which case a
    fresh :class:`ResultStore` is built), a :class:`RunResult` row is
    appended after evaluation completes.
    """
    L.seed_everything(seed, workers=True)
    LOGGER.info("run_experiment: seed=%s", seed)
    start_time = time.perf_counter()

    # Normalise data cfg (num_workers=auto, inject seed, ...).
    data_cfg = _normalize_data_cfg(benchmark_cfg.get("data", {}), seed)
    eval_cfg = dict(benchmark_cfg.get("eval", {}))

    benchmark_name = benchmark_cfg["name"]
    LOGGER.info("run_experiment: building benchmark=%s", benchmark_name)
    benchmark = BENCHMARK_REGISTRY.build(
        {
            "name": benchmark_name,
            "data_cfg": data_cfg,
            "eval_cfg": eval_cfg,
        }
    )
    data = benchmark.build()
    LOGGER.info(
        "run_experiment: benchmark ready. feature_map_size=%s",
        len(data.feature_map),
    )

    algo_build_cfg = dict(algo_cfg)
    # optimizer/loss live under the algo cfg as a convenience for the
    # torch path; they're not algorithm constructor kwargs.
    opt_override = algo_build_cfg.pop("optimizer", None)
    loss_override = algo_build_cfg.pop("loss", None)
    algo = ALGO_REGISTRY.build(algo_build_cfg, feature_map=data.feature_map)

    # In smoke mode (any limit_val_batches set) cap the ranking-metric
    # user loop so the gate stays cheap.
    max_users_override = None
    if trainer_overrides and trainer_overrides.get("limit_val_batches") is not None:
        max_users_override = 200

    # Classical bypass: algorithms that subclass the framework-agnostic
    # :class:`Algorithm` protocol directly (i.e. not an ``nn.Module``)
    # are fit via ``algo.fit(train, val)`` and handed straight to the
    # task evaluator — no Lightning wrapper, no Trainer.fit, no shim
    # optimizer. The torch path below is untouched for DeepFM/DIN.
    if isinstance(algo, Algorithm):
        LOGGER.info(
            "run_experiment: classical bypass — calling %s.fit directly",
            algo.__class__.__name__,
        )
        algo.fit(data.train, data.val)
        eval_target: Any = algo
    else:
        optimizer_cfg = {**_DEFAULT_OPTIMIZER, **(opt_override or {})}
        optimizer_cls = OPTIMIZER_REGISTRY.get(optimizer_cfg.pop("name"))
        loss_cfg = {**_DEFAULT_LOSS, **(loss_override or {})}
        loss_fn = LOSS_REGISTRY.get(loss_cfg["name"])
        lightning_task = LightningCTRTask(
            model=algo,
            optimizer_cls=optimizer_cls,
            optimizer_params=optimizer_cfg,
            loss_fn=loss_fn,
        )

        trainer_cfg: dict[str, Any] = {"max_epochs": 1}
        if trainer_overrides:
            trainer_cfg.update(trainer_overrides)
        trainer = L.Trainer(**trainer_cfg)

        LOGGER.info("run_experiment: starting trainer.fit")
        trainer.fit(lightning_task, datamodule=data.datamodule)

        device = getattr(trainer.strategy, "root_device", None)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lightning_task.eval()
        lightning_task.model.to(device)
        eval_target = lightning_task

    metrics = benchmark.task.evaluate(
        algo=eval_target,
        benchmark_data=data,
        metric_names=benchmark.metric_names,
        max_users_override=max_users_override,
    )

    runtime_s = time.perf_counter() - start_time
    LOGGER.info(
        "run_experiment: metrics: "
        + " ".join(f"{k}={v:.6f}" for k, v in metrics.items())
    )

    # Persist the run if a store (or a results_dir) was provided.
    effective_store = store
    if effective_store is None and results_dir is not None:
        effective_store = ResultStore(results_dir)
    if effective_store is not None:
        algo_name = str(algo_cfg.get("name", "unknown"))
        result = RunResult(
            benchmark=benchmark.name,
            benchmark_version=benchmark.version(),
            algo=algo_name,
            algo_config_hash=config_hash(algo_cfg),
            seed=int(seed),
            metrics={k: float(v) for k, v in metrics.items()},
            runtime_s=float(runtime_s),
            timestamp=datetime.now(timezone.utc).isoformat(),
            code_sha=_git_sha(),
            env_fingerprint=None,
        )
        effective_store.write(result)
        LOGGER.info(
            "run_experiment: wrote result to %s",
            effective_store._path(benchmark.name),
        )

    return metrics
