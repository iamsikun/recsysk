from __future__ import annotations

from pathlib import Path
import logging
import os
from typing import Any

import lightning as L
import torch
import yaml

from recsys.engine import CTRTask
from recsys.evaluation import CTREvaluator
from recsys.utils import (
    ALGO_REGISTRY,
    OPTIMIZER_REGISTRY,
    LOSS_REGISTRY,
    DATASET_REGISTRY,
)
import recsys.data  # Needed to register datasets
import recsys.algorithms  # Needed to register algorithms


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


def build_datamodule(cfg: dict[str, Any]):
    data_cfg = dict(cfg["data"])
    if "seed" not in data_cfg and "seed" in cfg:
        data_cfg["seed"] = cfg["seed"]
    if "num_workers" in data_cfg:
        resolved_workers = _resolve_num_workers(data_cfg["num_workers"])
        if resolved_workers is not None:
            data_cfg["num_workers"] = resolved_workers
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
    task = CTRTask(
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


def train(cfg: dict[str, Any]) -> None:
    seed = cfg.get("seed", 42)
    L.seed_everything(seed, workers=True)
    LOGGER.info("Seed set to %s", seed)

    dm = build_datamodule(cfg)
    model = build_model(cfg, dm.feature_map)
    # Classical-algorithm hook: one-shot fit on the train dataset before
    # Lightning runs. Used by baselines like popularity that don't need SGD.
    fit_hook = getattr(model, "fit_on_train_counts", None)
    if callable(fit_hook):
        LOGGER.info("Running classical one-shot fit_on_train_counts")
        fit_hook(dm.train_dataset)
    optimizer_cls, optimizer_params = build_optimizer(cfg)
    loss_fn = build_loss(cfg)
    task = build_task(model, optimizer_cls, optimizer_params, loss_fn)
    trainer = build_trainer(cfg)

    LOGGER.info("Starting training")
    trainer.fit(task, datamodule=dm)

    # Phase 2: post-fit CTR evaluation on the val dataloader.
    val_loader = dm.val_dataloader()
    if val_loader is None:
        LOGGER.warning("No val dataloader available; skipping post-fit evaluation")
        return
    device = getattr(trainer.strategy, "root_device", None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task.eval()
    task.model.to(device)
    try:
        metrics = CTREvaluator().evaluate(task.model, val_loader, device)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Post-fit evaluation failed: %s", exc)
        return
    if not metrics:
        LOGGER.warning("Post-fit val metrics: empty (no batches evaluated)")
        return
    LOGGER.info(
        "Post-fit val metrics: auc=%.6f logloss=%.6f",
        metrics.get("auc", float("nan")),
        metrics.get("logloss", float("nan")),
    )


def train_from_config(
    config_path: str | Path,
    log_level: str | int = "INFO",
    seed_override: int | None = None,
) -> None:
    configure_logging(log_level)

    LOGGER.info(f"Starting training from config: {config_path}")
    cfg = load_config(config_path)
    LOGGER.info(f"Using device: {cfg['trainer']['accelerator']}")

    if seed_override is not None:
        LOGGER.info(f"Overriding seed with: {seed_override}")
        cfg["seed"] = seed_override
    train(cfg)
