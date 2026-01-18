import yaml
import pprint
import lightning as L
from recsys.utils import (
    MODEL_REGISTRY,
    OPTIMIZER_REGISTRY,
    LOSS_REGISTRY,
    DATASET_REGISTRY,
)
from recsys.engine import CTRTask


def get_config() -> dict:
    with open("conf/deepfm.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    # 0. load config
    cfg = get_config()
    L.seed_everything(42)

    # 1. load data
    print(f'Building data: {cfg["data"]["name"]}')
    dm = DATASET_REGISTRY.build(cfg["data"])
    dm.setup(stage="fit")
    print(f"Data loaded successfully. Feature map: {dm.feature_map}")

    # 2. instantiate model and optimizer
    print(f'Building model: {cfg["model"]["name"]}')
    model = MODEL_REGISTRY.build(cfg["model"], feature_map=dm.feature_map)

    # 3. get optimizer and loss classes
    # NOTE: do not build (instantiate) optimizer because we don't know the model parameters yet
    opt_cfg = cfg["optimizer"]
    opt_name = opt_cfg.pop("name")
    opt_cls = OPTIMIZER_REGISTRY.get(opt_name)

    loss_fn = LOSS_REGISTRY.get(cfg["loss"]["name"])

    # 4. instantiate task and trainer
    task = CTRTask(model=model, optimizer_cls=opt_cls, loss_fn=loss_fn)
    print(
        f"Task built successfully. Model: {model}, Optimizer: {opt_cls}, Loss: {loss_fn}"
    )

    # 5. instantiate trainer
    trainer = L.Trainer(**cfg["trainer"])
    print(f"Trainer built successfully. Trainer: {trainer}")

    # 6. Ignite
    print(f"Start training...")
    trainer.fit(task, datamodule=dm)


if __name__ == "__main__":
    main()
