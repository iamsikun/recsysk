from __future__ import annotations

from typing import Callable, Any

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim
from recsys.utils import OPTIMIZER_REGISTRY, LOSS_REGISTRY


# Register standard torch optimizers
OPTIMIZER_REGISTRY.register("adamw")(torch.optim.AdamW)
OPTIMIZER_REGISTRY.register("adam")(torch.optim.Adam)
OPTIMIZER_REGISTRY.register("sgd")(torch.optim.SGD)

# Register standard torch losses
LOSS_REGISTRY.register("binary_cross_entropy_with_logits")(
    F.binary_cross_entropy_with_logits
)


# CTR Trainer
class CTRTask(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer_cls: type[torch.optim.Optimizer],
        optimizer_params: dict[str, Any],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        metrics: list[Callable[[torch.Tensor, torch.Tensor], float]] = [],
    ):
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_params
        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch

        logits = self.model(x)  # (B, 1)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        logits = self.model(x)  # (B, 1)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer_cls(self.model.parameters(), **self.optimizer_params)
