from __future__ import annotations

import random
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from recsys.data.builders.base import DatasetBuilder, DatasetBundle


class BuilderDataModule(LightningDataModule):
    """Lightning DataModule wrapper around a DatasetBuilder."""

    def __init__(
        self,
        builder: DatasetBuilder,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        seed: int | None = None,
    ):
        super().__init__()
        self.builder = builder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

        self._generator: torch.Generator | None = None
        self.feature_map: dict[str, int] = {}
        self.train_dataset = None
        self.val_dataset = None
        self.full_dataset = None

    def _get_generator(self) -> torch.Generator | None:
        if self.seed is None:
            return None
        if self._generator is None:
            self._generator = torch.Generator().manual_seed(self.seed)
        return self._generator

    def _seed_worker(self, worker_id: int) -> None:
        if self.seed is None:
            return
        worker_seed = self.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def setup(self, stage: str | None = None) -> None:
        bundle: DatasetBundle = self.builder.build()
        self.full_dataset = bundle.full
        self.train_dataset = bundle.train
        self.val_dataset = bundle.val
        self.feature_map = bundle.feature_map

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Must call setup() before train_dataloader()")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self._seed_worker if self.seed is not None else None,
            generator=self._get_generator(),
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Must call setup() before val_dataloader()")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self._seed_worker if self.seed is not None else None,
            generator=self._get_generator(),
        )
