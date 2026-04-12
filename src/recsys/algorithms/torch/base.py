"""TorchAlgorithm skeleton — thin base for Lightning-backed algorithms.

In Wave 1 this file exists only so that Wave 2 code can subclass it without
having to introduce the file itself. The full ``fit`` implementation that
owns a :class:`lightning.Trainer` and delegates to ``trainer.fit`` is wired
in Wave 3 (P5), together with the Task/Benchmark abstractions that define
what ``train`` / ``val`` look like at the algorithm boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from recsys.algorithms.base import Algorithm

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    import lightning as L
    from torch import nn


class TorchAlgorithm(Algorithm):
    """Base class for algorithms backed by a ``LightningModule``.

    Wraps a ``LightningModule`` plus a serializable Trainer config dict.
    Subclasses should set :attr:`module` in their ``__init__`` and then let
    :meth:`fit` (to be wired in P5) build a fresh trainer from
    :attr:`trainer_config` and call ``trainer.fit`` against the Lightning
    DataModule exposed by the benchmark's ``BenchmarkData``.

    Attributes:
        module: The wrapped ``LightningModule``. May be ``None`` until the
            algorithm has been built against a concrete feature map.
        trainer_config: Dict of keyword arguments forwarded to
            ``lightning.Trainer`` at fit time.
    """

    def __init__(
        self,
        module: "L.LightningModule | nn.Module | None" = None,
        trainer_config: dict[str, Any] | None = None,
    ) -> None:
        self.module = module
        self.trainer_config: dict[str, Any] = dict(trainer_config or {})

    def fit(self, train: Any, val: Any | None = None) -> None:
        # wired in Wave 3 (P5): build a lightning.Trainer from
        # self.trainer_config, adapt train/val to a DataModule, and call
        # trainer.fit(self.module, datamodule=...).
        raise NotImplementedError(
            "TorchAlgorithm.fit is a Wave 1 skeleton; the Lightning fit path "
            "is wired in Wave 3 (P5)."
        )
