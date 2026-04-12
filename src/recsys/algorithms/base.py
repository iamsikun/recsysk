"""Framework-agnostic Algorithm protocol and TaskType enum.

This module deliberately does not import :mod:`torch`. Classical algorithms
(popularity, item-KNN, BPR-MF, ...) must be importable and runnable without
paying the torch import cost, and keeping this base module torch-free is the
way we enforce that. Torch-only helpers live under :mod:`recsys.algorithms.torch`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    import torch


class TaskType(Enum):
    """Enumeration of supported task types.

    A task type defines the I/O contract between an :class:`Algorithm` and an
    evaluator. An algorithm declares which task types it supports via
    :attr:`Algorithm.supported_tasks`; benchmarks declare their task type and
    the runner checks compatibility before calling :meth:`Algorithm.fit`.
    """

    CTR = "ctr"
    RETRIEVAL = "retrieval"
    SEQUENTIAL = "sequential"


class Algorithm(ABC):
    """Abstract base class for all recommendation algorithms.

    Subclasses must declare :attr:`supported_tasks` and :attr:`required_roles`
    as class attributes. :meth:`fit` is the only method every algorithm must
    implement; the prediction / persistence methods default to
    :class:`NotImplementedError` so that algorithms only need to implement the
    surface relevant to the tasks they support.

    Note:
        ``required_roles`` is typed as ``set[str]`` rather than a
        ``FeatureRole`` enum to keep this module framework- and schema-free.
        Subclasses list role names (e.g. ``{"user", "item", "label"}``).
    """

    #: Task types this algorithm can be evaluated on.
    supported_tasks: set[TaskType] = set()

    #: Names of feature roles the algorithm requires its dataset to provide.
    required_roles: set[str] = set()

    @abstractmethod
    def fit(self, train: Any, val: Any | None = None) -> None:
        """Fit the algorithm on the training data.

        Args:
            train: Training data bundle (framework-specific; typically a
                ``BenchmarkData`` in later phases).
            val: Optional validation data bundle used for early stopping /
                hyperparameter monitoring.
        """
        raise NotImplementedError

    def predict_scores(self, batch: Any) -> "torch.Tensor":
        """Return per-row scores for a batch of (user, item) pairs.

        Used by CTR-style tasks. Classical algorithms may implement this over
        numpy arrays and wrap the result in a ``torch.Tensor`` at the boundary.
        """
        raise NotImplementedError

    def predict_topk(
        self,
        users: Any,
        k: int,
        candidates: Sequence[int] | None = None,
    ) -> Any:
        """Return the top-``k`` items for each user.

        Used by retrieval / sequential tasks.

        Args:
            users: Iterable of user ids.
            k: Number of items to return per user.
            candidates: Optional restricted candidate set. If ``None`` the
                algorithm is expected to score against the full catalog.
        """
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Persist algorithm state (weights, indices, ...) to ``path``."""
        raise NotImplementedError

    def load(self, path: Path) -> None:
        """Load algorithm state previously written by :meth:`save`."""
        raise NotImplementedError
