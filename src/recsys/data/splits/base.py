"""Splitter protocol.

A :class:`Splitter` partitions a dataset into ``(train, val, test)``.
Implementations may return the same object for ``val`` and ``test`` when
they don't support a real three-way split yet (Wave 4 aliases test=val
for MovieLens; Wave 5+ adds temporal/leave-last-out splits with a real
held-out test slice).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch.utils.data import Dataset


@runtime_checkable
class Splitter(Protocol):
    """Protocol all split strategies must implement."""

    def split(
        self,
        dataset: Dataset,
        *,
        seed: int | None = None,
    ) -> tuple[Dataset, Dataset, Dataset]:
        """Return ``(train, val, test)`` subsets of ``dataset``.

        Implementations may return the same object for ``val`` and
        ``test`` when a three-way split is not yet available.
        """
        ...
