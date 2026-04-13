"""Temporal global splitter (stub — Wave 5+)."""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import Dataset

from recsys.data.splits.base import Splitter


@dataclass
class TemporalGlobalSplit(Splitter):
    """Split at a global timestamp cutoff.

    Not yet implemented. Wave 5+ fills this in once the
    benchmarks expose timestamp columns through the builder.
    """

    cutoff_timestamp: int | float

    def split(
        self,
        dataset: Dataset,
        *,
        seed: int | None = None,
    ) -> tuple[Dataset, Dataset, Dataset]:
        raise NotImplementedError(
            "TemporalGlobalSplit is stubbed for Wave 4. Wave 5+ will "
            "implement global-cutoff temporal splits."
        )
