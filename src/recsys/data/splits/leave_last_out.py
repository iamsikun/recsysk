"""Leave-last-out splitter (stub — Wave 5+)."""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import Dataset

from recsys.data.splits.base import Splitter


@dataclass
class LeaveLastOutSplit(Splitter):
    """Per-user leave-last-out split.

    Not yet implemented. Wave 5+ will carve each user's final positive
    interaction as the test row and the one before that as val.
    """

    min_user_interactions: int = 2

    def split(
        self,
        dataset: Dataset,
        *,
        seed: int | None = None,
    ) -> tuple[Dataset, Dataset, Dataset]:
        raise NotImplementedError(
            "LeaveLastOutSplit is stubbed for Wave 4. Wave 5+ will add "
            "the per-user leave-last-out implementation."
        )
