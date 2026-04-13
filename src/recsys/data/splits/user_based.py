"""User-based splitter (stub — Wave 5+)."""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import Dataset

from recsys.data.splits.base import Splitter


@dataclass
class UserBasedSplit(Splitter):
    """Random partition of users into disjoint train/val/test cohorts.

    Not yet implemented. Wave 5+ will partition users deterministically
    so no user appears in more than one slice (user-cold evaluation).
    """

    train_fraction: float = 0.8
    val_fraction: float = 0.1

    def split(
        self,
        dataset: Dataset,
        *,
        seed: int | None = None,
    ) -> tuple[Dataset, Dataset, Dataset]:
        raise NotImplementedError(
            "UserBasedSplit is stubbed for Wave 4. Wave 5+ will add "
            "the user-cohort partition implementation."
        )
