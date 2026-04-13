"""Random fractional splitter.

Wraps :func:`torch.utils.data.random_split` to produce a
``(train, val, test)`` triple from a single fraction. Wave 4 aliases
``test`` to ``val`` — Wave 5+ replaces this with a real three-way split.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, random_split

from recsys.data.splits.base import Splitter


@dataclass
class RandomSplit(Splitter):
    """Random fractional split.

    Parameters
    ----------
    train_fraction:
        Fraction of the dataset assigned to the training slice. The
        remainder goes to val; test currently aliases val.
    """

    train_fraction: float

    def split(
        self,
        dataset: Dataset,
        *,
        seed: int | None = None,
    ) -> tuple[Dataset, Dataset, Dataset]:
        n = len(dataset)
        n_train = int(n * self.train_fraction)
        n_val = n - n_train
        generator: torch.Generator | None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        else:
            generator = None
        train, val = random_split(dataset, [n_train, n_val], generator=generator)
        # Wave 4: test aliases val. Wave 5+ will add a real held-out slice.
        return train, val, val
