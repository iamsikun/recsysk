"""Split modules.

Wave 4 (P6) introduces the Splitter protocol plus a random-fraction
implementation used today by the MovieLens benchmarks. Temporal, leave-
last-out, and user-based splits are stubbed and will be filled in in
Wave 5+.
"""

from recsys.data.splits.base import Splitter
from recsys.data.splits.random_split import RandomSplit

__all__ = ["Splitter", "RandomSplit"]
