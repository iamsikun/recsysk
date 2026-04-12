"""Framework-agnostic algorithm package.

Importing this package force-imports the subpackages so that any
algorithms registered via side-effect decorators (e.g. ``ALGO_REGISTRY``)
become available to consumers that only import ``recsys.algorithms``.
"""

from recsys.algorithms import torch as _torch  # noqa: F401
from recsys.algorithms import classical as _c  # noqa: F401
