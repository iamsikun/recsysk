"""Torch-backed algorithm implementations.

Importing this package imports each algorithm module for its registration
side effects (``@MODEL_REGISTRY.register(...)``).
"""

from recsys.algorithms.torch import deepfm  # noqa: F401
from recsys.algorithms.torch import din  # noqa: F401
