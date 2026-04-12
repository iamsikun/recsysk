"""Framework-agnostic algorithm package.

Importing this package force-imports the torch subpackage so that any
algorithms registered via side-effect decorators (e.g. ``MODEL_REGISTRY``)
become available to consumers that only import ``recsys.algorithms``.
"""

from recsys.algorithms import torch as _torch  # noqa: F401
