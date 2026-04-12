"""Compatibility shim — the real implementations now live under
:mod:`recsys.algorithms.torch`. Importing :mod:`recsys.models` triggers
importing :mod:`recsys.algorithms` so that all ``@MODEL_REGISTRY.register``
decorators run. Deleted in Phase 7.
"""

from recsys.algorithms import torch as _a  # noqa: F401
