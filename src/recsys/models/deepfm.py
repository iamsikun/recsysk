"""Compatibility shim — ``DeepFM`` moved to :mod:`recsys.algorithms.torch.deepfm`.

This shim is kept so that ``recsys.runner``'s ``import recsys.models`` line
and any user code doing ``from recsys.models.deepfm import DeepFM`` still
works. Deleted in Phase 7.
"""

from recsys.algorithms.torch.deepfm import *  # noqa: F401,F403
from recsys.algorithms.torch.deepfm import DeepFM  # noqa: F401
