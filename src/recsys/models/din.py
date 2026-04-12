"""Compatibility shim — ``DeepInterestNetwork`` moved to
:mod:`recsys.algorithms.torch.din`.

This shim is kept so that ``recsys.runner``'s ``import recsys.models`` line
and any user code doing ``from recsys.models.din import DeepInterestNetwork``
still works. Deleted in Phase 7.
"""

from recsys.algorithms.torch.din import *  # noqa: F401,F403
from recsys.algorithms.torch.din import (  # noqa: F401
    DeepInterestNetwork,
    LocalActivationUnit,
)
