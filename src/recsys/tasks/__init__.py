"""Task protocols and concrete task implementations.

Importing this package registers ``ctr``, ``retrieval``, and ``sequential``
tasks into :data:`recsys.utils.TASK_REGISTRY`.
"""

from recsys.tasks.base import Task
from recsys.tasks.ctr import CTRTask
from recsys.tasks.retrieval import RetrievalTask
from recsys.tasks.sequential import SequentialTask

__all__ = ["Task", "CTRTask", "RetrievalTask", "SequentialTask"]
