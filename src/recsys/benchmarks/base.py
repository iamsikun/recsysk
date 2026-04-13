"""Benchmark protocol + BenchmarkData bundle.

A Benchmark pins together a dataset, an evaluation protocol (split + metric
list + negative-sampling config), and a task. Calling ``build()`` returns
a :class:`BenchmarkData` bundle the runner hands to the task's evaluator.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from recsys.tasks.base import Task


@dataclass
class BenchmarkData:
    """Fitted / setup-ed datasets + metadata for a benchmark.

    Wave 3 note: the ``test`` slot historically aliased ``val``; for
    competition-style benchmarks it may also point at an unlabeled
    test split that :meth:`Task.export_predictions` iterates over.
    """

    train: Any
    val: Any
    test: Any
    feature_map: dict[str, int]
    feature_specs: list
    datamodule: Any
    metadata: dict = field(default_factory=dict)


class Benchmark(ABC):
    """Abstract base for benchmarks.

    Subclasses pin the following as class attributes:

    * :attr:`name` — the registry key / human-readable identifier.
    * :attr:`task` — the :class:`Task` used to evaluate algorithms.
    * :attr:`metric_names` — the full pinned metric list.

    :meth:`build` instantiates the underlying datamodule and returns a
    :class:`BenchmarkData`. :meth:`version` returns a short hash stable
    across runs for the same benchmark configuration, useful for keying
    result stores in Wave 5.
    """

    name: str
    task: "Task"
    metric_names: list[str]

    @abstractmethod
    def build(self) -> BenchmarkData:
        raise NotImplementedError

    def version(self) -> str:
        """Return an 8-char hash of (name, metric_names, eval_cfg)."""
        eval_cfg = getattr(self, "_eval_cfg", {}) or {}
        payload = {
            "name": self.name,
            "metrics": list(self.metric_names),
            "eval": eval_cfg,
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:8]

    def write_submission(
        self,
        predictions: Iterable[tuple[Any, float]],
        out_path: Path,
    ) -> None:
        """Write a stream of ``(row_id, score)`` tuples to ``out_path``.

        The default implementation produces a 2-column CSV with header
        ``row_id,score``. Competition-specific benchmarks can override
        this to emit a different shape (e.g. Kaggle ``id,target`` CSVs,
        parquet for fast large-N, or multi-column layouts for top-k
        retrieval submissions). Override-by-subclass keeps the output
        format owned by the benchmark, not the task.
        """
        import polars as pl

        rows = [{"row_id": r, "score": s} for r, s in predictions]
        df = pl.DataFrame(rows)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(str(out_path))
