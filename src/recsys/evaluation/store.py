"""Parquet-backed result store for Phase 7.

A ``ResultStore`` persists one parquet file per benchmark under a
``results_dir``. Each row is a :class:`RunResult` capturing the
(benchmark, algo, config_hash, seed, timestamp) key plus the metric
dict and a few provenance fields (runtime, code sha, environment).

Polars is used for read/write since it is already a direct dep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl


@dataclass
class RunResult:
    """One row in the result store."""

    benchmark: str
    benchmark_version: str
    algo: str
    algo_config_hash: str
    seed: int
    metrics: dict[str, float]
    runtime_s: float
    timestamp: str  # ISO 8601 UTC
    code_sha: str | None = None
    env_fingerprint: str | None = None
    # Path to a serialized model (Lightning ``.ckpt`` for torch algos or
    # a pickled classical algo). Consumed by ``recsys submit`` to load
    # the fitted model for offline prediction export. ``None`` when
    # persistence is disabled or the algo doesn't implement save/load.
    model_checkpoint_path: str | None = None


class ResultStore:
    """Append-only parquet store keyed by benchmark name.

    Each benchmark writes to ``results_dir / f"{benchmark}.parquet"``.
    ``write`` appends a single row, using ``diagonal_relaxed`` concat so
    metric columns can grow across runs without breaking older rows.
    """

    def __init__(self, results_dir: Path | str):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, benchmark: str) -> Path:
        return self.results_dir / f"{benchmark}.parquet"

    @staticmethod
    def _row_from_result(result: RunResult) -> dict[str, Any]:
        row: dict[str, Any] = {
            "benchmark": result.benchmark,
            "benchmark_version": result.benchmark_version,
            "algo": result.algo,
            "algo_config_hash": result.algo_config_hash,
            "seed": int(result.seed),
            "runtime_s": float(result.runtime_s),
            "timestamp": result.timestamp,
            "code_sha": result.code_sha,
            "env_fingerprint": result.env_fingerprint,
            "model_checkpoint_path": result.model_checkpoint_path,
        }
        for name, value in result.metrics.items():
            row[f"metric.{name}"] = float(value)
        return row

    def write(self, result: RunResult) -> None:
        row = self._row_from_result(result)
        new_df = pl.DataFrame([row])
        path = self._path(result.benchmark)
        if path.exists():
            old = pl.read_parquet(path)
            df = pl.concat([old, new_df], how="diagonal_relaxed")
        else:
            df = new_df
        df.write_parquet(path)

    def query(
        self,
        benchmark: str,
        algos: list[str] | None = None,
    ) -> pl.DataFrame:
        path = self._path(benchmark)
        if not path.exists():
            return pl.DataFrame()
        df = pl.read_parquet(path)
        if algos:
            df = df.filter(pl.col("algo").is_in(algos))
        return df

    def get_run(
        self,
        benchmark: str,
        algo: str | None = None,
        algo_config_hash: str | None = None,
        seed: int | None = None,
        latest: bool = True,
    ) -> RunResult | None:
        """Return a single ``RunResult`` matching the provided filters.

        When multiple rows match, ``latest=True`` (default) picks the
        most recent by timestamp. Returns ``None`` if no row matches.
        """
        df = self.query(benchmark, algos=[algo] if algo else None)
        if df.is_empty():
            return None
        if algo_config_hash is not None:
            df = df.filter(pl.col("algo_config_hash") == algo_config_hash)
        if seed is not None:
            df = df.filter(pl.col("seed") == int(seed))
        if df.is_empty():
            return None
        if latest:
            df = df.sort("timestamp", descending=True).head(1)
        row = df.row(0, named=True)

        metric_cols = [c for c in df.columns if c.startswith("metric.")]
        metrics = {c.removeprefix("metric."): float(row[c]) for c in metric_cols}
        return RunResult(
            benchmark=str(row["benchmark"]),
            benchmark_version=str(row["benchmark_version"]),
            algo=str(row["algo"]),
            algo_config_hash=str(row["algo_config_hash"]),
            seed=int(row["seed"]),
            metrics=metrics,
            runtime_s=float(row["runtime_s"]),
            timestamp=str(row["timestamp"]),
            code_sha=row.get("code_sha"),
            env_fingerprint=row.get("env_fingerprint"),
            model_checkpoint_path=row.get("model_checkpoint_path"),
        )
