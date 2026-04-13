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
