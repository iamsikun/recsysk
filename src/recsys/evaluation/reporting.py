"""Reporting helpers that aggregate stored runs into mean/std tables."""

from __future__ import annotations

import polars as pl

from recsys.evaluation.store import ResultStore


def summary_table(store: ResultStore, benchmark: str) -> pl.DataFrame:
    """Return one row per algorithm with per-metric mean/std and n_runs.

    Metric columns are discovered from ``metric.<name>`` prefixes so new
    metrics land automatically.
    """
    df = store.query(benchmark)
    if df.is_empty():
        return df
    metric_cols = [c for c in df.columns if c.startswith("metric.")]
    agg_exprs: list[pl.Expr] = []
    for col in metric_cols:
        short = col.removeprefix("metric.")
        agg_exprs.append(pl.col(col).mean().alias(f"{short}_mean"))
        agg_exprs.append(pl.col(col).std().alias(f"{short}_std"))
    agg_exprs.append(pl.len().alias("n_runs"))
    return df.group_by("algo").agg(agg_exprs).sort("algo")


def format_table(df: pl.DataFrame) -> str:
    """Render a summary DataFrame for the CLI."""
    if df.is_empty():
        return "(no results)"
    with pl.Config(
        tbl_rows=100,
        tbl_cols=100,
        tbl_width_chars=240,
        fmt_float="mixed",
    ):
        return str(df)
