"""Criteo Terabyte (24-day click logs) loader.

Wukong's largest public CTR benchmark — 24 daily TSV shards totaling
~1.3 TB uncompressed (~370 GB gzipped) and ~4.4B rows. The harness uses
the official HF Hub mirror at ``criteo/CriteoClickLogs`` because the
Criteo AI Lab portal URLs rot frequently and the HF mirror is
verified-source under CC-BY-NC-SA-4.0.

Schema per shard (TSV, no header):

    label  int_1  int_2 ... int_13  cat_1  cat_2 ... cat_26

The loader fetches only the days you ask for (defaults to ``["day_0"]``)
and returns a polars DataFrame ready for ``CsvCtrBuilder``. Combine
with ``CsvCtrConfig.max_rows`` to keep the full frame out of memory —
``max_rows: 1_000_000`` keeps peak memory under ~200 MB.

Note: This dataset is not viable on a laptop without ``max_rows``.
Pick a row cap and expect the first run to spend most of its time in
the HF Hub download.
"""

from __future__ import annotations

import gzip
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

from recsys.data._download import fetch_hf_dataset

LOGGER = logging.getLogger(__name__)

_HF_REPO_ID = "criteo/CriteoClickLogs"

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3] / "datasets"

_EXTRACT_DIRNAME = "CriteoClickLogs"

INT_COLS: tuple[str, ...] = tuple(f"int_{i}" for i in range(1, 14))
CAT_COLS: tuple[str, ...] = tuple(f"cat_{i}" for i in range(1, 27))
ALL_COLS: tuple[str, ...] = ("label", *INT_COLS, *CAT_COLS)


@dataclass
class CriteoPaths:
    dataset_root: Path
    extract_dir: Path
    days: dict[str, Path]


def _resolve_paths(
    dataset_root: Path | None, days: Iterable[str]
) -> CriteoPaths:
    base = (
        Path(dataset_root).expanduser().resolve() if dataset_root else DEFAULT_DATASET_ROOT
    )
    extract_dir = base / _EXTRACT_DIRNAME
    return CriteoPaths(
        dataset_root=base,
        extract_dir=extract_dir,
        days={d: extract_dir / f"{d}.tsv" for d in days},
    )


def _ensure_uncompressed(gz_path: Path, tsv_path: Path) -> None:
    """gunzip ``gz_path`` to ``tsv_path`` if the latter is missing."""
    if tsv_path.exists():
        return
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Criteo: decompressing %s -> %s", gz_path, tsv_path)
    tmp = tsv_path.with_suffix(tsv_path.suffix + ".part")
    try:
        with gzip.open(gz_path, "rb") as gz, tmp.open("wb") as out:
            shutil.copyfileobj(gz, out, length=1 << 22)
    except BaseException:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    tmp.replace(tsv_path)


def download(
    days: Iterable[str] = ("day_0",),
    dataset_root: Path | str | None = None,
) -> CriteoPaths:
    """Snapshot-download the requested days from the HF Hub mirror.

    ``days`` are repo-relative names, e.g. ``["day_0", "day_1"]``. Each
    day is a single ~57 GB gzipped TSV.
    """
    days = list(days)
    paths = _resolve_paths(Path(dataset_root) if dataset_root else None, days)
    paths.extract_dir.mkdir(parents=True, exist_ok=True)
    if all(p.exists() for p in paths.days.values()):
        return paths

    snapshot = fetch_hf_dataset(
        _HF_REPO_ID,
        cache_dir=paths.dataset_root / ".hf_cache",
        allow_patterns=[f"{d}.gz" for d in days],
    )
    for day, tsv_path in paths.days.items():
        gz_candidates = list(snapshot.glob(f"**/{day}.gz"))
        if not gz_candidates:
            raise FileNotFoundError(
                f"Criteo: HF snapshot at {snapshot} did not contain {day}.gz"
            )
        _ensure_uncompressed(gz_candidates[0], tsv_path)
    return paths


def load(
    dataset_root: Path | str | None = None,
    *,
    days: Iterable[str] = ("day_0",),
    max_rows: int | None = None,
    auto_download: bool = True,
) -> pl.DataFrame:
    """Load the requested Criteo Terabyte days into one polars DataFrame.

    Pass ``max_rows`` to truncate after concatenation; combine with
    ``days=["day_0"]`` for a smoke-gate-friendly cut. The TSV files have
    no header — the loader applies the canonical column names
    ``[label, int_1..13, cat_1..26]``.
    """
    days = list(days)
    paths = _resolve_paths(Path(dataset_root) if dataset_root else None, days)
    if not all(p.exists() for p in paths.days.values()):
        if not auto_download:
            raise FileNotFoundError(
                f"Criteo days missing under {paths.extract_dir}. "
                f"Re-run with auto_download=True or fetch from "
                f"https://huggingface.co/datasets/{_HF_REPO_ID}"
            )
        download(days=days, dataset_root=paths.dataset_root)

    frames: list[pl.DataFrame] = []
    rows_so_far = 0
    for day, path in paths.days.items():
        kwargs: dict = dict(
            separator="\t",
            has_header=False,
            new_columns=list(ALL_COLS),
            ignore_errors=True,
        )
        if max_rows is not None:
            remaining = max_rows - rows_so_far
            if remaining <= 0:
                break
            kwargs["n_rows"] = remaining
        df = pl.read_csv(path, **kwargs)
        frames.append(df)
        rows_so_far += df.height
    return pl.concat(frames)


__all__ = ["CriteoPaths", "INT_COLS", "CAT_COLS", "ALL_COLS", "download", "load"]
