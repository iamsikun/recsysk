"""KuaiVideo dataset loader.

InterFormer's "KuaiVideo" benchmark, distributed by the BARS framework
on Hugging Face Hub as ``reczoo/KuaiVideo_x1``. The release covers
10K users / 3.24M items / 13.7M interactions; label is binary click.

Sources:

* HF Hub repo: ``reczoo/KuaiVideo_x1`` (https://huggingface.co/datasets/reczoo/KuaiVideo_x1)
* Original paper: Li et al. 2019, "Routing Micro-videos via A Temporal
  Graph-guided Recommendation System" (the ALPINE paper).

The harness fetches the BARS-pinned CSV split (``train.csv``,
``valid.csv``, ``test.csv``) via ``fetch_hf_dataset`` with
``allow_patterns=["*.csv"]`` and concatenates them into a single
interaction frame for the standard ``CsvCtrBuilder`` path. The 2048-d
visual embedding columns shipped with the BARS snapshot are dropped by
default to keep the smoke gate cheap; opt back in via ``include_visual``
once you're ready to wire a ``DENSE_VECTOR`` feature.

Note: the BARS distribution is large (~2.27 GB CSV expanded). The
first call materialises the full concatenated frame in memory inside
``CsvCtrBuilder``; budget ~5–6 GB peak. Use the ``max_rows`` knob added
in PR8 if you need a smaller cut.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from recsys.data._download import fetch_hf_dataset

LOGGER = logging.getLogger(__name__)

_HF_REPO_ID = "reczoo/KuaiVideo_x1"

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3] / "datasets"

_EXTRACT_DIRNAME = "KuaiVideo_x1"

# BARS / FuxiCTR pin: split filenames in the repo.
_SPLIT_FILES: tuple[str, ...] = ("train.csv", "valid.csv", "test.csv")

# Default columns kept in the loaded frame. The visual embedding lives
# under ``feature_emb`` (2048-d packed) and is dropped here unless
# ``include_visual=True`` is passed.
_DEFAULT_KEEP_COLUMNS: tuple[str, ...] = ("user_id", "item_id", "is_click")


@dataclass
class KuaiVideoPaths:
    dataset_root: Path
    extract_dir: Path
    splits: dict[str, Path]


def _resolve_paths(dataset_root: Path | None) -> KuaiVideoPaths:
    base = (
        Path(dataset_root).expanduser().resolve() if dataset_root else DEFAULT_DATASET_ROOT
    )
    extract_dir = base / _EXTRACT_DIRNAME
    return KuaiVideoPaths(
        dataset_root=base,
        extract_dir=extract_dir,
        splits={s: extract_dir / s for s in _SPLIT_FILES},
    )


def download(
    dataset_root: Path | str | None = None,
) -> KuaiVideoPaths:
    """Snapshot-download the KuaiVideo_x1 CSVs from HF Hub.

    Re-runs are no-ops once the CSVs exist under ``dataset_root /
    KuaiVideo_x1/``.
    """
    paths = _resolve_paths(Path(dataset_root) if dataset_root else None)
    paths.extract_dir.mkdir(parents=True, exist_ok=True)
    if all(p.exists() for p in paths.splits.values()):
        return paths

    snapshot = fetch_hf_dataset(
        _HF_REPO_ID,
        cache_dir=paths.dataset_root / ".hf_cache",
        allow_patterns=["*.csv"],
    )

    # Symlink the snapshot's CSVs into the stable extract_dir so
    # downstream code doesn't depend on HF's content-addressed paths.
    for split, dest in paths.splits.items():
        if dest.exists():
            continue
        src_candidates = list(snapshot.glob(f"**/{split}"))
        if not src_candidates:
            raise FileNotFoundError(
                f"KuaiVideo: HF snapshot at {snapshot} did not contain {split}"
            )
        # symlink_to may fail across filesystems; fall back to copy.
        try:
            dest.symlink_to(src_candidates[0])
        except OSError:
            import shutil

            shutil.copy2(src_candidates[0], dest)
    return paths


def load(
    dataset_root: Path | str | None = None,
    *,
    auto_download: bool = True,
    include_visual: bool = False,
    max_rows: int | None = None,
) -> pl.DataFrame:
    """Load the concatenated KuaiVideo_x1 interaction frame.

    Returns a polars DataFrame with columns ``[user_id, item_id, is_click]``
    by default. Pass ``include_visual=True`` to keep the 2048-d
    ``feature_emb`` column (still string-packed; downstream code is
    responsible for parsing it into a dense vector).

    ``max_rows`` returns the first ``max_rows`` rows of the concatenated
    frame; useful for smoke runs without paying the full 13.7M-row cost.
    """
    paths = _resolve_paths(Path(dataset_root) if dataset_root else None)
    if not all(p.exists() for p in paths.splits.values()):
        if not auto_download:
            raise FileNotFoundError(
                f"KuaiVideo splits missing under {paths.extract_dir}. "
                f"Re-run with auto_download=True or fetch from "
                f"https://huggingface.co/datasets/{_HF_REPO_ID}"
            )
        download(paths.dataset_root)

    keep_cols = list(_DEFAULT_KEEP_COLUMNS)
    if include_visual:
        keep_cols.append("feature_emb")

    frames: list[pl.DataFrame] = []
    rows_so_far = 0
    for split, path in paths.splits.items():
        df = pl.read_csv(path)
        # Be defensive: only project columns that actually exist; fail
        # loudly if a required default column is missing.
        missing = [c for c in _DEFAULT_KEEP_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(
                f"KuaiVideo split {split} missing required columns {missing}; "
                f"got {df.columns}"
            )
        present_keep = [c for c in keep_cols if c in df.columns]
        df = df.select(present_keep)
        if max_rows is not None:
            remaining = max_rows - rows_so_far
            if remaining <= 0:
                break
            if df.height > remaining:
                df = df.head(remaining)
        frames.append(df)
        rows_so_far += df.height
    return pl.concat(frames)


__all__ = ["KuaiVideoPaths", "download", "load"]
