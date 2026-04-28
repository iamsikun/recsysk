"""MicroVideo (THACIL) dataset loader.

Wukong's MicroVideo dataset is the THACIL release (Chen et al. 2018, ACM
MM): ~10,986 users / 1,704,880 items / 12,737,619 interactions. The
release ships pickled behaviour sequences and pre-extracted multimodal
embeddings on a password-protected page (``ms7x``), so v1 does **not**
attempt to download — the loader validates the manual-extract layout
under ``./datasets/microvideo_thacil/`` and raises
:class:`FileNotFoundError` with the source URL and password hint
otherwise.

The harness consumes a tabular ``interactions.csv`` view of THACIL —
users converting the original pickle release should produce a CSV with
columns ``[user_id, item_id, label]`` (label is implicit click / sampled
negative, 0/1).

Source: https://github.com/Ocxs/THACIL
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl

LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3] / "datasets"

_SOURCE_URL = "https://github.com/Ocxs/THACIL"
_PASSWORD_HINT = (
    "The THACIL release page is password-protected; the README on the "
    "GitHub repo lists the password as `ms7x`."
)


@dataclass
class MicroVideoPaths:
    root: Path
    interactions: Path


def _resolve_paths(dataset_root: Path | None) -> MicroVideoPaths:
    base = Path(dataset_root).expanduser().resolve() if dataset_root else DEFAULT_DATASET_ROOT
    root = base / "microvideo_thacil"
    return MicroVideoPaths(root=root, interactions=root / "interactions.csv")


def _check_layout(paths: MicroVideoPaths) -> None:
    if paths.interactions.exists():
        return
    raise FileNotFoundError(
        "MicroVideo (THACIL) dataset not found. Expected:\n"
        f"  - {paths.interactions}\n"
        f"with columns [user_id, item_id, label].\n"
        f"Acquire from {_SOURCE_URL} (the original release ships as pickled "
        "behaviour sequences + multimodal embeddings; convert to a CSV with "
        "the columns above).\n"
        f"{_PASSWORD_HINT}\n"
        "v1 does not auto-download."
    )


def load(
    dataset_root: Path | str | None = None,
) -> pl.DataFrame:
    """Load the MicroVideo (THACIL) interactions table.

    Returns a polars DataFrame with columns ``[user_id, item_id, label]``.
    The label is binary (0/1).
    """
    paths = _resolve_paths(Path(dataset_root) if dataset_root else None)
    _check_layout(paths)
    return pl.read_csv(paths.interactions)


__all__ = ["MicroVideoPaths", "load"]
