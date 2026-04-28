"""Frappe dataset loader.

Frappe is a context-aware app-recommendation dataset (Baltrunas et al.
2015). The harness uses the **NFM CTR-formatted mirror** rather than the
raw Frappe TSV because Wukong, NFM, and most of the unified-seq line
benchmark on the NFM-prepared variant: each row is one positive
interaction plus a sampled negative, encoded as libfm with 10 fields
(user, item, daytime, weekday, isweekend, homework, cost, weather,
country, city). Total ~288k rows across train/validation/test.

Source: https://github.com/hexiangnan/neural_factorization_machine

The mirror is small enough (~12 MB total) that we download all three
splits unconditionally; the harness recombines them into a single
interaction frame and runs its own train/val split via the standard
``CsvCtrBuilder`` path. Raw label values are libfm-style ``-1``/``1``;
the loader normalises them to ``0``/``1``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

from recsys.data._download import http_download_atomic

LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3] / "datasets"

_BASE_URL = (
    "https://raw.githubusercontent.com/hexiangnan/"
    "neural_factorization_machine/master/data/frappe"
)
_SPLIT_FILES = {
    "train": "frappe.train.libfm",
    "validation": "frappe.validation.libfm",
    "test": "frappe.test.libfm",
}

# Field order pinned by the NFM mirror's libfm encoding.
FIELD_NAMES: tuple[str, ...] = (
    "user_id",
    "item_id",
    "daytime",
    "weekday",
    "isweekend",
    "homework",
    "cost",
    "weather",
    "country",
    "city",
)


@dataclass
class FrappePaths:
    root: Path
    train: Path
    validation: Path
    test: Path


def _resolve_paths(dataset_root: Path | None) -> FrappePaths:
    base = Path(dataset_root).expanduser().resolve() if dataset_root else DEFAULT_DATASET_ROOT
    root = base / "frappe"
    return FrappePaths(
        root=root,
        train=root / _SPLIT_FILES["train"],
        validation=root / _SPLIT_FILES["validation"],
        test=root / _SPLIT_FILES["test"],
    )


def _is_present(paths: FrappePaths) -> bool:
    return paths.train.exists() and paths.validation.exists() and paths.test.exists()


def download(
    dataset_root: Path | str | None = None,
    *,
    force: bool = False,
) -> FrappePaths:
    paths = _resolve_paths(Path(dataset_root) if dataset_root else None)
    paths.root.mkdir(parents=True, exist_ok=True)
    for split, fname in _SPLIT_FILES.items():
        dest = paths.root / fname
        if dest.exists() and not force:
            continue
        url = f"{_BASE_URL}/{fname}"
        # The NFM mirror is hosted on raw.githubusercontent.com which does
        # not always send Content-Length; disable verification so a
        # missing header doesn't trip the truncation guard.
        http_download_atomic(
            url,
            dest,
            verify_content_length=False,
            progress_label=f"frappe/{split}",
        )
    return paths


def _read_libfm(path: Path) -> pl.DataFrame:
    """Parse one libfm split into a polars DataFrame.

    Each line has the form ``<label> <idx>:1 <idx>:1 ...`` with exactly
    10 ``idx:1`` pairs corresponding to ``FIELD_NAMES``. Labels are
    ``-1`` / ``1`` and are normalised to ``0`` / ``1`` floats.
    """
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            tokens = raw.strip().split()
            if not tokens:
                continue
            if len(tokens) != 1 + len(FIELD_NAMES):
                raise ValueError(
                    f"{path.name}: expected {1 + len(FIELD_NAMES)} tokens "
                    f"(label + {len(FIELD_NAMES)} idx:1 pairs), got {len(tokens)}"
                )
            label_raw = int(tokens[0])
            label = 1.0 if label_raw > 0 else 0.0
            row: dict = {"label": label}
            for field, tok in zip(FIELD_NAMES, tokens[1:]):
                # tok is "<idx>:1"; we keep only the integer index.
                idx_str, _ = tok.split(":", 1)
                row[field] = int(idx_str)
            rows.append(row)
    return pl.DataFrame(rows)


def load(
    dataset_root: Path | str | None = None,
    *,
    splits: Iterable[str] | None = None,
    auto_download: bool = True,
) -> dict[str, pl.DataFrame]:
    """Load the Frappe NFM mirror.

    Returns a dict keyed by split name (``train``, ``validation``,
    ``test``) mapping to a polars DataFrame with columns
    ``[label, *FIELD_NAMES]``. The ``label`` column is float (0.0/1.0).

    If ``auto_download`` is True (default) and any split file is missing,
    the loader fetches it from the NFM GitHub mirror.
    """
    paths = _resolve_paths(Path(dataset_root) if dataset_root else None)
    if not _is_present(paths):
        if not auto_download:
            raise FileNotFoundError(
                f"Frappe split files not found under {paths.root}. "
                f"Re-run with auto_download=True or fetch them manually from "
                f"{_BASE_URL}/{{frappe.train,frappe.validation,frappe.test}}.libfm"
            )
        download(paths.root.parent)

    requested = list(splits) if splits is not None else list(_SPLIT_FILES)
    out: dict[str, pl.DataFrame] = {}
    for split in requested:
        if split not in _SPLIT_FILES:
            raise KeyError(
                f"Unknown Frappe split: {split!r}. Available: {sorted(_SPLIT_FILES)}"
            )
        out[split] = _read_libfm(getattr(paths, split))
    return out


def load_combined(
    dataset_root: Path | str | None = None,
    *,
    auto_download: bool = True,
) -> pl.DataFrame:
    """Concatenate all three Frappe splits into one interaction frame.

    The harness's ``CsvCtrBuilder`` runs its own random train/val split,
    so we hand it a single combined frame rather than the NFM splits.
    """
    splits = load(dataset_root, auto_download=auto_download)
    return pl.concat([splits["train"], splits["validation"], splits["test"]])


__all__ = ["FrappePaths", "FIELD_NAMES", "download", "load", "load_combined"]
