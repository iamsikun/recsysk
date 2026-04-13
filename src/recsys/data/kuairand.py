"""KuaiRand dataset loader.

Framework-agnostic loader for the KuaiRand short-video recommendation
dataset (Gao et al., CIKM 2022). No torch or Lightning imports live in
this module; it just acquires the raw files and reads them into polars
DataFrames.

Dataset homepage: https://kuairand.com/
Source archives:

* ``Pure``  — https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
* ``1K``    — https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz
* ``27K``   — https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz

The ``Pure`` variant is the smallest (~45 MB compressed, ~1.4M
interactions) and is used by the v2.0 starter benchmark for smoke
speed. ``1K`` and ``27K`` are configurable for production runs.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

from recsys.data._download import http_download_atomic

LOGGER = logging.getLogger(__name__)

# Default dataset cache — points at the repo-root ``datasets/`` directory
# regardless of the caller's CWD. ``parents[3]`` lifts from
# ``src/recsys/data/kuairand.py`` to the repo root.
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3] / "datasets"


@dataclass(frozen=True)
class KuaiRandVariantSpec:
    """Metadata for a specific KuaiRand archive variant."""

    key: str
    archive_url: str
    archive_name: str
    extract_dirname: str


_VARIANTS: dict[str, KuaiRandVariantSpec] = {
    "pure": KuaiRandVariantSpec(
        key="pure",
        archive_url="https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz",
        archive_name="KuaiRand-Pure.tar.gz",
        extract_dirname="KuaiRand-Pure",
    ),
    "1k": KuaiRandVariantSpec(
        key="1k",
        archive_url="https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz",
        archive_name="KuaiRand-1K.tar.gz",
        extract_dirname="KuaiRand-1K",
    ),
    "27k": KuaiRandVariantSpec(
        key="27k",
        archive_url="https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz",
        archive_name="KuaiRand-27K.tar.gz",
        extract_dirname="KuaiRand-27K",
    ),
}

_HTTP_HEADERS = {
    "User-Agent": "curl/8.4.0",
    "Accept": "*/*",
}


@dataclass
class KuaiRandPaths:
    """Resolved disk paths for a specific KuaiRand variant."""

    dataset_root: Path
    extract_dir: Path  # dataset_root / "KuaiRand-<Variant>"
    data_dir: Path  # extract_dir / "data"
    variant: KuaiRandVariantSpec


def _resolve_variant(variant: str) -> KuaiRandVariantSpec:
    key = variant.lower()
    if key not in _VARIANTS:
        raise ValueError(
            f"Unknown KuaiRand variant {variant!r}; expected one of "
            f"{sorted(_VARIANTS)}"
        )
    return _VARIANTS[key]


def _resolve_paths(
    dataset_root: Path | str | None,
    variant: str,
) -> KuaiRandPaths:
    root = Path(dataset_root or DEFAULT_DATASET_ROOT).expanduser().resolve()
    spec = _resolve_variant(variant)
    extract_dir = root / spec.extract_dirname
    data_dir = extract_dir / "data"
    return KuaiRandPaths(
        dataset_root=root,
        extract_dir=extract_dir,
        data_dir=data_dir,
        variant=spec,
    )


def _is_present(paths: KuaiRandPaths) -> bool:
    if not paths.data_dir.exists():
        return False
    return any(paths.data_dir.glob("*.csv"))


def _download_with_progress(url: str, dest: Path) -> None:
    """KuaiRand wrapper around the shared ``http_download_atomic`` helper."""
    http_download_atomic(
        url,
        dest,
        headers=_HTTP_HEADERS,
        progress_every_bytes=8 * (1 << 20),
        progress_label="KuaiRand download",
    )


def download(
    dataset_root: Path | str | None = None,
    *,
    variant: str = "pure",
    force: bool = False,
) -> KuaiRandPaths:
    """Download and extract a KuaiRand variant if missing.

    Parameters
    ----------
    dataset_root:
        Parent directory that should hold the ``KuaiRand-*`` subtree.
        Defaults to :data:`DEFAULT_DATASET_ROOT`.
    variant:
        One of ``"pure"`` (default), ``"1k"``, ``"27k"``.
    force:
        If ``True``, re-download and re-extract even if present.
    """
    paths = _resolve_paths(dataset_root, variant)
    if _is_present(paths) and not force:
        LOGGER.info("KuaiRand[%s] already present at %s", variant, paths.data_dir)
        return paths

    paths.dataset_root.mkdir(parents=True, exist_ok=True)
    archive_path = paths.dataset_root / paths.variant.archive_name
    _download_with_progress(paths.variant.archive_url, archive_path)

    staging = paths.dataset_root / f".{paths.variant.extract_dirname}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    LOGGER.info("KuaiRand: extracting %s -> %s", archive_path, staging)
    try:
        with tarfile.open(archive_path, "r:gz") as tf:
            try:
                tf.extractall(staging, filter="data")  # type: ignore[arg-type]
            except TypeError:  # Python < 3.12
                tf.extractall(staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    produced_roots = [p for p in staging.iterdir() if p.is_dir()]
    if (
        len(produced_roots) == 1
        and produced_roots[0].name == paths.variant.extract_dirname
    ):
        source = produced_roots[0]
    else:
        source = staging
    if paths.extract_dir.exists():
        shutil.rmtree(paths.extract_dir)
    source.replace(paths.extract_dir)
    shutil.rmtree(staging, ignore_errors=True)

    if not _is_present(paths):
        raise RuntimeError(
            f"KuaiRand[{variant}] extraction completed but no CSVs found "
            f"under {paths.data_dir}"
        )
    return paths


def _find_csvs(data_dir: Path, substring: str) -> list[Path]:
    """Return all CSVs under ``data_dir`` whose name contains ``substring``.

    KuaiRand splits some logs across multiple date-range files (e.g.
    ``log_standard_4_08_to_4_21_pure.csv`` and
    ``log_standard_4_22_to_5_08_pure.csv``); the loader concatenates them
    in :func:`load`. Raises if no matches.
    """
    matches = sorted(p for p in data_dir.glob("*.csv") if substring in p.name)
    if not matches:
        raise FileNotFoundError(
            f"No KuaiRand CSV matching {substring!r} under {data_dir}. "
            f"Found: {[p.name for p in data_dir.glob('*.csv')]}"
        )
    return matches


# Tables the loader knows how to return. Filenames in the KuaiRand
# archives include date ranges (e.g. ``log_standard_4_08_to_4_21_pure.csv``),
# so we locate them by substring.
_TABLE_SUBSTRINGS: dict[str, str] = {
    "log_standard": "log_standard",
    "log_random": "log_random",
    "user_features": "user_features",
    "video_features_basic": "video_features_basic",
    "video_features_statistic": "video_features_statistic",
}


def load(
    dataset_root: Path | str | None = None,
    *,
    variant: str = "pure",
    tables: Iterable[str] | None = None,
    auto_download: bool = True,
) -> dict[str, pl.DataFrame]:
    """Load KuaiRand tables into polars DataFrames.

    Parameters
    ----------
    dataset_root:
        Directory holding the ``KuaiRand-*`` subtree. Defaults to
        :data:`DEFAULT_DATASET_ROOT`.
    variant:
        One of ``"pure"``, ``"1k"``, ``"27k"``. Default ``"pure"``.
    tables:
        Iterable of table keys (see :data:`_TABLE_SUBSTRINGS`). If
        ``None``, defaults to ``["log_standard"]``.
    auto_download:
        If ``True`` (default) and data is missing, call :func:`download`
        first. If ``False`` and data is missing, raise
        :class:`FileNotFoundError`.
    """
    paths = _resolve_paths(dataset_root, variant)
    if not _is_present(paths):
        if not auto_download:
            raise FileNotFoundError(
                f"KuaiRand[{variant}] not found at {paths.data_dir}; "
                "pass auto_download=True or call recsys.data.kuairand.download()"
            )
        paths = download(dataset_root, variant=variant)

    if tables is None:
        tables = ["log_standard"]
    tables_list = list(tables)
    unknown = sorted(set(tables_list) - set(_TABLE_SUBSTRINGS))
    if unknown:
        raise ValueError(f"Unknown KuaiRand tables: {unknown}")

    out: dict[str, pl.DataFrame] = {}
    for name in tables_list:
        substring = _TABLE_SUBSTRINGS[name]
        fpaths = _find_csvs(paths.data_dir, substring)
        frames: list[pl.DataFrame] = []
        for fpath in fpaths:
            LOGGER.info("KuaiRand: reading %s", fpath)
            frames.append(pl.read_csv(fpath))
        out[name] = pl.concat(frames) if len(frames) > 1 else frames[0]
    return out


__all__ = [
    "DEFAULT_DATASET_ROOT",
    "KuaiRandPaths",
    "KuaiRandVariantSpec",
    "download",
    "load",
]
