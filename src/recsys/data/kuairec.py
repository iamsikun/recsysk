"""KuaiRec dataset loader.

Framework-agnostic loader for the KuaiRec short-video recommendation
dataset (Gao et al., CIKM 2022). No torch or Lightning imports live in
this module; it just knows how to acquire the raw files and read them
into polars DataFrames.

Dataset homepage: https://kuairec.com/
Source archive:   https://zenodo.org/records/18164998/files/KuaiRec.zip

The archive extracts to a ``KuaiRec/data/`` directory containing the
CSVs we care about:

* ``big_matrix.csv``      — full user x video interaction log
* ``small_matrix.csv``    — dense subset (~1.4K users x 3.3K videos)
* ``user_features.csv``
* ``item_categories.csv``
* ``item_daily_features.csv``
* ``social_network.csv``
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

from recsys.data._download import http_download_atomic

LOGGER = logging.getLogger(__name__)

# Default dataset cache — points at the repo-root ``datasets/`` directory
# regardless of the caller's CWD. ``parents[3]`` lifts from
# ``src/recsys/data/kuairec.py`` to the repo root.
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3] / "datasets"

KUAIREC_ARCHIVE_URL = "https://zenodo.org/records/18164998/files/KuaiRec.zip"
KUAIREC_ARCHIVE_NAME = "KuaiRec.zip"
# After unzip, files live under this relative directory inside the
# dataset root.
KUAIREC_EXTRACT_DIRNAME = "KuaiRec"
KUAIREC_DATA_SUBDIR = "data"

# Sentinel file used to check whether the dataset is already on disk.
KUAIREC_SENTINEL_FILE = "small_matrix.csv"

# Requests to Zenodo with a default Python urllib User-Agent currently
# get a 403. Use a curl-style UA which zenodo's filter lets through.
_HTTP_HEADERS = {
    "User-Agent": "curl/8.4.0",
    "Accept": "*/*",
}


@dataclass
class KuaiRecPaths:
    """Resolved disk paths for KuaiRec files."""

    dataset_root: Path
    extract_dir: Path  # dataset_root / "KuaiRec"
    data_dir: Path  # extract_dir / "data"


def _resolve_paths(dataset_root: Path | str | None) -> KuaiRecPaths:
    root = Path(dataset_root or DEFAULT_DATASET_ROOT).expanduser().resolve()
    extract_dir = root / KUAIREC_EXTRACT_DIRNAME
    data_dir = extract_dir / KUAIREC_DATA_SUBDIR
    return KuaiRecPaths(dataset_root=root, extract_dir=extract_dir, data_dir=data_dir)


def _is_present(paths: KuaiRecPaths) -> bool:
    return (paths.data_dir / KUAIREC_SENTINEL_FILE).exists()


def _download_with_progress(url: str, dest: Path) -> None:
    """KuaiRec wrapper around the shared ``http_download_atomic`` helper."""
    http_download_atomic(
        url,
        dest,
        headers=_HTTP_HEADERS,
        progress_label="KuaiRec download",
    )


def download(
    dataset_root: Path | str | None = None,
    *,
    force: bool = False,
) -> KuaiRecPaths:
    """Download and extract the KuaiRec archive if missing.

    Parameters
    ----------
    dataset_root:
        Parent directory that should hold the ``KuaiRec/`` subtree.
        Defaults to :data:`DEFAULT_DATASET_ROOT`.
    force:
        If ``True``, re-download and re-extract even if the sentinel
        file is already present.
    """
    paths = _resolve_paths(dataset_root)
    if _is_present(paths) and not force:
        LOGGER.info("KuaiRec already present at %s", paths.data_dir)
        return paths

    paths.dataset_root.mkdir(parents=True, exist_ok=True)
    archive_path = paths.dataset_root / KUAIREC_ARCHIVE_NAME

    _download_with_progress(KUAIREC_ARCHIVE_URL, archive_path)

    # Extract to a staging directory first so a mid-extract failure
    # leaves no partial KuaiRec/ tree behind.
    staging = paths.dataset_root / f".{KUAIREC_EXTRACT_DIRNAME}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    LOGGER.info("KuaiRec: extracting %s -> %s", archive_path, staging)
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # The archive root is ``KuaiRec/`` — promote it to its final home.
    produced_roots = [p for p in staging.iterdir() if p.is_dir()]
    if len(produced_roots) == 1 and produced_roots[0].name == KUAIREC_EXTRACT_DIRNAME:
        source = produced_roots[0]
    else:
        source = staging
    if paths.extract_dir.exists():
        shutil.rmtree(paths.extract_dir)
    source.replace(paths.extract_dir)
    shutil.rmtree(staging, ignore_errors=True)

    if not _is_present(paths):
        raise RuntimeError(
            f"KuaiRec extraction completed but sentinel file "
            f"{KUAIREC_SENTINEL_FILE!r} not found under {paths.data_dir}"
        )
    # Keep the zip archive — it's immutable and saves re-downloading.
    return paths


# Tables the loader knows how to return. Keyed by the short table name
# used by callers.
_TABLE_FILES: dict[str, str] = {
    "small_matrix": "small_matrix.csv",
    "big_matrix": "big_matrix.csv",
    "user_features": "user_features.csv",
    "item_categories": "item_categories.csv",
    "item_daily_features": "item_daily_features.csv",
    "social_network": "social_network.csv",
}


def load(
    dataset_root: Path | str | None = None,
    *,
    tables: Iterable[str] | None = None,
    variant: str = "small",
    auto_download: bool = True,
) -> dict[str, pl.DataFrame]:
    """Load KuaiRec tables into polars DataFrames.

    Parameters
    ----------
    dataset_root:
        Directory holding the ``KuaiRec/`` subtree. Defaults to
        :data:`DEFAULT_DATASET_ROOT`.
    tables:
        Iterable of table keys to load (see :data:`_TABLE_FILES`). If
        ``None``, defaults to ``{variant}_matrix`` only.
    variant:
        ``"small"`` (default) loads the dense 1.4K x 3.3K matrix;
        ``"big"`` loads the full 7.2K x 10.7K matrix.
    auto_download:
        If ``True`` (default) and the data is missing, call
        :func:`download` first. If ``False`` and the data is missing,
        raise :class:`FileNotFoundError`.
    """
    paths = _resolve_paths(dataset_root)
    if not _is_present(paths):
        if not auto_download:
            raise FileNotFoundError(
                f"KuaiRec not found at {paths.data_dir}; "
                "pass auto_download=True or call recsys.data.kuairec.download()"
            )
        paths = download(dataset_root)

    if variant not in {"small", "big"}:
        raise ValueError(f"variant must be 'small' or 'big', got {variant!r}")

    if tables is None:
        tables = [f"{variant}_matrix"]
    tables_list = list(tables)
    unknown = sorted(set(tables_list) - set(_TABLE_FILES))
    if unknown:
        raise ValueError(f"Unknown KuaiRec tables: {unknown}")

    out: dict[str, pl.DataFrame] = {}
    for name in tables_list:
        fname = _TABLE_FILES[name]
        fpath = paths.data_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Expected {fpath} but it does not exist")
        LOGGER.info("KuaiRec: reading %s", fpath)
        df = pl.read_csv(fpath)
        out[name] = df
    return out


__all__ = [
    "DEFAULT_DATASET_ROOT",
    "KUAIREC_ARCHIVE_URL",
    "KuaiRecPaths",
    "download",
    "load",
]
