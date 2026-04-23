"""Amazon Reviews 2023 dataset loader.

Framework-agnostic loader for the Amazon Reviews 2023 release by the
McAuley Lab (https://amazon-reviews-2023.github.io/). No torch or
Lightning imports live in this module; it acquires the raw files from
Hugging Face Hub and reads them into polars DataFrames.

Source:

* HF Hub repo: ``McAuley-Lab/Amazon-Reviews-2023``
* Per-category reviews: ``raw/review_categories/<Category>.jsonl``
* Per-category metadata: ``raw/meta_categories/meta_<Category>.jsonl``

Default category is ``All_Beauty`` — the smallest (~701K ratings, ~632K
users, ~112K items) and therefore the best fit for the repo's smoke
gate. ``Books`` (~29M ratings), ``Electronics`` (~44M), and
``Sports_and_Outdoors`` are also registered; those categories incur
multi-GB downloads on first run.

Acquisition is via ``fetch_hf_dataset`` (HF Hub ``snapshot_download``)
rather than the direct-archive ``http_download_atomic`` used by
KuaiRec/KuaiRand — the McAuley Lab mirror on HF Hub is public, no auth
needed, and integrity is handled by the Hub rather than
Content-Length.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

from recsys.data._download import fetch_hf_dataset

LOGGER = logging.getLogger(__name__)

_HF_REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"

# Default dataset cache — points at the repo-root ``datasets/`` directory
# regardless of the caller's CWD. ``parents[3]`` lifts from
# ``src/recsys/data/amazon.py`` to the repo root.
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3] / "datasets"

_EXTRACT_DIRNAME = "Amazon-Reviews-2023"


@dataclass(frozen=True)
class AmazonCategorySpec:
    """Metadata for a single Amazon Reviews category."""

    key: str  # user-facing identifier (lowercase, snake_case)
    hf_name: str  # filename component in the HF repo (e.g. ``All_Beauty``)

    @property
    def review_filename(self) -> str:
        return f"raw/review_categories/{self.hf_name}.jsonl"

    @property
    def meta_filename(self) -> str:
        return f"raw/meta_categories/meta_{self.hf_name}.jsonl"


_CATEGORIES: dict[str, AmazonCategorySpec] = {
    "all_beauty": AmazonCategorySpec(key="all_beauty", hf_name="All_Beauty"),
    "books": AmazonCategorySpec(key="books", hf_name="Books"),
    "electronics": AmazonCategorySpec(key="electronics", hf_name="Electronics"),
    "sports_and_outdoors": AmazonCategorySpec(
        key="sports_and_outdoors", hf_name="Sports_and_Outdoors"
    ),
}


@dataclass
class AmazonPaths:
    """Resolved disk paths for a specific Amazon Reviews category."""

    dataset_root: Path
    extract_dir: Path  # dataset_root / "Amazon-Reviews-2023"
    review_path: Path
    meta_path: Path
    category: AmazonCategorySpec


def _resolve_category(category: str) -> AmazonCategorySpec:
    key = category.lower()
    if key not in _CATEGORIES:
        raise ValueError(
            f"Unknown Amazon Reviews category {category!r}; expected one of "
            f"{sorted(_CATEGORIES)}"
        )
    return _CATEGORIES[key]


def _resolve_paths(
    dataset_root: Path | str | None,
    category: str,
) -> AmazonPaths:
    root = Path(dataset_root or DEFAULT_DATASET_ROOT).expanduser().resolve()
    spec = _resolve_category(category)
    extract_dir = root / _EXTRACT_DIRNAME
    return AmazonPaths(
        dataset_root=root,
        extract_dir=extract_dir,
        review_path=extract_dir / spec.review_filename,
        meta_path=extract_dir / spec.meta_filename,
        category=spec,
    )


def _is_present(paths: AmazonPaths) -> bool:
    return paths.review_path.exists() and paths.meta_path.exists()


def download(
    dataset_root: Path | str | None = None,
    *,
    category: str = "all_beauty",
    force: bool = False,
) -> AmazonPaths:
    """Download review + metadata JSONL files for a single Amazon category.

    Uses :func:`recsys.data._download.fetch_hf_dataset` to pull the two
    per-category files from the Hugging Face Hub. Subsequent calls for
    the same category hit the Hub cache.

    Parameters
    ----------
    dataset_root:
        Parent directory that will hold the ``Amazon-Reviews-2023`` subtree.
        Defaults to :data:`DEFAULT_DATASET_ROOT`.
    category:
        One of ``"all_beauty"`` (default), ``"books"``, ``"electronics"``,
        ``"sports_and_outdoors"``.
    force:
        If ``True``, re-download even if present (routes through HF Hub
        which will refresh from the remote).
    """
    paths = _resolve_paths(dataset_root, category)
    if _is_present(paths) and not force:
        LOGGER.info(
            "Amazon-Reviews-2023[%s] already present at %s",
            category,
            paths.extract_dir,
        )
        return paths

    paths.dataset_root.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Amazon-Reviews-2023[%s]: fetching from %s -> %s",
        category,
        _HF_REPO_ID,
        paths.extract_dir,
    )
    local_root = fetch_hf_dataset(
        _HF_REPO_ID,
        cache_dir=paths.extract_dir,
        allow_patterns=[
            paths.category.review_filename,
            paths.category.meta_filename,
        ],
    )
    # ``snapshot_download`` places files under an opaque snapshot path
    # inside ``cache_dir``. Re-resolve review/meta paths against that
    # snapshot root so downstream callers always see a stable location.
    review_src = local_root / paths.category.review_filename
    meta_src = local_root / paths.category.meta_filename
    if not review_src.exists() or not meta_src.exists():
        raise RuntimeError(
            f"HF snapshot for {_HF_REPO_ID} is missing expected files "
            f"review={review_src} meta={meta_src}"
        )
    paths.review_path.parent.mkdir(parents=True, exist_ok=True)
    paths.meta_path.parent.mkdir(parents=True, exist_ok=True)
    # Symlink rather than copy — HF snapshot files can be multi-GB and
    # there's no point duplicating them. If the symlink already points
    # somewhere stale (e.g. a different cache), replace it.
    for src, dst in [(review_src, paths.review_path), (meta_src, paths.meta_path)]:
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())

    if not _is_present(paths):
        raise RuntimeError(
            f"Amazon-Reviews-2023[{category}] acquisition completed but "
            f"expected files are missing under {paths.extract_dir}"
        )
    return paths


_REVIEW_COLUMNS = ["user_id", "parent_asin", "rating", "timestamp"]
_META_COLUMNS = ["parent_asin", "categories", "store"]


def _read_reviews(path: Path) -> pl.DataFrame:
    """Stream the reviews JSONL into a polars DataFrame, keeping only the
    columns needed downstream (``user_id``, ``parent_asin``, ``rating``,
    ``timestamp``). The raw file has ~10 columns including a nested
    ``images`` list and free-text ``title``/``text``; dropping those at
    read time keeps memory reasonable and avoids schema-inference
    surprises on the nested ``images`` field.
    """
    LOGGER.info("Amazon: reading reviews %s", path)
    lf = pl.scan_ndjson(path)
    return lf.select(_REVIEW_COLUMNS).collect()


def _read_meta(path: Path) -> pl.DataFrame:
    """Stream the metadata JSONL into a polars DataFrame, keeping only
    ``parent_asin``, ``categories`` (list of strings), and ``store``
    (brand; often null).
    """
    LOGGER.info("Amazon: reading metadata %s", path)
    lf = pl.scan_ndjson(path)
    return lf.select(_META_COLUMNS).collect()


def _join_reviews_with_meta(
    reviews: pl.DataFrame, meta: pl.DataFrame
) -> pl.DataFrame:
    """Left-join reviews with metadata on ``parent_asin`` and rename to
    ``item_id`` so the Amazon column name never leaks downstream.
    """
    joined = reviews.join(meta, on="parent_asin", how="left")
    # Ensure the side-info columns are never null at the dataframe level;
    # the encoder handles empty lists and missing strings but downstream
    # polars casts choke on nulls for some dtypes.
    joined = joined.with_columns(
        pl.col("categories").fill_null([]),
        pl.col("store").fill_null("<unk>"),
    )
    return joined.rename({"parent_asin": "item_id"})


def load(
    dataset_root: Path | str | None = None,
    *,
    category: str = "all_beauty",
    tables: Iterable[str] | None = None,
    auto_download: bool = True,
    max_rows: int | None = None,
) -> dict[str, pl.DataFrame]:
    """Load Amazon Reviews tables into polars DataFrames.

    Parameters
    ----------
    dataset_root:
        Directory holding the ``Amazon-Reviews-2023`` subtree. Defaults
        to :data:`DEFAULT_DATASET_ROOT`.
    category:
        One of ``"all_beauty"`` (default), ``"books"``, ``"electronics"``,
        ``"sports_and_outdoors"``.
    tables:
        Iterable containing any of ``"reviews"``, ``"meta"``. If
        ``None``, returns a joined ``"reviews"`` frame with metadata
        columns merged in (the default shape used by the Amazon
        datamodule).
    auto_download:
        If ``True`` (default) and data is missing, call :func:`download`
        first. If ``False`` and data is missing, raise ``FileNotFoundError``.
    max_rows:
        If set, sort the joined frame by ``timestamp`` ascending and
        return only the first ``max_rows`` rows. Deterministic,
        time-coherent subsample — matches how sequential-recsys
        benchmarks normally report truncated splits.

    Returns
    -------
    A dict with key ``"reviews"`` (reviews joined with metadata when
    ``tables`` is None or covers both) and/or ``"meta"`` (raw metadata).
    The ``reviews`` frame always has columns ``user_id``, ``item_id``
    (renamed from ``parent_asin``), ``rating``, ``timestamp``, plus —
    when the default join applies — ``categories`` (list[str]) and
    ``store`` (str).
    """
    paths = _resolve_paths(dataset_root, category)
    if not _is_present(paths):
        if not auto_download:
            raise FileNotFoundError(
                f"Amazon-Reviews-2023[{category}] not found at "
                f"{paths.extract_dir}; pass auto_download=True or call "
                "recsys.data.amazon.download()"
            )
        paths = download(dataset_root, category=category)

    tables_list = list(tables) if tables is not None else ["reviews", "meta"]
    unknown = sorted(set(tables_list) - {"reviews", "meta"})
    if unknown:
        raise ValueError(f"Unknown Amazon tables: {unknown}")

    out: dict[str, pl.DataFrame] = {}

    want_reviews = "reviews" in tables_list
    want_meta = "meta" in tables_list
    reviews_df = _read_reviews(paths.review_path) if want_reviews else None
    meta_df = _read_meta(paths.meta_path) if want_meta else None

    if want_reviews and want_meta:
        joined = _join_reviews_with_meta(reviews_df, meta_df)
        if max_rows is not None:
            joined = joined.sort("timestamp").head(int(max_rows))
        out["reviews"] = joined
        out["meta"] = meta_df
    elif want_reviews:
        # Reviews-only: still rename parent_asin -> item_id for consistency.
        df = reviews_df.rename({"parent_asin": "item_id"})
        if max_rows is not None:
            df = df.sort("timestamp").head(int(max_rows))
        out["reviews"] = df
    elif want_meta:
        out["meta"] = meta_df

    return out


__all__ = [
    "DEFAULT_DATASET_ROOT",
    "AmazonPaths",
    "AmazonCategorySpec",
    "download",
    "load",
]
