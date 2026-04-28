"""TaobaoAd dataset loader.

Tianchi Ad Display/Click Dataset (Alibaba, ~26M impressions across 8
days). Used by InterFormer, Wukong, and BST. The dataset is gated
behind a Tianchi account login (``tianchi.aliyun.com/dataset/56``), so
v1 does **not** auto-download — the loader validates the manual-extract
layout under ``./datasets/taobao_ad/`` and raises a
:class:`FileNotFoundError` with the source URL otherwise. This mirrors
the existing MovieLens-20m / MovieLens-1m manual-extract pattern.

Expected files under ``./datasets/taobao_ad/``:

* ``raw_sample.csv``      — impression log (user, ad, timestamp, click).
* ``ad_feature.csv``      — ad metadata (cate, brand, customer, price).
* ``user_profile.csv``    — user demographics.
* ``behavior_log.csv``    — historical user-on-category interactions
  (large, optional for tabular CTR — only loaded when ``include_behaviors=True``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3] / "datasets"

_SOURCE_URL = "https://tianchi.aliyun.com/dataset/56"

_REQUIRED_FILES: tuple[str, ...] = (
    "raw_sample.csv",
    "ad_feature.csv",
    "user_profile.csv",
)
_OPTIONAL_FILES: tuple[str, ...] = ("behavior_log.csv",)


@dataclass
class TaobaoAdPaths:
    root: Path
    raw_sample: Path
    ad_feature: Path
    user_profile: Path
    behavior_log: Path


def _resolve_paths(dataset_root: Path | None) -> TaobaoAdPaths:
    base = Path(dataset_root).expanduser().resolve() if dataset_root else DEFAULT_DATASET_ROOT
    root = base / "taobao_ad"
    return TaobaoAdPaths(
        root=root,
        raw_sample=root / "raw_sample.csv",
        ad_feature=root / "ad_feature.csv",
        user_profile=root / "user_profile.csv",
        behavior_log=root / "behavior_log.csv",
    )


def _check_layout(paths: TaobaoAdPaths, *, require_behaviors: bool = False) -> None:
    missing = [
        f for f in _REQUIRED_FILES if not (paths.root / f).exists()
    ]
    if require_behaviors and not paths.behavior_log.exists():
        missing.append("behavior_log.csv")
    if missing:
        raise FileNotFoundError(
            "TaobaoAd dataset not found. Expected the following files under "
            f"{paths.root}:\n"
            + "\n".join(f"  - {paths.root / f}" for f in (*_REQUIRED_FILES, *_OPTIONAL_FILES))
            + f"\nMissing: {missing}\n"
            f"Acquire from {_SOURCE_URL} (Tianchi login required) and "
            "extract under ./datasets/taobao_ad/. v1 does not auto-download."
        )


def load(
    dataset_root: Path | str | None = None,
    *,
    tables: Iterable[str] | None = None,
    join: bool = True,
) -> dict[str, pl.DataFrame] | pl.DataFrame:
    """Load TaobaoAd tables.

    With ``join=True`` (default), returns a single DataFrame: the
    impression log left-joined against ad and user metadata, ready for
    the standard ``CsvCtrBuilder`` path. With ``join=False``, returns a
    dict keyed by table name (``raw_sample``, ``ad_feature``,
    ``user_profile``, optionally ``behavior_log``).

    The ``clk`` column in ``raw_sample.csv`` is the binary click label
    (already 0/1, no thresholding needed).
    """
    paths = _resolve_paths(Path(dataset_root) if dataset_root else None)
    requested = list(tables) if tables is not None else list(_REQUIRED_FILES)
    require_behaviors = "behavior_log" in requested or "behavior_log.csv" in requested
    _check_layout(paths, require_behaviors=require_behaviors)

    raw_sample = pl.read_csv(paths.raw_sample)
    ad_feature = pl.read_csv(paths.ad_feature)
    user_profile = pl.read_csv(paths.user_profile)

    if not join:
        out: dict[str, pl.DataFrame] = {
            "raw_sample": raw_sample,
            "ad_feature": ad_feature,
            "user_profile": user_profile,
        }
        if require_behaviors:
            out["behavior_log"] = pl.read_csv(paths.behavior_log)
        return out

    # Joined view — column names follow the canonical Tianchi schema:
    # raw_sample: user, time_stamp, adgroup_id, pid, nonclk, clk
    # ad_feature: adgroup_id, cate_id, campaign_id, customer, brand, price
    # user_profile: userid, cms_segid, cms_group_id, final_gender_code,
    #               age_level, pvalue_level, shopping_level, occupation,
    #               new_user_class_level
    df = (
        raw_sample
        .join(ad_feature, on="adgroup_id", how="left")
        .join(user_profile.rename({"userid": "user"}), on="user", how="left")
    )
    return df


__all__ = ["TaobaoAdPaths", "load"]
