# Supported datasets

The harness ships with loaders and Lightning datamodules for four datasets. Each is wired through [DATASET_REGISTRY](../src/recsys/utils.py) so benchmark YAMLs can reference it by name; see [CLAUDE.md](../CLAUDE.md) for the registration conventions that make this work.

| Dataset | Registry key | Default variant / category | Acquisition | Pinned benchmarks |
|---|---|---|---|---|
| MovieLens 20m | `movielens` | `20m` | Local (manual extract to `./datasets/ml-20m/`) | `movielens_ctr`, `movielens_seq` |
| MovieLens 1m | `movielens` | `1m` | Local (manual extract to `./datasets/ml-1m/`) | `movielens_ctr` (via the `movielens_1m_ctr.yaml` config) |
| KuaiRec | `kuairec` | `small_matrix` | Zenodo auto-download | `kuairec_ctr` |
| KuaiRand | `kuairand` | `pure` | Zenodo auto-download | `kuairand_ctr` |
| Amazon Reviews 2023 | `amazon` | `all_beauty` | Hugging Face Hub auto-download | `amazon_ctr`, `amazon_seq` |
| Frappe (NFM mirror) | `frappe` | _(single)_ | GitHub raw auto-download | `frappe_ctr` |
| TaobaoAd (Tianchi) | `taobao_ad` | _(single)_ | Local (manual extract to `./datasets/taobao_ad/`) | `taobao_ad_ctr` |
| MicroVideo (THACIL) | `microvideo` | _(single)_ | Local (manual extract to `./datasets/microvideo_thacil/`) | `microvideo_ctr` |
| KuaiVideo (BARS / reczoo HF) | `kuaivideo` | _(single)_ | Hugging Face Hub auto-download | `kuaivideo_ctr` |

All loaders default their on-disk cache to the repo-root `./datasets/` directory via `Path(__file__).resolve().parents[3] / "datasets"`, so calls succeed regardless of CWD. The two acquisition helpers — [`http_download_atomic`](../src/recsys/data/_download.py) (Zenodo) and [`fetch_hf_dataset`](../src/recsys/data/_download.py) (HF Hub) — handle caching, partial-download protection, and integrity checks.

## MovieLens 20m

- Homepage: https://grouplens.org/datasets/movielens/20m/
- Loader: [src/recsys/data/loaders/movielens.py](../src/recsys/data/loaders/movielens.py)
- Datamodule: [src/recsys/data/datamodules/movielens.py](../src/recsys/data/datamodules/movielens.py)
- Label: `rating >= 4` (binary, MovieLens' classical threshold).
- Variants: the loader accepts a `dataset_size` field — `"20m"`, `"10m"`, and `"1m"` are wired in v1. See "MovieLens 1m" below for the smaller variant used by Wukong / HSTU.
- Acquisition: **manual**. v1 does not auto-download MovieLens. Extract the `ml-20m.zip` archive into `./datasets/ml-20m/` yourself before running the MovieLens benchmarks.

## MovieLens 1m

- Homepage: https://grouplens.org/datasets/movielens/1m/
- Loader: same `loaders/movielens.py` (variant config at [movielens.py:124–172](../src/recsys/data/loaders/movielens.py)).
- Datamodule: same as MovieLens 20m.
- Label: `rating >= 4` (same threshold as 20m).
- Format on disk: `::`-separated `.dat` files. Expected layout under `./datasets/ml-1m/`:
  - `ratings.dat` — `user_id::item_id::rating::timestamp`
  - `users.dat` — `user_id::gender::age::occupation::zip_code`
  - `movies.dat` — `item_id::title::genres` (encoded `latin-1`)
- Acquisition: **manual**. Download `ml-1m.zip` from https://files.grouplens.org/datasets/movielens/ml-1m.zip and extract under `./datasets/ml-1m/`. v1 does not auto-download.
- Benchmark configs: [conf/benchmarks/movielens_1m_ctr.yaml](../conf/benchmarks/movielens_1m_ctr.yaml). Run with `uv run recsys bench --experiment conf/experiments/deepfm_on_movielens_1m_ctr.yaml --seeds 1`.

## KuaiRec

- Homepage: https://kuairec.com/
- Loader: [src/recsys/data/kuairec.py](../src/recsys/data/kuairec.py)
- Datamodule: [src/recsys/data/datamodules/kuairec.py](../src/recsys/data/datamodules/kuairec.py)
- Label: `watch_ratio >= 2.0` (tunable via `watch_ratio_threshold` in the benchmark YAML).
- Variants: `small_matrix` (default; ~1.4K × 3.3K dense) or `big_matrix` (~7K × 10K).
- Acquisition: first `kuairec.load()` downloads `KuaiRec.zip` (~432 MB) from Zenodo and extracts it. The Zenodo mirror has had throttling problems (sustained ~0–22 KB/s) as of early 2026, which in practice blocks the first-run download; if you hit it, drop a pre-extracted `KuaiRec/data/` tree under `./datasets/` manually and the loader will skip the download step.

## KuaiRand

- Homepage: https://kuairand.com/
- Loader: [src/recsys/data/kuairand.py](../src/recsys/data/kuairand.py)
- Datamodule: [src/recsys/data/datamodules/kuairand.py](../src/recsys/data/datamodules/kuairand.py)
- Label: `is_click` (already binary; no threshold applied).
- Variants: `pure` (~45 MB compressed, ~1.4M interactions — the default, smoke-gate friendly), `1k` (~2 GB), `27k` (~40 GB).
- Acquisition: Zenodo, auto-downloaded on first load. `log_standard_*.csv` ships as multiple date-range shards; the loader concatenates them into a single DataFrame.

## Amazon Reviews 2023

- Homepage: https://amazon-reviews-2023.github.io/
- HF Hub mirror: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Loader: [src/recsys/data/amazon.py](../src/recsys/data/amazon.py)
- Datamodule: [src/recsys/data/datamodules/amazon.py](../src/recsys/data/datamodules/amazon.py)
- Label: `rating >= 4` (same threshold as MovieLens).
- Categories supported in the loader's `_CATEGORIES` table:

  | Key | HF name | # users | # items | # ratings | First-run download (reviews + meta) |
  |---|---|---|---|---|---|
  | `all_beauty` (default) | `All_Beauty` | 632K | 112.6K | 701.5K | ~515 MB |
  | `books` | `Books` | 10.3M | 4.4M | 29.5M | multi-GB (very slow) |
  | `electronics` | `Electronics` | 18.3M | 1.6M | 43.9M | ~27 GB (don't do this lightly) |
  | `sports_and_outdoors` | `Sports_and_Outdoors` | — | — | — | multi-GB |

- Acquisition: HF Hub via `fetch_hf_dataset`. The loader fetches exactly two files per category (`raw/review_categories/<Category>.jsonl` and `raw/meta_categories/meta_<Category>.jsonl`) via `snapshot_download`'s `allow_patterns` flag, then symlinks them into `./datasets/Amazon-Reviews-2023/...` so downstream paths are stable. `huggingface-hub` is already a hard dep in `pyproject.toml`.
- Metadata join: `amazon.load()` returns the reviews frame left-joined against per-item metadata, adding `categories` (list of ancestor→leaf strings) and `store` (brand, nullable — null is replaced with `"<unk>"`). The default datamodule features expose `store` as a `CATEGORICAL` side-feature; `categories` is **not** in the default features because the 2023 `All_Beauty` metadata has an empty `categories` field for every item. Users running on a category with populated `categories` can add it via their benchmark YAML as `MULTI_CATEGORICAL, max_len: 8`.
- Subsampling: the loader's `max_rows` parameter (and the benchmark YAML's `data.max_rows`) takes a deterministic time-ordered prefix of the joined frame. Useful for running Books / Electronics at smoke-gate size.

### Caveats for Books / Electronics

Neither `books` nor `electronics` is practical on a laptop without `max_rows`. Pick a row cap (e.g. 200K) and expect the first run to spend most of its time in the HF Hub download, not the benchmark itself. The JSONL files are not chunked on disk, so polars holds the full selected-column slice in memory during `load`; budget accordingly.

## Frappe (NFM mirror)

- Original homepage: https://www.baltrunas.info/context-aware/frappe (Baltrunas et al. 2015)
- NFM mirror: https://github.com/hexiangnan/neural_factorization_machine/tree/master/data/frappe
- Loader: [src/recsys/data/frappe.py](../src/recsys/data/frappe.py)
- Datamodule: [src/recsys/data/datamodules/frappe.py](../src/recsys/data/datamodules/frappe.py)
- Label: implicit click vs. NFM-sampled negative; libfm files use `1` / `-1`, the loader normalises to `1.0` / `0.0`.
- Variants: only the NFM-prepared mirror (~288k rows across train + validation + test).
- Acquisition: auto-download on first call from the NFM repo's raw GitHub URLs (~12 MB total). The combined frame is fed into the standard `CsvCtrBuilder`, which runs its own random train/val split — the NFM-supplied splits are concatenated rather than reused.
- Fields: 10 categorical features in pinned order — `user_id, item_id, daytime, weekday, isweekend, homework, cost, weather, country, city`.
- Used by: Wukong (one of its six small CTR datasets).

## TaobaoAd (Tianchi Ad Display Dataset)

- Homepage: https://tianchi.aliyun.com/dataset/56
- Loader: [src/recsys/data/taobao_ad.py](../src/recsys/data/taobao_ad.py)
- Datamodule: [src/recsys/data/datamodules/taobao_ad.py](../src/recsys/data/datamodules/taobao_ad.py)
- Label: `clk` (already binary 0/1, no thresholding).
- Acquisition: **manual**. Tianchi requires an Aliyun login to download; v1 does not auto-download. Extract the four CSVs under `./datasets/taobao_ad/`:
  - `raw_sample.csv` — 26M impression rows (`user, time_stamp, adgroup_id, pid, nonclk, clk`)
  - `ad_feature.csv` — ad metadata (`adgroup_id, cate_id, campaign_id, customer, brand, price`)
  - `user_profile.csv` — user demographics (`userid, cms_segid, cms_group_id, final_gender_code, age_level, pvalue_level, shopping_level, occupation, new_user_class_level`)
  - `behavior_log.csv` — historical user-on-category interactions (large; only loaded when explicitly requested)
- Joined view: `taobao_ad.load(join=True)` returns the impression log left-joined against ad and user metadata, ready for the standard `CsvCtrBuilder` path.
- Memory note: the joined frame is materialised in memory by `CsvCtrBuilder` before splitting (~26M rows; budget several GB). Use the `max_rows` knob added in PR8 if you need to fit a smaller machine.
- Used by: InterFormer, Wukong, BST.

## MicroVideo (THACIL)

- Source repo: https://github.com/Ocxs/THACIL (Chen et al. 2018, ACM MM)
- Loader: [src/recsys/data/microvideo.py](../src/recsys/data/microvideo.py)
- Datamodule: [src/recsys/data/datamodules/microvideo.py](../src/recsys/data/datamodules/microvideo.py)
- Label: implicit click vs. sampled negative (0/1).
- Stats: 10,986 users / 1,704,880 items / 12,737,619 interactions.
- Acquisition: **manual**. The THACIL release page is password-protected (the GitHub README lists the password as `ms7x`); v1 does not auto-download.
  - The original release ships as pickled behaviour sequences + pre-extracted multimodal embeddings.
  - For v1, convert that release to a tabular `interactions.csv` with columns `[user_id, item_id, label]` and place it at `./datasets/microvideo_thacil/interactions.csv`.
- Used by: Wukong (one of its six public CTR datasets — distinct from InterFormer's KuaiVideo, which is the ALPINE 2019 release).

## KuaiVideo (BARS / reczoo HF mirror)

- HF Hub repo: https://huggingface.co/datasets/reczoo/KuaiVideo_x1
- Original paper: Li et al. 2019, "Routing Micro-videos via A Temporal Graph-guided Recommendation System" (the ALPINE paper, ACM MM '19).
- Loader: [src/recsys/data/kuaivideo.py](../src/recsys/data/kuaivideo.py)
- Datamodule: [src/recsys/data/datamodules/kuaivideo.py](../src/recsys/data/datamodules/kuaivideo.py)
- Label: `is_click` (binary, no thresholding).
- Stats: 10,000 users / 3,239,534 items / 13,661,383 interactions.
- Acquisition: HF Hub via `fetch_hf_dataset`. The loader pulls `*.csv` only (drops the optional 2048-d visual embedding for v1) and symlinks the BARS train/valid/test CSVs into `./datasets/KuaiVideo_x1/`. The harness's `CsvCtrBuilder` then concatenates the three CSVs and runs its own random train/val split — the BARS-supplied splits are not preserved.
- Memory note: ~2.27 GB CSV; budget ~5–6 GB peak after polars load. Use `data.max_rows: 1_000_000` (added in PR8) for laptop-friendly smoke runs.
- Used by: InterFormer.
