# Supported datasets

The harness ships with loaders and Lightning datamodules for four datasets. Each is wired through [DATASET_REGISTRY](../src/recsys/utils.py) so benchmark YAMLs can reference it by name; see [CLAUDE.md](../CLAUDE.md) for the registration conventions that make this work.

| Dataset | Registry key | Default variant / category | Acquisition | Pinned benchmarks |
|---|---|---|---|---|
| MovieLens 20m | `movielens` | `20m` | Local (manual extract to `./datasets/ml-20m/`) | `movielens_ctr`, `movielens_seq` |
| KuaiRec | `kuairec` | `small_matrix` | Zenodo auto-download | `kuairec_ctr` |
| KuaiRand | `kuairand` | `pure` | Zenodo auto-download | `kuairand_ctr` |
| Amazon Reviews 2023 | `amazon` | `all_beauty` | Hugging Face Hub auto-download | `amazon_ctr`, `amazon_seq` |

All loaders default their on-disk cache to the repo-root `./datasets/` directory via `Path(__file__).resolve().parents[3] / "datasets"`, so calls succeed regardless of CWD. The two acquisition helpers — [`http_download_atomic`](../src/recsys/data/_download.py) (Zenodo) and [`fetch_hf_dataset`](../src/recsys/data/_download.py) (HF Hub) — handle caching, partial-download protection, and integrity checks.

## MovieLens 20m

- Homepage: https://grouplens.org/datasets/movielens/20m/
- Loader: [src/recsys/data/loaders/movielens.py](../src/recsys/data/loaders/movielens.py)
- Datamodule: [src/recsys/data/datamodules/movielens.py](../src/recsys/data/datamodules/movielens.py)
- Label: `rating >= 4` (binary, MovieLens' classical threshold).
- Variants: the loader accepts a `dataset_size` field (`"20m"` is the only one wired up in v1).
- Acquisition: **manual**. v1 does not auto-download MovieLens. Extract the `ml-20m.zip` archive into `./datasets/ml-20m/` yourself before running the MovieLens benchmarks.

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
