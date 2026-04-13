# Benchmarks

## What is a benchmark

A benchmark is an immutable pinning of
`(dataset, task, split, eval protocol, metric set)`. Once a benchmark is
defined, every algorithm runs against the *same* contract: the same split
seed, the same negative-sampling strategy, the same metrics at the same
cutoffs. Users cannot tune metrics, split, or eval protocol via config at
run time — that would silently make two result rows non-comparable. If
you want NDCG@20 instead of NDCG@10, or a temporal split instead of a
random one, that is a *new* benchmark with a new name, not a flag on an
existing one. See `docs/dev.md` ("Guiding principles", point 1, and the
`Benchmark` section under "Core abstractions") for the long-form
rationale.

## The v1 benchmarks

Two benchmarks ship in v1, both sourced from MovieLens 20M with
`rating >= 4` as the positive label. Both are registered under
`recsys.benchmarks` and have a pinned config under `conf/benchmarks/`.

### `movielens_ctr`

- **Source:** `src/recsys/benchmarks/movielens_ctr.py`, config
  `conf/benchmarks/movielens_ctr.yaml`.
- **Task:** `CTRTask` — dense tabular `(user_id, item_id)` batches of shape
  `(B, n_fields)` with binary click labels.
- **Split:** random 80/20 train/val performed inside the MovieLens
  builder. `BenchmarkData.test` currently aliases `val`; a proper 80/10/10
  temporal split is a documented TODO (Wave 4+). The `RandomSplit` object
  is still stored on the benchmark (`self.splitter`) so the result store
  hashes the split config even though the builder does the actual cut.
- **Target algorithm:** DeepFM.

### `movielens_seq`

- **Source:** `src/recsys/benchmarks/movielens_seq.py`, config
  `conf/benchmarks/movielens_seq.yaml`.
- **Task:** `CTRTask` again, but the datamodule is built with
  `model_input: sequence` so batches are dicts:
  `{item_id, hist_item_id, hist_item_id_mask, sparse_features, ...}`.
- **Split:** same random 80/20 as above, same `test = val` aliasing. The
  `MovieLensSeqBuilder` also runs its own train-time negative sampling
  inline — Wave 5+ will move this behind the `negative_sampler` module.
- **Target algorithm:** DIN.

### Eval protocol (both benchmarks)

Ranking metrics use **sampled-100 negatives per positive**, drawn
uniformly at random from the item catalog with a fixed seed (default
`42`, configurable via `eval.seed` / `eval.n_negatives` in the benchmark
YAML). The sampler is
`recsys.data.negatives.random_uniform.RandomUniform`, stored on the
benchmark as `self.negative_sampler` and threaded into
`CTREvaluator.evaluate_full` via
`BenchmarkData.metadata["negative_sampler"]`.

### Pinned metric set

Both benchmarks pin the same list:

```
auc, logloss, ndcg@10, ndcg@50, recall@10, recall@50, hr@10, hr@50, mrr
```

AUC and LogLoss are point-prediction metrics computed over the full val
set. The ranking metrics are sampled-100 metrics — each positive is
ranked against 100 uniform negatives and the resulting rank is scored
into NDCG / Recall / HR / MRR at the respective cutoffs.

### Known limitation: sequential ranking is NaN

Sequential-input algorithms (DIN) currently return `NaN` for the ranking
slice of the metric dict on `movielens_seq`. The reason is mechanical:
`CTREvaluator.evaluate_full` cannot synthesize dict-shaped batches with
fabricated history columns for the sampled negatives, so the ranking
pass is skipped. AUC and LogLoss are still produced for DIN. This is a
documented follow-up and does not affect `movielens_ctr` / DeepFM /
popularity.

## How to add a new benchmark

1. **Data plumbing (only if needed).** If the raw dataset isn't already
   loadable, add a loader under `src/recsys/data/loaders/`, a builder
   under `src/recsys/data/builders/`, and a Lightning-compatible
   datamodule under `src/recsys/data/datamodules/`. Register the
   datamodule with `DATASET_REGISTRY`.
2. **Benchmark class.** Create `src/recsys/benchmarks/<my_benchmark>.py`
   with a subclass of `recsys.benchmarks.base.Benchmark`. Pin `name`,
   `task`, and `metric_names` as class attributes. In `__init__`, accept
   `data_cfg` and `eval_cfg` dicts and explicitly construct
   `self.splitter` and `self.negative_sampler` from
   `recsys.data.splits` / `recsys.data.negatives` so those choices are
   recorded in metadata and hashed by `version()`. In `build()`,
   instantiate the datamodule via `DATASET_REGISTRY.build(data_cfg)`,
   call `dm.setup(stage="fit")`, and return a `BenchmarkData` populated
   with `train`, `val`, `test`, `feature_map`, `feature_specs`,
   `datamodule`, and a `metadata` dict carrying at least `eval`,
   `splitter`, and `negative_sampler`.
3. **Register.** Decorate the class with
   `@BENCHMARK_REGISTRY.register("my_benchmark")`.
4. **Wire the import.** Import the module from
   `src/recsys/benchmarks/__init__.py` so the decorator fires on package
   import.
5. **Benchmark YAML.** Create `conf/benchmarks/my_benchmark.yaml` with
   the `data:` block (dataset kwargs, feature specs) and the `eval:`
   block (`n_negatives`, `seed`, optional `max_users`). No algo config
   in here — benchmarks are algo-agnostic.
6. **Experiment YAML.** Create
   `conf/experiments/<algo>_on_<benchmark>.yaml` that points
   `benchmark:` at the new benchmark YAML, `algo:` at an algorithm YAML,
   and sets `seeds:` and `trainer:` overrides.

## Version hash

`Benchmark.version()` returns a stable 8-character SHA1 prefix of
`(name, metric_names, eval_cfg)`. The result store uses this hash to
disambiguate benchmark revisions stored under the same `name`. Changing
the metric list or the eval protocol on an existing benchmark will bump
the hash — old results written under the previous hash are still
readable but are not directly comparable across the revision boundary.
The convention is: if you change metrics or the eval protocol, bump the
benchmark's `name` (e.g. `movielens_ctr_v2`) rather than silently
breaking continuity of the old hash.
