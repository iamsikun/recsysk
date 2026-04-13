# recsys

A framework-agnostic benchmarking harness for recommendation algorithms. Benchmarks are pinned contracts (dataset + split + eval protocol + metrics), algorithms plug in behind a narrow protocol, and every run is persisted to a parquet store so multi-seed comparisons are one command away.

## Quickstart

One-time setup:

```bash
uv sync
```

**Dataset prerequisite.** v1 benchmarks assume MovieLens 20m is already extracted to `./datasets/ml-20m/` (the directory containing `ratings.csv`, `movies.csv`, `tags.csv`, ...). v1 does not auto-download.

Run an experiment (benchmark x algorithm x seeds):

```bash
uv run recsys bench --experiment conf/experiments/deepfm_on_movielens_ctr.yaml  --seeds 1,2,3
uv run recsys bench --experiment conf/experiments/din_on_movielens_seq.yaml     --seeds 1,2,3
uv run recsys bench --experiment conf/experiments/popularity_on_movielens_ctr.yaml --seeds 1,2,3
```

View aggregated results (mean +/- std across seeds):

```bash
uv run recsys report --benchmark movielens_ctr
uv run recsys report --benchmark movielens_seq
```

List what's registered:

```bash
uv run recsys list benchmarks
uv run recsys list algorithms
```

Results land in `results/<benchmark>.parquet` (git-ignored). The default experiment YAMLs bake in a fast "smoke" trainer profile (`limit_train_batches: 4`, `limit_val_batches: 4`, `max_epochs: 1`) so the gate runs in seconds; remove those overrides for a real training run.

## Adding a new algorithm

1. **Pick a backend.**
   - Classical / non-neural: subclass `recsys.algorithms.base.Algorithm` under `src/recsys/algorithms/classical/`. Do **not** import `torch`.
   - Lightning-backed neural: subclass `recsys.algorithms.torch.TorchAlgorithm` (or use an `nn.Module` that plugs into the current runner shim) under `src/recsys/algorithms/torch/`.
2. **Declare compatibility.** Set class attributes `supported_tasks: set[TaskType]` and `required_roles: set[str]` (role names from `FeatureRole`: `user`, `item`, `context`, `sequence`, `group`, `label`).
3. **Register.** Decorate the class with `@ALGO_REGISTRY.register("my_algo")` from `recsys.utils`.
4. **Side-effect import.** Add the module to `src/recsys/algorithms/classical/__init__.py` or `src/recsys/algorithms/torch/__init__.py` so the decorator fires at import time. *This step is mandatory.* If you skip it, `recsys list algorithms` will not show your algo.
5. **Write a config.** Add `conf/algorithms/my_algo.yaml` with the constructor kwargs.
6. **Write an experiment.** Add `conf/experiments/my_algo_on_<benchmark>.yaml` that points at the benchmark and algo YAMLs and sets `seeds:` + optional `trainer:` overrides.

## Adding a new benchmark

1. **Builder.** Add a data builder under `src/recsys/data/builders/` that produces a `DatasetBundle` (train/val/test + `feature_map` + `feature_specs`). Reuse the split modules under `data/splits/` and negative samplers under `data/negatives/`; write new ones there if needed.
2. **Datamodule (optional).** If the benchmark feeds a Lightning-backed algo, wrap the builder in a `BuilderDataModule` under `src/recsys/data/datamodules/`.
3. **Benchmark class.** Create `src/recsys/benchmarks/my_bench.py` that subclasses `recsys.benchmarks.base.Benchmark`, pins its `Task` (CTR / retrieval / sequential) and its `metric_names` list, and implements `build() -> BenchmarkData` and `version()`.
4. **Register.** `@BENCHMARK_REGISTRY.register("my_bench")` and import the module from `src/recsys/benchmarks/__init__.py`.
5. **Config.** Add `conf/benchmarks/my_bench.yaml` with the data/eval blocks the benchmark class consumes.

Metrics and splits are **pinned by the benchmark class**, not by config. If you need a different metric set or a different split, that is a new benchmark, not a new config — this is what makes benchmark comparisons meaningful.

## Architecture at a glance

**`FeatureSpec` + `FeatureRole`** (`src/recsys/schemas/features.py`). Every column in a benchmark is annotated with a role (`user`, `item`, `context`, `sequence`, `group`, `label`). Builders partition columns by role and algos declare which roles they need; the runner can reject incompatible combinations before `fit` is called.

**`Algorithm`** (`src/recsys/algorithms/base.py`). A framework-agnostic protocol: `fit(train, val)`, `predict_scores(batch)`, `predict_topk(users, k, candidates)`, `save/load`. Classical baselines like `popularity` live under `algorithms/classical/` and never import torch; Lightning-backed models live under `algorithms/torch/` and reuse a common `TorchAlgorithm` wrapper.

**`Task`** (`src/recsys/tasks/base.py`). Declares the I/O contract between an algo and the evaluator: which roles are required, which prediction method is called, and how predictions turn into metric values. v1 ships `CTRTask`, `RetrievalTask`, and `SequentialTask`.

**`Benchmark`** (`src/recsys/benchmarks/base.py`). Immutable bundle of dataset + task + split + eval protocol + metric set. `build()` returns a `BenchmarkData` with train/val/test, feature map, feature specs, and candidate item list. `version()` is a hash of `(dataset, split, eval_protocol)` so result rows are traceable.

**Runtime flow.** `recsys bench` loads an experiment YAML and calls `recsys.runner.run_experiment(algo_cfg, benchmark_cfg, seed, trainer_overrides, results_dir, store)`. That function builds the benchmark, builds the algo, runs the classical one-shot fit hook (if present) and/or the Lightning `Trainer.fit`, hands the fitted algo to `benchmark.task.evaluate(...)`, and writes a `RunResult` row to `results/<benchmark>.parquet`. `recsys report` reads the same parquet and prints a mean +/- std table across seeds.

## Scope (v1 + landed v2.0 work)

- **Benchmarks:** `movielens_ctr`, `movielens_seq`, `kuairec_ctr`, `kuairand_ctr`. KuaiRec/KuaiRand loaders auto-download the archives from Zenodo to `./datasets/` on first use; subsequent runs hit the cache.
- **Algorithms:** `deepfm` (CTR), `din` (sequential, with working ranking metrics), `popularity` (classical baseline — bypasses Lightning entirely via the framework-agnostic fit path).
- **Metrics:** AUC, LogLoss for CTR; NDCG@{10,50}, Recall@{10,50}, HR@{10,50}, MRR for ranking. Ranking metrics now compute correctly for sequential dict-batch algos like DIN.
- **Infrastructure:** parquet result store keyed by `(benchmark, algo, config_hash, seed, timestamp)`, argparse CLI (`bench`, `report`, `list`), multi-seed runs as the default.

## Deferred to v2

Session / conversational / cold-start tasks, beyond-accuracy metrics (coverage, diversity, novelty, fairness), statistical significance tests, experiment-tracker hooks (MLflow / W&B), hyperparameter sweeps, a typer-based CLI, additional classical / neural baselines (item-KNN, BPR-MF, SASRec, BERT4Rec, ...), and additional benchmarks (Amazon Reviews, Yoochoose / Diginetica / RetailRocket, MovieLens 1M, Netflix Prize, Yelp, Last.fm, Criteo / Avazu). See `docs/dev.md` for the full v1 plan and v2 roadmap.
