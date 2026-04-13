# CLAUDE.md

Guidance for Claude Code working in this repository. The project is a framework-agnostic recsys benchmarking harness — a `recsys` CLI runs pinned benchmarks against pluggable algorithms and persists results to parquet. See `docs/dev.md` for the full v1 plan and `README.md` for the user-facing walkthrough.

## Commands

- **Install / sync deps:** `uv sync`. Re-run after any `pyproject.toml` edit. Always use `uv run ...`; never call system python.
- **List what's registered:** `uv run recsys list benchmarks`, `uv run recsys list algorithms`.
- **Run an experiment:** `uv run recsys bench --experiment conf/experiments/<name>.yaml --seeds 1,2,3 --results-dir results`.
- **Report aggregated metrics:** `uv run recsys report --benchmark movielens_ctr` (reads `results/movielens_ctr.parquet`).
- **Dataset prerequisite:** MovieLens 20m must be extracted to `./datasets/ml-20m/`. v1 does not auto-download.
- **No tests.** `pytest` is installed as a dev dep but there is no project test suite under the repo. Don't claim test runs. If you add tests, keep them outside `src/` and run with `uv run pytest`.
- **Torch wheels.** `pyproject.toml` pins a Windows-only CUDA index (`pytorch-cu130`). On macOS/Linux `uv` resolves torch from the default index — do not add that index marker to new platforms.

## Architecture

The harness is structured around four core abstractions plus a single runner entrypoint.

### Registries (`src/recsys/utils.py`)

Six `Registry` instances drive name-to-class dispatch:

- `ALGO_REGISTRY` — every algorithm (classical + torch).
- `BENCHMARK_REGISTRY` — every benchmark class.
- `TASK_REGISTRY` — CTR / retrieval / sequential task types.
- `DATASET_REGISTRY` — datamodule factories used by benchmarks whose builder needs a Lightning datamodule.
- `OPTIMIZER_REGISTRY`, `LOSS_REGISTRY` — torch optimizers + loss fns for the Lightning task wrapper.

`Registry.build(cfg, **kwargs)` pops `name` from the config dict and constructs the class with the remaining keys plus any injected runtime kwargs (e.g. `feature_map`).

### Runner (`src/recsys/runner.py::run_experiment`)

Single pipeline used by the CLI and any higher-level tooling. Given `(algo_cfg, benchmark_cfg, seed, trainer_overrides, results_dir, store)` it:

1. Builds the benchmark via `BENCHMARK_REGISTRY` and calls `benchmark.build()` to get a `BenchmarkData`.
2. Builds the algo via `ALGO_REGISTRY.build(algo_cfg, feature_map=data.feature_map)`.
3. Classical one-shot fit: if the algo exposes a `fit_on_train_counts(train)` hook, it is called once before any Lightning work.
4. Wraps the algo in `recsys.engine.CTRTask` (a LightningModule) and calls `L.Trainer(**cfg).fit(...)`. For classical algos the caller sets `max_epochs: 0` so the trainer run is a no-op.
5. Evaluates via `benchmark.task.evaluate(algo=lightning_task, benchmark_data=data, metric_names=benchmark.metric_names, ...)`.
6. Writes a `RunResult(benchmark, benchmark_version, algo, algo_config_hash, seed, metrics, runtime_s, timestamp, code_sha, env_fingerprint)` row to `results/<benchmark>.parquet`.

### Algorithm protocol (`src/recsys/algorithms/base.py`)

Framework-agnostic. The module does not import torch. Subclasses declare `supported_tasks: set[TaskType]` and `required_roles: set[str]` and implement `fit`; `predict_scores` / `predict_topk` / `save` / `load` default to `NotImplementedError` so algos only implement the surface their task needs.

- **Classical** algos live under `src/recsys/algorithms/classical/` (e.g. `popularity.py`). They must not import torch at module scope.
- **Torch** algos live under `src/recsys/algorithms/torch/` and share a `TorchAlgorithm` base (`torch/base.py`). `deepfm.py` and `din.py` both use it.
- **Known limitation (Wave 2 workaround).** The runner still routes everything through a Lightning `CTRTask` wrapper, so popularity currently carries an `nn.Module` shim even though it is classical. The true framework-agnostic fit path — where classical algos bypass Lightning entirely — is a known follow-up and is tracked in `docs/dev.md`.

### Benchmark protocol (`src/recsys/benchmarks/base.py`)

Each benchmark pins its own task, split, eval protocol, and metric set. `metric_names` is a class attribute on the benchmark — **not a config key**. Config-tweakable metrics or splits would make benchmark comparisons meaningless. A user who wants different metrics or a different split should create a new benchmark class, not edit a YAML.

v1 ships `movielens_ctr.py` (CTR task, `CTRTask` eval, AUC/LogLoss + ranking metrics) and `movielens_seq.py` (sequential task, DIN target).

### Feature roles (`src/recsys/schemas/features.py`)

`FeatureRole` is `{user, item, context, sequence, group, label}`. Every `FeatureSpec` carries a role; builders partition columns by role; algos declare `required_roles` so the runner can reject incompatible combinations.

### Result store (`src/recsys/evaluation/store.py`)

Parquet, one file per benchmark under `results/`. Rows are keyed by `(benchmark, algo, config_hash, seed, timestamp)`. `config_hash` is `hashlib.sha256(json.dumps(cfg, sort_keys=True))[:12]` via `recsys.utils.config_hash`. `code_sha` is a best-effort `git rev-parse HEAD` captured at run time. `recsys.evaluation.reporting.summary_table(store, benchmark)` produces the mean +/- std table used by `recsys report`.

### Known metric limitation

Ranking metrics (NDCG, Recall, HR, MRR) are NaN for dict-batch (sequential) algos like DIN — the v1 ranking evaluator walks tabular batches and does not yet reconstruct per-user candidate lists for sequence inputs. CTR metrics (AUC, LogLoss) work for **all** algos, including DIN. Don't be surprised by NaN ranking rows in `results/movielens_seq.parquet`; that is expected in v1.

## Side-effect registration gotcha

This is the single biggest trap when adding an algo or benchmark: `@ALGO_REGISTRY.register(...)` and `@BENCHMARK_REGISTRY.register(...)` only fire when the defining module is actually imported. After creating a new module you **must** add an import for it in the nearest package `__init__.py`:

- New classical algo -> `src/recsys/algorithms/classical/__init__.py`
- New torch algo -> `src/recsys/algorithms/torch/__init__.py`
- New benchmark -> `src/recsys/benchmarks/__init__.py`
- New datamodule / builder -> `src/recsys/data/__init__.py` (via the appropriate subpackage)

To sanity-check coverage, `grep -r "@ALGO_REGISTRY" src/` and `grep -r "@BENCHMARK_REGISTRY" src/` and confirm each hit is reachable from its package `__init__.py`. `uv run recsys list algorithms` / `uv run recsys list benchmarks` is the ground-truth check.

## Development gate convention

Experiment YAMLs under `conf/experiments/` bake in a "smoke" trainer profile so the acceptance gate runs in seconds, not the half-hour a real MovieLens 20m run would take. The pattern is:

```yaml
trainer:
  accelerator: auto
  devices: 1
  max_epochs: 1
  limit_train_batches: 4
  limit_val_batches: 4
  enable_progress_bar: false
  enable_checkpointing: false
```

These come in via the `trainer_overrides` arg to `run_experiment`. When iterating on an algorithm or benchmark, keep these in place; remove them only for a real training run.

## Darwin-specific notes

- `accelerator: auto` resolves to MPS on macOS (no CUDA). Don't hard-code `gpu`.
- `num_workers: 0` in the data config is required on macOS inside the sandboxed harness — higher values trigger a `torch_shm_manager` permission error. The `_resolve_num_workers` helper in `runner.py` accepts `auto` and maps it to `max(1, cpu_count - 1)`, but benchmark YAMLs used in the gate set `num_workers: 0` explicitly.

## Conventions worth knowing

- `seed` at the top of an experiment/benchmark config is propagated into the data config if the data block doesn't set its own.
- `FeatureType` values in YAML may be upper- or lower-case; `build_feature_specs` normalizes them.
- The `trainer:` block in experiment YAMLs is passed through verbatim to `L.Trainer(**cfg)`, so any Lightning trainer kwarg works.
- Never edit files under `results/` by hand — it is the append-only run log and is git-ignored.
