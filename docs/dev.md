# dev.md — Recsys Benchmarking Harness: v1 Plan

This document is the concrete, step-by-step plan for restructuring `recsys` into a framework-agnostic benchmarking infrastructure for recommendation algorithms. It is the living reference we iterate against. Keep it up to date as decisions change.

## Guiding principles

1. **Benchmarks are pinned contracts.** A benchmark owns its dataset, split, eval protocol, and metrics. Users cannot change NDCG@10 to NDCG@20 via config — that would make benchmarks non-comparable. New metrics/splits = new benchmark.
2. **Framework-agnostic algorithm interface.** The base `Algorithm` class does not assume PyTorch or Lightning. Classical baselines (popularity, item-KNN, BPR-MF) are first-class citizens. Lightning is a preferred backend for NN algorithms, not a requirement.
3. **Tasks define the algo/eval I/O contract.** Whether an algorithm is compatible with a benchmark is determined by the task type (CTR vs retrieval vs sequential) and by the feature roles it consumes.
4. **Reproducibility is mandatory, not optional.** Every run is persisted with config hash, seed, dataset version, and code SHA. Multi-seed runs are the default.
5. **MovieLens is the development anchor.** DeepFM and DIN are the default algorithms used to validate every step of the migration. If a change breaks the MovieLens + DeepFM + DIN path, that change is not done.

## v1 scope

- **Default dataset:** MovieLens (20m) — already wired.
- **Default algorithms:** DeepFM (CTR) and DIN (sequential). Every migration step must keep both runnable end-to-end.
- **Task types in v1:** CTR, top-K retrieval, sequential next-item.
- **Metrics in v1:**
  - CTR: AUC, LogLoss
  - Ranking/Retrieval/Sequential: NDCG@{10, 50}, Recall@{10, 50}, HR@{10, 50}, MRR
- **Storage:** Parquet files under `results/`, keyed by `(benchmark, algo, config_hash, seed, timestamp)`.
- **CLI:** Thin layer (`recsys bench`, `recsys report`, `recsys list`). Library choice deferred to v2 — see the v2 section for typer vs click.
- **Classical baseline in v1:** Popularity. It is the forcing function that validates framework-agnosticism — if popularity can't run through the harness, the Algorithm protocol is wrong.

## Explicitly out of scope for v1 (deferred to v2)

Write these down so we don't lose them:

- **Simulation-based benchmarks.** The `recsys/simulation/addict_fatigue.py` DGP is left as-is in v1. v2 will wire it as a first-class benchmark for counterfactual/online policy evaluation. This is considered important for modern recsys given the growth of generative recommendation and RL-style policies, and is a v2 priority, not a "maybe."
- **Task types:** session-based recommendation, conversational recommendation, cold-start (user-cold and item-cold) evaluation.
- **Beyond-accuracy metrics:** coverage, diversity, novelty, fairness.
- **Experiment tracking integrations:** MLflow, W&B, TensorBoard hooks.
- **Statistical significance tests** beyond per-seed mean ± std (e.g. paired bootstrap, corrected t-tests between algos).
- **CLI library choice** (see "CLI in v2" below).
- **Hyperparameter search / sweeps.** v1 runs are single-config, multi-seed. Sweeps come later.
- **Dataset auto-download.** v1 assumes datasets are already present under `./datasets`.

## CLI in v2 — typer vs click

Both are mature Python CLI libraries that let you build `recsys bench ...` style command trees. Short version:

- **click** is the older, more widely used library. You define commands via decorators and manually declare each option/argument with `@click.option(...)`. Very flexible, huge ecosystem, but verbose.
- **typer** is built on top of click by the FastAPI author. You write a normal Python function with type annotations, and typer turns the annotations into CLI options/arguments automatically. Less boilerplate, and you get free type checking on the CLI surface.

**Recommendation for v2: typer.** Because our commands map cleanly to Python functions (`bench(benchmark: str, algo: str, seeds: list[int])`, etc.), typer removes most of the boilerplate click would force on us, and the type hints double as documentation. We lose nothing — typer is a thin layer on click, so if we ever need raw click features we can drop down. For v1 we can ship a minimal `argparse`-based CLI or a tiny hand-rolled dispatcher to avoid committing before we see how commands shape up; promote to typer in v2 once the command surface is stable.

## Target layout

```
src/recsys/
  benchmarks/            # NEW — Benchmark = dataset × task × split × eval protocol × metrics
    base.py              #   Benchmark protocol + BenchmarkData dataclass
    movielens_ctr.py     #   MovieLens CTR benchmark (DeepFM target)
    movielens_seq.py     #   MovieLens sequential benchmark (DIN target)
    registry.py          #   benchmark name → builder
  tasks/                 # NEW — Task = I/O contract between Algorithm and Evaluator
    base.py              #   Task protocol, TaskType enum
    ctr.py               #   CTRTask: (user, item, ctx) → p(click)
    retrieval.py         #   RetrievalTask: user → top-k items
    sequential.py        #   SequentialTask: (user, history) → next item
  data/
    loaders/             # (existing) raw dataset readers — tables only
    schemas/             # FeatureSpec + FeatureRole (user/item/context/sequence/group/label)
    transforms/          # (existing) reusable column/sequence transforms
    splits/              # NEW — temporal, leave-one-out, user-based, session-cut
    negatives/           # NEW — random, popularity, in-batch, full-catalog, sampled-k
    builders/            # (existing) loader + transforms + split + negs → DatasetBundle
    datamodules/         # (existing) Lightning wrappers — used only by torch algos
  algorithms/            # RENAMED from models/
    base.py              #   Algorithm protocol — framework-agnostic
    torch/               #   Lightning-backed NN algorithms
      base.py            #     TorchAlgorithm: wraps a LightningModule + Trainer
      deepfm.py
      din.py
    classical/           #   Non-neural baselines
      popularity.py      #     v1 forcing-function baseline
  metrics/               # NEW — all evaluation metrics centralized here
    ctr.py               #   auc, logloss
    ranking.py           #   ndcg_at_k, recall_at_k, hr_at_k, mrr
  evaluation/            # NEW — the benchmarking system itself
    evaluator.py         #   Evaluator.run(algo, benchmark, seed) → RunResult
    protocols.py         #   eval-time candidate / negative sampling
    runner.py            #   end-to-end: build benchmark, build algo, fit, evaluate, persist
    reporting.py         #   aggregate across seeds, print comparison tables
    store.py             #   parquet result store
  cli.py                 # NEW — thin CLI entrypoint (v1: argparse or minimal dispatcher)
  schemas/               # (existing) — extended in Phase 1
  utils.py               # (existing) — Registry stays here
conf/
  benchmarks/            # one YAML per benchmark (stable, pinned)
    movielens_ctr.yaml
    movielens_seq.yaml
  algorithms/            # one YAML per algo variant (hyperparameters)
    deepfm.yaml
    din.yaml
    popularity.yaml
  experiments/           # experiment = benchmark × algo(s) × seeds × trainer settings
    deepfm_on_movielens_ctr.yaml
    din_on_movielens_seq.yaml
results/                 # parquet store, git-ignored
```

## Core abstractions

### FeatureSpec + FeatureRole

Current `FeatureSpec(name, source_name, type)` is extended with a `role` field:

```python
class FeatureRole(Enum):
    USER = "user"          # user-level attributes
    ITEM = "item"          # item-level attributes
    CONTEXT = "context"    # request-time / environmental features
    SEQUENCE = "sequence"  # historical interaction sequences
    GROUP = "group"        # TikTok-style semantic group features
    LABEL = "label"        # target

@dataclass
class FeatureSpec:
    name: str
    source_name: str
    type: FeatureType          # CATEGORICAL | NUMERIC
    role: FeatureRole          # NEW
    group_id: str | None = None  # NEW — used when role == GROUP
    vocab_size: int | None = None
    fill_value: float = 0
```

Builders partition columns by role. Algorithms declare `required_roles` and `optional_roles`; the runner checks compatibility before calling `fit`.

### Benchmark

Immutable bundle of dataset + task + split + eval protocol + metrics. **Metrics and split are pinned by the benchmark**, not by user config.

```python
class Benchmark(Protocol):
    name: str
    task: Task
    def build(self) -> BenchmarkData: ...   # cached; returns train/val/test + candidate set
    def metrics(self) -> list[Metric]: ...
    def version(self) -> str: ...           # hash of (dataset_version, split, eval_protocol)

@dataclass
class BenchmarkData:
    train: DatasetLike
    val: DatasetLike | None
    test: DatasetLike
    feature_map: dict[str, int]
    feature_specs: list[FeatureSpec]
    candidate_items: Sequence[int] | None   # used by retrieval-style tasks
    metadata: dict[str, Any]
```

### Task

A task type declares:
- which feature roles it requires (e.g. `CTRTask` requires at least USER, ITEM, LABEL),
- which algorithm method(s) the evaluator will call (`predict_scores` vs `predict_topk`),
- how predictions are scored into metric values.

```python
class TaskType(Enum):
    CTR = "ctr"
    RETRIEVAL = "retrieval"
    SEQUENTIAL = "sequential"

class Task(Protocol):
    type: TaskType
    required_roles: set[FeatureRole]
    def evaluate(self, algo: Algorithm, data: BenchmarkData,
                 metrics: list[Metric]) -> MetricResults: ...
```

### Algorithm (framework-agnostic)

```python
class Algorithm(Protocol):
    supported_tasks: set[TaskType]
    required_roles: set[FeatureRole]

    def fit(self, train: BenchmarkData, val: BenchmarkData | None) -> None: ...
    def predict_scores(self, batch: Batch) -> Tensor: ...
    def predict_topk(self, users, k: int,
                     candidates: Sequence[int] | None = None) -> TopK: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
```

- `predict_scores` is required for CTR-style tasks; `predict_topk` for retrieval.
- Classical baselines implement `fit` directly over numpy/polars. No torch dependency.
- `TorchAlgorithm` (in `algorithms/torch/base.py`) is a thin subclass that owns a `LightningModule` and a `Trainer`, and implements `fit` by calling `trainer.fit(...)`.

### Evaluator + ResultStore

```python
@dataclass
class RunResult:
    benchmark: str
    benchmark_version: str
    algo: str
    algo_config_hash: str
    seed: int
    metrics: dict[str, float]
    runtime_s: float
    timestamp: str
    code_sha: str | None
    env_fingerprint: str | None

class Evaluator:
    def run(self, algo: Algorithm, benchmark: Benchmark, seed: int) -> RunResult: ...

class ResultStore:
    path: Path  # results/<benchmark>.parquet
    def write(self, result: RunResult) -> None: ...
    def query(self, benchmark: str, algos: list[str] | None = None) -> pd.DataFrame: ...
```

## Phase plan — incremental, each phase ends with a green MovieLens+DeepFM and MovieLens+DIN run

Every phase ends with the same acceptance gate: **both `deepfm.yaml` and `din.yaml` runs must execute end-to-end on MovieLens and produce metrics.** If a phase breaks either, it is not done.

### Phase 0 — Baseline checkpoint (no code changes)

- Confirm `uv run python scripts/train.py --config conf/deepfm.yaml` runs to completion on MovieLens.
- Confirm `uv run python scripts/train.py --config conf/din.yaml` runs to completion on MovieLens.
- Record final `val_loss` values for both as the reference baseline. Any later phase that changes these values without explanation is suspect.
- Create `results/` directory, add to `.gitignore` if not already there.

### Phase 1 — FeatureRole annotation (no behavior change)

- Add `FeatureRole` enum to `schemas/features.py`.
- Add `role` (required) and `group_id` (optional) fields to `FeatureSpec`.
- Update `schemas/builder.py::build_feature_specs` to parse `role` from config dicts; default to `USER` for existing MovieLens user-id, `ITEM` for item-id. Keep backwards-compatible parsing so old YAMLs still load.
- Update `conf/deepfm.yaml` and `conf/din.yaml` to specify roles explicitly.
- **Gate:** both configs still train to the same `val_loss` as Phase 0.

### Phase 2 — Metrics module + evaluator skeleton

- Create `metrics/ctr.py` with `auc` and `logloss` (numpy/sklearn).
- Create `metrics/ranking.py` with `ndcg_at_k`, `recall_at_k`, `hr_at_k`, `mrr`.
- Create `evaluation/evaluator.py` with a minimal `Evaluator.run` that only handles CTR tasks for now.
- Wire the evaluator into the existing `runner.train` so after `trainer.fit` it runs `evaluator` on the val set and logs AUC/LogLoss alongside `val_loss`. Do not persist results yet.
- **Gate:** both `deepfm.yaml` and `din.yaml` runs now also report AUC + LogLoss at the end of training.

### Phase 3 — Algorithm protocol + TorchAlgorithm wrapper

- Create `algorithms/base.py` with the `Algorithm` protocol and `TaskType` enum.
- Create `algorithms/torch/base.py` with `TorchAlgorithm`: holds a `LightningModule` and a `Trainer` config, implements `fit` by delegating to `trainer.fit`.
- Move `models/deepfm.py` → `algorithms/torch/deepfm.py`, wrapped as a `TorchAlgorithm` subclass.
- Move `models/din.py` → `algorithms/torch/din.py`, wrapped as a `TorchAlgorithm` subclass.
- Leave `recsys/models/` as a compatibility shim that re-exports from `algorithms/torch/`. Delete in Phase 7.
- **Gate:** both configs still train to the same metrics as Phase 2.

### Phase 4 — Classical baseline (forcing function)

- Implement `algorithms/classical/popularity.py`. No torch. Fits on the train split by counting item interactions; `predict_scores` returns popularity; `predict_topk` returns globally most popular items.
- Register it in `MODEL_REGISTRY` (or a new `ALGO_REGISTRY` — decide in this phase, see Open questions).
- Add `conf/algorithms/popularity.yaml`.
- Run popularity end-to-end through the existing `runner.train` path on MovieLens CTR. If anything in the runner assumes `nn.Module` or Lightning, fix the runner, not the baseline.
- **Gate:** popularity runs and produces AUC/LogLoss; DeepFM and DIN still run unchanged.

### Phase 5 — Task module + Benchmark module

- Create `tasks/base.py`, `tasks/ctr.py`, `tasks/retrieval.py`, `tasks/sequential.py`.
- Create `benchmarks/base.py` with `Benchmark` protocol and `BenchmarkData`.
- Implement `benchmarks/movielens_ctr.py` wrapping the existing MovieLens tabular pipeline. Pin its split, eval protocol (full-catalog sampled negatives for retrieval metrics — decide in this phase), and metrics (AUC, LogLoss, plus NDCG@10/50, Recall@10/50, HR@10/50, MRR).
- Implement `benchmarks/movielens_seq.py` wrapping the existing MovieLens sequence pipeline.
- Add `conf/benchmarks/movielens_ctr.yaml` and `conf/benchmarks/movielens_seq.yaml` as the pinned benchmark specs.
- Migrate `runner.train` to take `(algo, benchmark)` instead of a flat config. Old configs still work via a compatibility shim.
- **Gate:** DeepFM on `movielens_ctr` benchmark, DIN on `movielens_seq` benchmark, popularity on `movielens_ctr` — all three run end-to-end and produce the full v1 metric set.

### Phase 6 — Splits and negatives as modules

- Extract split logic out of builders into `data/splits/`:
  - `temporal_global.py` (cut at a timestamp)
  - `leave_last_out.py` (per-user last interaction = test)
  - `user_based.py` (random partition of users)
- Extract negative sampling into `data/negatives/`:
  - `random.py`, `popularity.py`, `full_catalog.py`, `sampled_k.py`, `in_batch.py`
- Refactor `movielens_ctr.py` and `movielens_seq.py` benchmarks to declare their split and negatives via these modules.
- **Gate:** same as Phase 5, metrics unchanged.

### Phase 7 — Result store + CLI + reporting

- Implement `evaluation/store.py` with parquet writer/reader, keyed by `(benchmark, algo, config_hash, seed, timestamp)`.
- Config hash = stable hash of the algo config dict (sorted, JSON-serialized). Benchmark version = hash of `(dataset_version, split_spec, eval_protocol)`.
- Capture `code_sha` via `git rev-parse HEAD` at run time (best-effort, allow dirty).
- Implement `evaluation/reporting.py`: given a benchmark, query all runs and print a table of algo × metrics with mean ± std across seeds.
- Implement `cli.py` with three subcommands (v1 can use argparse to avoid committing to typer/click prematurely):
  - `recsys bench --experiment conf/experiments/<name>.yaml` — run one experiment (benchmark × algo × seeds).
  - `recsys report --benchmark <name>` — print comparison table.
  - `recsys list {benchmarks,algorithms}` — list registered names.
- Add `conf/experiments/deepfm_on_movielens_ctr.yaml`, `conf/experiments/din_on_movielens_seq.yaml`, `conf/experiments/popularity_on_movielens_ctr.yaml`.
- Update `scripts/train.py` to delegate to `cli.py` (or deprecate in favor of `recsys` entry point — decide in this phase).
- Wire `recsys` as a `[project.scripts]` entry in `pyproject.toml`.
- Delete the `recsys/models/` compatibility shim from Phase 3.
- **Gate:** `uv run recsys bench --experiment conf/experiments/deepfm_on_movielens_ctr.yaml --seeds 1,2,3` runs three DeepFM seeds, writes to `results/movielens_ctr.parquet`, and `recsys report --benchmark movielens_ctr` prints a table with mean ± std. Same for DIN and popularity.

### Phase 8 — Polish, docs, README

- Update `README.md` with the new "add a benchmark / add an algorithm" workflow.
- Update `CLAUDE.md` with the new architecture.
- Add a short `docs/benchmarks.md` explaining how each v1 benchmark is constructed and why its metrics/split were chosen.
- Add a short `docs/algorithms.md` explaining the Algorithm protocol contract and how to add a new algorithm.
- Verify the full acceptance criteria below.

## v1 acceptance criteria

A user, starting from a clean clone with MovieLens already under `./datasets`, can:

1. Run `uv run recsys list benchmarks` and see `movielens_ctr` and `movielens_seq`.
2. Run `uv run recsys list algorithms` and see `deepfm`, `din`, `popularity`.
3. Run `uv run recsys bench --experiment conf/experiments/deepfm_on_movielens_ctr.yaml --seeds 1,2,3` and see three seeds complete, results written to parquet.
4. Run `uv run recsys bench --experiment conf/experiments/din_on_movielens_seq.yaml --seeds 1,2,3` and see the same for DIN.
5. Run `uv run recsys bench --experiment conf/experiments/popularity_on_movielens_ctr.yaml --seeds 1,2,3` — popularity is deterministic so the "seeds" here are no-ops but the run still persists 3 rows for protocol consistency.
6. Run `uv run recsys report --benchmark movielens_ctr` and see a table with DeepFM and popularity rows, each showing mean ± std of AUC, LogLoss, NDCG@10/50, Recall@10/50, HR@10/50, MRR.
7. Implement their own algorithm by subclassing `Algorithm` (or `TorchAlgorithm`), register it, add an experiment YAML, and run it through the same CLI without touching any evaluation code.

## Open questions to resolve during the phases (not blocking v1 start)

- **One registry or split registries?** Current code has `MODEL_REGISTRY`, `DATASET_REGISTRY`, etc. in `utils.py`. For the new layout, we likely want `ALGO_REGISTRY`, `BENCHMARK_REGISTRY`, `TASK_REGISTRY`. Resolve in Phase 4 when we introduce the first non-nn.Module algorithm.
- **Eval protocol for CTR retrieval metrics on MovieLens.** CTR-style benchmarks need a decision on how to compute ranking metrics: full-catalog ranking (expensive, unbiased), or sampled-1000 negatives (cheap, known to be biased per Krichene & Rendle 2020). Default to full-catalog for MovieLens 20m since it's feasible; revisit for larger datasets. Resolve in Phase 5.
- **Does `CTRTask` require a validation split, or can it share val and test?** Current code uses `train_split=0.8` and no separate test. v1 needs an explicit train/val/test three-way split. Resolve in Phase 6.
- **Deterministic ordering for parquet keys.** Need to decide on a canonical timestamp format (ISO 8601 UTC) and a config-hash algorithm (probably `hashlib.sha256(json.dumps(cfg, sort_keys=True))[:12]`). Resolve in Phase 7.
- **CLI library (typer vs click) and migration from argparse.** See the "CLI in v2" section — default v2 choice is typer. Resolve at the start of v2.

## v2 roadmap (write-down, not commitment)

- **Simulation-based benchmarks.** Wire `recsys/simulation/addict_fatigue.py` (and future DGPs) as first-class benchmarks supporting online interaction, counterfactual evaluation, and policy-style algorithms. Task API will need extension: current `Task.evaluate` assumes a static dataset; simulator benchmarks need an environment-loop variant. This is a v2 priority, not a stretch goal — generative and RL-style recommenders increasingly need offline-simulator evaluation.
- **Additional task types:** session-based recommendation, conversational recommendation, cold-start (user-cold and item-cold) evaluation.
- **Beyond-accuracy metrics:** coverage, diversity, novelty, fairness.
- **Statistical significance testing** between algos on the same benchmark (paired bootstrap, corrected t-tests).
- **Experiment tracking hooks:** MLflow / W&B / TensorBoard.
- **Typer-based CLI.** Migrate from the v1 argparse/minimal dispatcher to typer (see "CLI in v2" section for rationale).
- **Hyperparameter search / sweeps.** Integrate with Optuna or Ray Tune. The result store is already keyed by config hash, so sweeps slot in naturally.
- **Dataset auto-download + version pinning.** Hash-addressed preprocessed artifacts cached under `~/.cache/recsys/` so re-runs don't repeat the 20M-row preprocessing.
- **More classical baselines:** item-KNN, BPR-MF, ALS.
- **More neural baselines:** SASRec, BERT4Rec, NeuMF.
- **KuaiRec / KuaiRand benchmarks.** The loader stubs exist; promote them to full benchmarks once the v1 abstractions are proven on MovieLens.
