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

- **Task types:** session-based recommendation, conversational recommendation, cold-start (user-cold and item-cold) evaluation.
- **Beyond-accuracy metrics:** coverage, diversity, novelty, fairness.
- **Experiment tracking integrations:** MLflow, W&B, TensorBoard hooks.
- **Statistical significance tests** beyond per-seed mean ± std (e.g. paired bootstrap, corrected t-tests between algos).
- **CLI library choice** (see "CLI in v2" below).
- **Hyperparameter search / sweeps.** v1 runs are single-config, multi-seed. Sweeps come later.
- **Dataset auto-download.** v1 assumes datasets are already present under `./datasets`. *Partially landed in v2.0:* the KuaiRec and KuaiRand loaders (`src/recsys/data/{kuairec,kuairand}.py`) auto-download from Zenodo on first use and cache under `./datasets/`. MovieLens still has to be extracted manually.

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

*v2.0 extends this list.* `recsys list benchmarks` now also prints `kuairec_ctr` and `kuairand_ctr`; the corresponding `popularity_on_kuairec_ctr` / `popularity_on_kuairand_ctr` experiments exist and write to their own parquet files. DIN on `movielens_seq` now reports real NDCG/Recall/HR/MRR instead of NaN; the classical bypass means popularity skips the Lightning wrapper entirely.

## Open questions to resolve during the phases (not blocking v1 start)

- **One registry or split registries?** Current code has `MODEL_REGISTRY`, `DATASET_REGISTRY`, etc. in `utils.py`. For the new layout, we likely want `ALGO_REGISTRY`, `BENCHMARK_REGISTRY`, `TASK_REGISTRY`. Resolve in Phase 4 when we introduce the first non-nn.Module algorithm.
- **Eval protocol for CTR retrieval metrics on MovieLens.** CTR-style benchmarks need a decision on how to compute ranking metrics: full-catalog ranking (expensive, unbiased), or sampled-1000 negatives (cheap, known to be biased per Krichene & Rendle 2020). Default to full-catalog for MovieLens 20m since it's feasible; revisit for larger datasets. Resolve in Phase 5.
- **Does `CTRTask` require a validation split, or can it share val and test?** Current code uses `train_split=0.8` and no separate test. v1 needs an explicit train/val/test three-way split. Resolve in Phase 6.
- **Deterministic ordering for parquet keys.** Need to decide on a canonical timestamp format (ISO 8601 UTC) and a config-hash algorithm (probably `hashlib.sha256(json.dumps(cfg, sort_keys=True))[:12]`). Resolve in Phase 7.
- **CLI library (typer vs click) and migration from argparse.** See the "CLI in v2" section — default v2 choice is typer. Resolve at the start of v2.

## v2 roadmap (write-down, not commitment)

- **Additional task types:** session-based recommendation, conversational recommendation, cold-start (user-cold and item-cold) evaluation.
- **Beyond-accuracy metrics:** coverage, diversity, novelty, fairness.
- **Statistical significance testing** between algos on the same benchmark (paired bootstrap, corrected t-tests).
- **Experiment tracking hooks:** MLflow / W&B / TensorBoard.
- **Typer-based CLI.** Migrate from the v1 argparse/minimal dispatcher to typer (see "CLI in v2" section for rationale).
- **Hyperparameter search / sweeps.** Integrate with Optuna or Ray Tune. The result store is already keyed by config hash, so sweeps slot in naturally.
- **Dataset auto-download + version pinning.** Hash-addressed preprocessed artifacts cached under `~/.cache/recsys/` so re-runs don't repeat the 20M-row preprocessing.
- **More classical baselines:** item-KNN, BPR-MF, ALS.
- **More neural baselines:** SASRec, BERT4Rec, NeuMF.
- **More benchmarks.** Broaden coverage beyond MovieLens. Status:
  - **KuaiRec / KuaiRand** — *landed in v2.0.* Loaders at `src/recsys/data/{kuairec,kuairand}.py` (download-or-load-from-local, default to repo-root `datasets/`, atomic `.part` rename with Content-Length verification). Benchmarks: `kuairec_ctr` (small_matrix, `watch_ratio >= 2.0` label) and `kuairand_ctr` (KuaiRand-Pure standard logs, `is_click` label). **KuaiRand-Pure is validated end-to-end via the popularity smoke gate.** **KuaiRec data acquisition is currently deferred:** the 432 MB `KuaiRec.zip` on Zenodo is being served at ~0–22 KB/s (verified via both the loader and direct `curl --range`), making the first-run download infeasible in practice. The loader code is in place and will just work once Zenodo is less congested or the user drops a pre-extracted `KuaiRec/data/` tree under `datasets/` manually.
  - **Amazon Reviews** (Books, Beauty, Electronics, Sports, 5-core) — workhorse for sequential recommendation.
  - **Yoochoose, Diginetica, RetailRocket** — session-based sequential recommendation.
  - **MovieLens 1M** — smaller sibling of the existing MovieLens 20m benchmark, useful for fast iteration and apples-to-apples comparison with prior work.
  - **Netflix Prize** — classic rating-prediction benchmark.
  - **Yelp** — local/business recommendation.
  - **Last.fm** — music recommendation.
  - **Criteo / Avazu** — large-scale CTR-style ranking benchmarks (only if ad recommendation is in scope).
- ~~**Framework-agnostic fit path for classical algos.**~~ *Landed in v2.0.* The runner now branches on `isinstance(algo, Algorithm)`; classical algos like `popularity` skip Lightning entirely and have their `fit` called directly. `Popularity` no longer subclasses `nn.Module`.
- ~~**Ranking metrics for sequential / dict-batch algos.**~~ *Landed in v2.0.* `CTREvaluator.evaluate_full` now has a `_dict_batch_ranking` path that unwraps `torch.utils.data.Subset`, takes the first positive row per user, and scores it against sampled-K negatives by stacking the row's history/sparse/dense tensors. NDCG/Recall/HR/MRR work for DIN today.

## v2.1 — competition-style CTR primitives

Motivation: we want to run `recsys` inside downstream competition repos (TAAC2026 / Tencent KDD 2026 ads, future Kaggle drops) without baking any dataset-specific code into the harness. The only dataset-specific thing allowed here is the registry decorator — the loader, schema, benchmark class, and submission format live in the downstream repo. v2.1 adds the generic primitives every competition-style CTR dataset needs.

### New feature types

- `FeatureType.DENSE_VECTOR` — fixed-width float arrays for pretrained embedding columns. `FeatureSpec.vector_dim` carries the width; `feature_map[name]` stores the width (not a vocab size).
- `FeatureType.MULTI_CATEGORICAL` — variable-length id lists (tags, interest sets). `FeatureSpec.max_len` pads/truncates; 0 is reserved for padding. Optional `weighted=True` emits a paired `<name>_weight` Float32 array, consumed by `nn.EmbeddingBag` in `sum` mode.
- `FeatureSpec.stream: str | None` — semantic tag for sequence features, used by the multi-stream DIN path below.

### Encoder + tabular dataset shape

- `encode_features` in `data/transforms/tabular.py` handles all four types. Dense vectors pass through as `pl.Array(Float32, vector_dim)`; multi-cat goes through a Python-level vocab + pad/truncate and lands as `pl.Array(Int64, max_len)`.
- `build_tabular_dataset` auto-switches between the legacy flat-tensor `TabularDataset` (when every feature is scalar) and a new `TabularDictDataset` (when any non-scalar feature is present). Existing MovieLens / KuaiRec / KuaiRand benchmarks still hit the flat-tensor path and are bit-for-bit unchanged.

### Algo wiring

- The runner injects `feature_specs=data.feature_specs` into `ALGO_REGISTRY.build` alongside `feature_map`. Every built-in algo (`deepfm`, `din`, `popularity`) accepts the kwarg; classical algos that don't need it simply discard it.
- `DeepFM` has a dual-mode forward:
  - **Legacy / all-scalar:** `(B, n_fields)` integer tensor → single shared embedding table + FM interaction, same as pre-v2.1.
  - **Mixed:** dict batch with one tensor per feature; categoricals go through per-field `nn.Embedding`, dense vectors through `nn.Linear(vector_dim, embed_dim)`, multi-cat through `nn.EmbeddingBag`. FM second-order interaction fires over the stacked per-field embeddings; first-order term is skipped in mixed mode (no shared offset table).

### Multi-stream sequence features

- `SequenceSpec` refactored from one-item-one-history to `(target_feature, streams: tuple[SequenceStream, ...])`. A `SequenceSpec.single_stream(...)` factory preserves the pre-v2.1 MovieLens / KuaiRec layout; existing benchmark builders call it and are unchanged.
- `build_sequence_dataset` emits `hist_<history_feature>` + `hist_<history_feature>_mask` per stream. The first event per user still seeds history buffers without producing a training row.
- `DIN` gained an optional `streams: list[dict]` kwarg. When present, it builds one `nn.Embedding` + one `LocalActivationUnit` per stream and concatenates the K interest vectors into the final MLP. The target item is scored against the first stream's embedding table (the "primary" stream). Single-stream default preserves pre-v2.1 behavior exactly — the existing `din_on_movielens_seq` gate is unchanged.

### Download helpers + HF Hub fetcher

- Factored `http_download_atomic` out of `kuairec.py` / `kuairand.py` into a shared `src/recsys/data/_download.py`. Both loaders now wrap it; behavior (atomic `.part` → final rename, `Content-Length` verification, curl-style User-Agent to bypass Zenodo throttling) is identical.
- New `fetch_hf_dataset(repo_id, cache_dir, revision=None, allow_patterns=None, token=None)` helper wraps `huggingface_hub.snapshot_download`. Added `huggingface-hub>=0.24` to `[project.dependencies]`. Downstream competition loaders (e.g. TAAC2026) can call this directly without having to re-implement the atomic-rename dance.

### `Task.export_predictions` + `recsys submit` verb

- New `Task.export_predictions(algo, benchmark, benchmark_data, out_path)` abstract hook. Implemented on `CTRTask` by iterating the test dataset via a new module-level `iter_predictions(model, dataset)` helper in `evaluator.py` that handles both `(x, y)` tuple datasets and dict-valued (`SequenceDataset` / `TabularDictDataset`) datasets.
- New `Benchmark.write_submission(predictions, out_path)` default hook: writes a 2-column `row_id,score` CSV. Subclasses override to emit competition-specific shapes. Benchmark owns the format — the harness stays dataset-agnostic.
- Model persistence:
  - **Torch algos:** the runner attaches a Lightning `ModelCheckpoint` callback pointing at `<results_dir>/checkpoints/<benchmark>-<algo>-<config_hash>-seed<seed>.ckpt`, force-enabling `enable_checkpointing` even when the smoke-gate YAML sets it to `false`. A JSON sidecar next to the ckpt stores the exact `(algo_cfg, benchmark_cfg, seed)` used at fit time.
  - **Classical algos:** `Popularity.save/load` persist a pickle next to the checkpoints dir. Algos that don't implement `save` skip persistence and `recsys submit` raises a clear error if you try to load them.
- `RunResult` gained a `model_checkpoint_path` column; `ResultStore.get_run(benchmark, algo, config_hash=..., seed=..., latest=True)` resolves a single row back to a `RunResult`.
- New `recsys submit --benchmark <name> --algo <name> [--config-hash <h>] [--seed <n>] --out <path> [--results-dir <dir>]` verb:
  1. Query `ResultStore.get_run` for the matching run.
  2. Read the sidecar JSON next to `model_checkpoint_path` to recover the original configs.
  3. Rebuild the benchmark + algo (same `feature_map` / `feature_specs` injection as `bench`).
  4. Load the checkpoint (Lightning `.ckpt` via `torch.load` + `load_state_dict` onto a fresh `LightningCTRTask` shell since `LightningCTRTask` doesn't use `save_hyperparameters`; classical pickles via `algo.load(path)`).
  5. Call `benchmark.task.export_predictions`, which streams predictions into `benchmark.write_submission(...)`.

### Verified

- Smoke gate: `deepfm_on_movielens_ctr`, `din_on_movielens_seq`, `popularity_on_movielens_ctr`, `popularity_on_kuairand_ctr` all still pass (metrics unchanged modulo seed noise). `popularity_on_kuairec_ctr` is blocked on a pre-existing on-disk layout mismatch (`KuaiRec/KuaiRec 2.0/` vs expected `KuaiRec/data/`) — unrelated to v2.1.
- End-to-end `recsys submit` round-trips both `popularity_on_movielens_ctr` and `deepfm_on_movielens_ctr`, producing 4M-row `row_id,score` CSVs from the persisted checkpoints.

### Deferred (explicitly *not* in v2.1)

- **Memmap-backed long-sequence loader.** The original primitives list included a `np.memmap` / `.npy`-backed sequence store for datasets with >1k-step histories. No in-repo benchmark currently needs it — will land when the first dataset that actually requires it (likely TAAC2026 on the full split) arrives.
- **Unlabeled test split contract.** `BenchmarkData.test` still aliases `val` for every built-in benchmark. The new submit flow works fine on labeled test (labels are simply ignored by `iter_predictions`); a real unlabeled split lands when a benchmark subclass needs it.
