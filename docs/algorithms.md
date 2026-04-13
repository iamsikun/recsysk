# Algorithms

## The Algorithm protocol

The base class at `src/recsys/algorithms/base.py:Algorithm` is
deliberately framework-agnostic: the module has no top-level `torch`
import, so classical baselines (popularity, item-KNN, BPR-MF) can be
imported and run without paying for a torch import. It declares four
methods — `fit`, `predict_scores`, `predict_topk`, and `save`/`load` —
plus two class attributes every subclass is expected to override:
`supported_tasks: set[TaskType]` (which task types the algo can run
against) and `required_roles: set[str]` (which `FeatureRole` names it
needs the dataset to provide, e.g. `{"user", "item", "label"}`). The
runner checks compatibility between a benchmark's task and an algo's
`supported_tasks` / `required_roles` before calling `fit`, so an
algorithm that only supports CTR never gets handed a retrieval
benchmark.

## Two subclass flavors

### Torch algorithms — `src/recsys/algorithms/torch/`

Torch algorithms are `nn.Module` subclasses registered via
`@ALGO_REGISTRY.register("name")`. DeepFM (`torch/deepfm.py`) and DIN
(`torch/din.py`) are the v1 references. They do *not* currently subclass
`Algorithm` directly; instead they are plain `nn.Module`s, and at run
time the runner wraps them in a `CTRTask` Lightning module (see
`src/recsys/engine.py`) and drives training with `pl.Trainer.fit`. A
thin `TorchAlgorithm` base under `torch/base.py` was scaffolded in
Wave 2 but remains skeletal — the runner is the seam that actually
implements `Algorithm.fit` for torch models in v1. Collapsing this into
a proper `TorchAlgorithm.fit` wrapper is a v2 cleanup.

### Classical algorithms — `src/recsys/algorithms/classical/`

Classical algorithms are pure-Python / numpy and must run without any
Lightning machinery of their own. Popularity
(`classical/popularity.py`) is the v1 reference implementation and the
"forcing function" that validates the rest of the harness: if
popularity can't run end-to-end through the same runner as DeepFM, the
Algorithm protocol is wrong.

## v1 gotcha — the `nn.Module` workaround for classical algos

In v1, `run_experiment` always instantiates a Lightning `Trainer` and
calls `trainer.fit`, regardless of whether the algorithm is neural.
That means classical algos currently have to pretend to be torch
modules to flow through the pipeline. The popularity baseline works
around this with three concrete hacks, all visible in
`src/recsys/algorithms/classical/popularity.py`:

1. It subclasses `nn.Module` and stores its parameters as a frozen
   `nn.Parameter(..., requires_grad=False)` (`self.popularity_scores`)
   so the state dict travels with the model and rides the
   device-placement plumbing.
2. It exposes a one-shot `fit_on_train_counts(train_dataset)` method
   that the runner calls *before* `trainer.fit` — this is where the
   actual popularity counting happens.
3. It keeps a tiny unused `nn.Parameter(torch.zeros(1))` purely so
   Lightning's optimizer doesn't blow up on an empty parameter list,
   and the trainer is invoked with `max_epochs=0` so the no-op
   "training" loop runs once and exits.

This is mechanical, ugly, and documented. Splitting the runner into
separate classical and torch fit paths (so `Algorithm.fit` can be
implemented directly over numpy without touching Lightning) is a v2
task. Until then, follow the popularity pattern when adding a
classical baseline — `algorithms/classical/popularity.py` is the
concrete reference.

## How to add a new algorithm

1. **Pick the subpackage.** PyTorch / Lightning algos go under
   `src/recsys/algorithms/torch/`; pure-Python / numpy baselines go
   under `src/recsys/algorithms/classical/`.
2. **Implement the forward pass (torch).** Subclass `nn.Module`. For
   tabular CTR algos, `forward(x)` takes a tensor of shape
   `(B, n_fields)` of integer feature indices and returns logits of
   shape `(B, 1)` — see `DeepFM.forward` for the canonical shape. For
   sequential / dict-batch algos, `forward(x)` takes a dict and must
   handle at least `x[item_feature]`, `x[history_feature]`, the
   matching `_mask` key, and any `x["sparse_features"]` /
   `x["dense_features"]` columns — see
   `DeepInterestNetwork.forward`. Declare `supported_tasks` (e.g.
   `{TaskType.CTR}` or `{TaskType.SEQUENTIAL}`) and `required_roles`
   (e.g. `{"user", "item", "label"}`).
3. **Implement the classical path (if non-torch).** Follow the
   popularity pattern: wrap state in a frozen `nn.Parameter`, add a
   one-shot `fit_on_train_counts(train_dataset)` method the runner
   can call before `trainer.fit`, and keep a throwaway trainable
   parameter so Lightning is happy. Alternatively, implement
   `Algorithm.fit` directly over numpy and rewire the runner to
   dispatch on `isinstance(algo, nn.Module)` — that is a v2-scale
   change, not something to slip into a single-algo PR.
4. **Register.** Decorate the class with
   `@ALGO_REGISTRY.register("name")`.
5. **Wire the import.** Import the new module from
   `src/recsys/algorithms/torch/__init__.py` or
   `src/recsys/algorithms/classical/__init__.py` so the decorator
   fires on package import.
6. **Algorithm YAML.** Create `conf/algorithms/<name>.yaml` with
   algorithm hyperparameters only — embed dim, MLP dims, dropout, etc.
   Do *not* put `data:`, `optimizer:`, or `trainer:` sections in
   here; those belong on the benchmark YAML and the experiment YAML
   respectively.
7. **Experiment YAML.** Create
   `conf/experiments/<name>_on_<benchmark>.yaml` that references the
   benchmark config, the algo config, a seed list, and any trainer
   overrides.

## Smoke-test invocation pattern

All three experiment YAMLs under `conf/experiments/` (DeepFM, DIN,
popularity) use the same smoke-test trainer overrides:

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

Paired with a zero-workers dataloader setting, this is a smoke test,
not a real training run — 4 batches over 1 epoch is enough to prove
the data pipeline, the forward pass, the metric computation, and the
result store all line up end-to-end without burning cycles on actual
convergence. Real benchmarking runs will bump `max_epochs`, drop the
`limit_*_batches` caps, and typically run across multiple seeds. The
`accelerator: auto` / `devices: 1` / zero-workers pattern is chosen
so the same YAML works uniformly on CPU laptops, single-GPU desktops,
and CI without per-environment tuning.
