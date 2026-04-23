# Supported models and benchmarks

The harness ships with four algorithms and six benchmarks. Each algorithm is registered on [ALGO_REGISTRY](../src/recsys/utils.py) and each benchmark on [BENCHMARK_REGISTRY](../src/recsys/utils.py); see [CLAUDE.md](../CLAUDE.md) for the side-effect-import conventions that keep those registrations live.

## Algorithms

| Algo | Registry key | Kind | Batch shape | Required feature roles | Notes |
|---|---|---|---|---|---|
| Popularity | `popularity` | Classical | Flat `(B, n_fields)` tensor only | `item`, `label` | Scores each item by log1p-of-train-count. Runner uses the classical bypass — Lightning is never instantiated. Rejects dict batches. |
| DeepFM | `deepfm` | Torch (`nn.Module`) | Flat tensor **or** dict batch | `item`, `label` (typically plus `user`) | Dual-mode forward: legacy all-scalar path uses one shared embedding table; v2.1 mixed-type path (triggered when any `DENSE_VECTOR` / `MULTI_CATEGORICAL` feature is present) uses per-field embeddings + `nn.EmbeddingBag` for multi-cat. |
| DIN | `din` | Torch (`nn.Module`) | Dict batch only | `item`, `label`, plus a `hist_*` history feature | Attention over user-history via `LocalActivationUnit`. Consumes `{item_id, hist_item_id, hist_item_id_mask, user_id, ...}`. The v2.1 multi-stream refactor lets DIN attend over N parallel history streams; the default single-stream path matches the pre-v2.1 MovieLens/Amazon layout. |
| DIEN | `dien` | Torch (`nn.Module`) | Dict batch only | `item`, `label`, plus a `hist_*` history feature | DIN's successor (Zhou et al. AAAI 2019). Interest extractor `nn.GRU` over history embeddings → DIN-style softmax attention vs target → `AUGRU` (update gate scaled by attention score) → MLP. Adds a next-behavior auxiliary loss (Eq. 6) exposed via the opt-in `model.last_aux_loss` attribute that `CTRTask` picks up and weights by `aux_loss_weight` (algo YAML key, default `1.0`). Consumes the same batches as DIN. |

Sources: [popularity.py](../src/recsys/algorithms/classical/popularity.py), [deepfm.py](../src/recsys/algorithms/torch/deepfm.py), [din.py](../src/recsys/algorithms/torch/din.py), [dien.py](../src/recsys/algorithms/torch/dien.py).

### Key wiring gotcha

`Popularity` rejects dict batches. If a benchmark YAML lists a `MULTI_CATEGORICAL` or `DENSE_VECTOR` feature (which causes the datamodule to emit `TabularDictDataset`), the Popularity smoke gate will fail with `TypeError: Popularity baseline currently only supports tabular tensor batches`. Keep the default benchmark YAMLs all-scalar, or gate Popularity only against a scalar-features variant.

## Benchmarks

| Benchmark | Registry key | Dataset | Task | Model input | Compatible algos | Metrics |
|---|---|---|---|---|---|---|
| MovieLens CTR | `movielens_ctr` | `movielens` | CTR | Tabular | `popularity`, `deepfm` | AUC + LogLoss + NDCG/Recall/HR@{10,50} + MRR (sampled-100 negatives) |
| MovieLens Seq | `movielens_seq` | `movielens` | CTR (history-aware) | Sequence (dict batch) | `din`, `dien` | same 9 metrics |
| KuaiRec CTR | `kuairec_ctr` | `kuairec` | CTR | Tabular | `popularity`, `deepfm` | same 9 metrics |
| KuaiRand CTR | `kuairand_ctr` | `kuairand` | CTR | Tabular | `popularity`, `deepfm` | same 9 metrics |
| Amazon CTR | `amazon_ctr` | `amazon` | CTR | Tabular | `popularity`, `deepfm` | same 9 metrics |
| Amazon Seq | `amazon_seq` | `amazon` | CTR (history-aware) | Sequence (dict batch) | `din`, `dien` | same 9 metrics |

All benchmarks bind to [`CTRTask`](../src/recsys/tasks/ctr.py) and use `RandomSplit(train_fraction=0.8)` + `RandomUniform()` negatives by default (the benchmark class is free to pin different values — see each class's `__init__`).

### Cross-benchmark notes

- `metric_names` is a class attribute on each benchmark — it is not a YAML key. Changing metrics means writing a new benchmark class, not editing a config.
- `movielens_ctr` / `movielens_seq` and `amazon_ctr` / `amazon_seq` are each a pair over the same dataset: the only difference is `model_input: sequence` in the YAML, which routes the datamodule to produce dict batches with `hist_item_id` for DIN. Ranking metrics for DIN work via `CTREvaluator._dict_batch_ranking` (see [evaluator.py](../src/recsys/evaluation/evaluator.py)), which unwraps `torch.utils.data.Subset` and scores each user's held-out positive against sampled-K negatives.
- Cross-dataset AUC numbers are not comparable. The `config_hash` written to each `results/<benchmark>.parquet` row pins dataset + label + split; swapping, say, Amazon's `category: books` into the same benchmark YAML produces a different hash and lands in its own row.
