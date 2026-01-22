# personalized-recommendation

## Data pipeline layout
- `recsys/data/loaders`: raw dataset readers (tables only)
- `recsys/data/transforms`: reusable transforms (labels, encoders, sequences)
- `recsys/data/builders`: combine loaders + transforms into datasets
- `recsys/data/datamodules`: thin Lightning wrappers around builders

## Adding a new dataset (Amazon, Taobao, Simulators)
1. Add a loader under `recsys/data/loaders`.
2. Add a builder under `recsys/data/builders` (tabular and/or sequence).
3. Add a datamodule under `recsys/data/datamodules` and register it in the dataset registry.

`model_input` is validated and must be one of `tabular` or `sequence`.
