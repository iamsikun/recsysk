from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import torch

from recsys.data.datasets import SequenceDataset


@dataclass(frozen=True)
class SequenceStream:
    """One parallel history stream.

    ``source_feature`` is the column in the source dataframe whose value
    gets appended to the user's running history on each event (e.g.
    ``item_id`` in a single-stream MovieLens setup, or ``action_id`` /
    ``content_id`` in a multi-stream TAAC2026 setup).

    ``history_feature`` is the batch-dict key the resulting tensor will
    land under. A mask tensor is emitted alongside under
    ``f"{history_feature}_mask"``.
    """

    name: str
    source_feature: str
    history_feature: str
    max_len: int


@dataclass(frozen=True)
class SequenceSpec:
    """Configuration for building sequence features from event data.

    A sequence spec pins one *target* id column (the candidate being
    scored) plus 1..N parallel history streams. The common single-stream
    case (MovieLens/KuaiRec) can be constructed via
    :meth:`SequenceSpec.single_stream`.
    """

    target_feature: str
    streams: tuple[SequenceStream, ...]
    sparse_feature_names: list[str]
    dense_feature_names: list[str]

    @classmethod
    def single_stream(
        cls,
        *,
        item_feature: str,
        history_feature: str,
        max_history_len: int,
        sparse_feature_names: list[str] | None = None,
        dense_feature_names: list[str] | None = None,
    ) -> "SequenceSpec":
        """Construct a single-stream spec matching the pre-multi-stream
        MovieLens/KuaiRec layout. The one stream is named ``"item"`` and
        its source column == the target column, so the runner both
        scores on ``item_feature`` and builds history from it.
        """
        stream = SequenceStream(
            name="item",
            source_feature=item_feature,
            history_feature=history_feature,
            max_len=max_history_len,
        )
        return cls(
            target_feature=item_feature,
            streams=(stream,),
            sparse_feature_names=list(sparse_feature_names or []),
            dense_feature_names=list(dense_feature_names or []),
        )


def validate_sequence_spec(
    df: pl.DataFrame,
    feature_map: dict[str, int],
    spec: SequenceSpec,
) -> None:
    if spec.target_feature not in df.columns:
        raise ValueError(f"target_feature '{spec.target_feature}' not found in data")
    if spec.target_feature not in feature_map:
        raise ValueError(
            f"target_feature '{spec.target_feature}' must be included in features"
        )
    if "user_id" not in df.columns:
        raise ValueError("Sequence data requires a 'user_id' column")
    if not spec.streams:
        raise ValueError("SequenceSpec needs at least one stream")
    if spec.target_feature in spec.sparse_feature_names:
        raise ValueError("target_feature should not be included in sparse_feature_names")
    seen_hist_keys: set[str] = set()
    for stream in spec.streams:
        if stream.source_feature not in df.columns:
            raise ValueError(
                f"stream '{stream.name}' source_feature "
                f"'{stream.source_feature}' not found in data"
            )
        if stream.max_len <= 0:
            raise ValueError(
                f"stream '{stream.name}' max_len must be > 0, got {stream.max_len}"
            )
        if stream.history_feature in seen_hist_keys:
            raise ValueError(
                f"duplicate history_feature key '{stream.history_feature}'"
            )
        seen_hist_keys.add(stream.history_feature)
    missing_sparse = [
        name for name in spec.sparse_feature_names if name not in df.columns
    ]
    if missing_sparse:
        raise ValueError(f"Missing sparse feature columns: {missing_sparse}")
    missing_dense = [name for name in spec.dense_feature_names if name not in df.columns]
    if missing_dense:
        raise ValueError(f"Missing dense feature columns: {missing_dense}")


def build_sequence_dataset(
    df: pl.DataFrame,
    feature_map: dict[str, int],
    spec: SequenceSpec,
) -> SequenceDataset:
    """Build sequence inputs from an event-log dataframe.

    Each row becomes one training example whose label is the row's own
    label column and whose history (per stream) is the user's prior
    events truncated/padded to the stream's ``max_len``. The first event
    for each user is used to seed history buffers and does not itself
    produce a training row (matching the pre-multi-stream behavior).
    """
    validate_sequence_spec(df, feature_map, spec)

    sort_cols: list[str] = ["user_id"]
    if "timestamp" in df.columns:
        sort_cols.append("timestamp")
    df = df.sort(sort_cols)

    user_ids = df["user_id"].to_numpy()
    target_ids = df[spec.target_feature].to_numpy()
    labels = df["label"].to_numpy().astype("float32")

    stream_arrays = {
        stream.name: df[stream.source_feature].to_numpy()
        for stream in spec.streams
    }
    sparse_arrays = {
        name: df[name].to_numpy() for name in spec.sparse_feature_names
    }
    dense_arrays = {
        name: df[name].to_numpy().astype("float32")
        for name in spec.dense_feature_names
    }

    target_items_out: list[int] = []
    labels_out: list[float] = []
    sparse_rows: list[list[int]] = []
    dense_rows: list[list[float]] = []
    stream_hist_out: dict[str, list[list[int]]] = {
        s.name: [] for s in spec.streams
    }
    stream_mask_out: dict[str, list[list[bool]]] = {
        s.name: [] for s in spec.streams
    }

    current_user = None
    running_hists: dict[str, list[int]] = {s.name: [] for s in spec.streams}

    # Use the first stream's buffer length as the "has seen any event"
    # sentinel; all stream buffers advance in lock-step so any one works.
    sentinel_stream = spec.streams[0].name

    for idx in range(len(user_ids)):
        user_id = user_ids[idx]
        if current_user != user_id:
            current_user = user_id
            for name in running_hists:
                running_hists[name] = []

        if not running_hists[sentinel_stream]:
            # First event for this user: seed all stream buffers and
            # skip emitting a training row.
            for stream in spec.streams:
                running_hists[stream.name].append(
                    int(stream_arrays[stream.name][idx])
                )
            continue

        target_items_out.append(int(target_ids[idx]))
        labels_out.append(float(labels[idx]))
        for stream in spec.streams:
            buf = running_hists[stream.name]
            hist = buf[-stream.max_len :]
            pad_len = stream.max_len - len(hist)
            padded = ([0] * pad_len) + hist
            mask = ([False] * pad_len) + ([True] * len(hist))
            stream_hist_out[stream.name].append(padded)
            stream_mask_out[stream.name].append(mask)

        if spec.sparse_feature_names:
            sparse_rows.append(
                [int(sparse_arrays[name][idx]) for name in spec.sparse_feature_names]
            )
        if spec.dense_feature_names:
            dense_rows.append(
                [float(dense_arrays[name][idx]) for name in spec.dense_feature_names]
            )

        for stream in spec.streams:
            running_hists[stream.name].append(
                int(stream_arrays[stream.name][idx])
            )

    features: dict[str, torch.Tensor] = {
        spec.target_feature: torch.tensor(target_items_out, dtype=torch.long),
    }
    for stream in spec.streams:
        features[stream.history_feature] = torch.tensor(
            stream_hist_out[stream.name], dtype=torch.long
        )
        features[f"{stream.history_feature}_mask"] = torch.tensor(
            stream_mask_out[stream.name], dtype=torch.bool
        )

    if spec.sparse_feature_names:
        features["sparse_features"] = torch.tensor(sparse_rows, dtype=torch.long)
    if spec.dense_feature_names:
        features["dense_features"] = torch.tensor(dense_rows, dtype=torch.float32)

    labels_tensor = torch.tensor(labels_out, dtype=torch.float32).unsqueeze(-1)
    return SequenceDataset(features, labels_tensor)
