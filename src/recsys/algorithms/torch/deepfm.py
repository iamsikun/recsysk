from __future__ import annotations

import torch
from torch import nn

from recsys.schemas.features import FeatureSpec, FeatureType
from recsys.utils import ALGO_REGISTRY


@ALGO_REGISTRY.register("deepfm")
class DeepFM(nn.Module):
    """
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction.

    It combines the power of factorization machines for recommendation and deep learning
    for feature learning in a new neural network architecture.

    Reference:
        Guo, Huifeng, et al. "DeepFM: a factorization-machine based neural network for CTR prediction."
        IJCAI 2017.

    Input handling:

    * Legacy / all-scalar path: ``forward`` receives a 2-D tensor
      ``(B, n_fields)`` of integer ids. One global embedding table is
      used (offset per field). This is the path all currently-shipped
      benchmarks take.
    * Mixed-type path: when ``feature_specs`` is passed at construction
      and at least one spec is ``DENSE_VECTOR`` / ``MULTI_CATEGORICAL``,
      ``forward`` expects a dict batch with one tensor per feature name.
      Categorical fields go through per-field embedding lookup, dense
      vectors go through a per-field ``nn.Linear`` projection, and
      multi-categorical lists go through ``nn.EmbeddingBag`` (optionally
      weighted). All field embeddings are stacked into the
      ``(B, n_fields, embed_dim)`` shape the FM / DNN components expect.
    """

    def __init__(
        self,
        feature_map: dict[str, int],
        embed_dim: int,
        mlp_dims: list[int],
        feature_specs: list[FeatureSpec] | None = None,
    ):
        """
        Args:
            feature_map: mapping of feature names to their vocabulary sizes
                (or vector widths for dense features).
            embed_dim: dimension of the embedding vectors.
            mlp_dims: dimensions of the MLP layers.
            feature_specs: optional per-feature type info. When provided,
                DeepFM runs in mixed-type (dict-batch) mode; otherwise it
                falls back to the flat-tensor legacy path.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_specs = feature_specs

        has_non_scalar = feature_specs is not None and any(
            spec.type in (FeatureType.DENSE_VECTOR, FeatureType.MULTI_CATEGORICAL)
            for spec in feature_specs
        )
        self._mixed_mode = has_non_scalar

        if not self._mixed_mode:
            # Legacy flat-tensor path.
            n_fields = len(feature_map)
            total_dim = sum(feature_map.values())
            self.cat_embed = nn.Embedding(total_dim, embed_dim)
            self.o1_fc = nn.Embedding(total_dim, 1)
            self._mixed_cat_embeds = None
            self._mixed_dense_linears = None
            self._mixed_multi_embeds = None
        else:
            # Per-field heads for mixed mode.
            assert feature_specs is not None
            self._mixed_cat_embeds = nn.ModuleDict()
            self._mixed_dense_linears = nn.ModuleDict()
            self._mixed_multi_embeds = nn.ModuleDict()
            for spec in feature_specs:
                if spec.type == FeatureType.CATEGORICAL:
                    self._mixed_cat_embeds[spec.name] = nn.Embedding(
                        feature_map[spec.name], embed_dim
                    )
                elif spec.type == FeatureType.NUMERIC:
                    # Treat as a 1-d dense vector.
                    self._mixed_dense_linears[spec.name] = nn.Linear(
                        1, embed_dim
                    )
                elif spec.type == FeatureType.DENSE_VECTOR:
                    self._mixed_dense_linears[spec.name] = nn.Linear(
                        int(spec.vector_dim or feature_map[spec.name]), embed_dim
                    )
                elif spec.type == FeatureType.MULTI_CATEGORICAL:
                    mode = "sum" if spec.weighted else "mean"
                    self._mixed_multi_embeds[spec.name] = nn.EmbeddingBag(
                        feature_map[spec.name], embed_dim, mode=mode, padding_idx=0
                    )
            n_fields = len(feature_specs)

        # DNN layer (Deep part). Shared between the two modes.
        layers = [nn.Linear(n_fields * embed_dim, mlp_dims[0])]
        for i, dim in enumerate(mlp_dims[:-1]):
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(dim, mlp_dims[i + 1]))
        layers.append(nn.Linear(mlp_dims[-1], 1))
        self.dnn_mlp = nn.Sequential(*layers)

    def _embed_mixed(
        self, x: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Stack per-feature embeddings into ``(B, n_fields, embed_dim)``."""
        assert self.feature_specs is not None
        field_embs: list[torch.Tensor] = []
        for spec in self.feature_specs:
            if spec.type == FeatureType.CATEGORICAL:
                emb = self._mixed_cat_embeds[spec.name](x[spec.name].long())
            elif spec.type == FeatureType.NUMERIC:
                col = x[spec.name].float().view(-1, 1)
                emb = self._mixed_dense_linears[spec.name](col)
            elif spec.type == FeatureType.DENSE_VECTOR:
                emb = self._mixed_dense_linears[spec.name](
                    x[spec.name].float()
                )
            elif spec.type == FeatureType.MULTI_CATEGORICAL:
                ids = x[spec.name].long()
                weights = None
                if spec.weighted:
                    weights = x.get(f"{spec.name}_weight")
                    if weights is not None:
                        weights = weights.float()
                emb = self._mixed_multi_embeds[spec.name](
                    ids, per_sample_weights=weights
                )
            else:  # pragma: no cover - validated in encoder
                continue
            field_embs.append(emb)
        return torch.stack(field_embs, dim=1)  # (B, n_fields, E)

    def forward(self, x):
        """Forward pass.

        See class docstring for input shapes.
        """
        if self._mixed_mode:
            cat_embeddings = self._embed_mixed(x)  # (B, n_fields, E)
            batch_size = cat_embeddings.shape[0]
            # No first-order term in mixed mode (no shared offset embedding
            # table). The FM second-order interaction and the deep tower
            # still fire.
            o1_output = torch.zeros(
                (batch_size, 1), device=cat_embeddings.device
            )
        else:
            # Legacy path: x is (B, n_fields) of int ids.
            cat_embeddings = self.cat_embed(x)  # (B, n_fields, E)
            o1_output = torch.sum(self.o1_fc(x), dim=1)  # (B, 1)
            batch_size = x.shape[0]

        # FM second-order component.
        square_of_sums = torch.sum(cat_embeddings, dim=1) ** 2
        sum_of_squares = torch.sum(cat_embeddings ** 2, dim=1)
        o2_output = 0.5 * torch.sum(
            square_of_sums - sum_of_squares, dim=1, keepdim=True
        )

        dnn_input = cat_embeddings.view(batch_size, -1)
        dnn_output = self.dnn_mlp(dnn_input)

        return o1_output + o2_output + dnn_output
