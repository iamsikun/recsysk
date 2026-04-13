from __future__ import annotations

from typing import Any, Dict, List
from recsys.schemas.features import FeatureSpec, FeatureType, FeatureRole


def _infer_role(name: str) -> FeatureRole:
    """
    Infer a default FeatureRole from a feature name when the config does
    not specify one explicitly.
    """
    if name == "user_id":
        return FeatureRole.USER
    if name == "item_id":
        return FeatureRole.ITEM
    if name.startswith("hist_"):
        return FeatureRole.SEQUENCE
    raise ValueError(
        f"Cannot infer FeatureRole for feature '{name}'. "
        f"Please add an explicit `role:` field to this feature in the config "
        f"(one of: user, item, context, sequence, group, label)."
    )


def _parse_role(value: Any, name: str) -> FeatureRole:
    """Parse a role value that may be upper- or lowercase."""
    if isinstance(value, FeatureRole):
        return value
    if not isinstance(value, str):
        raise ValueError(
            f"Invalid role value for feature '{name}': {value!r}. Expected a string."
        )
    try:
        return FeatureRole(value.lower())
    except ValueError as e:
        valid = ", ".join(r.value for r in FeatureRole)
        raise ValueError(
            f"Invalid role '{value}' for feature '{name}'. Expected one of: {valid}."
        ) from e


def build_feature_specs(feature_configs: List[Dict[str, Any]]) -> List[FeatureSpec]:
    """
    Parses a list of dicts from YAML into a list of FeatureSpec objects.
    """
    specs = []
    for raw_cfg in feature_configs:
        cfg = dict(raw_cfg)  # shallow copy so we don't mutate the caller's dict

        # 1. Convert String to Enum for type
        f_type_str = cfg.pop("type").lower()
        f_type = FeatureType(f_type_str)  # Look up by value, not by name

        # 2. Resolve role: explicit if provided, otherwise inferred
        name = cfg.get("name")
        if "role" in cfg:
            role = _parse_role(cfg.pop("role"), name)
        else:
            role = _infer_role(name)

        # 3. Optional group_id (only meaningful for GROUP role, but we accept it)
        group_id = cfg.pop("group_id", None)

        # 4. Validate type-specific required fields.
        if f_type == FeatureType.DENSE_VECTOR:
            if cfg.get("vector_dim") is None:
                raise ValueError(
                    f"Feature '{name}' has type=dense_vector but no "
                    f"'vector_dim' is set. dense_vector features must "
                    f"declare a fixed width."
                )
        if f_type == FeatureType.MULTI_CATEGORICAL:
            if cfg.get("max_len") is None:
                raise ValueError(
                    f"Feature '{name}' has type=multi_categorical but no "
                    f"'max_len' is set. multi_categorical features must "
                    f"declare a pad/truncate length."
                )

        # 5. Create Dataclass
        spec = FeatureSpec(type=f_type, role=role, group_id=group_id, **cfg)
        specs.append(spec)

    return specs
