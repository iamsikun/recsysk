from __future__ import annotations

from typing import Any, Dict, List
from recsys.schemas.features import FeatureSpec, FeatureType


def build_feature_specs(feature_configs: List[Dict[str, Any]]) -> List[FeatureSpec]:
    """
    Parses a list of dicts from YAML into a list of FeatureSpec objects.
    """
    specs = []
    for cfg in feature_configs:
        # 1. Convert String to Enum
        # specific logic to map 'categorical' -> FeatureType.CATEGORICAL
        f_type_str = cfg.pop("type").lower()
        f_type = FeatureType(f_type_str)  # Look up by value, not by name

        # 2. Create Dataclass
        # We unpack the rest of the config (name, source_col)
        spec = FeatureSpec(type=f_type, **cfg)
        specs.append(spec)

    return specs
