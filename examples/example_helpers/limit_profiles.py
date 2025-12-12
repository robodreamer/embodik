"""
Utilities for discovering, loading, and applying joint-limit override profiles.

Profiles are described using YAML (or JSON) files that specify per-joint
overrides in terms of explicit min/max bounds and/or scale factors relative to
the baseline URDF range. Example YAML structure:

```yaml
name: alpha_extended
metadata:
  description: "Extend wrist roll by 20%"
defaults:
  scale: 1.0
overrides:
  left_wrist_roll_joint:
    scale: 1.2
  right_wrist_roll_joint:
    min: -2.6
    max: 2.6
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "PyYAML is required to load joint limit profiles. Please install 'pyyaml'."
    ) from exc


@dataclass(frozen=True)
class JointLimitOverride:
    """Description of how to modify a single joint's limits."""

    minimum: Optional[float] = None
    maximum: Optional[float] = None
    scale: Optional[float] = None
    scale_min: Optional[float] = None
    scale_max: Optional[float] = None

    def is_empty(self) -> bool:
        return (
            self.minimum is None
            and self.maximum is None
            and self.scale is None
            and self.scale_min is None
            and self.scale_max is None
        )


@dataclass(frozen=True)
class LimitProfile:
    name: str
    overrides: Dict[str, JointLimitOverride]
    metadata: Dict[str, object] = field(default_factory=dict)
    source: Optional[Path] = None

    def describe_joint(self, joint_name: str) -> Optional[JointLimitOverride]:
        return self.overrides.get(joint_name)


def _ensure_mapping(node: object, *, context: str) -> Dict[str, object]:
    if not isinstance(node, dict):
        raise ValueError(f"{context} must be a mapping, got {type(node).__name__}")
    return node


def _load_override(name: str, data: Dict[str, object]) -> JointLimitOverride:
    allowed_keys = {"min", "max", "minimum", "maximum", "scale", "scale_min", "scale_max"}
    for key in data:
        if key not in allowed_keys:
            raise ValueError(
                f"Unknown key '{key}' for joint '{name}'. "
                f"Allowed keys: {sorted(allowed_keys)}"
            )

    def _get_float(key: str) -> Optional[float]:
        value = data.get(key)
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        raise ValueError(f"Value for '{key}' on joint '{name}' must be numeric, got {value!r}")

    minimum = _get_float("min")
    if minimum is None:
        minimum = _get_float("minimum")
    maximum = _get_float("max")
    if maximum is None:
        maximum = _get_float("maximum")
    scale = _get_float("scale")
    scale_min = _get_float("scale_min")
    scale_max = _get_float("scale_max")

    override = JointLimitOverride(
        minimum=minimum,
        maximum=maximum,
        scale=scale,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    if override.is_empty():
        raise ValueError(f"Joint '{name}' override does not specify any changes.")
    return override


def load_limit_profile(path: Path) -> LimitProfile:
    """Load a limit profile from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Limit profile not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if payload is None:
        raise ValueError(f"Limit profile file {path} is empty.")
    payload = _ensure_mapping(payload, context=f"Profile {path}")

    name = payload.get("name") or path.stem
    metadata = payload.get("metadata") or {}
    metadata = dict(metadata) if isinstance(metadata, dict) else {"value": metadata}

    overrides_node = payload.get("overrides")
    if overrides_node is None:
        raise ValueError(f"Limit profile {path} does not define an 'overrides' section.")
    overrides_node = _ensure_mapping(overrides_node, context="overrides")

    overrides: Dict[str, JointLimitOverride] = {}
    for joint_name, entry in overrides_node.items():
        entry_mapping = _ensure_mapping(entry, context=f"override for joint '{joint_name}'")
        overrides[joint_name] = _load_override(joint_name, entry_mapping)

    return LimitProfile(name=str(name), overrides=overrides, metadata=metadata, source=path)


def discover_limit_profiles(search_directories: Iterable[Path]) -> Dict[str, LimitProfile]:
    """Discover profiles within the given directories."""

    discovered: Dict[str, LimitProfile] = {}
    used_labels: Dict[str, int] = {}

    def _make_unique_label(base: str, profile: LimitProfile) -> str:
        suffix_source = profile.source.stem if profile.source else base
        counter = 1
        while True:
            if counter == 1:
                label = f"{base} ({suffix_source})"
            else:
                label = f"{base} ({suffix_source}_{counter})"
            if label not in used_labels:
                return label
            counter += 1

    for directory in search_directories:
        if not directory.exists() or not directory.is_dir():
            continue
        profile_paths = sorted(directory.glob("*.yml")) + sorted(directory.glob("*.yaml"))
        for path in profile_paths:
            profile = load_limit_profile(path)
            label = profile.name
            if label in used_labels:
                label = _make_unique_label(profile.name, profile)
            used_labels[label] = 1
            discovered[label] = profile
    return discovered


def apply_limit_profile(
    joint_names: Sequence[str],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    profile: Optional[LimitProfile],
) -> Tuple[List[float], List[float], Dict[str, Tuple[float, float]]]:
    """Apply overrides, returning updated limits and a diff map."""

    lower = [float(value) for value in lower_limits]
    upper = [float(value) for value in upper_limits]
    diff: Dict[str, Tuple[float, float]] = {}

    if profile is None:
        return lower, upper, diff

    joint_lookup = {name: idx for idx, name in enumerate(joint_names)}
    for joint_name, override in profile.overrides.items():
        idx = joint_lookup.get(joint_name)
        if idx is None:
            # Not all joints in profile may exist in current model.
            continue

        base_min = lower[idx]
        base_max = upper[idx]
        if base_max < base_min:
            base_min, base_max = base_max, base_min

        new_min = base_min
        new_max = base_max

        if override.scale is not None:
            scale = override.scale
            if scale <= 0:
                raise ValueError(f"Scale for joint '{joint_name}' must be positive.")
            half_range = (base_max - base_min) * scale * 0.5
            center = (base_max + base_min) * 0.5
            new_min = center - half_range
            new_max = center + half_range

        if override.scale_min is not None:
            scale_min = override.scale_min
            if scale_min <= 0:
                raise ValueError(f"scale_min for joint '{joint_name}' must be positive.")
            offset = base_min * scale_min
            new_min = offset if base_min >= 0 else base_min * scale_min

        if override.scale_max is not None:
            scale_max = override.scale_max
            if scale_max <= 0:
                raise ValueError(f"scale_max for joint '{joint_name}' must be positive.")
            offset = base_max * scale_max
            new_max = offset if base_max >= 0 else base_max * scale_max

        if override.minimum is not None:
            new_min = override.minimum
        if override.maximum is not None:
            new_max = override.maximum

        if new_max < new_min:
            raise ValueError(
                f"Invalid override for joint '{joint_name}': max ({new_max}) < min ({new_min})."
            )

        if not (math.isfinite(new_min) and math.isfinite(new_max)):
            raise ValueError(
                f"Override produced non-finite limits for joint '{joint_name}': "
                f"({new_min}, {new_max})"
            )

        lower[idx] = new_min
        upper[idx] = new_max
        if (new_min, new_max) != (base_min, base_max):
            diff[joint_name] = (new_min, new_max)

    return lower, upper, diff


def profile_summary(profile: Optional[LimitProfile], diff: Dict[str, Tuple[float, float]]) -> str:
    """Generate human-readable summary for logging/diagnostics."""

    if profile is None or not diff:
        return "baseline limits (no overrides applied)"

    parts = [f"profile '{profile.name}'"]
    if profile.source:
        parts.append(f"({profile.source})")

    joint_parts = [f"{joint}: [{new_min:.4f}, {new_max:.4f}]" for joint, (new_min, new_max) in diff.items()]
    return f"{' '.join(parts)} -> {len(diff)} joints updated: " + ", ".join(joint_parts)
