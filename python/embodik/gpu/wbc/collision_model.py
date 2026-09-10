"""Panda collision geometry naming and exclusion policy shared by CPU/GPU paths."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence


def panda_collision_geometry_name(body_label: str, ordinal: int) -> str:
    """Return the Pinocchio/EmbodiK name for one URDF collision element."""

    if ordinal < 0:
        raise ValueError("collision geometry ordinal must be non-negative")
    body_name = body_label.rsplit("/", 1)[-1]
    if not body_name:
        raise ValueError("body label must not be empty")
    return f"{body_name}_{ordinal}"


def _link_index(name: str) -> int | None:
    match = re.search(r"(?:link|joint)[_-]?(\d+)", name.lower())
    return int(match.group(1)) if match else None


def panda_collision_pair_excluded(name_a: str, name_b: str) -> bool:
    """Match the latest EmbodiK Panda example's self-collision exclusions."""

    a_lower, b_lower = name_a.lower(), name_b.lower()
    a_ee = any(token in a_lower for token in ("finger", "hand"))
    b_ee = any(token in b_lower for token in ("finger", "hand"))
    index_a, index_b = _link_index(a_lower), _link_index(b_lower)
    if a_ee and b_ee:
        return True
    if a_ee != b_ee:
        other_index = index_b if a_ee else index_a
        return other_index is not None and other_index >= 5
    return index_a is not None and index_b is not None and abs(index_a - index_b) <= 2


def retained_collision_pairs(
    geometries: Sequence[tuple[str, int, int]],
) -> tuple[tuple[int, int], ...]:
    """Build explicit shape pairs from ``(name, shape, body)`` records."""

    if len({name for name, _, _ in geometries}) != len(geometries):
        raise ValueError("collision geometry names must be unique")
    pairs = []
    for index, (name_a, shape_a, body_a) in enumerate(geometries):
        for name_b, shape_b, body_b in geometries[index + 1 :]:
            if body_a == body_b or panda_collision_pair_excluded(name_a, name_b):
                continue
            pairs.append((shape_a, shape_b))
    return tuple(pairs)


def replicated_shape_pairs(
    source_pairs: Iterable[tuple[int, int]],
    *,
    shape_count: int,
    world_count: int,
) -> tuple[tuple[int, int], ...]:
    """Offset source shape pairs into contiguous replicated Newton worlds."""

    if shape_count <= 0 or world_count <= 0:
        raise ValueError("shape_count and world_count must be positive")
    source = tuple(source_pairs)
    if any(a < 0 or b < 0 or a >= shape_count or b >= shape_count for a, b in source):
        raise ValueError("source pair shape index is out of range")
    return tuple(
        (shape_a + world * shape_count, shape_b + world * shape_count)
        for world in range(world_count)
        for shape_a, shape_b in source
    )
