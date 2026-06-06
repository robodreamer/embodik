#!/usr/bin/env python3
"""Public-safe kinematic health and collision-distance metrics for harnesses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

import embodik as eik


@dataclass(frozen=True)
class HealthMetricSample:
    joint_limit_cost: float
    joint_limit_health: float
    manipulability: float
    limit_weighted_dexterity: float


@dataclass(frozen=True)
class CollisionDistanceStats:
    source: str
    sample_count: int
    minimum: float
    p05: float
    p25: float
    mean: float
    count_below_threshold: int
    worst_penetration: float


def _finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def kinematic_health_sample(
    robot: eik.RobotModel,
    q: np.ndarray,
    jacobian: np.ndarray | None = None,
    *,
    epsilon: float = 1e-6,
) -> HealthMetricSample:
    """Return health metrics matching the solver-side scoring convention."""
    q_arr = np.asarray(q, dtype=float)
    lower, upper = robot.get_joint_limits()
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    _per_joint, limit_cost = eik.joint_limit_distance(
        q_arr,
        lower_arr,
        upper_arr,
        epsilon,
    )
    joint_limit_health = 1.0 / (1.0 + float(limit_cost))
    if jacobian is None:
        manipulability = 0.0
        limit_weighted_dexterity = 0.0
    else:
        jacobian_arr = np.asarray(jacobian, dtype=float)
        manipulability = float(eik.velocity_manipulability(jacobian_arr))
        limit_weighted_dexterity = float(
            eik.singularity_joint_limit_metric(
                q_arr,
                jacobian_arr,
                lower_arr,
                upper_arr,
                epsilon,
            )
        )
    return HealthMetricSample(
        joint_limit_cost=float(limit_cost),
        joint_limit_health=float(joint_limit_health),
        manipulability=float(manipulability),
        limit_weighted_dexterity=float(limit_weighted_dexterity),
    )


def summarize_health_series(samples: Iterable[HealthMetricSample]) -> dict[str, float]:
    values = list(samples)
    if not values:
        return {}
    fields = (
        "joint_limit_cost",
        "joint_limit_health",
        "manipulability",
        "limit_weighted_dexterity",
    )
    out: dict[str, float] = {}
    for field in fields:
        arr = _finite_array(getattr(sample, field) for sample in values)
        if arr.size == 0:
            continue
        out[f"{field}_min"] = float(np.min(arr))
        out[f"{field}_p05"] = float(np.percentile(arr, 5.0))
        out[f"{field}_p25"] = float(np.percentile(arr, 25.0))
        out[f"{field}_mean"] = float(np.mean(arr))
    return out


def collision_distance_stats(
    distances: Iterable[float],
    *,
    threshold: float = 0.0,
    source: str = "joint_state_postprocess",
) -> CollisionDistanceStats:
    arr = _finite_array(distances)
    if arr.size == 0:
        return CollisionDistanceStats(
            source=source,
            sample_count=0,
            minimum=float("inf"),
            p05=float("inf"),
            p25=float("inf"),
            mean=float("inf"),
            count_below_threshold=0,
            worst_penetration=0.0,
        )
    return CollisionDistanceStats(
        source=source,
        sample_count=int(arr.size),
        minimum=float(np.min(arr)),
        p05=float(np.percentile(arr, 5.0)),
        p25=float(np.percentile(arr, 25.0)),
        mean=float(np.mean(arr)),
        count_below_threshold=int(np.sum(arr < threshold)),
        worst_penetration=float(min(0.0, np.min(arr))),
    )


def collision_distance_stats_dict(
    distances: Iterable[float],
    *,
    threshold: float = 0.0,
    source: str = "joint_state_postprocess",
) -> dict[str, float | int | str]:
    stats = collision_distance_stats(distances, threshold=threshold, source=source)
    return {
        "collision_distance_source": stats.source,
        "collision_distance_sample_count": stats.sample_count,
        "min_self_collision_dist": stats.minimum,
        "collision_distance_p05": stats.p05,
        "collision_distance_p25": stats.p25,
        "collision_distance_mean": stats.mean,
        "collision_count": stats.count_below_threshold,
        "worst_penetration": stats.worst_penetration,
    }
