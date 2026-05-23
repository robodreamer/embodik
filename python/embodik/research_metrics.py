"""Shared metrics for headless IK fluidity and recovery harnesses."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def _finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _percentile(values: Sequence[float], pct: float) -> float | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return None
    return float(np.percentile(finite, pct))


def _mean(values: Sequence[float]) -> float | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _min(values: Sequence[float]) -> float | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return None
    return float(np.min(finite))


def _max(values: Sequence[float]) -> float | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return None
    return float(np.max(finite))


def status_name_from_result(result_or_status: object) -> str:
    """Return a stable enum-like status name from a solver result or status."""
    status = getattr(result_or_status, "status", result_or_status)
    name = getattr(status, "name", None)
    if name is not None:
        return str(name)
    return str(status).split(".")[-1]


def joint_velocity_norm(result: object) -> float:
    """Return the L2 norm of ``result.joint_velocities`` or 0 when absent."""
    raw = getattr(result, "joint_velocities", None)
    if raw is None:
        return 0.0
    values = np.asarray(raw, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.linalg.norm(values)) if np.all(np.isfinite(values)) else float("inf")


def joint_velocity_tuple(result: object) -> tuple[float, ...]:
    """Return finite joint velocities as a tuple for sign-flip and jerk metrics."""
    raw = getattr(result, "joint_velocities", None)
    if raw is None:
        return ()
    values = np.asarray(raw, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return ()
    return tuple(float(v) for v in values)


def task_scale_from_result(result: object) -> float | None:
    """Return the minimum task scale reported by the solver, if any."""
    raw = getattr(result, "task_scales", None)
    if raw is None:
        return None
    values = np.asarray(raw, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.min(finite))


def solver_intervened(result: object) -> bool:
    """Return True when the solver applied collision rejection or stall escape."""
    return (
        int(getattr(result, "collision_rejection_count", 0) or 0) > 0
        or int(getattr(result, "stall_escape_count", 0) or 0) > 0
    )


@dataclass(slots=True)
class FluidityStepSample:
    """One IK tick from a headless recovery scenario."""

    step: int
    status_name: str
    accepted: bool
    dq_norm: float
    solve_ms: float
    task_scale: float | None = None
    task_error: float | None = None
    target_error: float | None = None
    min_collision_distance: float | None = None
    com_slack: float | None = None
    saturated_joints: tuple[str, ...] = ()
    joint_velocities: tuple[float, ...] = ()
    elastic_delta: float | None = None
    collision_rejection_count: int = 0
    stall_escape_count: int = 0
    stale_fallback: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FluiditySummary:
    """Aggregated IK fluidity metrics for one scenario."""

    scenario: str
    total_steps: int
    status_counts: dict[str, int]
    accepted_count: int
    rejected_count: int
    intervention_count: int
    collision_rejection_count: int
    stall_escape_count: int
    stale_fallback_count: int
    zero_motion_count: int
    stall_count: int
    max_zero_motion_streak: int
    max_stall_streak: int
    unrecoverable_stall_count: int
    saturated_joint_histogram: dict[str, int]
    task_scale_min: float | None
    task_scale_mean: float | None
    task_scale_p05: float | None
    dq_sign_flip_count: int
    jerk_l2_mean: float | None
    jerk_l2_max: float | None
    solve_ms_p50: float | None
    solve_ms_p95: float | None
    solve_ms_p99: float | None
    deadline_miss_count: int
    final_target_error: float | None
    peak_target_error: float | None
    min_collision_distance: float | None
    min_com_slack: float | None
    elastic_delta_min: float | None
    elastic_delta_mean: float | None
    elastic_delta_max: float | None

    @property
    def hard_failure_count(self) -> int:
        return self.unrecoverable_stall_count

    def as_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["hard_failure_count"] = self.hard_failure_count
        return record


def summarize_fluidity(
    scenario: str,
    samples: Iterable[FluidityStepSample],
    *,
    zero_motion_eps: float = 1e-6,
    sign_flip_eps: float = 1e-5,
    target_error_stall_eps: float = 1e-3,
    unrecoverable_stall_steps: int = 25,
    deadline_ms: float | None = None,
) -> FluiditySummary:
    """Aggregate per-step IK samples into quantitative fluidity metrics."""
    step_samples = list(samples)
    status_counts = Counter(sample.status_name for sample in step_samples)
    accepted_count = sum(1 for sample in step_samples if sample.accepted)
    intervention_count = sum(
        1
        for sample in step_samples
        if sample.collision_rejection_count > 0 or sample.stall_escape_count > 0
    )
    collision_rejection_count = sum(sample.collision_rejection_count for sample in step_samples)
    stall_escape_count = sum(sample.stall_escape_count for sample in step_samples)
    stale_fallback_count = sum(1 for sample in step_samples if sample.stale_fallback)
    zero_motion_flags = [
        bool(np.isfinite(sample.dq_norm) and sample.dq_norm <= zero_motion_eps)
        for sample in step_samples
    ]
    stall_flags: list[bool] = []
    for sample, zero_motion in zip(step_samples, zero_motion_flags):
        target_error = _finite_float(sample.target_error)
        unresolved_target = target_error is None or target_error > target_error_stall_eps
        saturated = bool(sample.saturated_joints)
        stall_flags.append(
            zero_motion and ((not sample.accepted) or unresolved_target or saturated)
        )

    max_zero_motion_streak = _max_bool_streak(zero_motion_flags)
    max_stall_streak = _max_bool_streak(stall_flags)
    unrecoverable_stall_count = sum(
        1
        for streak_len in _bool_streak_lengths(stall_flags)
        if streak_len >= int(unrecoverable_stall_steps)
    )

    saturated_counter: Counter[str] = Counter()
    for sample in step_samples:
        saturated_counter.update(str(item) for item in sample.saturated_joints)

    task_scales = [sample.task_scale for sample in step_samples if sample.task_scale is not None]
    solve_ms = [sample.solve_ms for sample in step_samples]
    target_errors = [
        sample.target_error for sample in step_samples if sample.target_error is not None
    ]
    collision_distances = [
        sample.min_collision_distance
        for sample in step_samples
        if sample.min_collision_distance is not None
    ]
    com_slacks = [sample.com_slack for sample in step_samples if sample.com_slack is not None]
    elastic_deltas = [
        sample.elastic_delta for sample in step_samples if sample.elastic_delta is not None
    ]

    dq_sign_flip_count, jerk_values = _motion_smoothness(step_samples, sign_flip_eps=sign_flip_eps)
    deadline_miss_count = 0
    if deadline_ms is not None:
        deadline = float(deadline_ms)
        deadline_miss_count = sum(
            1
            for sample in step_samples
            if np.isfinite(sample.solve_ms) and sample.solve_ms > deadline
        )

    return FluiditySummary(
        scenario=scenario,
        total_steps=len(step_samples),
        status_counts=dict(sorted(status_counts.items())),
        accepted_count=accepted_count,
        rejected_count=len(step_samples) - accepted_count,
        intervention_count=intervention_count,
        collision_rejection_count=collision_rejection_count,
        stall_escape_count=stall_escape_count,
        stale_fallback_count=stale_fallback_count,
        zero_motion_count=sum(1 for flag in zero_motion_flags if flag),
        stall_count=sum(1 for flag in stall_flags if flag),
        max_zero_motion_streak=max_zero_motion_streak,
        max_stall_streak=max_stall_streak,
        unrecoverable_stall_count=unrecoverable_stall_count,
        saturated_joint_histogram=dict(sorted(saturated_counter.items())),
        task_scale_min=_min(task_scales),
        task_scale_mean=_mean(task_scales),
        task_scale_p05=_percentile(task_scales, 5.0),
        dq_sign_flip_count=dq_sign_flip_count,
        jerk_l2_mean=_mean(jerk_values),
        jerk_l2_max=_max(jerk_values),
        solve_ms_p50=_percentile(solve_ms, 50.0),
        solve_ms_p95=_percentile(solve_ms, 95.0),
        solve_ms_p99=_percentile(solve_ms, 99.0),
        deadline_miss_count=deadline_miss_count,
        final_target_error=_finite_float(target_errors[-1]) if target_errors else None,
        peak_target_error=_max(target_errors),
        min_collision_distance=_min(collision_distances),
        min_com_slack=_min(com_slacks),
        elastic_delta_min=_min(elastic_deltas),
        elastic_delta_mean=_mean(elastic_deltas),
        elastic_delta_max=_max(elastic_deltas),
    )


def make_sample_from_result(
    *,
    step: int,
    result: object,
    accepted: bool,
    solve_ms: float,
    task_error: float | None = None,
    target_error: float | None = None,
    min_collision_distance: float | None = None,
    com_slack: float | None = None,
    elastic_delta: float | None = None,
    stale_fallback: bool = False,
    extra: dict[str, Any] | None = None,
) -> FluidityStepSample:
    """Build a :class:`FluidityStepSample` from an EmbodiK result object."""
    return FluidityStepSample(
        step=int(step),
        status_name=status_name_from_result(result),
        accepted=bool(accepted),
        dq_norm=joint_velocity_norm(result),
        solve_ms=float(solve_ms),
        task_scale=task_scale_from_result(result),
        task_error=_finite_float(task_error),
        target_error=_finite_float(target_error),
        min_collision_distance=_finite_float(min_collision_distance),
        com_slack=_finite_float(com_slack),
        saturated_joints=tuple(str(item) for item in getattr(result, "saturated_joints", ()) or ()),
        joint_velocities=joint_velocity_tuple(result),
        elastic_delta=_finite_float(elastic_delta),
        collision_rejection_count=int(getattr(result, "collision_rejection_count", 0) or 0),
        stall_escape_count=int(getattr(result, "stall_escape_count", 0) or 0),
        stale_fallback=bool(stale_fallback),
        extra=dict(extra or {}),
    )


def write_fluidity_report(path: Path, payload: dict[str, Any]) -> None:
    """Write a deterministic JSON metrics report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_samples_jsonl(path: Path, samples: Iterable[FluidityStepSample]) -> None:
    """Write per-step samples as JSONL for plotting or offline analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.as_dict(), sort_keys=True) + "\n")


def _bool_streak_lengths(flags: Sequence[bool]) -> list[int]:
    streaks: list[int] = []
    current = 0
    for flag in flags:
        if flag:
            current += 1
        elif current:
            streaks.append(current)
            current = 0
    if current:
        streaks.append(current)
    return streaks


def _max_bool_streak(flags: Sequence[bool]) -> int:
    streaks = _bool_streak_lengths(flags)
    return max(streaks) if streaks else 0


def _motion_smoothness(
    samples: Sequence[FluidityStepSample], *, sign_flip_eps: float
) -> tuple[int, list[float]]:
    vectors = [np.asarray(sample.joint_velocities, dtype=float) for sample in samples]
    if vectors and all(vec.size == vectors[0].size and vec.size > 0 for vec in vectors):
        return _vector_motion_smoothness(vectors, sign_flip_eps=sign_flip_eps)

    scalar_vectors = [
        np.asarray([sample.dq_norm], dtype=float)
        for sample in samples
        if np.isfinite(sample.dq_norm)
    ]
    return _vector_motion_smoothness(scalar_vectors, sign_flip_eps=sign_flip_eps)


def _vector_motion_smoothness(
    vectors: Sequence[np.ndarray], *, sign_flip_eps: float
) -> tuple[int, list[float]]:
    if len(vectors) < 2:
        return 0, []

    sign_flip_count = 0
    previous_sign = np.sign(np.where(np.abs(vectors[0]) > sign_flip_eps, vectors[0], 0.0))
    for vector in vectors[1:]:
        sign = np.sign(np.where(np.abs(vector) > sign_flip_eps, vector, 0.0))
        sign_flip_count += int(np.count_nonzero((previous_sign * sign) < 0.0))
        previous_sign = np.where(sign != 0.0, sign, previous_sign)

    if len(vectors) < 3:
        return sign_flip_count, []
    jerk_values = []
    for idx in range(2, len(vectors)):
        jerk = vectors[idx] - 2.0 * vectors[idx - 1] + vectors[idx - 2]
        jerk_values.append(float(np.linalg.norm(jerk)))
    return sign_flip_count, jerk_values
