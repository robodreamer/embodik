"""Small runtime helpers for interactive IK loops.

These helpers keep example code focused on tasks, targets, and visualization
while preserving the conservative status handling expected from constrained IK
teleoperation loops.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable, Sequence

import numpy as np


@dataclass
class RobustStepResult:
    """Normalized outcome for one interactive IK solve step."""

    q_next: np.ndarray
    solver_result: object
    elapsed_ms: float
    solver_calls: int


@dataclass(frozen=True)
class ConstraintBoundary:
    """Observed scalar boundary where larger values mean more clearance."""

    name: str
    value: float | None
    minimum: float
    enabled: bool = True
    violation_tolerance: float = 1e-5
    boundary_slack: float = 5e-3

    @property
    def observed(self) -> bool:
        return self.enabled and self.value is not None and np.isfinite(float(self.value))

    @property
    def violated(self) -> bool:
        return self.observed and float(self.value) < float(self.minimum) - float(self.violation_tolerance)

    @property
    def clear(self) -> bool:
        return (not self.enabled) or (not self.observed) or float(self.value) >= float(self.minimum) - float(self.violation_tolerance)

    @property
    def near(self) -> bool:
        return self.observed and float(self.value) <= float(self.minimum) + float(self.boundary_slack)


@dataclass
class ConstraintDecision:
    """Decision from ``ConstrainedStepGuard.evaluate``."""

    q_next: np.ndarray
    restored_last_safe: bool
    restore_labels: list[str]
    boundary_stall_labels: list[str]
    zero_motion_resync: bool
    zero_motion_snap: bool


def clip_configuration(robot, q: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> np.ndarray:
    """Clip joints to limits while preserving floating-base quaternion validity."""
    q_out = np.asarray(q, dtype=float).copy()
    if getattr(robot, "is_floating_base", False) and q_out.size >= 7:
        q_out[7:] = np.clip(q_out[7:], np.asarray(q_lo[7:], dtype=float), np.asarray(q_hi[7:], dtype=float))
        quat = q_out[3:7]
        norm = float(np.linalg.norm(quat))
        if np.isfinite(norm) and norm > 1e-12:
            q_out[3:7] = quat / norm
        else:
            q_out[3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return q_out
    return np.clip(q_out, np.asarray(q_lo, dtype=float), np.asarray(q_hi, dtype=float))


def clear_all_target_velocities_if_available(solver) -> None:
    """Clear stale target velocities when the runtime exposes that API."""
    if hasattr(solver, "clear_all_target_velocities"):
        solver.clear_all_target_velocities()


def solver_status_name(result: object) -> str:
    """Return the string name for a solver result status."""
    status = getattr(result, "status", None)
    return getattr(status, "name", str(status))


def joint_velocity_norm(result: object) -> float:
    """Return ``||dq||`` for a solver result, or zero when unavailable."""
    dq = getattr(result, "joint_velocities", None)
    if dq is None:
        return 0.0
    try:
        return float(np.linalg.norm(np.asarray(dq, dtype=float)))
    except Exception:
        return 0.0


def is_constraint_boundary_stall(
    *,
    result: object,
    boundary: ConstraintBoundary,
    status_names: set[str] | None = None,
    dq_stall_eps: float = 1e-5,
) -> bool:
    """Classify zero-motion near-boundary plateaus as constrained holds."""
    if not boundary.near:
        return False
    if solver_status_name(result) not in (status_names or {"NO_PROGRESS", "INFEASIBLE", "NUMERICAL_ERROR", "SUCCESS"}):
        return False
    return joint_velocity_norm(result) <= float(dq_stall_eps)


def is_collision_boundary_stall(
    *,
    result: object,
    collision_enabled: bool,
    current_collision_min: float | None,
    collision_min_distance_m: float,
    distance_slack_m: float = 5e-3,
    dq_stall_eps: float = 1e-5,
) -> bool:
    """Compatibility wrapper for collision-clearance boundary stalls."""
    return is_constraint_boundary_stall(
        result=result,
        boundary=ConstraintBoundary(
            "collision",
            current_collision_min,
            collision_min_distance_m,
            enabled=collision_enabled,
            boundary_slack=distance_slack_m,
        ),
        dq_stall_eps=dq_stall_eps,
    )


def is_com_boundary_stall(
    *,
    result: object,
    com_enabled: bool,
    current_com_min_slack: float | None,
    boundary_slack_m: float = 5e-3,
    dq_stall_eps: float = 1e-5,
) -> bool:
    """Compatibility wrapper for CoM support-polygon boundary stalls."""
    return is_constraint_boundary_stall(
        result=result,
        boundary=ConstraintBoundary(
            "CoM",
            current_com_min_slack,
            0.0,
            enabled=com_enabled,
            violation_tolerance=1e-4,
            boundary_slack=boundary_slack_m,
        ),
        dq_stall_eps=dq_stall_eps,
    )


class ConstrainedStepGuard:
    """Remember last-safe ``q`` and classify constrained zero-motion states."""

    def __init__(
        self,
        q_initial: np.ndarray,
        *,
        zero_motion_resync_frames: int = 5,
        zero_motion_snap_frames: int = 20,
        zero_motion_eps: float = 1e-8,
        success_zero_motion_eps: float = 1e-4,
    ) -> None:
        self.last_safe_q = np.asarray(q_initial, dtype=float).copy()
        self.zero_motion_count = 0
        self.zero_motion_resync_frames = int(zero_motion_resync_frames)
        self.zero_motion_snap_frames = int(zero_motion_snap_frames)
        self.zero_motion_eps = float(zero_motion_eps)
        self.success_zero_motion_eps = float(success_zero_motion_eps)

    def reset(self, q_safe: np.ndarray) -> None:
        """Reset guard state around a known-safe configuration."""
        self.last_safe_q = np.asarray(q_safe, dtype=float).copy()
        self.zero_motion_count = 0

    def remember_if_clear(self, q: np.ndarray, boundaries: Sequence[ConstraintBoundary]) -> None:
        """Store ``q`` as last safe when all observed enabled boundaries are clear."""
        if all(boundary.clear for boundary in boundaries):
            self.last_safe_q = np.asarray(q, dtype=float).copy()

    def evaluate(
        self,
        *,
        q_candidate: np.ndarray,
        result: object,
        max_task_error: float,
        boundaries: Sequence[ConstraintBoundary],
        task_deadband: float,
        constraints_enabled: bool,
    ) -> ConstraintDecision:
        """Apply generic constrained-step policy to a candidate configuration."""
        q_next = np.asarray(q_candidate, dtype=float).copy()
        restore_labels = [boundary.name for boundary in boundaries if boundary.violated]
        restored = bool(restore_labels) and np.linalg.norm(q_next - self.last_safe_q) > 1e-12
        if restored:
            q_next = self.last_safe_q.copy()

        boundary_stall_labels = [
            boundary.name
            for boundary in boundaries
            if is_constraint_boundary_stall(result=result, boundary=boundary)
        ]

        status_name = solver_status_name(result)
        dq_norm = joint_velocity_norm(result)
        zero_motion = (
            max_task_error > float(task_deadband)
            and (
                (
                    dq_norm <= self.zero_motion_eps
                    and (
                        status_name in {"NO_PROGRESS", "INFEASIBLE", "NUMERICAL_ERROR"}
                        or len(list(getattr(result, "saturated_joints", []) or [])) > 0
                    )
                )
                or (
                    constraints_enabled
                    and status_name == "SUCCESS"
                    and dq_norm <= self.success_zero_motion_eps
                )
            )
        )
        if zero_motion:
            self.zero_motion_count += 1
        else:
            self.zero_motion_count = 0

        zero_motion_resync = self.zero_motion_count >= self.zero_motion_resync_frames
        zero_motion_snap = self.zero_motion_count >= self.zero_motion_snap_frames
        if zero_motion_snap:
            self.zero_motion_count = 0

        return ConstraintDecision(
            q_next=q_next,
            restored_last_safe=restored,
            restore_labels=restore_labels,
            boundary_stall_labels=boundary_stall_labels,
            zero_motion_resync=zero_motion_resync,
            zero_motion_snap=zero_motion_snap,
        )


def _apply_targets_to_solver_tasks(solver, targets: Iterable[object]) -> None:
    """Mirror ``TaskTarget`` payloads onto solver tasks for velocity fallback."""
    for target in targets:
        task = solver.get_task(target.task_name)
        pose = np.asarray(target.target_pose, dtype=float)
        try:
            task.set_target_pose(pose[:3, 3], pose[:3, :3])
            continue
        except Exception:
            pass
        try:
            task.set_target_orientation(pose[:3, :3])
            continue
        except Exception:
            pass
        task.set_target_position(pose[:3, 3])


def robust_solve_position_step(
    *,
    robot,
    solver,
    q_current: np.ndarray,
    targets: Sequence[object],
    options,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    zero_velocity_indices: Sequence[int] | None = None,
    fallback_status_names: Sequence[str] = (),
    hold_status_names: Sequence[str] = ("NON_FINITE_INPUT",),
    allow_solver_intervention: bool = False,
    apply_collision_violated_q_solution: bool = False,
) -> RobustStepResult:
    """Run ``solve_position_step`` with consistent example-side recovery."""
    q_prev = np.asarray(q_current, dtype=float).copy()
    q_next = q_prev.copy()
    zero_velocity_indices = list(zero_velocity_indices or [])

    t0 = time.perf_counter()
    result = solver.solve_position_step(q_prev, list(targets), options)
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    solver_calls = 1
    status_name = solver_status_name(result)
    solver_intervened = (
        int(getattr(result, "collision_rejection_count", 0)) > 0
        or int(getattr(result, "stall_escape_count", 0)) > 0
    )

    if status_name in hold_status_names:
        return RobustStepResult(q_next=q_prev, solver_result=result, elapsed_ms=elapsed_ms, solver_calls=solver_calls)

    if allow_solver_intervention and solver_intervened and hasattr(result, "q_solution"):
        q_next = clip_configuration(robot, np.asarray(result.q_solution, dtype=float), q_lo, q_hi)
        return RobustStepResult(q_next=q_next, solver_result=result, elapsed_ms=elapsed_ms, solver_calls=solver_calls)

    if apply_collision_violated_q_solution and status_name == "COLLISION_VIOLATED" and hasattr(result, "q_solution"):
        q_next = clip_configuration(robot, np.asarray(result.q_solution, dtype=float), q_lo, q_hi)
        return RobustStepResult(q_next=q_next, solver_result=result, elapsed_ms=elapsed_ms, solver_calls=solver_calls)

    if status_name in set(fallback_status_names):
        _apply_targets_to_solver_tasks(solver, targets)
        t1 = time.perf_counter()
        vel_result = solver.solve_velocity(q_prev, apply_limits=True)
        elapsed_ms += (time.perf_counter() - t1) * 1e3
        solver_calls += 1
        if solver_status_name(vel_result) == "SUCCESS":
            dq = np.asarray(vel_result.joint_velocities, dtype=float).copy()
            if zero_velocity_indices:
                dq[zero_velocity_indices] = 0.0
            q_next = np.asarray(robot.integrate(q_prev, dq, solver.dt), dtype=float)
            q_next = clip_configuration(robot, q_next, q_lo, q_hi)
            result = vel_result
        elif hasattr(result, "q_solution"):
            q_next = clip_configuration(robot, np.asarray(result.q_solution, dtype=float), q_lo, q_hi)
        return RobustStepResult(q_next=q_next, solver_result=result, elapsed_ms=elapsed_ms, solver_calls=solver_calls)

    if hasattr(result, "q_solution"):
        q_next = clip_configuration(robot, np.asarray(result.q_solution, dtype=float), q_lo, q_hi)

    if not np.all(np.isfinite(q_next)):
        q_next = q_prev

    return RobustStepResult(q_next=q_next, solver_result=result, elapsed_ms=elapsed_ms, solver_calls=solver_calls)


def configure_primary_solve_mode(options, solve_mode, allow_fallback: bool) -> None:
    """Mirror latest main-branch primary solve-mode wiring when available."""
    if hasattr(options, "primary_solve_mode"):
        options.primary_solve_mode = solve_mode
    if hasattr(options, "primary_allow_min_error_fallback"):
        options.primary_allow_min_error_fallback = bool(allow_fallback)
