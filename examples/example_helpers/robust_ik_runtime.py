#!/usr/bin/env python3
"""Generic runtime helpers for interactive IK examples.

These helpers are intentionally example-agnostic so public examples can share
consistent constrained-solve behavior.
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


def _status_name(result: object) -> str:
    status = getattr(result, "status", None)
    return getattr(status, "name", str(status))


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
    status_name = _status_name(result)
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
        if _status_name(vel_result) == "SUCCESS":
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
