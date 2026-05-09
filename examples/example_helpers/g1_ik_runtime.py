#!/usr/bin/env python3
"""Shared G1 IK utilities for Viser apps and headless harnesses."""

from __future__ import annotations

import time

import embodik
import numpy as np
from embodik.interactive_ik import joint_velocity_norm
from embodik.utils import q2r, r2q


def _clip_q(robot, q: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).copy()
    if getattr(robot, "is_floating_base", False) and q.size >= 7:
        q[7:] = np.clip(q[7:], q_lo[7:], q_hi[7:])
        quat = q[3:7]
        n = np.linalg.norm(quat)
        q[3:7] = quat / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return q
    return np.clip(q, q_lo, q_hi)


def _apply_g1_soft_knee_seed(robot, q: np.ndarray) -> np.ndarray:
    """Start from a slight crouch so leg IK has a consistent bend direction."""
    q = np.asarray(q, dtype=float).copy()
    seed = {
        "left_hip_pitch_joint": -0.18,
        "left_knee_joint": 0.36,
        "left_ankle_pitch_joint": -0.18,
        "right_hip_pitch_joint": -0.18,
        "right_knee_joint": 0.36,
        "right_ankle_pitch_joint": -0.18,
    }
    for joint_name, value in seed.items():
        try:
            idx = int(robot.get_joint_config_index(joint_name))
        except Exception:
            continue
        if 0 <= idx < q.size:
            q[idx] = float(value)
    return q


def _pose_from_ctrl(ctrl) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = np.asarray(ctrl.position, dtype=float)
    wxyz = np.asarray(ctrl.wxyz, dtype=float)
    n = np.linalg.norm(wxyz)
    if not np.isfinite(n) or n < 1e-12:
        wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        wxyz = wxyz / n
    pose[:3, :3] = q2r(np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=float), order="xyzs")
    return pose


def _set_ctrl_from_pose(ctrl, pose: np.ndarray) -> None:
    pose = np.asarray(pose, dtype=float)
    ctrl.position = tuple(pose[:3, 3])
    q_xyzw = r2q(np.asarray(pose[:3, :3], dtype=float), order="xyzs")
    ctrl.wxyz = (float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))


def _pose_signature(pose: np.ndarray) -> tuple[float, ...]:
    pose = np.asarray(pose, dtype=float)
    return tuple(np.round(pose[:3, :].reshape(-1), 5))


def _sleep_for_loop_rate(loop_t0: float, hz: float = 60.0) -> None:
    period = 1.0 / max(float(hz), 1.0)
    remaining = period - (time.perf_counter() - loop_t0)
    if remaining > 0.0:
        time.sleep(remaining)


def _frame_delta6(anchor_pose: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
    delta = np.zeros(6, dtype=float)
    delta[:3] = np.asarray(current_pose[:3, 3] - anchor_pose[:3, 3], dtype=float)
    delta[3:] = np.asarray(
        embodik.log3(
            np.asarray(anchor_pose[:3, :3], dtype=float).T
            @ np.asarray(current_pose[:3, :3], dtype=float)
        ),
        dtype=float,
    )
    return delta


def _target_position_error(robot, frame_name: str, target_pose: np.ndarray) -> float:
    current_pose = np.asarray(robot.get_frame_pose(frame_name).homogeneous(), dtype=float)
    return float(np.linalg.norm(current_pose[:3, 3] - np.asarray(target_pose, dtype=float)[:3, 3]))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), pct))


def _apply_zero_based_task_hierarchy(
    primary_tasks: list,
    secondary_tasks: list,
    tertiary_tasks: list,
) -> None:
    """Assign solver priorities with an explicit zero-based hierarchy."""
    for t in primary_tasks:
        t.priority = 0
    for t in secondary_tasks:
        t.priority = 1
    for t in tertiary_tasks:
        t.priority = 2


def _task_solve_mode(name: str, fallback):
    """Return a TaskSolveMode by enum name with installed-binding fallback."""
    return getattr(embodik.TaskSolveMode, name, fallback)


def _target_solve_mode(name: str):
    """Resolve target solve mode with a fluid fallback for older bindings."""
    if name == "SCALE_ELASTIC":
        elastic = getattr(embodik.TaskSolveMode, "SCALE_ELASTIC", None)
        if elastic is not None:
            return elastic, "SCALE_ELASTIC", True
        return embodik.TaskSolveMode.MIN_ERROR, "SCALE_ELASTIC unavailable -> MIN_ERROR", True
    return _task_solve_mode(name, embodik.TaskSolveMode.MIN_ERROR), name, False


def _enum_names(values) -> str:
    names = []
    for value in values:
        names.append(getattr(value, "name", str(value)))
    return ",".join(names) if names else "--"


def _needs_min_error_recovery(result) -> bool:
    if result.status == embodik.SolverStatus.NUMERICAL_ERROR:
        return True
    if result.status != embodik.SolverStatus.NO_PROGRESS:
        return False
    dq = np.asarray(getattr(result, "joint_velocities", []), dtype=float)
    if dq.size and np.linalg.norm(dq) > 1e-10:
        return False
    msg = str(getattr(result, "status_message", "")).lower()
    scales = [abs(float(s)) for s in getattr(result, "task_scales", [])]
    zero_scales = bool(scales) and max(scales) <= 1e-10
    return "non-finite" in msg or zero_scales


def _scaled_targets(targets, gain_scale: float):
    return [
        embodik.TaskTarget(
            t.task_name,
            np.asarray(t.target_pose, dtype=float),
            float(t.position_gain) * gain_scale,
            float(t.orientation_gain) * gain_scale,
        )
        for t in targets
    ]


def _scaled_secondary_targets(targets, secondary_gain_scale: float):
    secondary_names = {"torso_upright_ori"}
    out = []
    for t in targets:
        scale = secondary_gain_scale if t.task_name in secondary_names else 1.0
        out.append(
            embodik.TaskTarget(
                t.task_name,
                np.asarray(t.target_pose, dtype=float),
                float(t.position_gain) * scale,
                float(t.orientation_gain) * scale,
            )
        )
    return out


def _clone_position_step_options(opts):
    clone = embodik.PositionStepOptions()
    for name in (
        "position_gain",
        "orientation_gain",
        "max_steps",
        "dt",
        "max_linear_speed",
        "max_angular_speed",
        "stall_recovery",
        "elastic_band",
        "limit_change_from_seed",
        "no_progress_max_steps",
        "no_progress_error_tolerance",
        "no_progress_dq_norm_tolerance",
        "adaptive_dt",
        "adaptive_dt_max_scale",
        "adaptive_dt_reference_distance",
    ):
        if hasattr(opts, name) and hasattr(clone, name):
            setattr(clone, name, getattr(opts, name))
    for name in (
        "excluded_joint_indices",
        "locked_joint_indices",
        "integration_zero_velocity_indices",
    ):
        if hasattr(opts, name) and hasattr(clone, name):
            setattr(clone, name, list(getattr(opts, name)))
    return clone


def _max_frame_target_position_error(robot, frame_targets) -> float:
    return max(
        (
            _target_position_error(robot, frame_name, target_pose)
            for frame_name, target_pose in frame_targets
        ),
        default=0.0,
    )


def _g1_interactive_hold_error_threshold() -> float:
    return 0.01


def _limit_tangent_step(robot, q_from: np.ndarray, q_to: np.ndarray, max_component: float):
    q_step = np.asarray(robot.difference(q_from, q_to), dtype=float)
    step_component = float(np.max(np.abs(q_step))) if q_step.size else 0.0
    if step_component <= max_component or step_component <= 1e-12:
        return q_to, step_component
    limited = np.asarray(robot.integrate(q_from, q_step * (max_component / step_component), 1.0))
    return limited, max_component


def _configure_interactive_elastic_band(solver) -> None:
    if not hasattr(solver, "enable_elastic_band") or not hasattr(solver, "configure_elastic_band"):
        return
    solver.enable_elastic_band(delta_max=0.015)
    solver.configure_elastic_band(
        delta_max=0.015,
        expand_rate=0.003,
        decay_rate=0.35,
        stall_threshold=2,
        expand_only_saturated=True,
    )


def _g1_posture_controlled_joint_indices(robot) -> list[int]:
    """Bias only articulated joints; leave floating-base DOFs to frame tasks."""
    if getattr(robot, "is_floating_base", False) and int(robot.nv) > 6:
        return list(range(6, int(robot.nv)))
    return list(range(int(robot.nv)))


def _configure_g1_posture_task(posture, robot, q_target: np.ndarray) -> None:
    posture.set_target_configuration(np.asarray(q_target, dtype=float).copy())
    if hasattr(posture, "set_controlled_joint_indices"):
        posture.set_controlled_joint_indices(_g1_posture_controlled_joint_indices(robot))


def _clear_interactive_solver_transients(solver) -> None:
    """Drop stateful recovery/constraint state that can outlive a UI reset."""
    for name in (
        "disable_stall_handler",
        "disable_elastic_band",
        "clear_contact_frames",
        "clear_tight_frame_pose_constraints",
        "clear_tight_point_constraints",
        "clear_linear_velocity_constraints",
        "clear_collision_constraint",
        "clear_com_constraint",
    ):
        fn = getattr(solver, name, None)
        if fn is None:
            continue
        try:
            fn()
        except Exception:
            pass
    _configure_interactive_elastic_band(solver)


def _post_step_collision_distance(solver, q: np.ndarray) -> float:
    """Use the solver's cached collision guard before falling back to a full scan."""
    for name in ("evaluate_post_step_collision_distance", "evaluate_min_collision_distance"):
        fn = getattr(solver, name, None)
        if fn is None:
            continue
        try:
            return float(fn(q))
        except Exception:
            continue
    return float("inf")


def _current_collision_min_distance(solver, q_eval: np.ndarray | None = None) -> float | None:
    """Read the current configured collision min distance without a full app-side scan."""
    try:
        if q_eval is not None and hasattr(solver, "evaluate_collision_debug"):
            dbg = solver.evaluate_collision_debug(np.asarray(q_eval, dtype=float))
            return None if dbg is None else float(dbg.distance)
        if hasattr(solver, "get_last_collision_debug_list"):
            rows = list(solver.get_last_collision_debug_list())
            if rows:
                return min(float(row.distance) for row in rows)
        if hasattr(solver, "get_last_collision_debug"):
            dbg = solver.get_last_collision_debug()
            return None if dbg is None else float(dbg.distance)
    except Exception:
        return None
    return None


def _apply_g1_collision_tuning_mode(solver, mode_label: str) -> None:
    """Apply the same collision-tuning knobs used by the AI worker teleop app."""
    label = str(mode_label).lower()
    if hasattr(solver, "set_collision_tuning_mode") and hasattr(embodik, "CollisionTuningMode"):
        mode_map = {
            "speed": embodik.CollisionTuningMode.BALANCED,
            "balanced": embodik.CollisionTuningMode.BALANCED,
            "precise": embodik.CollisionTuningMode.PRECISE,
        }
        solver.set_collision_tuning_mode(mode_map.get(label, embodik.CollisionTuningMode.BALANCED))
    if hasattr(solver, "set_proximity_gated_collision_activation_enabled"):
        solver.set_proximity_gated_collision_activation_enabled(label != "precise")
    if hasattr(solver, "set_collision_constraint_activation_multiplier"):
        solver.set_collision_constraint_activation_multiplier(0.0 if label == "precise" else 5.0)
    if hasattr(solver, "enable_collision_pair_cache"):
        solver.enable_collision_pair_cache(True, 20, 0.05, 128)
    if hasattr(solver, "set_collision_refinement_time_budget_us"):
        solver.set_collision_refinement_time_budget_us(0 if label != "precise" else 300)
    if hasattr(solver, "enable_sphere_broadphase"):
        solver.enable_sphere_broadphase(True)


def _configure_g1_collision_constraint(
    solver,
    *,
    enabled: bool,
    min_distance_m: float,
    max_constraints: int,
    tuning_mode: str,
    include_pairs: list[tuple[str, str]],
) -> None:
    """Configure G1 collision constraint plus the solver-native stall handler."""
    if not hasattr(solver, "configure_collision_constraint"):
        return
    if not enabled:
        if hasattr(solver, "clear_collision_constraint"):
            solver.clear_collision_constraint()
        if hasattr(solver, "disable_stall_handler"):
            solver.disable_stall_handler()
        return
    _apply_g1_collision_tuning_mode(solver, tuning_mode)
    try:
        solver.configure_collision_constraint(
            min_distance=float(min_distance_m),
            include_pairs=list(include_pairs),
            exclude_pairs=[],
            nearest_points_all_pairs=False,
            max_constraints=int(max_constraints),
        )
        if hasattr(solver, "enable_stall_handler"):
            solver.enable_stall_handler(float(min_distance_m))
            if hasattr(solver, "configure_stall_handler"):
                solver.configure_stall_handler(
                    stall_threshold=3,
                    restore_rate=0.2,
                    floor_fraction=0.0,
                )
    except RuntimeError:
        if hasattr(solver, "clear_collision_constraint"):
            solver.clear_collision_constraint()
        if hasattr(solver, "disable_stall_handler"):
            solver.disable_stall_handler()


def _attempt_g1_penetration_escape_burst(
    *,
    robot,
    solver,
    q_current: np.ndarray,
    targets,
    options,
    target_tasks,
    frame_targets,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    min_distance_m: float,
    max_constraints: int,
    tuning_mode: str,
    include_pairs: list[tuple[str, str]],
    trial_steps: int = 3,
    improvement_epsilon: float = 1e-4,
) -> np.ndarray | None:
    """Temporarily drop collision to accept only a measured pull-away improvement."""
    if not hasattr(solver, "evaluate_collision_debug") or not hasattr(
        solver, "clear_collision_constraint"
    ):
        return None
    try:
        dbg_before = solver.evaluate_collision_debug(np.asarray(q_current, dtype=float))
    except Exception:
        return None
    if dbg_before is None or not np.isfinite(float(dbg_before.distance)):
        return None
    before_distance = float(dbg_before.distance)
    if before_distance >= float(min_distance_m):
        return None

    q_seed = np.asarray(q_current, dtype=float).copy()
    q_trial = q_seed.copy()
    had_stall_recovery = bool(getattr(options, "stall_recovery", False))
    try:
        solver.clear_collision_constraint()
        if hasattr(solver, "disable_stall_handler"):
            solver.disable_stall_handler()
        if hasattr(options, "stall_recovery"):
            options.stall_recovery = False
        burst_opts = _clone_position_step_options(options)
        burst_opts.max_steps = max(2, int(getattr(options, "max_steps", 1)))
        for _ in range(max(int(trial_steps), 1)):
            result, _recovered = _solve_quality_step(
                solver,
                robot,
                q_trial,
                q_lo,
                q_hi,
                targets,
                burst_opts,
                target_tasks,
                frame_targets,
                prefer_min_error=True,
                error_limit=0.08,
            )
            if not hasattr(result, "q_solution"):
                break
            q_trial = _clip_q(robot, np.asarray(result.q_solution, dtype=float), q_lo, q_hi)
            if not np.all(np.isfinite(q_trial)):
                return None
            robot.update_configuration(q_trial)
        dbg_after = solver.evaluate_collision_debug(q_trial)
        if dbg_after is None or not np.isfinite(float(dbg_after.distance)):
            return None
        if float(dbg_after.distance) > before_distance + float(improvement_epsilon):
            return q_trial
        return None
    finally:
        if hasattr(options, "stall_recovery"):
            options.stall_recovery = had_stall_recovery
        _configure_g1_collision_constraint(
            solver,
            enabled=True,
            min_distance_m=min_distance_m,
            max_constraints=max_constraints,
            tuning_mode=tuning_mode,
            include_pairs=include_pairs,
        )
        robot.update_configuration(q_seed)


def _solve_targets_as_min_error(solver, q, targets, opts, target_tasks):
    previous_modes = [task.solve_mode for task in target_tasks]
    previous_fallbacks = [bool(task.allow_min_error_fallback) for task in target_tasks]
    try:
        if hasattr(solver, "elastic_band_enabled") and solver.elastic_band_enabled():
            solver.disable_elastic_band()
        for task in target_tasks:
            task.solve_mode = embodik.TaskSolveMode.MIN_ERROR
            task.allow_min_error_fallback = True
        result = solver.solve_position_step(q, targets, opts)
        if _needs_min_error_recovery(result):
            result = solver.solve_position_step(q, _scaled_targets(targets, 0.35), opts)
        if _needs_min_error_recovery(result):
            result = solver.solve_position_step(q, _scaled_targets(targets, 0.15), opts)
    finally:
        for task, mode, fallback in zip(target_tasks, previous_modes, previous_fallbacks):
            task.solve_mode = mode
            task.allow_min_error_fallback = fallback
        _configure_interactive_elastic_band(solver)
    return result


def _solve_with_min_error_recovery(solver, q, targets, opts, target_tasks, prefer_min_error=False):
    if prefer_min_error:
        return _solve_targets_as_min_error(solver, q, targets, opts, target_tasks), False

    result = solver.solve_position_step(q, targets, opts)
    if not _needs_min_error_recovery(result):
        return result, False

    retry = _solve_targets_as_min_error(solver, q, targets, opts, target_tasks)
    if hasattr(retry, "q_solution") and retry.status != embodik.SolverStatus.NUMERICAL_ERROR:
        return retry, True
    return result, False


def _solve_quality_step(
    solver,
    robot,
    q,
    q_lo,
    q_hi,
    targets,
    opts,
    target_tasks,
    frame_targets,
    prefer_min_error=False,
    error_limit=0.03,
):
    q_current = np.asarray(q, dtype=float)
    robot.update_configuration(q_current)
    pre_error = _max_frame_target_position_error(robot, frame_targets)
    attempts = [
        (targets, opts, prefer_min_error, False),
        (_scaled_secondary_targets(targets, 0.35), opts, True, True),
        (_scaled_secondary_targets(targets, 0.0), opts, True, True),
    ]
    extended_opts = _clone_position_step_options(opts)
    extended_opts.max_steps = max(int(getattr(opts, "max_steps", 1)), 10)
    attempts.append((_scaled_secondary_targets(targets, 0.0), extended_opts, True, True))

    best = None
    best_error = float("inf")
    for attempt_targets, attempt_opts, attempt_min_error, recovered in attempts:
        result, min_error_recovered = _solve_with_min_error_recovery(
            solver,
            q_current,
            attempt_targets,
            attempt_opts,
            target_tasks,
            prefer_min_error=attempt_min_error,
        )
        if (
            not hasattr(result, "q_solution")
            or result.status == embodik.SolverStatus.COLLISION_VIOLATED
        ):
            candidate_error = float("inf")
        else:
            q_candidate = _clip_q(robot, np.asarray(result.q_solution, dtype=float), q_lo, q_hi)
            if np.all(np.isfinite(q_candidate)):
                robot.update_configuration(q_candidate)
                candidate_error = _max_frame_target_position_error(robot, frame_targets)
            else:
                candidate_error = float("inf")

        if candidate_error < best_error:
            best = (result, bool(recovered or min_error_recovered))
            best_error = candidate_error
        result_dq = np.asarray(getattr(result, "joint_velocities", []), dtype=float)
        result_moved = bool(result_dq.size and np.linalg.norm(result_dq) > 1e-10)
        result_productive = result.status == embodik.SolverStatus.SUCCESS or result_moved
        improved = candidate_error <= max(pre_error - 1e-4, pre_error * 0.98)
        already_accurate = candidate_error <= 0.005
        if already_accurate or (result_productive and (candidate_error <= error_limit or improved)):
            break

    robot.update_configuration(q_current)
    if best is None:
        return _solve_with_min_error_recovery(
            solver,
            q_current,
            targets,
            opts,
            target_tasks,
            prefer_min_error=prefer_min_error,
        )
    return best
