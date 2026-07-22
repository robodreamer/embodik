#!/usr/bin/env python3
"""Mode-agnostic smoothness matrix for public position-step IK examples."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

import embodik as eik

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from utils.robot_models import resolve_robot_configuration  # noqa: E402


@dataclass(frozen=True)
class PublicRobotCase:
    case_id: str
    preset: str
    limit_joint: str


@dataclass(frozen=True)
class SmoothModeCase:
    case_id: str
    solve_mode: eik.TaskSolveMode
    weighted_advisor: bool = False


ROBOT_CASES = (
    PublicRobotCase("example-panda", "panda", "panda_joint4"),
    PublicRobotCase("example-iiwa", "iiwa", "iiwa_joint_2"),
)

MODE_CASES = (
    SmoothModeCase("scale", eik.TaskSolveMode.SCALE),
    SmoothModeCase("scale-elastic", eik.TaskSolveMode.SCALE_ELASTIC),
    SmoothModeCase("min-error", eik.TaskSolveMode.MIN_ERROR),
    SmoothModeCase("weighted-advisor", eik.TaskSolveMode.SCALE_ELASTIC, True),
)


def _load_public_robot(case: PublicRobotCase) -> tuple[eik.RobotModel, str, np.ndarray]:
    cfg = resolve_robot_configuration(case.preset)
    robot = cfg["robot"]
    q = np.asarray(cfg["default_configuration"], dtype=float)
    if q.size != robot.nq:
        q = np.asarray(robot.neutral_configuration(), dtype=float)
    q_lower, q_upper = robot.get_joint_limits()
    finite_lower = np.isfinite(q_lower)
    finite_upper = np.isfinite(q_upper)
    q[finite_lower] = np.maximum(q[finite_lower], q_lower[finite_lower])
    q[finite_upper] = np.minimum(q[finite_upper], q_upper[finite_upper])
    finite_both = finite_lower & finite_upper
    interior_margin = np.minimum(1e-4, 0.1 * np.maximum(q_upper - q_lower, 0.0))
    q[finite_both] = np.maximum(q[finite_both], q_lower[finite_both] + interior_margin[finite_both])
    q[finite_both] = np.minimum(q[finite_both], q_upper[finite_both] - interior_margin[finite_both])
    return robot, str(cfg["target_link"]), q


def _pose_matrix(robot: eik.RobotModel, frame_name: str) -> np.ndarray:
    pose = robot.get_frame_pose(frame_name)
    out = np.eye(4, dtype=float)
    out[:3, :3] = np.asarray(pose.rotation, dtype=float)
    out[:3, 3] = np.asarray(pose.translation, dtype=float)
    return out


def _single_dof_joint_indices(robot: eik.RobotModel, joint_name: str) -> tuple[int, int]:
    assert robot.has_joint(joint_name), f"matrix joint not found: {joint_name}"
    assert int(robot.get_joint_velocity_size(joint_name)) == 1
    return int(robot.get_joint_config_index(joint_name)), int(
        robot.get_joint_velocity_index(joint_name)
    )


def _drive_target_step_with_configuration_cap(
    robot_case: PublicRobotCase,
    mode_case: SmoothModeCase,
    *,
    max_step_norm: float,
) -> dict[str, float | int | str | bool]:
    robot, frame_name, q = _load_public_robot(robot_case)
    robot.update_configuration(q)
    initial_pose = _pose_matrix(robot, frame_name)
    target_pose = initial_pose.copy()
    target_pose[:3, 3] += np.array([0.35, 0.10, 0.10], dtype=float)
    initial_error = float(np.linalg.norm(target_pose[:3, 3] - initial_pose[:3, 3]))

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    cfg = eik.SolverRuntimeConfig()
    cfg.weighted_advisor_enabled = bool(mode_case.weighted_advisor)
    cfg.weighted_fallback_enabled = bool(mode_case.weighted_advisor)
    cfg.enable_auto_task_layout = False
    solver.configure_runtime(cfg)

    task = solver.add_frame_task("target_pose", frame_name, eik.TaskType.FRAME_POSE)
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = mode_case.solve_mode

    opts = eik.PositionStepOptions()
    assert hasattr(opts, "max_configuration_step_norm")
    opts.max_steps = 4
    opts.dt = 0.02
    opts.position_gain = 80.0
    opts.orientation_gain = 0.0
    opts.primary_solve_mode = mode_case.solve_mode
    opts.primary_allow_min_error_fallback = bool(mode_case.weighted_advisor)
    opts.max_configuration_step_norm = float(max_step_norm)

    result = solver.solve_position_step(q, target_pose, "target_pose", opts)
    q_solution = np.asarray(result.q_solution, dtype=float)
    step_norm = float(np.linalg.norm(q_solution - q))
    return {
        "robot": robot_case.case_id,
        "mode": mode_case.case_id,
        "status": getattr(result.status, "name", str(result.status)),
        "step_norm": step_norm,
        "final_error": float(result.position_error),
        "initial_error": initial_error,
        "iterations_used": int(result.iterations_used),
        "weighted_advisory_available": bool(getattr(result, "weighted_advisory_available", False)),
        "weighted_fallback_used": bool(getattr(result, "weighted_fallback_used", False)),
    }


def _drive_joint_limit_ramp_with_configuration_cap(
    robot_case: PublicRobotCase,
    mode_case: SmoothModeCase,
    *,
    max_step_norm: float,
) -> dict[str, float | int | str | bool | list[str]]:
    robot, frame_name, q = _load_public_robot(robot_case)
    q_lower, q_upper = robot.get_joint_limits()
    limit_q, limit_v = _single_dof_joint_indices(robot, robot_case.limit_joint)
    assert np.isfinite(q_lower[limit_q])
    assert np.isfinite(q_upper[limit_q])

    q[limit_q] = float(q_upper[limit_q] - 0.04)
    robot.update_configuration(q)
    initial_pose = _pose_matrix(robot, frame_name)

    q_inside = q.copy()
    q_inside[limit_q] -= 1e-4
    robot.update_configuration(q_inside)
    inside_pose = _pose_matrix(robot, frame_name)
    direction = (initial_pose[:3, 3] - inside_pose[:3, 3]) / 1e-4
    direction_norm = float(np.linalg.norm(direction))
    assert direction_norm > 1e-8
    direction /= direction_norm

    robot.update_configuration(q)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    cfg = eik.SolverRuntimeConfig()
    cfg.weighted_advisor_enabled = bool(mode_case.weighted_advisor)
    cfg.weighted_fallback_enabled = bool(mode_case.weighted_advisor)
    cfg.enable_auto_task_layout = False
    solver.configure_runtime(cfg)

    task = solver.add_frame_task("target_pose", frame_name, eik.TaskType.FRAME_POSITION)
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = mode_case.solve_mode
    task.set_excluded_joint_indices([idx for idx in range(robot.nv) if idx != limit_v])

    q_trace = [q.copy()]
    target_progress: list[float] = []
    ee_progress: list[float] = []
    statuses: list[str] = []
    weighted_advisory_seen = False

    for offset_magnitude in np.linspace(0.005, 0.24, 20):
        target_pose = initial_pose.copy()
        target_pose[:3, 3] += offset_magnitude * direction

        opts = eik.PositionStepOptions()
        opts.max_steps = 2
        opts.dt = 0.02
        opts.position_gain = 60.0
        opts.orientation_gain = 0.0
        opts.primary_solve_mode = mode_case.solve_mode
        opts.primary_allow_min_error_fallback = bool(mode_case.weighted_advisor)
        opts.max_configuration_step_norm = float(max_step_norm)

        result = solver.solve_position_step(q, target_pose, "target_pose", opts)
        statuses.append(getattr(result.status, "name", str(result.status)))
        weighted_advisory_seen = weighted_advisory_seen or bool(
            getattr(result, "weighted_advisory_available", False)
        )
        q_next = np.asarray(result.q_solution, dtype=float)
        assert q_next.shape == q.shape
        assert np.all(np.isfinite(q_next))
        q = q_next
        robot.update_configuration(q)
        current_pose = _pose_matrix(robot, frame_name)
        q_trace.append(q.copy())
        target_progress.append(float(offset_magnitude))
        ee_progress.append(float(np.dot(current_pose[:3, 3] - initial_pose[:3, 3], direction)))

    q_arr = np.vstack(q_trace)
    q_steps = np.asarray(
        [
            np.linalg.norm(robot.difference(q_arr[idx], q_arr[idx + 1]))
            for idx in range(len(q_arr) - 1)
        ],
        dtype=float,
    )
    q_accel = np.linalg.norm(np.diff(q_arr, n=2, axis=0), axis=1)
    q_jerk = np.linalg.norm(np.diff(q_arr, n=3, axis=0), axis=1)
    ee_progress_arr = np.asarray(ee_progress, dtype=float)
    limit_positions = q_arr[:, limit_q]
    return {
        "robot": robot_case.case_id,
        "mode": mode_case.case_id,
        "limit_joint": robot_case.limit_joint,
        "statuses": sorted(set(statuses)),
        "max_step_norm": float(q_steps.max()) if q_steps.size else 0.0,
        "p95_step_norm": float(np.percentile(q_steps, 95.0)) if q_steps.size else 0.0,
        "max_accel_norm": float(q_accel.max()) if q_accel.size else 0.0,
        "max_jerk_norm": float(q_jerk.max()) if q_jerk.size else 0.0,
        "total_motion_norm": float(np.sum(q_steps)),
        "backsteps": int(np.sum(np.diff(ee_progress_arr) < -1e-4)),
        "final_target_progress": float(target_progress[-1]),
        "final_ee_progress": float(ee_progress_arr[-1]) if ee_progress_arr.size else 0.0,
        "min_limit_slack": float(np.min(q_upper[limit_q] - limit_positions)),
        "weighted_advisory_available": weighted_advisory_seen,
    }


def _drive_target_sequence_with_configuration_cap(
    robot_case: PublicRobotCase,
    mode_case: SmoothModeCase,
    *,
    target_offsets: list[np.ndarray],
    max_step_norm: float,
) -> dict[str, float | int | str | bool | list[str]]:
    robot, frame_name, q = _load_public_robot(robot_case)
    robot.update_configuration(q)
    initial_pose = _pose_matrix(robot, frame_name)
    final_offset = np.asarray(target_offsets[-1], dtype=float)
    progress_axis = final_offset / max(float(np.linalg.norm(final_offset)), 1e-12)

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    cfg = eik.SolverRuntimeConfig()
    cfg.weighted_advisor_enabled = bool(mode_case.weighted_advisor)
    cfg.weighted_fallback_enabled = bool(mode_case.weighted_advisor)
    cfg.enable_auto_task_layout = False
    solver.configure_runtime(cfg)

    task = solver.add_frame_task("target_pose", frame_name, eik.TaskType.FRAME_POSE)
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = mode_case.solve_mode

    q_trace = [q.copy()]
    errors: list[float] = []
    progress: list[float] = []
    statuses: list[str] = []
    weighted_advisory_seen = False

    for offset in target_offsets:
        target_pose = initial_pose.copy()
        target_pose[:3, 3] += np.asarray(offset, dtype=float)
        opts = eik.PositionStepOptions()
        opts.max_steps = 2
        opts.dt = 0.02
        opts.position_gain = 60.0
        opts.orientation_gain = 0.0
        opts.primary_solve_mode = mode_case.solve_mode
        opts.primary_allow_min_error_fallback = bool(mode_case.weighted_advisor)
        opts.max_configuration_step_norm = float(max_step_norm)

        result = solver.solve_position_step(q, target_pose, "target_pose", opts)
        statuses.append(getattr(result.status, "name", str(result.status)))
        weighted_advisory_seen = weighted_advisory_seen or bool(
            getattr(result, "weighted_advisory_available", False)
        )
        q_next = np.asarray(result.q_solution, dtype=float)
        assert q_next.shape == q.shape
        assert np.all(np.isfinite(q_next))
        q = q_next
        robot.update_configuration(q)
        q_trace.append(q.copy())
        current_pose = _pose_matrix(robot, frame_name)
        errors.append(float(np.linalg.norm(target_pose[:3, 3] - current_pose[:3, 3])))
        progress.append(float(np.dot(current_pose[:3, 3] - initial_pose[:3, 3], progress_axis)))

    q_arr = np.vstack(q_trace)
    q_steps = np.linalg.norm(np.diff(q_arr, axis=0), axis=1)
    q_accel = np.linalg.norm(np.diff(q_arr, n=2, axis=0), axis=1)
    q_jerk = np.linalg.norm(np.diff(q_arr, n=3, axis=0), axis=1)
    error_arr = np.asarray(errors, dtype=float)
    progress_arr = np.asarray(progress, dtype=float)
    return {
        "robot": robot_case.case_id,
        "mode": mode_case.case_id,
        "statuses": sorted(set(statuses)),
        "final_error": float(error_arr[-1]) if error_arr.size else 0.0,
        "error_increases": int(np.sum(np.diff(error_arr) > 1e-4)),
        "backsteps": int(np.sum(np.diff(progress_arr) < -1e-4)),
        "max_step_norm": float(q_steps.max()) if q_steps.size else 0.0,
        "p95_step_norm": float(np.percentile(q_steps, 95.0)) if q_steps.size else 0.0,
        "max_accel_norm": float(q_accel.max()) if q_accel.size else 0.0,
        "max_jerk_norm": float(q_jerk.max()) if q_jerk.size else 0.0,
        "total_motion_norm": float(np.sum(q_steps)),
        "stationary_drift_norm": float(np.linalg.norm(q_arr[-1] - q_arr[0])),
        "weighted_advisory_available": weighted_advisory_seen,
    }


@pytest.mark.benchmark
@pytest.mark.parametrize("robot_case", ROBOT_CASES, ids=[case.case_id for case in ROBOT_CASES])
@pytest.mark.parametrize("mode_case", MODE_CASES, ids=[case.case_id for case in MODE_CASES])
def test_target_step_configuration_delta_is_bounded_across_public_modes(
    robot_case: PublicRobotCase,
    mode_case: SmoothModeCase,
    record_property: Callable[[str, object], None],
) -> None:
    max_step_norm = 0.08

    metrics = _drive_target_step_with_configuration_cap(
        robot_case, mode_case, max_step_norm=max_step_norm
    )
    for key, value in metrics.items():
        record_property(key, value)

    assert metrics["status"] in {"SUCCESS", "NO_PROGRESS", "INFEASIBLE"}, metrics
    assert float(metrics["step_norm"]) <= max_step_norm + 1e-9, metrics
    assert float(metrics["final_error"]) <= float(metrics["initial_error"]) + 1e-9, metrics
    if mode_case.weighted_advisor:
        assert metrics["weighted_advisory_available"], metrics


@pytest.mark.benchmark
@pytest.mark.parametrize("robot_case", ROBOT_CASES, ids=[case.case_id for case in ROBOT_CASES])
@pytest.mark.parametrize("mode_case", MODE_CASES, ids=[case.case_id for case in MODE_CASES])
def test_stationary_reachable_target_has_no_self_motion_across_public_modes(
    robot_case: PublicRobotCase,
    mode_case: SmoothModeCase,
    record_property: Callable[[str, object], None],
) -> None:
    metrics = _drive_target_sequence_with_configuration_cap(
        robot_case,
        mode_case,
        target_offsets=[np.zeros(3, dtype=float) for _ in range(6)],
        max_step_norm=0.08,
    )
    for key, value in metrics.items():
        record_property(key, value)

    assert float(metrics["stationary_drift_norm"]) <= 1e-9, metrics
    assert float(metrics["total_motion_norm"]) <= 1e-9, metrics
    assert int(metrics["error_increases"]) == 0, metrics
    if mode_case.weighted_advisor:
        assert metrics["weighted_advisory_available"], metrics


@pytest.mark.benchmark
@pytest.mark.parametrize("robot_case", ROBOT_CASES, ids=[case.case_id for case in ROBOT_CASES])
@pytest.mark.parametrize("mode_case", MODE_CASES, ids=[case.case_id for case in MODE_CASES])
def test_slow_target_ramp_stays_bounded_across_public_modes(
    robot_case: PublicRobotCase,
    mode_case: SmoothModeCase,
    record_property: Callable[[str, object], None],
) -> None:
    final_offset = np.array([0.18, 0.05, 0.04], dtype=float)
    offsets = [final_offset * (idx + 1) / 18.0 for idx in range(18)]
    max_step_norm = 0.08

    metrics = _drive_target_sequence_with_configuration_cap(
        robot_case,
        mode_case,
        target_offsets=offsets,
        max_step_norm=max_step_norm,
    )
    for key, value in metrics.items():
        record_property(key, value)

    assert float(metrics["max_step_norm"]) <= max_step_norm + 1e-9, metrics
    assert int(metrics["backsteps"]) == 0, metrics
    assert float(metrics["max_accel_norm"]) <= max_step_norm, metrics
    assert float(metrics["max_jerk_norm"]) <= 2.0 * max_step_norm, metrics
    if mode_case.weighted_advisor:
        assert metrics["weighted_advisory_available"], metrics


@pytest.mark.benchmark
@pytest.mark.parametrize("robot_case", ROBOT_CASES, ids=[case.case_id for case in ROBOT_CASES])
@pytest.mark.parametrize("mode_case", MODE_CASES, ids=[case.case_id for case in MODE_CASES])
def test_slow_target_ramp_across_joint_limit_stays_bounded_across_public_modes(
    robot_case: PublicRobotCase,
    mode_case: SmoothModeCase,
    record_property: Callable[[str, object], None],
) -> None:
    max_step_norm = 0.08

    metrics = _drive_joint_limit_ramp_with_configuration_cap(
        robot_case,
        mode_case,
        max_step_norm=max_step_norm,
    )
    for key, value in metrics.items():
        record_property(key, value)

    assert "NUMERICAL_ERROR" not in metrics["statuses"], metrics
    assert float(metrics["max_step_norm"]) <= max_step_norm + 1e-9, metrics
    assert float(metrics["min_limit_slack"]) >= -1e-9, metrics
    assert int(metrics["backsteps"]) <= 1, metrics
    assert float(metrics["max_accel_norm"]) <= 2.0 * max_step_norm, metrics
    assert float(metrics["max_jerk_norm"]) <= 3.0 * max_step_norm, metrics
    if mode_case.weighted_advisor:
        assert metrics["weighted_advisory_available"], metrics
