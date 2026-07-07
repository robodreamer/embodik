#!/usr/bin/env python3
"""Public constrained-ROM recovery analogs for private Alpha WBC rows."""

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
class RecoveryRobotCase:
    case_id: str
    preset: str
    limit_joint: str
    singularity_seed: dict[str, float]


ROBOT_CASES = (
    RecoveryRobotCase(
        "example-panda",
        "panda",
        "panda_joint4",
        {"panda_joint2": 0.02, "panda_joint4": -0.02, "panda_joint6": 0.02},
    ),
    RecoveryRobotCase(
        "example-iiwa",
        "iiwa",
        "iiwa_joint_2",
        {"iiwa_joint_2": 0.02, "iiwa_joint_4": -0.02, "iiwa_joint_6": 0.02},
    ),
)


@dataclass(frozen=True)
class RecoveryScenario:
    case_id: str
    kind: str
    steps: int
    return_start_step: int | None
    backstep_limit: int


SCENARIOS = (
    RecoveryScenario("far-then-reachable", "far_return", 34, 17, 20),
    RecoveryScenario("joint-limit-out-and-back", "joint_limit", 30, 15, 4),
    RecoveryScenario("singularity-perturbation", "singularity", 28, 14, 2),
)


def _load_robot(case: RecoveryRobotCase) -> tuple[eik.RobotModel, str, np.ndarray]:
    cfg = resolve_robot_configuration(case.preset)
    robot = cfg["robot"]
    q = np.asarray(cfg["default_configuration"], dtype=float)
    if q.size != robot.nq:
        q = np.asarray(robot.neutral_configuration(), dtype=float)
    lower, upper = robot.get_joint_limits()
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    q[finite_lower] = np.maximum(q[finite_lower], lower[finite_lower] + 1e-4)
    q[finite_upper] = np.minimum(q[finite_upper], upper[finite_upper] - 1e-4)
    robot.update_configuration(q)
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


def _singularity_proxy(robot: eik.RobotModel, frame_name: str) -> float:
    jac = np.asarray(robot.get_frame_jacobian(frame_name), dtype=float)[:3, :]
    singular_values = np.linalg.svd(jac, compute_uv=False)
    return float(singular_values[-1]) if singular_values.size else 0.0


def _configure_solver(robot: eik.RobotModel, frame_name: str) -> eik.KinematicsSolver:
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    cfg = eik.SolverRuntimeConfig()
    cfg.enable_auto_task_layout = False
    cfg.weighted_fallback_enabled = True
    solver.configure_runtime(cfg)
    task = solver.add_frame_task("target_pose", frame_name, eik.TaskType.FRAME_POSITION)
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = eik.TaskSolveMode.SCALE_ELASTIC
    if hasattr(task, "allow_min_error_fallback"):
        task.allow_min_error_fallback = True
    return solver


def _targets(
    robot: eik.RobotModel,
    frame_name: str,
    q: np.ndarray,
    robot_case: RecoveryRobotCase,
    scenario: RecoveryScenario,
) -> tuple[np.ndarray, list[np.ndarray]]:
    lower, upper = robot.get_joint_limits()
    if scenario.kind == "joint_limit":
        limit_q, _limit_v = _single_dof_joint_indices(robot, robot_case.limit_joint)
        q = q.copy()
        q[limit_q] = float(upper[limit_q] - 0.04)
        robot.update_configuration(q)
        initial_pose = _pose_matrix(robot, frame_name)
        q_inside = q.copy()
        q_inside[limit_q] -= 1e-4
        robot.update_configuration(q_inside)
        inside_pose = _pose_matrix(robot, frame_name)
        direction = (initial_pose[:3, 3] - inside_pose[:3, 3]) / 1e-4
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        robot.update_configuration(q)
        poses = []
        half = int(scenario.return_start_step or scenario.steps // 2)
        for step in range(scenario.steps):
            alpha = (
                (step + 1) / max(half, 1)
                if step < half
                else max(0.0, 1.0 - (step - half + 1) / max(scenario.steps - half, 1))
            )
            target = initial_pose.copy()
            target[:3, 3] += 0.18 * alpha * direction
            poses.append(target)
        return q, poses

    if scenario.kind == "singularity":
        q = q.copy()
        for joint_name, value in robot_case.singularity_seed.items():
            if robot.has_joint(joint_name):
                idx = int(robot.get_joint_config_index(joint_name))
                q[idx] = float(np.clip(value, lower[idx] + 1e-3, upper[idx] - 1e-3))
        robot.update_configuration(q)
        initial_pose = _pose_matrix(robot, frame_name)
        poses = []
        half = int(scenario.return_start_step or scenario.steps // 2)
        for step in range(scenario.steps):
            alpha = (
                (step + 1) / max(half, 1)
                if step < half
                else max(0.0, 1.0 - (step - half + 1) / max(scenario.steps - half, 1))
            )
            target = initial_pose.copy()
            target[:3, 3] += alpha * np.array([0.035, -0.015, 0.025], dtype=float)
            poses.append(target)
        return q, poses

    robot.update_configuration(q)
    initial_pose = _pose_matrix(robot, frame_name)
    poses = []
    delta = np.array([0.30, 0.10, 0.10], dtype=float)
    for step in range(scenario.steps):
        target = initial_pose.copy()
        target[:3, 3] += delta if step < int(scenario.return_start_step or 0) else 0.0
        poses.append(target)
    return q, poses


def _run_recovery_case(
    robot_case: RecoveryRobotCase,
    scenario: RecoveryScenario,
) -> dict[str, float | int | str | list[str]]:
    robot, frame_name, q = _load_robot(robot_case)
    q, poses = _targets(robot, frame_name, q, robot_case, scenario)
    robot.update_configuration(q)
    solver = _configure_solver(robot, frame_name)
    start_pose = _pose_matrix(robot, frame_name)
    progress_axis = poses[0][:3, 3] - start_pose[:3, 3]
    progress_axis /= max(float(np.linalg.norm(progress_axis)), 1e-12)

    q_trace = [q.copy()]
    errors: list[float] = []
    progress: list[float] = []
    target_progress: list[float] = []
    singularity: list[float] = []
    statuses: list[str] = []

    for target in poses:
        opts = eik.PositionStepOptions()
        opts.max_steps = 1
        opts.dt = 0.02
        opts.position_gain = 20.0
        opts.orientation_gain = 0.0
        opts.primary_solve_mode = eik.TaskSolveMode.SCALE_ELASTIC
        opts.primary_allow_min_error_fallback = True
        opts.max_configuration_step_norm = 0.08
        result = solver.solve_position_step(q, target, "target_pose", opts)
        statuses.append(getattr(result.status, "name", str(result.status)))
        q = np.asarray(result.q_solution, dtype=float)
        assert q.shape == q_trace[-1].shape
        assert np.all(np.isfinite(q))
        robot.update_configuration(q)
        q_trace.append(q.copy())
        current_pose = _pose_matrix(robot, frame_name)
        errors.append(float(np.linalg.norm(target[:3, 3] - current_pose[:3, 3])))
        progress.append(float(np.dot(current_pose[:3, 3] - start_pose[:3, 3], progress_axis)))
        target_progress.append(float(np.dot(target[:3, 3] - start_pose[:3, 3], progress_axis)))
        singularity.append(_singularity_proxy(robot, frame_name))

    q_arr = np.vstack(q_trace)
    q_steps = np.linalg.norm(np.diff(q_arr, axis=0), axis=1)
    q_accel = np.linalg.norm(np.diff(q_arr, n=2, axis=0), axis=1)
    q_jerk = np.linalg.norm(np.diff(q_arr, n=3, axis=0), axis=1)
    errors_arr = np.asarray(errors, dtype=float)
    progress_arr = np.asarray(progress, dtype=float)
    target_delta = np.diff(np.asarray(target_progress, dtype=float))
    stationary = np.abs(target_delta) <= 1e-6
    non_retracting = target_delta >= -1e-4
    zero_motion = (q_steps < 1e-5) & (errors_arr > 2e-2)
    max_zero_streak = 0
    streak = 0
    for value in zero_motion:
        streak = streak + 1 if bool(value) else 0
        max_zero_streak = max(max_zero_streak, streak)
    recovery_lag = 0
    if scenario.return_start_step is not None:
        recovery_lag = scenario.steps - scenario.return_start_step
        for idx in range(scenario.return_start_step, len(q_steps)):
            previous_error = errors_arr[idx - 1] if idx > 0 else float("inf")
            if q_steps[idx] > 1e-5 and errors_arr[idx] <= previous_error + 1e-4:
                recovery_lag = idx - scenario.return_start_step
                break

    lower, upper = robot.get_joint_limits()
    slack = np.minimum(q_arr - lower, upper - q_arr)
    return {
        "robot": robot_case.case_id,
        "scenario": scenario.case_id,
        "final_error_m": float(errors_arr[-1]),
        "error_increases": int(np.sum((np.diff(errors_arr) > 1e-4) & stationary)),
        "backsteps": int(np.sum((np.diff(progress_arr) < -1e-4) & non_retracting)),
        "max_step_norm": float(q_steps.max(initial=0.0)),
        "p95_step_norm": float(np.percentile(q_steps, 95.0)) if q_steps.size else 0.0,
        "max_accel_norm": float(q_accel.max(initial=0.0)),
        "max_jerk_norm": float(q_jerk.max(initial=0.0)),
        "limit_slack_min": float(np.min(slack)),
        "singularity_proxy_min": float(np.min(singularity)) if singularity else 0.0,
        "max_zero_motion_streak": int(max_zero_streak),
        "recovery_lag_steps": int(recovery_lag),
        "statuses": sorted(set(statuses)),
    }


@pytest.mark.benchmark
@pytest.mark.parametrize("robot_case", ROBOT_CASES, ids=[case.case_id for case in ROBOT_CASES])
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[scenario.case_id for scenario in SCENARIOS])
def test_public_constrained_recovery_continuity_matrix(
    robot_case: RecoveryRobotCase,
    scenario: RecoveryScenario,
    record_property: Callable[[str, object], None],
) -> None:
    metrics = _run_recovery_case(robot_case, scenario)
    for key, value in metrics.items():
        record_property(key, value)

    assert "NUMERICAL_ERROR" not in metrics["statuses"], metrics
    assert float(metrics["limit_slack_min"]) >= -1e-8, metrics
    assert float(metrics["max_step_norm"]) <= 0.08 + 1e-9, metrics
    assert float(metrics["max_accel_norm"]) <= 0.16, metrics
    assert float(metrics["max_jerk_norm"]) <= 0.24, metrics
    assert int(metrics["max_zero_motion_streak"]) <= 2, metrics
    assert int(metrics["recovery_lag_steps"]) <= 6, metrics
    assert int(metrics["error_increases"]) <= 2, metrics
    assert int(metrics["backsteps"]) <= scenario.backstep_limit, metrics
    assert float(metrics["singularity_proxy_min"]) >= 0.0, metrics
