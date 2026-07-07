#!/usr/bin/env python3
"""Matrix benchmark for infeasible-target hold behavior across example robots."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pytest

import embodik as eik

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from example_helpers.common_bimanual_model_utils import (  # noqa: E402
    default_common_bimanual_ik_joint_names,
    resolve_common_bimanual_frames,
)
from example_helpers.g1_model_utils import (  # noqa: E402
    create_g1_robot_model,
    resolve_frames_for_g1_base_mode,
)
from example_helpers.public_ai_worker_paths import (  # noqa: E402
    resolve_public_ai_worker_urdf_paths,
)
from example_helpers.spot_whole_body_ik import (  # noqa: E402
    TOOL_FRAME_CANDIDATES,
    resolve_spot_ik_urdf,
)
from utils.robot_models import resolve_robot_configuration  # noqa: E402


@dataclass(frozen=True)
class RobotMatrixCase:
    case_id: str
    load: Callable[[], tuple[eik.RobotModel, str, np.ndarray]]
    primary_joint: str
    posture_joint: str


def _load_example_preset(key: str) -> tuple[eik.RobotModel, str, np.ndarray]:
    cfg = resolve_robot_configuration(key)
    return (
        cfg["robot"],
        str(cfg["target_link"]),
        np.asarray(cfg["default_configuration"], dtype=float),
    )


def _load_ai_worker_sg2() -> tuple[eik.RobotModel, str, np.ndarray]:
    urdf, collision_urdf = resolve_public_ai_worker_urdf_paths(variant="sg2")
    model_urdf = collision_urdf or urdf
    full_robot = eik.RobotModel(str(model_urdf), floating_base=False)
    robot = eik.RobotModel(
        str(model_urdf),
        actuated_joint_names=default_common_bimanual_ik_joint_names(full_robot.get_joint_names()),
        floating_base=False,
    )
    frames = resolve_common_bimanual_frames(robot.get_frame_names())
    return robot, frames["right_tool"], np.asarray(robot.neutral_configuration(), dtype=float)


def _load_g1() -> tuple[eik.RobotModel, str, np.ndarray]:
    robot = create_g1_robot_model(floating_base=False, reduced_ik=True)
    frames = resolve_frames_for_g1_base_mode(robot.get_frame_names())
    return robot, frames["right_palm"], np.asarray(robot.neutral_configuration(), dtype=float)


def _load_spot() -> tuple[eik.RobotModel, str, np.ndarray]:
    urdf = resolve_spot_ik_urdf()
    if urdf is None:
        pytest.skip("Spot whole-body URDF not available")
    robot = eik.RobotModel(str(urdf), floating_base=False)
    frame_names = set(robot.get_frame_names())
    frame = next(name for name in TOOL_FRAME_CANDIDATES if name in frame_names)
    return robot, frame, np.asarray(robot.neutral_configuration(), dtype=float)


def _load_rby1() -> tuple[eik.RobotModel, str, np.ndarray]:
    rby1_description = pytest.importorskip("robot_descriptions.rby1_description")
    full_robot = eik.RobotModel(str(rby1_description.URDF_PATH), floating_base=False)
    robot = eik.RobotModel(
        str(rby1_description.URDF_PATH),
        actuated_joint_names=default_common_bimanual_ik_joint_names(full_robot.get_joint_names()),
        floating_base=False,
    )
    frames = resolve_common_bimanual_frames(robot.get_frame_names())
    return robot, frames["right_tool"], np.asarray(robot.neutral_configuration(), dtype=float)


MATRIX_CASES = (
    RobotMatrixCase(
        "example-panda",
        lambda: _load_example_preset("panda"),
        "panda_joint4",
        "panda_joint1",
    ),
    RobotMatrixCase(
        "example-iiwa",
        lambda: _load_example_preset("iiwa"),
        "iiwa_joint_2",
        "iiwa_joint_1",
    ),
    RobotMatrixCase(
        "example-ai-worker-sg2",
        _load_ai_worker_sg2,
        "lift_joint",
        "arm_l_joint1",
    ),
    RobotMatrixCase(
        "example-g1",
        _load_g1,
        "right_shoulder_pitch_joint",
        "left_hip_pitch_joint",
    ),
    RobotMatrixCase(
        "example-spot",
        _load_spot,
        "arm0_sh1",
        "arm0_sh0",
    ),
    RobotMatrixCase(
        "example-rby1",
        _load_rby1,
        "right_arm_1",
        "torso_0",
    ),
)


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


def _disable_weighted_fallback(solver: eik.KinematicsSolver) -> None:
    cfg = eik.SolverRuntimeConfig()
    cfg.weighted_fallback_enabled = False
    cfg.enable_auto_task_layout = False
    solver.configure_runtime(cfg)


def _make_soft_infeasible_matrix_step(
    case: RobotMatrixCase,
) -> tuple[eik.RobotModel, eik.KinematicsSolver, np.ndarray, np.ndarray, eik.PositionStepOptions]:
    robot, frame_name, q_seed = case.load()
    assert robot.has_frame(frame_name), f"matrix frame not found: {frame_name}"

    q = np.asarray(q_seed, dtype=float).copy()
    if q.size != robot.nq:
        q = np.asarray(robot.neutral_configuration(), dtype=float)
    q_lower, q_upper = robot.get_joint_limits()
    primary_q, primary_v = _single_dof_joint_indices(robot, case.primary_joint)
    posture_q, posture_v = _single_dof_joint_indices(robot, case.posture_joint)

    for q_index in (primary_q, posture_q):
        assert np.isfinite(q_lower[q_index])
        assert np.isfinite(q_upper[q_index])
    q[posture_q] = float(np.clip(q[posture_q], q_lower[posture_q] + 0.2, q_upper[posture_q] - 0.2))
    q[primary_q] = float(q_upper[primary_q] - 1e-3)

    robot.update_configuration(q)
    current_pose = _pose_matrix(robot, frame_name)
    q_inside = q.copy()
    q_inside[primary_q] -= 1e-4
    robot.update_configuration(q_inside)
    inside_pose = _pose_matrix(robot, frame_name)
    direction = (current_pose[:3, 3] - inside_pose[:3, 3]) / (q[primary_q] - q_inside[primary_q])
    direction_norm = float(np.linalg.norm(direction))
    assert direction_norm > 1e-8

    target_pose = current_pose.copy()
    target_pose[:3, 3] += 0.75 * direction / direction_norm

    robot.update_configuration(q)
    solver = eik.KinematicsSolver(robot)
    _disable_weighted_fallback(solver)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    task = solver.add_frame_task("target_pose", frame_name, eik.TaskType.FRAME_POSE)
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = eik.TaskSolveMode.SCALE
    task.set_excluded_joint_indices([idx for idx in range(robot.nv) if idx != primary_v])

    posture = solver.add_posture_task("posture")
    posture.priority = 1
    posture.weight = 10.0
    posture_velocity = np.zeros(robot.nv, dtype=float)
    posture_velocity[posture_v] = 10.0
    posture.set_target_velocity(posture_velocity)

    opts = eik.PositionStepOptions()
    opts.max_steps = 1
    opts.dt = 0.02
    opts.position_gain = 8000.0
    opts.orientation_gain = 0.0
    opts.primary_solve_mode = eik.TaskSolveMode.SCALE
    opts.primary_allow_min_error_fallback = False
    return robot, solver, q, target_pose, opts


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "case",
    MATRIX_CASES,
    ids=[case.case_id for case in MATRIX_CASES],
)
def test_soft_infeasible_target_hold_matrix(
    case: RobotMatrixCase, record_property: Callable[[str, object], None]
) -> None:
    robot, frame_name, q_seed = case.load()
    assert robot.has_frame(frame_name), f"matrix frame not found: {frame_name}"

    q = np.asarray(q_seed, dtype=float).copy()
    if q.size != robot.nq:
        q = np.asarray(robot.neutral_configuration(), dtype=float)
    q_lower, q_upper = robot.get_joint_limits()
    primary_q, primary_v = _single_dof_joint_indices(robot, case.primary_joint)
    posture_q, posture_v = _single_dof_joint_indices(robot, case.posture_joint)

    for q_index in (primary_q, posture_q):
        assert np.isfinite(q_lower[q_index])
        assert np.isfinite(q_upper[q_index])
    q[posture_q] = float(np.clip(q[posture_q], q_lower[posture_q] + 0.2, q_upper[posture_q] - 0.2))
    q[primary_q] = float(q_upper[primary_q] - 1e-3)

    robot.update_configuration(q)
    current_pose = _pose_matrix(robot, frame_name)
    q_inside = q.copy()
    q_inside[primary_q] -= 1e-4
    robot.update_configuration(q_inside)
    inside_pose = _pose_matrix(robot, frame_name)
    direction = (current_pose[:3, 3] - inside_pose[:3, 3]) / (q[primary_q] - q_inside[primary_q])
    direction_norm = float(np.linalg.norm(direction))
    assert direction_norm > 1e-8

    target_pose = current_pose.copy()
    target_pose[:3, 3] += 0.75 * direction / direction_norm

    robot.update_configuration(q)
    solver = eik.KinematicsSolver(robot)
    _disable_weighted_fallback(solver)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    task = solver.add_frame_task("target_pose", frame_name, eik.TaskType.FRAME_POSE)
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = eik.TaskSolveMode.SCALE
    task.set_excluded_joint_indices([idx for idx in range(robot.nv) if idx != primary_v])

    posture = solver.add_posture_task("posture")
    posture.priority = 1
    posture.weight = 10.0
    posture_velocity = np.zeros(robot.nv, dtype=float)
    posture_velocity[posture_v] = 10.0
    posture.set_target_velocity(posture_velocity)

    opts = eik.PositionStepOptions()
    opts.max_steps = 1
    opts.dt = 0.02
    opts.position_gain = 8000.0
    opts.orientation_gain = 0.0
    opts.primary_solve_mode = eik.TaskSolveMode.SCALE
    opts.primary_allow_min_error_fallback = False

    started = perf_counter()
    result = solver.solve_position_step(q, target_pose, "target_pose", opts)
    elapsed_ms = (perf_counter() - started) * 1e3

    q_solution = np.asarray(result.q_solution, dtype=float)
    task_scale = float(np.asarray(result.task_scales, dtype=float)[0])
    record_property("case_id", case.case_id)
    record_property("target_frame", frame_name)
    record_property("primary_joint", case.primary_joint)
    record_property("posture_joint", case.posture_joint)
    record_property("task_scale", task_scale)
    record_property("position_error", float(result.position_error))
    record_property("solve_ms", elapsed_ms)

    assert result.status == eik.SolverStatus.NO_PROGRESS
    assert "held current configuration" in result.status_message
    assert result.task_modes_effective[0] == eik.TaskSolveMode.SCALE
    assert not result.task_used_fallback[0]
    assert abs(task_scale) <= 1e-4
    assert result.position_error > 0.05
    np.testing.assert_allclose(q_solution, q, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(result.joint_velocities, dtype=float),
        np.zeros(robot.nv, dtype=float),
        atol=1e-12,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "case",
    MATRIX_CASES,
    ids=[case.case_id for case in MATRIX_CASES],
)
def test_stationary_far_target_stays_held_after_soft_infeasible_hold(
    case: RobotMatrixCase,
) -> None:
    robot, solver, q, target_pose, opts = _make_soft_infeasible_matrix_step(case)

    first = solver.solve_position_step(q, target_pose, "target_pose", opts)
    assert first.status == eik.SolverStatus.NO_PROGRESS
    assert "held current configuration" in first.status_message
    np.testing.assert_allclose(np.asarray(first.q_solution, dtype=float), q, atol=1e-12)

    q_stationary = q.copy()
    q_trace = [q_stationary.copy()]
    statuses: list[str] = []
    for _ in range(12):
        result = solver.solve_position_step(q_stationary, target_pose, "target_pose", opts)
        q_next = np.asarray(result.q_solution, dtype=float)
        assert q_next.shape == q_stationary.shape
        assert np.all(np.isfinite(q_next))
        q_stationary = q_next
        robot.update_configuration(q_stationary)
        q_trace.append(q_stationary.copy())
        statuses.append(result.status.name)

    q_arr = np.vstack(q_trace)
    q_steps = np.linalg.norm(np.diff(q_arr, axis=0), axis=1)
    assert "NUMERICAL_ERROR" not in statuses
    assert float(q_steps.max(initial=0.0)) <= 1e-2
    assert float(q_steps.sum()) <= 1e-2
