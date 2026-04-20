#!/usr/bin/env python3
"""Headless motion regressions for the incubating ROBOTIS AI worker example."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


embodik = pytest.importorskip("embodik")

from examples.incubating.example_helpers.robotis_ai_worker_utils import (  # noqa: E402
    default_worker_ik_joint_names,
    resolve_ai_worker_frames,
    resolve_ffw_urdf_path,
)


def _load_worker_robot():
    urdf = resolve_ffw_urdf_path("sg2")
    if not Path(urdf).is_file():
        pytest.skip("local FFW SG2 URDF not available")
    robot = embodik.RobotModel(str(urdf), floating_base=False)
    frames = resolve_ai_worker_frames(robot.get_frame_names())
    return robot, frames


def _frame_pose_matrix(robot, frame_name: str) -> np.ndarray:
    pose = robot.get_frame_pose(frame_name)
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(pose.rotation, dtype=float)
    T[:3, 3] = np.asarray(pose.translation, dtype=float)
    return T


def _build_locked_velocity_indices(robot) -> list[int]:
    joint_names = list(robot.get_joint_names())
    allowed_joint_names = set(default_worker_ik_joint_names(joint_names))
    locked: list[int] = []
    for joint_name in joint_names:
        if joint_name in allowed_joint_names:
            continue
        idx_v = int(robot.get_joint_velocity_index(joint_name))
        nv_joint = int(robot.get_joint_velocity_size(joint_name)) if hasattr(robot, "get_joint_velocity_size") else 1
        for offset in range(max(nv_joint, 1)):
            locked.append(idx_v + offset)
    return sorted(set(locked))


def test_reduced_worker_ik_joint_set_excludes_wheels_and_head() -> None:
    urdf = resolve_ffw_urdf_path("sg2")
    reduced = embodik.RobotModel(
        str(urdf),
        actuated_joint_names=default_worker_ik_joint_names(embodik.RobotModel(str(urdf), floating_base=False).get_joint_names()),
        floating_base=False,
    )
    joint_names = list(reduced.get_joint_names())
    assert "lift_joint" in joint_names
    assert any(name.startswith("arm_l_") for name in joint_names)
    assert any(name.startswith("arm_r_") for name in joint_names)
    assert not any("wheel_" in name for name in joint_names)
    assert not any(name.startswith("head_") for name in joint_names)
    assert not any(name.startswith("gripper_") for name in joint_names)


def _solve_single_step(
    robot,
    frames: dict[str, str],
    *,
    right_target_offset: np.ndarray,
    exclude_right: list[int] | None = None,
    exclude_left: list[int] | None = None,
) -> np.ndarray:
    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)
    solver.set_tolerance(0.1)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    right_task = solver.add_frame_task("right_tool_pose", frames["right_tool"], embodik.TaskType.FRAME_POSE)
    left_task = solver.add_frame_task("left_tool_pose", frames["left_tool"], embodik.TaskType.FRAME_POSE)
    for task in (right_task, left_task):
        task.priority = 0
        task.weight = 1.0
        task.solve_mode = embodik.TaskSolveMode.SCALE_ELASTIC

    if exclude_right:
        right_task.set_excluded_joint_indices(list(exclude_right))
    if exclude_left:
        left_task.set_excluded_joint_indices(list(exclude_left))

    q0 = robot.neutral_configuration()
    robot.update_configuration(q0)
    right0 = _frame_pose_matrix(robot, frames["right_tool"])
    left0 = _frame_pose_matrix(robot, frames["left_tool"])

    right_target = right0.copy()
    right_target[:3, 3] += np.asarray(right_target_offset, dtype=float)

    opts = embodik.PositionStepOptions()
    opts.max_steps = 1
    opts.position_gain = 10.0
    opts.orientation_gain = 10.0
    locked = _build_locked_velocity_indices(robot)
    opts.excluded_joint_indices = list(locked)
    opts.integration_zero_velocity_indices = list(locked)

    result = solver.solve_position_step(
        q0,
        [
            embodik.TaskTarget("right_tool_pose", right_target, 10.0, 10.0),
            embodik.TaskTarget("left_tool_pose", left0, 10.0, 10.0),
        ],
        opts,
    )
    return np.asarray(result.q_solution, dtype=float)


def test_direct_joint_configuration_moves_right_tool() -> None:
    robot, frames = _load_worker_robot()
    q0 = robot.neutral_configuration()
    robot.update_configuration(q0)
    p0 = _frame_pose_matrix(robot, frames["right_tool"])[:3, 3].copy()

    idx = int(robot.get_joint_config_index("arm_r_joint4"))
    q1 = q0.copy()
    q1[idx] += 0.2
    q_lo, q_hi = robot.get_joint_limits()
    q1 = np.clip(q1, q_lo, q_hi)

    robot.update_configuration(q1)
    p1 = _frame_pose_matrix(robot, frames["right_tool"])[:3, 3].copy()
    assert np.linalg.norm(p1 - p0) > 1e-4


def test_single_step_ik_moves_right_tool_toward_target() -> None:
    robot, frames = _load_worker_robot()
    q0 = robot.neutral_configuration()
    robot.update_configuration(q0)
    right0 = _frame_pose_matrix(robot, frames["right_tool"])
    q1 = _solve_single_step(robot, frames, right_target_offset=np.array([0.10, 0.0, 0.0], dtype=float))
    assert np.linalg.norm(q1 - q0) > 1e-6

    robot.update_configuration(q1)
    right1 = _frame_pose_matrix(robot, frames["right_tool"])
    right_target = right0.copy()
    right_target[:3, 3] += np.array([0.10, 0.0, 0.0], dtype=float)
    moved = np.linalg.norm(right1[:3, 3] - right0[:3, 3])
    residual0 = np.linalg.norm(right_target[:3, 3] - right0[:3, 3])
    residual1 = np.linalg.norm(right_target[:3, 3] - right1[:3, 3])

    assert moved > 1e-4
    assert residual1 < residual0


def test_unlocked_lift_can_participate_in_single_arm_solve() -> None:
    robot, frames = _load_worker_robot()
    joint_names = list(robot.get_joint_names())
    left_arm_vel = []
    lift_vel = []
    for joint_name in joint_names:
        idx_v = int(robot.get_joint_velocity_index(joint_name))
        nv_joint = int(robot.get_joint_velocity_size(joint_name)) if hasattr(robot, "get_joint_velocity_size") else 1
        expanded = [idx_v + offset for offset in range(max(nv_joint, 1))]
        if joint_name.startswith(("arm_l_", "gripper_l_")):
            left_arm_vel.extend(expanded)
        if joint_name.startswith("lift_"):
            lift_vel.extend(expanded)

    q_unlocked = _solve_single_step(
        robot,
        frames,
        right_target_offset=np.array([0.0, 0.0, 0.08], dtype=float),
        exclude_right=left_arm_vel,
        exclude_left=[],
    )
    q_locked = _solve_single_step(
        robot,
        frames,
        right_target_offset=np.array([0.0, 0.0, 0.08], dtype=float),
        exclude_right=left_arm_vel + lift_vel,
        exclude_left=lift_vel,
    )

    lift_idx = int(robot.get_joint_config_index("lift_joint"))
    lift_unlocked = float(q_unlocked[lift_idx])
    lift_locked = float(q_locked[lift_idx])
    assert abs(lift_unlocked) > 1e-5
    assert abs(lift_locked) <= 1e-9
