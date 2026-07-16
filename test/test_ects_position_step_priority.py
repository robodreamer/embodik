"""Position-step regressions for nonlinear ECTS task priority."""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

import embodik as eik

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from utils.dual_iiwa_urdf import (  # noqa: E402
    build_dual_iiwa_urdf,
    get_dual_iiwa_default_configuration,
    get_dual_iiwa_frame_names,
)


def _rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _pose_matrix(pose: object) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.asarray(pose.rotation, dtype=float)
    matrix[:3, 3] = np.asarray(pose.translation, dtype=float)
    return matrix


@pytest.mark.parametrize("acceleration_limits_enabled", [False, True])
@pytest.mark.parametrize("relative_target_first", [False, True])
def test_lower_priority_rigid_motion_preserves_satisfied_relative_grasp(
    relative_target_first: bool,
    acceleration_limits_enabled: bool,
) -> None:
    """A reachable rigid tote command must not break its priority-0 grasp."""
    with tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False) as urdf_file:
        urdf_file.write(build_dual_iiwa_urdf())
        urdf_path = urdf_file.name
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
    finally:
        os.unlink(urdf_path)

    q0 = np.asarray(get_dual_iiwa_default_configuration(), dtype=float)
    robot.update_configuration(q0)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    if acceleration_limits_enabled:
        solver.set_acceleration_limits(np.full(robot.nv, 100.0, dtype=float))
        solver.enable_acceleration_limits(True)

    left_frame, right_frame = get_dual_iiwa_frame_names()
    absolute_task = solver.add_absolute_frame_task("tote_absolute", left_frame, right_frame, 0.5)
    relative_task = solver.add_relative_frame_task("tote_relative", left_frame, right_frame)
    relative_task.priority = 0
    absolute_task.priority = 1

    left_start = robot.get_frame_pose(left_frame)
    right_start = robot.get_frame_pose(right_frame)
    absolute_task.update(robot)
    absolute_task.calibrate_grasp_offsets(_pose_matrix(left_start), _pose_matrix(right_start))
    relative_task.update(robot)
    relative_task.capture_current_as_target()
    relative_target = eik.SE3(
        rotation=np.asarray(relative_task.current_orientation, dtype=float),
        translation=np.asarray(relative_task.current_position, dtype=float),
    )

    rotation_delta = _rotation_from_rpy(math.radians(12.0), math.radians(12.0), math.radians(18.0))
    translation_delta = np.array([0.045, 0.045, 0.045], dtype=float)
    midpoint = 0.5 * (
        np.asarray(left_start.translation, dtype=float)
        + np.asarray(right_start.translation, dtype=float)
    )

    def rigid_target(start_pose: object) -> eik.SE3:
        start_position = np.asarray(start_pose.translation, dtype=float)
        return eik.SE3(
            rotation=rotation_delta @ np.asarray(start_pose.rotation, dtype=float),
            translation=(
                midpoint + translation_delta + rotation_delta @ (start_position - midpoint)
            ),
        )

    left_target = rigid_target(left_start)
    right_target = rigid_target(right_start)
    absolute_task.set_target_from_arm_targets(_pose_matrix(left_target), _pose_matrix(right_target))
    absolute_task.update(robot)
    initial_absolute_error = float(np.linalg.norm(absolute_task.get_error()))

    options = eik.PositionStepOptions()
    options.dt = 0.02
    options.max_steps = 2
    options.adaptive_dt = True
    options.adaptive_dt_max_scale = 5.0
    options.adaptive_dt_reference_distance = 0.02
    options.max_linear_speed = 1.0
    options.max_angular_speed = 3.0
    options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
    targets = [
        eik.TaskTarget.from_se3_pair("tote_absolute", left_target, right_target, 12.0, 8.0),
        eik.TaskTarget.from_se3("tote_relative", relative_target, 12.0, 8.0),
    ]
    if relative_target_first:
        targets.reverse()

    result = solver.solve_position_step(q0, targets, options)
    q1 = np.asarray(result.q_solution, dtype=float)
    robot.update_configuration(q1)
    absolute_task.update(robot)
    relative_task.update(robot)
    relative_error = np.asarray(relative_task.get_error(), dtype=float)
    final_absolute_error = float(np.linalg.norm(absolute_task.get_error()))
    lower_limits, upper_limits = robot.get_joint_limits()

    assert result.status == eik.SolverStatus.SUCCESS, (
        result.status_message,
        final_absolute_error,
        relative_error,
        float(np.linalg.norm(q1 - q0)),
    )
    assert final_absolute_error < initial_absolute_error - 1e-3
    assert np.linalg.norm(q1 - q0) > 1e-3
    assert np.linalg.norm(relative_error[:3]) < 5e-3
    assert np.linalg.norm(relative_error[3:]) < math.radians(2.0)
    assert np.all(q1 >= np.asarray(lower_limits, dtype=float) - 1e-9)
    assert np.all(q1 <= np.asarray(upper_limits, dtype=float) + 1e-9)
