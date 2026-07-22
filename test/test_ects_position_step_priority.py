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


@pytest.mark.parametrize(
    (
        "relative_target_first",
        "acceleration_limits_enabled",
        "initial_relative_orientation_error",
    ),
    [
        (False, False, 0.0),
        (True, False, 0.0),
        (False, True, 0.0),
        (True, True, 0.0),
        pytest.param(
            True,
            True,
            2e-3 - 5e-8,
            id="near-satisfied-relative-orientation",
        ),
    ],
)
def test_lower_priority_motion_preserves_satisfied_relative_grasp(
    relative_target_first: bool,
    acceleration_limits_enabled: bool,
    initial_relative_orientation_error: float,
) -> None:
    """Lower-priority tote motion must not break its priority-0 grasp."""
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
    acceleration_limit = 10.0
    if acceleration_limits_enabled:
        solver.set_acceleration_limits(np.full(robot.nv, acceleration_limit, dtype=float))
        solver.enable_acceleration_limits(True)
        solver.set_previous_joint_velocities(np.zeros(robot.nv, dtype=float))

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
    relative_target_rotation = np.asarray(relative_task.current_orientation, dtype=float)
    relative_target_translation = np.asarray(relative_task.current_position, dtype=float)
    if initial_relative_orientation_error > 0.0:
        relative_target_rotation = (
            _rotation_from_rpy(initial_relative_orientation_error, 0.0, 0.0)
            @ relative_target_rotation
        )
    relative_target = eik.SE3(
        rotation=relative_target_rotation,
        translation=relative_target_translation,
    )

    translation_delta = np.array([0.045, 0.045, 0.045], dtype=float)
    midpoint = 0.5 * (
        np.asarray(left_start.translation, dtype=float)
        + np.asarray(right_start.translation, dtype=float)
    )

    def rigid_target(start_pose: object, fraction: float = 1.0) -> eik.SE3:
        rotation = _rotation_from_rpy(
            fraction * math.radians(12.0),
            fraction * math.radians(12.0),
            fraction * math.radians(18.0),
        )
        start_position = np.asarray(start_pose.translation, dtype=float)
        return eik.SE3(
            rotation=rotation @ np.asarray(start_pose.rotation, dtype=float),
            translation=(
                midpoint + fraction * translation_delta + rotation @ (start_position - midpoint)
            ),
        )

    left_target = rigid_target(left_start)
    right_target = rigid_target(right_start)
    absolute_task.set_target_from_arm_targets(_pose_matrix(left_target), _pose_matrix(right_target))
    absolute_task.update(robot)
    initial_absolute_error = float(np.linalg.norm(absolute_task.get_error()))

    options = eik.PositionStepOptions()
    options.dt = 0.02
    options.max_steps = 3
    options.adaptive_dt = True
    options.adaptive_dt_max_scale = 10.0
    options.adaptive_dt_reference_distance = 0.02
    options.max_linear_speed = 0.5
    options.max_angular_speed = 1.0
    options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR

    def make_targets(
        fraction: float = 1.0, grasp_conflict_fraction: float = 0.0
    ) -> list[eik.TaskTarget]:
        relative_step_target = eik.TaskTarget.from_se3("tote_relative", relative_target, 12.0, 8.0)
        if initial_relative_orientation_error > 0.0:
            relative_step_target.priority_position_tolerance = 5e-3
            relative_step_target.priority_orientation_tolerance = math.radians(2.0)
        left_command = rigid_target(left_start, fraction)
        right_command = rigid_target(right_start, fraction)
        if grasp_conflict_fraction > 0.0:
            right_command = eik.SE3(
                rotation=np.asarray(right_command.rotation, dtype=float),
                translation=(
                    np.asarray(right_command.translation, dtype=float)
                    + grasp_conflict_fraction * np.array([0.06, 0.0, 0.0], dtype=float)
                ),
            )
        targets = [
            eik.TaskTarget.from_se3_pair(
                "tote_absolute",
                left_command,
                right_command,
                12.0,
                8.0,
            ),
            relative_step_target,
        ]
        if relative_target_first:
            targets.reverse()
        return targets

    q1 = q0.copy()
    previous_velocity = np.zeros(robot.nv, dtype=float)
    moving_ticks = 0
    if initial_relative_orientation_error > 0.0:
        # Cover two complete cycles so reaching the protected grasp boundary
        # cannot turn into a permanent hold under the acceleration corridor.
        fractions = np.sin(2.0 * math.pi * 0.5 * options.dt * np.arange(200))
    else:
        fractions = np.linspace(0.05, 1.0, 20) if acceleration_limits_enabled else [1.0]
    for tick, fraction in enumerate(fractions):
        grasp_conflict_fraction = (
            min(1.0, (tick + 1) / 20.0) if initial_relative_orientation_error > 0.0 else 0.0
        )
        result = solver.solve_position_step(
            q1,
            make_targets(float(fraction), grasp_conflict_fraction),
            options,
        )
        q_next = np.asarray(result.q_solution, dtype=float)
        if initial_relative_orientation_error > 0.0:
            assert result.status == eik.SolverStatus.SUCCESS, result.status_message
        moving_ticks += int(np.linalg.norm(q_next - q1) > 1e-8)
        if acceleration_limits_enabled:
            applied_velocity = (q_next - q1) / options.dt
            assert (
                np.max(np.abs(applied_velocity - previous_velocity)) / options.dt
                <= acceleration_limit * 1.02
            )
            previous_velocity = applied_velocity
        q1 = q_next
        robot.update_configuration(q1)
        relative_task.update(robot)
        tick_relative_error = np.asarray(relative_task.get_error(), dtype=float)
        assert np.linalg.norm(tick_relative_error[:3]) < 5e-3
        assert np.linalg.norm(tick_relative_error[3:]) < math.radians(2.0)
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
    assert moving_ticks >= max(1, len(fractions) // 2)
    assert np.linalg.norm(relative_error[:3]) < 5e-3
    assert np.linalg.norm(relative_error[3:]) < math.radians(2.0)
    assert np.all(q1 >= np.asarray(lower_limits, dtype=float) - 1e-9)
    assert np.all(q1 <= np.asarray(upper_limits, dtype=float) + 1e-9)
