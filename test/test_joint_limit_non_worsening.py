#!/usr/bin/env python3
"""Primary-task continuity at active joint-limit margins."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

import embodik as eik


def test_joint_limit_non_worsening_policy_is_default_off_and_validates_margin(
    tmp_path: Path,
) -> None:
    runtime = eik.SolverRuntimeConfig()
    assert runtime.joint_limit_non_worsening_enabled is False
    assert runtime.joint_limit_non_worsening_margin == pytest.approx(0.04)

    runtime.joint_limit_non_worsening_enabled = True
    robot = eik.RobotModel(str(_write_decoupled_xy_stage_urdf(tmp_path)), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    for invalid_margin in (0.0, -0.01, float("nan")):
        runtime.joint_limit_non_worsening_margin = invalid_margin
        with pytest.raises(ValueError, match="joint_limit_non_worsening_margin"):
            solver.configure_runtime(runtime)


def _write_decoupled_xy_stage_urdf(tmp_path: Path) -> Path:
    urdf = textwrap.dedent("""
        <robot name="joint_limit_tangent_stage">
          <link name="base"/>
          <link name="x_stage"/>
          <link name="tool"/>
          <joint name="slide_x" type="prismatic">
            <parent link="base"/><child link="x_stage"/>
            <axis xyz="1 0 0"/>
            <limit lower="-0.05" upper="0.05" effort="10" velocity="2"/>
          </joint>
          <joint name="slide_y" type="prismatic">
            <parent link="x_stage"/><child link="tool"/>
            <axis xyz="0 1 0"/>
            <limit lower="-0.20" upper="0.20" effort="10" velocity="2"/>
          </joint>
        </robot>
        """).strip()
    path = tmp_path / "joint_limit_tangent_stage.urdf"
    path.write_text(urdf, encoding="utf-8")
    return path


@pytest.mark.parametrize("boundary", ("lower", "upper"))
@pytest.mark.parametrize(
    "solve_mode",
    (
        eik.TaskSolveMode.SCALE,
        eik.TaskSolveMode.SCALE_ELASTIC,
        eik.TaskSolveMode.MIN_ERROR,
    ),
)
def test_primary_cartesian_motion_preserves_active_limit_slack_and_tangent_progress(
    tmp_path: Path,
    boundary: str,
    solve_mode: eik.TaskSolveMode,
) -> None:
    robot = eik.RobotModel(str(_write_decoupled_xy_stage_urdf(tmp_path)), floating_base=False)
    normal_sign = -1.0 if boundary == "lower" else 1.0
    q = np.array([normal_sign * 0.04, 0.0], dtype=float)
    robot.update_configuration(q)
    entry_position = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)
    lower, upper = (np.asarray(value, dtype=float) for value in robot.get_joint_limits())
    entry_limit_slack = float(q[0] - lower[0] if boundary == "lower" else upper[0] - q[0])

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    runtime = eik.SolverRuntimeConfig()
    runtime.weighted_fallback_enabled = False
    runtime.enable_auto_task_layout = False
    runtime.joint_limit_non_worsening_enabled = True
    runtime.joint_limit_non_worsening_margin = 0.02
    solver.configure_runtime(runtime)

    task = solver.add_frame_task("tool_position", "tool", eik.TaskType.FRAME_POSITION)
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = solve_mode

    target_pose = np.eye(4, dtype=float)
    target_pose[:3, 3] = entry_position + np.array([normal_sign * 0.03, 0.08, 0.0])
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.dt = solver.dt
    options.position_gain = 10.0
    options.orientation_gain = 0.0
    options.primary_solve_mode = solve_mode
    options.primary_allow_min_error_fallback = False
    options.continuity_command_revision = 1

    limit_slacks: list[float] = []
    tangent_progress: list[float] = []
    step_norms: list[float] = []
    statuses: list[str] = []
    for _ in range(10):
        result = solver.solve_position_step(q, target_pose, "tool_position", options)
        q_next = np.asarray(result.q_solution, dtype=float)
        step_norms.append(float(np.linalg.norm(robot.difference(q, q_next))))
        q = q_next
        robot.update_configuration(q)
        position = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)
        limit_slacks.append(float(q[0] - lower[0] if boundary == "lower" else upper[0] - q[0]))
        tangent_progress.append(float(position[1] - entry_position[1]))
        statuses.append(result.status.name)

    zero_motion_streak = 0
    max_zero_motion_streak = 0
    for step_norm in step_norms:
        zero_motion_streak = zero_motion_streak + 1 if step_norm <= 1e-8 else 0
        max_zero_motion_streak = max(max_zero_motion_streak, zero_motion_streak)

    assert min(limit_slacks) >= entry_limit_slack - 1e-10
    assert tangent_progress[-1] >= 0.065
    assert sum(np.diff(np.array([0.0, *tangent_progress])) > 1e-5) >= 8
    assert max_zero_motion_streak <= 1
    assert statuses == ["SUCCESS"] * 10


def test_active_limit_cone_overrides_stale_outward_acceleration_history(
    tmp_path: Path,
) -> None:
    robot = eik.RobotModel(str(_write_decoupled_xy_stage_urdf(tmp_path)), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.set_acceleration_limits(np.full(robot.nv, 0.1, dtype=float))
    solver.enable_acceleration_limits(True)

    runtime = eik.SolverRuntimeConfig()
    runtime.weighted_fallback_enabled = False
    runtime.enable_auto_task_layout = False
    runtime.joint_limit_non_worsening_enabled = True
    runtime.joint_limit_non_worsening_margin = 0.02
    solver.configure_runtime(runtime)

    task = solver.add_frame_task("tool_position", "tool", eik.TaskType.FRAME_POSITION)
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR

    q_clear = np.zeros(robot.nq, dtype=float)
    robot.update_configuration(q_clear)
    clear_position = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)
    task.set_target_pose(clear_position + np.array([0.20, 0.0, 0.0]), np.eye(3))

    previous_velocity = np.zeros(robot.nv, dtype=float)
    for _ in range(12):
        result = solver.solve_velocity(q_clear, apply_limits=True)
        assert result.status == eik.SolverStatus.SUCCESS
        previous_velocity = np.asarray(result.joint_velocities, dtype=float)
    assert previous_velocity[0] > 0.015

    q_active = np.array([0.04, 0.0], dtype=float)
    robot.update_configuration(q_active)
    active_position = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)
    task.set_target_pose(active_position + np.array([0.03, 0.08, 0.0]), np.eye(3))

    result = solver.solve_velocity(q_active, apply_limits=True)
    velocity = np.asarray(result.joint_velocities, dtype=float)

    assert result.status == eik.SolverStatus.SUCCESS
    assert velocity[0] <= 1e-10
    assert velocity[1] >= 1e-3


def test_approach_brakes_into_active_limit_cone_with_bounded_acceleration(
    tmp_path: Path,
) -> None:
    robot = eik.RobotModel(str(_write_decoupled_xy_stage_urdf(tmp_path)), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    acceleration_limit = 0.5
    solver.set_acceleration_limits(np.full(robot.nv, acceleration_limit, dtype=float))
    solver.enable_acceleration_limits(True)

    runtime = eik.SolverRuntimeConfig()
    runtime.weighted_fallback_enabled = False
    runtime.enable_auto_task_layout = False
    runtime.joint_limit_non_worsening_enabled = True
    runtime.joint_limit_non_worsening_margin = 0.02
    solver.configure_runtime(runtime)

    task = solver.add_frame_task("tool_position", "tool", eik.TaskType.FRAME_POSITION)
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR

    q = np.zeros(robot.nq, dtype=float)
    robot.update_configuration(q)
    entry_position = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)
    task.set_target_pose(entry_position + np.array([0.20, 0.15, 0.0]), np.eye(3))

    previous_velocity = np.zeros(robot.nv, dtype=float)
    upper_limit = float(np.asarray(robot.get_joint_limits()[1], dtype=float)[0])
    active_slacks: list[float] = []
    tangent_positions: list[float] = []
    max_acceleration_step = 0.0
    max_unfinished_zero_motion_streak = 0
    unfinished_zero_motion_streak = 0
    for _ in range(80):
        result = solver.solve_velocity(q, apply_limits=True)
        assert result.status in (
            eik.SolverStatus.SUCCESS,
            eik.SolverStatus.NO_PROGRESS,
        ), result.status_message
        velocity = np.asarray(result.joint_velocities, dtype=float)
        max_acceleration_step = max(
            max_acceleration_step,
            float(np.max(np.abs(velocity - previous_velocity))),
        )

        unfinished_tangent_motion = float(q[1]) < 0.11
        zero_motion = float(np.linalg.norm(velocity)) < 1e-6
        unfinished_zero_motion_streak = (
            unfinished_zero_motion_streak + 1 if unfinished_tangent_motion and zero_motion else 0
        )
        max_unfinished_zero_motion_streak = max(
            max_unfinished_zero_motion_streak,
            unfinished_zero_motion_streak,
        )

        q = np.asarray(robot.integrate(q, velocity * solver.dt), dtype=float)
        robot.update_configuration(q)
        upper_slack = upper_limit - float(q[0])
        if upper_slack <= runtime.joint_limit_non_worsening_margin:
            active_slacks.append(upper_slack)
        tangent_positions.append(float(q[1]))
        previous_velocity = velocity

    assert active_slacks
    assert min(active_slacks) >= active_slacks[0] - 1e-10
    assert tangent_positions[-1] >= 0.11
    assert max_unfinished_zero_motion_streak <= 1
    assert max_acceleration_step <= acceleration_limit * solver.dt * 1.02
