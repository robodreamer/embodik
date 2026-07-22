#!/usr/bin/env python3
"""Tests for solve_position_step safety helpers and speed caps."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import embodik as eik


def _create_two_joint_urdf(*, velocity_limit: float = 100.0) -> str:
    urdf_content = f"""<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="{velocity_limit}"/>
  </joint>
  <link name="link1"/>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="ee"/>
    <origin xyz="0.2 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="{velocity_limit}"/>
  </joint>
  <link name="ee"/>
</robot>
"""
    fd, path = tempfile.mkstemp(suffix=".urdf")
    with os.fdopen(fd, "w") as f:
        f.write(urdf_content)
    return path


def _make_solver():
    urdf_path = _create_two_joint_urdf()
    robot = eik.RobotModel(urdf_path, floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    return urdf_path, robot, solver


def test_position_step_resets_stale_target_velocities_on_all_tasks():
    urdf_path, robot, solver = _make_solver()
    try:
        ee_task = solver.add_frame_task("ee_task", "ee")
        ee_task.priority = 0
        ee_task.weight = 1.0

        posture = solver.add_posture_task("posture")
        posture.priority = 1
        posture.weight = 1.0
        posture.set_target_velocity(np.array([0.5, -0.5], dtype=float))

        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[0, 3] += 0.02

        opts = eik.PositionStepOptions()
        opts.max_steps = 1
        solver.solve_position_step(q, target, "ee_task", opts)

        ee_task.active = False
        check = solver.solve_velocity(q, apply_limits=False)
        dq = np.array(check.joint_velocities, dtype=float)
        assert np.linalg.norm(dq) < 1e-10
    finally:
        os.unlink(urdf_path)


@pytest.mark.parametrize("bad_value", [-1.0, float("nan")])
def test_position_step_configuration_step_cap_rejects_invalid_values(bad_value: float):
    urdf_path, robot, solver = _make_solver()
    try:
        task = solver.add_frame_task("ee_task", "ee")
        task.priority = 0
        task.weight = 1.0

        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)
        target = np.eye(4, dtype=float)
        target[:3, 3] = np.array(robot.get_frame_pose("ee").translation, dtype=float)
        target[1, 3] += 0.1

        opts = eik.PositionStepOptions()
        opts.max_configuration_step_norm = bad_value

        out = solver.solve_position_step(q, target, "ee_task", opts)
        assert out.status == eik.SolverStatus.INVALID_INPUT
        assert "max_configuration_step_norm" in out.status_message
    finally:
        os.unlink(urdf_path)


def test_position_step_linear_speed_cap_limits_motion():
    urdf_path, robot, solver = _make_solver()
    try:
        task = solver.add_frame_task("ee_task", "ee")
        task.priority = 0
        task.weight = 1.0

        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)
        start_pos = np.array(robot.get_frame_pose("ee").translation, dtype=float)
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.eye(3)
        target[:3, 3] = start_pos + np.array([0.0, 0.15, 0.0], dtype=float)

        opts_unlimited = eik.PositionStepOptions()
        opts_unlimited.position_gain = 80.0
        opts_unlimited.orientation_gain = 80.0
        opts_unlimited.max_steps = 1
        q_unlimited = np.array(
            solver.solve_position_step(q, target, "ee_task", opts_unlimited).q_solution,
            dtype=float,
        )
        robot.update_configuration(q_unlimited)
        pos_unlimited = np.array(robot.get_frame_pose("ee").translation, dtype=float)
        disp_unlimited = np.linalg.norm(pos_unlimited - start_pos)

        robot.update_configuration(q)
        opts_limited = eik.PositionStepOptions()
        opts_limited.position_gain = 80.0
        opts_limited.orientation_gain = 80.0
        opts_limited.max_steps = 1
        opts_limited.max_linear_speed = 0.01
        q_limited = np.array(
            solver.solve_position_step(q, target, "ee_task", opts_limited).q_solution,
            dtype=float,
        )
        robot.update_configuration(q_limited)
        pos_limited = np.array(robot.get_frame_pose("ee").translation, dtype=float)
        disp_limited = np.linalg.norm(pos_limited - start_pos)

        assert disp_limited < disp_unlimited
        assert disp_limited <= 5e-4
    finally:
        os.unlink(urdf_path)
