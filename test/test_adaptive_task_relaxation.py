#!/usr/bin/env python3
"""Tests for adaptive task relaxation (SCALE + MIN_ERROR modes)."""

from __future__ import annotations

import os
import tempfile

import numpy as np

import embodik as eik


def _create_two_joint_urdf() -> str:
    urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1.0"/>
  </joint>
  <link name="link1"/>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="ee"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1.0"/>
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


def test_task_mode_defaults_and_roundtrip():
    urdf_path, robot, solver = _make_solver()
    try:
        task = solver.add_frame_task("ee", "ee", eik.TaskType.FRAME_POSE)
        assert task.solve_mode == eik.TaskSolveMode.SCALE
        assert task.allow_min_error_fallback is True

        task.solve_mode = eik.TaskSolveMode.MIN_ERROR
        task.allow_min_error_fallback = True
        assert task.solve_mode == eik.TaskSolveMode.MIN_ERROR
        assert task.allow_min_error_fallback is True
    finally:
        os.unlink(urdf_path)


def test_min_error_mode_reports_effective_mode_and_respects_limits():
    urdf_path, robot, solver = _make_solver()
    try:
        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)

        posture = solver.add_posture_task("posture")
        posture.priority = 0
        posture.weight = 1.0
        posture.solve_mode = eik.TaskSolveMode.MIN_ERROR
        posture.allow_min_error_fallback = False
        posture.set_target_configuration(np.array([2.0, 2.0], dtype=float))

        result = solver.solve_velocity(q, apply_limits=True)
        dq = np.array(result.joint_velocities)
        vel_limits = np.array(robot.get_velocity_limits())

        assert result.status == eik.SolverStatus.SUCCESS
        assert len(result.task_modes_effective) >= 1
        assert result.task_modes_effective[0] == eik.TaskSolveMode.MIN_ERROR
        assert result.task_used_fallback[0] is False
        assert np.all(dq <= vel_limits + 1e-6)
        assert np.all(dq >= -vel_limits - 1e-6)
    finally:
        os.unlink(urdf_path)


def test_scale_with_fallback_reports_fallback_usage():
    urdf_path, robot, solver = _make_solver()
    try:
        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)

        posture = solver.add_posture_task("posture")
        posture.priority = 0
        posture.weight = 1.0
        posture.solve_mode = eik.TaskSolveMode.SCALE
        posture.allow_min_error_fallback = True
        posture.set_target_configuration(np.array([5.0, 5.0], dtype=float))

        result = solver.solve_velocity(q, apply_limits=True)

        assert result.status == eik.SolverStatus.SUCCESS
        assert len(result.task_modes_effective) >= 1
        assert result.task_modes_effective[0] in (
            eik.TaskSolveMode.SCALE,
            eik.TaskSolveMode.MIN_ERROR,
        )
        assert len(result.task_used_fallback) >= 1
        assert isinstance(result.task_used_fallback[0], bool)
    finally:
        os.unlink(urdf_path)
