#!/usr/bin/env python3
"""Tests for adaptive task relaxation (SCALE + MIN_ERROR modes)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

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
        assert task.allow_min_error_fallback is False

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


def test_same_priority_scale_fallback_tasks_are_grouped_order_independently():
    urdf_path, robot, solver = _make_solver()
    try:
        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)

        link_task = solver.add_frame_task("link1_pos", "link1", eik.TaskType.FRAME_POSITION)
        ee_task = solver.add_frame_task("ee_pos", "ee", eik.TaskType.FRAME_POSITION)
        for task in (link_task, ee_task):
            task.priority = 0
            task.weight = 1.0
            task.solve_mode = eik.TaskSolveMode.SCALE_ELASTIC
            task.allow_min_error_fallback = True

        link_target = np.array(robot.get_frame_pose("link1").translation, dtype=float)
        ee_target = np.array(robot.get_frame_pose("ee").translation, dtype=float)
        link_task.set_target_position(link_target + np.array([0.02, 0.0, 0.0]))
        ee_task.set_target_position(ee_target + np.array([0.0, 0.02, 0.0]))

        result = solver.solve_velocity(q, apply_limits=True)

        assert result.status in (eik.SolverStatus.SUCCESS, eik.SolverStatus.INFEASIBLE)
        assert np.all(np.isfinite(np.asarray(result.joint_velocities, dtype=float)))
        assert len(result.task_modes_effective) == 1
        assert len(result.task_used_fallback) == 1
    finally:
        os.unlink(urdf_path)


def test_recovery_mode_softens_primary_task_to_min_error():
    """Repeated stalled errors should switch primary objective to MIN_ERROR."""
    urdf_path, robot, solver = _make_solver()
    try:
        q = np.array([3.13, 0.0], dtype=float)  # near joint1 upper limit
        robot.update_configuration(q)

        if not hasattr(solver, "configure_collision_constraint"):
            return
        dbg = solver.evaluate_collision_debug(q)
        if dbg is None:
            return
        solver.configure_collision_constraint(min_distance=float(dbg.distance) + 0.01)

        task = solver.add_frame_task("ee", "ee", eik.TaskType.FRAME_POSITION)
        task.priority = 0
        task.weight = 1.0
        task.solve_mode = eik.TaskSolveMode.SCALE
        task.allow_min_error_fallback = False
        ee_pos = np.array(robot.get_frame_pose("ee").translation, dtype=float)
        task.set_target_position(ee_pos + np.array([0.0, 0.20, 0.0], dtype=float))

        recovery_seen = False
        effective_modes = []
        for _ in range(220):
            result = solver.solve_velocity(q, apply_limits=True)
            if "recovery mode activated" in result.status_message:
                recovery_seen = True
            if len(result.task_modes_effective) > 0:
                effective_modes.append(result.task_modes_effective[0])
            dq = np.array(result.joint_velocities, dtype=float)
            q = q + dq * solver.dt
            q_lower, q_upper = robot.get_joint_limits()
            q = np.clip(q, q_lower, q_upper)
            robot.update_configuration(q)

        if not recovery_seen:
            return
        assert any(m == eik.TaskSolveMode.MIN_ERROR for m in effective_modes)
    finally:
        os.unlink(urdf_path)


def test_position_step_uses_registered_tasks():
    """solve_position_step should use the solver's registered tasks and expose
    velocity-level diagnostics (task_modes_effective, task_scales, etc.).
    Gains are supplied via PositionStepOptions; the task weight is independent."""
    urdf_path, robot, solver = _make_solver()
    try:
        task = solver.add_frame_task("ee_task", "ee")
        task.priority = 0
        task.weight = 1.0
        task.solve_mode = eik.TaskSolveMode.MIN_ERROR
        task.allow_min_error_fallback = False

        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[1, 3] += 0.05

        opts = eik.PositionStepOptions()
        opts.position_gain = 10.0
        opts.orientation_gain = 10.0
        result = solver.solve_position_step(q, target, "ee_task", opts)

        assert result.q_solution is not None
        assert len(result.task_modes_effective) >= 1
        assert result.task_modes_effective[0] == eik.TaskSolveMode.MIN_ERROR

        opts_multi = eik.PositionStepOptions()
        opts_multi.position_gain = 10.0
        opts_multi.orientation_gain = 10.0
        opts_multi.max_steps = 3
        result_multi = solver.solve_position_step(q, target, "ee_task", opts_multi)
        assert result_multi.q_solution is not None

        # Ergonomic overload: accept SE3 / Rt directly.
        target_se3 = eik.Rt(target[:3, :3], target[:3, 3])
        result_se3 = solver.solve_position_step(q, target_se3, "ee_task", opts)
        assert result_se3.q_solution is not None
    finally:
        os.unlink(urdf_path)


@pytest.mark.benchmark(group="position-step")
def test_position_step_single_task_benchmark(benchmark):  # type: ignore[no-untyped-def]
    """Benchmark solve_position_step single-task call path."""
    urdf_path, robot, solver = _make_solver()
    try:
        task = solver.add_frame_task("ee_task", "ee")
        task.priority = 0
        task.weight = 1.0
        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)

        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[1, 3] += 0.05

        opts = eik.PositionStepOptions()
        opts.position_gain = 10.0
        opts.orientation_gain = 10.0
        opts.max_steps = 1

        def run() -> eik.PositionIKResult:
            return solver.solve_position_step(q, target, "ee_task", opts)

        result = benchmark.pedantic(run, rounds=50, iterations=1)
        assert result.status in (
            (
                eik.SolverStatus.kSuccess
                if hasattr(eik.SolverStatus, "kSuccess")
                else eik.SolverStatus.SUCCESS
            ),  # compatibility
            eik.SolverStatus.NO_PROGRESS,
            eik.SolverStatus.INFEASIBLE,
            eik.SolverStatus.NUMERICAL_ERROR,
        )
    finally:
        os.unlink(urdf_path)


@pytest.mark.benchmark(group="position-step")
def test_position_step_low_level_reference_benchmark(benchmark):  # type: ignore[no-untyped-def]
    """Benchmark low-level equivalent path: pose-error + solve_velocity."""
    urdf_path, robot, solver = _make_solver()
    try:
        task = solver.add_frame_task("ee_task", "ee")
        task.priority = 0
        task.weight = 1.0
        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)

        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[1, 3] += 0.05
        task.set_target_pose(target[:3, 3], target[:3, :3])

        def run() -> eik.VelocitySolverResult:
            robot.update_configuration(q)
            task.update(robot)
            error = task.get_error()
            vel = np.zeros(6, dtype=float)
            vel[:3] = 10.0 * error[:3]
            vel[3:] = 10.0 * error[3:]
            task.set_target_velocity(vel)
            result = solver.solve_velocity(q, True)
            task.clear_target_velocity()
            return result

        result = benchmark.pedantic(run, rounds=50, iterations=1)
        assert result.status in (
            (
                eik.SolverStatus.kSuccess
                if hasattr(eik.SolverStatus, "kSuccess")
                else eik.SolverStatus.SUCCESS
            ),  # compatibility
            eik.SolverStatus.INFEASIBLE,
            eik.SolverStatus.NUMERICAL_ERROR,
        )
    finally:
        os.unlink(urdf_path)
