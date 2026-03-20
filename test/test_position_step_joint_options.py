#!/usr/bin/env python3
"""Tests for PositionStepOptions joint-index fields (IK-aligned API)."""

from __future__ import annotations

import os
import tempfile

import numpy as np

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


def _make_solver_with_posture():
    urdf_path = _create_two_joint_urdf()
    robot = eik.RobotModel(urdf_path, floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    ee_task = solver.add_frame_task("ee_task", "ee")
    ee_task.priority = 0
    ee_task.weight = 1.0
    posture = solver.add_posture_task("posture")
    posture.priority = 1
    posture.weight = 1.0
    posture.set_target_velocity(np.array([0.4, -0.4], dtype=float))
    return urdf_path, robot, solver, ee_task, posture


def test_integration_zero_velocity_indices_invalid_returns_invalid_input():
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01
        task = solver.add_frame_task("ee_task", "ee")
        task.priority = 0
        task.weight = 1.0

        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[0, 3] += 0.02

        opts = eik.PositionStepOptions()
        opts.max_steps = 1
        opts.integration_zero_velocity_indices = [2]
        result = solver.solve_position_step(q, target, "ee_task", opts)
        assert result.status == eik.SolverStatus.INVALID_INPUT
        assert "integration_zero_velocity_indices" in result.status_message
    finally:
        os.unlink(urdf_path)


def test_locked_joint_indices_invalid():
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        task = solver.add_frame_task("ee_task", "ee")
        task.priority = 0
        task.weight = 1.0
        q = np.zeros(2, dtype=float)
        target = np.eye(4, dtype=float)
        opts = eik.PositionStepOptions()
        opts.locked_joint_indices = [-1]
        result = solver.solve_position_step(q, target, "ee_task", opts)
        assert result.status == eik.SolverStatus.INVALID_INPUT
    finally:
        os.unlink(urdf_path)


def test_integration_zero_velocity_indices_masks_before_integrate():
    """With all nv-indices masked, configuration does not move despite EE error."""
    urdf_path, robot, solver, _, _ = _make_solver_with_posture()
    try:
        q = np.array([0.2, -0.15], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[0, 3] += 0.12

        opts = eik.PositionStepOptions()
        opts.max_steps = 2
        opts.position_gain = 50.0
        opts.orientation_gain = 50.0
        opts.integration_zero_velocity_indices = [0, 1]
        res = solver.solve_position_step(q, target, "ee_task", opts)
        assert res.status == eik.SolverStatus.SUCCESS
        dq = np.asarray(res.joint_velocities, dtype=float)
        assert abs(dq[0]) < 1e-12 and abs(dq[1]) < 1e-12
        np.testing.assert_allclose(np.asarray(res.q_solution, dtype=float), q, atol=1e-10)
    finally:
        os.unlink(urdf_path)


def test_locked_joint_indices_qp_zero_and_consistent_velocity():
    urdf_path, robot, solver, _, _ = _make_solver_with_posture()
    try:
        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[0, 3] += 0.05

        opts = eik.PositionStepOptions()
        opts.max_steps = 1
        opts.position_gain = 40.0
        opts.orientation_gain = 40.0
        opts.locked_joint_indices = [1]
        res = solver.solve_position_step(q, target, "ee_task", opts)
        assert res.status == eik.SolverStatus.SUCCESS
        q_out = np.array(res.q_solution, dtype=float)
        assert abs(q_out[1] - q[1]) < 1e-9
        assert float(np.asarray(res.joint_velocities, dtype=float)[1]) == 0.0
    finally:
        os.unlink(urdf_path)


def test_integration_mask_applied_each_substep_when_max_steps_gt_1():
    urdf_path, robot, solver, _, _ = _make_solver_with_posture()
    try:
        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[0, 3] += 0.08

        opts = eik.PositionStepOptions()
        opts.max_steps = 3
        opts.position_gain = 35.0
        opts.orientation_gain = 35.0
        opts.integration_zero_velocity_indices = [1]
        res = solver.solve_position_step(q, target, "ee_task", opts)
        assert res.status == eik.SolverStatus.SUCCESS
        q_out = np.array(res.q_solution, dtype=float)
        assert abs(q_out[1] - q[1]) < 1e-8
        assert float(np.asarray(res.joint_velocities, dtype=float)[1]) == 0.0
    finally:
        os.unlink(urdf_path)


def test_integration_mask_multi_task_path():
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01
        ee_task = solver.add_frame_task("ee_task", "ee")
        ee_task.priority = 0
        ee_task.weight = 1.0
        mid_task = solver.add_frame_task("mid_task", "link1")
        mid_task.priority = 0
        mid_task.weight = 1.0
        posture = solver.add_posture_task("posture")
        posture.priority = 1
        posture.weight = 1.0
        posture.set_target_velocity(np.array([0.35, -0.35], dtype=float))

        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)
        t_ee = np.eye(4, dtype=float)
        t_ee[:3, :3] = np.array(robot.get_frame_pose("ee").rotation, dtype=float)
        t_ee[:3, 3] = np.array(robot.get_frame_pose("ee").translation, dtype=float)
        t_ee[0, 3] += 0.04
        t_mid = np.eye(4, dtype=float)
        t_mid[:3, :3] = np.array(robot.get_frame_pose("link1").rotation, dtype=float)
        t_mid[:3, 3] = np.array(robot.get_frame_pose("link1").translation, dtype=float)

        targets = [
            eik.TaskTarget("ee_task", t_ee, 30.0, 30.0),
            eik.TaskTarget("mid_task", t_mid, 15.0, 15.0),
        ]
        opts = eik.PositionStepOptions()
        opts.max_steps = 1
        opts.integration_zero_velocity_indices = [1]
        res = solver.solve_position_step(q, targets, opts)
        assert res.status == eik.SolverStatus.SUCCESS
        q_out = np.array(res.q_solution, dtype=float)
        assert abs(q_out[1] - q[1]) < 1e-8
        assert float(np.asarray(res.joint_velocities, dtype=float)[1]) == 0.0
    finally:
        os.unlink(urdf_path)


def test_excluded_joint_indices_do_not_mutate_registered_task_exclusions():
    """solve_position_step exclusion handling must not rewrite task state."""
    urdf_path, robot, solver, ee_task, posture = _make_solver_with_posture()
    try:
        ee_task.set_excluded_joint_indices([0])
        posture.set_excluded_joint_indices([1])
        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[0, 3] += 0.02

        opts = eik.PositionStepOptions()
        opts.max_steps = 1
        opts.position_gain = 40.0
        opts.orientation_gain = 40.0
        opts.excluded_joint_indices = [1]
        res = solver.solve_position_step(q, target, "ee_task", opts)
        assert res.status == eik.SolverStatus.SUCCESS

        assert list(ee_task.get_excluded_joint_indices()) == [0]
        assert list(posture.get_excluded_joint_indices()) == [1]
    finally:
        os.unlink(urdf_path)


def test_excluded_joint_indices_step_matches_solve_position_lock_behavior():
    urdf_path, robot, solver, _, _ = _make_solver_with_posture()
    try:
        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[0, 3] += 0.03

        step_opts = eik.PositionStepOptions()
        step_opts.max_steps = 1
        step_opts.position_gain = 40.0
        step_opts.orientation_gain = 40.0
        step_opts.excluded_joint_indices = [1]
        step_out = solver.solve_position_step(q, target, "ee_task", step_opts)
        assert step_out.status == eik.SolverStatus.SUCCESS

        ik_opts = eik.PositionIKOptions()
        ik_opts.max_iterations = 1
        ik_opts.position_gain = 40.0
        ik_opts.orientation_gain = 40.0
        ik_opts.excluded_joint_indices = [1]
        ik_out = solver.solve_position(q, target, "ee", ik_opts)
        assert ik_out.status in (eik.SolverStatus.SUCCESS, eik.SolverStatus.INFEASIBLE)

        assert abs(float(np.asarray(step_out.q_solution, dtype=float)[1]) - q[1]) < 1e-9
        assert abs(float(np.asarray(ik_out.q_solution, dtype=float)[1]) - q[1]) < 1e-9
    finally:
        os.unlink(urdf_path)


def test_duplicate_integration_indices_idempotent():
    urdf_path, robot, solver, _, _ = _make_solver_with_posture()
    try:
        q = np.array([0.1, -0.1], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[1, 3] += 0.03

        opts = eik.PositionStepOptions()
        opts.max_steps = 1
        opts.position_gain = 40.0
        opts.orientation_gain = 40.0
        opts.integration_zero_velocity_indices = [1, 1]
        res = solver.solve_position_step(q, target, "ee_task", opts)
        assert res.status == eik.SolverStatus.SUCCESS
        assert abs(float(np.asarray(res.q_solution, dtype=float)[1]) - q[1]) < 1e-8
    finally:
        os.unlink(urdf_path)
