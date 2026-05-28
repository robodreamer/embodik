#!/usr/bin/env python3
"""Tests for PositionStepOptions joint-index fields (IK-aligned API)."""

from __future__ import annotations

import os
import tempfile

import numpy as np

import embodik as eik
from embodik import _embodik_impl as _eik_native


def _torso_rel_state_vs_reference(ref_pose, cur_pose) -> np.ndarray:
    """Match solve_position torso pose-box definition (kinematics_solver.cpp).

    6D state [x,y,z,rx,ry,rz]: translation delta in world (m); rotation delta as
    log(R_ref^T R_cur) (rad), consistent with native log3 / pinocchio::log3 in C++.
    """
    R_ref = np.asarray(ref_pose.rotation, dtype=float)
    t_ref = np.asarray(ref_pose.translation, dtype=float)
    R_cur = np.asarray(cur_pose.rotation, dtype=float)
    t_cur = np.asarray(cur_pose.translation, dtype=float)
    rel = np.zeros(6, dtype=float)
    rel[:3] = t_cur - t_ref
    rel[3:] = _eik_native.log3(R_ref.T @ R_cur)
    return rel


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


def _create_lower_limited_two_joint_urdf(*, velocity_limit: float = 100.0) -> str:
    urdf_content = f"""<?xml version="1.0"?>
<robot name="test_robot_lower_limited">
  <link name="base_link"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="0.0" upper="3.14" effort="100" velocity="{velocity_limit}"/>
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


def test_adaptive_dt_applies_to_multi_target_overload():
    """Multi-target solve_position_step should honor PositionStepOptions.adaptive_dt."""
    urdf_path, robot, solver, _, _ = _make_solver_with_posture()
    try:
        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[1, 3] += 0.12
        targets = [eik.TaskTarget("ee_task", target, 20.0, 20.0)]

        opts_base = eik.PositionStepOptions()
        opts_base.dt = 0.01
        opts_base.max_steps = 1
        opts_base.adaptive_dt = False

        opts_adapt = eik.PositionStepOptions()
        opts_adapt.dt = 0.01
        opts_adapt.max_steps = 1
        opts_adapt.adaptive_dt = True
        opts_adapt.adaptive_dt_max_scale = 4.0
        opts_adapt.adaptive_dt_reference_distance = 0.01

        base = solver.solve_position_step(q, targets, opts_base)
        robot.update_configuration(q)
        adapt = solver.solve_position_step(q, targets, opts_adapt)

        base_delta = np.linalg.norm(np.asarray(base.q_solution, dtype=float) - q)
        adapt_delta = np.linalg.norm(np.asarray(adapt.q_solution, dtype=float) - q)

        assert adapt.status == eik.SolverStatus.SUCCESS
        assert adapt_delta > base_delta * 2.0
    finally:
        os.unlink(urdf_path)


def test_adaptive_dt_respects_position_limits_during_single_target_integration():
    """Adaptive integration must not step past native position limits."""
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.enable_position_limits(True)
        task = solver.add_frame_task("ee_task", "ee")
        task.priority = 0
        task.weight = 1.0

        q = np.array([3.13, 0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.asarray(pose.rotation, dtype=float)
        target[:3, 3] = np.asarray(pose.translation, dtype=float)
        target[1, 3] -= 0.2

        opts = eik.PositionStepOptions()
        opts.dt = 0.01
        opts.max_steps = 1
        opts.position_gain = 50.0
        opts.orientation_gain = 50.0
        opts.adaptive_dt = True
        opts.adaptive_dt_max_scale = 10.0
        opts.adaptive_dt_reference_distance = 0.01

        res = solver.solve_position_step(q, target, "ee_task", opts)

        assert res.status == eik.SolverStatus.SUCCESS
        assert np.asarray(res.q_solution, dtype=float)[0] <= 3.14
    finally:
        os.unlink(urdf_path)


def test_adaptive_dt_respects_position_limits_during_multi_target_integration():
    """Multi-target adaptive integration must use the effective dt for limits."""
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.enable_position_limits(True)
        task = solver.add_frame_task("ee_task", "ee")
        task.priority = 0
        task.weight = 1.0

        q = np.array([3.13, 0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.asarray(pose.rotation, dtype=float)
        target[:3, 3] = np.asarray(pose.translation, dtype=float)
        target[1, 3] -= 0.2
        targets = [eik.TaskTarget("ee_task", target, 50.0, 50.0)]

        opts = eik.PositionStepOptions()
        opts.dt = 0.01
        opts.max_steps = 1
        opts.adaptive_dt = True
        opts.adaptive_dt_max_scale = 10.0
        opts.adaptive_dt_reference_distance = 0.01

        res = solver.solve_position_step(q, targets, opts)

        assert res.status == eik.SolverStatus.SUCCESS
        assert np.asarray(res.q_solution, dtype=float)[0] <= 3.14
    finally:
        os.unlink(urdf_path)


def test_elastic_position_step_returns_true_joint_limits_after_integration():
    """Elastic-band internal limit expansion must not leak into q_solution."""
    urdf_path = _create_lower_limited_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.enable_position_limits(True)
        solver.dt = 0.01
        task = solver.add_frame_task("ee_task", "ee")
        task.priority = 0
        task.weight = 1.0

        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.asarray(pose.rotation, dtype=float)
        target[:3, 3] = np.asarray(pose.translation, dtype=float)
        target[1, 3] -= 0.1

        opts = eik.PositionStepOptions()
        opts.dt = 0.01
        opts.max_steps = 1
        opts.position_gain = 100.0
        opts.orientation_gain = 0.0
        opts.elastic_band = True

        res = solver.solve_position_step(q, target, "ee_task", opts)
        q_lower, q_upper = robot.get_joint_limits()
        q_solution = np.asarray(res.q_solution, dtype=float)

        assert res.status in (eik.SolverStatus.SUCCESS, eik.SolverStatus.INFEASIBLE)
        assert q_solution[0] >= q_lower[0] - 1e-12
        assert q_solution[0] <= q_upper[0] + 1e-12
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


def test_solve_position_integrates_inside_scalar_joint_limits_without_python_clip():
    urdf_path = _create_two_joint_urdf(velocity_limit=1000.0)
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        q_lower, q_upper = robot.get_joint_limits()

        q = np.array([q_upper[0] - 5e-5, 0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[1, 3] += 0.5

        opts = eik.PositionIKOptions()
        opts.max_iterations = 1
        opts.dt = 0.05
        opts.position_gain = 500.0
        opts.orientation_gain = 0.0
        opts.max_linear_step = 0.0
        opts.max_angular_step = 0.0
        opts.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
        opts.primary_allow_min_error_fallback = True
        out = solver.solve_position(q, target, "ee", opts)

        assert out.status in (eik.SolverStatus.SUCCESS, eik.SolverStatus.INFEASIBLE)
        q_solution = np.asarray(out.q_solution, dtype=float)
        assert q_solution[0] <= q_upper[0] - 1e-4 + 1e-10
        assert q_solution[0] >= q_lower[0] + 1e-4 - 1e-10
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


def test_single_step_iterations_used_reflects_executed_steps():
    urdf_path, robot, solver, _, _ = _make_solver_with_posture()
    try:
        q = np.array([0.1, -0.1], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[0, 3] += 0.03

        opts = eik.PositionStepOptions()
        opts.max_steps = 5
        opts.no_progress_max_steps = 1
        opts.locked_joint_indices = [0, 1]
        res = solver.solve_position_step(q, target, "ee_task", opts)
        assert res.status == eik.SolverStatus.NO_PROGRESS
        assert res.iterations_used == 1
    finally:
        os.unlink(urdf_path)


def test_reference_corridor_limits_total_displacement_from_seed():
    urdf_path = _create_two_joint_urdf(velocity_limit=0.1)
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01
        ee_task = solver.add_frame_task("ee_task", "ee")
        ee_task.priority = 0
        ee_task.weight = 1.0
        posture = solver.add_posture_task("posture")
        posture.priority = 1
        posture.weight = 1.0
        posture.set_target_velocity(np.array([0.1, -0.1], dtype=float))

        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)

        opts_no_corridor = eik.PositionStepOptions()
        opts_no_corridor.max_steps = 20
        out_no_corridor = solver.solve_position_step(q, target, "ee_task", opts_no_corridor)

        opts_corridor = eik.PositionStepOptions()
        opts_corridor.max_steps = 20
        opts_corridor.limit_change_from_seed = True
        out_corridor = solver.solve_position_step(q, target, "ee_task", opts_corridor)

        dq_total_no = np.abs(np.asarray(out_no_corridor.q_solution, dtype=float) - q)
        dq_total_yes = np.abs(np.asarray(out_corridor.q_solution, dtype=float) - q)
        corridor_cap = 0.1 * solver.dt + 1e-9
        assert np.max(dq_total_yes) <= corridor_cap
        assert np.max(dq_total_no) > np.max(dq_total_yes) + 1e-6
    finally:
        os.unlink(urdf_path)


def test_step_no_progress_exit_for_locked_configuration():
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
        opts.max_steps = 5
        opts.no_progress_max_steps = 1
        opts.no_progress_error_tolerance = 1e-12
        opts.no_progress_dq_norm_tolerance = 1e-12
        opts.locked_joint_indices = [0, 1]
        result = solver.solve_position_step(q, target, "ee_task", opts)
        assert result.status == eik.SolverStatus.NO_PROGRESS
        assert "no progress" in result.status_message.lower()
        assert result.iterations_used < opts.max_steps
    finally:
        os.unlink(urdf_path)


def test_one_step_matches_direct_solve_velocity_update():
    """Bounded-output update in solve_position_step should match solve_velocity."""
    urdf_path = _create_two_joint_urdf(velocity_limit=0.5)
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01
        task = solver.add_frame_task("ee_task", "ee")
        task.priority = 0
        task.weight = 1.0

        q = np.array([0.2, -0.2], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[0, 3] += 0.05

        opts = eik.PositionStepOptions()
        opts.max_steps = 1
        opts.dt = solver.dt
        opts.position_gain = 30.0
        opts.orientation_gain = 30.0
        out_step = solver.solve_position_step(q, target, "ee_task", opts)
        assert out_step.status in (
            eik.SolverStatus.SUCCESS,
            eik.SolverStatus.INFEASIBLE,
            eik.SolverStatus.NO_PROGRESS,
        )

        robot.update_configuration(q)
        task.set_target_pose(target[:3, 3], target[:3, :3])
        task.update(robot)
        err = np.asarray(task.get_error(), dtype=float)
        vel = np.zeros(6, dtype=float)
        vel[:3] = opts.position_gain * err[:3]
        vel[3:] = opts.orientation_gain * err[3:]
        task.set_target_velocity(vel)
        out_vel = solver.solve_velocity(q, apply_limits=True)
        task.clear_target_velocity()
        assert out_vel.joint_velocities.shape[0] == robot.nv

        dq_step = (np.asarray(out_step.q_solution, dtype=float) - q) / opts.dt
        np.testing.assert_allclose(
            dq_step,
            np.asarray(out_vel.joint_velocities, dtype=float),
            atol=1e-6,
            rtol=0.0,
        )
    finally:
        os.unlink(urdf_path)


def test_position_ik_nullspace_active_joints_and_weights_behave_like_selection_mask():
    """Nullspace active-joint selection and per-joint weights should gate motion."""
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        q0 = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q0)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[0, 3] += 0.03  # avoid immediate convergence check short-circuit

        opts_a = eik.PositionIKOptions()
        opts_a.max_iterations = 1
        opts_a.position_gain = 0.0
        opts_a.orientation_gain = 0.0
        opts_a.nullspace_bias = np.array([0.4, -0.2], dtype=float)
        opts_a.nullspace_gain = 1.0
        opts_a.nullspace_active_joints = [0]
        opts_a.nullspace_joint_weights = np.array([6.0], dtype=float)
        out_a = solver.solve_position(q0, target, "ee", opts_a)
        assert out_a.status in (eik.SolverStatus.SUCCESS, eik.SolverStatus.INFEASIBLE)
        dq_a = np.asarray(out_a.q_solution, dtype=float) - q0
        assert abs(dq_a[0]) > abs(dq_a[1]) + 1e-8

        robot.update_configuration(q0)
        opts_b = eik.PositionIKOptions()
        opts_b.max_iterations = 1
        opts_b.position_gain = 0.0
        opts_b.orientation_gain = 0.0
        opts_b.nullspace_bias = np.array([0.4, -0.2], dtype=float)
        opts_b.nullspace_gain = 1.0
        opts_b.nullspace_active_joints = [1]
        opts_b.nullspace_joint_weights = np.array([6.0], dtype=float)
        out_b = solver.solve_position(q0, target, "ee", opts_b)
        assert out_b.status in (eik.SolverStatus.SUCCESS, eik.SolverStatus.INFEASIBLE)
        dq_b = np.asarray(out_b.q_solution, dtype=float) - q0
        assert abs(dq_b[1]) > abs(dq_b[0]) + 1e-8
    finally:
        os.unlink(urdf_path)


def test_position_step_floating_base_torso_pose_bounds_are_enforced():
    """PositionStepOptions torso pose bounds should constrain floating-base motion."""
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=True)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.02
        ee_task = solver.add_frame_task("ee_task", "ee")
        ee_task.priority = 0
        ee_task.weight = 1.0

        q = np.asarray(robot.get_current_configuration(), dtype=float)
        assert q.size >= 9, "expected free-flyer (7) + 2 arm joints for this URDF"
        q[:7] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
        q[7:] = 0.0
        robot.update_configuration(q)

        # Last two nv-indices = arm joints; floating base uses nv 0..5.
        arm_v0, arm_v1 = robot.nv - 2, robot.nv - 1
        excluded = [arm_v0, arm_v1]

        ref_torso = robot.get_frame_pose("base_link")
        pose_ee = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.asarray(pose_ee.rotation, dtype=float)
        target[:3, 3] = np.asarray(pose_ee.translation, dtype=float)
        target[0, 3] += 0.35

        x_lo, x_hi = -0.04, 0.04
        mask = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

        bounded = eik.PositionStepOptions()
        bounded.max_steps = 80
        bounded.dt = 0.02
        bounded.position_gain = 50.0
        bounded.orientation_gain = 50.0
        bounded.excluded_joint_indices = excluded
        bounded.torso_constraint.enabled = True
        bounded.torso_constraint.frame_name = "base_link"
        bounded.torso_constraint.pose_lower_bounds = np.array(
            [x_lo, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=float
        )
        bounded.torso_constraint.pose_upper_bounds = np.array(
            [x_hi, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float
        )
        bounded.torso_constraint.pose_axis_mask = mask
        bounded.torso_constraint.velocity_limits = np.full(6, 0.5, dtype=float)
        bounded.torso_constraint.acceleration_limits = np.full(6, 1.0, dtype=float)
        bounded.torso_constraint.pose_bounds_reference_pose = np.asarray(
            ref_torso.homogeneous(), dtype=float
        )

        out_bounded = solver.solve_position_step(q, target, "ee_task", bounded)
        assert out_bounded.status in (
            eik.SolverStatus.SUCCESS,
            eik.SolverStatus.INFEASIBLE,
            eik.SolverStatus.NO_PROGRESS,
        )
        robot.update_configuration(np.asarray(out_bounded.q_solution, dtype=float))
        rel_bounded = _torso_rel_state_vs_reference(ref_torso, robot.get_frame_pose("base_link"))
        assert rel_bounded[0] <= x_hi + 2e-3
        assert rel_bounded[0] >= x_lo - 2e-3

        # Without torso bounds, the same step should move farther in +x.
        robot.update_configuration(q)
        unbounded = eik.PositionStepOptions()
        unbounded.max_steps = bounded.max_steps
        unbounded.dt = bounded.dt
        unbounded.position_gain = bounded.position_gain
        unbounded.orientation_gain = bounded.orientation_gain
        unbounded.excluded_joint_indices = excluded
        out_free = solver.solve_position_step(q, target, "ee_task", unbounded)
        assert out_free.status in (
            eik.SolverStatus.SUCCESS,
            eik.SolverStatus.INFEASIBLE,
            eik.SolverStatus.NO_PROGRESS,
        )
        robot.update_configuration(np.asarray(out_free.q_solution, dtype=float))
        rel_free = _torso_rel_state_vs_reference(ref_torso, robot.get_frame_pose("base_link"))
        assert rel_free[0] > x_hi + 0.02
    finally:
        os.unlink(urdf_path)


def test_position_ik_rejects_invalid_new_torso_and_nullspace_options():
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[0, 3] += 0.01

        bad_nullspace = eik.PositionIKOptions()
        bad_nullspace.max_iterations = 1
        bad_nullspace.nullspace_bias = np.array([0.2, -0.2], dtype=float)
        bad_nullspace.nullspace_active_joints = [0, 0]
        bad_nullspace.nullspace_joint_weights = np.array([1.0], dtype=float)
        out_dup = solver.solve_position(q, target, "ee", bad_nullspace)
        assert out_dup.status == eik.SolverStatus.INVALID_INPUT
        assert "nullspace_active_joints" in out_dup.status_message

        bad_torso = eik.PositionIKOptions()
        bad_torso.max_iterations = 1
        bad_torso.torso_constraint.enabled = True
        bad_torso.torso_constraint.frame_name = "link1"
        bad_torso.torso_constraint.pose_lower_bounds = np.zeros(6, dtype=float)
        # Missing upper bounds by design.
        out_torso = solver.solve_position(q, target, "ee", bad_torso)
        assert out_torso.status == eik.SolverStatus.INVALID_INPUT
        assert "pose bounds require both lower and upper" in out_torso.status_message
    finally:
        os.unlink(urdf_path)


def test_position_ik_accepts_asymmetric_torso_pose_bounds():
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[0, 3] += 0.02

        opts = eik.PositionIKOptions()
        opts.max_iterations = 2
        opts.torso_constraint.enabled = True
        opts.torso_constraint.frame_name = "link1"
        opts.torso_constraint.orientation_mask = np.array([1.0, 1.0, 0.0], dtype=float)
        # Asymmetric bounds by design.
        opts.torso_constraint.pose_lower_bounds = np.array(
            [-0.01, -0.02, -0.03, -0.10, -0.08, -0.05], dtype=float
        )
        opts.torso_constraint.pose_upper_bounds = np.array(
            [0.03, 0.01, 0.02, 0.12, 0.06, 0.04], dtype=float
        )
        opts.torso_constraint.pose_axis_mask = np.ones(6, dtype=float)
        opts.torso_constraint.velocity_limits = np.full(6, 0.5, dtype=float)
        opts.torso_constraint.acceleration_limits = np.full(6, 1.0, dtype=float)

        out = solver.solve_position(q, target, "ee", opts)
        assert out.status != eik.SolverStatus.INVALID_INPUT
    finally:
        os.unlink(urdf_path)


def test_position_ik_floating_base_torso_pose_bounds_respected():
    """Torso pose box rows must keep bounded axes inside [lower, upper] vs seed.

    Arm joints are excluded so only the floating base can move; an aggressive +x EE
    target would otherwise require large base translation.
    """
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=True)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.02

        q = np.asarray(robot.get_current_configuration(), dtype=float)
        assert q.size >= 9, "expected free-flyer (7) + 2 arm joints for this URDF"
        q[:7] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
        q[7:] = 0.0
        robot.update_configuration(q)

        # Last two nv-indices = arm joints; floating base uses nv 0..5.
        arm_v0, arm_v1 = robot.nv - 2, robot.nv - 1
        excluded = [arm_v0, arm_v1]

        ref_torso = robot.get_frame_pose("base_link")
        pose_ee = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.asarray(pose_ee.rotation, dtype=float)
        target[:3, 3] = np.asarray(pose_ee.translation, dtype=float)
        target[0, 3] += 0.35

        x_lo, x_hi = -0.04, 0.04
        mask = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

        opts = eik.PositionIKOptions()
        opts.max_iterations = 200
        opts.dt = 0.02
        opts.position_gain = 50.0
        opts.orientation_gain = 50.0
        opts.stagnation_iterations = 20
        opts.primary_solve_mode = eik.TaskSolveMode.SCALE
        opts.primary_allow_min_error_fallback = False
        opts.excluded_joint_indices = excluded

        opts.torso_constraint.enabled = True
        opts.torso_constraint.frame_name = "base_link"
        opts.torso_constraint.orientation_mask = np.zeros(3, dtype=float)
        opts.torso_constraint.orientation_gain = 1e-3
        opts.torso_constraint.pose_lower_bounds = np.array(
            [x_lo, -1.0, -1.0, -3.15, -3.15, -3.15], dtype=float
        )
        opts.torso_constraint.pose_upper_bounds = np.array(
            [x_hi, 1.0, 1.0, 3.15, 3.15, 3.15], dtype=float
        )
        opts.torso_constraint.pose_axis_mask = mask
        opts.torso_constraint.velocity_limits = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float
        )
        opts.torso_constraint.acceleration_limits = np.array(
            [2.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float
        )
        # Explicit anchor matches seed torso (same as default); documents teleop-style API.
        opts.torso_constraint.pose_bounds_reference_pose = np.asarray(
            ref_torso.homogeneous(), dtype=float
        )

        out = solver.solve_position(q, target, "ee", opts)
        assert out.status != eik.SolverStatus.INVALID_INPUT
        q_fin = np.asarray(out.q_solution, dtype=float)
        np.testing.assert_allclose(q_fin[7:], q[7:], atol=1e-5)
        robot.update_configuration(q_fin)
        rel = _torso_rel_state_vs_reference(ref_torso, robot.get_frame_pose("base_link"))
        tol = 8e-3
        assert x_lo - tol <= rel[0] <= x_hi + tol, (
            f"torso x delta {rel[0]} outside [{x_lo}, {x_hi}] (tol={tol}); "
            f"status={out.status} msg={out.status_message!r}"
        )

        # Same seed/target without pose bounds: base should translate much farther in x.
        robot.update_configuration(q)
        ref_b = robot.get_frame_pose("base_link")
        opts_b = eik.PositionIKOptions()
        opts_b.max_iterations = 200
        opts_b.dt = 0.02
        opts_b.position_gain = 50.0
        opts_b.orientation_gain = 50.0
        opts_b.stagnation_iterations = 20
        opts_b.primary_solve_mode = eik.TaskSolveMode.SCALE
        opts_b.primary_allow_min_error_fallback = False
        opts_b.excluded_joint_indices = excluded

        out_b = solver.solve_position(q, target, "ee", opts_b)
        assert out_b.status != eik.SolverStatus.INVALID_INPUT
        q_b = np.asarray(out_b.q_solution, dtype=float)
        np.testing.assert_allclose(q_b[7:], q[7:], atol=1e-5)
        robot.update_configuration(q_b)
        rel_b = _torso_rel_state_vs_reference(ref_b, robot.get_frame_pose("base_link"))
        assert abs(rel_b[0]) > abs(rel[0]) + 0.02, (
            "expected unconstrained solve to use more base-x motion than bounded run: "
            f"bounded |dx|={abs(rel[0]):.4g}, unbounded |dx|={abs(rel_b[0]):.4g}"
        )
    finally:
        os.unlink(urdf_path)


def test_position_ik_rejects_nonfinite_torso_bounds_reference_pose():
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)

        M_bad = np.eye(4, dtype=float)
        M_bad[0, 0] = float("nan")

        opts = eik.PositionIKOptions()
        opts.max_iterations = 1
        opts.torso_constraint.enabled = True
        opts.torso_constraint.frame_name = "link1"
        opts.torso_constraint.pose_bounds_reference_pose = M_bad
        opts.torso_constraint.pose_lower_bounds = np.zeros(6, dtype=float)
        opts.torso_constraint.pose_upper_bounds = np.ones(6, dtype=float)
        opts.torso_constraint.pose_axis_mask = np.ones(6, dtype=float)
        opts.torso_constraint.velocity_limits = np.full(6, 0.5, dtype=float)
        opts.torso_constraint.acceleration_limits = np.full(6, 1.0, dtype=float)

        out = solver.solve_position(q, target, "ee", opts)
        assert out.status == eik.SolverStatus.INVALID_INPUT
        assert "pose_bounds_reference_pose" in out.status_message
    finally:
        os.unlink(urdf_path)


def test_position_ik_stagnation_classification_toggle_changes_status():
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[0, 3] += 0.05

        opts_no_progress = eik.PositionIKOptions()
        opts_no_progress.max_iterations = 6
        opts_no_progress.stagnation_iterations = 2
        opts_no_progress.position_gain = 30.0
        opts_no_progress.orientation_gain = 30.0
        opts_no_progress.classify_stagnation_as_no_progress = True
        opts_no_progress.excluded_joint_indices = [0, 1]

        out_no_progress = solver.solve_position(q, target, "ee", opts_no_progress)
        assert out_no_progress.status == eik.SolverStatus.NO_PROGRESS

        opts_infeasible = eik.PositionIKOptions()
        opts_infeasible.max_iterations = 6
        opts_infeasible.stagnation_iterations = 2
        opts_infeasible.position_gain = 30.0
        opts_infeasible.orientation_gain = 30.0
        opts_infeasible.classify_stagnation_as_no_progress = False
        opts_infeasible.excluded_joint_indices = [0, 1]

        out_infeasible = solver.solve_position(q, target, "ee", opts_infeasible)
        assert out_infeasible.status == eik.SolverStatus.INFEASIBLE
    finally:
        os.unlink(urdf_path)


def test_position_ik_rejects_invalid_torso_pose_bound_softening_fraction():
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[0, 3] += 0.01

        opts = eik.PositionIKOptions()
        opts.max_iterations = 1
        opts.torso_constraint.enabled = True
        opts.torso_constraint.frame_name = "link1"
        opts.torso_constraint.pose_lower_bounds = np.full(6, -0.1, dtype=float)
        opts.torso_constraint.pose_upper_bounds = np.full(6, 0.1, dtype=float)
        opts.torso_constraint.pose_axis_mask = np.ones(6, dtype=float)
        opts.torso_constraint.velocity_limits = np.full(6, 0.5, dtype=float)
        opts.torso_constraint.acceleration_limits = np.full(6, 1.0, dtype=float)
        opts.torso_constraint.pose_bound_softening_enabled = True
        opts.torso_constraint.pose_bound_softening_fraction = 1.5

        out = solver.solve_position(q, target, "ee", opts)
        assert out.status == eik.SolverStatus.INVALID_INPUT
        assert "pose_bound_softening_fraction" in out.status_message
    finally:
        os.unlink(urdf_path)


def test_position_ik_torso_pose_bound_softening_reduces_jump_infeasible_count():
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=True)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01

        q0 = np.asarray(robot.get_current_configuration(), dtype=float)
        q0[:7] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
        q0[7:] = 0.0
        robot.update_configuration(q0)

        torso_ref = np.asarray(robot.get_frame_pose("base_link").homogeneous(), dtype=float)
        ee_pose = robot.get_frame_pose("ee")
        base_target = np.asarray(ee_pose.homogeneous(), dtype=float)
        excluded = [robot.nv - 2, robot.nv - 1]

        def _run_sequence(enable_softening: bool) -> int:
            q = q0.copy()
            last_feasible_q = q.copy()
            prev_pos_error = float("inf")
            infeasible = 0
            for step in range(140):
                target = np.array(base_target, dtype=float, order="F")
                t = 0.03 * float(step)
                target[0, 3] += 0.14 * np.cos(t)
                target[1, 3] += 0.08 * np.sin(0.8 * t)
                if step > 0 and step % 35 == 0:
                    target[0, 3] += 0.10
                    target[1, 3] -= 0.06

                opts = eik.PositionIKOptions()
                opts.max_iterations = 10
                opts.dt = 0.01
                opts.position_gain = 10.0
                opts.orientation_gain = 10.0
                opts.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
                opts.primary_allow_min_error_fallback = True
                opts.classify_stagnation_as_no_progress = False
                opts.limit_change_from_seed = False
                opts.stagnation_iterations = 10
                opts.excluded_joint_indices = excluded
                opts.nullspace_bias = q.copy()
                opts.nullspace_gain = 0.002
                opts.nullspace_active_joints = list(range(6, robot.nv))
                opts.nullspace_joint_weights = np.ones(robot.nv - 6, dtype=float)

                opts.torso_constraint.enabled = True
                opts.torso_constraint.frame_name = "base_link"
                opts.torso_constraint.target_orientation = np.asarray(
                    robot.get_frame_pose("base_link").rotation, dtype=float
                )
                opts.torso_constraint.orientation_mask = np.array([1.0, 1.0, 0.0], dtype=float)
                opts.torso_constraint.orientation_gain = 0.05
                opts.torso_constraint.pose_lower_bounds = np.array(
                    [-0.06, -0.08, -0.10, -0.3, -0.3, -0.3], dtype=float
                )
                opts.torso_constraint.pose_upper_bounds = np.array(
                    [0.06, 0.08, 0.10, 0.3, 0.3, 0.3], dtype=float
                )
                opts.torso_constraint.pose_axis_mask = np.ones(6, dtype=float)
                opts.torso_constraint.velocity_limits = np.full(6, 0.8, dtype=float)
                opts.torso_constraint.acceleration_limits = np.full(6, 1.0, dtype=float)
                opts.torso_constraint.pose_bounds_reference_pose = torso_ref
                opts.torso_constraint.pose_bound_softening_enabled = enable_softening
                opts.torso_constraint.pose_bound_softening_fraction = 0.1

                out = solver.solve_position(q, target, "ee", opts)
                if out.status == eik.SolverStatus.SUCCESS:
                    q_next = np.asarray(out.q_solution, dtype=float).copy()
                    q_next[3:7] /= max(np.linalg.norm(q_next[3:7]), 1e-12)
                    q[:] = q_next
                    last_feasible_q[:] = q
                    prev_pos_error = float(out.position_error)
                elif out.status in (eik.SolverStatus.INFEASIBLE, eik.SolverStatus.NO_PROGRESS):
                    improved = float(out.position_error) < (prev_pos_error - 1e-6)
                    if improved:
                        q_next = np.asarray(out.q_solution, dtype=float).copy()
                        q_next[3:7] /= max(np.linalg.norm(q_next[3:7]), 1e-12)
                        q[:] = q_next
                        last_feasible_q[:] = q
                        prev_pos_error = float(out.position_error)
                    else:
                        q[:] = last_feasible_q
                    if out.status == eik.SolverStatus.INFEASIBLE:
                        infeasible += 1
                else:
                    q[:] = last_feasible_q
                robot.update_configuration(q)
            return infeasible

        infeasible_hard = _run_sequence(enable_softening=False)
        robot.update_configuration(q0)
        infeasible_soft = _run_sequence(enable_softening=True)
        assert infeasible_soft <= infeasible_hard
    finally:
        os.unlink(urdf_path)


def test_position_ik_rejects_invalid_torso_velocity_box_headroom_fraction():
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[0, 3] += 0.01

        opts = eik.PositionIKOptions()
        opts.max_iterations = 1
        opts.torso_constraint.enabled = True
        opts.torso_constraint.frame_name = "link1"
        opts.torso_constraint.pose_lower_bounds = np.full(6, -0.1, dtype=float)
        opts.torso_constraint.pose_upper_bounds = np.full(6, 0.1, dtype=float)
        opts.torso_constraint.pose_axis_mask = np.ones(6, dtype=float)
        opts.torso_constraint.velocity_limits = np.full(6, 0.5, dtype=float)
        opts.torso_constraint.acceleration_limits = np.full(6, 1.0, dtype=float)
        opts.torso_constraint.velocity_box_headroom.enabled = True
        opts.torso_constraint.velocity_box_headroom.fraction = 1.5

        out = solver.solve_position(q, target, "ee", opts)
        assert out.status == eik.SolverStatus.INVALID_INPUT
        assert "velocity_box_headroom.fraction" in out.status_message
    finally:
        os.unlink(urdf_path)


def test_position_step_orientation_only_task_uses_orientation_gain():
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01
        ori_task = solver.add_frame_task("ori_task", "ee", eik.TaskType.FRAME_ORIENTATION)
        ori_task.priority = 0
        ori_task.weight = 1.0

        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        # Y-axis rotation target to create pure orientation error.
        ang = 0.25
        Ry = np.array(
            [
                [np.cos(ang), 0.0, np.sin(ang)],
                [0.0, 1.0, 0.0],
                [-np.sin(ang), 0.0, np.cos(ang)],
            ],
            dtype=float,
        )
        target[:3, :3] = target[:3, :3] @ Ry

        opts = eik.PositionStepOptions()
        opts.max_steps = 3
        opts.position_gain = 0.0
        opts.orientation_gain = 40.0
        out = solver.solve_position_step(q, target, "ori_task", opts)
        assert out.status in (
            eik.SolverStatus.SUCCESS,
            eik.SolverStatus.INFEASIBLE,
            eik.SolverStatus.NO_PROGRESS,
        )
        dq = np.asarray(out.joint_velocities, dtype=float)
        assert np.linalg.norm(dq) > 1e-6
    finally:
        os.unlink(urdf_path)


def test_multi_target_orientation_task_uses_orientation_gain():
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01
        ori_task = solver.add_frame_task("ori_task", "ee", eik.TaskType.FRAME_ORIENTATION)
        ori_task.priority = 0
        ori_task.weight = 1.0

        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        ang = -0.20
        Ry = np.array(
            [
                [np.cos(ang), 0.0, np.sin(ang)],
                [0.0, 1.0, 0.0],
                [-np.sin(ang), 0.0, np.cos(ang)],
            ],
            dtype=float,
        )
        target[:3, :3] = target[:3, :3] @ Ry

        opts = eik.PositionStepOptions()
        opts.max_steps = 3
        targets = [eik.TaskTarget("ori_task", target, 0.0, 40.0)]
        out = solver.solve_position_step(q, targets, opts)
        assert out.status in (
            eik.SolverStatus.SUCCESS,
            eik.SolverStatus.INFEASIBLE,
            eik.SolverStatus.NO_PROGRESS,
        )
        dq = np.asarray(out.joint_velocities, dtype=float)
        assert np.linalg.norm(dq) > 1e-6
    finally:
        os.unlink(urdf_path)
