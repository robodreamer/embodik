#!/usr/bin/env python3
"""Tests for linear and equality-style epsilon-box constraints."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

import embodik as eik
from embodik import _embodik_impl as _eik_native


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


def _pose_error6(anchor_pose, cur_pose) -> np.ndarray:
    err = np.zeros(6, dtype=float)
    t_ref = np.asarray(anchor_pose.translation, dtype=float)
    t_cur = np.asarray(cur_pose.translation, dtype=float)
    r_ref = np.asarray(anchor_pose.rotation, dtype=float)
    r_cur = np.asarray(cur_pose.rotation, dtype=float)
    err[:3] = t_cur - t_ref
    err[3:] = _eik_native.log3(r_ref.T @ r_cur)
    return err


def test_linear_velocity_constraints_validate_inputs():
    urdf = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        with pytest.raises(ValueError):
            solver.set_linear_velocity_constraints(
                np.zeros((2, robot.nv + 1), dtype=float),
                np.zeros(2, dtype=float),
                np.zeros(2, dtype=float),
            )
        with pytest.raises(ValueError):
            solver.set_linear_velocity_constraints(
                np.zeros((2, robot.nv), dtype=float),
                np.zeros(2, dtype=float),
                np.zeros(1, dtype=float),
            )
        with pytest.raises(ValueError):
            solver.set_linear_velocity_constraints(
                np.zeros((1, robot.nv), dtype=float),
                np.array([1.0], dtype=float),
                np.array([0.0], dtype=float),
            )
    finally:
        os.unlink(urdf)


def test_linear_velocity_constraints_lock_dq_to_zero():
    urdf = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf, floating_base=False)
        solver = eik.KinematicsSolver(robot)
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
        target[0, 3] += 0.1
        task.set_target_pose(target[:3, 3], target[:3, :3])

        C = np.eye(robot.nv, dtype=float)
        solver.set_linear_velocity_constraints(C, np.zeros(robot.nv), np.zeros(robot.nv))
        out = solver.solve_velocity(q, apply_limits=True)
        dq = np.asarray(out.joint_velocities, dtype=float)
        np.testing.assert_allclose(dq, np.zeros_like(dq), atol=1e-10, rtol=0.0)
        solver.clear_linear_velocity_constraints()
        assert solver.get_linear_velocity_constraint_rows() == 0
    finally:
        os.unlink(urdf)


def test_tight_frame_pose_constraint_rollout_no_drift():
    urdf = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01
        task = solver.add_frame_task("ee_task", "ee")
        task.priority = 0
        task.weight = 1.0
        q = np.array([0.1, 0.0], dtype=float)
        robot.update_configuration(q)

        anchor = robot.get_frame_pose("link1")
        solver.add_tight_frame_pose_constraint(
            "link1",
            np.asarray(anchor.homogeneous(), dtype=float),
            position_epsilon=1e-5,
            orientation_epsilon=1e-4,
        )

        ee_pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.asarray(ee_pose.rotation, dtype=float)
        target[:3, 3] = np.asarray(ee_pose.translation, dtype=float)
        target[2, 3] += 0.05

        opts = eik.PositionStepOptions()
        opts.max_steps = 1
        opts.position_gain = 40.0
        opts.orientation_gain = 20.0

        for _ in range(40):
            out = solver.solve_position_step(q, target, "ee_task", opts)
            assert out.status in (
                eik.SolverStatus.SUCCESS,
                eik.SolverStatus.INFEASIBLE,
                eik.SolverStatus.NO_PROGRESS,
            )
            q = np.asarray(out.q_solution, dtype=float)
            robot.update_configuration(q)
            err6 = _pose_error6(anchor, robot.get_frame_pose("link1"))
            assert np.max(np.abs(err6[:3])) <= 1e-5 + 2e-6
            assert np.max(np.abs(err6[3:])) <= 1e-4 + 2e-5
    finally:
        os.unlink(urdf)


def test_tight_point_constraint_conflict_classification_or_bounded_success():
    urdf = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01
        task = solver.add_frame_task("ee_task", "ee")
        task.priority = 0
        task.weight = 1.0
        q = np.array([0.0, 0.0], dtype=float)
        robot.update_configuration(q)

        p_anchor = np.asarray(robot.get_frame_pose("ee").translation, dtype=float)
        eps = 1e-5
        solver.add_tight_point_constraint("ee", p_anchor, position_epsilon=eps)

        target = np.eye(4, dtype=float)
        target[:3, :3] = np.asarray(robot.get_frame_pose("ee").rotation, dtype=float)
        target[:3, 3] = p_anchor.copy()
        target[0, 3] += 0.1

        opts = eik.PositionStepOptions()
        opts.max_steps = 5
        opts.position_gain = 60.0
        opts.orientation_gain = 20.0
        out = solver.solve_position_step(q, target, "ee_task", opts)

        robot.update_configuration(np.asarray(out.q_solution, dtype=float))
        p_now = np.asarray(robot.get_frame_pose("ee").translation, dtype=float)
        err = np.abs(p_now - p_anchor)
        if out.status == eik.SolverStatus.SUCCESS:
            assert np.max(err) <= eps + 2e-6
        else:
            assert out.status in (eik.SolverStatus.INFEASIBLE, eik.SolverStatus.NO_PROGRESS)
    finally:
        os.unlink(urdf)


def test_tight_constraint_invalid_inputs_raise():
    urdf = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        with pytest.raises(ValueError):
            solver.add_tight_frame_pose_constraint(
                "missing_frame", np.eye(4, dtype=float), 1e-5, 1e-4
            )
        with pytest.raises(ValueError):
            solver.add_tight_frame_pose_constraint("ee", np.eye(4, dtype=float), -1e-5, 1e-4)
        with pytest.raises(ValueError):
            solver.add_tight_point_constraint("ee", np.array([np.nan, 0.0, 0.0], dtype=float), 1e-5)
    finally:
        os.unlink(urdf)


def test_tight_constraints_allow_progress_without_nonfinite_failures():
    urdf = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01
        task = solver.add_frame_task("ee_task", "ee")
        task.priority = 0
        task.weight = 1.0

        q = np.array([0.2, 0.1], dtype=float)
        robot.update_configuration(q)
        p_anchor = np.asarray(robot.get_frame_pose("ee").translation, dtype=float)
        solver.add_tight_point_constraint(
            "ee",
            p_anchor.copy(),
            position_epsilon=1e-4,
            axis_mask=np.array([1.0, 0.0, 0.0], dtype=float),
        )

        target = np.eye(4, dtype=float)
        ee_pose = robot.get_frame_pose("ee")
        target[:3, :3] = np.asarray(ee_pose.rotation, dtype=float)
        target[:3, 3] = np.asarray(ee_pose.translation, dtype=float)
        target[2, 3] += 0.03

        opts = eik.PositionStepOptions()
        opts.max_steps = 1
        opts.position_gain = 30.0
        opts.orientation_gain = 10.0

        statuses = []
        for _ in range(20):
            out = solver.solve_position_step(q, target, "ee_task", opts)
            statuses.append(out.status)
            assert out.status in (
                eik.SolverStatus.SUCCESS,
                eik.SolverStatus.INFEASIBLE,
                eik.SolverStatus.NO_PROGRESS,
            )
            assert out.status != eik.SolverStatus.NON_FINITE_INPUT
            assert out.status != eik.SolverStatus.NUMERICAL_ERROR
            q_next = np.asarray(out.q_solution, dtype=float)
            q = q_next
            robot.update_configuration(q)
        assert any(s in (eik.SolverStatus.SUCCESS, eik.SolverStatus.NO_PROGRESS) for s in statuses)
    finally:
        os.unlink(urdf)


def test_conflicting_linear_constraints_do_not_produce_nonfinite():
    urdf = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf, floating_base=False)
        solver = eik.KinematicsSolver(robot)
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
        target[0, 3] += 0.1
        task.set_target_pose(target[:3, 3], target[:3, :3])

        C = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float)
        lower = np.array([0.2, -1e10], dtype=float)
        upper = np.array([1e10, 0.0], dtype=float)
        solver.set_linear_velocity_constraints(C, lower, upper)

        out = solver.solve_velocity(q, apply_limits=True)
        dq = np.asarray(out.joint_velocities, dtype=float)
        assert np.all(np.isfinite(dq))
        assert out.status in (
            eik.SolverStatus.INFEASIBLE,
            eik.SolverStatus.NO_PROGRESS,
            eik.SolverStatus.SUCCESS,
        )
        assert out.status != eik.SolverStatus.NON_FINITE_INPUT
        assert out.status != eik.SolverStatus.NUMERICAL_ERROR
    finally:
        os.unlink(urdf)


def test_repeated_steps_remain_stable_under_tight_constraints():
    urdf = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01
        task = solver.add_frame_task("ee_task", "ee")
        task.priority = 0
        task.weight = 1.0
        q = np.array([0.15, -0.2], dtype=float)
        robot.update_configuration(q)

        anchor_pose = np.asarray(robot.get_frame_pose("link1").homogeneous(), dtype=float)
        solver.add_tight_frame_pose_constraint(
            "link1",
            anchor_pose,
            position_epsilon=1e-5,
            orientation_epsilon=1e-4,
        )

        target = np.eye(4, dtype=float)
        ee_pose = robot.get_frame_pose("ee")
        target[:3, :3] = np.asarray(ee_pose.rotation, dtype=float)
        target[:3, 3] = np.asarray(ee_pose.translation, dtype=float)
        target[2, 3] += 0.03

        opts = eik.PositionStepOptions()
        opts.max_steps = 1
        opts.position_gain = 40.0
        opts.orientation_gain = 20.0

        times_ms = []
        for _ in range(40):
            out = solver.solve_position_step(q, target, "ee_task", opts)
            assert out.status in (
                eik.SolverStatus.SUCCESS,
                eik.SolverStatus.INFEASIBLE,
                eik.SolverStatus.NO_PROGRESS,
            )
            assert out.status != eik.SolverStatus.NON_FINITE_INPUT
            assert out.status != eik.SolverStatus.NUMERICAL_ERROR
            times_ms.append(float(out.solver_computation_time_ms))
            q = np.asarray(out.q_solution, dtype=float)
            robot.update_configuration(q)

        first_avg = float(np.mean(times_ms[:10]))
        last_avg = float(np.mean(times_ms[-10:]))
        assert np.isfinite(first_avg) and np.isfinite(last_avg)
        assert last_avg <= first_avg * 2.0 + 1e-9
    finally:
        os.unlink(urdf)
