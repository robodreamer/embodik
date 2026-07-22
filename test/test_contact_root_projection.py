#!/usr/bin/env python3
"""Regression tests for contact-root Jacobian projection."""

from __future__ import annotations

import os
import tempfile

import numpy as np

import embodik as eik


def _create_floating_contact_urdf(*, velocity_limit: float = 10.0) -> str:
    urdf = f"""<?xml version="1.0"?>
<robot name="floating_contact_test">
  <link name="base_link"/>

  <joint name="arm_joint" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <origin xyz="0.25 0.0 0.5" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="{velocity_limit}"/>
  </joint>
  <link name="arm_link"/>

  <joint name="hand_fixed" type="fixed">
    <parent link="arm_link"/>
    <child link="hand_link"/>
    <origin xyz="0.35 0.0 0.0" rpy="0 0 0"/>
  </joint>
  <link name="hand_link"/>

  <joint name="left_contact_fixed" type="fixed">
    <parent link="base_link"/>
    <child link="left_contact"/>
    <origin xyz="0.0 0.12 -0.8" rpy="0 0 0"/>
  </joint>
  <link name="left_contact"/>

  <joint name="right_contact_fixed" type="fixed">
    <parent link="base_link"/>
    <child link="right_contact"/>
    <origin xyz="0.0 -0.12 -0.8" rpy="0 0 0"/>
  </joint>
  <link name="right_contact"/>
</robot>
"""
    fd, path = tempfile.mkstemp(suffix=".urdf")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(urdf)
    return path


def _rx(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    R = np.eye(3, dtype=float)
    R[1, 1] = c
    R[1, 2] = -s
    R[2, 1] = s
    R[2, 2] = c
    return R


def _foot_pose_error6(anchor_pose, current_pose) -> np.ndarray:
    err = np.zeros(6, dtype=float)
    t_ref = np.asarray(anchor_pose.translation, dtype=float)
    t_cur = np.asarray(current_pose.translation, dtype=float)
    r_ref = np.asarray(anchor_pose.rotation, dtype=float)
    r_cur = np.asarray(current_pose.rotation, dtype=float)
    err[:3] = t_cur - t_ref
    err[3:] = eik.log3(r_ref.T @ r_cur)
    return err


def _make_solver_with_hand_task(urdf_path: str):
    robot = eik.RobotModel(urdf_path, floating_base=True)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    task = solver.add_frame_task("hand_pose", "hand_link", eik.TaskType.FRAME_POSE)
    task.priority = 0
    task.weight = 1.0
    return robot, solver, task


def test_point_contact_zeros_linear_velocity_and_allows_rotation():
    urdf = _create_floating_contact_urdf()
    try:
        robot, solver, task = _make_solver_with_hand_task(urdf)
        q = robot.neutral_configuration()
        robot.update_configuration(q)

        pose = robot.get_frame_pose("hand_link")
        target = np.eye(4, dtype=float)
        target[:3, 3] = np.asarray(pose.translation, dtype=float)
        target[:3, :3] = _rx(0.35) @ np.asarray(pose.rotation, dtype=float)
        task.set_target_pose(target[:3, 3], target[:3, :3])

        solver.add_contact_frame("left_contact", eik.ContactType.POINT_CONTACT)
        result = solver.solve_velocity(q, apply_limits=True)
        assert result.status in (eik.SolverStatus.SUCCESS, eik.SolverStatus.NO_PROGRESS)
        dq = np.asarray(result.joint_velocities, dtype=float)

        J = np.asarray(robot.get_frame_jacobian("left_contact"), dtype=float)
        linear_vel = J[:3, :] @ dq
        angular_vel = J[3:, :] @ dq
        assert np.linalg.norm(linear_vel) < 1e-8
        assert np.linalg.norm(angular_vel) > 1e-8
    finally:
        os.remove(urdf)


def test_rigid_contact_zeros_full_spatial_velocity():
    urdf = _create_floating_contact_urdf()
    try:
        robot, solver, task = _make_solver_with_hand_task(urdf)
        q = robot.neutral_configuration()
        robot.update_configuration(q)

        pose = robot.get_frame_pose("hand_link")
        target_pos = np.asarray(pose.translation, dtype=float) + np.array([0.08, 0.05, 0.03])
        task.set_target_pose(target_pos, np.asarray(pose.rotation, dtype=float))

        solver.add_contact_frame("left_contact", eik.ContactType.RIGID_CONTACT)
        result = solver.solve_velocity(q, apply_limits=True)
        assert result.status in (eik.SolverStatus.SUCCESS, eik.SolverStatus.NO_PROGRESS)
        dq = np.asarray(result.joint_velocities, dtype=float)

        J = np.asarray(robot.get_frame_jacobian("left_contact"), dtype=float)
        spatial_vel = J @ dq
        assert np.linalg.norm(spatial_vel) < 1e-8
    finally:
        os.remove(urdf)


def test_mixed_contact_types_enforce_expected_axes():
    urdf = _create_floating_contact_urdf()
    try:
        robot, solver, task = _make_solver_with_hand_task(urdf)
        q = robot.neutral_configuration()
        robot.update_configuration(q)

        pose = robot.get_frame_pose("hand_link")
        target = np.eye(4, dtype=float)
        target[:3, 3] = np.asarray(pose.translation, dtype=float) + np.array([0.1, -0.03, 0.0])
        target[:3, :3] = _rx(0.25) @ np.asarray(pose.rotation, dtype=float)
        task.set_target_pose(target[:3, 3], target[:3, :3])

        solver.add_contact_frame("left_contact", eik.ContactType.RIGID_CONTACT)
        solver.add_contact_frame("right_contact", eik.ContactType.POINT_CONTACT)
        result = solver.solve_velocity(q, apply_limits=True)
        assert result.status in (eik.SolverStatus.SUCCESS, eik.SolverStatus.NO_PROGRESS)
        dq = np.asarray(result.joint_velocities, dtype=float)

        J_left = np.asarray(robot.get_frame_jacobian("left_contact"), dtype=float)
        J_right = np.asarray(robot.get_frame_jacobian("right_contact"), dtype=float)
        assert np.linalg.norm(J_left @ dq) < 1e-8
        assert np.linalg.norm(J_right[:3, :] @ dq) < 1e-8
    finally:
        os.remove(urdf)


def test_rigid_contact_projection_preserves_active_joint_limit_slack():
    urdf = _create_floating_contact_urdf()
    try:
        robot, solver, task = _make_solver_with_hand_task(urdf)
        lower, upper = (np.asarray(value, dtype=float) for value in robot.get_joint_limits())
        q = np.asarray(robot.neutral_configuration(), dtype=float)
        arm_q_index = int(robot.get_joint_config_index("arm_joint"))
        arm_v_index = int(robot.get_joint_velocity_index("arm_joint"))
        q[arm_q_index] = upper[arm_q_index] - 0.01
        robot.update_configuration(q)

        runtime = solver.runtime_config()
        runtime.joint_limit_non_worsening_enabled = True
        runtime.joint_limit_non_worsening_margin = 0.04
        solver.configure_runtime(runtime)

        hand_pose = robot.get_frame_pose("hand_link")
        hand_jacobian = np.asarray(robot.get_frame_jacobian("hand_link"), dtype=float)
        outward_direction = hand_jacobian[:3, arm_v_index]
        outward_direction /= np.linalg.norm(outward_direction)
        target_position = np.asarray(hand_pose.translation, dtype=float) + 0.08 * outward_direction
        task.set_target_pose(target_position, np.asarray(hand_pose.rotation, dtype=float))

        solver.add_contact_frame("left_contact", eik.ContactType.RIGID_CONTACT)
        result = solver.solve_velocity(q, apply_limits=True)
        assert result.status in (
            eik.SolverStatus.SUCCESS,
            eik.SolverStatus.NO_PROGRESS,
        )
        velocity = np.asarray(result.joint_velocities, dtype=float)

        contact_jacobian = np.asarray(robot.get_frame_jacobian("left_contact"), dtype=float)
        assert np.linalg.norm(contact_jacobian @ velocity) < 1e-8
        assert velocity[arm_v_index] <= 1e-10
        q_next = np.asarray(robot.integrate(q, velocity * solver.dt), dtype=float)
        assert q_next[arm_q_index] <= q[arm_q_index] + 1e-12
        assert np.min(np.minimum(q_next - lower, upper - q_next)) >= -1e-10
    finally:
        os.remove(urdf)


def test_rigid_contact_active_limit_overrides_stale_outward_acceleration_history():
    urdf = _create_floating_contact_urdf()
    try:
        robot, solver, task = _make_solver_with_hand_task(urdf)
        lower, upper = (np.asarray(value, dtype=float) for value in robot.get_joint_limits())
        arm_q_index = int(robot.get_joint_config_index("arm_joint"))
        arm_v_index = int(robot.get_joint_velocity_index("arm_joint"))

        runtime = solver.runtime_config()
        runtime.joint_limit_non_worsening_enabled = True
        runtime.joint_limit_non_worsening_margin = 0.04
        solver.configure_runtime(runtime)
        solver.set_acceleration_limits(np.full(robot.nv, 0.1, dtype=float))
        solver.enable_acceleration_limits(True)
        solver.add_contact_frame("left_contact", eik.ContactType.RIGID_CONTACT)

        q_clear = np.asarray(robot.neutral_configuration(), dtype=float)
        q_clear[arm_q_index] = 0.0
        robot.update_configuration(q_clear)
        clear_pose = robot.get_frame_pose("hand_link")
        arm_direction = np.asarray(robot.get_frame_jacobian("hand_link"), dtype=float)[
            :3, arm_v_index
        ]
        arm_direction /= np.linalg.norm(arm_direction)
        task.set_target_pose(
            np.asarray(clear_pose.translation, dtype=float) + 0.30 * arm_direction,
            np.asarray(clear_pose.rotation, dtype=float),
        )
        previous_velocity = np.zeros(robot.nv, dtype=float)
        for _ in range(12):
            result = solver.solve_velocity(q_clear, apply_limits=True)
            assert result.status == eik.SolverStatus.SUCCESS
            previous_velocity = np.asarray(result.joint_velocities, dtype=float)
        assert previous_velocity[arm_v_index] > 0.01

        q_active = q_clear.copy()
        q_active[arm_q_index] = upper[arm_q_index] - 0.01
        robot.update_configuration(q_active)
        active_pose = robot.get_frame_pose("hand_link")
        task.set_target_pose(
            np.asarray(active_pose.translation, dtype=float) + 0.08 * arm_direction,
            np.asarray(active_pose.rotation, dtype=float),
        )

        result = solver.solve_velocity(q_active, apply_limits=True)
        velocity = np.asarray(result.joint_velocities, dtype=float)

        assert result.status in (
            eik.SolverStatus.SUCCESS,
            eik.SolverStatus.NO_PROGRESS,
        ), result.status_message
        contact_jacobian = np.asarray(robot.get_frame_jacobian("left_contact"), dtype=float)
        assert np.linalg.norm(contact_jacobian @ velocity) < 1e-8
        assert velocity[arm_v_index] <= 1e-10
        q_next = np.asarray(robot.integrate(q_active, velocity * solver.dt), dtype=float)
        assert q_next[arm_q_index] <= q_active[arm_q_index] + 1e-12
        assert np.min(np.minimum(q_next - lower, upper - q_next)) >= -1e-10
    finally:
        os.remove(urdf)


def _run_position_loop(mode: str, steps: int = 50):
    urdf = _create_floating_contact_urdf()
    try:
        robot, solver, task = _make_solver_with_hand_task(urdf)
        q = robot.neutral_configuration()
        robot.update_configuration(q)
        left_anchor = robot.get_frame_pose("left_contact")
        right_anchor = robot.get_frame_pose("right_contact")

        opts = eik.PositionStepOptions()
        opts.max_steps = 1
        opts.position_gain = 12.0
        opts.orientation_gain = 8.0
        opts.max_linear_speed = 0.5
        opts.max_angular_speed = 1.2

        if mode == "rigid":
            solver.add_contact_frame("left_contact", eik.ContactType.RIGID_CONTACT)
            solver.add_contact_frame("right_contact", eik.ContactType.RIGID_CONTACT)
        elif mode == "tight":
            solver.add_tight_frame_pose_constraint(
                "left_contact",
                np.asarray(left_anchor.homogeneous(), dtype=float),
                position_epsilon=2e-3,
                orientation_epsilon=np.deg2rad(0.8),
            )
            solver.add_tight_frame_pose_constraint(
                "right_contact",
                np.asarray(right_anchor.homogeneous(), dtype=float),
                position_epsilon=2e-3,
                orientation_epsilon=np.deg2rad(0.8),
            )
        else:
            raise ValueError(mode)

        max_left = 0.0
        max_right = 0.0
        status_ok = True
        for i in range(steps):
            cur = robot.get_frame_pose("hand_link")
            target = np.eye(4, dtype=float)
            target[:3, 3] = np.asarray(cur.translation, dtype=float) + np.array(
                [0.01 * np.sin(i * 0.15), 0.015 * np.cos(i * 0.2), 0.0]
            )
            target[:3, :3] = _rx(0.03 * np.sin(i * 0.1)) @ np.asarray(cur.rotation, dtype=float)
            task.set_target_pose(target[:3, 3], target[:3, :3])

            r = solver.solve_position_step(q, target, "hand_pose", opts)
            status_ok = status_ok and (
                r.status in (eik.SolverStatus.SUCCESS, eik.SolverStatus.NO_PROGRESS)
            )
            q = np.asarray(r.q_solution, dtype=float)
            robot.update_configuration(q)

            left_now = robot.get_frame_pose("left_contact")
            right_now = robot.get_frame_pose("right_contact")
            max_left = max(
                max_left, float(np.max(np.abs(_foot_pose_error6(left_anchor, left_now))))
            )
            max_right = max(
                max_right, float(np.max(np.abs(_foot_pose_error6(right_anchor, right_now))))
            )
        return max(max_left, max_right), status_ok
    finally:
        os.remove(urdf)


def test_projected_solve_zero_foot_drift_rigid():
    drift, status_ok = _run_position_loop("rigid", steps=60)
    assert status_ok
    assert drift < 1e-8


def test_contact_projection_reduces_drift_vs_tight_epsilon():
    rigid_drift, rigid_ok = _run_position_loop("rigid", steps=60)
    tight_drift, tight_ok = _run_position_loop("tight", steps=60)
    assert rigid_ok
    # Tight-epsilon mode can intermittently report NO_PROGRESS/INFEASIBLE under
    # aggressive targets; this comparison focuses on drift magnitude.
    assert tight_drift > 0.0
    assert rigid_drift < 1e-8
    assert rigid_drift < 0.2 * tight_drift
