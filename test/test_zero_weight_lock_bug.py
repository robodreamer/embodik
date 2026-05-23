#!/usr/bin/env python3
"""Reproduce: solve_velocity with all zero-weight tasks permanently locks solver.

Bug: Calling solve_velocity once with all zero-weight tasks causes subsequent
calls with non-zero weights to still return zero.
"""

import os
import tempfile

import numpy as np

import embodik


def create_test_urdf():
    urdf = """<?xml version="1.0"?>
<robot name="test">
  <link name="base"><inertial><mass value="1"/><origin xyz="0 0 0"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial></link>
  <link name="link1"><inertial><mass value="1"/><origin xyz="0 0 0.5"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial></link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="link1"/><origin xyz="0 0 1"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" velocity="1" effort="10"/></joint>
  <link name="ee"><inertial><mass value="0.1"/><origin xyz="0 0 0"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial></link>
  <joint name="j2" type="revolute"><parent link="link1"/><child link="ee"/><origin xyz="0 0 1"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" velocity="1" effort="10"/></joint>
</robot>"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(urdf)
        return f.name


def test_zero_weight_then_nonzero():
    """Reproduce: solve with weight=0, then weight=1 should produce non-zero dq."""
    from test_ects import create_dual_arm_urdf

    path = create_dual_arm_urdf()
    try:
        model = embodik.RobotModel(path, floating_base=False)
        solver = embodik.KinematicsSolver(model)
        solver.dt = 0.01
        task = solver.add_frame_task("left", "left_ee")
        task.priority = 0

        q = np.array([0.3, 0.5, 0.1, -0.3, 0.5, -0.1])
        model.update_configuration(q)
        current_pose = model.get_frame_pose("left_ee")
        target_pose = np.eye(4)
        target_pose[:3, :3] = current_pose.rotation
        target_pose[:3, 3] = current_pose.translation + np.array([0.1, 0.0, 0.0])
        task.set_target_pose(
            target_pose[:3, 3],
            target_pose[:3, :3],
        )

        # Step 1: All zero weight, call solve_velocity
        task.weight = 0.0
        r1 = solver.solve_velocity(q, apply_limits=True)
        dq1 = np.array(r1.joint_velocities)
        assert np.allclose(dq1, 0.0), "With weight=0, dq should be zero"
        print("Step 1 (weight=0): |dq| =", np.linalg.norm(dq1))

        # Step 2: Set non-zero weight, call solve_velocity again
        task.weight = 1.0
        r2 = solver.solve_velocity(q, apply_limits=True)
        dq2 = np.array(r2.joint_velocities)
        print("Step 2 (weight=1): |dq| =", np.linalg.norm(dq2))

        # BUG: dq2 should be non-zero (we have pose error), but if bug exists it stays zero
        if np.linalg.norm(dq2) < 1e-10:
            print("BUG CONFIRMED: Step 2 returned zero despite weight=1 and pose error")
            assert False, "Solver locked: weight=1 should produce non-zero dq"
        print("OK: Step 2 produced non-zero dq as expected")
    finally:
        os.unlink(path)


def test_no_solve_before_weights():
    """Control: set weight=1 and target_velocity BEFORE any solve - should work."""
    from test_ects import create_dual_arm_urdf

    path = create_dual_arm_urdf()
    try:
        model = embodik.RobotModel(path, floating_base=False)
        solver = embodik.KinematicsSolver(model)
        solver.dt = 0.01
        task = solver.add_frame_task("left", "left_ee")
        task.priority = 0

        q = np.array([0.3, 0.5, 0.1, -0.3, 0.5, -0.1])
        # Use set_target_velocity directly (like WBC does) - explicit non-zero velocity
        task.set_target_velocity(np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]))
        task.weight = 1.0

        r = solver.solve_velocity(q, apply_limits=True)
        dq = np.array(r.joint_velocities)
        print("No prior solve (weight=1, target_vel=[0.5,0,0,0,0,0]): |dq| =", np.linalg.norm(dq))
        assert np.linalg.norm(dq) > 1e-10, "Should produce non-zero dq"
    finally:
        os.unlink(path)


if __name__ == "__main__":
    print("=== Control: weight=1 before any solve ===")
    test_no_solve_before_weights()
    print("\n=== Bug repro: weight=0 solve, then weight=1 solve ===")
    test_zero_weight_then_nonzero()
