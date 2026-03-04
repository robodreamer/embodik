#!/usr/bin/env python3
"""Tests for relative pose inequality constraint"""

import pytest
import numpy as np
import tempfile
import os

import embodik

from test_ects import create_dual_arm_urdf


@pytest.fixture
def dual_arm_solver():
    path = create_dual_arm_urdf()
    robot = embodik.RobotModel(path)
    q = np.array([0.3, 0.5, 0.1, -0.3, 0.5, -0.1])
    robot.update_configuration(q)
    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    yield solver, robot, q
    os.unlink(path)


def get_relative_position(robot, frame_a, frame_b):
    """Compute relative position T_a^{-1} * T_b"""
    T_a = robot.get_frame_pose(frame_a).homogeneous()
    T_b = robot.get_frame_pose(frame_b).homogeneous()
    T_rel = np.linalg.inv(T_a) @ T_b
    return T_rel[:3, 3]


class TestRelativePoseConstraint:
    def test_bounded_relative_position(self, dual_arm_solver):
        """Apply tight bounds on relative position and verify they hold"""
        solver, robot, q = dual_arm_solver

        initial_rel_pos = get_relative_position(robot, "left_ee", "right_ee")

        bound = 0.005  # 5mm
        lower = np.array([
            initial_rel_pos[0] - bound,
            initial_rel_pos[1] - bound,
            initial_rel_pos[2] - bound,
            -10.0, -10.0, -10.0  # orientation unconstrained
        ])
        upper = np.array([
            initial_rel_pos[0] + bound,
            initial_rel_pos[1] + bound,
            initial_rel_pos[2] + bound,
            10.0, 10.0, 10.0
        ])
        mask = np.array([1, 1, 1, 0, 0, 0], dtype=np.float64)

        solver.configure_relative_pose_constraint(
            "left_ee", "right_ee", lower, upper, mask
        )

        left_task = solver.add_frame_task("left", "left_ee")
        left_task.weight = 10.0
        left_pose = robot.get_frame_pose("left_ee")
        left_task.set_target_pose(
            left_pose.translation + np.array([0.05, 0.03, 0.0]),
            left_pose.rotation
        )

        for _ in range(200):
            result = solver.solve_velocity(q)
            q = q + np.array(result.joint_velocities) * solver.dt
            robot.update_configuration(q)

        final_rel_pos = get_relative_position(robot, "left_ee", "right_ee")
        for i in range(3):
            assert final_rel_pos[i] >= lower[i] - 0.002, \
                f"Axis {i}: {final_rel_pos[i]:.4f} < lower {lower[i]:.4f}"
            assert final_rel_pos[i] <= upper[i] + 0.002, \
                f"Axis {i}: {final_rel_pos[i]:.4f} > upper {upper[i]:.4f}"

    def test_per_axis_bounds(self, dual_arm_solver):
        """Different bounds on different axes"""
        solver, robot, q = dual_arm_solver

        initial_rel_pos = get_relative_position(robot, "left_ee", "right_ee")

        lower = np.array([
            initial_rel_pos[0] - 0.002,  # tight X
            initial_rel_pos[1] - 0.05,   # loose Y
            initial_rel_pos[2] - 0.002,  # tight Z
            -10.0, -10.0, -10.0
        ])
        upper = np.array([
            initial_rel_pos[0] + 0.002,
            initial_rel_pos[1] + 0.05,
            initial_rel_pos[2] + 0.002,
            10.0, 10.0, 10.0
        ])
        mask = np.array([1, 1, 1, 0, 0, 0], dtype=np.float64)

        solver.configure_relative_pose_constraint(
            "left_ee", "right_ee", lower, upper, mask
        )

        left_task = solver.add_frame_task("left", "left_ee")
        left_task.weight = 10.0
        left_pose = robot.get_frame_pose("left_ee")
        left_task.set_target_pose(
            left_pose.translation + np.array([0.04, 0.0, 0.0]),
            left_pose.rotation
        )

        for _ in range(100):
            result = solver.solve_velocity(q)
            q = q + np.array(result.joint_velocities) * solver.dt
            robot.update_configuration(q)

        # Solver should run without errors with per-axis bounds

    def test_combined_with_absolute_task(self, dual_arm_solver):
        """Drive absolute task while relative constraint holds"""
        solver, robot, q = dual_arm_solver

        abs_task = solver.add_absolute_frame_task("abs", "left_ee", "right_ee", 0.5)
        abs_task.weight = 10.0
        abs_task.update(robot)

        initial_rel_pos = get_relative_position(robot, "left_ee", "right_ee")

        bound = 0.005
        lower = np.array([
            initial_rel_pos[0] - bound,
            initial_rel_pos[1] - bound,
            initial_rel_pos[2] - bound,
            -10.0, -10.0, -10.0
        ])
        upper = np.array([
            initial_rel_pos[0] + bound,
            initial_rel_pos[1] + bound,
            initial_rel_pos[2] + bound,
            10.0, 10.0, 10.0
        ])
        mask = np.array([1, 1, 1, 0, 0, 0], dtype=np.float64)

        solver.configure_relative_pose_constraint(
            "left_ee", "right_ee", lower, upper, mask
        )

        target_pos = abs_task.current_position + np.array([0.03, 0.0, 0.0])
        abs_task.set_target_pose(target_pos, abs_task.current_orientation)

        for _ in range(200):
            result = solver.solve_velocity(q)
            q = q + np.array(result.joint_velocities) * solver.dt
            robot.update_configuration(q)

        final_rel_pos = get_relative_position(robot, "left_ee", "right_ee")
        for i in range(3):
            assert final_rel_pos[i] >= lower[i] - 0.003, \
                f"Axis {i}: constraint violated ({final_rel_pos[i]:.4f} < {lower[i]:.4f})"
            assert final_rel_pos[i] <= upper[i] + 0.003, \
                f"Axis {i}: constraint violated ({final_rel_pos[i]:.4f} > {upper[i]:.4f})"

    def test_clear_constraint(self, dual_arm_solver):
        """Verify clear_relative_pose_constraint removes the constraint"""
        solver, robot, q = dual_arm_solver

        solver.configure_relative_pose_constraint(
            "left_ee", "right_ee",
            np.zeros(6), np.zeros(6),
            np.ones(6)
        )
        solver.clear_relative_pose_constraint()

        left_task = solver.add_frame_task("left", "left_ee")
        left_task.weight = 10.0
        left_pose = robot.get_frame_pose("left_ee")
        left_task.set_target_pose(
            left_pose.translation + np.array([0.05, 0.0, 0.0]),
            left_pose.rotation
        )

        # Should solve without constraint issues
        for _ in range(50):
            result = solver.solve_velocity(q)
            q = q + np.array(result.joint_velocities) * solver.dt
            robot.update_configuration(q)
