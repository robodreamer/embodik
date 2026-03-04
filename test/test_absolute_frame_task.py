#!/usr/bin/env python3
"""Tests for AbsoluteFrameTask"""

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
    solver.enable_velocity_limits(False)
    solver.enable_position_limits(False)
    yield solver, robot, q
    os.unlink(path)


class TestAbsoluteFrameTaskSolve:
    def test_basic_solve_moves_toward_target(self, dual_arm_solver):
        solver, robot, q = dual_arm_solver
        abs_task = solver.add_absolute_frame_task("abs", "left_ee", "right_ee", 0.5)
        abs_task.weight = 10.0

        abs_task.update(robot)
        initial_pos = abs_task.current_position.copy()
        target_pos = initial_pos + np.array([0.005, 0.0, 0.0])
        abs_task.set_target_pose(target_pos, abs_task.current_orientation)

        for _ in range(500):
            result = solver.solve_velocity(q)
            q = q + np.array(result.joint_velocities) * solver.dt
            robot.update_configuration(q)

        abs_task.update(robot)
        final_pos = abs_task.current_position
        moved = np.linalg.norm(final_pos - initial_pos)
        assert moved > 0.001, f"Solver should move the absolute frame, moved {moved:.6f}"

    def test_alpha_variation_serial_left(self, dual_arm_solver):
        """With alpha=1.0, only left arm Jacobian columns are active"""
        solver, robot, q = dual_arm_solver
        abs_task = solver.add_absolute_frame_task("abs", "left_ee", "right_ee", 1.0)
        abs_task.weight = 10.0

        abs_task.update(robot)
        target_pos = abs_task.current_position + np.array([0.02, 0.0, 0.0])
        abs_task.set_target_pose(target_pos, abs_task.current_orientation)

        q_init = q.copy()
        for _ in range(200):
            result = solver.solve_velocity(q)
            q = q + np.array(result.joint_velocities) * solver.dt
            robot.update_configuration(q)

        left_change = np.linalg.norm(q[:3] - q_init[:3])
        right_change = np.linalg.norm(q[3:] - q_init[3:])
        assert left_change > 0.001, f"Left arm should move, got {left_change:.6f}"
        assert right_change < left_change, \
            f"Right arm moved more than left: {right_change:.4f} vs {left_change:.4f}"

    def test_alpha_variation_serial_right(self, dual_arm_solver):
        """With alpha=0.0, only right arm Jacobian columns are active"""
        solver, robot, q = dual_arm_solver
        abs_task = solver.add_absolute_frame_task("abs", "left_ee", "right_ee", 0.0)
        abs_task.weight = 10.0

        abs_task.update(robot)
        target_pos = abs_task.current_position + np.array([0.02, 0.0, 0.0])
        abs_task.set_target_pose(target_pos, abs_task.current_orientation)

        q_init = q.copy()
        for _ in range(200):
            result = solver.solve_velocity(q)
            q = q + np.array(result.joint_velocities) * solver.dt
            robot.update_configuration(q)

        left_change = np.linalg.norm(q[:3] - q_init[:3])
        right_change = np.linalg.norm(q[3:] - q_init[3:])
        assert right_change > 0.001, f"Right arm should move, got {right_change:.6f}"
        assert left_change < right_change, \
            f"Left arm moved more than right: {left_change:.4f} vs {right_change:.4f}"

    def test_set_alpha_runtime(self, dual_arm_solver):
        """Verify alpha can be changed at runtime"""
        solver, robot, q = dual_arm_solver
        abs_task = solver.add_absolute_frame_task("abs", "left_ee", "right_ee", 0.5)
        assert abs_task.get_alpha() == pytest.approx(0.5)

        abs_task.set_alpha(0.75)
        assert abs_task.get_alpha() == pytest.approx(0.75)

    def test_with_posture_regularization(self, dual_arm_solver):
        """Verify absolute task works alongside posture task"""
        solver, robot, q = dual_arm_solver
        abs_task = solver.add_absolute_frame_task("abs", "left_ee", "right_ee", 0.5)
        abs_task.weight = 10.0
        abs_task.priority = 0

        posture = solver.add_posture_task("posture")
        posture.set_target_configuration(q.copy())
        posture.weight = 0.01
        posture.priority = 1

        abs_task.update(robot)
        initial_pos = abs_task.current_position.copy()
        target_pos = abs_task.current_position + np.array([0.005, 0.0, 0.0])
        abs_task.set_target_pose(target_pos, abs_task.current_orientation)

        for _ in range(500):
            result = solver.solve_velocity(q)
            q = q + np.array(result.joint_velocities) * solver.dt
            robot.update_configuration(q)

        abs_task.update(robot)
        moved = np.linalg.norm(abs_task.current_position - initial_pos)
        assert moved > 0.001, f"Solver should move the absolute frame, moved {moved:.6f}"
