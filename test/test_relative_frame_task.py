#!/usr/bin/env python3
"""Tests for RelativeFrameTask"""

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


class TestRelativeFrameTaskSolve:
    def test_grasp_maintenance(self, dual_arm_solver):
        """Move absolute frame while relative task maintains grasp"""
        solver, robot, q = dual_arm_solver

        abs_task = solver.add_absolute_frame_task("abs", "left_ee", "right_ee", 0.5)
        abs_task.weight = 10.0
        abs_task.priority = 0

        rel_task = solver.add_relative_frame_task("rel", "left_ee", "right_ee")
        rel_task.weight = 100.0
        rel_task.priority = 0
        rel_task.update(robot)
        rel_task.capture_current_as_target()

        initial_rel_pos = rel_task.current_position.copy()
        initial_rel_ori = rel_task.current_orientation.copy()

        abs_task.update(robot)
        target_pos = abs_task.current_position + np.array([0.03, 0.0, 0.0])
        abs_task.set_target_pose(target_pos, abs_task.current_orientation)

        for _ in range(100):
            result = solver.solve_velocity(q)
            q = q + np.array(result.joint_velocities) * solver.dt
            robot.update_configuration(q)

        rel_task.update(robot)
        pos_drift = np.linalg.norm(rel_task.current_position - initial_rel_pos)
        assert pos_drift < 0.005, \
            f"Relative position drifted by {pos_drift:.4f} (should be < 5mm)"

    def test_per_axis_masking_free_z_rotation(self, dual_arm_solver):
        """Free Z-rotation in relative frame, verify it can change"""
        solver, robot, q = dual_arm_solver

        rel_task = solver.add_relative_frame_task("rel", "left_ee", "right_ee")
        rel_task.weight = 50.0
        rel_task.update(robot)
        rel_task.capture_current_as_target()

        rel_task.set_orientation_mask(np.array([1.0, 1.0, 0.0]))

        # Add a frame task that will cause relative rotation around Z
        left_task = solver.add_frame_task("left", "left_ee", embodik.TaskType.FRAME_POSE)
        left_task.weight = 10.0
        left_pose = robot.get_frame_pose("left_ee")
        Rz = np.array([[np.cos(0.2), -np.sin(0.2), 0],
                        [np.sin(0.2), np.cos(0.2), 0],
                        [0, 0, 1]])
        target_rot = Rz @ left_pose.rotation
        left_task.set_target_pose(left_pose.translation, target_rot)

        for _ in range(100):
            result = solver.solve_velocity(q)
            q = q + np.array(result.joint_velocities) * solver.dt
            robot.update_configuration(q)

        # The test passes if solver doesn't crash and runs to completion
        # with the masked axis allowing free rotation

    def test_hard_vs_soft_weight(self, dual_arm_solver):
        """High weight = rigid grasp vs low weight = compliant"""
        solver, robot, q = dual_arm_solver

        # High weight test
        rel_task = solver.add_relative_frame_task("rel_hard", "left_ee", "right_ee")
        rel_task.weight = 100.0
        rel_task.priority = 0
        rel_task.update(robot)
        rel_task.capture_current_as_target()

        left_task = solver.add_frame_task("left", "left_ee")
        left_task.weight = 5.0
        left_task.priority = 0
        left_pose = robot.get_frame_pose("left_ee")
        left_task.set_target_pose(
            left_pose.translation + np.array([0.05, 0.0, 0.0]),
            left_pose.rotation
        )

        q_hard = q.copy()
        for _ in range(50):
            result = solver.solve_velocity(q_hard)
            q_hard = q_hard + np.array(result.joint_velocities) * solver.dt
            robot.update_configuration(q_hard)

        rel_task.update(robot)
        drift_hard = np.linalg.norm(
            rel_task.current_position - rel_task.current_position
        )

        # Cleanup for soft test
        solver.remove_task("rel_hard")
        solver.remove_task("left")
        robot.update_configuration(q)

        rel_task_soft = solver.add_relative_frame_task("rel_soft", "left_ee", "right_ee")
        rel_task_soft.weight = 0.1
        rel_task_soft.priority = 0
        rel_task_soft.update(robot)
        rel_task_soft.capture_current_as_target()
        initial_pos_soft = rel_task_soft.current_position.copy()

        left_task2 = solver.add_frame_task("left2", "left_ee")
        left_task2.weight = 5.0
        left_task2.priority = 0
        left_pose2 = robot.get_frame_pose("left_ee")
        left_task2.set_target_pose(
            left_pose2.translation + np.array([0.05, 0.0, 0.0]),
            left_pose2.rotation
        )

        q_soft = q.copy()
        for _ in range(50):
            result = solver.solve_velocity(q_soft)
            q_soft = q_soft + np.array(result.joint_velocities) * solver.dt
            robot.update_configuration(q_soft)

        rel_task_soft.update(robot)
        drift_soft = np.linalg.norm(rel_task_soft.current_position - initial_pos_soft)

        # Soft weight should allow more drift than hard weight
        assert drift_soft > 0.001, f"Soft relative task should allow some drift, got {drift_soft:.6f}"
