#!/usr/bin/env python3
"""Tests for AbsoluteFrameTask grasp-offset target mapping."""

from __future__ import annotations

import os

import numpy as np
import pytest
from test_ects import create_dual_arm_urdf

import embodik


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _make_se3(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform


def _inverse_se3(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def _frame_pose(robot: embodik.RobotModel, frame_name: str) -> np.ndarray:
    pose = robot.get_frame_pose(frame_name)
    return np.asarray(pose.homogeneous(), dtype=float)


def _task_pose(task: embodik.AbsoluteFrameTask) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(task.current_orientation, dtype=float)
    transform[:3, 3] = np.asarray(task.current_position, dtype=float)
    return transform


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
    yield solver, robot
    os.unlink(path)


def _calibrated_task(solver: embodik.KinematicsSolver, robot: embodik.RobotModel):
    task = solver.add_absolute_frame_task("abs", "left_ee", "right_ee", 0.5)
    task.update(robot)
    left_fk = _frame_pose(robot, "left_ee")
    right_fk = _frame_pose(robot, "right_ee")
    object_pose = _task_pose(task)
    task.calibrate_grasp_offsets(left_fk, right_fk)
    return task, object_pose, left_fk, right_fk


def test_set_target_from_arm_targets_requires_calibration(dual_arm_solver):
    solver, _robot = dual_arm_solver
    task = solver.add_absolute_frame_task("abs", "left_ee", "right_ee", 0.5)

    assert task.grasp_offsets_calibrated is False
    with pytest.raises(RuntimeError, match="grasp offsets not calibrated"):
        task.set_target_from_arm_targets(np.eye(4), np.eye(4))


def test_calibration_requires_current_task_update(dual_arm_solver):
    solver, robot = dual_arm_solver
    task = solver.add_absolute_frame_task("abs", "left_ee", "right_ee", 0.5)

    with pytest.raises(RuntimeError, match="call update"):
        task.calibrate_grasp_offsets(
            _frame_pose(robot, "left_ee"),
            _frame_pose(robot, "right_ee"),
        )


def test_consistent_arm_targets_recover_object_target(dual_arm_solver):
    solver, robot = dual_arm_solver
    task, object_calib, left_fk, right_fk = _calibrated_task(solver, robot)

    left_in_object = _inverse_se3(object_calib) @ left_fk
    right_in_object = _inverse_se3(object_calib) @ right_fk
    object_target = _make_se3(_rot_z(0.35), np.array([0.05, -0.02, 0.04])) @ object_calib

    task.set_target_from_arm_targets(
        object_target @ left_in_object,
        object_target @ right_in_object,
    )
    task.update(robot)

    diag = task.get_grasp_divergence()
    assert diag.linear_m == pytest.approx(0.0, abs=1e-10)
    assert diag.angular_rad == pytest.approx(0.0, abs=1e-10)
    error = np.asarray(task.get_error(), dtype=float)
    np.testing.assert_allclose(error[:3], object_target[:3, 3] - object_calib[:3, 3], atol=1e-10)


def test_divergent_arm_targets_use_midpoint_and_report_diagnostic(dual_arm_solver):
    solver, robot = dual_arm_solver
    task, object_calib, left_fk, right_fk = _calibrated_task(solver, robot)

    left_in_object = _inverse_se3(object_calib) @ left_fk
    right_in_object = _inverse_se3(object_calib) @ right_fk
    object_from_left = _make_se3(object_calib[:3, :3], object_calib[:3, 3] + [0.1, 0.0, 0.0])
    object_from_right = _make_se3(
        object_calib[:3, :3] @ _rot_z(0.2),
        object_calib[:3, 3] + [0.3, 0.0, 0.0],
    )

    task.set_target_from_arm_targets(
        object_from_left @ left_in_object,
        object_from_right @ right_in_object,
    )
    task.update(robot)

    diag = task.get_grasp_divergence()
    assert diag.linear_m == pytest.approx(0.2, abs=1e-10)
    assert diag.angular_rad == pytest.approx(0.2, abs=1e-8)
    midpoint = embodik.compute_absolute_frame(
        embodik.SE3(rotation=object_from_left[:3, :3], translation=object_from_left[:3, 3]),
        embodik.SE3(rotation=object_from_right[:3, :3], translation=object_from_right[:3, 3]),
        0.5,
    )
    error = np.asarray(task.get_error(), dtype=float)
    np.testing.assert_allclose(error[:3], midpoint.translation - object_calib[:3, 3], atol=1e-10)


def test_recalibration_resets_divergence(dual_arm_solver):
    solver, robot = dual_arm_solver
    task, object_calib, left_fk, right_fk = _calibrated_task(solver, robot)
    left_in_object = _inverse_se3(object_calib) @ left_fk
    right_in_object = _inverse_se3(object_calib) @ right_fk

    task.set_target_from_arm_targets(
        _make_se3(object_calib[:3, :3], object_calib[:3, 3] + [0.0, 0.0, 0.0]) @ left_in_object,
        _make_se3(object_calib[:3, :3], object_calib[:3, 3] + [0.2, 0.0, 0.0]) @ right_in_object,
    )
    assert task.get_grasp_divergence().linear_m > 0.0

    task.calibrate_grasp_offsets(left_fk, right_fk)

    assert task.grasp_offsets_calibrated is True
    assert task.get_grasp_divergence().linear_m == pytest.approx(0.0)
    assert task.get_grasp_divergence().angular_rad == pytest.approx(0.0)
