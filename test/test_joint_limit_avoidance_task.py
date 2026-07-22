#!/usr/bin/env python3
"""Tests for smooth joint-limit avoidance as an explicit hierarchy task."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import embodik as eik

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from utils.robot_models import resolve_robot_configuration  # noqa: E402


def _panda_elbow() -> tuple[eik.RobotModel, int, int, np.ndarray, np.ndarray]:
    robot = resolve_robot_configuration("panda")["robot"]
    config_index = int(robot.get_joint_config_index("panda_joint4"))
    velocity_index = int(robot.get_joint_velocity_index("panda_joint4"))
    lower, upper = (np.asarray(value, dtype=float) for value in robot.get_joint_limits())
    return robot, config_index, velocity_index, lower, upper


def test_joint_limit_avoidance_is_smooth_signed_and_inactive_when_clear() -> None:
    robot, config_index, velocity_index, lower, upper = _panda_elbow()
    task = eik.JointLimitAvoidanceTask("limits", robot, [velocity_index])
    task.set_activation_margin(0.05)

    def error_at(value: float) -> float:
        q = np.asarray(robot.neutral_configuration(), dtype=float)
        q[config_index] = value
        robot.update_configuration(q)
        task.update(robot)
        return float(np.asarray(task.get_error(), dtype=float)[0])

    assert error_at(lower[config_index] + 1e-6) > 0.999
    assert error_at(upper[config_index] - 1e-6) < -0.999
    assert error_at(0.5 * (lower[config_index] + upper[config_index])) == 0.0

    just_inside = error_at(lower[config_index] + 0.05 - 1e-6)
    just_outside = error_at(lower[config_index] + 0.05 + 1e-6)
    assert 0.0 < just_inside < 1e-6
    assert just_outside == 0.0

    inactive_jacobian = np.asarray(task.get_jacobian(), dtype=float)
    assert inactive_jacobian.shape == (1, robot.nv)
    np.testing.assert_allclose(inactive_jacobian, 0.0)

    error_at(lower[config_index] + 1e-6)
    active_jacobian = np.asarray(task.get_jacobian(), dtype=float)
    np.testing.assert_allclose(active_jacobian[0, velocity_index], 1.0)
    np.testing.assert_allclose(np.delete(active_jacobian[0], velocity_index), 0.0)


def test_inactive_joint_limit_avoidance_leaves_lower_priority_motion_free() -> None:
    robot, config_index, velocity_index, lower, upper = _panda_elbow()
    q = np.asarray(robot.neutral_configuration(), dtype=float)
    q[config_index] = 0.5 * (lower[config_index] + upper[config_index])
    robot.update_configuration(q)

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    avoidance = solver.add_joint_limit_avoidance_task("limits", [velocity_index])
    avoidance.priority = 0
    avoidance.solve_mode = eik.TaskSolveMode.MIN_ERROR
    avoidance.set_activation_margin(0.05)
    tracking = solver.add_joint_task(
        "elbow_tracking", "panda_joint4", target_value=q[config_index] + 0.1
    )
    tracking.priority = 1
    tracking.solve_mode = eik.TaskSolveMode.MIN_ERROR

    result = solver.solve_velocity(q, apply_limits=True)
    velocity = np.asarray(result.joint_velocities, dtype=float)

    assert result.status == eik.SolverStatus.SUCCESS
    assert velocity[velocity_index] > 1e-3


def test_joint_limit_avoidance_drives_inward_without_crossing_limits() -> None:
    robot, config_index, velocity_index, lower, upper = _panda_elbow()
    q = np.asarray(robot.neutral_configuration(), dtype=float)
    q[config_index] = upper[config_index] - 1e-6
    robot.update_configuration(q)

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    task = solver.add_joint_limit_avoidance_task("limits", [velocity_index])
    task.priority = 0
    task.weight = 0.012
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    task.set_activation_margin(0.002)

    result = solver.solve_velocity(q, apply_limits=True)
    velocity = np.asarray(result.joint_velocities, dtype=float)
    q_next = np.asarray(robot.integrate(q, velocity, solver.dt), dtype=float)

    assert result.status == eik.SolverStatus.SUCCESS
    assert velocity[velocity_index] < 0.0
    assert lower[config_index] <= q_next[config_index] <= upper[config_index]
    assert upper[config_index] - q_next[config_index] > 1e-6


def test_joint_limit_avoidance_validates_parameters() -> None:
    robot, _config_index, velocity_index, _lower, _upper = _panda_elbow()
    task = eik.JointLimitAvoidanceTask("limits", robot, [velocity_index])
    with pytest.raises(ValueError):
        task.set_activation_margin(0.0)
    with pytest.raises(ValueError):
        task.set_activation_margin(float("nan"))
    with pytest.raises(ValueError):
        task.set_controlled_joint_indices([robot.nv])
