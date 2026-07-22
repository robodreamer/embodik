#!/usr/bin/env python3
"""Tests for the analytic singularity-conditioning objective."""

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


def _panda_arm_velocity_indices(robot: eik.RobotModel) -> list[int]:
    return [
        int(robot.get_joint_velocity_index(name))
        for name in robot.get_joint_names()
        if name.startswith("panda_joint")
    ]


def _panda_arm_config_indices(robot: eik.RobotModel) -> list[int]:
    return [
        int(robot.get_joint_config_index(name))
        for name in robot.get_joint_names()
        if name.startswith("panda_joint")
    ]


def test_manipulability_gradient_matches_finite_difference() -> None:
    config = resolve_robot_configuration("panda")
    robot = config["robot"]
    frame_name = str(config["target_link"])
    upper = np.asarray(robot.get_joint_limits()[1], dtype=float)
    elbow_index = int(robot.get_joint_config_index("panda_joint4"))
    q = np.array(
        [
            0.0,
            -0.017618,
            0.0,
            upper[elbow_index] - 1e-6,
            0.0,
            2.59284,
            0.0,
            0.02,
            0.02,
        ],
        dtype=float,
    )
    controlled_indices = _panda_arm_velocity_indices(robot)
    task = eik.ManipulabilityTask("conditioning", robot, frame_name, eik.TaskType.FRAME_POSITION)
    task.set_controlled_joint_indices(controlled_indices)
    task.set_regularization(0.03)

    def update_and_score(configuration: np.ndarray) -> float:
        robot.update_configuration(configuration)
        task.update(robot)
        return float(task.score)

    update_and_score(q)
    analytic = np.asarray(task.get_error(), dtype=float)
    finite_difference = np.zeros(len(controlled_indices), dtype=float)
    epsilon = 1e-6
    for row, velocity_index in enumerate(controlled_indices):
        tangent = np.zeros(robot.nv, dtype=float)
        tangent[velocity_index] = epsilon
        q_plus = np.asarray(robot.integrate(q, tangent, 1.0), dtype=float)
        q_minus = np.asarray(robot.integrate(q, -tangent, 1.0), dtype=float)
        finite_difference[row] = (update_and_score(q_plus) - update_and_score(q_minus)) / (
            2.0 * epsilon
        )

    expected = finite_difference / np.sqrt(
        1.0 + float(np.dot(finite_difference, finite_difference))
    )
    update_and_score(q)
    analytic = np.asarray(task.get_error(), dtype=float)

    assert np.isfinite(task.score)
    assert np.all(np.isfinite(analytic))
    assert np.linalg.norm(analytic) <= 1.0 + 1e-12
    np.testing.assert_allclose(analytic, expected, atol=2e-5, rtol=2e-4)


def test_manipulability_task_rejects_nonpositive_regularization() -> None:
    config = resolve_robot_configuration("iiwa")
    task = eik.ManipulabilityTask(
        "conditioning",
        config["robot"],
        str(config["target_link"]),
        eik.TaskType.FRAME_POSITION,
    )
    with pytest.raises(ValueError):
        task.set_regularization(0.0)
    with pytest.raises(ValueError):
        task.set_regularization(float("nan"))
    with pytest.raises(ValueError):
        task.set_joint_limit_penalty(-1.0)
    with pytest.raises(ValueError):
        task.set_joint_limit_penalty(float("nan"))
    with pytest.raises(ValueError):
        task.set_joint_limit_penalty(0.0, 0.0)


def test_joint_limit_aware_manipulability_moves_inward_near_limit() -> None:
    config = resolve_robot_configuration("panda")
    robot = config["robot"]
    frame_name = str(config["target_link"])
    controlled_indices = _panda_arm_velocity_indices(robot)
    config_index = int(robot.get_joint_config_index("panda_joint5"))
    velocity_index = int(robot.get_joint_velocity_index("panda_joint5"))
    task = eik.ManipulabilityTask(
        "conditioning",
        robot,
        frame_name,
        eik.TaskType.FRAME_POSITION,
    )
    task.set_controlled_joint_indices(controlled_indices)
    task.set_regularization(0.03)
    task.set_joint_limit_penalty(0.002)

    lower, upper = (np.asarray(value, dtype=float) for value in robot.get_joint_limits())
    q = np.asarray(robot.neutral_configuration(), dtype=float)
    row = controlled_indices.index(velocity_index)

    q[config_index] = lower[config_index] + 1e-5
    robot.update_configuration(q)
    task.update(robot)
    lower_gradient = float(np.asarray(task.get_error(), dtype=float)[row])

    q[config_index] = upper[config_index] - 1e-5
    robot.update_configuration(q)
    task.update(robot)
    upper_gradient = float(np.asarray(task.get_error(), dtype=float)[row])

    assert lower_gradient > 0.0
    assert upper_gradient < 0.0


def test_joint_limit_aware_manipulability_does_not_trade_away_limit_distance() -> None:
    config = resolve_robot_configuration("panda")
    robot = config["robot"]
    controlled_indices = _panda_arm_velocity_indices(robot)
    config_indices = _panda_arm_config_indices(robot)
    task = eik.ManipulabilityTask(
        "conditioning",
        robot,
        str(config["target_link"]),
        eik.TaskType.FRAME_POSITION,
    )
    task.set_controlled_joint_indices(controlled_indices)
    task.set_regularization(0.03)

    q = np.array(
        [
            0.033001725625,
            -0.509618685124,
            -2.727151753011,
            -1.706703998913,
            0.321117397083,
            2.368536371246,
            -0.807538834925,
            0.0,
            0.0,
        ],
        dtype=float,
    )
    lower, upper = (np.asarray(value, dtype=float) for value in robot.get_joint_limits())
    limit_descent_all = np.asarray(
        eik.joint_limit_distance_gradient(q, lower, upper, 0.04),
        dtype=float,
    )
    limit_descent = limit_descent_all[config_indices]

    robot.update_configuration(q)
    task.update(robot)
    raw_manipulability_direction = np.asarray(task.get_error(), dtype=float)
    conflict_cosine = float(
        np.dot(raw_manipulability_direction, limit_descent)
        / (np.linalg.norm(raw_manipulability_direction) * np.linalg.norm(limit_descent))
    )
    assert conflict_cosine < -0.95

    task.set_joint_limit_penalty(0.002, 0.04)
    task.update(robot)
    combined_direction = np.asarray(task.get_error(), dtype=float)

    assert float(np.dot(combined_direction, limit_descent)) > 0.0
