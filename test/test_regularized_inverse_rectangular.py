"""Rectangular legacy SRINV remains finite with damped singular directions."""

from pathlib import Path

import numpy as np
import pytest

import embodik as eik
from examples.harnesses.joint_limit_recovery_harness import (
    BOUND_TOLERANCE,
    DT,
    ROBOT_URDF,
    TARGET_SPEED,
)

INSTANCE_COUNT = 1000
REGULARIZATION_EPSILON = 0.1
REGULARIZATION_FACTOR = 0.1
VELOCITY_LIMIT = 1.0
INVERSE_TOLERANCE = 1e-8


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
        np.zeros((3, 2)),
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, 0.0]]),
    ],
    ids=["tall-partial", "tall-zero", "wide-partial", "square-partial"],
)
def test_rectangular_singular_inverse_matches_regularized_gram(matrix: np.ndarray) -> None:
    columns = matrix.shape[1]
    target = matrix[:, 0] * TARGET_SPEED
    singular_vectors, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    gram = matrix @ matrix.T
    threshold_squared = REGULARIZATION_EPSILON**2
    determinant = np.linalg.det(gram)
    global_damping = (
        (1.0 - np.clip(determinant / threshold_squared, 0.0, 1.0) ** 2) * threshold_squared
        if determinant < threshold_squared
        else 0.0
    )
    per_direction = REGULARIZATION_FACTOR * np.maximum(
        0.0, 1.0 - np.minimum(singular_values / REGULARIZATION_EPSILON, 1.0) ** 2
    )
    regularized_gram = (
        gram
        + global_damping * np.eye(matrix.shape[0])
        + (singular_vectors * per_direction) @ singular_vectors.T
    )
    expected = matrix.T @ np.linalg.solve(regularized_gram, target)
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [target],
        [matrix],
        np.eye(columns),
        np.full(columns, -VELOCITY_LIMIT),
        np.full(columns, VELOCITY_LIMIT),
        sr_tolerance=REGULARIZATION_EPSILON,
        sr_damping=REGULARIZATION_FACTOR,
    )
    assert result.status == eik.SolverStatus.SUCCESS
    assert np.all(np.isfinite(result.solution))
    np.testing.assert_allclose(result.solution, expected, atol=INVERSE_TOLERANCE)


def test_exact_lower_bound_recovery_survives_repeated_native_allocations(tmp_path: Path) -> None:
    urdf = tmp_path / "two_sliders.urdf"
    urdf.write_text(ROBOT_URDF)
    for instance in range(INSTANCE_COUNT):
        robot = eik.RobotModel(str(urdf), floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.dt = DT
        runtime = eik.SolverRuntimeConfig()
        runtime.weighted_fallback_enabled = False
        solver.configure_runtime(runtime)
        solver.enable_position_limits(True)
        solver.enable_velocity_limits(True)
        lower, upper = robot.get_joint_limits()
        configuration = np.zeros(robot.nq)
        configuration[1] = lower[1]
        robot.update_configuration(configuration)
        task = solver.add_frame_task("tool_motion", "tool", eik.TaskType.FRAME_POSITION)
        task.solve_mode = eik.TaskSolveMode.SCALE
        task.allow_min_error_fallback = True
        task.set_target_velocity(np.array([TARGET_SPEED, -TARGET_SPEED, 0.0]))
        result = solver.solve_velocity(configuration, apply_limits=True)
        assert result.status == eik.SolverStatus.SUCCESS, (instance, result.status_message)
        assert result.task_used_fallback == [True]
        velocity = np.asarray(result.joint_velocities)
        assert velocity[0] > 0.0
        assert np.all(np.isfinite(velocity))
        candidate = configuration + DT * velocity
        assert np.all(candidate >= np.asarray(lower) - BOUND_TOLERANCE)
        assert np.all(candidate <= np.asarray(upper) + BOUND_TOLERANCE)
