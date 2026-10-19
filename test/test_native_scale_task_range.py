"""A zeroed joint column must not make an unsupported task ray succeed."""

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("locked_column_zeroed", [False, True])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_legacy_scale_rejects_unsupported_primary_component(
    locked_column_zeroed: bool, direction: float
) -> None:
    matrix = np.zeros((6, 2))
    matrix[5] = [1.0, 0.0 if locked_column_zeroed else 1.0]
    goal = np.array([0.0, 0.0, 0.0, 0.2, 0.0, direction * 0.099])
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [goal],
        [matrix],
        np.eye(2),
        np.array([-1.0, 0.0]),
        np.array([1.0, 0.0]),
        sr_tolerance=0.1,
    )
    assert result.task_modes_effective == [eik.TaskSolveMode.SCALE]
    assert result.task_used_fallback == [False]
    if result.status == eik.SolverStatus.SUCCESS:
        assert result.task_scales[0] <= 1e-6
        np.testing.assert_allclose(matrix @ np.asarray(result.solution), 0.0, atol=1e-8)
    else:
        assert result.status == eik.SolverStatus.NO_PROGRESS


@pytest.mark.parametrize("regularization", [1e-6, 0.1])
def test_supported_redundant_primary_retains_regularized_behavior(
    regularization: float,
) -> None:
    matrix = np.zeros((6, 2))
    matrix[5, 0] = 1.0
    goal = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.099])
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [goal],
        [matrix],
        np.eye(2),
        np.array([-1.0, 0.0]),
        np.array([1.0, 0.0]),
        sr_tolerance=regularization,
    )
    assert result.status == eik.SolverStatus.SUCCESS
    assert result.task_scales == pytest.approx([1.0])
    assert result.task_used_fallback == [False]
    velocity = np.asarray(result.solution)
    assert velocity[0] < -0.09 and abs(velocity[1]) < 1e-12
    np.testing.assert_allclose(matrix @ velocity, goal, atol=0.002)
