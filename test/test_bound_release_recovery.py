"""A feasible task ray must survive release of an unnecessary saturated bound."""

import numpy as np
import pytest
from scipy.optimize import linprog

import embodik as eik


@pytest.mark.parametrize("target_scale", [0.5, 1.0, 10.0])
@pytest.mark.parametrize("duplicate_row", [False, True])
def test_zero_scale_releases_a_bound(target_scale: float, duplicate_row: bool) -> None:
    matrix = np.array([[-0.25, -0.16, -0.17, 0.0], [-0.93, -0.93, 0.0, -0.1]])
    target = target_scale * np.array([-0.04, -0.02])
    if duplicate_row:
        matrix = np.vstack([matrix, 2.0 * matrix[0]])
        target = np.r_[target, 2.0 * target[0]]
    lower, upper = np.array([-1.0, 0.0, -1.0, -1.0]), np.array([0.0, 1.0, 1.0, 0.0])
    reference = linprog(
        [0.0, 0.0, 0.0, 0.0, -1.0],
        A_eq=np.column_stack([matrix, -target]),
        b_eq=np.zeros(len(target)),
        bounds=list(zip(lower, upper)) + [(0.0, 1.0)],
        method="highs",
    )
    assert reference.success
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [target], [matrix], np.eye(4), lower, upper
    )
    velocity = np.asarray(result.solution)
    assert result.status == eik.SolverStatus.SUCCESS
    assert result.task_used_fallback == [False]
    assert result.task_modes_effective == [eik.TaskSolveMode.SCALE]
    np.testing.assert_allclose(result.task_scales[0], reference.x[-1], atol=1e-8)
    np.testing.assert_allclose(matrix @ velocity, result.task_scales[0] * target, atol=1e-8)
    assert np.all(velocity >= lower - 1e-8)
    assert np.all(velocity <= upper + 1e-8)
    assert velocity[1] > 0.0


def test_bound_release_preserves_higher_priority_task() -> None:
    matrix = np.array([[-0.25, -0.16, -0.17, 0.0], [-0.93, -0.93, 0.0, -0.1]])
    primary = np.array([[0.0, 0.0, 0.0, 1.0]])
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [np.zeros(1), np.array([-0.04, -0.02])],
        [primary, matrix],
        np.eye(4),
        np.array([-1.0, 0.0, -1.0, -1.0]),
        np.array([0.0, 1.0, 1.0, 0.0]),
    )
    velocity = np.asarray(result.solution)
    assert result.status == eik.SolverStatus.SUCCESS
    np.testing.assert_allclose(result.task_scales[1], 1.0, atol=1e-8)
    np.testing.assert_allclose(primary @ velocity, 0.0, atol=1e-8)
    np.testing.assert_allclose(matrix @ velocity, [-0.04, -0.02], atol=1e-8)


def test_later_task_cannot_undo_recovery_protection() -> None:
    matrix = np.array([[-0.25, -0.16, -0.17, 0.0, 0.0], [-0.93, -0.93, 0.0, -0.1, 0.0]])
    primary = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [np.zeros(1), np.array([-0.04, -0.02]), np.array([0.1])],
        [primary, matrix, primary],
        np.eye(5),
        np.array([-1.0, 0.0, -1.0, -1.0, -1.0]),
        np.array([0.0, 1.0, 1.0, 0.0, 1.0]),
    )
    velocity = np.asarray(result.solution)
    assert result.status == eik.SolverStatus.SUCCESS
    np.testing.assert_allclose(primary @ velocity, 0.0, atol=1e-8)
    np.testing.assert_allclose(matrix @ velocity, [-0.04, -0.02], atol=1e-8)
