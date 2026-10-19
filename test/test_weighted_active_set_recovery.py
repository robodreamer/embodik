"""Failed weighted active sets recover the same feasible constrained optimum."""

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("row_scale", [1.0, 10.0, 100.0, 1000.0])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("goal,expected", [([1.0, 1.0], [0.5, 0.5]), ([10.0, 1.0], [0.8, 0.2])])
def test_weighted_feasible_recovery(row_scale, reverse, goal, expected):
    c = np.array([[row_scale, 0.0], [1.0, 1.0], [0.0, 1.0]])
    lower = np.array([-np.inf, -np.inf, 0.2])
    upper = np.array([0.9 * row_scale, 1.0, np.inf])
    if reverse:
        c, lower, upper = c[::-1].copy(), lower[::-1].copy(), upper[::-1].copy()
    result = eik.computeConstrainedWeightedVelocitySolutionEigen(
        [np.asarray(goal)],
        [np.eye(2)],
        c,
        lower,
        upper,
        solver_tolerance=1e-8,
        sr_tolerance=1e-8,
        sr_damping=0.0,
        max_iters=128,
    )
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    v = np.asarray(result.solution)
    np.testing.assert_allclose(v, expected, atol=1e-8)
    assert np.max(lower - c @ v) <= 1e-8 and np.max(c @ v - upper) <= 1e-8


def test_weighted_recovery_preserves_fixed_coordinate_and_duplicate_rows():
    c = np.array(
        [[100.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [200.0, 0.0, 0.0]]
    )
    lower = np.array([-np.inf, -np.inf, 0.2, 0.3, -np.inf])
    upper = np.array([90.0, 1.0, np.inf, 0.3, 180.0])
    result = eik.computeConstrainedWeightedVelocitySolutionEigen(
        [np.array([1.0, 1.0, 0.4])],
        [np.eye(3)],
        c,
        lower,
        upper,
        solver_tolerance=1e-8,
        sr_tolerance=1e-8,
        sr_damping=0.0,
        max_iters=128,
    )
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    np.testing.assert_allclose(result.solution, [0.5, 0.5, 0.3], atol=1e-8)


def test_weighted_recovery_does_not_return_success_for_infeasible_bounds():
    c = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    result = eik.computeConstrainedWeightedVelocitySolutionEigen(
        [np.ones(2)],
        [np.eye(2)],
        c,
        np.array([0.2, -np.inf, 0.2]),
        np.array([0.9, 0.1, np.inf]),
        solver_tolerance=1e-8,
        sr_tolerance=1e-8,
        sr_damping=0.0,
        max_iters=128,
    )
    assert result.status != eik.SolverStatus.SUCCESS
