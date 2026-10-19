"""Exact zero hard rows have an infeasibility certificate, not a rank cutoff."""

import numpy as np
import pytest

import embodik as eik


def solve(coefficients, lower, upper, *, target=0.0, tolerance=1e-10):
    return eik.computeMultiObjectiveVelocitySolutionEigen(
        [np.array([target])],
        [np.ones((1, 1))],
        np.asarray(coefficients, dtype=float),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
        solver_tolerance=tolerance,
        solver_tight_tolerance=1e-12,
        max_iters=50,
        sr_tolerance=1e-10,
        sr_damping=0.0,
    )


@pytest.mark.parametrize("lower,upper", [(1.0, 2.0), (-2.0, -1.0)])
@pytest.mark.parametrize("coefficient", [0.0, -0.0])
def test_exact_zero_hard_row_excluding_zero_is_infeasible(lower, upper, coefficient):
    result = solve([[coefficient]], [lower], [upper])
    assert result.status == eik.SolverStatus.INFEASIBLE
    np.testing.assert_array_equal(result.solution, [0.0])
    assert result.task_scales == [0.0]
    assert result.task_errors == [0.0]
    assert result.task_modes_effective == [eik.TaskSolveMode.SCALE]
    assert result.task_used_fallback == [False]


@pytest.mark.parametrize("lower,upper", [(-1.0, 1.0), (0.0, 0.0), (1e-10, 1.0), (-1.0, -1e-10)])
def test_zero_row_retains_original_bound_tolerance(lower, upper):
    result = solve([[0.0]], [lower], [upper], target=1.0)
    assert result.status == eik.SolverStatus.SUCCESS
    np.testing.assert_array_equal(result.solution, [1.0])


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("magnitude", [1.0, 1e-12, 1e-100, 1e-200, 1e-300, np.nextafter(0.0, 1.0)])
def test_tiny_nonzero_rows_are_not_reclassified_as_zero(sign, magnitude):
    coefficient = sign * magnitude
    result = solve(
        [[coefficient]],
        [min(coefficient, 2.0 * coefficient)],
        [max(coefficient, 2.0 * coefficient)],
        target=1.0,
        tolerance=magnitude * 1e-4,
    )
    assert result.status == eik.SolverStatus.SUCCESS
    np.testing.assert_array_equal(result.solution, [1.0])
    assert result.task_scales == [1.0]


def test_mixed_ordinary_and_impossible_zero_rows_are_infeasible():
    result = solve([[1.0], [-0.0]], [-1.0, 1.0], [1.0, 2.0])
    assert result.status == eik.SolverStatus.INFEASIBLE


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_zero_row_just_outside_existing_epsilon_is_infeasible(sign):
    outside = np.nextafter(1e-10, np.inf)
    lower, upper = (outside, 1.0) if sign > 0 else (-1.0, -outside)
    assert solve([[0.0]], [lower], [upper]).status == eik.SolverStatus.INFEASIBLE
