"""A validated seed initializes optimization without fixing the answer."""

import numpy as np
import pytest

import embodik as eik


def solve(seed: np.ndarray | None):
    return eik.computeConstrainedWeightedVelocitySolutionEigen(
        [np.array([1.0, 0.0])],
        [np.eye(2)],
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -2.0], [2.0, -4.0]]),
        np.array([-1.0, -1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0, 0.0]),
        sr_tolerance=1e-8,
        solver_tolerance=1e-8,
        initial_solution=seed,
    )


@pytest.mark.parametrize("seed", [None, [0.0, 0.0], [0.5, 0.25], [-1.0, -0.5]])
def test_feasible_seed_reaches_same_constrained_optimum(seed) -> None:
    result = solve(None if seed is None else np.asarray(seed))
    assert result.status == eik.SolverStatus.SUCCESS
    np.testing.assert_allclose(result.solution, [0.8, 0.4], atol=1e-8)
    assert result.task_modes_effective == [eik.TaskSolveMode.MIN_ERROR]


@pytest.mark.parametrize(
    "seed",
    [[], [0.0], [0.0, 0.0, 0.0], [np.nan, 0.0], [np.inf, 0.0], [0.5, 0.0], [1.1, 0.55]],
)
def test_invalid_or_stale_seed_is_rejected(seed) -> None:
    result = solve(np.asarray(seed))
    assert result.status == eik.SolverStatus.INVALID_INPUT
    assert "initial solution" in result.status_message


def test_feasible_seed_preserves_one_sided_infinite_bound() -> None:
    result = eik.computeConstrainedWeightedVelocitySolutionEigen(
        [np.array([2.0])],
        [np.eye(1)],
        np.eye(1),
        np.array([0.0]),
        np.array([np.inf]),
        sr_tolerance=1e-8,
        initial_solution=np.zeros(1),
    )
    assert result.status == eik.SolverStatus.SUCCESS
    np.testing.assert_allclose(result.solution, [2.0], atol=1e-8)
