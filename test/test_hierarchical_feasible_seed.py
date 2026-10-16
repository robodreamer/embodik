"""A compatible lower objective must not block primary-null initialization."""

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("damping", [0.0, 0.1])
def test_lower_objective_does_not_erase_feasible_primary_scale(
    seed: int, direction: float, damping: float
) -> None:
    rng = np.random.default_rng(seed)
    primary = np.column_stack((rng.normal(size=(6, 7)), np.zeros((6, 2))))
    goal = np.r_[np.full(3, 5.0), np.zeros(3)]
    lower, upper = -np.ones(9), np.ones(9)
    if direction < 0:
        upper[7:] = -0.01
    else:
        lower[7:] = 0.01
    reference = eik.computeMultiObjectiveVelocitySolutionEigen(
        [goal],
        [primary],
        np.eye(9),
        lower,
        upper,
        max_iters=128,
        sr_damping=damping,
    )
    assert reference.status == eik.SolverStatus.SUCCESS
    assert reference.task_scales[0] > 1e-5
    secondary_target = np.asarray(reference.solution).copy()
    secondary_target[7:] = direction * 0.1
    # This lower target is inside every bound and has exactly the primary
    # task value already achieved by the independent reference.
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [goal, secondary_target],
        [primary, np.eye(9)],
        np.eye(9),
        lower,
        upper,
        max_iters=128,
        sr_damping=damping,
    )
    assert result.status == eik.SolverStatus.SUCCESS
    np.testing.assert_allclose(result.task_scales[0], reference.task_scales[0], atol=1e-9)
    velocity = np.asarray(result.solution)
    np.testing.assert_allclose(
        primary @ velocity, primary @ np.asarray(reference.solution), atol=1e-8
    )
    if damping == 0.0:
        np.testing.assert_allclose(primary @ velocity, result.task_scales[0] * goal, atol=1e-8)
        np.testing.assert_allclose(velocity, secondary_target, atol=1e-8)
    else:
        # Default inverse damping can leave a small secondary residual; it
        # must preserve the primary task and substantially fit this goal.
        entry_error = np.linalg.norm(np.asarray(reference.solution) - secondary_target)
        assert np.linalg.norm(velocity - secondary_target) < 0.02 * entry_error
    assert np.all(velocity >= lower - 1e-9)
    assert np.all(velocity <= upper + 1e-9)
