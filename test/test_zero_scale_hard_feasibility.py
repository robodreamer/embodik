"""A zero task scale never certifies an infeasible published velocity."""

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("additional_row", [False, True])
@pytest.mark.parametrize("damping", [0.0, 0.1])
def test_zero_scale_does_not_exempt_required_hard_motion(direction, additional_row, damping):
    matrix = np.eye(2)
    lower, upper = -np.ones(2), np.ones(2)
    if additional_row:
        matrix = np.vstack((matrix, [direction, 0.0]))
        lower, upper = np.r_[lower, 0.75], np.r_[upper, 1.0]
    elif direction > 0:
        lower[0] = 0.75
    else:
        upper[0] = -0.75
    # Hard constraints are feasible, but every scaled primary target opposes
    # the required motion. Zero velocity violates the mandatory bound too.
    witness = np.array([direction * 0.75, 0.0])
    assert np.all(matrix @ witness >= lower)
    assert np.all(matrix @ witness <= upper)
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [np.array([-direction * 0.5])],
        [np.array([[1.0, 0.0]])],
        matrix,
        lower,
        upper,
        sr_damping=damping,
    )
    actual = matrix @ np.asarray(result.solution)
    violation = max(0.0, float(np.max(lower - actual)), float(np.max(actual - upper)))
    assert result.status != eik.SolverStatus.SUCCESS or violation <= 1e-6


@pytest.mark.parametrize("seed", range(32))
def test_stationary_task_compensates_mandatory_inward_bound(seed):
    rng = np.random.default_rng(seed)
    jacobian = rng.normal(size=(3, 5))
    lower, upper = -np.ones(5), np.ones(5)
    lower[0] = 0.01
    _, _, vt = np.linalg.svd(jacobian, full_matrices=True)
    free = vt[3:].T
    witness = free @ free[0]
    witness *= lower[0] / witness[0]
    assert np.linalg.norm(jacobian @ witness) <= 1e-12
    assert np.all(witness >= lower - 1e-12)
    assert np.all(witness <= upper + 1e-12)
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [np.zeros(3)],
        [jacobian],
        np.eye(5),
        lower,
        upper,
        sr_damping=0.0,
        max_iters=128,
    )
    velocity = np.asarray(result.solution)
    assert result.status == eik.SolverStatus.SUCCESS
    assert np.all(velocity >= lower - 1e-8)
    assert np.all(velocity <= upper + 1e-8)
    assert np.linalg.norm(jacobian @ velocity) <= 1e-8


@pytest.mark.parametrize("coefficient", [-1.0, 1.0])
def test_conflicting_hard_intervals_are_reported_infeasible(coefficient):
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [np.array([0.1])],
        [np.array([[1.0]])],
        np.array([[coefficient], [coefficient]]),
        np.array([0.2, -1.0]),
        np.array([1.0, 0.0]),
        sr_damping=0.0,
    )
    assert result.status == eik.SolverStatus.INFEASIBLE
    assert np.all(np.isfinite(result.solution))


@pytest.mark.parametrize("coefficient", [-2.0, 0.04375, 3.0])
def test_scalar_recovery_row_conflicts_with_joint_velocity_box(coefficient):
    matrix = np.vstack((np.eye(2), [0.0, coefficient]))
    lower, upper = np.array([-5.0, -5.0, -1e10]), np.array([5.0, 5.0, 1e10])
    if coefficient > 0:
        lower[2] = coefficient * 9.0
    else:
        upper[2] = coefficient * 9.0
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [np.zeros(2)],
        [np.eye(2)],
        matrix,
        lower,
        upper,
        sr_damping=0.0,
    )
    assert result.status == eik.SolverStatus.INFEASIBLE
