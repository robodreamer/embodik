"""Task scaling must retain mandatory motion outside the task workspace."""

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_required_inactive_joint_motion_preserves_productive_task_scale(
    seed: int, direction: float
) -> None:
    rng = np.random.default_rng(seed)
    jacobian = np.column_stack((rng.normal(size=(6, 7)), np.zeros((6, 2))))
    goal = np.r_[np.full(3, 5.0), np.zeros(3)]
    lower = -np.ones(9)
    upper = np.ones(9)
    if direction < 0:
        upper[7:] = -0.01
    else:
        lower[7:] = 0.01

    # The arm alone supplies a productive reference; mandatory motion in two
    # task-independent joints must not erase that solution.
    reference = eik.computeMultiObjectiveVelocitySolutionEigen(
        [goal],
        [jacobian[:, :7]],
        np.eye(7),
        lower[:7],
        upper[:7],
        max_iters=128,
        sr_damping=0.0,
    )
    assert reference.status == eik.SolverStatus.SUCCESS
    assert reference.task_scales[0] > 1e-5
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [goal],
        [jacobian],
        np.eye(9),
        lower,
        upper,
        max_iters=128,
        sr_damping=0.0,
    )
    assert result.status == eik.SolverStatus.SUCCESS
    velocity = np.asarray(result.solution)
    assert result.task_scales[0] > 1e-5
    np.testing.assert_allclose(result.task_scales, reference.task_scales, atol=1e-9)
    np.testing.assert_allclose(velocity[:7], reference.solution, atol=1e-9)
    np.testing.assert_allclose(velocity[7:], direction * 0.01, atol=1e-10)
    assert np.all(velocity >= lower - 1e-9)
    assert np.all(velocity <= upper + 1e-9)
    np.testing.assert_allclose(jacobian @ velocity, result.task_scales[0] * goal, atol=1e-8)
