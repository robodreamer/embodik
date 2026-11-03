"""Task accuracy must not decide the rank of a certified hierarchy prefix."""

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("weak", [1e-4, 1e-6, 1e-8])
@pytest.mark.parametrize("redundant", [False, True])
@pytest.mark.parametrize("lower_mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.MIN_ERROR])
def test_refinement_preserves_weak_primary_image_against_lower_motion(weak, redundant, lower_mode):
    primary = np.array([[1.0, 0.0, 0.0], [0.0, weak, 0.0]])
    goal = np.array([0.5, 0.25 * weak])
    if redundant:
        primary = np.vstack([primary, 2.0 * primary[1]])
        goal = np.r_[goal, 2.0 * goal[1]]

    def solve(include_lower):
        problem = eik.VelocityProblemTrace()
        problem.constraint_matrix = np.eye(3)
        problem.lower_bounds = -np.ones(3)
        problem.upper_bounds = np.ones(3)
        problem.stationary_primary = primary
        problem.validate_primary_range = False
        configs = []
        for priority, mode in enumerate([eik.TaskSolveMode.SCALE, lower_mode]):
            config = eik.ObjectiveSolveConfig()
            config.priority = priority
            config.solve_mode = mode
            config.allow_min_error_fallback = False
            configs.append(config)
        count = 2 if include_lower else 1
        problem.objective_configs = configs[:count]
        problem.objective_jacobians = [primary, np.eye(3)][:count]
        problem.objective_goals = [goal, np.array([0.5, -0.75, 0.5])][:count]
        solver = eik.VelocitySolverConfig()
        solver.epsilon = 1e-3
        solver.precision_threshold = 1e-12
        solver.iteration_limit = 100
        solver.refine_partial_scale = True
        regularization = eik.RegularizedInverseConfig()
        regularization.epsilon = 1e-12
        regularization.regularization_factor = 0.0
        solver.regularization_config = regularization
        problem.solver_config = solver
        return eik.replay_velocity_problem(problem)

    prefix = solve(False)
    result = solve(True)
    assert prefix.status == result.status == eik.SolverStatus.SUCCESS
    # Certification allows a 1e-10 ray residual. Protect the actual achieved
    # image, including that existing residual, rather than re-solving it.
    np.testing.assert_allclose(primary @ prefix.solution, goal, atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        primary @ result.solution, primary @ prefix.solution, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(result.solution[:2], prefix.solution[:2], atol=1e-9, rtol=0)
    assert result.solution[2] == pytest.approx(0.5, abs=1e-9)
    assert result.task_scales[0] == pytest.approx(1.0)
    assert result.task_modes_effective[-1] == lower_mode
    assert not any(result.task_used_fallback)


@pytest.mark.parametrize("empty_index", [0, 1])
@pytest.mark.parametrize("refine", [False, True])
def test_empty_objective_returns_explicit_error(empty_index, refine):
    goals = [np.array([0.2, 0.3, 0.4]), np.array([0.2, 0.3, 0.4])]
    jacobians = [np.eye(3), np.eye(3)]
    goals[empty_index] = np.zeros(0)
    jacobians[empty_index] = np.zeros((0, 3))
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        goals, jacobians, np.eye(3), -np.ones(3), np.ones(3), refine_partial_scale=refine
    )
    assert result.status == eik.SolverStatus.EMPTY_PROBLEM
    assert "at least one row" in result.status_message
    assert not result.solution


def test_refined_zero_rank_prefix_leaves_all_lower_directions_available():
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [np.zeros(2), np.array([0.2, 0.3, 0.4])],
        [np.zeros((2, 3)), np.eye(3)],
        np.eye(3),
        -np.ones(3),
        np.ones(3),
        sr_tolerance=1e-10,
        sr_damping=0.0,
        refine_partial_scale=True,
    )
    assert result.status == eik.SolverStatus.SUCCESS
    np.testing.assert_allclose(result.solution, [0.2, 0.3, 0.4], atol=1e-10, rtol=0)
