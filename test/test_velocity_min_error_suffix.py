"""Numerical recovery must optimize a suffix without changing its solved prefix."""

import numpy as np
import pytest
from scipy.optimize import minimize

import embodik as eik


def solve(jacobians, goals, modes, lower=None, upper=None, refine=True):
    size = jacobians[0].shape[1]
    problem = eik.VelocityProblemTrace()
    problem.constraint_matrix = np.eye(size)
    problem.lower_bounds = -np.ones(size) if lower is None else lower
    problem.upper_bounds = np.ones(size) if upper is None else upper
    problem.stationary_primary = jacobians[0]
    problem.validate_primary_range = False
    configs = []
    for priority, mode in enumerate(modes):
        config = eik.ObjectiveSolveConfig()
        config.priority = priority
        config.solve_mode = mode
        config.allow_min_error_fallback = False
        configs.append(config)
    problem.objective_configs = configs
    problem.objective_jacobians = jacobians
    problem.objective_goals = goals
    solver = eik.VelocitySolverConfig()
    solver.epsilon = 1e-3
    solver.precision_threshold = 1e-10
    solver.iteration_limit = 128
    solver.refine_partial_scale = refine
    regularization = eik.RegularizedInverseConfig()
    regularization.epsilon = 1e-12
    regularization.regularization_factor = 0.0
    solver.regularization_config = regularization
    problem.solver_config = solver
    return eik.replay_velocity_problem(problem)


@pytest.mark.parametrize("seed", range(100))
@pytest.mark.parametrize("refine", [False, True])
def test_near_parallel_min_error_suffix_preserves_primary_and_bounds(seed, refine):
    rng = np.random.default_rng(20261021 + seed)
    primary = rng.normal(size=(1, 4))
    primary /= np.linalg.norm(primary)
    lower_task = np.vstack(
        [primary[0] + 10 ** rng.uniform(-8, -4) * rng.normal(size=4), rng.normal(size=4)]
    )
    goal = lower_task @ rng.uniform(-1, 1, size=4)
    modes = [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.MIN_ERROR]
    result = solve([primary, lower_task], [np.array([0.2]), goal], modes, refine=refine)
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    x = np.asarray(result.solution)
    np.testing.assert_allclose(primary @ x, [0.2], atol=1e-10, rtol=0)
    assert np.max(np.abs(x)) <= 1 + 1e-10
    np.testing.assert_allclose(result.task_scales, [1, -1], atol=1e-10, rtol=0)
    assert list(result.task_modes_effective) == modes
    assert not any(result.task_used_fallback)
    if seed < 10:
        completed = solve(
            [primary, lower_task, np.eye(4)],
            [np.array([0.2]), goal, np.full(4, 0.75)],
            modes + [eik.TaskSolveMode.MIN_ERROR],
            refine=refine,
        )
        assert completed.status == eik.SolverStatus.SUCCESS, completed.status_message
        np.testing.assert_allclose(primary @ completed.solution, [0.2], atol=1e-10, rtol=0)
        np.testing.assert_allclose(
            lower_task @ completed.solution, lower_task @ x, atol=1e-10, rtol=0
        )
        assert not any(completed.task_used_fallback)
        # Independent constrained least-squares reference, initialized at an
        # analytic feasible point. Joint-null choices need not be identical.
        reference = minimize(
            lambda q: 0.5 * np.linalg.norm(lower_task @ q - goal) ** 2,
            0.2 * primary[0],
            jac=lambda q: lower_task.T @ (lower_task @ q - goal),
            bounds=[(-1, 1)] * 4,
            constraints={
                "type": "eq",
                "fun": lambda q: primary @ q - 0.2,
                "jac": lambda q: primary,
            },
            method="SLSQP",
            options={"ftol": 1e-13, "maxiter": 300},
        )
        assert reference.success, reference.message
        assert 0.5 * np.linalg.norm(lower_task @ x - goal) ** 2 <= reference.fun + 1e-8


@pytest.mark.parametrize("seed", range(20))
def test_recovery_preserves_projected_scale_prefix_and_mandatory_inward_row(seed):
    rng = np.random.default_rng(20261022 + seed)
    primary = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
    second = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
    lower_task = np.vstack([second[1] + 1e-6 * rng.normal(size=5), rng.normal(size=5)])
    lower = np.array([-1.0, -1.0, 0.2, -1.0, -1.0])
    matrices = [primary, second]
    goals = [np.array([0.5]), np.array([1.0, 0.25])]
    modes = [eik.TaskSolveMode.SCALE] * 2
    prefix = solve(matrices, goals, modes, lower=lower)
    result = solve(
        matrices + [lower_task],
        goals + [lower_task @ rng.uniform(-1, 1, size=5)],
        modes + [eik.TaskSolveMode.MIN_ERROR],
        lower=lower,
    )
    assert prefix.status == result.status == eik.SolverStatus.SUCCESS
    x = np.asarray(result.solution)
    np.testing.assert_allclose(x[:2], np.asarray(prefix.solution)[:2], atol=1e-10, rtol=0)
    np.testing.assert_allclose(x[:2], [0.5, 0.25], atol=1e-10, rtol=0)
    assert np.all(x >= lower - 1e-10)
    assert np.all(x <= 1 + 1e-10)
    assert list(result.task_modes_effective) == modes + [eik.TaskSolveMode.MIN_ERROR]
    assert not any(result.task_used_fallback)


@pytest.mark.parametrize("weak", [1e-4, 1e-6, 1e-8])
@pytest.mark.parametrize("mandatory_inward", [False, True])
def test_suffix_preserves_legacy_projected_scale_prefix(weak, mandatory_inward):
    jacobians = [np.array([[1.0, 0.0, 0.0, 0.0]]), np.eye(4)[:2]]
    goals = [np.array([0.5]), np.array([1.0, 0.25])]
    modes = [eik.TaskSolveMode.SCALE] * 2
    lower = np.array([-1.0, -1.0, 0.2 if mandatory_inward else -1.0, -1.0])
    prefix = solve(jacobians, goals, modes, lower=lower)
    result = solve(
        jacobians + [np.array([[1.0, 2.0, weak, 0.0], [-1.0, -2.0, 0.0, weak]])],
        goals + [np.array([-0.4, 0.3])],
        modes + [eik.TaskSolveMode.MIN_ERROR],
        lower=lower,
    )
    assert prefix.status == result.status == eik.SolverStatus.SUCCESS
    x = np.asarray(result.solution)
    np.testing.assert_allclose(x[:2], [0.5, 0.25], atol=1e-10, rtol=0)
    np.testing.assert_allclose(x[:2], np.asarray(prefix.solution)[:2], atol=1e-10, rtol=0)
    assert np.all(x >= lower - 1e-10)
    assert np.all(x <= 1 + 1e-10)
    assert list(result.task_modes_effective) == modes + [eik.TaskSolveMode.MIN_ERROR]
    assert not any(result.task_used_fallback)


def test_multiple_min_error_suffix_bands_retain_completed_images():
    primary = np.array([[1.0, 0.0, 0.0, 0.0]])
    near_primary = np.array([[1.0, 1e-6, 0.0, 0.0], [-1.0, 0.0, 1e-6, 0.0]])
    modes = [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.MIN_ERROR]
    matrices = [primary, near_primary]
    goals = [np.array([0.2]), np.array([-0.4, 0.3])]
    prefix = solve(matrices, goals, modes)
    result = solve(
        matrices + [np.eye(4)], goals + [np.ones(4) * 0.75], modes + [eik.TaskSolveMode.MIN_ERROR]
    )
    assert prefix.status == result.status == eik.SolverStatus.SUCCESS
    x = np.asarray(result.solution)
    np.testing.assert_allclose(primary @ x, [0.2], atol=1e-10, rtol=0)
    np.testing.assert_allclose(near_primary @ x, near_primary @ prefix.solution, atol=1e-10, rtol=0)
    assert x[-1] == pytest.approx(0.75, abs=1e-8)
    assert not any(result.task_used_fallback)
