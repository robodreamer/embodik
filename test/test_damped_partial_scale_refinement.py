"""An off-ray damped seed must not prevent explicitly requested SCALE refinement."""

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("protect_x", [False, True])
def test_refinement_builds_exact_ray_without_relying_on_damped_partial_seed(protect_x):
    trace = eik.VelocityProblemTrace()
    trace.backend_column_scale = np.ones(3)
    trace.constraint_matrix = np.vstack([np.eye(3), [-1.0, 0.1, 0.1]])
    trace.lower_bounds = np.array([-2.0, -2.0, -1.0, -1e10])
    trace.upper_bounds = np.array([2.0, 2.0, 1.0, 0.01])
    trace.maximum_softening_factors = np.array([])
    trace.stationary_primary = np.empty((0, 3))
    trace.validate_primary_range = False
    primary = eik.ObjectiveSolveConfig()
    primary.solve_mode = eik.TaskSolveMode.SCALE
    primary.allow_min_error_fallback = False
    jacobians = [np.diag([0.0, 0.0, 1.0])]
    goals = [np.array([0.0, 0.0, 10.0])]
    configs = [primary]
    if protect_x:
        protected = eik.ObjectiveSolveConfig()
        protected.priority = 0
        protected.allow_min_error_fallback = False
        primary.priority = 1
        jacobians.insert(0, np.array([[1.0, 0.0, 0.0]]))
        goals.insert(0, np.zeros(1))
        configs.insert(0, protected)
    trace.objective_jacobians = jacobians
    trace.objective_goals = goals
    trace.objective_configs = configs
    config = eik.VelocitySolverConfig()
    regularization = eik.RegularizedInverseConfig()
    regularization.epsilon = 0.1
    regularization.regularization_factor = 0.1
    config.regularization_config = regularization
    trace.solver_config = config
    baseline = eik.replay_velocity_problem(trace)
    config.refine_partial_scale = True
    trace.solver_config = config
    refined = eik.replay_velocity_problem(trace)
    velocity = np.asarray(refined.solution)
    assert refined.status == eik.SolverStatus.SUCCESS
    assert refined.task_modes_effective[-1] == eik.TaskSolveMode.SCALE
    assert not refined.task_used_fallback[-1]
    # The yaw speed bound proves alpha <= 0.1; a feasible witness reaches it.
    np.testing.assert_allclose(refined.task_scales[-1], 0.1, atol=1e-9, rtol=0)
    np.testing.assert_allclose(jacobians[-1] @ velocity, 0.1 * goals[-1], atol=1e-9, rtol=0)
    assert refined.task_scales[-1] > baseline.task_scales[-1] + 0.01
    assert np.all(trace.constraint_matrix @ velocity >= trace.lower_bounds - 1e-10)
    assert np.all(trace.constraint_matrix @ velocity <= trace.upper_bounds + 1e-10)
    if protect_x:
        assert abs(velocity[0]) <= 1e-10


@pytest.mark.parametrize("protect_prefix", [False, True])
@pytest.mark.parametrize("certified_scale", [0.25, 1.0])
@pytest.mark.parametrize("mandatory_inward_motion", [False, True])
def test_refinement_certifies_a_reported_full_scale_that_is_actually_off_ray(
    protect_prefix, certified_scale, mandatory_inward_motion
):
    n = 3 if protect_prefix else 2
    trace = eik.VelocityProblemTrace()
    trace.backend_column_scale = np.ones(n)
    trace.constraint_matrix = np.eye(n)
    trace.lower_bounds, trace.upper_bounds = -np.ones(n), np.ones(n)
    upper = np.asarray(trace.upper_bounds).copy()
    upper[-1] = certified_scale
    trace.upper_bounds = upper
    if mandatory_inward_motion:
        lower = np.asarray(trace.lower_bounds).copy()
        lower[-2] = 0.1
        trace.lower_bounds = lower
    trace.maximum_softening_factors = np.array([])
    trace.stationary_primary = np.empty((0, n))
    trace.validate_primary_range = False
    J = np.diag([1.0, 0.01])
    if protect_prefix:
        J = np.column_stack((np.zeros(2), J))
    goal = np.array([0.5, 0.01])
    config = eik.ObjectiveSolveConfig()
    config.solve_mode = eik.TaskSolveMode.SCALE
    config.allow_min_error_fallback = False
    trace.objective_jacobians = [J]
    trace.objective_goals = [goal]
    trace.objective_configs = [config]
    if protect_prefix:
        prefix = eik.ObjectiveSolveConfig()
        prefix.solve_mode = eik.TaskSolveMode.SCALE
        prefix.allow_min_error_fallback = False
        config.priority = 1
        trace.objective_jacobians = [np.array([[1.0, 0.0, 0.0]]), J]
        trace.objective_goals = [np.array([0.2]), goal]
        trace.objective_configs = [prefix, config]
    runtime = eik.VelocitySolverConfig()
    regularization = eik.RegularizedInverseConfig()
    regularization.epsilon, regularization.regularization_factor = 0.1, 0.1
    runtime.regularization_config = regularization
    trace.solver_config = runtime
    baseline = eik.replay_velocity_problem(trace)
    assert baseline.task_scales[-1] == pytest.approx(1.0)
    assert np.linalg.norm(J @ np.asarray(baseline.solution) - goal) > 0.001
    runtime.refine_partial_scale = True
    trace.solver_config = runtime
    refined = eik.replay_velocity_problem(trace)
    assert refined.status == eik.SolverStatus.SUCCESS
    np.testing.assert_allclose(
        J @ np.asarray(refined.solution), certified_scale * goal, atol=1e-9, rtol=0
    )
    assert refined.task_scales[-1] == pytest.approx(certified_scale, abs=1e-9)
    assert not any(refined.task_used_fallback)
    assert np.all(np.abs(refined.solution) <= 1.0 + 1e-10)
    if protect_prefix:
        assert refined.solution[0] == pytest.approx(0.2, abs=1e-9)


@pytest.mark.parametrize("coupling", [6e-11, 7e-11, 9e-11])
def test_ray_repair_keeps_verified_progress_when_numeric_search_has_no_blocker(coupling):
    # The coupled face leaves a tiny free direction. Its norm exceeds the
    # stationary threshold while every individual row slope is below the
    # blocker threshold. Discarding the checkpoint restores an off-ray result.
    trace = eik.VelocityProblemTrace()
    trace.backend_column_scale = np.ones(4)
    trace.constraint_matrix = np.vstack((np.eye(4), [coupling] * 3 + [1.0]))
    trace.lower_bounds = np.r_[-np.ones(4), -10.0]
    trace.upper_bounds = np.r_[np.ones(4), 0.4]
    trace.maximum_softening_factors = np.array([])
    trace.stationary_primary = np.empty((0, 4))
    trace.validate_primary_range = False
    jacobian, goal = np.array([[0.0, 0.0, 0.0, 0.01]]), np.array([0.01])
    trace.objective_jacobians, trace.objective_goals = [jacobian], [goal]
    task = eik.ObjectiveSolveConfig()
    task.solve_mode = eik.TaskSolveMode.SCALE
    task.allow_min_error_fallback = False
    trace.objective_configs = [task]
    config = eik.VelocitySolverConfig()
    regularization = eik.RegularizedInverseConfig()
    regularization.epsilon, regularization.regularization_factor = 0.1, 0.1
    config.regularization_config = regularization
    config.refine_partial_scale = True
    trace.solver_config = config
    result = eik.replay_velocity_problem(trace)
    velocity = np.asarray(result.solution)
    assert result.status == eik.SolverStatus.SUCCESS
    assert result.task_scales[0] >= 0.4 - 1e-9
    assert result.task_scales[0] <= 0.4 + 3 * coupling + 1e-9
    np.testing.assert_allclose(
        jacobian @ velocity, result.task_scales[0] * goal, atol=1e-10, rtol=0
    )
    assert np.all(trace.constraint_matrix @ velocity >= trace.lower_bounds - 1e-10)
    assert np.all(trace.constraint_matrix @ velocity <= trace.upper_bounds + 1e-10)
    assert not any(result.task_used_fallback)


@pytest.mark.parametrize("prefix", [0.0, 0.25, 0.5, 0.75])
def test_refinement_keeps_legacy_lower_projected_preference(prefix):
    # The primary fixes x. Lower SCALE fills the available y direction; its
    # projected preference does not require [x,y] to lie on the raw [1,1] ray.
    trace = eik.VelocityProblemTrace()
    trace.backend_column_scale = np.ones(2)
    trace.constraint_matrix = np.eye(2)
    trace.lower_bounds, trace.upper_bounds = -np.ones(2), np.ones(2)
    trace.maximum_softening_factors = np.array([])
    trace.stationary_primary = np.empty((0, 2))
    trace.validate_primary_range = False
    primary, lower = eik.ObjectiveSolveConfig(), eik.ObjectiveSolveConfig()
    primary.priority, lower.priority = 0, 1
    primary.allow_min_error_fallback = lower.allow_min_error_fallback = False
    trace.objective_configs = [primary, lower]
    trace.objective_jacobians = [np.array([[1.0, 0.0]]), np.eye(2)]
    trace.objective_goals = [np.array([prefix]), np.ones(2)]
    config = eik.VelocitySolverConfig()
    regularization = eik.RegularizedInverseConfig()
    regularization.regularization_factor = 0.0
    config.regularization_config = regularization
    trace.solver_config = config
    baseline = eik.replay_velocity_problem(trace)
    np.testing.assert_allclose(baseline.solution, [prefix, 1.0], atol=1e-9, rtol=0)
    config.refine_partial_scale = True
    trace.solver_config = config
    refined = eik.replay_velocity_problem(trace)
    assert refined.status == eik.SolverStatus.SUCCESS
    np.testing.assert_allclose(refined.solution, baseline.solution, atol=1e-9, rtol=0)
    assert refined.task_scales == pytest.approx(baseline.task_scales)
    assert not any(refined.task_used_fallback)
