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
