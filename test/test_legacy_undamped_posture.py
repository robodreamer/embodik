"""A singular trailing posture must not invalidate a feasible undamped ray."""

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("redundant", [False, True])
@pytest.mark.parametrize("refine", [False, True])
def test_undamped_legacy_posture_remains_finite_and_preserves_primary(direction, redundant, refine):
    trace = eik.VelocityProblemTrace()
    trace.seed_q = np.zeros(2)
    trace.validation_dt = 0.01
    trace.constraint_matrix = np.diag([direction, 1.0])
    trace.lower_bounds = np.array([-1.0, -1.0])
    trace.upper_bounds = np.array([0.0, 1.0])
    trace.maximum_softening_factors = np.empty(0)
    trace.backend_column_scale = np.ones(2)
    trace.stationary_primary = np.ones((1, 2))
    trace.validate_primary_range = True
    posture = np.vstack((np.eye(2), np.zeros((1, 2)))) if redundant else np.eye(2)
    goal = direction * np.array([0.5, -0.5, 0.0] if redundant else [0.5, -0.5])
    trace.objective_jacobians = [np.ones((1, 2)), posture]
    trace.objective_goals = [np.array([direction * 0.1]), goal]
    primary, secondary = eik.ObjectiveSolveConfig(), eik.ObjectiveSolveConfig()
    primary.solve_mode = eik.TaskSolveMode.SCALE
    primary.allow_min_error_fallback = False
    secondary.priority = 1
    secondary.solve_mode = eik.TaskSolveMode.MIN_ERROR
    trace.objective_configs = [primary, secondary]
    config = eik.VelocitySolverConfig()
    config.epsilon = 1e-9
    config.precision_threshold = 1e-11
    config.refine_partial_scale = refine
    config.regularization_config.epsilon = 1e-10
    config.regularization_config.regularization_factor = 0.0
    trace.solver_config = config
    result = eik.replay_velocity_problem(trace)
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    np.testing.assert_allclose(result.solution, [0.0, direction * 0.1], atol=1e-8)
    assert np.isfinite(result.solution).all()
    assert result.task_scales == pytest.approx([1.0, -1.0])
    assert result.task_modes_effective == [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.MIN_ERROR]
