"""Corrective CoM solves retain physical rows and fail closed on exhaustion."""

import numpy as np
import pytest

import embodik as eik


@pytest.fixture
def pendulum(tmp_path):
    urdf = tmp_path / "pendulum.urdf"
    urdf.write_text(
        '<robot name="pendulum"><link name="base"/>'
        '<link name="body"><inertial><origin xyz="0 1 0"/><mass value="1"/>'
        '<inertia ixx=".1" ixy="0" ixz="0" iyy=".1" iyz="0" izz=".1"/>'
        '</inertial></link><joint name="pivot" type="revolute">'
        '<parent link="base"/><child link="body"/><axis xyz="0 0 1"/>'
        '<limit lower="-2" upper="2" velocity="10" effort="100"/>'
        "</joint></robot>"
    )
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.AccelerationSolver(robot)
    task = solver.add_joint_task("move", "pivot", 0.0)
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    reference = eik.AccelerationTaskReference()
    reference.proportional_gain = reference.derivative_gain = 0.0
    reference.desired_acceleration = np.array([2.0])
    solver.set_task_reference("move", reference)
    definition = eik.ComSupportPolygonConstraintDefinition()
    definition.support_polygon = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])
    definition.frame_name, definition.margin = "base", 0.0
    policy = eik.ComSupportPolygonAccelerationPolicy()
    policy.rate_limit = 10.0
    policy.acceleration_limit = policy.braking_acceleration = 1.0
    constraints = []
    for name in ("support", "second-support"):
        constraint = eik.ComSupportPolygonAccelerationConstraint()
        constraint.source_id = name
        constraint.definition, constraint.policy = definition, policy
        constraints.append(constraint)
    options = eik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.array([10.0])
    options.com_support_polygon_constraints = constraints
    return solver, options


def _physical_accelerations(q, dq, ddq, dt):
    times = np.linspace(0.0, dt, 101)
    angles = q + times * dq + 0.5 * times**2 * ddq
    rates = dq + times * ddq
    return np.array(
        [
            -np.cos(angles) * ddq + np.sin(angles) * rates**2,
            -np.sin(angles) * ddq - np.cos(angles) * rates**2,
        ]
    )


@pytest.mark.parametrize("allocated", [False, True])
def test_refinement_retains_affine_bias_and_allocation(pendulum, allocated):
    solver, options = pendulum
    row = eik.AffineAccelerationConstraint()
    row.source_id = "caller-row"
    row.coefficient_matrix = np.ones((1, 1))
    row.affine_bias = np.array([0.25])
    row.lower_bounds, row.upper_bounds = np.array([1.25]), np.array([1.5])
    options.affine_constraints = [row]
    if allocated:
        allocation = eik.GeneralizedAccelerationAllocation()
        allocation.metric_diagonal = np.array([4.0])
        allocation.reference_acceleration = np.array([0.3])
        options.generalized_acceleration_allocation = allocation
    q, dq, dt = np.array([-0.5]), np.array([0.1]), 0.01
    result = solver.solve(q, dq, dt, options)
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    assert 1 <= result.nonlinear_refinement_iterations <= 8
    acceleration = result.joint_accelerations[0]
    assert 1.0 <= acceleration <= 1.25
    assert np.max(np.abs(_physical_accelerations(q[0], dq[0], acceleration, dt))) <= 1 + 1e-8
    np.testing.assert_allclose(
        result.q_solution, q + dt * dq + 0.5 * dt**2 * acceleration, atol=1e-14
    )
    np.testing.assert_allclose(result.joint_velocities_next, dq + dt * acceleration, atol=1e-14)
    assert result.backend_computation_time_ms >= 0.0
    assert result.computation_time_ms >= result.preprocessing_time_ms


def test_unrecoverable_local_rows_do_not_claim_infeasibility(pendulum):
    solver, options = pendulum
    forced = eik.AffineAccelerationConstraint()
    forced.source_id = "forced-initial-face"
    forced.coefficient_matrix = np.ones((1, 1))
    forced.affine_bias = np.zeros(1)
    forced.lower_bounds = forced.upper_bounds = np.array([1 / np.cos(0.5)])
    options.affine_constraints = [forced]
    result = solver.solve(np.array([-0.5]), np.zeros(1), 0.01, options)
    assert result.status == eik.SolverStatus.NUMERICAL_ERROR, result.status_message
    assert 1 <= result.nonlinear_refinement_iterations <= 8
    assert result.joint_accelerations.size == 0
    assert result.joint_velocities_next.size == 0
    assert result.q_solution.size == 0


def test_safe_original_candidate_needs_no_refinement(pendulum):
    solver, options = pendulum
    reference = eik.AccelerationTaskReference()
    reference.proportional_gain = reference.derivative_gain = 0.0
    reference.desired_acceleration = np.array([0.5])
    solver.set_task_reference("move", reference)
    result = solver.solve(np.array([-0.5]), np.zeros(1), 0.01, options)
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    assert result.nonlinear_refinement_iterations == 0
    np.testing.assert_allclose(result.joint_accelerations, [0.5], atol=1e-8)


def test_curvature_correction_preserves_physical_acceptance(pendulum):
    solver, options = pendulum
    reference = eik.AccelerationTaskReference()
    reference.proportional_gain = reference.derivative_gain = 0.0
    reference.desired_acceleration = np.array([-2.0])
    solver.set_task_reference("move", reference)
    q, dq, dt = np.array([-0.518]), np.array([0.333]), 0.283

    result = solver.solve(q, dq, dt, options)

    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    assert 1 <= result.nonlinear_refinement_iterations <= 8
    acceleration = result.joint_accelerations[0]
    assert abs(acceleration) <= options.acceleration_limits_override[0]
    assert np.max(np.abs(_physical_accelerations(q[0], dq[0], acceleration, dt))) <= 1 + 1e-8
    np.testing.assert_allclose(
        result.q_solution, q + dt * dq + 0.5 * dt**2 * acceleration, atol=1e-14
    )
    np.testing.assert_allclose(result.joint_velocities_next, dq + dt * acceleration, atol=1e-14)


@pytest.mark.parametrize("allocated", [False, True])
def test_linear_trace_preserves_recursive_problems_and_replays(pendulum, allocated):
    solver, options = pendulum
    if allocated:
        allocation = eik.GeneralizedAccelerationAllocation()
        allocation.metric_diagonal = np.array([4.0])
        allocation.reference_acceleration = np.array([0.3])
        options.generalized_acceleration_allocation = allocation
    q, dq, dt = np.array([-0.5]), np.array([0.1]), 0.01
    original = solver.solve(q, dq, dt, options)
    assert original.linear_problems == []
    options.record_linear_problems = True
    result = solver.solve(q, dq, dt, options)
    assert result.status == original.status == eik.SolverStatus.SUCCESS
    np.testing.assert_array_equal(result.joint_accelerations, original.joint_accelerations)
    traces = result.linear_problems
    assert len(traces) == result.nonlinear_refinement_iterations + 1
    assert len(traces) > 1
    assert [t.refinement_budget_remaining for t in traces] == list(range(9, 9 - len(traces), -1))
    for trace in traces:
        np.testing.assert_array_equal(trace.q, q)
        np.testing.assert_array_equal(trace.dq, dq)
        assert trace.dt == dt
        assert trace.allocation_applied == allocated
        assert len(trace.constraint_row_sources) == trace.constraint_matrix.shape[0]
        assert len(trace.objective_targets) == len(trace.objective_biases)
        assert trace.diagnostics.phase_one_status == "feasible"
        replay = eik.replay_acceleration_linear_problem(trace)
        assert replay.status == trace.backend_result.status
        np.testing.assert_array_equal(replay.solution, trace.backend_result.solution)
    assert traces[-1].constraint_matrix.shape[0] > traces[0].constraint_matrix.shape[0]
    # Subsequent solves cannot alter a returned diagnostic snapshot.
    matrix = traces[-1].constraint_matrix.copy()
    options.record_linear_problems = False
    assert solver.solve(q, dq, dt, options).linear_problems == []
    np.testing.assert_array_equal(traces[-1].constraint_matrix, matrix)


def test_linear_trace_survives_failed_refinement_and_output_clearing(pendulum):
    solver, options = pendulum
    forced = eik.AffineAccelerationConstraint()
    forced.source_id = "forced-initial-face"
    forced.coefficient_matrix = np.ones((1, 1))
    forced.affine_bias = np.zeros(1)
    forced.lower_bounds = forced.upper_bounds = np.array([1 / np.cos(0.5)])
    options.affine_constraints = [forced]
    options.record_linear_problems = True
    result = solver.solve(np.array([-0.5]), np.zeros(1), 0.01, options)
    assert result.status == eik.SolverStatus.NUMERICAL_ERROR
    assert result.joint_accelerations.size == 0
    assert len(result.linear_problems) == result.nonlinear_refinement_iterations + 1
    assert result.linear_problems[0].backend_result.status == eik.SolverStatus.SUCCESS
    assert result.linear_problems[-1].backend_result.status != eik.SolverStatus.SUCCESS
    for trace in result.linear_problems:
        assert "forced-initial-face" in trace.constraint_row_sources
        replay = eik.replay_acceleration_linear_problem(trace)
        assert replay.status == trace.backend_result.status
        assert replay.status_message == trace.backend_result.status_message


def test_linear_trace_is_empty_when_validation_fails_before_backend(pendulum):
    solver, options = pendulum
    options.record_linear_problems = True
    result = solver.solve(np.array([np.nan]), np.zeros(1), 0.01, options)
    assert result.status == eik.SolverStatus.NON_FINITE_INPUT
    assert result.linear_problems == []
