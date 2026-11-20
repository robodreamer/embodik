"""Prescribed commands use native admission without becoming optimization tasks."""

from pathlib import Path

import numpy as np
import pytest
from test_acceleration_com_refinement import pendulum

import embodik as eik


@pytest.fixture
def system(tmp_path: Path):
    path = tmp_path / "candidate_sliders.urdf"
    path.write_text("""<robot name="candidate_sliders">
      <link name="base"/><link name="x_link"/>
      <link name="tip"><inertial><mass value="1"/>
      <inertia ixx=".1" iyy=".1" izz=".1" ixy="0" ixz="0" iyz="0"/>
      </inertial></link>
      <joint name="x" type="prismatic"><parent link="base"/><child link="x_link"/>
      <axis xyz="1 0 0"/><limit lower="-1" upper="1" velocity="2" effort="100"/></joint>
      <joint name="y" type="prismatic"><parent link="x_link"/><child link="tip"/>
      <axis xyz="0 1 0"/><limit lower="-1" upper="1" velocity="2" effort="100"/></joint>
    </robot>""")
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.AccelerationSolver(robot)
    task = solver.add_posture_task("requested_motion")
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    reference = eik.AccelerationTaskReference()
    reference.proportional_gain = reference.derivative_gain = 0.0
    reference.desired_acceleration = np.array([1.0, -1.0])
    solver.set_task_reference("requested_motion", reference)
    options = eik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.full(2, 4.0)
    return robot, solver, options


def assert_rejected(result) -> None:
    assert result.status != eik.SolverStatus.SUCCESS
    assert not result.solution
    assert np.isnan(result.final_error) and np.isnan(result.condition_number)
    for field in ("joint_accelerations", "q_solution", "joint_velocities_next"):
        assert np.asarray(getattr(result, field)).size == 0


def test_candidate_is_not_optimized_and_diagnostics_do_not_claim_achievement(system):
    _, solver, options = system
    q, dq, candidate, dt = np.zeros(2), np.array([0.1, -0.2]), np.array([0.3, 0.4]), 0.01
    options.record_linear_problems = True
    result = solver.validate_candidate(q, dq, dt, candidate, options)
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    np.testing.assert_array_equal(result.joint_accelerations, candidate)
    np.testing.assert_allclose(result.q_solution, q + dt * dq + 0.5 * dt**2 * candidate)
    np.testing.assert_allclose(result.joint_velocities_next, dq + dt * candidate)
    assert result.task_diagnostics == [] and result.task_scales == []
    assert result.task_errors == [] and result.task_modes_effective == []
    assert result.task_used_fallback == [] and result.linear_problems == []
    assert not result.state_box_task_fallback_applied
    assert result.iterations == 0 and result.backend_computation_time_ms == 0
    assert np.isnan(result.final_error) and np.isnan(result.condition_number)
    assert result.computation_time_ms >= 0
    np.testing.assert_array_equal(candidate, [0.3, 0.4])


def test_native_solve_output_passes_the_same_candidate_admission(system):
    _, solver, options = system
    q, dq, dt = np.array([0.2, -0.1]), np.array([-0.1, 0.2]), 0.01
    solved = solver.solve(q, dq, dt, options)
    assert solved.status == eik.SolverStatus.SUCCESS, solved.status_message
    checked = solver.validate_candidate(q, dq, dt, solved.joint_accelerations, options)
    assert checked.status == eik.SolverStatus.SUCCESS, checked.status_message
    np.testing.assert_array_equal(checked.joint_accelerations, solved.joint_accelerations)
    np.testing.assert_array_equal(checked.q_solution, solved.q_solution)
    np.testing.assert_array_equal(checked.joint_velocities_next, solved.joint_velocities_next)


@pytest.mark.parametrize("scale", [1.0, 1e8, 1e9])
def test_affine_row_units_preserve_ordinary_solve_acceptance(system, scale):
    _, solver, options = system
    near = np.nextafter(np.nextafter(1 / 3, np.inf), np.inf)
    reference = eik.AccelerationTaskReference()
    reference.proportional_gain = reference.derivative_gain = 0.0
    reference.desired_acceleration = np.array([near, 0.0])
    solver.set_task_reference("requested_motion", reference)
    row = eik.AffineAccelerationConstraint()
    row.source_id = "row_units"
    row.coefficient_matrix = np.array([[scale, 0.0]])
    row.affine_bias = np.zeros(1)
    row.lower_bounds, row.upper_bounds = np.array([-scale]), np.array([scale * (1 / 3)])
    options.affine_constraints = [row]
    q = np.zeros(2)
    solved = solver.solve(q, q, 0.01, options)
    assert solved.status == eik.SolverStatus.SUCCESS, solved.status_message
    checked = solver.validate_candidate(q, q, 0.01, solved.joint_accelerations, options)
    assert checked.status == eik.SolverStatus.SUCCESS, checked.status_message
    np.testing.assert_array_equal(checked.joint_accelerations, solved.joint_accelerations)


def test_task_bound_rows_are_prepared_even_without_objective_optimization(system):
    _, solver, options = system
    bound = eik.TaskAccelerationBounds()
    bound.source_id, bound.task_name = "task_hard_rows", "requested_motion"
    bound.lower_bounds, bound.upper_bounds = np.full(2, -0.1), np.full(2, 0.1)
    options.task_acceleration_bounds = [bound]
    assert_rejected(solver.validate_candidate(np.zeros(2), np.zeros(2), 0.01, np.ones(2), options))


def test_nonlinear_com_rejection_never_refines_the_supplied_command(pendulum):
    solver, options = pendulum
    q, dq = np.array([-0.5]), np.array([0.1])
    # Meets the incoming x acceleration bound but fails later in the interval.
    candidate = np.array([(1 + np.sin(q[0]) * dq[0] ** 2) / np.cos(q[0])])
    before = candidate.copy()
    result = solver.validate_candidate(q, dq, 0.01, candidate, options)
    assert_rejected(result)
    assert result.status == eik.SolverStatus.NO_PROGRESS
    assert "path" in result.status_message and "acceleration" in result.status_message
    assert result.nonlinear_refinement_iterations == 0
    np.testing.assert_array_equal(candidate, before)


@pytest.mark.parametrize(
    "candidate", [[1.0], [np.nan, 0.0], [np.inf, 0.0], [4.1, 0.0], [-4.1, 0.0]]
)
def test_invalid_or_hard_invalid_candidate_has_no_replacement(system, candidate):
    _, solver, options = system
    before = solver.get_elastic_band_deltas().copy()
    result = solver.validate_candidate(np.zeros(2), np.zeros(2), 0.01, np.array(candidate), options)
    assert_rejected(result)
    np.testing.assert_array_equal(solver.get_elastic_band_deltas(), before)


def test_affine_and_lock_rows_reject_an_otherwise_feasible_command(system):
    _, solver, options = system
    row = eik.AffineAccelerationConstraint()
    row.source_id = "coupled_acceleration"
    row.coefficient_matrix = np.array([[1.0, 1.0]])
    row.affine_bias = np.zeros(1)
    row.lower_bounds, row.upper_bounds = np.array([-0.1]), np.array([0.1])
    options.affine_constraints = [row]
    assert_rejected(solver.validate_candidate(np.zeros(2), np.zeros(2), 0.01, np.ones(2), options))
    options.affine_constraints = []
    options.zero_acceleration_joint_indices = [1]
    assert_rejected(solver.validate_candidate(np.zeros(2), np.zeros(2), 0.01, np.ones(2), options))


def test_com_rows_remain_active_for_a_prescribed_command(system):
    _, solver, options = system
    definition = eik.ComSupportPolygonConstraintDefinition()
    definition.support_polygon = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    definition.frame_name = "base"
    policy = eik.ComSupportPolygonAccelerationPolicy()
    policy.rate_limit = 1.0
    policy.acceleration_limit = policy.braking_acceleration = 0.25
    constraint = eik.ComSupportPolygonAccelerationConstraint()
    constraint.source_id = "support"
    constraint.definition, constraint.policy = definition, policy
    options.com_support_polygon_constraints = [constraint]
    okay = solver.validate_candidate(np.zeros(2), np.zeros(2), 0.01, np.array([0.1, 0.1]), options)
    assert okay.status == eik.SolverStatus.SUCCESS, okay.status_message
    assert_rejected(solver.validate_candidate(np.zeros(2), np.zeros(2), 0.01, np.ones(2), options))


def test_nominal_envelopes_and_existing_elastic_history_are_not_mutated(system):
    _, solver, options = system
    task = solver.get_task("requested_motion")
    task.solve_mode = eik.TaskSolveMode.SCALE_ELASTIC
    options.nominal_position_lower_bounds = np.full(2, -0.5)
    options.nominal_position_upper_bounds = np.full(2, 0.5)
    result = solver.solve(np.array([0.5, 0.0]), np.zeros(2), 0.01, options)
    assert np.any(np.asarray(result.elastic_band_capacity_lower) > 0)
    before = solver.get_elastic_band_deltas().copy()
    assert_rejected(solver.validate_candidate(np.zeros(2), np.zeros(2), 0.01, np.zeros(2), options))
    np.testing.assert_array_equal(solver.get_elastic_band_deltas(), before)
    options.nominal_position_lower_bounds = options.nominal_position_upper_bounds = None
    # Nonzero stored capacity is unsupported even if the caller removes its envelope.
    assert_rejected(solver.validate_candidate(np.zeros(2), np.zeros(2), 0.01, np.zeros(2), options))
    np.testing.assert_array_equal(solver.get_elastic_band_deltas(), before)
    malformed = solver.validate_candidate(
        np.zeros(2), np.zeros(2), 0.01, np.array([np.nan, 0.0]), options
    )
    assert_rejected(malformed)
    assert malformed.status == eik.SolverStatus.NON_FINITE_INPUT
    np.testing.assert_array_equal(solver.get_elastic_band_deltas(), before)


def test_prescribed_stopping_rollout_includes_actual_terminal_rest(system):
    _, solver, options = system
    q, dq = np.zeros(2), np.array([0.1, -0.1])
    for candidate in (np.array([-2.0, 2.0]), np.zeros(2)):
        result = solver.validate_candidate(q, dq, 0.05, candidate, options)
        assert result.status == eik.SolverStatus.SUCCESS, result.status_message
        q, dq = np.array(result.q_solution), np.array(result.joint_velocities_next)
    np.testing.assert_array_equal(dq, np.zeros(2))


def test_collision_validation_reuses_guards_and_refuses_refinement(tmp_path: Path):
    from test_acceleration_collision_refinement import DT, Q, V
    from test_acceleration_collision_refinement import system as collision_system

    _, solver, _, provider, options, lift = collision_system(tmp_path)
    # This passes initial affine rows but violates finite collision curvature.
    candidate = np.array([0.0, 10.0])
    result = solver.validate_candidate_with_velocity_collision(
        provider, Q, V, DT, candidate, options, lift
    )
    assert_rejected(result)
    assert result.status == eik.SolverStatus.COLLISION_VIOLATED
    assert result.collision_refinement_iterations == 0
    lift.max_refinement_iterations = 1
    result = solver.validate_candidate_with_velocity_collision(
        provider, Q, V, DT, candidate, options, lift
    )
    assert_rejected(result)
    assert "zero collision refinement budget" in result.status_message


def test_collision_solve_command_revalidates_unchanged(tmp_path: Path):
    from test_acceleration_collision_refinement import DT, Q, V
    from test_acceleration_collision_refinement import system as collision_system

    _, solver, _, provider, options, lift = collision_system(tmp_path)
    lift.max_refinement_iterations = 2
    solved = solver.solve_with_velocity_collision(provider, Q, V, DT, options, lift)
    assert solved.status == eik.SolverStatus.SUCCESS, solved.status_message
    lift.max_refinement_iterations = 0
    checked = solver.validate_candidate_with_velocity_collision(
        provider, Q, V, DT, solved.joint_accelerations, options, lift
    )
    assert checked.status == eik.SolverStatus.SUCCESS, checked.status_message
    np.testing.assert_array_equal(checked.joint_accelerations, solved.joint_accelerations)
    assert checked.collision_endpoint_validated and not checked.collision_step_certified
