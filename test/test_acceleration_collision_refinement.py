"""Bounded acceleration collision recovery preserves physical acceptance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import embodik as eik

URDF = """<robot name="lift_pendulum"><link name="base"><collision name="floor"><origin xyz="0 0 .469"/><geometry><box size="4 4 1"/></geometry></collision></link>
<link name="carriage"/><joint name="lift" type="prismatic"><parent link="base"/><child link="carriage"/><axis xyz="0 0 1"/><limit lower="-.02" upper=".02" velocity=".5" effort="1000"/></joint>
<link name="pendulum"><inertial><origin xyz="0 0 .5"/><mass value="1"/><inertia ixx=".1" ixy="0" ixz="0" iyy=".1" iyz="0" izz=".1"/></inertial><collision name="tip"><origin xyz="0 0 1"/><geometry><sphere radius=".01"/></geometry></collision></link>
<joint name="angle" type="revolute"><parent link="carriage"/><child link="pendulum"/><axis xyz="0 1 0"/><limit lower="-.2" upper=".2" velocity="2" effort="1000"/></joint></robot>"""
Q, V, DT = np.zeros(2), np.array([0.0, 0.1]), 0.1


def system(tmp_path: Path):
    path = tmp_path / "lift_pendulum.urdf"
    path.write_text(URDF)
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.AccelerationSolver(robot)
    task = solver.add_posture_task("command")
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    task.allow_min_error_fallback = False
    reference = eik.AccelerationTaskReference()
    reference.proportional_gain = reference.derivative_gain = 0.0
    reference.desired_acceleration = np.array([0.0, 10.0])
    solver.set_task_reference("command", reference)
    provider = eik.KinematicsSolver(robot)
    provider.dt = DT
    provider.configure_collision_constraint(0.02, [], [], True, 1)
    provider.set_collision_tuning_mode(eik.CollisionTuningMode.PRECISE)
    provider.set_proximity_gated_collision_activation_enabled(False)
    provider.set_acceleration_limits(np.full(2, 15.0))
    provider.enable_acceleration_limits(True)
    provider.set_previous_joint_velocities(V.copy())
    options = eik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.full(2, 15.0)
    lift = eik.VelocityCollisionLiftOptions()
    return robot, solver, task, provider, options, lift


def cleared(result):
    for field in ("q_solution", "joint_accelerations", "joint_velocities_next"):
        assert np.asarray(getattr(result, field)).size == 0, (field, result.status_message)


def assert_physical(result, q=Q):
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    a = np.asarray(result.joint_accelerations)
    expected_q, expected_v = q + DT * V + 0.5 * DT**2 * a, V + DT * a
    np.testing.assert_allclose(result.q_solution, expected_q, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(result.joint_velocities_next, expected_v, rtol=0.0, atol=1e-14)
    assert np.all(np.abs(a) <= 15.0 + 1e-8)
    assert np.all(np.abs(expected_v) <= np.array([0.5, 2.0]) + 1e-8)
    assert np.all(expected_q >= [-0.02, -0.2]) and np.all(expected_q <= [0.02, 0.2])
    assert result.collision_endpoint_validated and not result.collision_step_certified
    return float(expected_q[0] + np.cos(expected_q[1]) - 0.979)


def test_sampled_curvature_default_rejects_and_opt_in_refines(tmp_path):
    robot, solver, _, provider, options, lift = system(tmp_path)
    baseline = solver.solve_with_velocity_collision(provider, Q, V, DT, options, lift)
    assert baseline.status == eik.SolverStatus.COLLISION_VIOLATED
    assert baseline.collision_refinement_iterations == 0
    cleared(baseline)
    # This is separated geometry: min_distance=.02, actual distance=.01920054.
    assert 0.0 < np.cos(0.06) - 0.979 < 0.02
    lift.max_refinement_iterations = 2
    fixed = solver.solve_with_velocity_collision(provider, Q, V, DT, options, lift)
    assert assert_physical(fixed) >= 0.02 - 1e-8
    assert 1 <= fixed.collision_refinement_iterations <= 2
    assert fixed.collision_refinement_exact_queries == fixed.collision_refinement_iterations
    assert fixed.collision_validation_samples >= 2
    np.testing.assert_array_equal(robot.get_current_configuration(), Q)
    np.testing.assert_array_equal(robot.get_current_velocity(), V)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_refinement_iterations", -1),
        ("max_refinement_iterations", 9),
        ("refinement_trust_radius", 0.0),
        ("refinement_trust_radius", np.nan),
        ("refinement_clearance_reserve", -1.0),
        ("refinement_clearance_reserve", np.inf),
    ],
)
def test_invalid_refinement_options_fail_before_motion(tmp_path, field, value):
    _, solver, _, provider, options, lift = system(tmp_path)
    setattr(lift, field, value)
    result = solver.solve_with_velocity_collision(provider, Q, V, DT, options, lift)
    assert result.status == eik.SolverStatus.INVALID_INPUT
    cleared(result)


@pytest.mark.parametrize("budget", [1, 2])
def test_tiny_internal_trust_consumes_budget_before_physical_domain_retry(tmp_path, budget):
    _, solver, _, provider, options, lift = system(tmp_path)
    lift.max_refinement_iterations = budget
    lift.refinement_trust_radius = 1e-5
    result = solver.solve_with_velocity_collision(provider, Q, V, DT, options, lift)
    assert result.collision_refinement_exact_queries == 1
    assert result.collision_refinement_iterations == budget
    if budget == 1:
        assert result.status == eik.SolverStatus.NUMERICAL_ERROR, result.status_message
        cleared(result)
    else:
        assert assert_physical(result) >= 0.02 - 1e-8


@pytest.mark.parametrize("source_id", ["__collision_sample_refinement", "__collision_sample_trust"])
def test_caller_lift_lock_survives_internal_trust_removal(tmp_path, source_id):
    _, solver, _, provider, options, lift = system(tmp_path)
    lock = eik.AffineAccelerationConstraint()
    lock.source_id = source_id
    lock.coefficient_matrix, lock.affine_bias = np.array([[1.0, 0.0]]), np.zeros(1)
    lock.lower_bounds = lock.upper_bounds = np.zeros(1)
    options.affine_constraints = [lock]
    lift.max_refinement_iterations = 4
    result = solver.solve_with_velocity_collision(provider, Q, V, DT, options, lift)
    # Slowing the pendulum can legitimately satisfy the physical floor with
    # lift locked. The caller row must survive even a reserved-name collision.
    assert assert_physical(result) >= 0.02 - 1e-8
    assert abs(result.joint_accelerations[0]) <= 1e-8
    assert 1 <= result.collision_refinement_iterations <= 4


@pytest.mark.parametrize("mode", [eik.TaskSolveMode.MIN_ERROR, eik.TaskSolveMode.SCALE_ELASTIC])
def test_all_unsafe_accelerations_locked_retains_caller_rows_and_rolls_back(tmp_path, mode):
    _, solver, task, provider, options, lift = system(tmp_path)
    task.solve_mode = mode
    lock = eik.AffineAccelerationConstraint()
    lock.source_id = "__collision_sample_trust"
    lock.coefficient_matrix, lock.affine_bias = np.eye(2), np.zeros(2)
    lock.lower_bounds = lock.upper_bounds = np.array([0.0, 10.0])
    options.affine_constraints = [lock]
    if mode == eik.TaskSolveMode.SCALE_ELASTIC:
        options.nominal_position_lower_bounds = np.array([0.0, -0.2])
        options.nominal_position_upper_bounds = np.array([0.02, 0.2])
    before = np.asarray(solver.get_elastic_band_deltas()).copy()
    lift.max_refinement_iterations = 2
    lift.refinement_trust_radius = 1e-5
    result = solver.solve_with_velocity_collision(provider, Q, V, DT, options, lift)
    assert result.status == eik.SolverStatus.NUMERICAL_ERROR, result.status_message
    assert result.collision_refinement_iterations == 2
    assert result.collision_refinement_exact_queries == 1
    cleared(result)
    np.testing.assert_array_equal(solver.get_elastic_band_deltas(), before)
    assert np.asarray(result.elastic_band_delta).size == 0
    assert not result.elastic_band_enabled


def test_failed_refinement_rolls_back_elastic_state_and_diagnostics(tmp_path):
    _, solver, task, provider, options, lift = system(tmp_path)
    task.solve_mode = eik.TaskSolveMode.SCALE_ELASTIC
    # Initial lift is exactly on a nominal face, with physical room below it.
    # A solve prepares a nonzero band; an unaccepted wrapper must not retain it.
    options.nominal_position_lower_bounds = np.array([0.0, -0.2])
    options.nominal_position_upper_bounds = np.array([0.02, 0.2])
    before = np.asarray(solver.get_elastic_band_deltas()).copy()
    lift.max_refinement_iterations = 1
    lift.refinement_trust_radius = 1e-5
    result = solver.solve_with_velocity_collision(provider, Q, V, DT, options, lift)
    assert result.status != eik.SolverStatus.SUCCESS
    cleared(result)
    np.testing.assert_array_equal(solver.get_elastic_band_deltas(), before)
    assert np.asarray(result.elastic_band_delta).size == 0  # Original unprepared state.
    assert not result.elastic_band_enabled


def test_initially_below_floor_retains_nonworsening_threshold(tmp_path):
    _, solver, _, provider, options, lift = system(tmp_path)
    provider.set_non_worsening_collision_floor_enabled(False)
    q = np.array([-0.002, 0.0])
    initial_distance = q[0] + 1.0 - 0.979
    assert initial_distance < 0.02
    lift.max_refinement_iterations = 2
    result = solver.solve_with_velocity_collision(provider, q, V, DT, options, lift)
    assert assert_physical(result, q) >= initial_distance - 1e-8
    assert result.collision_refinement_iterations >= 1
    assert abs(result.collision_refinement_last_required_distance - initial_distance) < 1e-10
