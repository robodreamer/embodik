"""Affine coordinates reuse state shaping without endpoint-only projection."""

from pathlib import Path

import numpy as np
import pytest

import embodik as eik


@pytest.fixture
def setup(tmp_path: Path):
    path = tmp_path / "two_slides.urdf"
    path.write_text("""<robot name="two_slides">
      <link name="base"/><link name="one"/><link name="two"/>
      <joint name="x" type="prismatic"><parent link="base"/><child link="one"/>
      <axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="5" effort="100"/></joint>
      <joint name="y" type="prismatic"><parent link="one"/><child link="two"/>
      <axis xyz="0 1 0"/><limit lower="-2" upper="2" velocity="5" effort="100"/></joint>
    </robot>""")
    robot = eik.RobotModel(str(path))
    solver = eik.AccelerationSolver(robot)
    task = solver.add_posture_task("motion")
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    constraint = eik.AffineStateConstraint()
    constraint.source_id = "coupled"
    constraint.coefficient_matrix = np.array([[1.0, 1.0]])
    constraint.offset = np.array([0.25])
    constraint.lower_bounds = np.array([-0.5])
    constraint.upper_bounds = np.array([0.5])
    policy = eik.GeometricConstraintAccelerationPolicy()
    policy.rate_limits = np.array([1.0])
    policy.acceleration_limits = np.array([10.0])
    policy.lower_braking_accelerations = np.array([2.0])
    policy.upper_braking_accelerations = np.array([2.0])
    constraint.policy = policy
    options = eik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.full(2, 10.0)
    options.affine_state_constraints = [constraint]
    return robot, solver, constraint, options


def command(solver, acceleration):
    reference = eik.AccelerationTaskReference()
    reference.proportional_gain = reference.derivative_gain = 0.0
    reference.desired_acceleration = np.full(2, acceleration)
    solver.set_task_reference("motion", reference)


def assert_no_candidate(result):
    assert result.status != eik.SolverStatus.SUCCESS
    assert not result.solution
    assert len(result.q_solution) == len(result.joint_accelerations) == 0


def test_coupled_envelope_reverses_without_reset_and_preserves_full_quadratics(setup):
    _, solver, _, options = setup
    q, dq, dt = np.array([0.12, 0.12]), np.zeros(2), 0.05
    extremes = []
    for index in range(160):
        command(solver, 20.0 if index < 80 else -20.0)
        result = solver.solve(q, dq, dt, options)
        assert result.status == eik.SolverStatus.SUCCESS, result.status_message
        ddq = np.array(result.joint_accelerations)
        y, v, a = q.sum() + 0.25, dq.sum(), ddq.sum()
        times = [0.0, dt]
        if a != 0 and 0 < -v / a < dt:
            times.append(-v / a)
        values = [y + t * v + 0.5 * t * t * a for t in times]
        assert min(values) >= -0.5 - 1e-9
        assert max(values) <= 0.5 + 1e-9
        assert abs(v) <= 1 + 1e-8 and abs(v + dt * a) <= 1 + 1e-8
        assert abs(a) <= 10 + 1e-8
        q, dq = np.array(result.q_solution), np.array(result.joint_velocities_next)
        margin = 0.5 - (q.sum() + 0.25) if dq.sum() > 0 else q.sum() + 0.75
        assert dq.sum() ** 2 <= 4 * max(0, margin) + 1e-8
        extremes.append(q.sum() + 0.25)
    assert max(extremes) > 0.49 and min(extremes) < -0.49


@pytest.mark.parametrize(
    "field,value",
    [
        ("coefficient_matrix", [[1.0]]),
        ("coefficient_matrix", [[np.nan, 1.0]]),
        ("offset", [np.inf]),
        ("offset", [0.0, 0.0]),
        ("lower_bounds", [0.6]),
        ("upper_bounds", [np.nan]),
    ],
)
def test_invalid_affine_data_has_no_motion_output(setup, field, value):
    _, solver, constraint, options = setup
    setattr(constraint, field, np.asarray(value))
    options.affine_state_constraints = [constraint]
    command(solver, 1.0)
    assert_no_candidate(solver.solve(np.zeros(2), np.zeros(2), 0.05, options))


def test_duplicate_source_cannot_hide_a_caller_row(setup):
    _, solver, _, options = setup
    caller = eik.AffineAccelerationConstraint()
    caller.source_id = "coupled"
    caller.coefficient_matrix = np.array([[1.0, 0.0]])
    caller.affine_bias = np.zeros(1)
    caller.lower_bounds, caller.upper_bounds = np.array([-1.0]), np.array([1.0])
    options.affine_constraints = [caller]
    result = solver.solve(np.zeros(2), np.zeros(2), 0.05, options)
    assert_no_candidate(result)
    assert "duplicate" in result.status_message


def test_records_own_matrices_and_policy(setup):
    _, _, constraint, options = setup
    constraint.coefficient_matrix = np.zeros((1, 2))
    policy = constraint.policy
    policy.rate_limits = np.array([10.0])
    constraint.policy = policy
    np.testing.assert_array_equal(
        options.affine_state_constraints[0].coefficient_matrix, [[1.0, 1.0]]
    )
    np.testing.assert_array_equal(options.affine_state_constraints[0].policy.rate_limits, [1.0])


def test_coupled_state_outside_envelope_rejects_without_projection(setup):
    _, solver, _, options = setup
    command(solver, -20.0)
    assert_no_candidate(solver.solve(np.array([0.2, 0.2]), np.zeros(2), 0.05, options))


def test_mixed_sign_rows_and_asymmetric_braking(setup):
    _, solver, constraint, options = setup
    constraint.coefficient_matrix = np.array([[1.0, 1.0], [1.0, -1.0]])
    constraint.offset = np.array([0.25, -0.1])
    constraint.lower_bounds, constraint.upper_bounds = np.full(2, -0.5), np.full(2, 0.5)
    policy = constraint.policy
    policy.rate_limits = np.ones(2)
    policy.acceleration_limits = np.full(2, 10.0)
    policy.lower_braking_accelerations = np.array([0.5, 1.0])
    policy.upper_braking_accelerations = np.array([1.0, 0.5])
    constraint.policy = policy
    options.affine_state_constraints = [constraint]
    q, dq = np.zeros(2), np.zeros(2)
    for index in range(40):
        reference = eik.AccelerationTaskReference()
        reference.proportional_gain = reference.derivative_gain = 0.0
        reference.desired_acceleration = np.array([0.05, -0.1]) * (1 if index < 20 else -1)
        solver.set_task_reference("motion", reference)
        result = solver.solve(q, dq, 0.01, options)
        assert result.status == eik.SolverStatus.SUCCESS, result.status_message
        q, dq = np.array(result.q_solution), np.array(result.joint_velocities_next)
        y = constraint.coefficient_matrix @ q + constraint.offset
        assert np.all(y >= constraint.lower_bounds - 1e-9)
        assert np.all(y <= constraint.upper_bounds + 1e-9)


def test_sampled_collision_adapter_rejects_unqualified_stopping_family(setup):
    robot, solver, _, options = setup
    provider = eik.KinematicsSolver(robot)
    result = solver.solve_with_velocity_collision(provider, np.zeros(2), np.zeros(2), 0.05, options)
    assert_no_candidate(result)
    assert "does not certify coupled" in result.status_message


def test_fixed_base_continuous_joint_is_rejected(setup, tmp_path):
    path = tmp_path / "two_slides.urdf"
    path.write_text(path.read_text().replace('type="prismatic"', 'type="continuous"', 1))
    robot = eik.RobotModel(str(path))
    try:
        solver = eik.AccelerationSolver(robot)
    except ValueError:
        return  # Constructor can reject unsupported configuration manifolds.
    _, _, _, options = setup
    result = solver.solve(np.array([1.0, 0.0, 0.0]), np.zeros(2), 0.05, options)
    assert_no_candidate(result)
