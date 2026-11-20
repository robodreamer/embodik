"""Prescribed physical samples distinguish curvature from construction rows."""

from pathlib import Path

import numpy as np
import pytest

import embodik as eik

Q, DQ, DDQ, DT = np.array([0.0005]), np.array([-0.1]), np.array([10.0]), 0.01


def system(tmp_path: Path, separation: float = 0.11):
    path = tmp_path / "curved_distance.urdf"
    path.write_text(f"""<robot name="curved_distance"><link name="world">
      <collision name="obstacle"><geometry><sphere radius=".05"/></geometry></collision>
      </link><link name="tool"><inertial><mass value="1"/>
      <inertia ixx=".1" iyy=".1" izz=".1" ixy="0" ixz="0" iyz="0"/>
      </inertial><collision name="moving"><geometry><sphere radius=".05"/></geometry>
      </collision></link><joint name="slide" type="prismatic">
      <parent link="world"/><child link="tool"/><origin xyz="{separation} 0 0"/>
      <axis xyz="0 1 0"/><limit lower="-1" upper="1" velocity="2" effort="100"/>
      </joint></robot>""")
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.AccelerationSolver(robot)
    provider = eik.KinematicsSolver(robot)
    provider.dt = DT
    provider.configure_collision_constraint(0.01, [], [], True, 1)
    provider.set_proximity_gated_collision_activation_enabled(False)
    provider.set_collision_repulsion_deadband(0.0)
    provider.set_acceleration_limits(np.array([15.0]))
    provider.enable_acceleration_limits(True)
    options = eik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.array([15.0])
    lift = eik.VelocityCollisionLiftOptions()
    lift.validation_substeps = 4
    return solver, provider, options, lift


def checked(system, candidate=DDQ):
    solver, provider, options, lift = system
    return solver.validate_candidate_with_velocity_collision(
        provider, Q, DQ, DT, candidate, options, lift
    )


def assert_rejected(result) -> None:
    assert result.status != eik.SolverStatus.SUCCESS
    for name in ("joint_accelerations", "joint_velocities_next", "q_solution"):
        assert np.asarray(getattr(result, name)).size == 0


def physical(system):
    system[-1].candidate_validation = eik.CollisionCandidateValidation.PHYSICAL_SAMPLES
    return system


def test_curved_floor_safe_interval_can_end_at_exact_rest(tmp_path: Path):
    setup = system(tmp_path)
    assert setup[-1].candidate_validation == eik.CollisionCandidateValidation.FROZEN_ROWS
    assert_rejected(checked(setup))
    result = checked(physical(setup))
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    np.testing.assert_array_equal(result.joint_accelerations, DDQ)
    np.testing.assert_allclose(result.q_solution, 0.0, atol=1e-18, rtol=0)
    np.testing.assert_array_equal(result.joint_velocities_next, np.zeros(1))
    assert result.collision_endpoint_validated and not result.collision_step_certified
    assert result.collision_validation_samples == 4
    assert result.collision_validation_pairs_checked == 4
    assert result.collision_lift_row_pairs == 0
    # Analytically the whole interval is safe, independently of the four checks:
    # q(t)=q0*(1-t/DT)^2 and d(t)=sqrt(.11^2+q(t)^2)-.1 >= .01.
    assert result.task_diagnostics == [] and result.linear_problems == []


def test_physical_samples_reject_actual_floor_loss(tmp_path: Path):
    setup = physical(system(tmp_path, separation=0.109999))
    result = checked(setup)
    assert_rejected(result)
    assert result.status == eik.SolverStatus.COLLISION_VIOLATED
    assert result.collision_validation_samples > 0


@pytest.mark.parametrize("kind", ["affine", "frozen", "acceleration", "com"])
def test_physical_samples_preserve_caller_constraints(tmp_path: Path, kind: str):
    setup = physical(system(tmp_path))
    options = setup[2]
    if kind in ("affine", "frozen"):
        row = (
            eik.AffineAccelerationConstraint()
            if kind == "affine"
            else eik.FrozenNextVelocityConstraint()
        )
        row.source_id = "caller_contract"
        row.coefficient_matrix, row.affine_bias = np.ones((1, 1)), np.zeros(1)
        if kind == "affine":
            row.lower_bounds, row.upper_bounds = np.array([-1.0]), np.array([1.0])
            options.affine_constraints = [row]
        else:
            row.lower_bounds, row.upper_bounds = np.array([0.01]), np.array([1.0])
            options.frozen_next_velocity_constraints = [row]
    elif kind == "acceleration":
        options.acceleration_limits_override = np.array([5.0])
    else:
        definition = eik.ComSupportPolygonConstraintDefinition()
        definition.frame_name = "world"
        definition.support_polygon = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        policy = eik.ComSupportPolygonAccelerationPolicy()
        policy.rate_limit = 1.0
        policy.acceleration_limit = policy.braking_acceleration = 0.1
        constraint = eik.ComSupportPolygonAccelerationConstraint()
        constraint.source_id = "caller_support"
        constraint.definition, constraint.policy = definition, policy
        options.com_support_polygon_constraints = [constraint]
    assert_rejected(checked(setup))


def test_physical_samples_refuse_optimization(tmp_path: Path):
    solver, provider, options, lift = physical(system(tmp_path))
    result = solver.solve_with_velocity_collision(provider, Q, DQ, DT, options, lift)
    assert_rejected(result)
    assert result.status == eik.SolverStatus.INVALID_INPUT
    assert "prescribed candidate" in result.status_message


@pytest.mark.parametrize("reason", ["deadband", "initial_floor"])
def test_physical_samples_explicitly_refuse_unsupported_recovery(tmp_path: Path, reason: str):
    setup = physical(system(tmp_path, separation=0.1099 if reason == "initial_floor" else 0.11))
    if reason == "deadband":
        setup[1].set_collision_repulsion_deadband(1e-4)
    result = checked(setup)
    assert_rejected(result)
    assert result.status == eik.SolverStatus.INVALID_INPUT
    assert ("deadband" if reason == "deadband" else "initially satisfied") in result.status_message
