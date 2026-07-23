import pathlib

import numpy as np
import pytest

import embodik as eik


def _write_urdf(tmp_path: pathlib.Path, name: str, contents: str) -> pathlib.Path:
    path = tmp_path / name
    path.write_text(contents)
    return path


def _two_joint_urdf() -> str:
    return """<?xml version="1.0"?>
<robot name="acceleration_solver_python_test_robot">
  <link name="base_link"/>
  <link name="link1"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" velocity="10.0" effort="20.0"/>
  </joint>
  <link name="tip"/>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="tip"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" velocity="10.0" effort="20.0"/>
  </joint>
</robot>
"""


def _single_prismatic_sphere_pair_urdf() -> str:
    return """<?xml version="1.0"?>
<robot name="acceleration_solver_python_collision_robot">
  <link name="world"/>
  <link name="obstacle">
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <joint name="obstacle_fixed" type="fixed">
    <parent link="world"/>
    <child link="obstacle"/>
  </joint>
  <link name="moving">
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <joint name="moving_slide" type="prismatic">
    <parent link="world"/>
    <child link="moving"/>
    <origin xyz="0.12 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.0" upper="1.0" effort="1000" velocity="100"/>
  </joint>
</robot>
"""


def _options_with_limits(limits: list[float] | np.ndarray) -> eik.AccelerationSolveOptions:
    options = eik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.asarray(limits, dtype=float)
    options.apply_position_limits = False
    options.apply_velocity_limits = False
    return options


def _two_joint_robot(tmp_path: pathlib.Path) -> eik.RobotModel:
    urdf = _write_urdf(tmp_path, "acceleration_two_joint.urdf", _two_joint_urdf())
    return eik.RobotModel(str(urdf), floating_base=False)


def _collision_robot(tmp_path: pathlib.Path) -> eik.RobotModel:
    urdf = _write_urdf(
        tmp_path,
        "acceleration_prismatic_collision.urdf",
        _single_prismatic_sphere_pair_urdf(),
    )
    robot = eik.RobotModel(str(urdf), floating_base=False)
    robot.set_gravity(np.zeros(3))
    return robot


def test_capabilities_and_simple_fixed_base_solve(tmp_path):
    capabilities = eik.AccelerationSolver.capabilities()
    assert capabilities.supports_fixed_base_scalar_joints
    assert not capabilities.supports_floating_base
    assert capabilities.supports_effort_constraints
    assert capabilities.supports_fixed_base_contact_kinematics
    assert capabilities.supports_tight_point_constraints
    assert capabilities.supports_tight_frame_pose_constraints
    assert capabilities.supports_relative_pose_constraints
    assert capabilities.supports_torso_pose_bound_constraints
    assert capabilities.supports_com_support_polygon_constraints
    assert not capabilities.supports_dynamic_contact

    robot = _two_joint_robot(tmp_path)
    solver = eik.AccelerationSolver(robot)
    task = solver.add_joint_task("joint1_task", "joint1", 0.0)
    assert task.name == "joint1_task"

    reference = eik.AccelerationTaskReference()
    reference.desired_acceleration = np.array([3.0])
    solver.set_task_reference("joint1_task", reference)

    result = solver.solve(
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        0.1,
        _options_with_limits([10.0, 10.0]),
    )

    assert isinstance(result, eik.SolverResult)
    assert result.status == eik.SolverStatus.SUCCESS
    assert np.allclose(result.joint_accelerations, [3.0, 0.0], atol=1e-9)
    assert np.allclose(result.joint_velocities_next, [0.3, 0.0], atol=1e-9)
    assert np.allclose(result.q_solution, [0.015, 0.0], atol=1e-9)
    assert result.acceleration_limits_applied
    assert len(result.task_diagnostics) == 1
    assert result.task_diagnostics[0].task_name == "joint1_task"
    assert np.allclose(result.task_diagnostics[0].reference_acceleration, [3.0])


def test_invalid_acceleration_solve_fails_clear(tmp_path):
    robot = _two_joint_robot(tmp_path)
    solver = eik.AccelerationSolver(robot)
    solver.add_joint_task("joint1_task", "joint1", 0.0)

    result = solver.solve(
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        0.01,
        _options_with_limits([1.0]),
    )

    assert result.status == eik.SolverStatus.INVALID_INPUT
    assert result.solution == []
    assert result.joint_accelerations.size == 0
    assert result.joint_velocities_next.size == 0
    assert result.q_solution.size == 0
    assert result.predicted_torques.size == 0


def test_native_collision_lifecycle_and_solve(tmp_path):
    capabilities = eik.AccelerationSolver.capabilities()
    if not capabilities.supports_analytic_sphere_collision_constraints:
        pytest.skip("native sphere collision acceleration constraints unavailable")

    robot = _collision_robot(tmp_path)
    solver = eik.AccelerationSolver(robot)

    definition = eik.CollisionConstraintDefinition()
    definition.min_distance = 0.02
    policy = eik.CollisionConstraintAccelerationPolicy()
    policy.outside_policy = eik.CollisionConstraintOutsidePolicy.REJECT

    solver.configure_collision_constraint(definition, policy)
    assert solver.has_collision_constraint()
    assert solver.get_collision_min_distance() == pytest.approx(0.02)
    assert solver.get_collision_constraint_definition().min_distance == pytest.approx(0.02)
    assert (
        solver.get_collision_constraint_policy().outside_policy
        == eik.CollisionConstraintOutsidePolicy.REJECT
    )
    assert len(solver.get_active_collision_pairs()) == 1

    posture = solver.add_posture_task("posture")
    assert posture.name == "posture"
    reference = eik.AccelerationTaskReference()
    reference.desired_acceleration = np.array([-5.0])
    solver.set_task_reference("posture", reference)

    result = solver.solve(
        np.array([0.0]),
        np.array([0.0]),
        0.01,
        _options_with_limits([100.0]),
    )

    assert result.status == eik.SolverStatus.SUCCESS
    assert result.native_collision_constraint_applied
    assert not result.velocity_collision_lift_applied
    assert len(result.native_collision_diagnostics) == 1
    assert (
        result.native_collision_diagnostics[0].regime == eik.AccelerationCollisionRegime.EXACT_FLOOR
    )

    solver.clear_collision_constraint()
    assert not solver.has_collision_constraint()
    assert solver.get_collision_constraint_definition() is None
    assert solver.get_collision_constraint_policy() is None
    assert solver.get_active_collision_pairs() == []


def test_sampled_velocity_collision_lift_is_non_certifying(tmp_path):
    capabilities = eik.AccelerationSolver.capabilities()
    if not capabilities.supports_velocity_collision_lift:
        pytest.skip("velocity collision lift unavailable")

    robot = _collision_robot(tmp_path)
    solver = eik.AccelerationSolver(robot)
    collision_solver = eik.KinematicsSolver(robot)
    collision_solver.dt = 0.01
    collision_solver.configure_collision_constraint(0.01, [], [], True, 1)
    collision_solver.set_collision_tuning_mode(eik.CollisionTuningMode.PRECISE)
    collision_solver.set_proximity_gated_collision_activation_enabled(False)

    lift = eik.VelocityCollisionLiftOptions()
    lift.validation_substeps = 3
    result = solver.solve_with_velocity_collision(
        collision_solver,
        np.array([0.0]),
        np.array([0.0]),
        0.01,
        _options_with_limits([10.0]),
        lift,
    )

    assert result.status == eik.SolverStatus.SUCCESS
    assert result.velocity_collision_lift_applied
    assert result.collision_endpoint_validated
    assert not result.collision_step_certified
    assert result.collision_validation_samples == 3
    assert result.collision_validation_allowed_pairs == 1
    assert result.collision_validation_pairs_checked == 3
    assert result.collision_lift_pairs_considered == 1
