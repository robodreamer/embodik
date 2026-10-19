"""SCALE and priority behavior during nonlinear translation refinement."""

import numpy as np
import pytest

import embodik as eik


@pytest.fixture
def case(tmp_path):
    urdf = tmp_path / "frame_translation_modes.urdf"
    urdf.write_text("""<robot name="frame_translation_modes">
<link name="base"/><link name="first"/><link name="second"/><link name="tool"/>
<joint name="joint1" type="revolute"><parent link="base"/><child link="first"/>
<axis xyz="0 0 1"/><limit lower="-3.2" upper="3.2" velocity="500" effort="100"/></joint>
<joint name="joint2" type="revolute"><parent link="first"/><child link="second"/>
<origin xyz="1 0 0"/><axis xyz="0 0 1"/>
<limit lower="-3.2" upper="3.2" velocity="500" effort="100"/></joint>
<joint name="tip" type="fixed"><parent link="second"/><child link="tool"/>
<origin xyz="1 0 0"/></joint></robot>""")
    robot = eik.RobotModel(str(urdf), floating_base=False)
    q, dq = np.array([-1.2, 0.0]), np.array([0.2, 0.0])
    robot.update_kinematics(q, dq)
    pose = robot.get_frame_pose("tool")
    target = np.eye(4)
    target[:3, :3], target[:3, 3] = pose.rotation, pose.translation
    definition = eik.TightFramePoseConstraintDefinition()
    definition.frame_name, definition.target_pose = "tool", target
    definition.axis_mask = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    definition.position_epsilon = 0.010276011537982174
    definition.orientation_epsilon = 0.1
    policy = eik.GeometricConstraintAccelerationPolicy()
    policy.rate_limits = np.ones(6)
    policy.acceleration_limits = np.full(6, 10.0)
    policy.lower_braking_accelerations = policy.upper_braking_accelerations = np.full(6, 10.0)
    constraint = eik.TightFramePoseAccelerationConstraint()
    constraint.source_id, constraint.definition, constraint.policy = "held_x", definition, policy
    options = eik.AccelerationSolveOptions()
    options.apply_position_limits = options.apply_velocity_limits = False
    options.acceleration_limits_override = np.full(2, 15.0)
    options.tight_frame_pose_constraints = [constraint]
    hard = eik.AffineAccelerationConstraint()
    hard.source_id = "second_joint_equality"
    hard.coefficient_matrix, hard.affine_bias = np.array([[0.0, 1.0]]), np.zeros(1)
    hard.lower_bounds = hard.upper_bounds = np.zeros(1)
    options.affine_constraints = [hard]
    return robot, q, dq, options


def reference(acceleration):
    result = eik.AccelerationTaskReference()
    result.proportional_gain = result.derivative_gain = 0.0
    result.desired_acceleration = np.asarray(acceleration)
    return result


@pytest.mark.parametrize("fallback", [False, True])
def test_refinement_respects_explicit_scale_fallback_policy(case, fallback):
    robot, q, dq, options = case
    solver = eik.AccelerationSolver(robot)
    task = solver.add_posture_task("drive")
    task.solve_mode, task.allow_min_error_fallback = eik.TaskSolveMode.SCALE, fallback
    solver.set_task_reference("drive", reference([-0.5, 0.0]))
    result = solver.solve(q, dq, 0.01, options)
    assert result.nonlinear_refinement_iterations > 0
    if not fallback:
        # Nonlinear viability needs more braking than the unit-scale target.
        assert result.status != eik.SolverStatus.SUCCESS
        assert result.joint_accelerations.size == result.q_solution.size == 0
    else:
        assert result.status == eik.SolverStatus.SUCCESS, result.status_message
        assert result.task_modes_effective[0] == eik.TaskSolveMode.MIN_ERROR
        assert result.task_used_fallback[0]
        assert result.joint_accelerations[0] < -0.5
        assert result.joint_accelerations[1] == pytest.approx(0.0, abs=1e-10)


def test_refinement_preserves_earlier_scale_hold_task(case):
    robot, q, dq, options = case
    solver = eik.AccelerationSolver(robot)
    hold = solver.add_joint_task("hold", "joint2", 0.0)
    hold.solve_mode, hold.priority = eik.TaskSolveMode.SCALE, 0
    solver.set_task_reference("hold", reference([0.0]))
    drive = solver.add_posture_task("drive")
    drive.solve_mode, drive.priority = eik.TaskSolveMode.MIN_ERROR, 1
    solver.set_task_reference("drive", reference([-0.5, 0.0]))
    result = solver.solve(q, dq, 0.01, options)
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    assert result.nonlinear_refinement_iterations > 0
    assert result.task_modes_effective[0] == eik.TaskSolveMode.SCALE
    assert result.task_scales[0] == pytest.approx(1.0)
    assert not result.task_used_fallback[0]
    assert result.joint_accelerations[0] < -0.5
    assert result.joint_accelerations[1] == pytest.approx(0.0, abs=1e-10)
