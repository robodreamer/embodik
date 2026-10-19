"""Task-policy and hard-row preservation during SO(3) braking refinement."""

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("mode,fallback", [("MIN_ERROR", False), ("SCALE", False), ("SCALE", True)])
def test_rotation_refinement_preserves_task_policy_and_joint_equality(tmp_path, mode, fallback):
    urdf = tmp_path / "rotation_braking.urdf"
    urdf.write_text("""<robot name="rotation_braking">
<link name="base"/><link name="first"/><link name="second"/><link name="tool"/>
<joint name="joint1" type="revolute"><parent link="base"/><child link="first"/>
<axis xyz="0 1 0"/><limit lower="-3.2" upper="3.2" velocity="500" effort="100"/></joint>
<joint name="joint2" type="revolute"><parent link="first"/><child link="second"/>
<origin xyz="1 0 0"/><axis xyz="0 0 1"/>
<limit lower="-3.2" upper="3.2" velocity="500" effort="100"/></joint>
<joint name="tip" type="fixed"><parent link="second"/><child link="tool"/>
<origin xyz="1 0 0"/></joint></robot>""")
    robot = eik.RobotModel(str(urdf), floating_base=False)
    q, dq = np.array([0.4, -0.3]), np.array([0.2, 0.1])
    robot.update_kinematics(q, dq)
    pose = robot.get_frame_pose("tool")
    target = np.eye(4)
    target[:3, :3], target[:3, 3] = pose.rotation, pose.translation
    definition = eik.TightFramePoseConstraintDefinition()
    definition.frame_name, definition.target_pose = "tool", target
    definition.axis_mask = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    definition.position_epsilon = 0.1
    definition.orientation_epsilon = 0.0010007881529456087
    policy = eik.GeometricConstraintAccelerationPolicy()
    policy.rate_limits = np.ones(6)
    policy.acceleration_limits = np.full(6, 20.0)
    policy.lower_braking_accelerations = policy.upper_braking_accelerations = np.full(6, 4.0)
    constraint = eik.TightFramePoseAccelerationConstraint()
    constraint.source_id, constraint.definition, constraint.policy = "held_rx", definition, policy
    options = eik.AccelerationSolveOptions()
    options.apply_position_limits = options.apply_velocity_limits = False
    options.acceleration_limits_override = np.full(2, 15.0)
    options.tight_frame_pose_constraints = [constraint]
    hard = eik.AffineAccelerationConstraint()
    hard.source_id = "second_joint_equality"
    hard.coefficient_matrix, hard.affine_bias = np.array([[0.0, 1.0]]), np.zeros(1)
    hard.lower_bounds = hard.upper_bounds = np.array([0.15])
    options.affine_constraints = [hard]
    solver = eik.AccelerationSolver(robot)
    task = solver.add_posture_task("drive")
    task.solve_mode = getattr(eik.TaskSolveMode, mode)
    task.allow_min_error_fallback = fallback
    reference = eik.AccelerationTaskReference()
    reference.proportional_gain = reference.derivative_gain = 0.0
    reference.desired_acceleration = np.array([-0.4, 0.15])
    solver.set_task_reference("drive", reference)
    result = solver.solve(q, dq, 0.01, options)
    assert result.nonlinear_refinement_iterations > 0
    if mode == "SCALE" and not fallback:
        assert result.status != eik.SolverStatus.SUCCESS
        assert result.joint_accelerations.size == result.q_solution.size == 0
        return
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    assert result.task_modes_effective[0] == eik.TaskSolveMode.MIN_ERROR
    assert bool(result.task_used_fallback[0]) == fallback
    assert result.joint_accelerations[1] == pytest.approx(0.15, abs=1e-10)
    assert np.max(np.abs(result.joint_accelerations)) <= 15.0
    np.testing.assert_allclose(
        result.q_solution, q + 0.01 * dq + 0.00005 * result.joint_accelerations, atol=1e-12, rtol=0
    )
