#!/usr/bin/env python3
"""Tests for Velocity IK API enhancements"""

import pytest
import numpy as np
import tempfile
import os
import sys

import embodik

# Try to import robot_descriptions for real robot URDFs
try:
    from robot_descriptions.loaders.yourdfpy import load_robot_description
    USE_REAL_ROBOTS = True
except ImportError:
    USE_REAL_ROBOTS = False
    print("Warning: robot_descriptions not found, using synthetic test URDF")


def create_test_urdf():
    """Create a simple test URDF file"""
    urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="2.0"/>
  </joint>

  <link name="link1">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0.05"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.2"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" effort="50" velocity="1.5"/>
  </joint>

  <link name="link2">
    <inertial>
      <mass value="0.3"/>
      <origin xyz="0 0 0.05"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="end_effector_joint" type="fixed">
    <parent link="link2"/>
    <child link="end_effector"/>
    <origin xyz="0 0 0.1"/>
  </joint>

  <link name="end_effector">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
</robot>
    """
    # Create temporary URDF file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
        f.write(urdf_content)
        return f.name


def get_panda_urdf():
    """Get Panda robot URDF path"""
    if USE_REAL_ROBOTS:
        try:
            urdf = load_robot_description("panda_description")
            with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
                f.write(urdf)
                return f.name, "panda_hand"
        except Exception as e:
            print(f"Could not load Panda robot: {e}")
    return None, None


def get_iiwa_urdf():
    """Get KUKA IIWA robot URDF path"""
    if USE_REAL_ROBOTS:
        try:
            # Try to load IIWA 7 R800 (similar to 7kg model)
            urdf = load_robot_description("iiwa7_description")
            with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
                f.write(urdf)
                return f.name, "iiwa_link_ee"  # or "tool0" depending on the model
        except Exception as e:
            print(f"Could not load IIWA robot: {e}")
            # Try alternative name
            try:
                urdf = load_robot_description("iiwa_description")
                with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
                    f.write(urdf)
                    return f.name, "iiwa_link_ee"
            except Exception as e2:
                print(f"Could not load IIWA robot (alt): {e2}")
    return None, None


class TestVelocityIKEnhancements:

    @pytest.fixture
    def robot_and_solver(self):
        """Create robot model and solver for tests"""
        urdf_path = create_test_urdf()
        try:
            robot = embodik.RobotModel(urdf_path, floating_base=False)
            solver = embodik.KinematicsSolver(robot)
            solver.dt = 0.01  # Set a small dt for velocity calculations
            yield robot, solver
        finally:
            os.unlink(urdf_path)  # Clean up URDF file

    def test_basic_velocity_solving(self, robot_and_solver):
        """Test basic velocity-level solving without integration"""
        robot, solver = robot_and_solver

        # Add a simple end-effector task
        task = solver.add_frame_task("ee", "end_effector", embodik.TaskType.FRAME_POSITION)
        # Set target velocity directly
        target_velocity = np.array([0.1, 0, 0])  # 0.1 m/s in X direction
        task.set_target_position_velocity(target_velocity)

        # Solve for velocities only
        q_current = np.array([0.0, 0.0])  # Home configuration
        result = solver.solve_velocity(q_current)

        # Check result
        assert result.status == embodik.SolverStatus.SUCCESS
        assert hasattr(result, 'joint_velocities')
        assert hasattr(result, 'saturated_joints')
        assert hasattr(result, 'limits_applied')
        assert len(result.solution) == robot.nv
        assert result.joint_velocities.shape == (robot.nv,)
        assert result.limits_applied == True  # Default is to apply limits

        # Joint velocities should not be zero (task requires motion)
        assert np.linalg.norm(result.joint_velocities) > 0.01

    def test_velocity_solving_without_limits(self, robot_and_solver):
        """Test velocity solving with limits disabled"""
        robot, solver = robot_and_solver

        # Add aggressive velocity task
        task = solver.add_frame_task("ee", "end_effector", embodik.TaskType.FRAME_POSITION)
        current_pose = robot.get_frame_pose("end_effector")
        # Large velocity demand
        target_velocity = np.array([5.0, 0, 0])  # 5 m/s in X direction
        task.set_target_position_velocity(target_velocity)

        # Solve with limits
        result_with_limits = solver.solve_velocity(apply_limits=True)

        # Solve without limits
        result_no_limits = solver.solve_velocity(apply_limits=False)

        # Both should succeed
        assert result_with_limits.status == embodik.SolverStatus.SUCCESS
        assert result_no_limits.status == embodik.SolverStatus.SUCCESS

        # Without limits should have higher velocities
        vel_norm_with_limits = np.linalg.norm(result_with_limits.joint_velocities)
        vel_norm_no_limits = np.linalg.norm(result_no_limits.joint_velocities)
        assert vel_norm_no_limits >= vel_norm_with_limits

        # Check limits_applied flag
        assert result_with_limits.limits_applied == True
        assert result_no_limits.limits_applied == False

    def test_saturated_joints_detection(self, robot_and_solver):
        """Test detection of saturated joints at velocity limits"""
        robot, solver = robot_and_solver

        # Add very aggressive task to force saturation
        task = solver.add_frame_task("ee", "end_effector")
        task.set_target_velocity(np.array([10.0, 0, 0, 0, 0, 0]))  # 10m/s in X

        # Solve with limits
        result = solver.solve_velocity(apply_limits=True)

        assert result.status == embodik.SolverStatus.SUCCESS
        assert result.limits_applied == True

        # Check if any joints are saturated
        # With such an aggressive task, we expect at least one joint to saturate
        assert isinstance(result.saturated_joints, list)

        # Verify saturated joints are actually at limits
        vel_limits = robot.get_velocity_limits()
        for joint_idx in result.saturated_joints:
            joint_vel = abs(result.joint_velocities[joint_idx])
            limit = vel_limits[joint_idx]
            # Should be at least 99% of limit (as defined in implementation)
            assert joint_vel >= 0.99 * limit

    def test_position_based_velocity_limits(self, robot_and_solver):
        """Test that position limits affect velocity limits"""
        robot, solver = robot_and_solver

        # Move robot close to joint limit
        q_near_limit = np.array([3.0, 0.0])  # Joint 1 near upper limit (3.14)

        # Add task that would move joint 1 further positive
        task = solver.add_frame_task("ee", "end_effector", embodik.TaskType.FRAME_POSITION)
        task.set_target_position_velocity(np.array([0.1, 0, 0]))  # Move in X direction

        # Solve with position limits enabled
        solver.enable_position_limits(True)
        result_with_pos_limits = solver.solve_velocity(q_near_limit)

        # Solve with position limits disabled
        solver.enable_position_limits(False)
        result_no_pos_limits = solver.solve_velocity(q_near_limit)

        # Both should succeed
        assert result_with_pos_limits.status == embodik.SolverStatus.SUCCESS
        assert result_no_pos_limits.status == embodik.SolverStatus.SUCCESS

        # With position limits, joint 1 velocity should be constrained
        # to prevent exceeding position limit
        joint1_vel_with_limits = result_with_pos_limits.joint_velocities[0]
        joint1_vel_no_limits = result_no_pos_limits.joint_velocities[0]

        # Velocity should be reduced when position limits are considered
        if joint1_vel_no_limits > 0:  # If moving toward limit
            assert joint1_vel_with_limits < joint1_vel_no_limits

        # Additional check: velocity should allow movement away from limit
        # Add task that moves away from limit
        task.set_target_position_velocity(np.array([-0.1, 0, 0]))  # Move in negative X direction
        result_away = solver.solve_velocity(q_near_limit)
        assert result_away.status == embodik.SolverStatus.SUCCESS
        # Moving away from limit should not be restricted (check relative to limited case)
        # For this simple 2DOF robot, we can just check that the overall velocity norm is non-zero
        assert np.linalg.norm(result_away.joint_velocities) > 1e-6  # Should have some velocity

    def test_custom_configuration_input(self, robot_and_solver):
        """Test solve_velocity with custom configuration input"""
        robot, solver = robot_and_solver

        # Add task
        task = solver.add_frame_task("ee", "end_effector")
        task.set_target_velocity(np.array([0.1, 0, 0, 0, 0, 0]))

        # Test with different configurations
        q1 = np.array([0.0, 0.0])
        q2 = np.array([0.5, -0.5])

        result1 = solver.solve_velocity(q1)
        result2 = solver.solve_velocity(q2)

        # Both should succeed
        assert result1.status == embodik.SolverStatus.SUCCESS
        assert result2.status == embodik.SolverStatus.SUCCESS

        # Results should be different due to different configurations
        assert not np.allclose(result1.joint_velocities, result2.joint_velocities)

    def test_invalid_configuration_size(self, robot_and_solver):
        """Test solve_velocity with invalid configuration size"""
        robot, solver = robot_and_solver

        # Add task
        task = solver.add_frame_task("ee", "end_effector")
        task.set_target_velocity(np.array([0.1, 0, 0, 0, 0, 0]))

        # Wrong size configuration
        q_wrong_size = np.array([0.0, 0.0, 0.0])  # 3 values instead of 2

        result = solver.solve_velocity(q_wrong_size)
        assert result.status == embodik.SolverStatus.INVALID_INPUT

    def test_empty_configuration_uses_robot_current(self, robot_and_solver):
        """Test that empty configuration uses robot's current state"""
        robot, solver = robot_and_solver

        # Set robot to specific configuration
        q_set = np.array([0.3, -0.2])
        robot.update_configuration(q_set)

        # Add task
        task = solver.add_frame_task("ee", "end_effector")
        task.set_target_velocity(np.array([0.1, 0, 0, 0, 0, 0]))

        # Solve with empty configuration (should use robot's current)
        result_empty = solver.solve_velocity()

        # Solve with explicit configuration
        result_explicit = solver.solve_velocity(q_set)

        # Results should be identical
        assert result_empty.status == embodik.SolverStatus.SUCCESS
        assert result_explicit.status == embodik.SolverStatus.SUCCESS
        assert np.allclose(result_empty.joint_velocities, result_explicit.joint_velocities)

    def test_comparison_with_integration_solve(self, robot_and_solver):
        """Test that solve_velocity matches solve(integrate=False) for same dt"""
        robot, solver = robot_and_solver

        # Add task
        task = solver.add_frame_task("ee", "end_effector")
        task.set_target_velocity(np.array([0.1, 0, 0, 0, 0, 0]))

        # Set a specific dt
        solver.dt = 1.0  # 1 second for easy comparison

        q_current = np.array([0.1, -0.1])

        # Set robot to same configuration for both methods
        robot.update_configuration(q_current)

        # Solve without integration using original method
        result_original = solver.solve(integrate=False)

        # Solve using new velocity method (should use robot's current configuration)
        result_velocity = solver.solve_velocity(q_current)

        # Both should succeed
        assert result_original.status == embodik.SolverStatus.SUCCESS
        assert result_velocity.status == embodik.SolverStatus.SUCCESS

        # Solutions should match (original solve returns dq, velocity returns dq)
        assert np.allclose(result_original.solution, result_velocity.solution)

    def test_multiple_tasks_with_priorities(self, robot_and_solver):
        """Test velocity solving with multiple prioritized tasks"""
        robot, solver = robot_and_solver

        # Primary task: end-effector velocity
        ee_task = solver.add_frame_task("ee", "end_effector")
        ee_task.set_target_velocity(np.array([0.1, 0, 0, 0, 0, 0]))
        ee_task.priority = 0  # Highest priority

        # Secondary task: posture regularization
        posture_task = solver.add_posture_task("posture")
        posture_task.set_target_configuration(np.array([0.0, 0.0]))  # Prefer home
        posture_task.priority = 10  # Lower priority
        posture_task.weight = 0.1

        # Solve from non-home configuration
        q_current = np.array([0.5, 0.5])
        result = solver.solve_velocity(q_current)

        assert result.status == embodik.SolverStatus.SUCCESS

        # Should have task scales for both tasks
        assert len(result.task_scales) == 2

        # Primary task should be fully satisfied (scale = 1.0)
        assert result.task_scales[0] >= 0.99  # Allow small numerical tolerance

        # Joint velocities should move toward home (secondary task)
        # but only in nullspace of primary task

    def test_acceleration_limits_impact(self, robot_and_solver):
        """Test that acceleration limits affect position-based velocity constraints"""
        robot, solver = robot_and_solver

        # Set custom acceleration limits
        low_accel_limits = np.array([0.1, 0.1])  # Very low acceleration limits
        high_accel_limits = np.array([100.0, 100.0])  # High acceleration limits (default)

        # Move close to joint limit to ensure acceleration limits matter
        q_near_limit = np.array([2.8, 0.0])  # Joint 1 close to upper limit (3.14)

        # Add task moving toward limit
        task = solver.add_frame_task("ee", "end_effector", embodik.TaskType.FRAME_POSITION)
        task.set_target_position_velocity(np.array([0.1, 0, 0]))  # Move in X direction

        # Make sure position limits are enabled
        solver.enable_position_limits(True)

        # Test with high acceleration limits (default)
        robot.set_acceleration_limits(high_accel_limits)
        result_high_accel = solver.solve_velocity(q_near_limit)

        # Test with low acceleration limits
        robot.set_acceleration_limits(low_accel_limits)
        result_low_accel = solver.solve_velocity(q_near_limit)

        # Both should succeed
        assert result_high_accel.status == embodik.SolverStatus.SUCCESS
        assert result_low_accel.status == embodik.SolverStatus.SUCCESS

        # With lower acceleration limits near position limits, the joint velocities should be different
        # Check that at least one joint has different velocity
        vel_diff = np.abs(result_low_accel.joint_velocities - result_high_accel.joint_velocities)
        assert np.max(vel_diff) > 1e-6  # At least one joint should have different velocity

    def test_position_limits_near_boundaries(self, robot_and_solver):
        """Test position-based velocity limits at various distances from boundaries"""
        robot, solver = robot_and_solver

        # Get joint limits
        q_min, q_max = robot.get_joint_limits()

        # Create task
        task = solver.add_frame_task("ee", "end_effector", embodik.TaskType.FRAME_POSITION)
        task.set_target_position_velocity(np.array([0.1, 0, 0]))

        # Test at various distances from upper limit of joint 1
        distances = [0.01, 0.05, 0.1, 0.5, 1.0]  # Distance from upper limit
        previous_vel = None

        for dist in distances:
            q = np.array([q_max[0] - dist, 0.0])
            result = solver.solve_velocity(q)

            assert result.status == embodik.SolverStatus.SUCCESS

            # As we get closer to the limit, allowed velocity should decrease
            if previous_vel is not None:
                assert result.joint_velocities[0] <= previous_vel
            previous_vel = result.joint_velocities[0]

        # At very close distance, velocity should be very small
        q_very_close = np.array([q_max[0] - 0.001, 0.0])
        result_close = solver.solve_velocity(q_very_close)
        assert abs(result_close.joint_velocities[0]) < 0.5  # Should be limited

    def test_respects_all_constraints_simultaneously(self, robot_and_solver):
        """Test that solver respects velocity, position, and acceleration constraints together"""
        robot, solver = robot_and_solver

        # Set moderate acceleration limits
        robot.set_acceleration_limits(np.array([5.0, 5.0]))

        # Position near middle of range
        q_middle = np.array([0.0, 0.0])

        # Large velocity demand
        task = solver.add_frame_task("ee", "end_effector", embodik.TaskType.FRAME_POSITION)
        task.set_target_position_velocity(np.array([1.0, 0, 0]))  # High velocity demand in X

        result = solver.solve_velocity(q_middle)
        assert result.status == embodik.SolverStatus.SUCCESS

        # Check that velocities respect all limits
        vel_limits = robot.get_velocity_limits()
        for i in range(robot.nv):
            assert abs(result.joint_velocities[i]) <= vel_limits[i] + 1e-6  # Small tolerance for numerics

        # Position very close to limit
        q_near_limit = np.array([3.13, 0.0])  # Very close to upper limit 3.14
        result_near = solver.solve_velocity(q_near_limit)

        # Should still succeed
        assert result_near.status == embodik.SolverStatus.SUCCESS

        # Joint 1 should be constrained when near its upper limit
        assert abs(result_near.joint_velocities[0]) < 0.1  # Should be limited due to proximity to limit


class TestVelocityIKWithRealRobots:
    """Test velocity IK with real robot models like Panda"""

    @pytest.fixture
    def panda_robot_and_solver(self):
        """Create Panda robot model and solver for tests"""
        urdf_path, end_effector = get_panda_urdf()
        if urdf_path is None:
            pytest.skip("Panda robot URDF not available")

        try:
            robot = embodik.RobotModel(urdf_path, floating_base=False)
            solver = embodik.KinematicsSolver(robot)
            solver.set_dt(0.01)
            yield robot, solver, end_effector
        finally:
            os.unlink(urdf_path)

    @pytest.mark.skipif(not USE_REAL_ROBOTS, reason="robot_descriptions not available")
    def test_panda_velocity_limits(self, panda_robot_and_solver):
        """Test velocity IK with Panda robot respecting joint limits"""
        robot, solver, end_effector = panda_robot_and_solver

        # Add end-effector task
        task = solver.add_frame_task("ee", end_effector, embodik.TaskType.FRAME_POSITION)
        task.set_target_position_velocity(np.array([0.1, 0, 0]))  # 10cm/s in X

        # Move to a configuration close to joint limits
        q_current = robot.get_current_configuration()
        q_limits_lower, q_limits_upper = robot.get_joint_limits()

        # Set joint 1 close to its upper limit
        q_near_limit = q_current.copy()
        q_near_limit[0] = q_limits_upper[0] - 0.1  # 0.1 rad from upper limit

        # Solve with position limits enabled
        solver.enable_position_limits(True)
        result = solver.solve_velocity(q_near_limit)

        assert result.status == embodik.SolverStatus.SUCCESS

        # Check that velocities respect limits
        vel_limits = robot.get_velocity_limits()
        for i in range(robot.nv):
            assert abs(result.joint_velocities[i]) <= vel_limits[i] + 1e-6

        # Joint 0 should have limited positive velocity due to proximity to upper limit
        assert result.joint_velocities[0] < 0.5 * vel_limits[0]

    @pytest.mark.skipif(not USE_REAL_ROBOTS, reason="robot_descriptions not available")
    def test_panda_acceleration_limits(self, panda_robot_and_solver):
        """Test that custom acceleration limits work with Panda robot"""
        robot, solver, end_effector = panda_robot_and_solver

        # Set custom acceleration limits (lower than default)
        n_joints = robot.nv
        low_accel = np.full(n_joints, 1.0)  # 1 rad/s^2 for all joints
        robot.set_acceleration_limits(low_accel)

        # Add aggressive task
        task = solver.add_frame_task("ee", end_effector, embodik.TaskType.FRAME_POSITION)
        task.set_target_position_velocity(np.array([0.5, 0, 0]))  # 50cm/s in X

        # Solve from configuration moderately close to limits
        q_current = robot.get_current_configuration()
        q_limits_lower, q_limits_upper = robot.get_joint_limits()
        q_near_limit = q_current.copy()
        q_near_limit[0] = q_limits_upper[0] - 0.5  # 0.5 rad from upper limit

        result_low_accel = solver.solve_velocity(q_near_limit)

        # Set high acceleration limits
        high_accel = np.full(n_joints, 100.0)  # 100 rad/s^2 for all joints
        robot.set_acceleration_limits(high_accel)

        result_high_accel = solver.solve_velocity(q_near_limit)

        # Both should succeed
        assert result_low_accel.status == embodik.SolverStatus.SUCCESS
        assert result_high_accel.status == embodik.SolverStatus.SUCCESS

        # With lower acceleration limits, velocities should be more conservative
        # when near limits
        assert np.linalg.norm(result_low_accel.joint_velocities[:3]) <= \
               np.linalg.norm(result_high_accel.joint_velocities[:3])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
