#!/usr/bin/env python3
"""Tests for embodiK Task framework"""

import pytest
import numpy as np
import tempfile
import os
import sys

import embodik


def create_test_urdf():
    """Create a simple test URDF file"""
    urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="100"/>
  </joint>

  <link name="link1">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0.05"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="100"/>
  </joint>

  <link name="link2">
    <inertial>
      <mass value="0.3"/>
      <origin xyz="0 0 0.05"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
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
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
</robot>
"""
    fd, path = tempfile.mkstemp(suffix=".urdf")
    with os.fdopen(fd, 'w') as f:
        f.write(urdf_content)
    return path


class TestTasks:

    @pytest.fixture
    def robot_model(self):
        """Create a test robot model"""
        urdf_path = create_test_urdf()
        model = embodik.RobotModel(urdf_path, floating_base=False)
        yield model
        os.remove(urdf_path)

    def test_frame_task_position(self, robot_model):
        """Test FrameTask for position tracking"""
        # Create a position-only frame task
        task = embodik.FrameTask("ee_position", robot_model, "end_effector",
                                  embodik.TaskType.FRAME_POSITION, priority=0, weight=1.0)

        # Set target position
        target_pos = np.array([0.1, 0.2, 0.3])
        task.set_target_position(target_pos)

        # Update task with current robot state
        task.update(robot_model)

        # Check dimensions
        assert task.get_dimension() == 3
        assert task.get_type() == embodik.TaskType.FRAME_POSITION

        # Get error and jacobian
        error = task.get_error()
        jacobian = task.get_jacobian()

        assert error.shape == (3,)
        assert jacobian.shape == (3, robot_model.nv)

        # Check that current position is accessible
        current_pos = task.current_position
        assert current_pos.shape == (3,)

    def test_frame_task_orientation(self, robot_model):
        """Test FrameTask for orientation tracking"""
        # Create an orientation-only frame task
        task = embodik.FrameTask("ee_orientation", robot_model, "end_effector",
                                  embodik.TaskType.FRAME_ORIENTATION, priority=0, weight=1.0)

        # Set target orientation (rotation matrix)
        target_rot = embodik.rotation_from_rpy(0.1, 0.2, 0.3)
        task.set_target_orientation(target_rot)

        # Update task
        task.update(robot_model)

        # Check dimensions
        assert task.get_dimension() == 3
        assert task.get_type() == embodik.TaskType.FRAME_ORIENTATION

        # Get error and jacobian
        error = task.get_error()
        jacobian = task.get_jacobian()

        assert error.shape == (3,)
        assert jacobian.shape == (3, robot_model.nv)

    def test_frame_task_pose(self, robot_model):
        """Test FrameTask for full pose tracking"""
        # Create a pose frame task
        task = embodik.FrameTask("ee_pose", robot_model, "end_effector",
                                  embodik.TaskType.FRAME_POSE, priority=0, weight=1.0)

        # Set target pose
        target_pos = np.array([0.1, 0.2, 0.3])
        target_rot = embodik.rotation_from_rpy(0.1, 0.2, 0.3)
        task.set_target_pose(target_pos, target_rot)

        # Update task
        task.update(robot_model)

        # Check dimensions
        assert task.get_dimension() == 6
        assert task.get_type() == embodik.TaskType.FRAME_POSE

        # Get error and jacobian
        error = task.get_error()
        jacobian = task.get_jacobian()

        assert error.shape == (6,)
        assert jacobian.shape == (6, robot_model.nv)

    def test_com_task(self, robot_model):
        """Test COMTask"""
        # Create COM task
        task = embodik.COMTask("com", robot_model, priority=1, weight=1.0)

        # Set target COM position
        target_com = np.array([0.0, 0.1, 0.15])
        task.set_target_position(target_com)

        # Update task
        task.update(robot_model)

        # Check dimensions
        assert task.get_dimension() == 3
        assert task.get_type() == embodik.TaskType.COM

        # Get error and jacobian
        error = task.get_error()
        jacobian = task.get_jacobian()

        assert error.shape == (3,)
        assert jacobian.shape == (3, robot_model.nv)

        # Check current COM
        current_com = task.current_position
        assert current_com.shape == (3,)

    def test_posture_task_all_joints(self, robot_model):
        """Test PostureTask for all joints"""
        # Create posture task for all joints
        task = embodik.PostureTask("posture", robot_model, priority=10, weight=0.1)

        # Set target configuration
        q_target = np.array([0.5, -0.5])
        task.set_target_configuration(q_target)

        # Update task
        task.update(robot_model)

        # Check dimensions
        assert task.get_dimension() == robot_model.nv
        assert task.get_type() == embodik.TaskType.POSTURE

        # Get error and jacobian
        error = task.get_error()
        jacobian = task.get_jacobian()

        assert error.shape == (robot_model.nv,)
        assert jacobian.shape == (robot_model.nv, robot_model.nv)

    def test_posture_task_partial_joints(self, robot_model):
        """Test PostureTask for specific joints"""
        # Create posture task for specific joints (only joint1)
        controlled_indices = [0]  # Only control first joint
        task = embodik.PostureTask("posture_partial", robot_model,
                                    controlled_indices, priority=10, weight=0.1)

        # Set target values for controlled joints
        target_values = np.array([0.5])
        task.set_controlled_joint_targets(target_values)

        # Update task
        task.update(robot_model)

        # Check dimensions
        assert task.get_dimension() == 1
        assert task.get_type() == embodik.TaskType.POSTURE

        # Get error and jacobian
        error = task.get_error()
        jacobian = task.get_jacobian()

        assert error.shape == (1,)
        assert jacobian.shape == (1, robot_model.nv)

        # Check controlled indices
        indices = task.controlled_joint_indices
        assert indices == controlled_indices

    def test_joint_task(self, robot_model):
        """Test JointTask"""
        # Create joint task by index
        task = embodik.JointTask("joint1", robot_model, 0,
                                  target_value=0.5, priority=0, weight=1.0)

        # Update task
        task.update(robot_model)

        # Check dimensions
        assert task.get_dimension() == 1
        assert task.get_type() == embodik.TaskType.JOINT

        # Get error and jacobian
        error = task.get_error()
        jacobian = task.get_jacobian()

        assert error.shape == (1,)
        assert jacobian.shape == (1, robot_model.nv)

        # Change target
        task.set_target_value(1.0)
        error2 = task.get_error()
        assert np.abs(error2[0] - error[0] - 0.5) < 1e-6

    def test_multi_joint_task(self, robot_model):
        """Test MultiJointTask"""
        # Create multi-joint task controlling both joints
        joint_indices = [0, 1]
        target_values = np.array([0.5, -0.5])
        task = embodik.MultiJointTask("multi_joint", robot_model,
                                       joint_indices, target_values,
                                       priority=0, weight=1.0)

        # Update task
        task.update(robot_model)

        # Check dimensions
        assert task.get_dimension() == 2
        assert task.get_type() == embodik.TaskType.JOINT

        # Get error and jacobian
        error = task.get_error()
        jacobian = task.get_jacobian()

        assert error.shape == (2,)
        assert jacobian.shape == (2, robot_model.nv)

        # Check joint indices
        indices = task.joint_indices
        assert indices == joint_indices

        # Check target values
        targets = task.target_values
        assert np.allclose(targets, target_values)

    def test_task_priorities_and_weights(self, robot_model):
        """Test task priority and weight settings"""
        task = embodik.FrameTask("test", robot_model, "end_effector")

        # Check initial values
        assert task.priority == 0
        assert task.weight == 1.0
        assert task.active == True

        # Set new values
        task.priority = 5
        task.weight = 2.0
        task.active = False

        # Verify changes
        assert task.priority == 5
        assert task.weight == 2.0
        assert task.active == False

        # Check that velocity is scaled by weight
        task.active = True
        task.update(robot_model)
        error = task.get_error()
        velocity = task.get_velocity()
        assert np.allclose(velocity, -2.0 * error)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
