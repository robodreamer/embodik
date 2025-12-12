#!/usr/bin/env python3
"""Tests for RobotModel Python bindings"""

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

  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.5"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="1.0" effort="10.0"/>
  </joint>

  <link name="end_effector">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="end_effector"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" velocity="1.0" effort="10.0"/>
  </joint>
</robot>"""

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
        f.write(urdf_content)
        return f.name


class TestRobotModel:
    """Test suite for RobotModel"""

    @pytest.fixture
    def urdf_path(self):
        """Create a test URDF file"""
        path = create_test_urdf()
        yield path
        # Cleanup
        os.unlink(path)

    def test_load_from_urdf(self, urdf_path):
        """Test loading robot model from URDF"""
        model = embodik.RobotModel(urdf_path, floating_base=False)
        assert model.nq == 2
        assert model.nv == 2
        assert not model.is_floating_base

    def test_load_with_floating_base(self, urdf_path):
        """Test loading robot with floating base"""
        model = embodik.RobotModel(urdf_path, floating_base=True)
        assert model.nq == 9  # 7 for floating base + 2 joints
        assert model.nv == 8  # 6 for floating base + 2 joints
        assert model.is_floating_base

    def test_update_configuration(self, urdf_path):
        """Test updating robot configuration"""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        q = np.array([0.5, -0.5])
        model.update_configuration(q)

        # Check that configuration is stored
        np.testing.assert_array_almost_equal(model.get_current_configuration(), q)

    def test_get_frame_pose(self, urdf_path):
        """Test getting frame pose"""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        q = np.zeros(2)
        model.update_configuration(q)

        # Get end effector pose
        pose = model.get_frame_pose("end_effector")

        # Check that it returns SE3 object
        assert hasattr(pose, 'translation')
        assert hasattr(pose, 'rotation')

        # At zero configuration, end effector should be at (0, 0, 2)
        assert abs(pose.translation[2] - 2.0) < 1e-6

    def test_get_frame_jacobian(self, urdf_path):
        """Test getting frame Jacobian"""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        q = np.zeros(2)
        model.update_configuration(q)

        # Get Jacobian with default reference frame
        J = model.get_frame_jacobian("end_effector")

        assert J.shape == (6, 2)

        # Test with different reference frames
        J_world = model.get_frame_jacobian("end_effector", embodik.ReferenceFrame.WORLD)
        J_local = model.get_frame_jacobian("end_effector", embodik.ReferenceFrame.LOCAL)

        assert J_world.shape == (6, 2)
        assert J_local.shape == (6, 2)

    def test_center_of_mass(self, urdf_path):
        """Test center of mass computations"""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        q = np.zeros(2)
        model.update_configuration(q)

        # Get COM position
        com = model.get_com_position()
        assert com.shape == (3,)
        assert com[2] > 0  # COM should be above ground

        # Get COM Jacobian
        J_com = model.get_com_jacobian()
        assert J_com.shape == (3, 2)

    def test_get_frame_names(self, urdf_path):
        """Test getting frame names"""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        frame_names = model.get_frame_names()
        assert isinstance(frame_names, list)
        assert "end_effector" in frame_names
        assert "base_link" in frame_names

    def test_get_joint_names(self, urdf_path):
        """Test getting joint names"""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        joint_names = model.get_joint_names()
        assert isinstance(joint_names, list)
        assert len(joint_names) == 2
        assert "joint1" in joint_names
        assert "joint2" in joint_names

    def test_get_joint_limits(self, urdf_path):
        """Test getting joint limits"""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        lower, upper = model.get_joint_limits()

        assert lower.shape == (2,)
        assert upper.shape == (2,)

        # Check limits match URDF
        np.testing.assert_almost_equal(lower[0], -3.14, decimal=5)
        np.testing.assert_almost_equal(upper[0], 3.14, decimal=5)

    def test_has_frame(self, urdf_path):
        """Test frame existence check"""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        assert model.has_frame("end_effector")
        assert model.has_frame("base_link")
        assert not model.has_frame("nonexistent_frame")

    def test_invalid_configuration_size(self, urdf_path):
        """Test error handling for wrong configuration size"""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        q_wrong = np.zeros(3)  # Wrong size

        with pytest.raises(RuntimeError):
            model.update_configuration(q_wrong)

    def test_invalid_frame_name(self, urdf_path):
        """Test error handling for invalid frame name"""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        q = np.zeros(2)
        model.update_configuration(q)

        with pytest.raises(RuntimeError):
            model.get_frame_pose("nonexistent_frame")

    def test_repr(self, urdf_path):
        """Test string representation"""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        repr_str = repr(model)
        assert "RobotModel" in repr_str
        assert "nq=2" in repr_str
        assert "nv=2" in repr_str
        assert "floating_base=False" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
