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

    # =================================================================
    # Configuration-space (Lie-group) operations
    # =================================================================

    def test_integrate_fixed_base(self, urdf_path):
        """Test integrate() for fixed-base (revolute) joints == q + v*dt."""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        q = np.array([0.5, -0.5])
        v = np.array([0.1, -0.2])
        dt = 0.01

        q_new = model.integrate(q, v, dt)

        # For revolute joints, integrate is simply q + v*dt
        np.testing.assert_allclose(q_new, q + v * dt, atol=1e-12)

    def test_integrate_default_dt(self, urdf_path):
        """Test integrate() with default dt=1.0."""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        q = np.zeros(2)
        v = np.array([0.3, -0.4])

        q_new = model.integrate(q, v)  # dt defaults to 1.0
        np.testing.assert_allclose(q_new, q + v, atol=1e-12)

    def test_integrate_floating_base(self, urdf_path):
        """Test integrate() for floating-base robot preserves quaternion unit norm."""
        model = embodik.RobotModel(urdf_path, floating_base=True)
        assert model.nq == 9  # 7 (SE3: xyz + wxyz quat) + 2 revolute
        assert model.nv == 8  # 6 (SE3 tangent) + 2 revolute

        q0 = model.neutral_configuration()
        # Small velocity: translate x + rotate around z + move joints
        v = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.2, 0.05, -0.05])
        dt = 0.01

        q1 = model.integrate(q0, v, dt)

        # Result must still have unit quaternion (indices 3..6 for freeflyer)
        quat = q1[3:7]
        np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-12)

        # Revolute joints should match simple addition
        np.testing.assert_allclose(q1[7:], q0[7:] + v[6:] * dt, atol=1e-12)

        # Simple addition would NOT preserve unit quaternion -- verify that
        q_naive = q0 + np.zeros(9)
        q_naive[:3] += v[:3] * dt
        q_naive[3:7] += np.array([0, 0, v[5], 0]) * dt  # WRONG: breaks quaternion norm
        q_naive[7:] += v[6:] * dt
        quat_naive = q_naive[3:7]
        # The naive quaternion norm should NOT be 1.0
        assert abs(np.linalg.norm(quat_naive) - 1.0) > 1e-6

    def test_difference_fixed_base(self, urdf_path):
        """Test difference() for fixed-base == q1 - q0."""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        q0 = np.array([0.1, 0.2])
        q1 = np.array([0.5, -0.3])

        v = model.difference(q0, q1)
        np.testing.assert_allclose(v, q1 - q0, atol=1e-12)

    def test_difference_floating_base(self, urdf_path):
        """Test difference() is inverse of integrate() for floating-base."""
        model = embodik.RobotModel(urdf_path, floating_base=True)

        q0 = model.neutral_configuration()
        v_original = np.array([0.1, 0.05, -0.02, 0.01, -0.03, 0.04, 0.1, -0.1])

        q1 = model.integrate(q0, v_original)
        v_recovered = model.difference(q0, q1)

        np.testing.assert_allclose(v_recovered, v_original, atol=1e-10)

    def test_integrate_difference_roundtrip(self, urdf_path):
        """Test that integrate(q0, difference(q0, q1)) == q1."""
        model = embodik.RobotModel(urdf_path, floating_base=True)

        q0 = model.neutral_configuration()
        v = np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.3, 0.1, -0.2])
        q1 = model.integrate(q0, v)

        v_diff = model.difference(q0, q1)
        q1_roundtrip = model.integrate(q0, v_diff)

        np.testing.assert_allclose(q1_roundtrip, q1, atol=1e-10)

    def test_neutral_configuration_fixed_base(self, urdf_path):
        """Test neutral_configuration() for fixed-base robot."""
        model = embodik.RobotModel(urdf_path, floating_base=False)
        q_neutral = model.neutral_configuration()
        assert q_neutral.shape == (2,)
        np.testing.assert_allclose(q_neutral, np.zeros(2), atol=1e-12)

    def test_neutral_configuration_floating_base(self, urdf_path):
        """Test neutral_configuration() for floating-base robot has valid quaternion."""
        model = embodik.RobotModel(urdf_path, floating_base=True)
        q_neutral = model.neutral_configuration()
        assert q_neutral.shape == (9,)
        # Quaternion part should be unit quaternion [0, 0, 0, 1] (pinocchio uses xyzw internally but stores as wxyz)
        quat = q_neutral[3:7]
        np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-12)

    def test_random_configuration(self, urdf_path):
        """Test random_configuration() returns valid configurations."""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        q = model.random_configuration()
        assert q.shape == (2,)

        # Should be within joint limits
        lower, upper = model.get_joint_limits()
        assert np.all(q >= lower - 1e-6)
        assert np.all(q <= upper + 1e-6)

    def test_random_configuration_floating_base(self, urdf_path):
        """Test random_configuration() for floating-base has valid quaternion."""
        model = embodik.RobotModel(urdf_path, floating_base=True)

        q = model.random_configuration()
        assert q.shape == (9,)

        # Quaternion should be normalized
        quat = q[3:7]
        np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-12)

    def test_normalize_fixed_base(self, urdf_path):
        """Test normalize() is no-op for fixed-base (all revolute)."""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        q = np.array([1.5, -2.3])
        q_norm = model.normalize(q)
        np.testing.assert_allclose(q_norm, q, atol=1e-12)

    def test_normalize_floating_base(self, urdf_path):
        """Test normalize() re-normalizes quaternion for floating-base."""
        model = embodik.RobotModel(urdf_path, floating_base=True)

        q = model.neutral_configuration()
        # Perturb quaternion to have non-unit norm
        q[3:7] = np.array([0.0, 0.0, 0.1, 2.0])
        assert abs(np.linalg.norm(q[3:7]) - 1.0) > 0.1  # Not normalized

        q_norm = model.normalize(q)
        # After normalize, quaternion should be unit
        np.testing.assert_allclose(np.linalg.norm(q_norm[3:7]), 1.0, atol=1e-12)

    def test_integrate_size_mismatch(self, urdf_path):
        """Test integrate() raises on wrong-sized inputs."""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        q = np.zeros(3)  # Wrong size
        v = np.zeros(2)
        with pytest.raises(RuntimeError):
            model.integrate(q, v)

        q = np.zeros(2)
        v = np.zeros(3)  # Wrong size
        with pytest.raises(RuntimeError):
            model.integrate(q, v)

    def test_difference_size_mismatch(self, urdf_path):
        """Test difference() raises on wrong-sized inputs."""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        q0 = np.zeros(3)  # Wrong size
        q1 = np.zeros(2)
        with pytest.raises(RuntimeError):
            model.difference(q0, q1)

    # =================================================================
    # Inverse dynamics / gravity
    # =================================================================

    def test_compute_generalized_gravity(self, urdf_path):
        """Test compute_generalized_gravity returns correct-sized vector."""
        model = embodik.RobotModel(urdf_path, floating_base=False)
        q = np.zeros(model.nq)
        g = model.compute_generalized_gravity(q)
        assert g.shape == (model.nv,)
        # At zero configuration, gravity torques should be finite
        assert np.all(np.isfinite(g))

    def test_gravity_matches_rnea_zeros(self, urdf_path):
        """Test that gravity == rnea(q, 0, 0)."""
        model = embodik.RobotModel(urdf_path, floating_base=False)
        q = np.array([0.5, -0.3])
        v_zero = np.zeros(model.nv)
        a_zero = np.zeros(model.nv)

        g = model.compute_generalized_gravity(q)
        tau_rnea = model.rnea(q, v_zero, a_zero)

        np.testing.assert_allclose(g, tau_rnea, atol=1e-12)

    def test_rnea_full_dynamics(self, urdf_path):
        """Test rnea with nonzero velocities and accelerations."""
        model = embodik.RobotModel(urdf_path, floating_base=False)
        q = np.array([0.5, -0.3])
        v = np.array([0.1, -0.2])
        a = np.array([1.0, -0.5])

        tau = model.rnea(q, v, a)
        assert tau.shape == (model.nv,)
        assert np.all(np.isfinite(tau))

        # Full dynamics should differ from gravity-only (nonzero v, a)
        g = model.compute_generalized_gravity(q)
        assert not np.allclose(tau, g)

    def test_rnea_decomposition(self, urdf_path):
        """Test that tau = g + coriolis + inertial via RNEA calls."""
        model = embodik.RobotModel(urdf_path, floating_base=False)
        q = np.array([0.5, -0.3])
        v = np.array([0.1, -0.2])
        a = np.array([1.0, -0.5])
        v_zero = np.zeros(model.nv)
        a_zero = np.zeros(model.nv)

        tau_gravity = model.rnea(q, v_zero, a_zero)
        tau_coriolis_grav = model.rnea(q, v, a_zero)
        tau_total = model.rnea(q, v, a)

        # Coriolis = rnea(q,v,0) - rnea(q,0,0)
        coriolis = tau_coriolis_grav - tau_gravity
        # Inertial = rnea(q,v,a) - rnea(q,v,0)
        inertial = tau_total - tau_coriolis_grav

        # Verify decomposition: gravity + coriolis + inertial == total
        np.testing.assert_allclose(
            tau_gravity + coriolis + inertial, tau_total, atol=1e-12
        )

    def test_compute_coriolis(self, urdf_path):
        """Test compute_coriolis returns C(q,v)*v."""
        model = embodik.RobotModel(urdf_path, floating_base=False)
        q = np.array([0.5, -0.3])
        v = np.array([0.1, -0.2])
        v_zero = np.zeros(model.nv)
        a_zero = np.zeros(model.nv)

        coriolis = model.compute_coriolis(q, v)
        assert coriolis.shape == (model.nv,)

        # Should match rnea(q,v,0) - gravity(q)
        expected = model.rnea(q, v, a_zero) - model.compute_generalized_gravity(q)
        np.testing.assert_allclose(coriolis, expected, atol=1e-12)

    def test_compute_mass_matrix(self, urdf_path):
        """Test compute_mass_matrix returns symmetric positive-definite matrix."""
        model = embodik.RobotModel(urdf_path, floating_base=False)
        q = np.array([0.5, -0.3])

        M = model.compute_mass_matrix(q)
        assert M.shape == (model.nv, model.nv)

        # Should be symmetric
        np.testing.assert_allclose(M, M.T, atol=1e-12)

        # Should be positive-definite (all eigenvalues > 0)
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues > 0)

    def test_mass_matrix_times_accel(self, urdf_path):
        """Test that M(q)*a matches inertial component of RNEA."""
        model = embodik.RobotModel(urdf_path, floating_base=False)
        q = np.array([0.5, -0.3])
        v_zero = np.zeros(model.nv)
        a = np.array([1.0, -0.5])

        M = model.compute_mass_matrix(q)
        g = model.compute_generalized_gravity(q)
        # tau = M*a + g (when v=0)
        tau_rnea = model.rnea(q, v_zero, a)
        tau_expected = M @ a + g

        np.testing.assert_allclose(tau_rnea, tau_expected, atol=1e-10)

    def test_set_get_gravity(self, urdf_path):
        """Test set_gravity and get_gravity."""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        # Default gravity should be [0, 0, -9.81]
        g = model.get_gravity()
        np.testing.assert_allclose(g, [0, 0, -9.81], atol=1e-6)

        # Set custom gravity
        model.set_gravity(np.array([0.0, 0.0, 0.0]))
        g = model.get_gravity()
        np.testing.assert_allclose(g, [0, 0, 0], atol=1e-12)

        # With zero gravity, gravity torques should be zero
        q = np.zeros(model.nq)
        grav_torques = model.compute_generalized_gravity(q)
        np.testing.assert_allclose(grav_torques, np.zeros(model.nv), atol=1e-12)

    def test_gravity_floating_base(self, urdf_path):
        """Test gravity computation for floating-base robot."""
        model = embodik.RobotModel(urdf_path, floating_base=True)
        q = model.neutral_configuration()
        g = model.compute_generalized_gravity(q)
        assert g.shape == (model.nv,)
        assert np.all(np.isfinite(g))

    def test_rnea_size_mismatch(self, urdf_path):
        """Test rnea raises on wrong-sized inputs."""
        model = embodik.RobotModel(urdf_path, floating_base=False)
        q = np.zeros(model.nq)
        v = np.zeros(model.nv)

        with pytest.raises(RuntimeError):
            model.rnea(np.zeros(3), v, v)  # Wrong q size
        with pytest.raises(RuntimeError):
            model.rnea(q, np.zeros(3), v)  # Wrong v size
        with pytest.raises(RuntimeError):
            model.rnea(q, v, np.zeros(3))  # Wrong a size

    # =================================================================
    # Boolean collision checking
    # =================================================================

    def test_check_collision_no_geometry(self, urdf_path):
        """Test check_collision returns False when no collision geometry."""
        model = embodik.RobotModel(urdf_path, floating_base=False)
        q = np.zeros(model.nq)
        model.update_configuration(q)
        # Our simple URDF has no collision geometry
        assert model.check_collision() == False

    def test_check_collision_with_geometry(self, tmp_path):
        """Test check_collision with collision geometry URDF."""
        # Create a URDF with collision geometry that will collide
        urdf_content = """<?xml version="1.0"?>
<robot name="collision_test">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.2"/>
      </geometry>
    </collision>
  </link>
  <link name="link1">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
    <collision>
      <origin xyz="0.05 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0.05 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" lower="-3.14" upper="3.14" velocity="1.0"/>
  </joint>
</robot>"""
        urdf_path = tmp_path / "collision_test.urdf"
        urdf_path.write_text(urdf_content)

        model = embodik.RobotModel(str(urdf_path), floating_base=False)
        if not model.has_collision_geometry():
            pytest.skip("Collision geometry not loaded for this URDF")

        # At zero config, boxes overlap -> should be in collision
        q_zero = np.zeros(model.nq)
        model.update_configuration(q_zero)
        assert model.check_collision() == True

        # With margin check
        assert model.check_collision(min_distance=0.01) == True

    def test_check_collision_with_distance_threshold(self, tmp_path):
        """Test check_collision with distance threshold (near-miss detection)."""
        urdf_content = """<?xml version="1.0"?>
<robot name="near_miss">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
    </collision>
  </link>
  <link name="link1">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
    <collision>
      <origin xyz="0.15 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
    </collision>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" lower="-3.14" upper="3.14" velocity="1.0"/>
  </joint>
</robot>"""
        urdf_path = tmp_path / "near_miss.urdf"
        urdf_path.write_text(urdf_content)

        model = embodik.RobotModel(str(urdf_path), floating_base=False)
        if not model.has_collision_geometry():
            pytest.skip("Collision geometry not loaded")

        q = np.zeros(model.nq)
        model.update_configuration(q)

        min_dist = model.compute_min_collision_distance()
        if np.isfinite(min_dist) and min_dist > 0:
            # Not in contact, but check with large threshold
            assert model.check_collision() == False
            assert model.check_collision(min_distance=min_dist + 0.01) == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
