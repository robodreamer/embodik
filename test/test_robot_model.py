#!/usr/bin/env python3
"""Tests for RobotModel Python bindings"""

import os
import sys
import tempfile

import numpy as np
import pytest

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
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
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
        assert hasattr(pose, "translation")
        assert hasattr(pose, "rotation")

        # At zero configuration, end effector should be at (0, 0, 2)
        assert abs(pose.translation[2] - 2.0) < 1e-6

    def test_se3_composition_operator(self):
        """Test SE3 composition via operator* (torso_emb * target_se3)"""
        # Create two SE3 transforms
        R1 = np.eye(3)
        t1 = np.array([1.0, 0.0, 0.0])
        T1 = embodik.SE3(R1, t1)

        R2 = np.eye(3)
        t2 = np.array([0.0, 1.0, 0.0])
        T2 = embodik.SE3(R2, t2)

        # Compose: T1 * T2 (Pinocchio: ^A M_B * ^B M_C = ^A M_C)
        T_composed = T1 * T2

        # Expected: R = R1 @ R2, t = R1 @ t2 + t1
        R_expected = R1 @ R2
        t_expected = R1 @ t2 + t1
        np.testing.assert_array_almost_equal(T_composed.rotation, R_expected)
        np.testing.assert_array_almost_equal(T_composed.translation, t_expected)

        # Verify inverse: T * T.inverse() = identity
        T_inv = T1.inverse()
        T_id = T1 * T_inv
        np.testing.assert_array_almost_equal(T_id.rotation, np.eye(3))
        np.testing.assert_array_almost_equal(T_id.translation, np.zeros(3))

    def test_se3_equality_operators(self):
        """Test SE3 operator== and operator!="""
        T1 = embodik.SE3(np.eye(3), np.array([1.0, 0.0, 0.0]))
        T2 = embodik.SE3(np.eye(3), np.array([1.0, 0.0, 0.0]))
        T3 = embodik.SE3(np.eye(3), np.array([0.0, 1.0, 0.0]))
        assert T1 == T2
        assert T1 != T3
        assert not (T1 != T2)
        assert not (T1 == T3)

    def test_se3_act_actInv(self):
        """Test SE3 act() and actInv() for 3D point transform"""
        R = np.eye(3)
        t = np.array([1.0, 2.0, 3.0])
        T = embodik.SE3(R, t)

        p_local = np.array([0.0, 0.0, 0.0])
        p_world = T.act(p_local)
        np.testing.assert_array_almost_equal(p_world, t)

        p_local = np.array([1.0, 0.0, 0.0])
        p_world = T.act(p_local)
        np.testing.assert_array_almost_equal(p_world, np.array([2.0, 2.0, 3.0]))

        # Roundtrip: actInv(act(p)) == p
        p_roundtrip = T.actInv(p_world)
        np.testing.assert_array_almost_equal(p_roundtrip, p_local)

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
        np.testing.assert_allclose(tau_gravity + coriolis + inertial, tau_total, atol=1e-12)

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

    def test_collision_geometry_survives_visual_mesh_failure(self, tmp_path):
        """Collision import should survive unrelated visual mesh failures."""
        urdf_content = """<?xml version="1.0"?>
<robot name="broken_visual_valid_collision">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="missing_visual_mesh.stl" scale="1 1 1"/>
      </geometry>
    </visual>
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
      <origin xyz="0.0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.2"/>
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
        urdf_path = tmp_path / "broken_visual_valid_collision.urdf"
        urdf_path.write_text(urdf_content)

        model = embodik.RobotModel(str(urdf_path), floating_base=False)
        assert model.has_collision_geometry()
        assert len(model.get_collision_geometry_names()) == 2
        assert len(model.get_collision_pair_names()) >= 1

    # =================================================================
    # Joint index access
    # =================================================================

    def test_has_joint(self, urdf_path):
        """Test has_joint() returns True for existing joints."""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        assert model.has_joint("joint1")
        assert model.has_joint("joint2")
        assert not model.has_joint("nonexistent_joint")

    def test_get_joint_id(self, urdf_path):
        """Test get_joint_id() returns valid joint index."""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        joint_id = model.get_joint_id("joint1")
        assert isinstance(joint_id, int)
        assert joint_id >= 0

        joint_id2 = model.get_joint_id("joint2")
        assert isinstance(joint_id2, int)
        assert joint_id2 != joint_id  # Different joints have different IDs

        with pytest.raises(RuntimeError):
            model.get_joint_id("nonexistent_joint")

    def test_get_joint_config_index(self, urdf_path):
        """Test get_joint_config_index() returns correct idx_q."""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        idx_q1 = model.get_joint_config_index("joint1")
        idx_q2 = model.get_joint_config_index("joint2")

        assert isinstance(idx_q1, int)
        assert isinstance(idx_q2, int)
        assert idx_q1 >= 0
        assert idx_q2 >= 0
        # For a 2-joint robot, joint2 should come after joint1
        assert idx_q2 >= idx_q1

        with pytest.raises(RuntimeError):
            model.get_joint_config_index("nonexistent_joint")

    def test_get_joint_config_size(self, urdf_path):
        """Test get_joint_config_size() returns correct nq."""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        nq1 = model.get_joint_config_size("joint1")
        nq2 = model.get_joint_config_size("joint2")

        assert isinstance(nq1, int)
        assert isinstance(nq2, int)
        # Revolute joints typically have nq=1
        assert nq1 == 1
        assert nq2 == 1

        with pytest.raises(RuntimeError):
            model.get_joint_config_size("nonexistent_joint")

    def test_get_joint_velocity_index(self, urdf_path):
        """Test get_joint_velocity_index() returns correct idx_v."""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        idx_v1 = model.get_joint_velocity_index("joint1")
        idx_v2 = model.get_joint_velocity_index("joint2")

        assert isinstance(idx_v1, int)
        assert isinstance(idx_v2, int)
        assert idx_v1 >= 0
        assert idx_v2 >= 0
        # For a 2-joint robot, joint2 should come after joint1
        assert idx_v2 >= idx_v1

        with pytest.raises(RuntimeError):
            model.get_joint_velocity_index("nonexistent_joint")

    def test_get_joint_velocity_size(self, urdf_path):
        """Test get_joint_velocity_size() returns correct nv."""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        nv1 = model.get_joint_velocity_size("joint1")
        nv2 = model.get_joint_velocity_size("joint2")

        assert isinstance(nv1, int)
        assert isinstance(nv2, int)
        # Revolute joints typically have nv=1
        assert nv1 == 1
        assert nv2 == 1

        with pytest.raises(RuntimeError):
            model.get_joint_velocity_size("nonexistent_joint")

    def test_joint_indices_floating_base(self, urdf_path):
        """Test joint indices work correctly for floating-base robot."""
        model = embodik.RobotModel(urdf_path, floating_base=True)

        # Floating base should have nq=7 (xyz + quaternion), nv=6
        # But we can't query "floating_base" as a joint name - it's implicit
        # Test that regular joints still work
        idx_q1 = model.get_joint_config_index("joint1")
        idx_v1 = model.get_joint_velocity_index("joint1")

        # Joint1 should start after floating base (7 config vars, 6 velocity vars)
        assert idx_q1 >= 7  # After floating base config
        assert idx_v1 >= 6  # After floating base velocity

    def test_joint_index_consistency(self, urdf_path):
        """Test that joint indices are consistent with model structure."""
        model = embodik.RobotModel(urdf_path, floating_base=False)

        joint_names = model.get_joint_names()
        assert len(joint_names) == 2

        # Verify indices are sequential and match model.nq/nv
        idx_q_list = [model.get_joint_config_index(name) for name in joint_names]
        idx_v_list = [model.get_joint_velocity_index(name) for name in joint_names]
        nq_list = [model.get_joint_config_size(name) for name in joint_names]
        nv_list = [model.get_joint_velocity_size(name) for name in joint_names]

        # Total nq/nv should match model
        assert sum(nq_list) == model.nq
        assert sum(nv_list) == model.nv

        # Indices should be sequential (no gaps)
        idx_q_sorted = sorted(idx_q_list)
        idx_v_sorted = sorted(idx_v_list)
        for i in range(len(idx_q_sorted) - 1):
            assert (
                idx_q_sorted[i + 1] == idx_q_sorted[i] + nq_list[idx_q_list.index(idx_q_sorted[i])]
            )
        for i in range(len(idx_v_sorted) - 1):
            assert (
                idx_v_sorted[i + 1] == idx_v_sorted[i] + nv_list[idx_v_list.index(idx_v_sorted[i])]
            )


def create_4joint_urdf():
    """Create a 4-joint chain URDF for reduced model tests."""
    urdf_content = """<?xml version="1.0"?>
<robot name="four_joint_robot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
  <link name="link_a">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.25"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <joint name="joint_a" type="revolute">
    <parent link="base_link"/>
    <child link="link_a"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10.0"/>
  </joint>
  <link name="link_b">
    <inertial>
      <mass value="0.8"/>
      <origin xyz="0 0 0.25"/>
      <inertia ixx="0.08" ixy="0" ixz="0" iyy="0.08" iyz="0" izz="0.08"/>
    </inertial>
  </link>
  <joint name="joint_b" type="revolute">
    <parent link="link_a"/>
    <child link="link_b"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="1.5" effort="8.0"/>
  </joint>
  <link name="link_c">
    <inertial>
      <mass value="0.6"/>
      <origin xyz="0 0 0.25"/>
      <inertia ixx="0.06" ixy="0" ixz="0" iyy="0.06" iyz="0" izz="0.06"/>
    </inertial>
  </link>
  <joint name="joint_c" type="revolute">
    <parent link="link_b"/>
    <child link="link_c"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" velocity="1.0" effort="5.0"/>
  </joint>
  <link name="link_d">
    <inertial>
      <mass value="0.4"/>
      <origin xyz="0 0 0.1"/>
      <inertia ixx="0.04" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.04"/>
    </inertial>
  </link>
  <joint name="joint_d" type="revolute">
    <parent link="link_c"/>
    <child link="link_d"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" velocity="1.0" effort="5.0"/>
  </joint>
</robot>"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        f.write(urdf_content)
        return f.name


class TestReducedModel:
    """Tests for RobotModel(urdf_path, actuated_joint_names) constructor."""

    @pytest.fixture
    def urdf_path(self):
        path = create_4joint_urdf()
        yield path
        os.unlink(path)

    def test_all_joints_actuated_matches_full(self, urdf_path):
        """When all joints are listed, reduced model == full model."""
        full = embodik.RobotModel(urdf_path)
        reduced = embodik.RobotModel(
            urdf_path,
            actuated_joint_names=[
                "joint_a",
                "joint_b",
                "joint_c",
                "joint_d",
            ],
        )
        assert reduced.nq == full.nq
        assert reduced.nv == full.nv

    def test_subset_reduces_nq(self, urdf_path):
        """Selecting a subset of joints produces a smaller model."""
        full = embodik.RobotModel(urdf_path)
        assert full.nq == 4

        reduced = embodik.RobotModel(
            urdf_path,
            actuated_joint_names=[
                "joint_a",
                "joint_c",
            ],
        )
        assert reduced.nq == 2
        assert reduced.nv == 2

    def test_single_joint(self, urdf_path):
        """A single actuated joint gives nq=1."""
        reduced = embodik.RobotModel(urdf_path, actuated_joint_names=["joint_b"])
        assert reduced.nq == 1
        assert reduced.nv == 1

    def test_joint_names_match(self, urdf_path):
        """Reduced model only has the requested actuated joints."""
        reduced = embodik.RobotModel(
            urdf_path,
            actuated_joint_names=[
                "joint_a",
                "joint_d",
            ],
        )
        joint_names = reduced.get_joint_names()
        assert "joint_a" in joint_names
        assert "joint_d" in joint_names
        assert "joint_b" not in joint_names
        assert "joint_c" not in joint_names

    def test_frames_preserved(self, urdf_path):
        """Locked joints become fixed frames — links remain accessible."""
        reduced = embodik.RobotModel(urdf_path, actuated_joint_names=["joint_a"])
        assert reduced.has_frame("link_d")
        assert reduced.has_frame("link_b")

    def test_fk_consistent(self, urdf_path):
        """FK at neutral q should match full model at neutral q."""
        full = embodik.RobotModel(urdf_path)
        reduced = embodik.RobotModel(
            urdf_path,
            actuated_joint_names=[
                "joint_a",
                "joint_d",
            ],
        )

        q_full = full.neutral_configuration()
        q_reduced = reduced.neutral_configuration()

        full.update_configuration(q_full)
        reduced.update_configuration(q_reduced)

        full_pose = full.get_frame_pose("link_d")
        reduced_pose = reduced.get_frame_pose("link_d")

        np.testing.assert_allclose(full_pose.translation, reduced_pose.translation, atol=1e-10)
        np.testing.assert_allclose(full_pose.rotation, reduced_pose.rotation, atol=1e-10)

    def test_integrate_works(self, urdf_path):
        """integrate() works on the reduced model."""
        reduced = embodik.RobotModel(
            urdf_path,
            actuated_joint_names=[
                "joint_b",
                "joint_c",
            ],
        )
        q = reduced.neutral_configuration()
        v = np.array([0.1, -0.2])
        dt = 0.01
        q_new = reduced.integrate(q, v, dt)
        np.testing.assert_allclose(q_new, q + v * dt, atol=1e-12)

    def test_jacobian_shape(self, urdf_path):
        """Jacobian has correct shape for the reduced model."""
        reduced = embodik.RobotModel(
            urdf_path,
            actuated_joint_names=[
                "joint_a",
                "joint_c",
            ],
        )
        q = reduced.neutral_configuration()
        reduced.update_configuration(q)
        J = reduced.get_frame_jacobian("link_d")
        assert J.shape == (6, 2)

    def test_gravity_torques(self, urdf_path):
        """Gravity computation works on reduced model."""
        reduced = embodik.RobotModel(
            urdf_path,
            actuated_joint_names=[
                "joint_a",
                "joint_b",
            ],
        )
        q = reduced.neutral_configuration()
        g = reduced.compute_generalized_gravity(q)
        assert g.shape == (reduced.nv,)
        assert np.all(np.isfinite(g))

    def test_invalid_joint_name_raises(self, urdf_path):
        """Requesting a non-existent joint raises RuntimeError."""
        with pytest.raises(RuntimeError, match="not found"):
            embodik.RobotModel(urdf_path, actuated_joint_names=["bogus_joint"])

    def test_floating_base_reduced(self, urdf_path):
        """Reduced model with floating base works correctly."""
        reduced = embodik.RobotModel(
            urdf_path,
            actuated_joint_names=["joint_a", "joint_b"],
            floating_base=True,
        )
        # Floating base adds 7 config vars (xyz + quaternion) and 6 velocity vars
        assert reduced.nq == 7 + 2
        assert reduced.nv == 6 + 2
        assert reduced.is_floating_base

    def test_com_on_reduced(self, urdf_path):
        """CoM computation works on the reduced model."""
        reduced = embodik.RobotModel(
            urdf_path,
            actuated_joint_names=[
                "joint_a",
                "joint_d",
            ],
        )
        q = reduced.neutral_configuration()
        reduced.update_configuration(q)
        com = reduced.get_com_position()
        assert com.shape == (3,)
        assert com[2] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
