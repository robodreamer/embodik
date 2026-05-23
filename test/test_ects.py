#!/usr/bin/env python3
"""Tests for ECTS (Extended Cooperative Task Space) math utilities"""

import os
import tempfile

import numpy as np
import pytest

import embodik


def create_dual_arm_urdf():
    """Create a dual-arm URDF: base -> {left 3DOF chain, right 3DOF chain}"""
    urdf_content = """<?xml version="1.0"?>
<robot name="dual_arm_robot">
  <link name="base_link">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>

  <!-- Left arm -->
  <joint name="left_j1" type="revolute">
    <parent link="base_link"/>
    <child link="left_link1"/>
    <origin xyz="0 0.2 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
  <link name="left_link1">
    <inertial><mass value="0.5"/><origin xyz="0 0 0.1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
  </link>

  <joint name="left_j2" type="revolute">
    <parent link="left_link1"/>
    <child link="left_link2"/>
    <origin xyz="0 0 0.2"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
  <link name="left_link2">
    <inertial><mass value="0.3"/><origin xyz="0 0 0.1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
  </link>

  <joint name="left_j3" type="revolute">
    <parent link="left_link2"/>
    <child link="left_link3"/>
    <origin xyz="0 0 0.2"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
  <link name="left_link3">
    <inertial><mass value="0.2"/><origin xyz="0 0 0.05"/><inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/></inertial>
  </link>

  <joint name="left_ee_joint" type="fixed">
    <parent link="left_link3"/>
    <child link="left_ee"/>
    <origin xyz="0 0 0.15"/>
  </joint>
  <link name="left_ee">
    <inertial><mass value="0.05"/><origin xyz="0 0 0"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
  </link>

  <!-- Right arm -->
  <joint name="right_j1" type="revolute">
    <parent link="base_link"/>
    <child link="right_link1"/>
    <origin xyz="0 -0.2 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
  <link name="right_link1">
    <inertial><mass value="0.5"/><origin xyz="0 0 0.1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
  </link>

  <joint name="right_j2" type="revolute">
    <parent link="right_link1"/>
    <child link="right_link2"/>
    <origin xyz="0 0 0.2"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
  <link name="right_link2">
    <inertial><mass value="0.3"/><origin xyz="0 0 0.1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
  </link>

  <joint name="right_j3" type="revolute">
    <parent link="right_link2"/>
    <child link="right_link3"/>
    <origin xyz="0 0 0.2"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
  <link name="right_link3">
    <inertial><mass value="0.2"/><origin xyz="0 0 0.05"/><inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/></inertial>
  </link>

  <joint name="right_ee_joint" type="fixed">
    <parent link="right_link3"/>
    <child link="right_ee"/>
    <origin xyz="0 0 0.15"/>
  </joint>
  <link name="right_ee">
    <inertial><mass value="0.05"/><origin xyz="0 0 0"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
  </link>
</robot>"""
    fd, path = tempfile.mkstemp(suffix=".urdf")
    with os.fdopen(fd, "w") as f:
        f.write(urdf_content)
    return path


@pytest.fixture
def dual_arm():
    path = create_dual_arm_urdf()
    robot = embodik.RobotModel(path)
    q = np.array([0.3, 0.5, 0.1, -0.3, 0.5, -0.1])
    robot.update_configuration(q)
    yield robot, q
    os.unlink(path)


class TestMapEctsMode:
    def test_orthogonal(self):
        cfg = embodik.map_ects_mode("orthogonal")
        assert cfg.alpha == pytest.approx(0.5)
        assert cfg.coordinated is False

    def test_serial_left(self):
        cfg = embodik.map_ects_mode("serial_left")
        assert cfg.alpha == pytest.approx(1.0)
        assert cfg.coordinated is True

    def test_serial_right(self):
        cfg = embodik.map_ects_mode("serial_right")
        assert cfg.alpha == pytest.approx(0.0)
        assert cfg.coordinated is True

    def test_parallel(self):
        cfg = embodik.map_ects_mode("parallel")
        assert cfg.alpha == pytest.approx(0.5)
        assert cfg.coordinated is True

    def test_blended(self):
        cfg = embodik.map_ects_mode("blended")
        assert cfg.alpha == pytest.approx(0.5)
        assert cfg.coordinated is True

    def test_blended_custom(self):
        cfg = embodik.map_ects_mode_blended(0.75)
        assert cfg.alpha == pytest.approx(0.75)
        assert cfg.coordinated is True

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            embodik.map_ects_mode("invalid_mode")


class TestAbsoluteFrame:
    def test_midpoint_alpha_half(self, dual_arm):
        robot, q = dual_arm
        solver = embodik.KinematicsSolver(robot)
        abs_task = solver.add_absolute_frame_task("abs", "left_ee", "right_ee", 0.5)
        abs_task.update(robot)

        left_pose = robot.get_frame_pose("left_ee")
        right_pose = robot.get_frame_pose("right_ee")

        expected_pos = 0.5 * left_pose.translation + 0.5 * right_pose.translation
        np.testing.assert_allclose(abs_task.current_position, expected_pos, atol=1e-10)

    def test_alpha_one_equals_frame_a(self, dual_arm):
        robot, q = dual_arm
        solver = embodik.KinematicsSolver(robot)
        abs_task = solver.add_absolute_frame_task("abs", "left_ee", "right_ee", 1.0)
        abs_task.update(robot)

        left_pose = robot.get_frame_pose("left_ee")
        np.testing.assert_allclose(abs_task.current_position, left_pose.translation, atol=1e-10)

    def test_alpha_zero_equals_frame_b(self, dual_arm):
        robot, q = dual_arm
        solver = embodik.KinematicsSolver(robot)
        abs_task = solver.add_absolute_frame_task("abs", "left_ee", "right_ee", 0.0)
        abs_task.update(robot)

        right_pose = robot.get_frame_pose("right_ee")
        np.testing.assert_allclose(abs_task.current_position, right_pose.translation, atol=1e-10)


class TestRelativeFrame:
    def test_identity_at_same_config(self):
        """When both frames are at a known relative pose, verify it matches FK"""
        path = create_dual_arm_urdf()
        robot = embodik.RobotModel(path)
        q = np.zeros(6)
        robot.update_configuration(q)

        solver = embodik.KinematicsSolver(robot)
        rel_task = solver.add_relative_frame_task("rel", "left_ee", "right_ee")
        rel_task.update(robot)

        left_pose = robot.get_frame_pose("left_ee")
        right_pose = robot.get_frame_pose("right_ee")
        T_left = left_pose.homogeneous()
        T_right = right_pose.homogeneous()
        T_rel_expected = np.linalg.inv(T_left) @ T_right

        np.testing.assert_allclose(rel_task.current_position, T_rel_expected[:3, 3], atol=1e-10)
        np.testing.assert_allclose(rel_task.current_orientation, T_rel_expected[:3, :3], atol=1e-10)
        os.unlink(path)

    def test_capture_and_error_zero(self, dual_arm):
        robot, q = dual_arm
        solver = embodik.KinematicsSolver(robot)
        rel_task = solver.add_relative_frame_task("rel", "left_ee", "right_ee")
        rel_task.update(robot)
        rel_task.capture_current_as_target()

        error = rel_task.get_error()
        np.testing.assert_allclose(error, np.zeros(6), atol=1e-10)


class TestJacobianNumerical:
    """Verify ECTS Jacobians by comparing with finite-difference approximations"""

    def test_absolute_jacobian_numerical(self, dual_arm):
        robot, q = dual_arm
        solver = embodik.KinematicsSolver(robot)
        abs_task = solver.add_absolute_frame_task("abs", "left_ee", "right_ee", 0.5)
        abs_task.update(robot)
        J_abs = abs_task.get_jacobian()

        eps = 1e-7
        nv = robot.nv
        J_num = np.zeros((3, nv))

        for i in range(nv):
            q_plus = q.copy()
            q_plus[i] += eps
            robot.update_configuration(q_plus)
            left_plus = robot.get_frame_pose("left_ee").translation
            right_plus = robot.get_frame_pose("right_ee").translation
            pos_plus = 0.5 * left_plus + 0.5 * right_plus

            q_minus = q.copy()
            q_minus[i] -= eps
            robot.update_configuration(q_minus)
            left_minus = robot.get_frame_pose("left_ee").translation
            right_minus = robot.get_frame_pose("right_ee").translation
            pos_minus = 0.5 * left_minus + 0.5 * right_minus

            J_num[:, i] = (pos_plus - pos_minus) / (2 * eps)

        robot.update_configuration(q)
        np.testing.assert_allclose(J_abs[:3, :], J_num, atol=1e-4)

    def test_relative_position_jacobian_numerical(self, dual_arm):
        robot, q = dual_arm
        solver = embodik.KinematicsSolver(robot)
        rel_task = solver.add_relative_frame_task("rel", "left_ee", "right_ee")
        rel_task.update(robot)
        J_rel = rel_task.get_jacobian()

        eps = 1e-7
        nv = robot.nv
        J_num = np.zeros((3, nv))

        def get_rel_pos(q_val):
            robot.update_configuration(q_val)
            T_l = robot.get_frame_pose("left_ee").homogeneous()
            T_r = robot.get_frame_pose("right_ee").homogeneous()
            T_rel = np.linalg.inv(T_l) @ T_r
            return T_rel[:3, 3]

        for i in range(nv):
            q_plus = q.copy()
            q_plus[i] += eps
            pos_plus = get_rel_pos(q_plus)

            q_minus = q.copy()
            q_minus[i] -= eps
            pos_minus = get_rel_pos(q_minus)

            J_num[:, i] = (pos_plus - pos_minus) / (2 * eps)

        robot.update_configuration(q)
        np.testing.assert_allclose(J_rel[:3, :], J_num, atol=1e-4)


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


class TestComputeAbsoluteFrame:
    def test_pure_translation_midpoint(self):
        identity = np.eye(3)
        T_1 = embodik.SE3(rotation=identity, translation=np.array([1.0, 0.0, 0.0]))
        T_2 = embodik.SE3(rotation=identity, translation=np.array([3.0, 0.0, 0.0]))

        T_abs = embodik.compute_absolute_frame(T_1, T_2, 0.5)

        np.testing.assert_allclose(T_abs.translation, np.array([2.0, 0.0, 0.0]), atol=1e-12)
        np.testing.assert_allclose(T_abs.rotation, identity, atol=1e-12)

    def test_slerp_rotation_midpoint(self):
        identity = np.eye(3)
        zero = np.zeros(3)
        T_1 = embodik.SE3(rotation=identity, translation=zero)
        T_2 = embodik.SE3(rotation=_rot_z(np.pi / 2.0), translation=zero)

        T_abs = embodik.compute_absolute_frame(T_1, T_2, 0.5)

        np.testing.assert_allclose(T_abs.rotation, _rot_z(np.pi / 4.0), atol=1e-9)
        np.testing.assert_allclose(T_abs.translation, zero, atol=1e-12)

    def test_alpha_one_returns_first_frame(self):
        T_1 = embodik.SE3(rotation=_rot_z(0.7), translation=np.array([0.1, 0.2, 0.3]))
        T_2 = embodik.SE3(rotation=_rot_y(0.4), translation=np.array([1.1, -0.4, 0.5]))

        T_abs = embodik.compute_absolute_frame(T_1, T_2, 1.0)

        np.testing.assert_allclose(T_abs.translation, T_1.translation, atol=1e-12)
        np.testing.assert_allclose(T_abs.rotation, T_1.rotation, atol=1e-12)

    def test_alpha_zero_returns_second_frame(self):
        T_1 = embodik.SE3(rotation=_rot_z(0.7), translation=np.array([0.1, 0.2, 0.3]))
        T_2 = embodik.SE3(rotation=_rot_y(0.4), translation=np.array([1.1, -0.4, 0.5]))

        T_abs = embodik.compute_absolute_frame(T_1, T_2, 0.0)

        np.testing.assert_allclose(T_abs.translation, T_2.translation, atol=1e-12)
        np.testing.assert_allclose(T_abs.rotation, T_2.rotation, atol=1e-12)

    def test_slerp_midpoint_is_not_endpoint_shortcut(self):
        identity = np.eye(3)
        T_1 = embodik.SE3(rotation=identity, translation=np.zeros(3))
        T_2 = embodik.SE3(rotation=_rot_y(np.pi / 2.0), translation=np.zeros(3))

        T_abs = embodik.compute_absolute_frame(T_1, T_2, 0.5)

        assert not np.allclose(T_abs.rotation, T_1.rotation, atol=1e-6)
        assert not np.allclose(T_abs.rotation, T_2.rotation, atol=1e-6)


class TestComputeRelativeFrame:
    def test_relative_frame_round_trip(self):
        T_1 = embodik.SE3(
            rotation=_rot_z(0.3) @ _rot_y(-0.2),
            translation=np.array([0.2, -0.1, 0.4]),
        )
        T_2 = embodik.SE3(
            rotation=_rot_x(0.5) @ _rot_z(-0.4),
            translation=np.array([-0.3, 0.6, 0.05]),
        )

        T_rel = embodik.compute_relative_frame(T_1, T_2)
        T_reconstructed = T_1 * T_rel

        np.testing.assert_allclose(T_reconstructed.homogeneous(), T_2.homogeneous(), atol=1e-9)
