from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import embodik as eik


def _write_centroidal_urdf(tmp_path: Path) -> Path:
    path = tmp_path / "centroidal_binding_robot.urdf"
    path.write_text(
        """<?xml version="1.0"?>
<robot name="centroidal_binding_robot">
  <link name="base_link">
    <inertial>
      <origin xyz="0.07 -0.03 0.11" rpy="0.05 -0.02 0.04"/>
      <mass value="3.25"/>
      <inertia ixx="0.31" ixy="0.012" ixz="-0.017" iyy="0.43" iyz="0.023" izz="0.52"/>
    </inertial>
  </link>
  <link name="shoulder">
    <inertial>
      <origin xyz="0.34 0.06 -0.02" rpy="-0.03 0.04 0.07"/>
      <mass value="1.75"/>
      <inertia ixx="0.08" ixy="-0.006" ixz="0.004" iyy="0.11" iyz="-0.003" izz="0.13"/>
    </inertial>
  </link>
  <joint name="yaw" type="revolute">
    <parent link="base_link"/>
    <child link="shoulder"/>
    <origin xyz="0.13 -0.08 0.22" rpy="0.01 0.03 -0.02"/>
    <axis xyz="0.2 0.1 0.97"/>
    <limit lower="-2.4" upper="2.3" velocity="7.0" effort="40.0"/>
  </joint>
  <link name="forearm">
    <inertial>
      <origin xyz="-0.08 0.27 0.05" rpy="0.09 -0.06 0.02"/>
      <mass value="0.95"/>
      <inertia ixx="0.044" ixy="0.005" ixz="-0.002" iyy="0.061" iyz="0.007" izz="0.073"/>
    </inertial>
  </link>
  <joint name="slide" type="prismatic">
    <parent link="shoulder"/>
    <child link="forearm"/>
    <origin xyz="0.41 0.18 -0.09" rpy="-0.04 0.08 0.03"/>
    <axis xyz="-0.1 0.95 0.2"/>
    <limit lower="-0.35" upper="0.45" velocity="3.0" effort="35.0"/>
  </joint>
  <link name="tool">
    <inertial>
      <origin xyz="0.12 -0.16 0.19" rpy="-0.02 0.11 -0.05"/>
      <mass value="0.55"/>
      <inertia ixx="0.019" ixy="-0.002" ixz="0.001" iyy="0.026" iyz="-0.003" izz="0.031"/>
    </inertial>
  </link>
  <joint name="pitch" type="revolute">
    <parent link="forearm"/>
    <child link="tool"/>
    <origin xyz="-0.23 0.36 0.14" rpy="0.06 -0.03 0.05"/>
    <axis xyz="0.25 -0.4 0.88"/>
    <limit lower="-1.7" upper="1.8" velocity="5.0" effort="20.0"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    return path


def test_centroidal_methods_are_exported_and_consistent(tmp_path: Path) -> None:
    robot = eik.RobotModel(str(_write_centroidal_urdf(tmp_path)), floating_base=False)
    q = np.array([0.37, -0.11, -0.42])
    v = np.array([-0.23, 0.31, 0.17])
    a = np.array([0.41, -0.19, 0.29])

    robot.update_kinematics(q, v)

    assert robot.get_total_mass() == pytest.approx(6.5)
    ag = robot.get_centroidal_momentum_matrix()
    h = robot.get_centroidal_momentum()
    dag = robot.get_centroidal_momentum_matrix_time_variation()
    bias = robot.get_centroidal_momentum_matrix_bias()

    assert ag.shape == (6, robot.nv)
    assert h.shape == (6,)
    assert dag.shape == (6, robot.nv)
    assert bias.shape == (6,)
    np.testing.assert_allclose(h, ag @ v, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(bias, dag @ v, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        robot.compute_centroidal_momentum_matrix(q),
        robot.compute_centroidal_momentum_matrix(q, np.zeros(robot.nv)),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(robot.compute_centroidal_momentum(q, v), h, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        robot.compute_centroidal_momentum_matrix_time_variation(q, v),
        dag,
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        robot.compute_centroidal_momentum_matrix_bias(q, v),
        bias,
        rtol=0.0,
        atol=1e-10,
    )
    assert np.all(np.isfinite(ag @ a + bias))


def test_centroidal_bindings_reject_wrong_size_and_nonfinite(tmp_path: Path) -> None:
    robot = eik.RobotModel(str(_write_centroidal_urdf(tmp_path)), floating_base=False)
    q = np.array([0.37, -0.11, -0.42])
    v = np.array([-0.23, 0.31, 0.17])

    with pytest.raises(ValueError, match="Configuration vector size mismatch"):
        robot.compute_centroidal_momentum_matrix(q[:2])
    with pytest.raises(ValueError, match="Velocity vector size mismatch"):
        robot.compute_centroidal_momentum(q, v[:2])
    with pytest.raises(ValueError, match="finite"):
        robot.compute_centroidal_momentum_matrix(np.array([0.0, math.nan, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        robot.compute_centroidal_momentum_matrix_bias(q, np.array([0.0, math.inf, 0.0]))
