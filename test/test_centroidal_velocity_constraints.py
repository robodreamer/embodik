"""Centroidal momentum velocity IK behavior.

These tests exercise the public Python API and compare against RobotModel's
centroidal quantities so row ordering, masks, and exclusions stay observable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt

import embodik as eik


def _write_centroidal_urdf(tmp_path: Path) -> Path:
    path = tmp_path / "centroidal_velocity_robot.urdf"
    path.write_text(
        """<?xml version="1.0"?>
<robot name="centroidal_velocity_robot">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.03" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.05"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0.1 0.0 0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="20" velocity="5"/>
  </joint>
  <link name="link1">
    <inertial>
      <origin xyz="0.35 0.05 0.0" rpy="0 0 0"/>
      <mass value="1.3"/>
      <inertia ixx="0.02" ixy="0.001" ixz="0" iyy="0.025" iyz="0" izz="0.03"/>
    </inertial>
  </link>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0.45 0.0 0.0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" effort="20" velocity="5"/>
  </joint>
  <link name="link2">
    <inertial>
      <origin xyz="0.2 -0.08 0.1" rpy="0 0 0"/>
      <mass value="0.9"/>
      <inertia ixx="0.012" ixy="0" ixz="0.001" iyy="0.017" iyz="0" izz="0.021"/>
    </inertial>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    return path


def test_centroidal_momentum_task_tracks_selected_absolute_momentum_and_exclusions(
    tmp_path: Path,
) -> None:
    robot = eik.RobotModel(str(_write_centroidal_urdf(tmp_path)), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    q = np.array([0.35, -0.45])
    robot.update_configuration(q)

    task = solver.add_centroidal_momentum_task("momentum")
    assert task.get_type() == eik.TaskType.CENTROIDAL_MOMENTUM
    task.set_target_momentum(np.array([0.4, -0.2, 0.1, 0.05, 0.0, -0.03]))
    task.set_axis_mask(np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]))
    task.set_excluded_joint_indices([1])
    task.update(robot)

    ag = robot.get_centroidal_momentum_matrix()
    expected_jacobian = ag[[0, 2, 4], :].copy()
    expected_jacobian[:, 1] = 0.0
    npt.assert_allclose(task.get_jacobian(), expected_jacobian, rtol=0.0, atol=1e-12)
    npt.assert_allclose(task.get_error(), np.array([0.4, 0.1, 0.0]), rtol=0.0, atol=1e-12)
    npt.assert_allclose(task.get_velocity(), np.array([0.4, 0.1, 0.0]), rtol=0.0, atol=1e-12)


def test_hard_centroidal_momentum_bounds_ignore_task_local_exclusions(
    tmp_path: Path,
) -> None:
    robot = eik.RobotModel(str(_write_centroidal_urdf(tmp_path)), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    q = np.array([0.2, -0.3])
    robot.update_configuration(q)
    ag = robot.get_centroidal_momentum_matrix()

    row = int(np.argmax(np.linalg.norm(ag, axis=1)))
    target = np.zeros(6)
    target[row] = 10.0
    mask = np.zeros(6)
    mask[row] = 1.0

    task = solver.add_centroidal_momentum_task("momentum")
    task.set_target_momentum(target)
    task.set_axis_mask(mask)
    task.set_excluded_joint_indices([0, 1])
    solver.configure_centroidal_momentum_bounds(
        lower_h=np.array([-0.05]), upper_h=np.array([0.05]), axis_mask=mask
    )

    readback = solver.get_centroidal_momentum_bounds()
    npt.assert_allclose(readback[0], np.array([-0.05]), rtol=0.0, atol=1e-12)
    npt.assert_allclose(readback[1], np.array([0.05]), rtol=0.0, atol=1e-12)
    npt.assert_allclose(readback[2], mask, rtol=0.0, atol=1e-12)

    result = solver.solve_velocity(q, apply_limits=True)
    h_command = ag @ result.joint_velocities
    assert h_command[row] <= 0.05 + 1e-8
    assert h_command[row] >= -0.05 - 1e-8


def test_capture_point_debug_uses_commanded_com_velocity_formula(tmp_path: Path) -> None:
    robot = eik.RobotModel(str(_write_centroidal_urdf(tmp_path)), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    q = np.array([0.25, -0.4])
    dq = np.array([0.3, -0.2])
    robot.update_configuration(q)
    omega = 3.0
    polygon = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])
    solver.configure_capture_point_constraint(polygon, omega=omega)

    debug = solver.evaluate_capture_point_constraint(q, dq)
    expected_xi = robot.get_com_position()[:2] + (robot.get_com_jacobian()[:2, :] @ dq) / omega
    assert debug["status"] == eik.SolverStatus.SUCCESS
    npt.assert_allclose(debug["point"], expected_xi, rtol=0.0, atol=1e-12)
    assert np.min(debug["slacks"]) > 0.0


def test_velocity_zmp_requires_explicit_current_dq_and_matches_explicit_state(
    tmp_path: Path,
) -> None:
    robot = eik.RobotModel(str(_write_centroidal_urdf(tmp_path)), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    q = np.array([0.15, -0.2])
    current_dq = np.array([0.1, -0.05])
    polygon = np.array([[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]])
    task = solver.add_centroidal_momentum_task("momentum")
    task.set_target_momentum(np.array([0.2, -0.1, 0.05, 0.03, -0.02, 0.01]))
    solver.configure_velocity_zmp_constraint(polygon, fz_min=1.0)

    regular = solver.solve_velocity(q, apply_limits=True)
    assert regular.status == eik.SolverStatus.INVALID_INPUT
    npt.assert_allclose(regular.joint_velocities, np.zeros(robot.nv), rtol=0.0, atol=0.0)

    explicit = solver.solve_velocity_with_state(q, current_dq, apply_limits=True)
    assert explicit.status == eik.SolverStatus.SUCCESS
    debug = solver.evaluate_velocity_zmp_constraint(q, current_dq, explicit.joint_velocities)
    assert debug["status"] == eik.SolverStatus.SUCCESS
    assert debug["force_z"] >= 1.0 - 1e-8
    assert np.min(debug["slacks"]) >= -1e-8
