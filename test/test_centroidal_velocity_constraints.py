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
  <joint name="support_joint" type="fixed">
    <parent link="base"/>
    <child link="support_frame"/>
    <origin xyz="0.2 -0.1 0.05" rpy="0 0 1.5707963267948966"/>
  </joint>
  <link name="support_frame"/>
  <joint name="high_support_joint" type="fixed">
    <parent link="base"/>
    <child link="high_support_frame"/>
    <origin xyz="0 0 10" rpy="0 0 0"/>
  </joint>
  <link name="high_support_frame"/>
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

    after_explicit = solver.solve_velocity(q, apply_limits=True)
    assert after_explicit.status == eik.SolverStatus.INVALID_INPUT
    npt.assert_allclose(after_explicit.joint_velocities, np.zeros(robot.nv), rtol=0.0, atol=0.0)


def test_velocity_zmp_invalid_explicit_state_clears_and_fails_closed(
    tmp_path: Path,
) -> None:
    robot = eik.RobotModel(str(_write_centroidal_urdf(tmp_path)), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    q = np.array([0.15, -0.2])
    polygon = np.array([[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]])
    solver.add_centroidal_momentum_task("momentum").set_target_momentum(np.ones(6) * 0.01)
    solver.configure_velocity_zmp_constraint(polygon, fz_min=1.0)

    seeded = solver.solve_velocity_with_state(q, np.array([0.1, -0.05]), apply_limits=True)
    assert seeded.status == eik.SolverStatus.SUCCESS

    bad = solver.solve_velocity_with_state(q, np.array([np.nan, 0.0]), apply_limits=True)
    assert bad.status == eik.SolverStatus.NON_FINITE_INPUT
    npt.assert_allclose(bad.joint_velocities, np.zeros(robot.nv), rtol=0.0, atol=0.0)

    regular = solver.solve_velocity(q, apply_limits=True)
    assert regular.status == eik.SolverStatus.INVALID_INPUT
    npt.assert_allclose(regular.joint_velocities, np.zeros(robot.nv), rtol=0.0, atol=0.0)


def test_velocity_zmp_repeated_explicit_calls_are_stationary_and_deterministic(
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

    first = solver.solve_velocity_with_state(q, current_dq, apply_limits=True)
    second = solver.solve_velocity_with_state(q, current_dq, apply_limits=True)
    assert first.status == eik.SolverStatus.SUCCESS
    assert second.status == eik.SolverStatus.SUCCESS
    npt.assert_allclose(second.joint_velocities, first.joint_velocities, rtol=0.0, atol=1e-12)


def test_cleared_centroidal_support_constraints_are_disabled_invariant(
    tmp_path: Path,
) -> None:
    robot = eik.RobotModel(str(_write_centroidal_urdf(tmp_path)), floating_base=False)
    q = np.array([0.1, -0.2])
    target = np.array([0.1, -0.05, 0.02, 0.01, -0.03, 0.04])
    polygon = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

    baseline_solver = eik.KinematicsSolver(robot)
    baseline_solver.add_centroidal_momentum_task("momentum").set_target_momentum(target)
    baseline = baseline_solver.solve_velocity(q, apply_limits=True)

    constrained_solver = eik.KinematicsSolver(robot)
    constrained_solver.add_centroidal_momentum_task("momentum").set_target_momentum(target)
    constrained_solver.configure_capture_point_constraint(polygon, frame_name="high_support_frame")
    constrained_solver.clear_capture_point_constraint()
    constrained_solver.configure_velocity_zmp_constraint(polygon, fz_min=1.0)
    constrained_solver.clear_velocity_zmp_constraint()
    cleared = constrained_solver.solve_velocity(q, apply_limits=True)

    assert cleared.status == baseline.status
    npt.assert_allclose(cleared.joint_velocities, baseline.joint_velocities, rtol=0.0, atol=1e-12)


def test_capture_point_derives_omega_from_support_frame_com_height(
    tmp_path: Path,
) -> None:
    robot = eik.RobotModel(str(_write_centroidal_urdf(tmp_path)), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    q = np.array([0.25, -0.4])
    dq = np.array([0.3, -0.2])
    robot.update_configuration(q)
    polygon = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])
    solver.configure_capture_point_constraint(polygon, frame_name="support_frame")

    pose = robot.get_frame_pose("support_frame")
    rotation = np.asarray(pose.rotation)
    translation = np.asarray(pose.translation)
    com_local = rotation.T @ (robot.get_com_position() - translation)
    jcom_local = rotation.T @ robot.get_com_jacobian()
    omega = np.sqrt(9.81 / com_local[2])

    debug = solver.evaluate_capture_point_constraint(q, dq)
    assert debug["status"] == eik.SolverStatus.SUCCESS
    npt.assert_allclose(
        debug["point"], com_local[:2] + (jcom_local[:2, :] @ dq) / omega, rtol=0.0, atol=1e-12
    )


def test_capture_point_rejects_nonpositive_derived_height(tmp_path: Path) -> None:
    robot = eik.RobotModel(str(_write_centroidal_urdf(tmp_path)), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    q = np.array([0.0, 0.0])
    dq = np.array([0.0, 0.0])
    polygon = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])
    solver.configure_capture_point_constraint(polygon, frame_name="high_support_frame")

    debug = solver.evaluate_capture_point_constraint(q, dq)
    assert debug["status"] == eik.SolverStatus.INVALID_INPUT


def test_local_support_frame_zmp_formula_and_debug_evaluators_do_not_mutate(
    tmp_path: Path,
) -> None:
    robot = eik.RobotModel(str(_write_centroidal_urdf(tmp_path)), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    q = np.array([0.25, -0.35])
    current_dq = np.array([0.18, -0.07])
    command = np.array([0.22, -0.03])
    held_q = np.array([-0.1, 0.2])
    held_v = np.array([0.4, -0.2])
    robot.update_kinematics(held_q, held_v)
    polygon = np.array([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]])
    solver.configure_velocity_zmp_constraint(polygon, frame_name="support_frame", fz_min=0.1)

    debug = solver.evaluate_velocity_zmp_constraint(q, current_dq, command)
    npt.assert_allclose(robot.get_current_configuration(), held_q, rtol=0.0, atol=0.0)
    npt.assert_allclose(robot.get_current_velocity(), held_v, rtol=0.0, atol=0.0)

    robot.update_kinematics(q, current_dq)
    pose = robot.get_frame_pose("support_frame")
    rotation = np.asarray(pose.rotation)
    translation = np.asarray(pose.translation)
    com_local = rotation.T @ (robot.get_com_position() - translation)
    ag = robot.compute_centroidal_momentum_matrix(q, current_dq)
    bias = robot.compute_centroidal_momentum_matrix_bias(q, current_dq)
    hdot_world = ag @ ((command - current_dq) / solver.dt) + bias
    force_world = hdot_world[:3] + np.array([0.0, 0.0, robot.get_total_mass() * 9.81])
    force_local = rotation.T @ force_world
    moment_local = rotation.T @ hdot_world[3:]
    expected_zmp = np.array(
        [
            com_local[0] - (moment_local[1] + com_local[2] * force_local[0]) / force_local[2],
            com_local[1] + (moment_local[0] - com_local[2] * force_local[1]) / force_local[2],
        ]
    )

    assert debug["status"] == eik.SolverStatus.SUCCESS
    npt.assert_allclose(debug["point"], expected_zmp, rtol=0.0, atol=1e-10)

    robot.update_kinematics(held_q, held_v)
    solver.evaluate_capture_point_constraint(q, command)
    npt.assert_allclose(robot.get_current_configuration(), held_q, rtol=0.0, atol=0.0)
    npt.assert_allclose(robot.get_current_velocity(), held_v, rtol=0.0, atol=0.0)


def test_moving_support_frames_fail_closed_at_configuration(tmp_path: Path) -> None:
    robot = eik.RobotModel(str(_write_centroidal_urdf(tmp_path)), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    polygon = np.array([[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]])
    with np.testing.assert_raises(ValueError):
        solver.configure_capture_point_constraint(polygon, frame_name="link1")
    with np.testing.assert_raises(ValueError):
        solver.configure_velocity_zmp_constraint(polygon, frame_name="link1", fz_min=0.1)


def test_position_step_fail_closed_when_support_contract_is_invalid(
    tmp_path: Path,
) -> None:
    robot = eik.RobotModel(str(_write_centroidal_urdf(tmp_path)), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    q = np.array([0.0, 0.0])
    robot.update_configuration(q)
    task = solver.add_frame_task("tip", "link2", eik.TaskType.FRAME_POSE)
    pose = robot.get_frame_pose("link2")
    target = np.eye(4)
    target[:3, :3] = np.asarray(pose.rotation)
    target[:3, 3] = np.asarray(pose.translation) + np.array([0.02, 0.0, 0.0])
    solver.configure_capture_point_constraint(
        np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]),
        frame_name="high_support_frame",
    )

    result = solver.solve_position_step(q, target, "tip")
    assert result.status != eik.SolverStatus.SUCCESS
    npt.assert_allclose(result.q_solution, q, rtol=0.0, atol=0.0)
    npt.assert_allclose(result.joint_velocities, np.zeros(robot.nv), rtol=0.0, atol=0.0)


def test_final_acceptance_revalidates_centroidal_bounds_and_zmp(tmp_path: Path) -> None:
    robot = eik.RobotModel(str(_write_centroidal_urdf(tmp_path)), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    q = np.array([0.2, -0.3])
    current_dq = np.array([0.1, -0.05])
    task = solver.add_centroidal_momentum_task("momentum")
    task.set_target_momentum(np.ones(6) * 100.0)
    solver.configure_centroidal_momentum_bounds(lower_h=np.ones(6) * 0.2, upper_h=np.ones(6) * 0.3)
    solver.configure_velocity_zmp_constraint(
        np.array([[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]]),
        fz_min=0.1,
    )

    result = solver.solve_velocity_with_state(q, current_dq, apply_limits=True)
    assert result.status != eik.SolverStatus.SUCCESS
    npt.assert_allclose(result.joint_velocities, np.zeros(robot.nv), rtol=0.0, atol=0.0)
