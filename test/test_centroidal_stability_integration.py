from __future__ import annotations

from pathlib import Path

import numpy as np

import embodik


def _write_model(tmp_path: Path) -> Path:
    path = tmp_path / "centroidal_stability_integration.urdf"
    path.write_text(
        """<?xml version="1.0"?>
<robot name="centroidal_stability_integration">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.03" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.05"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0.1 0 0.2" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="200" velocity="20"/>
  </joint>
  <link name="link1">
    <inertial>
      <origin xyz="0.35 0.05 0.1" rpy="0 0 0"/>
      <mass value="1.3"/>
      <inertia ixx="0.02" ixy="0.001" ixz="0" iyy="0.025" iyz="0" izz="0.03"/>
    </inertial>
  </link>
  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0.45 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" effort="200" velocity="20"/>
  </joint>
  <link name="link2">
    <inertial>
      <origin xyz="0.2 -0.08 0.12" rpy="0 0 0"/>
      <mass value="0.9"/>
      <inertia ixx="0.012" ixy="0" ixz="0.001" iyy="0.017" iyz="0" izz="0.021"/>
    </inertial>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    return path


def _support_polygon(robot: embodik.RobotModel) -> np.ndarray:
    x, y = robot.get_com_position()[:2]
    return np.array(
        [[x - 0.75, y - 0.75], [x + 0.75, y - 0.75], [x + 0.75, y + 0.75], [x - 0.75, y + 0.75]]
    )


def test_position_step_centroidal_constraints_compose_with_explicit_state(
    tmp_path: Path,
) -> None:
    robot = embodik.RobotModel(str(_write_model(tmp_path)), floating_base=False)
    robot.set_gravity(np.array([0.0, 0.0, -9.81]))
    q = np.array([0.25, -0.35])
    current_dq = np.zeros(robot.nv)
    robot.update_kinematics(q, current_dq)
    polygon = _support_polygon(robot)

    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    pose = robot.get_frame_pose("link2")
    target = np.eye(4)
    target[:3, :3] = np.asarray(pose.rotation)
    target[:3, 3] = np.asarray(pose.translation)
    solver.add_frame_task("tip", "link2", embodik.TaskType.FRAME_POSE)
    momentum = solver.add_centroidal_momentum_task("momentum")
    momentum.set_target_momentum(robot.compute_centroidal_momentum(q, current_dq))
    solver.configure_centroidal_momentum_bounds(np.full(6, -5.0), np.full(6, 5.0))
    solver.configure_capture_point_constraint(polygon, omega=3.0)
    solver.configure_velocity_zmp_constraint(polygon, fz_min=1.0)

    options = embodik.PositionStepOptions()
    options.current_joint_velocity = current_dq
    result = solver.solve_position_step(q, target, "tip", options)
    assert result.status is embodik.SolverStatus.SUCCESS
    cp = solver.evaluate_capture_point_constraint(q, result.joint_velocities)
    zmp = solver.evaluate_velocity_zmp_constraint(q, current_dq, result.joint_velocities)
    assert np.min(cp["slacks"]) >= -1e-8
    assert np.min(zmp["slacks"]) >= -1e-8
    assert zmp["force_z"] >= 1.0


def test_acceleration_centroidal_constraints_compose_at_accepted_state(tmp_path: Path) -> None:
    robot = embodik.RobotModel(str(_write_model(tmp_path)), floating_base=False)
    robot.set_gravity(np.array([0.0, 0.0, -9.81]))
    q = np.array([0.25, -0.35])
    dq = np.zeros(robot.nv)
    robot.update_kinematics(q, dq)
    polygon = _support_polygon(robot)

    solver = embodik.AccelerationSolver(robot)
    support = embodik.ComSupportPolygonConstraintDefinition()
    support.support_polygon = polygon
    support.frame_name = "world"

    objective = embodik.CentroidalMomentumRateObjective()
    objective.source_id = "stationary_momentum"
    objective.h_target = robot.compute_centroidal_momentum(q, dq)
    objective.hdot_feedforward = np.zeros(6)
    objective.proportional_gain = 2.0

    bounds = embodik.CentroidalMomentumRateBounds()
    bounds.source_id = "momentum_rate_envelope"
    bounds.lower_bounds = np.full(6, -50.0)
    bounds.upper_bounds = np.full(6, 50.0)

    capture_point = embodik.CapturePointAccelerationConstraint()
    capture_point.source_id = "predicted_capture_point"
    capture_point.definition = support
    capture_point.omega = 3.0

    zmp = embodik.ZmpAccelerationConstraint()
    zmp.source_id = "physical_zmp"
    zmp.definition = support
    zmp.fz_min = 1.0

    options = embodik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.full(robot.nv, 50.0)
    options.apply_position_limits = False
    options.apply_velocity_limits = False
    options.centroidal_momentum_rate_objectives = [objective]
    options.centroidal_momentum_rate_bounds = [bounds]
    options.capture_point_constraints = [capture_point]
    options.zmp_constraints = [zmp]

    result = solver.solve(q, dq, 0.01, options)
    assert result.status is embodik.SolverStatus.SUCCESS
    assert result.capture_point_diagnostics[0].min_slack >= -1e-8
    assert result.zmp_diagnostics[0].min_slack >= -1e-8
    assert result.zmp_diagnostics[0].force_z >= 1.0
    assert embodik.AccelerationSolver.capabilities().supports_dynamic_balance is False
