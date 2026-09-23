import pathlib

import numpy as np

import embodik as eik


def _write_urdf(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "acceleration_dynamic_support_exports.urdf"
    path.write_text("""<?xml version="1.0"?>
<robot name="acceleration_dynamic_support_exports_robot">
  <link name="base_link">
    <inertial>
      <origin xyz="0.0 0.0 0.1" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.2"/>
    </inertial>
  </link>
  <link name="link1">
    <inertial>
      <origin xyz="0.5 0.0 0.2" rpy="0 0 0"/>
      <mass value="1.5"/>
      <inertia ixx="0.08" ixy="0" ixz="0" iyy="0.12" iyz="0" izz="0.10"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="500.0" effort="1000.0"/>
  </joint>
  <link name="tip">
    <inertial>
      <origin xyz="0.35 0.05 0.1" rpy="0 0 0"/>
      <mass value="0.8"/>
      <inertia ixx="0.04" ixy="0" ixz="0" iyy="0.06" iyz="0" izz="0.05"/>
    </inertial>
  </link>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="tip"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="500.0" effort="1000.0"/>
  </joint>
</robot>
""")
    return path


def _square(center: np.ndarray, half_span: float) -> np.ndarray:
    x, y = center
    return np.array(
        [
            [x - half_span, y - half_span],
            [x + half_span, y - half_span],
            [x + half_span, y + half_span],
            [x - half_span, y + half_span],
        ],
        dtype=float,
    )


def _options(nv: int) -> eik.AccelerationSolveOptions:
    options = eik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.full(nv, 50.0)
    options.apply_position_limits = False
    options.apply_velocity_limits = False
    return options


def test_dynamic_support_records_export_and_solve(tmp_path):
    for name in (
        "CapturePointAccelerationConstraint",
        "ZmpAccelerationConstraint",
        "CapturePointAccelerationDiagnostics",
        "ZmpAccelerationDiagnostics",
    ):
        assert hasattr(eik, name)

    capabilities = eik.AccelerationSolver.capabilities()
    assert capabilities.supports_fixed_base_capture_point_constraints
    assert capabilities.supports_fixed_base_zmp_constraints
    assert not capabilities.supports_dynamic_contact
    assert not capabilities.supports_dynamic_balance

    robot = eik.RobotModel(str(_write_urdf(tmp_path)), floating_base=False)
    robot.set_gravity(np.array([0.0, 0.0, -9.81]))
    solver = eik.AccelerationSolver(robot)
    q = np.array([0.2, -0.15])
    dq = np.zeros(2)
    robot.update_kinematics(q, dq)
    support = eik.ComSupportPolygonConstraintDefinition()
    support.support_polygon = _square(robot.get_com_position()[:2], 0.75)
    support.frame_name = "world"
    support.margin = 0.0

    cp = eik.CapturePointAccelerationConstraint()
    cp.source_id = "python_cp"
    cp.definition = support
    assert cp.omega is None
    cp.omega = 3.0

    zmp = eik.ZmpAccelerationConstraint()
    zmp.source_id = "python_zmp"
    zmp.definition = support
    zmp.fz_min = 1.0

    options = _options(robot.nv)
    options.capture_point_constraints = [cp]
    options.zmp_constraints = [zmp]
    result = solver.solve(q, dq, 0.01, options)

    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    assert result.q_solution.shape == (robot.nq,)
    assert len(result.capture_point_diagnostics) == 1
    assert len(result.zmp_diagnostics) == 1
    cp_diag = result.capture_point_diagnostics[0]
    assert cp_diag.source_id == "python_cp"
    assert cp_diag.postvalidated
    assert cp_diag.frozen_omega == 3.0
    robot.update_kinematics(result.q_solution, result.joint_velocities_next)
    expected_cp = (
        robot.get_com_position()[:2]
        + (robot.get_com_jacobian() @ result.joint_velocities_next)[:2] / 3.0
    )
    np.testing.assert_allclose(cp_diag.predicted_point_xy, expected_cp, atol=1e-8)
    assert cp_diag.half_plane_slacks.shape == (4,)
    assert np.isclose(cp_diag.min_slack, np.min(cp_diag.half_plane_slacks))
    assert cp_diag.min_slack > 0.0

    zmp_diag = result.zmp_diagnostics[0]
    assert zmp_diag.source_id == "python_zmp"
    assert zmp_diag.postvalidated
    hdot = robot.compute_centroidal_momentum_matrix(
        result.q_solution, result.joint_velocities_next
    ) @ result.joint_accelerations + robot.compute_centroidal_momentum_matrix_bias(
        result.q_solution, result.joint_velocities_next
    )
    force = hdot[:3] - robot.get_total_mass() * robot.get_gravity()
    expected_force_z = force[2]
    expected_zmp = np.array(
        [
            robot.get_com_position()[0]
            - (hdot[4] + robot.get_com_position()[2] * force[0]) / expected_force_z,
            robot.get_com_position()[1]
            + (hdot[3] - robot.get_com_position()[2] * force[1]) / expected_force_z,
        ]
    )
    assert np.isclose(zmp_diag.force_z, expected_force_z)
    np.testing.assert_allclose(zmp_diag.predicted_point_xy, expected_zmp, atol=1e-8)
    assert zmp_diag.half_plane_slacks.shape == (4,)
    assert np.isclose(zmp_diag.min_slack, np.min(zmp_diag.half_plane_slacks))
    assert zmp_diag.min_slack > 0.0

    options.zmp_constraints[0].fz_min = 1e9
    failed = solver.solve(q, dq, 0.01, options)
    assert failed.status == eik.SolverStatus.INFEASIBLE
    assert failed.q_solution.size == 0
