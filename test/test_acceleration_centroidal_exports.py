import pathlib

import numpy as np

import embodik as eik


def _write_urdf(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "acceleration_centroidal_exports.urdf"
    path.write_text(
        """<?xml version="1.0"?>
<robot name="acceleration_centroidal_exports_robot">
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
    <limit lower="-2.4" upper="2.3" velocity="70.0" effort="400.0"/>
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
    <limit lower="-1.35" upper="1.45" velocity="30.0" effort="350.0"/>
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
    <limit lower="-1.7" upper="1.8" velocity="50.0" effort="200.0"/>
  </joint>
</robot>
"""
    )
    return path


def _options() -> eik.AccelerationSolveOptions:
    options = eik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.array([100.0, 100.0, 100.0])
    options.apply_position_limits = False
    options.apply_velocity_limits = False
    return options


def test_centroidal_momentum_rate_exports_and_solve(tmp_path):
    capabilities = eik.AccelerationSolver.capabilities()
    assert capabilities.supports_fixed_base_centroidal_momentum_rate_objective
    assert capabilities.supports_fixed_base_centroidal_momentum_rate_bounds
    assert not capabilities.supports_floating_base_centroidal_momentum_rate
    assert not capabilities.supports_dynamic_balance

    robot = eik.RobotModel(str(_write_urdf(tmp_path)), floating_base=False)
    solver = eik.AccelerationSolver(robot)
    q = np.array([0.37, -0.11, -0.42])
    dq = np.array([-0.23, 0.31, 0.17])
    ddq_ref = np.array([0.41, -0.19, 0.29])
    current_h = robot.compute_centroidal_momentum(q, dq)
    hdot_ref = robot.compute_centroidal_momentum_matrix(q, dq) @ ddq_ref
    hdot_ref += robot.compute_centroidal_momentum_matrix_bias(q, dq)

    objective = eik.CentroidalMomentumRateObjective()
    objective.source_id = "python_centroidal"
    objective.h_target = current_h
    objective.hdot_feedforward = hdot_ref
    objective.proportional_gain = 0.0
    objective.axis_mask = [True, True, True, True, True, True]
    objective.priority = 0
    objective.solve_mode = eik.TaskSolveMode.SCALE
    objective.allow_min_error_fallback = True

    bounds = eik.CentroidalMomentumRateBounds()
    bounds.source_id = "python_linear_y"
    bounds.lower_bounds = np.full(6, -100.0)
    bounds.upper_bounds = np.full(6, 100.0)
    bounds.axis_mask = [False, False, False, False, True, False]

    options = _options()
    options.centroidal_momentum_rate_objectives = [objective]
    options.centroidal_momentum_rate_bounds = [bounds]
    result = solver.solve(q, dq, 0.01, options)

    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    achieved = robot.compute_centroidal_momentum_matrix(q, dq) @ result.joint_accelerations
    achieved += robot.compute_centroidal_momentum_matrix_bias(q, dq)
    np.testing.assert_allclose(achieved, hdot_ref, atol=1e-8)
    assert len(result.centroidal_momentum_rate_diagnostics) == 1
    diagnostic = result.centroidal_momentum_rate_diagnostics[0]
    assert diagnostic.source_id == "python_centroidal"
    np.testing.assert_allclose(diagnostic.current_momentum, current_h, atol=1e-8)
    np.testing.assert_allclose(diagnostic.reference_momentum_rate, hdot_ref, atol=1e-8)
    np.testing.assert_allclose(diagnostic.achieved_momentum_rate, achieved, atol=1e-8)
    assert diagnostic.selected_axes == [True, True, True, True, True, True]
