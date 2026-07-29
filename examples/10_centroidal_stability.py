#!/usr/bin/env python3
"""Headless centroidal momentum, capture-point, and ZMP example.

The example uses a small generated fixed-base model so it runs without network
downloads or visualization dependencies. It exercises the explicit-state
velocity API and the fixed-base acceleration API, then reports the accepted
capture-point and ZMP margins.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import embodik

try:
    from example_helpers.ik_common import configure_solver_runtime_policy
except ModuleNotFoundError:
    from examples.example_helpers.ik_common import configure_solver_runtime_policy

_DEMO_URDF = """<?xml version="1.0"?>
<robot name="embodik_centroidal_stability_demo">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.03" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.05"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0.1 0.0 0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
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
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0.45 0.0 0.0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
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
"""


def _square(center: np.ndarray, half_span: float) -> np.ndarray:
    x, y = np.asarray(center, dtype=float)
    return np.array(
        [
            [x - half_span, y - half_span],
            [x + half_span, y - half_span],
            [x + half_span, y + half_span],
            [x - half_span, y + half_span],
        ],
        dtype=float,
    )


def _status_name(status: Any) -> str:
    return str(getattr(status, "name", status))


def _velocity_report(
    robot: embodik.RobotModel,
    q: np.ndarray,
    dq: np.ndarray,
    support_polygon: np.ndarray,
) -> dict[str, Any]:
    solver = embodik.KinematicsSolver(robot)
    configure_solver_runtime_policy(solver)
    solver.dt = 0.01

    momentum = solver.add_centroidal_momentum_task("momentum")
    momentum.set_target_momentum(robot.compute_centroidal_momentum(q, dq))
    solver.configure_centroidal_momentum_bounds(
        lower_h=np.full(6, -5.0),
        upper_h=np.full(6, 5.0),
    )
    solver.configure_capture_point_constraint(support_polygon, omega=3.0)
    solver.configure_velocity_zmp_constraint(support_polygon, fz_min=1.0)

    result = solver.solve_velocity_with_state(q, dq, apply_limits=True)
    if result.status != embodik.SolverStatus.SUCCESS:
        raise RuntimeError(f"velocity centroidal solve failed: {result.status_message}")
    cp = solver.evaluate_capture_point_constraint(q, result.joint_velocities)
    zmp = solver.evaluate_velocity_zmp_constraint(q, dq, result.joint_velocities)
    return {
        "status": _status_name(result.status),
        "capture_point_min_slack": float(np.min(cp["slacks"])),
        "zmp_min_slack": float(np.min(zmp["slacks"])),
        "force_z": float(zmp["force_z"]),
        "command_norm": float(np.linalg.norm(result.joint_velocities)),
    }


def _acceleration_report(
    robot: embodik.RobotModel,
    q: np.ndarray,
    dq: np.ndarray,
    support_polygon: np.ndarray,
) -> dict[str, Any]:
    solver = embodik.AccelerationSolver(robot)
    support = embodik.ComSupportPolygonConstraintDefinition()
    support.support_polygon = support_polygon
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

    cp = embodik.CapturePointAccelerationConstraint()
    cp.source_id = "predicted_capture_point"
    cp.definition = support
    cp.omega = 3.0

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
    options.capture_point_constraints = [cp]
    options.zmp_constraints = [zmp]

    result = solver.solve(q, dq, 0.01, options)
    if result.status != embodik.SolverStatus.SUCCESS:
        raise RuntimeError(f"acceleration centroidal solve failed: {result.status_message}")
    cp_diagnostic = result.capture_point_diagnostics[0]
    zmp_diagnostic = result.zmp_diagnostics[0]
    return {
        "status": _status_name(result.status),
        "capture_point_min_slack": float(cp_diagnostic.min_slack),
        "zmp_min_slack": float(zmp_diagnostic.min_slack),
        "force_z": float(zmp_diagnostic.force_z),
        "acceleration_norm": float(np.linalg.norm(result.joint_accelerations)),
    }


def run_headless_demo() -> dict[str, Any]:
    """Run both centroidal solver levels and return machine-readable evidence."""
    with tempfile.TemporaryDirectory(prefix="embodik-centroidal-") as directory:
        urdf_path = Path(directory) / "centroidal_stability_demo.urdf"
        urdf_path.write_text(_DEMO_URDF, encoding="utf-8")
        robot = embodik.RobotModel(str(urdf_path), floating_base=False)
        robot.set_gravity(np.array([0.0, 0.0, -9.81]))
        q = np.array([0.25, -0.35])
        dq = np.zeros(robot.nv)
        robot.update_kinematics(q, dq)
        support_polygon = _square(robot.get_com_position()[:2], 0.75)

        capabilities = embodik.AccelerationSolver.capabilities()
        return {
            "velocity": _velocity_report(robot, q, dq, support_polygon),
            "acceleration": _acceleration_report(robot, q, dq, support_polygon),
            "capabilities": {
                "fixed_base_capture_point": bool(
                    capabilities.supports_fixed_base_capture_point_constraints
                ),
                "fixed_base_zmp": bool(capabilities.supports_fixed_base_zmp_constraints),
                "supports_dynamic_balance": bool(capabilities.supports_dynamic_balance),
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed-base centroidal stability constraints headlessly."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit only a machine-readable JSON report.",
    )
    args = parser.parse_args()
    report = run_headless_demo()
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
