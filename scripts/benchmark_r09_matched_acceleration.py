#!/usr/bin/env python3
"""R09 matched velocity-vs-acceleration benchmark harness.

The report is comparative hot-cache telemetry for fixed-base scalar-joint
models. It is not a WCET or real-time guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_descriptions.panda_description import URDF_PATH as PANDA_URDF_PATH  # noqa: E402

import embodik as eik  # noqa: E402
from examples.utils.dual_iiwa_urdf import build_dual_iiwa_urdf  # noqa: E402
from examples.utils.robot_models import ensure_ros_package_path  # noqa: E402

BENCHMARK_ID = "R09_matched_velocity_acceleration"
SCHEMA_VERSION = 2
ACCEL_MEDIAN_OVERHEAD_LIMIT = 0.15
ACCEL_P95_OVERHEAD_LIMIT = 0.20
CONSTRAINT_TOL = 1e-8
REQUIRED_FIXED_SCENARIOS = {
    "panda_fixed_base_frame_position",
    "dual_iiwa_fixed_base_frame_position",
}
REQUIRED_DIAGNOSTIC_SCENARIOS = {
    "dual_iiwa_masked_collision_velocity_lift",
}
REQUIRED_SCENARIO_KEYS = {"panda", "iiwa", "iiwa_collision"}
MIN_ACCEPTANCE_WARMUP = 80
MIN_ACCEPTANCE_REPETITIONS = 600
ACCEPTANCE_DT = 0.01

PANDA_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
PANDA_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
PANDA_DQ = np.zeros(7, dtype=float)
PANDA_TASK_V = np.array([0.035, -0.015, 0.02], dtype=float)
IIWA_Q = np.array(
    [
        math.radians(0),
        math.radians(45),
        math.radians(0),
        math.radians(-90),
        math.radians(0),
        math.radians(45),
        math.radians(0),
        math.radians(0),
        math.radians(45),
        math.radians(0),
        math.radians(-90),
        math.radians(0),
        math.radians(45),
        math.radians(0),
    ],
    dtype=float,
)
IIWA_DQ = np.zeros(14, dtype=float)
IIWA_LEFT_V = np.array([0.025, 0.0, 0.012], dtype=float)
IIWA_RIGHT_V = np.array([-0.025, 0.0, 0.012], dtype=float)

SPHERE_COLLISION_URDF = """<?xml version="1.0"?>
<robot name="r09_collision_pair">
  <link name="world"/>
  <link name="obstacle">
    <collision name="obstacle_sphere"><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <joint name="obstacle_fixed" type="fixed">
    <parent link="world"/>
    <child link="obstacle"/>
  </joint>
  <link name="moving">
    <collision name="moving_sphere"><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <joint name="moving_slide" type="prismatic">
    <parent link="world"/>
    <child link="moving"/>
    <origin xyz="0.12 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.0" upper="1.0" effort="1000" velocity="100"/>
  </joint>
</robot>
"""


@dataclass
class ScenarioRuntime:
    name: str
    model: str
    q: np.ndarray
    dq: np.ndarray
    dt: float
    velocity_robot: Any
    acceleration_robot: Any
    velocity_solver: Any
    acceleration_solver: Any
    acceleration_options: Any
    task_contract: list[dict[str, Any]]
    constraint_contract: dict[str, Any]
    collision_mode: str = "none"
    collision_solver: Any | None = None
    lift_options: Any | None = None
    temp_path: Path | None = None

    def cleanup(self) -> None:
        if self.temp_path is not None:
            try:
                self.temp_path.unlink()
            except FileNotFoundError:
                pass


def _status_name(status: Any) -> str:
    return str(getattr(status, "name", str(status))).split(".")[-1]


def _project_version() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    return str(data["project"]["version"])


def _build_config() -> dict[str, Any]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    cmake_define = data.get("tool", {}).get("scikit-build", {}).get("cmake", {}).get("define", {})
    return {
        "cmake_build_type": os.environ.get(
            "CMAKE_BUILD_TYPE", str(cmake_define.get("CMAKE_BUILD_TYPE", "unknown"))
        ),
        "build_examples": str(cmake_define.get("BUILD_EXAMPLES", "unknown")),
        "build_tests": str(cmake_define.get("BUILD_TESTS", "unknown")),
        "python_executable": sys.executable,
    }


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_dirty() -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return bool(status.strip())


def _current_metadata() -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    return {
        "git": {"head": _git_head(), "dirty": _git_dirty()},
        "project_version": _project_version(),
        "build_config": _build_config(),
        "python_version": sys.version.split()[0],
        "python_compiler": platform.python_compiler(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_affinity": affinity,
    }


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    sorted_values = sorted(float(v) for v in values)
    p95_idx = max(0, min(len(sorted_values) - 1, int(math.ceil(0.95 * len(sorted_values)) - 1)))
    return {
        "min": float(sorted_values[0]),
        "mean": float(statistics.fmean(sorted_values)),
        "median": float(statistics.median(sorted_values)),
        "p95": float(sorted_values[p95_idx]),
        "max": float(sorted_values[-1]),
    }


def _sha256_array(values: np.ndarray) -> str:
    data = np.asarray(values, dtype=np.float64)
    return hashlib.sha256(data.tobytes()).hexdigest()


def _reference_for_velocity(target_velocity: np.ndarray, dt: float) -> Any:
    reference = eik.AccelerationTaskReference()
    reference.desired_velocity = np.asarray(target_velocity, dtype=float)
    reference.desired_acceleration = np.zeros(target_velocity.size, dtype=float)
    reference.proportional_gain = 0.0
    reference.derivative_gain = 1.0 / dt
    return reference


def _options(robot: Any) -> Any:
    opts = eik.AccelerationSolveOptions()
    opts.acceleration_limits_override = _acceleration_limits(robot)
    opts.apply_position_limits = True
    opts.apply_velocity_limits = True
    opts.collect_task_diagnostics = False
    return opts


def _add_position_task_pair(
    velocity_solver: Any,
    acceleration_solver: Any,
    name: str,
    frame: str,
    target_velocity: np.ndarray,
    dt: float,
) -> None:
    vtask = velocity_solver.add_frame_task(name, frame, eik.TaskType.FRAME_POSITION)
    vtask.priority = 0
    vtask.weight = 10.0
    vtask.set_target_position_velocity(np.asarray(target_velocity, dtype=float))

    atask = acceleration_solver.add_frame_task(name, frame, eik.TaskType.FRAME_POSITION)
    atask.priority = 0
    atask.weight = 10.0
    acceleration_solver.set_task_reference(name, _reference_for_velocity(target_velocity, dt))


def _acceleration_limits(robot: Any) -> np.ndarray:
    return np.minimum(
        np.asarray(robot.get_acceleration_limits(), dtype=float),
        np.full(robot.nv, 20.0),
    )


def _configure_velocity_solver(robot: Any, dq: np.ndarray, dt: float) -> Any:
    solver = eik.KinematicsSolver(robot)
    solver.dt = dt
    solver.set_damping(0.1)
    solver.enable_timing_breakdown(True)
    solver.enable_position_limits(True)
    solver.set_acceleration_limits(_acceleration_limits(robot))
    solver.enable_acceleration_limits(True)
    solver.set_previous_joint_velocities(np.asarray(dq, dtype=float))
    return solver


def _build_panda(dt: float) -> ScenarioRuntime:
    ensure_ros_package_path(Path(PANDA_URDF_PATH))
    v_robot = eik.RobotModel(str(PANDA_URDF_PATH), actuated_joint_names=PANDA_JOINTS)
    a_robot = eik.RobotModel(str(PANDA_URDF_PATH), actuated_joint_names=PANDA_JOINTS)
    v_robot.update_kinematics(PANDA_Q, PANDA_DQ)
    a_robot.update_kinematics(PANDA_Q, PANDA_DQ)
    v_solver = _configure_velocity_solver(v_robot, PANDA_DQ, dt)
    a_solver = eik.AccelerationSolver(a_robot)
    _add_position_task_pair(
        v_solver, a_solver, "panda_hand_position", "panda_hand", PANDA_TASK_V, dt
    )
    return ScenarioRuntime(
        name="panda_fixed_base_frame_position",
        model="panda_reduced_7dof_fixed_base",
        q=PANDA_Q.copy(),
        dq=PANDA_DQ.copy(),
        dt=dt,
        velocity_robot=v_robot,
        acceleration_robot=a_robot,
        velocity_solver=v_solver,
        acceleration_solver=a_solver,
        acceleration_options=_options(a_robot),
        task_contract=[
            {
                "name": "panda_hand_position",
                "frame": "panda_hand",
                "type": "FRAME_POSITION",
                "target_velocity": PANDA_TASK_V.tolist(),
                "priority": 0,
            }
        ],
        constraint_contract={
            "seed_provenance": (
                "R00 full-model Panda arm-state projection; finger joints are locked "
                "because its 0.05 finger seed exceeds the current URDF limit"
            ),
            "position_limits": True,
            "velocity_limits": True,
            "acceleration_limits": True,
            "collect_task_diagnostics": False,
            "collision": "none",
        },
    )


def _build_dual_iiwa_runtime(dt: float, *, collision: bool) -> ScenarioRuntime:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_r09_dual_iiwa.urdf", delete=False
    ) as handle:
        handle.write(build_dual_iiwa_urdf(replace_mesh_collision=True))
        urdf_path = Path(handle.name)
    ensure_ros_package_path(urdf_path)
    v_robot = eik.RobotModel(str(urdf_path), floating_base=False)
    a_robot = v_robot if collision else eik.RobotModel(str(urdf_path), floating_base=False)
    v_robot.update_kinematics(IIWA_Q, IIWA_DQ)
    a_robot.update_kinematics(IIWA_Q, IIWA_DQ)
    v_solver = _configure_velocity_solver(v_robot, IIWA_DQ, dt)
    a_solver = eik.AccelerationSolver(a_robot)
    _add_position_task_pair(
        v_solver, a_solver, "left_link7_position", "iiwa_left_iiwa_link_7", IIWA_LEFT_V, dt
    )
    _add_position_task_pair(
        v_solver,
        a_solver,
        "right_link7_position",
        "iiwa_right_iiwa_link_7",
        IIWA_RIGHT_V,
        dt,
    )
    include_pairs: list[tuple[str, str]] = []
    lift = None
    if collision:
        geometry_names = list(v_robot.get_collision_geometry_names())
        left_candidates = [
            name
            for name in geometry_names
            if "left_iiwa_link_6" in name or "left_iiwa_link_7" in name
        ]
        right_candidates = [
            name
            for name in geometry_names
            if "right_iiwa_link_6" in name or "right_iiwa_link_7" in name
        ]
        include_pairs = [
            (left_name, right_name)
            for left_name in left_candidates
            for right_name in right_candidates
        ]
        v_solver.set_collision_refinement_time_budget_us(0)
        v_solver.enable_collision_pair_cache(False, 20, 0.03, 128)
        v_solver.configure_collision_constraint(
            min_distance=0.02,
            include_pairs=include_pairs,
            exclude_pairs=[],
            nearest_points_all_pairs=False,
            max_constraints=4,
        )
        lift = eik.VelocityCollisionLiftOptions()
        lift.validation_substeps = 1

    name = (
        "dual_iiwa_masked_collision_velocity_lift"
        if collision
        else "dual_iiwa_fixed_base_frame_position"
    )
    return ScenarioRuntime(
        name=name,
        model="dual_iiwa_simplified_collision_fixed_base",
        q=IIWA_Q.copy(),
        dq=IIWA_DQ.copy(),
        dt=dt,
        velocity_robot=v_robot,
        acceleration_robot=a_robot,
        velocity_solver=v_solver,
        acceleration_solver=a_solver,
        acceleration_options=_options(a_robot),
        task_contract=[
            {
                "name": "left_link7_position",
                "frame": "iiwa_left_iiwa_link_7",
                "type": "FRAME_POSITION",
                "target_velocity": IIWA_LEFT_V.tolist(),
                "priority": 0,
            },
            {
                "name": "right_link7_position",
                "frame": "iiwa_right_iiwa_link_7",
                "type": "FRAME_POSITION",
                "target_velocity": IIWA_RIGHT_V.tolist(),
                "priority": 0,
            },
        ],
        constraint_contract={
            "position_limits": True,
            "velocity_limits": True,
            "acceleration_limits": True,
            "collect_task_diagnostics": False,
            "collision": "velocity_collision_lift" if collision else "none",
            "min_distance": 0.02 if collision else None,
            "include_pairs": include_pairs,
            "exclude_pairs": [],
            "nearest_points_all_pairs": False,
            "max_constraints": 4 if collision else 0,
            "collision_pair_cache_enabled": False,
            "collision_refinement_budget_us": 0,
            "validation_substeps": 1 if collision else 0,
        },
        collision_mode="velocity_collision_lift" if collision else "none",
        collision_solver=v_solver if collision else None,
        lift_options=lift,
        temp_path=urdf_path,
    )


def _build_dual_iiwa(dt: float) -> ScenarioRuntime:
    return _build_dual_iiwa_runtime(dt, collision=False)


def _build_dual_iiwa_collision(dt: float) -> ScenarioRuntime:
    return _build_dual_iiwa_runtime(dt, collision=True)


def _build_collision_lift(dt: float) -> ScenarioRuntime:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_r09_collision_pair.urdf", delete=False
    ) as handle:
        handle.write(SPHERE_COLLISION_URDF)
        urdf_path = Path(handle.name)
    v_robot = eik.RobotModel(str(urdf_path), floating_base=False)
    a_robot = v_robot
    q = np.array([0.0], dtype=float)
    dq = np.array([0.0], dtype=float)
    target_v = np.array([0.02, 0.0, 0.0], dtype=float)
    v_robot.update_kinematics(q, dq)
    v_solver = _configure_velocity_solver(v_robot, dq, dt)
    v_solver.configure_collision_constraint(
        min_distance=0.01,
        include_pairs=[],
        exclude_pairs=[],
        nearest_points_all_pairs=False,
        max_constraints=1,
    )
    a_solver = eik.AccelerationSolver(a_robot)
    _add_position_task_pair(v_solver, a_solver, "moving_position", "moving", target_v, dt)
    lift = eik.VelocityCollisionLiftOptions()
    lift.validation_substeps = 3
    return ScenarioRuntime(
        name="sphere_prismatic_velocity_collision_lift",
        model="single_prismatic_sphere_pair_fixed_base",
        q=q,
        dq=dq,
        dt=dt,
        velocity_robot=v_robot,
        acceleration_robot=a_robot,
        velocity_solver=v_solver,
        acceleration_solver=a_solver,
        acceleration_options=_options(a_robot),
        task_contract=[
            {
                "name": "moving_position",
                "frame": "moving",
                "type": "FRAME_POSITION",
                "target_velocity": target_v.tolist(),
                "priority": 0,
            }
        ],
        constraint_contract={
            "position_limits": True,
            "velocity_limits": True,
            "acceleration_limits": True,
            "collect_task_diagnostics": False,
            "collision": "velocity_collision_lift",
            "min_distance": 0.01,
            "validation_substeps": 3,
        },
        collision_mode="velocity_collision_lift",
        collision_solver=v_solver,
        lift_options=lift,
        temp_path=urdf_path,
    )


def _scenario_builders() -> dict[str, Any]:
    return {
        "panda": _build_panda,
        "iiwa": _build_dual_iiwa,
        "iiwa_collision": _build_dual_iiwa_collision,
        "collision_lift": _build_collision_lift,
    }


def _safe_constraint_violation(robot: Any, options: Any, result: Any) -> float:
    q_solution = np.asarray(result.q_solution, dtype=float)
    dq_next = np.asarray(result.joint_velocities_next, dtype=float)
    ddq = np.asarray(result.joint_accelerations, dtype=float)
    lower_limits, upper_limits = robot.get_joint_limits()
    limits = np.asarray(options.acceleration_limits_override, dtype=float)
    checks = [
        np.maximum(np.abs(ddq) - limits, 0.0),
        np.maximum(np.abs(dq_next) - np.asarray(robot.get_velocity_limits(), dtype=float), 0.0),
        np.maximum(np.asarray(lower_limits, dtype=float) - q_solution, 0.0),
        np.maximum(q_solution - np.asarray(upper_limits, dtype=float), 0.0),
    ]
    return float(max(float(np.max(check)) for check in checks if check.size > 0))


def _run_acceleration(runtime: ScenarioRuntime) -> Any:
    if runtime.collision_mode == "velocity_collision_lift":
        assert runtime.collision_solver is not None
        assert runtime.lift_options is not None
        return runtime.acceleration_solver.solve_with_velocity_collision(
            runtime.collision_solver,
            runtime.q,
            runtime.dq,
            runtime.dt,
            runtime.acceleration_options,
            runtime.lift_options,
        )
    return runtime.acceleration_solver.solve(
        runtime.q, runtime.dq, runtime.dt, runtime.acceleration_options
    )


def _collision_identity(
    runtime: ScenarioRuntime, velocity_result: Any, acceleration_result: Any
) -> dict[str, Any]:
    declared = runtime.collision_mode != "none"
    active_velocity_pairs = (
        len(runtime.velocity_solver.get_active_collision_pairs()) if declared else 0
    )
    identity = {
        "declared": declared,
        "mode": runtime.collision_mode,
        "velocity_active_pair_count": int(active_velocity_pairs),
        "velocity_pairs_considered": int(getattr(velocity_result, "collision_pairs_considered", 0)),
        "velocity_exact_distance_queries": int(
            getattr(velocity_result, "collision_exact_distance_queries", 0)
        ),
        "acceleration_endpoint_validated": bool(
            getattr(acceleration_result, "collision_endpoint_validated", False)
        ),
        "acceleration_step_certified": bool(
            getattr(acceleration_result, "collision_step_certified", False)
        ),
        "acceleration_expected_step_certified": False,
        "acceleration_validation_samples": int(
            getattr(acceleration_result, "collision_validation_samples", 0)
        ),
        "acceleration_validation_allowed_pairs": int(
            getattr(acceleration_result, "collision_validation_allowed_pairs", 0)
        ),
        "acceleration_validation_pairs_checked": int(
            getattr(acceleration_result, "collision_validation_pairs_checked", 0)
        ),
        "acceleration_lift_pairs_considered": int(
            getattr(acceleration_result, "collision_lift_pairs_considered", 0)
        ),
        "native_collision_constraint_applied": bool(
            getattr(acceleration_result, "native_collision_constraint_applied", False)
        ),
    }
    if declared:
        expected_checks = (
            identity["velocity_active_pair_count"] * identity["acceleration_validation_samples"]
        )
        identity["checks_passed"] = (
            identity["velocity_active_pair_count"] > 0
            and identity["acceleration_endpoint_validated"]
            and not identity["acceleration_step_certified"]
            and identity["acceleration_validation_allowed_pairs"]
            == identity["velocity_active_pair_count"]
            and identity["acceleration_validation_pairs_checked"] == expected_checks
        )
    else:
        identity["checks_passed"] = (
            identity["velocity_pairs_considered"] == 0
            and not identity["acceleration_endpoint_validated"]
            and not identity["acceleration_step_certified"]
            and identity["acceleration_validation_pairs_checked"] == 0
        )
    return identity


def _run_scenario(runtime: ScenarioRuntime, warmup: int, repetitions: int) -> dict[str, Any]:
    velocity_walls: list[float] = []
    acceleration_walls: list[float] = []
    velocity_backend: list[float] = []
    acceleration_backend: list[float] = []
    acceleration_cpp_total: list[float] = []
    acceleration_preprocessing: list[float] = []
    acceleration_postprocessing: list[float] = []
    velocity_status_counts: dict[str, int] = {}
    acceleration_status_counts: dict[str, int] = {}
    velocity_checksum: str | None = None
    acceleration_checksum: str | None = None
    velocity_mismatches = 0
    acceleration_mismatches = 0
    max_constraint_violation = 0.0
    max_task_velocity_delta = 0.0
    max_velocity_target_error = 0.0
    max_acceleration_target_error = 0.0
    last_velocity = None
    last_acceleration = None

    for _ in range(warmup):
        runtime.velocity_solver.set_previous_joint_velocities(runtime.dq)
        v_result = runtime.velocity_solver.solve_velocity(runtime.q, apply_limits=True)
        a_result = _run_acceleration(runtime)
        if _status_name(v_result.status) != "SUCCESS":
            raise RuntimeError(f"{runtime.name} velocity warmup failed: {v_result.status_message}")
        if _status_name(a_result.status) != "SUCCESS":
            raise RuntimeError(
                f"{runtime.name} acceleration warmup failed: {a_result.status_message}"
            )

    for _ in range(repetitions):
        v_start = perf_counter_ns()
        runtime.velocity_solver.set_previous_joint_velocities(runtime.dq)
        v_result = runtime.velocity_solver.solve_velocity(runtime.q, apply_limits=True)
        v_wall = (perf_counter_ns() - v_start) / 1e6
        a_start = perf_counter_ns()
        a_result = _run_acceleration(runtime)
        a_wall = (perf_counter_ns() - a_start) / 1e6

        v_status = _status_name(v_result.status)
        a_status = _status_name(a_result.status)
        velocity_status_counts[v_status] = velocity_status_counts.get(v_status, 0) + 1
        acceleration_status_counts[a_status] = acceleration_status_counts.get(a_status, 0) + 1
        if v_status != "SUCCESS":
            raise RuntimeError(f"{runtime.name} velocity solve failed: {v_result.status_message}")
        if a_status != "SUCCESS":
            raise RuntimeError(
                f"{runtime.name} acceleration solve failed: {a_result.status_message}"
            )

        v_digest = _sha256_array(np.asarray(v_result.joint_velocities, dtype=float))
        a_digest = _sha256_array(np.asarray(a_result.joint_velocities_next, dtype=float))
        if velocity_checksum is None:
            velocity_checksum = v_digest
        elif v_digest != velocity_checksum:
            velocity_mismatches += 1
        if acceleration_checksum is None:
            acceleration_checksum = a_digest
        elif a_digest != acceleration_checksum:
            acceleration_mismatches += 1

        max_constraint_violation = max(
            max_constraint_violation,
            _safe_constraint_violation(
                runtime.acceleration_robot, runtime.acceleration_options, a_result
            ),
        )
        for task in runtime.task_contract:
            jacobian = np.asarray(
                runtime.velocity_robot.get_frame_jacobian(task["frame"]),
                dtype=float,
            )[:3, :]
            target_velocity = np.asarray(task["target_velocity"], dtype=float)
            velocity_achieved = jacobian @ np.asarray(v_result.joint_velocities, dtype=float)
            acceleration_achieved = jacobian @ np.asarray(
                a_result.joint_velocities_next, dtype=float
            )
            max_task_velocity_delta = max(
                max_task_velocity_delta,
                float(np.max(np.abs(velocity_achieved - acceleration_achieved))),
            )
            max_velocity_target_error = max(
                max_velocity_target_error,
                float(np.linalg.norm(velocity_achieved - target_velocity)),
            )
            max_acceleration_target_error = max(
                max_acceleration_target_error,
                float(np.linalg.norm(acceleration_achieved - target_velocity)),
            )
        velocity_walls.append(v_wall)
        acceleration_walls.append(a_wall)
        velocity_backend.append(float(getattr(v_result, "computation_time_ms", 0.0)))
        acceleration_backend.append(float(getattr(a_result, "backend_computation_time_ms", 0.0)))
        acceleration_cpp_total.append(float(getattr(a_result, "computation_time_ms", 0.0)))
        acceleration_preprocessing.append(float(getattr(a_result, "preprocessing_time_ms", 0.0)))
        acceleration_postprocessing.append(float(getattr(a_result, "postprocessing_time_ms", 0.0)))
        last_velocity = v_result
        last_acceleration = a_result

    if last_velocity is None or last_acceleration is None:
        raise RuntimeError(f"{runtime.name} did not produce any measured samples")
    collision = _collision_identity(runtime, last_velocity, last_acceleration)
    correctness = {
        "velocity_status_counts": velocity_status_counts,
        "acceleration_status_counts": acceleration_status_counts,
        "velocity_deterministic_mismatches": velocity_mismatches,
        "acceleration_deterministic_mismatches": acceleration_mismatches,
        "max_constraint_violation": max_constraint_violation,
        "max_task_velocity_delta": max_task_velocity_delta,
        "max_velocity_target_error": max_velocity_target_error,
        "max_acceleration_target_error": max_acceleration_target_error,
        "checks_passed": (
            velocity_status_counts == {"SUCCESS": repetitions}
            and acceleration_status_counts == {"SUCCESS": repetitions}
            and velocity_mismatches == 0
            and acceleration_mismatches == 0
            and max_constraint_violation <= CONSTRAINT_TOL
            and max_acceleration_target_error <= max_velocity_target_error + CONSTRAINT_TOL
            and bool(collision["checks_passed"])
        ),
    }
    return {
        "model": runtime.model,
        "performance_contract": {
            "thresholds_enforced": runtime.collision_mode == "none",
            "scope": (
                "matched_solver_wall"
                if runtime.collision_mode == "none"
                else "diagnostic_unmatched_sample_validation"
            ),
            "reason": (
                ""
                if runtime.collision_mode == "none"
                else (
                    "acceleration includes sampled endpoint validation that the "
                    "velocity call does not perform"
                )
            ),
        },
        "matched_inputs": {
            "q": runtime.q.tolist(),
            "dq": runtime.dq.tolist(),
            "dt": runtime.dt,
            "q_sha256": _sha256_array(runtime.q),
            "dq_sha256": _sha256_array(runtime.dq),
            "tasks": runtime.task_contract,
            "constraints": runtime.constraint_contract,
        },
        "correctness": correctness,
        "collision_identity": collision,
        "velocity": {
            "timing_ms": {
                "wall": _summary(velocity_walls),
                "reported_backend": _summary(velocity_backend),
            },
            "output_checksum_sha256": velocity_checksum,
        },
        "acceleration": {
            "timing_ms": {
                "wall": _summary(acceleration_walls),
                "reported_full_cpp_solve": _summary(acceleration_cpp_total),
                "reported_preprocessing": _summary(acceleration_preprocessing),
                "reported_backend": _summary(acceleration_backend),
                "reported_postprocessing": _summary(acceleration_postprocessing),
            },
            "output_checksum_sha256": acceleration_checksum,
        },
    }


def _run_selected_scenarios(
    names: list[str], warmup: int, repetitions: int, dt: float
) -> dict[str, Any]:
    builders = _scenario_builders()
    scenarios: dict[str, Any] = {}
    for key in names:
        runtime = builders[key](dt)
        try:
            scenarios[runtime.name] = _run_scenario(runtime, warmup, repetitions)
        finally:
            runtime.cleanup()
    return scenarios


def _write_immutable_json(output: Path, data: dict[str, Any]) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing benchmark artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _public_benchmarks_path(path: Path) -> bool:
    resolved = path.resolve()
    public_root = (REPO_ROOT / ".benchmarks").resolve()
    return resolved == public_root or public_root in resolved.parents


def _load_artifact(
    path: Path, allowed_benchmark_ids: tuple[str, ...] = (BENCHMARK_ID,)
) -> dict[str, Any]:
    if _public_benchmarks_path(path):
        raise ValueError(f"comparison refuses public .benchmarks artifact: {path}")
    data = json.loads(path.read_text())
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path} has unsupported schema_version")
    if data.get("benchmark_id") not in allowed_benchmark_ids:
        allowed = ", ".join(allowed_benchmark_ids)
        raise ValueError(f"{path} benchmark_id must be one of: {allowed}")
    return data


def _validate_artifact_current(path: Path, artifact: dict[str, Any]) -> None:
    metadata = artifact.get("metadata", {})
    git_meta = metadata.get("git", {})
    if git_meta.get("dirty"):
        raise ValueError(f"{path} was produced from a dirty worktree")
    if git_meta.get("head") != _git_head():
        raise ValueError(f"{path} is stale for current HEAD {_git_head()}")
    if metadata.get("project_version") != _project_version():
        raise ValueError(f"{path} has stale project_version")
    if metadata.get("build_config") != _build_config():
        raise ValueError(f"{path} has wrong build_config")


def _ratio(candidate: float, baseline: float) -> float:
    if baseline <= 0.0:
        raise ValueError("baseline timing must be positive")
    return (candidate - baseline) / baseline


def _assert_correctness_before_thresholds(label: str, artifact: dict[str, Any]) -> None:
    for name, scenario in artifact.get("scenarios", {}).items():
        correctness = scenario.get("correctness", {})
        collision = scenario.get("collision_identity", {})
        if not correctness.get("checks_passed", False):
            raise ValueError(f"{label}:{name} failed correctness checks")
        if not collision.get("checks_passed", False):
            raise ValueError(f"{label}:{name} failed collision identity checks")


def evaluate_artifact(candidate_path: Path) -> dict[str, Any]:
    candidate = _load_artifact(candidate_path)
    _validate_artifact_current(candidate_path, candidate)
    _assert_correctness_before_thresholds("candidate", candidate)
    scenario_names = set(candidate.get("scenarios", {}))
    required_scenarios = REQUIRED_FIXED_SCENARIOS | REQUIRED_DIAGNOSTIC_SCENARIOS
    if scenario_names != required_scenarios:
        raise ValueError(
            "candidate scenarios must be exactly: " + ", ".join(sorted(required_scenarios))
        )
    parameters = candidate.get("parameters", {})
    scenario_keys = set(parameters.get("scenario_keys", []))
    if scenario_keys != REQUIRED_SCENARIO_KEYS:
        raise ValueError(
            "candidate scenario_keys must be exactly: " + ", ".join(sorted(REQUIRED_SCENARIO_KEYS))
        )
    if (
        parameters.get("warmup", -1) < MIN_ACCEPTANCE_WARMUP
        or parameters.get("repetitions", -1) < MIN_ACCEPTANCE_REPETITIONS
    ):
        raise ValueError(
            f"candidate requires warmup >= {MIN_ACCEPTANCE_WARMUP} and "
            f"repetitions >= {MIN_ACCEPTANCE_REPETITIONS}"
        )
    if not math.isclose(
        float(parameters.get("dt", math.nan)),
        ACCEPTANCE_DT,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"candidate dt must be {ACCEPTANCE_DT}")

    failures: list[str] = []
    comparisons: dict[str, Any] = {}
    for name, scenario in candidate["scenarios"].items():
        velocity = scenario["velocity"]["timing_ms"]["wall"]
        acceleration = scenario["acceleration"]["timing_ms"]["wall"]
        median_overhead = _ratio(float(acceleration["median"]), float(velocity["median"]))
        p95_overhead = _ratio(float(acceleration["p95"]), float(velocity["p95"]))
        scenario_result = {
            "acceleration_over_velocity_median_overhead": median_overhead,
            "acceleration_over_velocity_p95_overhead": p95_overhead,
        }
        performance_contract = scenario.get("performance_contract")
        if not isinstance(performance_contract, dict):
            raise ValueError(f"candidate:{name} has no performance contract")
        enforce_thresholds = performance_contract.get("thresholds_enforced")
        if not isinstance(enforce_thresholds, bool):
            raise ValueError(f"candidate:{name} has invalid performance threshold policy")
        scenario_result["thresholds_enforced"] = enforce_thresholds
        scenario_result["scope"] = performance_contract.get("scope", "")
        if name in REQUIRED_FIXED_SCENARIOS and (
            not enforce_thresholds or performance_contract.get("scope") != "matched_solver_wall"
        ):
            raise ValueError(f"candidate:{name} must enforce the matched solver wall-time gates")
        if name in REQUIRED_DIAGNOSTIC_SCENARIOS and enforce_thresholds:
            raise ValueError(
                f"candidate:{name} must report unmatched collision validation " "as diagnostic"
            )
        if not enforce_thresholds:
            collision = scenario.get("collision_identity", {})
            if (
                not collision.get("declared", False)
                or collision.get("mode") != "velocity_collision_lift"
                or performance_contract.get("scope") != "diagnostic_unmatched_sample_validation"
            ):
                raise ValueError(
                    f"candidate:{name} may skip thresholds only for the "
                    "velocity-collision lift's unmatched sampled validation"
                )
            comparisons[name] = scenario_result
            continue
        if median_overhead > ACCEL_MEDIAN_OVERHEAD_LIMIT:
            failures.append(f"{name} acceleration median overhead {median_overhead:.3%} > 15%")
        if p95_overhead > ACCEL_P95_OVERHEAD_LIMIT:
            failures.append(f"{name} acceleration p95 overhead {p95_overhead:.3%} > 20%")
        comparisons[name] = scenario_result

    passed = not failures
    return {"passed": passed, "failures": failures, "comparisons": comparisons}


def _run_command(args: argparse.Namespace) -> None:
    selected = args.scenario or ["panda", "iiwa", "iiwa_collision"]
    unknown = sorted(set(selected) - set(_scenario_builders()))
    if unknown:
        raise SystemExit(f"unknown scenarios: {unknown}")
    if args.warmup < 0 or args.repetitions <= 0 or args.dt <= 0.0:
        raise SystemExit("warmup must be >= 0; repetitions and dt must be positive")
    metadata = _current_metadata()
    report = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "artifact_kind": "matched_velocity_acceleration_benchmark",
        "measurement_scope": "comparative hot-cache telemetry; not WCET or real-time",
        "created_unix": time.time(),
        "metadata": metadata,
        "parameters": {
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "dt": args.dt,
            "scenario_keys": selected,
        },
        "thresholds": {
            "acceleration_median_overhead_limit": ACCEL_MEDIAN_OVERHEAD_LIMIT,
            "acceleration_p95_overhead_limit": ACCEL_P95_OVERHEAD_LIMIT,
        },
        "capabilities": {
            name: bool(getattr(eik.AccelerationSolver.capabilities(), name))
            for name in dir(eik.AccelerationSolver.capabilities())
            if name.startswith("supports_")
        },
        "scenarios": _run_selected_scenarios(selected, args.warmup, args.repetitions, args.dt),
    }
    _write_immutable_json(args.output, report)
    print(f"Wrote immutable R09 benchmark artifact: {args.output}")
    for name, scenario in report["scenarios"].items():
        v = scenario["velocity"]["timing_ms"]["wall"]
        a = scenario["acceleration"]["timing_ms"]["wall"]
        print(
            f"{name}: velocity_median={v['median']:.4f} ms "
            f"acceleration_median={a['median']:.4f} ms correctness={scenario['correctness']['checks_passed']}"
        )


def _compare_command(args: argparse.Namespace) -> None:
    result = evaluate_artifact(args.candidate)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run matched R09 benchmark scenarios")
    run_parser.add_argument("--warmup", type=int, default=80)
    run_parser.add_argument("--repetitions", type=int, default=600)
    run_parser.add_argument("--dt", type=float, default=0.01)
    run_parser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(_scenario_builders()),
        help="Scenario key to run. Can be repeated; default runs all R09 scenarios.",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Immutable output JSON path. Existing files are refused.",
    )
    run_parser.set_defaults(func=_run_command)

    compare_parser = subparsers.add_parser(
        "compare",
        help="validate one clean R09 artifact against the frozen overhead budgets",
    )
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.set_defaults(func=_compare_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
