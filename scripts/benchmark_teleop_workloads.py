#!/usr/bin/env python3
"""Closed-loop teleop-style workload benchmark for EmbodiK.

Runs representative solve_velocity workloads and records timing breakdown stats:
- no collision (single-arm)
- collision enabled (single-arm)
- CoM constraint enabled (single-arm)
- dual-arm frame tracking

Outputs JSON report with mean/median/p95 for each timing field.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import sys

import numpy as np

import embodik as eik

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.utils.dual_iiwa_urdf import build_dual_iiwa_urdf
from examples.utils.robot_models import ensure_ros_package_path

try:
    from robot_descriptions.panda_description import URDF_PATH as PANDA_URDF_PATH
except ImportError as exc:
    raise SystemExit("robot_descriptions is required (panda_description).") from exc


MetricMap = dict[str, list[float]]

COLLISION_CACHE_ENABLED = False
COLLISION_CACHE_REFRESH_INTERVAL = 20
COLLISION_CACHE_DISTANCE_MARGIN = 0.03
COLLISION_CACHE_MAX_CANDIDATES = 128
COLLISION_REFINEMENT_BUDGET_US = 0


@dataclass
class Workload:
    name: str
    teleop_relevant: bool
    builder: Callable[[], tuple[eik.RobotModel, eik.KinematicsSolver, Callable[[int], None]]]


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    sorted_vals = sorted(values)
    p95_idx = max(0, min(len(sorted_vals) - 1, int(math.ceil(0.95 * len(sorted_vals)) - 1)))
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p95": float(sorted_vals[p95_idx]),
    }


def _default_metrics() -> MetricMap:
    return {
        "wall_time_ms": [],
        "pinocchio_kinematics_time_ms": [],
        "collision_constraint_time_ms": [],
        "task_update_time_ms": [],
        "constraint_setup_time_ms": [],
        "solver_computation_time_ms": [],
        "backend_computation_time_ms": [],
        "collision_pairs_considered": [],
        "collision_exact_distance_queries": [],
        "collision_bound_culled_pairs": [],
        "collision_budget_exhausted": [],
    }


def _add_sample(metrics: MetricMap, result: eik.VelocitySolverResult, wall_ms: float) -> None:
    metrics["wall_time_ms"].append(float(wall_ms))
    metrics["pinocchio_kinematics_time_ms"].append(
        float(getattr(result, "pinocchio_kinematics_time_ms", 0.0))
    )
    metrics["collision_constraint_time_ms"].append(
        float(getattr(result, "collision_constraint_time_ms", 0.0))
    )
    metrics["task_update_time_ms"].append(float(getattr(result, "task_update_time_ms", 0.0)))
    metrics["constraint_setup_time_ms"].append(
        float(getattr(result, "constraint_setup_time_ms", 0.0))
    )
    metrics["solver_computation_time_ms"].append(
        float(getattr(result, "solver_computation_time_ms", 0.0))
    )
    metrics["backend_computation_time_ms"].append(float(getattr(result, "computation_time_ms", 0.0)))
    metrics["collision_pairs_considered"].append(
        float(getattr(result, "collision_pairs_considered", 0.0))
    )
    metrics["collision_exact_distance_queries"].append(
        float(getattr(result, "collision_exact_distance_queries", 0.0))
    )
    metrics["collision_bound_culled_pairs"].append(
        float(getattr(result, "collision_bound_culled_pairs", 0.0))
    )
    metrics["collision_budget_exhausted"].append(
        1.0 if bool(getattr(result, "collision_budget_exhausted", False)) else 0.0
    )


def _panda_base_solver() -> tuple[eik.RobotModel, eik.KinematicsSolver, np.ndarray]:
    ensure_ros_package_path(Path(PANDA_URDF_PATH))
    robot = eik.RobotModel(str(PANDA_URDF_PATH), floating_base=False)
    q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.05, 0.05], dtype=float)
    robot.update_configuration(q)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)
    solver.enable_timing_breakdown(True)
    solver.enable_position_limits(True)
    task = solver.add_frame_task("ee_task", "panda_hand")
    task.priority = 0
    task.weight = 10.0
    return robot, solver, q


def _moving_target_controller(
    robot: eik.RobotModel,
    solver: eik.KinematicsSolver,
    base_frame: str,
    radius_xy: float = 0.035,
    amp_z: float = 0.015,
    omega: float = 0.09,
) -> Callable[[int], None]:
    start_pose = robot.get_frame_pose(base_frame)
    base_pos = np.asarray(start_pose.translation, dtype=float)
    base_rot = np.asarray(start_pose.rotation, dtype=float)
    task = solver.get_task("ee_task")

    def update_target(step: int) -> None:
        phase = omega * float(step)
        target_pos = base_pos + np.array(
            [
                radius_xy * math.cos(phase),
                radius_xy * math.sin(phase),
                amp_z * math.sin(0.5 * phase),
            ],
            dtype=float,
        )
        task.set_target_pose(target_pos, base_rot)

    return update_target


def _workload_panda_no_collision() -> tuple[eik.RobotModel, eik.KinematicsSolver, Callable[[int], None]]:
    robot, solver, _ = _panda_base_solver()
    updater = _moving_target_controller(robot, solver, "panda_hand")
    return robot, solver, updater


def _workload_panda_collision() -> tuple[eik.RobotModel, eik.KinematicsSolver, Callable[[int], None]]:
    robot, solver, _ = _panda_base_solver()
    if hasattr(solver, "set_collision_refinement_time_budget_us"):
        solver.set_collision_refinement_time_budget_us(COLLISION_REFINEMENT_BUDGET_US)
    if hasattr(solver, "enable_collision_pair_cache"):
        solver.enable_collision_pair_cache(
            COLLISION_CACHE_ENABLED,
            COLLISION_CACHE_REFRESH_INTERVAL,
            COLLISION_CACHE_DISTANCE_MARGIN,
            COLLISION_CACHE_MAX_CANDIDATES,
        )
    solver.configure_collision_constraint(
        min_distance=0.03,
        include_pairs=[],
        exclude_pairs=[],
        nearest_points_all_pairs=False,
        max_constraints=3,
    )
    updater = _moving_target_controller(robot, solver, "panda_hand", radius_xy=0.04, amp_z=0.02)
    return robot, solver, updater


def _workload_panda_com() -> tuple[eik.RobotModel, eik.KinematicsSolver, Callable[[int], None]]:
    robot, solver, _ = _panda_base_solver()
    square = np.array(
        [
            [-0.20, -0.20],
            [0.20, -0.20],
            [0.20, 0.20],
            [-0.20, 0.20],
        ],
        dtype=float,
    )
    solver.configure_com_constraint(square, margin=0.0, frame_name="world", proximity_fraction=0.3)
    updater = _moving_target_controller(robot, solver, "panda_hand", radius_xy=0.03, amp_z=0.01)
    return robot, solver, updater


def _configure_dual_collision(
    robot: eik.RobotModel,
    solver: eik.KinematicsSolver,
    masked: bool,
) -> None:
    if hasattr(solver, "set_collision_refinement_time_budget_us"):
        solver.set_collision_refinement_time_budget_us(COLLISION_REFINEMENT_BUDGET_US)
    if hasattr(solver, "enable_collision_pair_cache"):
        solver.enable_collision_pair_cache(
            COLLISION_CACHE_ENABLED,
            COLLISION_CACHE_REFRESH_INTERVAL,
            COLLISION_CACHE_DISTANCE_MARGIN,
            COLLISION_CACHE_MAX_CANDIDATES,
        )
    include_pairs: list[tuple[str, str]] = []
    if masked:
        geom_names = list(robot.get_collision_geometry_names())
        left_candidates = [
            name
            for name in geom_names
            if ("left_iiwa_link_6" in name) or ("left_iiwa_link_7" in name)
        ]
        right_candidates = [
            name
            for name in geom_names
            if ("right_iiwa_link_6" in name) or ("right_iiwa_link_7" in name)
        ]
        include_pairs = [(ln, rn) for ln in left_candidates for rn in right_candidates]

    solver.configure_collision_constraint(
        min_distance=0.02,
        include_pairs=include_pairs,
        exclude_pairs=[],
        nearest_points_all_pairs=False,
        max_constraints=4,
    )


def _workload_dual_arm(
    *,
    replace_mesh_collision: bool,
    collision: bool,
    masked_pairs: bool,
) -> tuple[eik.RobotModel, eik.KinematicsSolver, Callable[[int], None]]:
    with tempfile.NamedTemporaryFile(mode="w", suffix="_dual_iiwa.urdf", delete=False) as f:
        f.write(build_dual_iiwa_urdf(replace_mesh_collision=replace_mesh_collision))
        urdf_path = f.name
    ensure_ros_package_path(Path(urdf_path))
    robot = eik.RobotModel(urdf_path, floating_base=False)
    q = np.array(
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
    robot.update_configuration(q)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)
    solver.enable_timing_breakdown(True)
    solver.enable_position_limits(True)
    l_task = solver.add_frame_task("left_task", "iiwa_left_iiwa_link_7")
    r_task = solver.add_frame_task("right_task", "iiwa_right_iiwa_link_7")
    l_task.priority = 0
    r_task.priority = 0
    l_task.weight = 10.0
    r_task.weight = 10.0
    if collision:
        _configure_dual_collision(robot, solver, masked=masked_pairs)

    l_start = robot.get_frame_pose("iiwa_left_iiwa_link_7")
    r_start = robot.get_frame_pose("iiwa_right_iiwa_link_7")
    l_pos = np.asarray(l_start.translation, dtype=float)
    r_pos = np.asarray(r_start.translation, dtype=float)
    l_rot = np.asarray(l_start.rotation, dtype=float)
    r_rot = np.asarray(r_start.rotation, dtype=float)

    def updater(step: int) -> None:
        phase = 0.07 * float(step)
        l_target = l_pos + np.array([0.03 * math.cos(phase), 0.0, 0.015 * math.sin(phase)], dtype=float)
        r_target = r_pos + np.array([-0.03 * math.cos(phase), 0.0, 0.015 * math.sin(phase)], dtype=float)
        l_task.set_target_pose(l_target, l_rot)
        r_task.set_target_pose(r_target, r_rot)

    return robot, solver, updater


def _workload_dual_arm_tracking() -> tuple[eik.RobotModel, eik.KinematicsSolver, Callable[[int], None]]:
    return _workload_dual_arm(
        replace_mesh_collision=True,
        collision=False,
        masked_pairs=False,
    )


def _workload_dual_arm_collision_mesh() -> tuple[eik.RobotModel, eik.KinematicsSolver, Callable[[int], None]]:
    return _workload_dual_arm(
        replace_mesh_collision=False,
        collision=True,
        masked_pairs=False,
    )


def _workload_dual_arm_collision_simplified() -> tuple[eik.RobotModel, eik.KinematicsSolver, Callable[[int], None]]:
    return _workload_dual_arm(
        replace_mesh_collision=True,
        collision=True,
        masked_pairs=False,
    )


def _workload_dual_arm_collision_simplified_masked() -> tuple[eik.RobotModel, eik.KinematicsSolver, Callable[[int], None]]:
    return _workload_dual_arm(
        replace_mesh_collision=True,
        collision=True,
        masked_pairs=True,
    )


WORKLOADS: list[Workload] = [
    Workload("panda_no_collision", teleop_relevant=True, builder=_workload_panda_no_collision),
    Workload("panda_collision", teleop_relevant=True, builder=_workload_panda_collision),
    Workload("panda_com_constraint", teleop_relevant=True, builder=_workload_panda_com),
    Workload("dual_arm_tracking", teleop_relevant=False, builder=_workload_dual_arm_tracking),
    Workload(
        "dual_arm_collision_mesh",
        teleop_relevant=False,
        builder=_workload_dual_arm_collision_mesh,
    ),
    Workload(
        "dual_arm_collision_simplified",
        teleop_relevant=False,
        builder=_workload_dual_arm_collision_simplified,
    ),
    Workload(
        "dual_arm_collision_simplified_masked",
        teleop_relevant=False,
        builder=_workload_dual_arm_collision_simplified_masked,
    ),
]


def run_workload(workload: Workload, warmup: int, steps: int) -> dict:
    robot, solver, update_target = workload.builder()
    q = np.asarray(robot.get_current_configuration(), dtype=float)
    metrics = _default_metrics()
    status_counts: dict[str, int] = {}

    total_steps = warmup + steps
    for step in range(total_steps):
        update_target(step)
        t0 = time.perf_counter()
        result = solver.solve_velocity(q, apply_limits=True)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        status = str(getattr(result, "status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

        dq = np.asarray(result.joint_velocities, dtype=float)
        q = np.asarray(robot.integrate(q, solver.dt * dq), dtype=float)
        robot.update_configuration(q)

        if step >= warmup:
            _add_sample(metrics, result, wall_ms)

    return {
        "teleop_relevant": workload.teleop_relevant,
        "samples": steps,
        "status_counts": status_counts,
        "stats_ms": {name: _stats(values) for name, values in metrics.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark teleop-style EmbodiK workloads.")
    parser.add_argument("--warmup", type=int, default=80, help="Warmup steps per workload.")
    parser.add_argument("--steps", type=int, default=600, help="Measured steps per workload.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cursor/reports/perf_runs/baseline_teleop_workloads.json"),
        help="Output JSON path.",
    )
    parser.add_argument("--collision-cache", action="store_true", help="Enable conservative collision pair cache.")
    parser.add_argument("--cache-refresh-interval", type=int, default=20, help="Full refresh interval for collision cache.")
    parser.add_argument("--cache-distance-margin", type=float, default=0.03, help="Candidate retention margin above min_distance.")
    parser.add_argument("--cache-max-candidates", type=int, default=128, help="Maximum cached collision candidates.")
    parser.add_argument(
        "--collision-refinement-budget-us",
        type=int,
        default=0,
        help="Optional exact collision refinement budget per solve in microseconds (<=0 disables).",
    )
    args = parser.parse_args()

    global COLLISION_CACHE_ENABLED
    global COLLISION_CACHE_REFRESH_INTERVAL
    global COLLISION_CACHE_DISTANCE_MARGIN
    global COLLISION_CACHE_MAX_CANDIDATES
    global COLLISION_REFINEMENT_BUDGET_US
    COLLISION_CACHE_ENABLED = args.collision_cache
    COLLISION_CACHE_REFRESH_INTERVAL = args.cache_refresh_interval
    COLLISION_CACHE_DISTANCE_MARGIN = args.cache_distance_margin
    COLLISION_CACHE_MAX_CANDIDATES = args.cache_max_candidates
    COLLISION_REFINEMENT_BUDGET_US = max(0, int(args.collision_refinement_budget_us))

    report: dict[str, object] = {
        "timestamp_unix": time.time(),
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "collision_cache_enabled": COLLISION_CACHE_ENABLED,
        "collision_cache_refresh_interval": COLLISION_CACHE_REFRESH_INTERVAL,
        "collision_cache_distance_margin": COLLISION_CACHE_DISTANCE_MARGIN,
        "collision_cache_max_candidates": COLLISION_CACHE_MAX_CANDIDATES,
        "collision_refinement_budget_us": COLLISION_REFINEMENT_BUDGET_US,
        "workloads": {},
    }
    for workload in WORKLOADS:
        report["workloads"][workload.name] = run_workload(workload, args.warmup, args.steps)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Wrote benchmark report: {args.output}")
    for name, data in report["workloads"].items():
        wall = data["stats_ms"]["wall_time_ms"]["median"]
        coll = data["stats_ms"]["collision_constraint_time_ms"]["median"]
        solv = data["stats_ms"]["solver_computation_time_ms"]["median"]
        print(f"{name:24s} median_wall={wall:7.4f} ms  collision={coll:7.4f} ms  solver={solv:7.4f} ms")


if __name__ == "__main__":
    main()
