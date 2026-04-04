#!/usr/bin/env python3
"""Validate conservative collision-cache behavior against full-scan baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

import embodik as eik

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.utils.robot_models import ensure_ros_package_path

try:
    from robot_descriptions.panda_description import URDF_PATH as PANDA_URDF_PATH
except ImportError as exc:
    raise SystemExit("robot_descriptions is required (panda_description).") from exc


def make_solver(
    enable_cache: bool, refresh_interval: int
) -> tuple[eik.RobotModel, eik.KinematicsSolver, callable]:
    ensure_ros_package_path(Path(PANDA_URDF_PATH))
    robot = eik.RobotModel(str(PANDA_URDF_PATH), floating_base=False)
    q0 = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.05, 0.05], dtype=float)
    robot.update_configuration(q0)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)
    solver.enable_position_limits(True)
    solver.enable_timing_breakdown(True)
    if hasattr(solver, "enable_sphere_broadphase"):
        solver.enable_sphere_broadphase(True)
    if hasattr(solver, "enable_collision_pair_cache"):
        solver.enable_collision_pair_cache(enable_cache, refresh_interval, 0.03, 128)
    solver.configure_collision_constraint(
        min_distance=0.03,
        include_pairs=[],
        exclude_pairs=[],
        nearest_points_all_pairs=False,
        max_constraints=3,
    )
    task = solver.add_frame_task("ee_task", "panda_hand")
    task.priority = 0
    task.weight = 10.0
    start = robot.get_frame_pose("panda_hand")
    base_pos = np.asarray(start.translation, dtype=float)
    base_rot = np.asarray(start.rotation, dtype=float)

    def update(step: int) -> None:
        phase = 0.09 * float(step)
        target = base_pos + np.array(
            [0.04 * math.cos(phase), 0.04 * math.sin(phase), 0.02 * math.sin(0.5 * phase)],
            dtype=float,
        )
        task.set_target_pose(target, base_rot)

    return robot, solver, update


def run_trace(enable_cache: bool, refresh_interval: int, steps: int) -> dict:
    robot, solver, update_target = make_solver(enable_cache, refresh_interval)
    q = np.asarray(robot.get_current_configuration(), dtype=float)
    statuses: list[str] = []
    min_dists: list[float] = []
    collision_ms: list[float] = []
    pairs_considered: list[float] = []
    exact_queries: list[float] = []
    bound_culled: list[float] = []
    sphere_culled: list[float] = []
    budget_exhausted: list[float] = []
    for step in range(steps):
        update_target(step)
        res = solver.solve_velocity(q, apply_limits=True)
        statuses.append(str(res.status))
        dbg = solver.evaluate_collision_debug(q)
        min_dists.append(float(dbg.distance) if dbg is not None else float("inf"))
        collision_ms.append(float(getattr(res, "collision_constraint_time_ms", 0.0)))
        pairs_considered.append(float(getattr(res, "collision_pairs_considered", 0.0)))
        exact_queries.append(float(getattr(res, "collision_exact_distance_queries", 0.0)))
        bound_culled.append(float(getattr(res, "collision_bound_culled_pairs", 0.0)))
        sphere_culled.append(float(getattr(res, "collision_sphere_culled_pairs", 0)))
        budget_exhausted.append(
            1.0 if bool(getattr(res, "collision_budget_exhausted", False)) else 0.0
        )
        dq = np.asarray(res.joint_velocities, dtype=float)
        q = np.asarray(robot.integrate(q, solver.dt * dq), dtype=float)
        robot.update_configuration(q)
    return {
        "statuses": statuses,
        "min_dists": min_dists,
        "collision_time_ms": collision_ms,
        "collision_pairs_considered": pairs_considered,
        "collision_exact_distance_queries": exact_queries,
        "collision_bound_culled_pairs": bound_culled,
        "collision_sphere_culled_pairs": sphere_culled,
        "collision_budget_exhausted": budget_exhausted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate collision cache against full scan baseline.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--refresh-interval", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cursor/reports/perf_runs/collision_cache_validation.json"),
    )
    args = parser.parse_args()

    baseline = run_trace(enable_cache=False, refresh_interval=1, steps=args.steps)
    cached = run_trace(enable_cache=True, refresh_interval=args.refresh_interval, steps=args.steps)

    status_mismatch = sum(1 for a, b in zip(baseline["statuses"], cached["statuses"]) if a != b)
    max_dist_delta = max(
        abs(a - b)
        for a, b in zip(baseline["min_dists"], cached["min_dists"])
        if np.isfinite(a) and np.isfinite(b)
    )
    median_baseline_coll = float(np.median(np.array(baseline["collision_time_ms"], dtype=float)))
    median_cached_coll = float(np.median(np.array(cached["collision_time_ms"], dtype=float)))

    result = {
        "steps": args.steps,
        "refresh_interval": args.refresh_interval,
        "status_mismatch_count": int(status_mismatch),
        "max_abs_min_distance_delta_m": float(max_dist_delta),
        "median_collision_time_ms_baseline": median_baseline_coll,
        "median_collision_time_ms_cached": median_cached_coll,
        "collision_time_speedup_x": (median_baseline_coll / median_cached_coll)
        if median_cached_coll > 1e-9
        else float("inf"),
        "total_pairs_considered_baseline": float(np.sum(np.array(baseline["collision_pairs_considered"], dtype=float))),
        "total_pairs_considered_cached": float(np.sum(np.array(cached["collision_pairs_considered"], dtype=float))),
        "total_exact_distance_queries_baseline": float(np.sum(np.array(baseline["collision_exact_distance_queries"], dtype=float))),
        "total_exact_distance_queries_cached": float(np.sum(np.array(cached["collision_exact_distance_queries"], dtype=float))),
        "total_bound_culled_pairs_baseline": float(np.sum(np.array(baseline["collision_bound_culled_pairs"], dtype=float))),
        "total_bound_culled_pairs_cached": float(np.sum(np.array(cached["collision_bound_culled_pairs"], dtype=float))),
        "median_sphere_culled_pairs_baseline": float(np.median(np.array(baseline["collision_sphere_culled_pairs"], dtype=float))),
        "median_sphere_culled_pairs_cached": float(np.median(np.array(cached["collision_sphere_culled_pairs"], dtype=float))),
        "mean_sphere_culled_pairs_baseline": float(np.mean(np.array(baseline["collision_sphere_culled_pairs"], dtype=float))),
        "mean_sphere_culled_pairs_cached": float(np.mean(np.array(cached["collision_sphere_culled_pairs"], dtype=float))),
        "total_sphere_culled_pairs_baseline": float(np.sum(np.array(baseline["collision_sphere_culled_pairs"], dtype=float))),
        "total_sphere_culled_pairs_cached": float(np.sum(np.array(cached["collision_sphere_culled_pairs"], dtype=float))),
        "budget_exhausted_steps_baseline": float(np.sum(np.array(baseline["collision_budget_exhausted"], dtype=float))),
        "budget_exhausted_steps_cached": float(np.sum(np.array(cached["collision_budget_exhausted"], dtype=float))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
