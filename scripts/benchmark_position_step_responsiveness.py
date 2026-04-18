#!/usr/bin/env python3
"""Benchmark: solve_position_step responsiveness for large target jumps.

Measures how many control cycles (at fixed 200 Hz) it takes to reach
position_error < TARGET_ERROR_M after a 0.4 m target jump.

Usage:
    pixi run python scripts/benchmark_position_step_responsiveness.py
    pixi run python scripts/benchmark_position_step_responsiveness.py --single-case adaptive_dt=True,max_scale=5,ref_dist=0.05
    pixi run python scripts/benchmark_position_step_responsiveness.py --no-collision

Output: JSON to scripts/results/position_step_bench_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any

import numpy as np

# ── project path so we can import examples.utils ──────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import embodik

# ── constants ──────────────────────────────────────────────────────────────────
CONTROL_HZ = 200
MAX_CYCLES = 500
TARGET_ERROR_M = 0.01       # convergence threshold
TARGET_JUMP_M = 0.4         # how far the target jumps from initial EE pose
DEFAULT_Q = np.array([0., -0.785, 0., -2.356, 0., 1.571, 0.785, 0.035, 0.035])
MIN_DISTANCE_COLLISION = 0.03

N_SEEDS = 5                  # number of random jump directions to average over


def _setup_ros_package_path(urdf_path: pathlib.Path) -> None:
    existing = os.environ.get("ROS_PACKAGE_PATH", "")
    paths: set[str] = set()
    p = urdf_path.parent
    while p != p.parent:
        paths.add(str(p))
        p = p.parent
    new_part = ":".join(paths)
    os.environ["ROS_PACKAGE_PATH"] = f"{existing}:{new_part}" if existing else new_part


def _load_robot() -> embodik.RobotModel:
    from robot_descriptions.panda_description import URDF_PATH
    urdf_path = pathlib.Path(URDF_PATH)
    _setup_ros_package_path(urdf_path)
    return embodik.RobotModel(str(urdf_path))


def _make_solver(robot: embodik.RobotModel, with_collision: bool) -> embodik.KinematicsSolver:
    solver = embodik.KinematicsSolver(robot)
    solver.dt = 1.0 / CONTROL_HZ
    solver.set_damping(0.01)
    ee_task = solver.add_frame_task("ee_task", "panda_hand")
    ee_task.weight = 10.0
    ee_task.priority = 0

    if with_collision:
        solver.configure_collision_constraint(min_distance=MIN_DISTANCE_COLLISION)
        solver.set_collision_tuning_mode(embodik.CollisionTuningMode.BALANCED)
        solver.enable_sphere_broadphase(True)

    return solver


def _random_target_pose(robot: embodik.RobotModel, q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a target 4x4 pose TARGET_JUMP_M away from the EE at q."""
    robot.update_configuration(q)
    ee_pose = robot.get_frame_pose("panda_hand")
    direction = rng.standard_normal(3)
    direction /= np.linalg.norm(direction)
    target_pos = ee_pose.translation + TARGET_JUMP_M * direction
    target_rot = ee_pose.rotation
    T = np.eye(4)
    T[:3, :3] = target_rot
    T[:3, 3] = target_pos
    return T


def run_case(
    robot: embodik.RobotModel,
    solver: embodik.KinematicsSolver,
    adaptive_dt: bool,
    max_scale: float,
    ref_dist: float,
    position_gain: float,
    with_collision: bool,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Run N_SEEDS trials, return aggregated metrics."""
    approach_cycles_list = []
    final_errors = []
    violation_counts = []
    wall_times = []

    for seed_i in range(N_SEEDS):
        q = DEFAULT_Q.copy()
        target_pose = _random_target_pose(robot, q, rng)
        robot.update_configuration(q)

        opts = embodik.PositionStepOptions()
        opts.position_gain = position_gain
        opts.max_linear_speed = 2.0
        opts.max_steps = 1
        opts.adaptive_dt = adaptive_dt
        opts.adaptive_dt_max_scale = max_scale
        opts.adaptive_dt_reference_distance = ref_dist

        approach_cycle = MAX_CYCLES  # default: didn't converge
        n_violations = 0
        t0 = time.perf_counter()

        for cyc in range(MAX_CYCLES):
            result = solver.solve_position_step(q, target_pose, "ee_task", opts)
            if result.status == embodik.SolverStatus.SUCCESS:
                q = result.q_solution
            elif result.status == embodik.SolverStatus.COLLISION_VIOLATED:
                n_violations += 1
                # q stays at safe position
            # else: kInfeasible etc — also hold q

            if result.position_error < TARGET_ERROR_M and approach_cycle == MAX_CYCLES:
                approach_cycle = cyc + 1

        wall_ms = (time.perf_counter() - t0) * 1000
        approach_cycles_list.append(approach_cycle)
        final_errors.append(result.position_error)
        violation_counts.append(n_violations)
        wall_times.append(wall_ms)

    return {
        "approach_cycles_mean": float(np.mean(approach_cycles_list)),
        "approach_cycles_median": float(np.median(approach_cycles_list)),
        "approach_cycles_list": [int(x) for x in approach_cycles_list],
        "final_error_mean_m": float(np.mean(final_errors)),
        "collision_violation_count_mean": float(np.mean(violation_counts)),
        "wall_ms_per_seed": float(np.mean(wall_times)),
    }


def parse_single_case(s: str) -> dict[str, Any]:
    """Parse 'adaptive_dt=True,max_scale=5,ref_dist=0.05' into a dict."""
    params: dict[str, Any] = {}
    for part in s.split(","):
        k, v = part.split("=")
        k = k.strip()
        v = v.strip()
        if v.lower() in ("true", "false"):
            params[k] = v.lower() == "true"
        else:
            params[k] = float(v)
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark position_step responsiveness")
    parser.add_argument("--single-case", type=str, default=None,
                        help="Run one case: 'adaptive_dt=True,max_scale=5,ref_dist=0.05'")
    parser.add_argument("--no-collision", action="store_true",
                        help="Disable collision constraints for all cases")
    parser.add_argument("--position-gain", type=float, default=10.0)
    args = parser.parse_args()

    robot = _load_robot()
    with_collision = not args.no_collision
    solver = _make_solver(robot, with_collision)
    rng = np.random.default_rng(42)

    if args.single_case:
        params = parse_single_case(args.single_case)
        metrics = run_case(
            robot, solver,
            adaptive_dt=bool(params.get("adaptive_dt", False)),
            max_scale=float(params.get("max_scale", 5.0)),
            ref_dist=float(params.get("ref_dist", 0.05)),
            position_gain=args.position_gain,
            with_collision=with_collision,
            rng=rng,
        )
        print(json.dumps(metrics, indent=2))
        return

    # Full sweep
    sweep_cases = [
        # (adaptive_dt, max_scale, ref_dist)
        (False, 1.0, 0.05),   # baseline
        (True,  2.0, 0.05),
        (True,  5.0, 0.05),
        (True,  5.0, 0.02),
        (True,  5.0, 0.10),
        (True,  8.0, 0.05),
        (True, 10.0, 0.05),
        (True, 10.0, 0.02),
    ]

    results: dict[str, Any] = {}
    print(f"Running {len(sweep_cases)} cases × {N_SEEDS} seeds each...")
    for i, (adt, ms, rd) in enumerate(sweep_cases):
        label = f"adaptive_dt={adt},max_scale={ms},ref_dist={rd}"
        print(f"  [{i+1}/{len(sweep_cases)}] {label} ...", end="", flush=True)
        metrics = run_case(robot, solver, adt, ms, rd, args.position_gain, with_collision, rng)
        results[label] = metrics
        print(f"  approach_cycles_mean={metrics['approach_cycles_mean']:.1f}  "
              f"violations={metrics['collision_violation_count_mean']:.1f}")

    # Save JSON
    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"position_step_bench_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"config": {"control_hz": CONTROL_HZ, "max_cycles": MAX_CYCLES,
                               "target_error_m": TARGET_ERROR_M, "target_jump_m": TARGET_JUMP_M,
                               "n_seeds": N_SEEDS, "with_collision": with_collision,
                               "position_gain": args.position_gain},
                   "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    baseline = results.get("adaptive_dt=False,max_scale=1.0,ref_dist=0.05", {})
    baseline_cycles = baseline.get("approach_cycles_mean", MAX_CYCLES)
    print(f"\n{'Case':<45} {'cycles':>8} {'vs baseline':>12} {'violations':>11}")
    print("-" * 78)
    for label, m in results.items():
        cyc = m["approach_cycles_mean"]
        ratio = cyc / baseline_cycles if baseline_cycles > 0 else 1.0
        viol = m["collision_violation_count_mean"]
        print(f"{label:<45} {cyc:>8.1f} {ratio:>11.2f}x {viol:>11.1f}")


if __name__ == "__main__":
    main()
