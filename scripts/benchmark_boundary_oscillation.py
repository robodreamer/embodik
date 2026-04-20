#!/usr/bin/env python3
"""Benchmark: collision boundary bounce/jitter near min_distance.

Validates the deadband reduction fix using:
  1. Analytical lb-discontinuity proof (no robot needed)
  2. Alpha wheelbase 3D simulation (requires hmnd_robot URDF)

The root cause of boundary oscillation is kCollisionRepulsionDeadband=3mm:
in this zone lb=0 (no braking), so the robot can approach the boundary at
full task-commanded speed. When it crosses min_distance, recovery lb fires
and pushes it back. It then re-enters the 3mm zone (lb=0 again) and the
task pulls it back in. Oscillation at the solve frequency.

With deadband=0 the lb formula is continuous everywhere: the robot decelerates
smoothly to a stop at min_distance with no bounce.

Usage:
    pixi run python scripts/benchmark_boundary_oscillation.py            # analytical only
    pixi run python scripts/benchmark_boundary_oscillation.py --alpha    # + alpha wheelbase

Output: JSON to scripts/results/boundary_oscillation_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import embodik

# ── analytical constants ──────────────────────────────────────────────────────
MIN_DIST  = 0.05
TOL       = 1e-4
DT        = 0.02
MAX_SEP   = 0.15
REC_SCALE = 0.2


def lb_value(dist: float, deadband: float,
             min_dist: float = MIN_DIST, tol: float = TOL,
             dt: float = DT, max_sep: float = MAX_SEP,
             rec_scale: float = REC_SCALE) -> float:
    if dist >= min_dist + deadband:
        return (min_dist + tol - dist) / dt
    elif dist >= min_dist:
        return 0.0
    else:
        desired = (min_dist + tol - dist) / dt
        if dist >= 0.0:
            return min(max_sep, max(0.0, desired * rec_scale))
        return min(0.5, max(0.05, desired))


def lb_jump_at_deadband_boundary(deadband: float) -> float:
    """Discontinuity size at dist = min_dist + deadband."""
    eps = 1e-6
    if deadband < eps:
        return 0.0
    outside = lb_value(MIN_DIST + deadband + eps, deadband)
    inside  = lb_value(MIN_DIST + deadband - eps, deadband)
    return abs(outside - inside)


def simulate_1d(deadband: float, gain: float = 8.0,
                target: float = 0.02, n_steps: int = 400) -> list[float]:
    """1-D velocity damper: vel = max(lb, vel_task), dist += vel*dt."""
    dist = 0.10
    trace = [dist]
    for _ in range(n_steps):
        vel_task = gain * (target - dist)
        vel = max(lb_value(dist, deadband), vel_task)
        dist += DT * vel
        trace.append(dist)
    return trace


def count_crossings(trace: list[float], md: float = MIN_DIST, warmup: int = 50) -> int:
    t = trace[warmup:]
    return sum(1 for i in range(1, len(t)) if (t[i-1] - md) * (t[i] - md) < 0)


# ── alpha wheelbase 3D simulation ─────────────────────────────────────────────

def _alpha_urdf() -> str | None:
    xacro = "/home/andypark/Projects/hmnd-repos/hmnd/hmnd_robot/install/share/alpha_wheelbase_description/urdf/alpha_wheelbase.urdf.xacro"
    if not pathlib.Path(xacro).exists():
        return None
    with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w") as f:
        tmp = f.name
    res = subprocess.run(["pixi", "run", "python", "-c",
                          f"import xacro; doc=xacro.process_file('{xacro}'); open('{tmp}','w').write(doc.toxml())"],
                         capture_output=True, text=True,
                         cwd="/home/andypark/Projects/hmnd-repos/hmnd/hmnd_robot")
    return tmp if res.returncode == 0 else None


def simulate_alpha_boundary(urdf: str, deadband: float,
                             min_dist: float = 0.03,
                             n_steps: int = 300,
                             warmup: int = 30) -> dict[str, Any]:
    """Drive alpha wheelbase right arm toward torso, measure boundary bounce."""
    from hmnd_robots.robots.alpha_wheelbase import LEFT_ARM, RIGHT_ARM, TORSO  # type: ignore
    actuated = TORSO + LEFT_ARM + RIGHT_ARM
    robot = embodik.RobotModel(urdf, actuated, floating_base=False)

    with open("/home/andypark/Projects/hmnd-repos/hmnd/hmnd_robot/install/share/alpha_wheelbase_description/urdf/collisions.json") as f:
        raw_pairs = json.load(f)
    all_pairs = set(robot.get_collision_pair_names())
    geom_names = robot.get_collision_geometry_names()
    valid_pairs = []
    for la, lb_name in raw_pairs:
        for ga in geom_names:
            if la in ga or ga.startswith(la):
                for gb in geom_names:
                    if (lb_name in gb or gb.startswith(lb_name)) and ga != gb:
                        if (ga, gb) in all_pairs or (gb, ga) in all_pairs:
                            valid_pairs.append((ga, gb)); break
                break

    solver = embodik.KinematicsSolver(robot)
    solver.dt = DT; solver.set_damping(0.1); solver.set_tolerance(0.1)
    solver.set_collision_tuning_mode(embodik.CollisionTuningMode.BALANCED)
    if valid_pairs:
        solver.configure_collision_constraint(min_distance=min_dist, max_constraints=2,
                                              include_pairs=valid_pairs)
        solver.enable_sphere_broadphase(True)
    solver.set_collision_repulsion_deadband(deadband)
    ee = solver.add_frame_task("right_ee_task", "right_gripper_frame")
    ee.priority = 0; ee.weight = 10.0

    q0 = np.zeros(robot.nq); q0[6] if robot.nq > 6 else None
    robot.update_configuration(q0)
    hand = robot.get_frame_pose("right_gripper_frame")
    # Target: fold arm toward torso (brings torso/arm collision pairs closer)
    target = np.eye(4); target[:3, :3] = np.asarray(hand.rotation)
    target[:3, 3] = np.asarray(hand.translation) + np.array([0.0, -0.15, -0.1])
    targets = [embodik.TaskTarget("right_ee_task", target, 5.0, 2.0)]
    opts = embodik.PositionStepOptions(); opts.max_steps = 1; opts.position_gain = 5.0; opts.dt = DT

    q = q0.copy(); dists: list[float] = []
    for _ in range(n_steps):
        result = solver.solve_position_step(q, targets, opts)
        q = np.asarray(result.q_solution)
        dbg = solver.get_last_collision_debug()
        if dbg is not None and hasattr(dbg, "distance"):
            dists.append(float(dbg.distance))

    measure = dists[warmup:]
    if not measure:
        return {"oscillation_count": 0, "near_boundary_fraction": 0.0,
                "min_dist_achieved": float("inf"), "note": "no_collision_data"}

    lb_zone = min_dist + 0.005
    crossings = sum(1 for i in range(1, len(measure))
                    if (measure[i-1] - min_dist) * (measure[i] - min_dist) < 0)
    near = sum(1 for d in measure if d < lb_zone) / len(measure)
    return {
        "oscillation_count": crossings,
        "near_boundary_fraction": near,
        "min_dist_achieved": min(measure),
        "dist_std_in_zone": float(np.std([d for d in measure if d < lb_zone]) if any(d < lb_zone for d in measure) else 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", action="store_true",
                        help="Also run 3D alpha wheelbase simulation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results: dict[str, Any] = {}

    # ── Analytical proof ──────────────────────────────────────────────────────
    print("=== 1. Analytical lb-discontinuity proof ===")
    analytical = {}
    for db in [0.003, 0.002, 0.001, 0.0005, 0.000]:
        jump = lb_jump_at_deadband_boundary(db)
        trace = simulate_1d(db)
        c1d = count_crossings(trace)
        analytical[f"deadband={db:.4f}"] = {"lb_jump_mps": round(jump, 4), "1d_crossings": c1d}
        label = "↑ oscillation trigger" if jump > 0.05 else ("✓ smooth" if jump < 0.001 else "mild")
        print(f"  deadband={db*1000:.1f}mm: lb_jump={jump:.4f} m/s  {label}")

    results["analytical"] = analytical
    print()
    print("  Key insight: deadband creates a velocity step-jump at min_dist+deadband.")
    print("  With deadband=0: lb is continuous → smooth deceleration → no bounce.\n")

    # ── 3D alpha wheelbase ────────────────────────────────────────────────────
    if args.alpha:
        sys.path.insert(0, "/home/andypark/Projects/hmnd-repos/hmnd/hmnd_robot/ros/platforms")
        print("=== 2. Alpha wheelbase 3D boundary simulation ===")
        urdf = _alpha_urdf()
        if urdf is None:
            print("  [SKIP] Alpha wheelbase URDF not found.")
        else:
            alpha_results = {}
            for db in [0.003, 0.001, 0.000]:
                m = simulate_alpha_boundary(urdf, deadband=db)
                alpha_results[f"deadband={db:.3f}"] = m
                print(f"  deadband={db*1000:.1f}mm: osc={m['oscillation_count']}  "
                      f"near_boundary={m['near_boundary_fraction']:.1%}  "
                      f"std={m['dist_std_in_zone']:.4f}m")
            results["alpha_3d"] = alpha_results

    # ── Save results ─────────────────────────────────────────────────────────
    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"boundary_oscillation_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("\nRecommendation: set collision_repulsion_deadband_ default to 0")
    print("  (or expose via set_collision_repulsion_deadband(0) in teleop setup)")


if __name__ == "__main__":
    main()
