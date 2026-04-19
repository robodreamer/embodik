#!/usr/bin/env python3
"""Benchmark: collision boundary oscillation (bounce) near min_distance.

Sets up Panda with a target that requires the arm to press against a collision
boundary.  The task continuously tries to push the arm through the boundary;
the velocity-damper constraint pushes back.  Measures how much the distance to
the boundary oscillates (sign changes in velocity-toward-boundary) under
different tuning parameters.

Usage:
    pixi run python scripts/benchmark_boundary_oscillation.py
    pixi run python scripts/benchmark_boundary_oscillation.py --single-case deadband=0,recovery_scale=0.2,max_sep=0.15

Output: JSON to scripts/results/boundary_oscillation_<timestamp>.json
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

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import embodik

# ── constants ─────────────────────────────────────────────────────────────────
N_STEPS   = 300          # total steps per trial
WARMUP    = 50           # steps before measuring (let robot settle)
DT        = 0.02         # solver dt
POS_GAIN  = 15.0         # aggressive enough to always push toward boundary
MIN_DIST  = 0.05         # collision min_distance (chosen to create boundary situation)
N_SEEDS   = 3            # number of arm orientations to average over
OZONE_FACTOR = 2.0       # measure oscillations when dist < OZONE_FACTOR * MIN_DIST


def _setup_ros_package_path(urdf_path: pathlib.Path) -> None:
    existing = os.environ.get("ROS_PACKAGE_PATH", "")
    paths: set[str] = set()
    p = urdf_path.parent
    while p != p.parent:
        paths.add(str(p)); p = p.parent
    new_part = ":".join(paths)
    os.environ["ROS_PACKAGE_PATH"] = f"{existing}:{new_part}" if existing else new_part


def _load_robot() -> embodik.RobotModel:
    from robot_descriptions.panda_description import URDF_PATH
    _setup_ros_package_path(pathlib.Path(URDF_PATH))
    return embodik.RobotModel(str(URDF_PATH))


def _make_solver(robot: embodik.RobotModel,
                 deadband: float,
                 recovery_scale: float,
                 max_sep_speed: float) -> embodik.KinematicsSolver:
    solver = embodik.KinematicsSolver(robot)
    solver.dt = DT
    solver.set_damping(0.01)
    solver.set_tolerance(0.1)
    solver.set_collision_tuning_mode(embodik.CollisionTuningMode.BALANCED)
    solver.configure_collision_constraint(min_distance=MIN_DIST, max_constraints=3)
    solver.enable_sphere_broadphase(True)
    # Apply tunable boundary parameters
    solver.set_collision_repulsion_deadband(deadband)
    solver.set_collision_recovery_scale(recovery_scale)
    solver.set_collision_max_separation_speed_nonpenetrating(max_sep_speed)
    ee = solver.add_frame_task("ee_task", "panda_hand")
    ee.priority = 0; ee.weight = 10.0
    return solver


def _boundary_target(robot: embodik.RobotModel, q0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a target that pushes the arm toward the collision boundary."""
    robot.update_configuration(q0)
    hand = robot.get_frame_pose("panda_hand")
    # Push in a random direction — some seeds will hit the boundary, others won't
    # Use a direction biased toward bringing link4/link6 closer together
    direction = rng.standard_normal(3)
    direction /= np.linalg.norm(direction)
    target = np.eye(4)
    target[:3, :3] = np.asarray(hand.rotation)
    target[:3, 3] = np.asarray(hand.translation) + 0.25 * direction
    return target


def _count_oscillations(dist_trace: list[float]) -> int:
    """Count sign changes in velocity-toward-boundary in the measurement zone."""
    osc = 0
    prev_vel = None
    for i in range(1, len(dist_trace)):
        vel = dist_trace[i] - dist_trace[i-1]  # positive = moving away, negative = approaching
        in_zone = dist_trace[i] < OZONE_FACTOR * MIN_DIST
        if in_zone and prev_vel is not None:
            if prev_vel * vel < 0:  # sign change
                osc += 1
        if in_zone:
            prev_vel = vel
    return osc


def _run_trial(robot: embodik.RobotModel,
               solver: embodik.KinematicsSolver,
               rng: np.random.Generator) -> dict[str, Any]:
    """Run N_SEEDS seeds and aggregate oscillation metrics."""
    osc_counts, near_boundary_fractions, violation_counts = [], [], []
    q0_base = np.array([0., -0.785, 0., -2.356, 0., 1.571, 0.785, 0.04, 0.04])

    for _ in range(N_SEEDS):
        target = _boundary_target(robot, q0_base, rng)
        targets = [embodik.TaskTarget("ee_task", target, POS_GAIN, 3.0)]
        opts = embodik.PositionStepOptions()
        opts.max_steps = 1; opts.position_gain = POS_GAIN; opts.dt = DT

        q = q0_base.copy()
        dist_trace = []
        n_violations = 0

        for step in range(N_STEPS):
            result = solver.solve_position_step(q, targets, opts)
            q = np.asarray(result.q_solution)
            if result.status == embodik.SolverStatus.SUCCESS:
                dist = solver.evaluate_min_collision_distance(q)
                if dist is not None:
                    dist_trace.append(float(dist))
                    if dist < MIN_DIST:
                        n_violations += 1

        measure = dist_trace[WARMUP:]
        if not measure:
            continue
        osc_counts.append(_count_oscillations(measure))
        near_boundary_fractions.append(
            sum(1 for d in measure if d < OZONE_FACTOR * MIN_DIST) / len(measure)
        )
        violation_counts.append(n_violations)

    if not osc_counts:
        return {"oscillation_count_mean": 999, "near_boundary_fraction": 0.0,
                "violation_count_mean": 0.0}

    return {
        "oscillation_count_mean":      float(np.mean(osc_counts)),
        "oscillation_count_median":    float(np.median(osc_counts)),
        "near_boundary_fraction":      float(np.mean(near_boundary_fractions)),
        "violation_count_mean":        float(np.mean(violation_counts)),
    }


def parse_single_case(s: str) -> dict[str, float]:
    return {k.strip(): float(v.strip()) for k, v in (p.split("=") for p in s.split(","))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-case", type=str, default=None,
                        help="e.g. 'deadband=0,recovery_scale=0.2,max_sep=0.15'")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    robot = _load_robot()
    rng = np.random.default_rng(args.seed)

    if args.single_case:
        p = parse_single_case(args.single_case)
        solver = _make_solver(robot,
                              deadband=p.get("deadband", 0.003),
                              recovery_scale=p.get("recovery_scale", 0.2),
                              max_sep=p.get("max_sep", 0.15))
        metrics = _run_trial(robot, solver, rng)
        print(json.dumps(metrics, indent=2))
        return

    # Full sweep
    sweep = [
        # (deadband_m, recovery_scale, max_sep_speed_m/s)
        (0.003, 0.20, 0.15),   # baseline (current defaults)
        (0.002, 0.20, 0.15),   # narrower deadband
        (0.001, 0.20, 0.15),   # narrow deadband
        (0.000, 0.20, 0.15),   # no deadband
        (0.000, 0.10, 0.15),   # no deadband + gentler recovery
        (0.000, 0.30, 0.15),   # no deadband + stronger recovery
        (0.000, 0.20, 0.08),   # no deadband + slower max sep
        (0.000, 0.20, 0.25),   # no deadband + faster max sep
        (0.000, 0.10, 0.08),   # no deadband + gentle everything
    ]

    results: dict[str, Any] = {}
    print(f"Sweeping {len(sweep)} cases × {N_SEEDS} seeds...")
    for i, (db, rs, ms) in enumerate(sweep):
        label = f"deadband={db:.3f} recovery={rs:.2f} max_sep={ms:.2f}"
        print(f"  [{i+1}/{len(sweep)}] {label} ...", end="", flush=True)
        solver = _make_solver(robot, db, rs, ms)
        m = _run_trial(robot, solver, rng)
        results[label] = m
        print(f"  osc={m['oscillation_count_mean']:.1f}  "
              f"violations={m['violation_count_mean']:.1f}")

    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"boundary_oscillation_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"config": {"n_steps": N_STEPS, "warmup": WARMUP, "min_dist": MIN_DIST,
                               "pos_gain": POS_GAIN, "n_seeds": N_SEEDS},
                   "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    baseline = results.get(f"deadband=0.003 recovery=0.20 max_sep=0.15", {})
    b_osc = baseline.get("oscillation_count_mean", 1.0)
    print(f"\n{'Case':<50} {'osc':>6} {'vs baseline':>12} {'violations':>11}")
    print("-" * 80)
    for label, m in results.items():
        osc = m["oscillation_count_mean"]
        ratio = osc / b_osc if b_osc > 0 else 1.0
        viol = m["violation_count_mean"]
        print(f"{label:<50} {osc:>6.1f} {ratio:>11.2f}x {viol:>11.1f}")


if __name__ == "__main__":
    main()
