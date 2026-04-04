#!/usr/bin/env python3
"""Benchmark collision tuning modes for solve_position_step (Panda).

Mirrors the usage in examples/02_collision_aware_IK.py: Panda robot with
self-collision exclusions, frame task tracking a moving target, all three
collision tuning modes compared.

Expected baselines (from original collision tuning implementation):
  - Speed:    ~0.1ms per step (cache + budget + proximity gating)
  - Balanced: ~0.4-0.5ms per step (cache, no budget)
  - Precise:  ~7-8ms per step (full scan every step)
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np

import embodik as eik


def _ensure_ros_package_path(urdf_path: Path) -> None:
    resolved = urdf_path.resolve()
    current = os.environ.get("ROS_PACKAGE_PATH", "")
    paths = [Path(p) for p in current.split(":") if p]
    for depth in range(1, 5):
        if len(resolved.parents) > depth:
            root = resolved.parents[depth]
            if root.is_dir() and root not in paths:
                paths.insert(0, root)
    os.environ["ROS_PACKAGE_PATH"] = ":".join(str(p) for p in paths)


def _panda_collision_exclusions(robot):
    excl = []
    for a, b in robot.get_collision_pair_names():
        al, bl = a.lower(), b.lower()
        if "finger" in al or "finger" in bl:
            excl.append((a, b))
            continue
        m_a = re.search(r"link[_-]?(\d+)", al)
        m_b = re.search(r"link[_-]?(\d+)", bl)
        if m_a and m_b and abs(int(m_a.group(1)) - int(m_b.group(1))) <= 2:
            excl.append((a, b))
    return excl


def benchmark_mode(mode_label: str, warmup: int = 20, steps: int = 100):
    from robot_descriptions.panda_description import URDF_PATH

    urdf_path = Path(URDF_PATH)
    _ensure_ros_package_path(urdf_path)

    robot = eik.RobotModel(str(urdf_path), floating_base=False)
    exclusions = _panda_collision_exclusions(robot)
    if exclusions:
        try:
            robot.apply_collision_exclusions(exclusions)
        except Exception:
            pass

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)
    solver.configure_collision_constraint(
        min_distance=0.04,
        include_pairs=[],
        exclude_pairs=list(exclusions),
        nearest_points_all_pairs=False,
        max_constraints=3,
    )

    mode_map = {
        "speed": eik.CollisionTuningMode.SPEED,
        "balanced": eik.CollisionTuningMode.BALANCED,
        "precise": eik.CollisionTuningMode.PRECISE,
    }
    solver.set_collision_tuning_mode(mode_map[mode_label])

    q = np.array([0.0, -0.3, 0.0, -2.8, 0.0, 2.5, 0.785, 0.04, 0.04])
    robot.update_configuration(q)

    solver.clear_tasks()
    task = solver.add_frame_task("panda_ee", "panda_hand")
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = eik.TaskSolveMode.SCALE

    hand_pose = robot.get_frame_pose("panda_hand")
    base_pos = np.asarray(hand_pose.translation, dtype=float)
    base_rot = np.asarray(hand_pose.rotation, dtype=float)

    opts = eik.PositionStepOptions()
    opts.dt = solver.dt
    opts.max_steps = 1
    opts.stall_recovery = False
    opts.position_gain = 1.0
    opts.orientation_gain = 1.0

    def make_target(i):
        t = i * 0.05
        tgt = np.eye(4)
        tgt[:3, :3] = base_rot
        tgt[:3, 3] = base_pos + np.array(
            [0.08 * np.cos(t), 0.08 * np.sin(t), 0.02 * np.sin(t * 0.7)]
        )
        return tgt

    # Warmup
    for i in range(warmup):
        target = make_target(i)
        res = solver.solve_position_step(q, target, "panda_ee", opts)
        q = np.asarray(res.q_solution, dtype=float)
        robot.update_configuration(q)

    # Benchmark
    timings = []
    last_queries = 0
    last_pairs = 0
    for i in range(steps):
        target = make_target(warmup + i)
        t0 = time.perf_counter()
        res = solver.solve_position_step(q, target, "panda_ee", opts)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        timings.append(dt_ms)
        last_queries = res.collision_exact_distance_queries
        last_pairs = res.collision_pairs_considered
        q = np.asarray(res.q_solution, dtype=float)
        robot.update_configuration(q)

    return {
        "mode": mode_label,
        "median_ms": float(np.median(timings)),
        "p5_ms": float(np.percentile(timings, 5)),
        "p95_ms": float(np.percentile(timings, 95)),
        "mean_ms": float(np.mean(timings)),
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings)),
        "last_queries": last_queries,
        "last_pairs": last_pairs,
    }


def main():
    print("Collision Tuning Mode Benchmark (Panda, solve_position_step)")
    print("=" * 65)
    print()

    results = {}
    for mode in ["speed", "balanced", "precise"]:
        r = benchmark_mode(mode)
        results[mode] = r
        print(
            f"  {mode:>8s}: median={r['median_ms']:6.2f}ms  "
            f"p5={r['p5_ms']:6.2f}ms  p95={r['p95_ms']:6.2f}ms  "
            f"queries={r['last_queries']}  pairs={r['last_pairs']}"
        )

    print()
    print("Baselines (original tuning mode implementation):")
    print("  speed: ~0.1ms  |  balanced: ~0.4-0.5ms  |  precise: ~7-8ms")
    print()

    speed_ok = results["speed"]["median_ms"] < 3.0
    balanced_ok = results["balanced"]["median_ms"] < 4.0
    differentiation_ok = results["speed"]["median_ms"] < results["precise"]["median_ms"] * 0.5

    print(f"Speed < 3ms:     {'PASS' if speed_ok else 'FAIL'} ({results['speed']['median_ms']:.2f}ms)")
    print(f"Balanced < 4ms:  {'PASS' if balanced_ok else 'FAIL'} ({results['balanced']['median_ms']:.2f}ms)")
    print(f"Speed << Precise: {'PASS' if differentiation_ok else 'FAIL'}")

    if speed_ok and balanced_ok and differentiation_ok:
        print("\nAll checks passed.")
        return 0
    else:
        print("\nSome checks failed.")
        return 1


if __name__ == "__main__":
    exit(main())
