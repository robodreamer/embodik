#!/usr/bin/env python3
"""Reproduce and diagnose sluggish SCALE_ELASTIC behavior near joint limits.

Compares SCALE, SCALE_ELASTIC, and MIN_ERROR on the same trajectory that
pushes the Panda near its joint limits. Prints per-step diagnostics to
identify where SCALE_ELASTIC is slow.
"""

from __future__ import annotations

import numpy as np
import embodik as eik

_PANDA_DEFAULT_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
_PANDA_GRIPPER_EXTRA = np.array([0.05, 0.05])
_PANDA_EE_FRAME = "panda_hand"


def _load_panda_narrow(margin=0.15):
    from robot_descriptions.panda_description import URDF_PATH
    robot = eik.RobotModel(URDF_PATH, floating_base=False)
    q_init = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
    robot.update_configuration(q_init)

    q_lower_orig, q_upper_orig = robot.get_joint_limits()
    q_lower = q_lower_orig.copy()
    q_upper = q_upper_orig.copy()
    q_center = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
    for i in range(len(_PANDA_DEFAULT_Q)):
        q_lower[i] = max(q_lower_orig[i], q_center[i] - margin)
        q_upper[i] = min(q_upper_orig[i], q_center[i] + margin)
    robot.set_joint_limits(q_lower, q_upper)

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    return robot, solver


def run_mode(mode_name, solve_mode, allow_fallback=False, steps=100,
             offset=None, margin=0.15):
    """Run a trajectory and collect per-step diagnostics."""
    robot, solver = _load_panda_narrow(margin=margin)
    q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
    robot.update_configuration(q)
    robot.update_kinematics(q)

    solver.clear_tasks()
    task = solver.add_frame_task("ee", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSE)
    task.priority = 0
    task.weight = 10.0
    task.solve_mode = solve_mode
    task.allow_min_error_fallback = allow_fallback

    robot.update_kinematics(q)
    ee_pos = np.array(task.current_position)
    ee_rot = np.array(task.current_orientation)
    if offset is None:
        offset = np.array([0.08, 0.0, 0.0])
    task.set_target_pose(ee_pos + offset, ee_rot)

    q_lower, q_upper = robot.get_joint_limits()
    dt = solver.dt

    scales = []
    dq_norms = []
    ee_positions = []
    statuses = []
    saturated_counts = []

    for step in range(steps):
        result = solver.solve_velocity(q)
        dq = result.joint_velocities
        scale = result.task_scales[0] if result.task_scales else 1.0

        scales.append(scale)
        dq_norms.append(float(np.linalg.norm(dq)))
        statuses.append(str(result.status).split(".")[-1])
        saturated_counts.append(len(result.saturated_joints))

        q = robot.integrate(q, dq, dt)
        q = np.clip(q, q_lower, q_upper)
        robot.update_kinematics(q)
        ee_positions.append(np.array(task.current_position).copy())

    solver.clear_tasks()
    if hasattr(solver, 'disable_elastic_band') and solver.elastic_band_enabled():
        solver.disable_elastic_band()

    return {
        "name": mode_name,
        "scales": np.array(scales),
        "dq_norms": np.array(dq_norms),
        "ee_positions": np.array(ee_positions),
        "statuses": statuses,
        "saturated_counts": np.array(saturated_counts),
    }


def print_comparison(results, offset_label=""):
    """Print side-by-side comparison of modes."""
    print(f"\n{'='*90}")
    print(f"Comparison: {offset_label}")
    print(f"{'='*90}")

    # Summary stats
    print(f"\n{'Mode':20s} {'MeanScale':>10s} {'MinScale':>10s} {'MeanDqNorm':>11s} "
          f"{'ZeroScaleN':>11s} {'MeanSat':>8s} {'EE_Dist':>10s}")
    print("-" * 90)
    for r in results:
        ee_dist = float(np.linalg.norm(r["ee_positions"][-1] - r["ee_positions"][0]))
        zero_scale_count = int(np.sum(r["scales"] < 1e-6))
        print(f"{r['name']:20s} {np.mean(r['scales']):10.4f} {np.min(r['scales']):10.4f} "
              f"{np.mean(r['dq_norms']):11.6f} {zero_scale_count:11d} "
              f"{np.mean(r['saturated_counts']):8.1f} {ee_dist:10.4f}")

    # Per-step trace (first 30 steps where differences matter most)
    print(f"\n--- Per-step trace (first 30 steps) ---")
    print(f"{'Step':>4s}", end="")
    for r in results:
        print(f"  {r['name'][:12]:>12s}_scl {r['name'][:12]:>12s}_dq", end="")
    print()

    n_steps = min(30, len(results[0]["scales"]))
    for i in range(n_steps):
        print(f"{i:4d}", end="")
        for r in results:
            print(f"  {r['scales'][i]:16.4f} {r['dq_norms'][i]:15.6f}", end="")
        print()

    # Identify sluggish region: steps where SCALE_ELASTIC has low scale but MIN_ERROR has high dq
    if len(results) >= 3:
        elastic = results[1]  # SCALE_ELASTIC
        minerr = results[2]   # MIN_ERROR
        sluggish_steps = []
        for i in range(len(elastic["scales"])):
            if elastic["scales"][i] < 0.3 and minerr["dq_norms"][i] > 0.1:
                sluggish_steps.append(i)
        if sluggish_steps:
            print(f"\n--- Sluggish steps (elastic scale<0.3, min_error dq>0.1): "
                  f"{len(sluggish_steps)} steps ---")
            print(f"Steps: {sluggish_steps[:20]}{'...' if len(sluggish_steps) > 20 else ''}")


if __name__ == "__main__":
    for margin, offset_label, offset in [
        (0.15, "X +0.08, margin=0.15", np.array([0.08, 0.0, 0.0])),
        (0.15, "Y +0.08, margin=0.15", np.array([0.0, 0.08, 0.0])),
        (0.25, "X +0.10, margin=0.25", np.array([0.10, 0.0, 0.0])),
        (0.10, "X +0.05, margin=0.10 (very narrow)", np.array([0.05, 0.0, 0.0])),
    ]:
        results = [
            run_mode("SCALE", eik.TaskSolveMode.SCALE, steps=100,
                     offset=offset, margin=margin),
            run_mode("SCALE_ELASTIC", eik.TaskSolveMode.SCALE_ELASTIC, steps=100,
                     offset=offset, margin=margin),
            run_mode("MIN_ERROR", eik.TaskSolveMode.SCALE, allow_fallback=True,
                     steps=100, offset=offset, margin=margin),
        ]
        print_comparison(results, offset_label)
