#!/usr/bin/env python3
"""Focused perf gate for solve_position_step collision/stall paths."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np

import embodik as eik

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _ensure_ros_package_path(urdf_path: Path) -> None:
    resolved = urdf_path.resolve()
    current = os.environ.get("ROS_PACKAGE_PATH", "")
    paths = [Path(p) for p in current.split(":") if p]
    updated = False
    for depth in range(1, 5):
        if len(resolved.parents) > depth:
            root = resolved.parents[depth]
            if root.is_dir() and root not in paths:
                paths.insert(0, root)
                updated = True
    if updated:
        os.environ["ROS_PACKAGE_PATH"] = ":".join(str(p) for p in paths)


def _extract_link_index(name: str) -> int | None:
    m = re.search(r"link[_-]?(\d+)", name.lower())
    if m:
        return int(m.group(1))
    return None


def _panda_collision_exclusions(robot: eik.RobotModel) -> list[tuple[str, str]]:
    excl: list[tuple[str, str]] = []
    for a, b in robot.get_collision_pair_names():
        a_l, b_l = a.lower(), b.lower()
        if "finger" in a_l or "finger" in b_l:
            excl.append((a, b))
            continue
        ia, ib = _extract_link_index(a_l), _extract_link_index(b_l)
        if ia is not None and ib is not None and abs(ia - ib) <= 2:
            excl.append((a, b))
    return excl


def _setup_panda_position_step_stall():
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
    solver.enable_stall_handler(0.04)

    q = np.array([0.0, -0.3, 0.0, -2.8, 0.0, 2.5, 0.785, 0.04, 0.04], dtype=float)
    robot.update_configuration(q)

    solver.clear_tasks()
    task = solver.add_frame_task("panda_stall", "panda_hand")
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = eik.TaskSolveMode.SCALE

    hand_pose = robot.get_frame_pose("panda_hand")
    target = np.eye(4, dtype=float)
    target[:3, :3] = np.asarray(hand_pose.rotation, dtype=float)
    target[:3, 3] = np.asarray(hand_pose.translation, dtype=float) + np.array(
        [-0.30, 0.0, -0.25], dtype=float
    )

    opts = eik.PositionStepOptions()
    opts.dt = solver.dt
    opts.max_steps = 1
    opts.stall_recovery = True
    opts.position_gain = 1.0
    opts.orientation_gain = 1.0
    return robot, solver, q, target, "panda_stall", opts


def _setup_dual_iiwa_position_step_stall():
    examples_dir = _REPO_ROOT / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    from utils.dual_iiwa_urdf import build_dual_iiwa_urdf, get_dual_iiwa_frame_names

    urdf_str = build_dual_iiwa_urdf()
    with tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False) as f:
        f.write(urdf_str)
        urdf_path = f.name

    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
    finally:
        os.unlink(urdf_path)

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)
    solver.configure_collision_constraint(
        min_distance=0.35,
        include_pairs=[],
        exclude_pairs=[],
        nearest_points_all_pairs=False,
        max_constraints=3,
    )
    solver.enable_stall_handler(0.35)

    q = np.asarray(robot.get_current_configuration(), dtype=float)
    if q.size == 0:
        q = np.asarray(robot.neutral_configuration(), dtype=float)
    robot.update_configuration(q)
    left_frame, right_frame = get_dual_iiwa_frame_names()
    solver.clear_tasks()
    left_task = solver.add_frame_task("left_body", left_frame)
    left_task.priority = 0
    left_task.weight = 1.0
    left_task.solve_mode = eik.TaskSolveMode.SCALE
    right_task = solver.add_frame_task("right_body", right_frame)
    right_task.priority = 0
    right_task.weight = 1.0
    right_task.solve_mode = eik.TaskSolveMode.SCALE
    left_pose = robot.get_frame_pose(left_frame)
    right_pose = robot.get_frame_pose(right_frame)
    left_target = np.eye(4, dtype=float)
    left_target[:3, :3] = np.asarray(left_pose.rotation, dtype=float)
    left_target[:3, 3] = np.asarray(left_pose.translation, dtype=float) + np.array(
        [0.15, -0.35, -0.10], dtype=float
    )
    right_target = np.eye(4, dtype=float)
    right_target[:3, :3] = np.asarray(right_pose.rotation, dtype=float)
    right_target[:3, 3] = np.asarray(right_pose.translation, dtype=float) + np.array(
        [0.15, 0.35, -0.10], dtype=float
    )
    targets = [eik.TaskTarget("left_body", left_target), eik.TaskTarget("right_body", right_target)]

    opts = eik.PositionStepOptions()
    opts.dt = solver.dt
    opts.max_steps = 1
    opts.stall_recovery = True
    opts.position_gain = 1.0
    opts.orientation_gain = 1.0
    return robot, solver, q, targets, opts


def _run_single_target(steps: int, warmup: int) -> dict[str, float]:
    robot, solver, q, target, task_name, opts = _setup_panda_position_step_stall()
    timings = []
    rejects = 0
    escapes = 0
    for i in range(warmup + steps):
        t0 = time.perf_counter()
        res = solver.solve_position_step(q, target, task_name, opts)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            timings.append(dt_ms)
            rejects += int(getattr(res, "collision_rejection_count", 0))
            escapes += int(getattr(res, "stall_escape_count", 0))
        q = np.asarray(res.q_solution, dtype=float)
        robot.update_configuration(q)
    timings_sorted = sorted(timings)
    return {
        "p50_ms": float(np.median(timings_sorted)),
        "p95_ms": float(timings_sorted[max(0, int(0.95 * len(timings_sorted)) - 1)]),
        "mean_ms": float(np.mean(timings_sorted)),
        "collision_rejections": float(rejects),
        "stall_escapes": float(escapes),
    }


def _run_multi_target(steps: int, warmup: int) -> dict[str, float]:
    try:
        robot, solver, q, targets, opts = _setup_dual_iiwa_position_step_stall()
    except Exception:
        return {"skipped": 1.0}
    timings = []
    rejects = 0
    escapes = 0
    for i in range(warmup + steps):
        t0 = time.perf_counter()
        res = solver.solve_position_step(q, targets, opts)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            timings.append(dt_ms)
            rejects += int(getattr(res, "collision_rejection_count", 0))
            escapes += int(getattr(res, "stall_escape_count", 0))
        q = np.asarray(res.q_solution, dtype=float)
        robot.update_configuration(q)
    timings_sorted = sorted(timings)
    return {
        "p50_ms": float(np.median(timings_sorted)),
        "p95_ms": float(timings_sorted[max(0, int(0.95 * len(timings_sorted)) - 1)]),
        "mean_ms": float(np.mean(timings_sorted)),
        "collision_rejections": float(rejects),
        "stall_escapes": float(escapes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Position-step perf gate benchmark.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".editor/reports/perf_runs/position_step_refactor_gate.json"),
    )
    args = parser.parse_args()

    report = {
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "workloads": {
            "panda_position_step_stall": _run_single_target(args.steps, args.warmup),
            "dual_iiwa_position_step_stall": _run_multi_target(args.steps, args.warmup),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

