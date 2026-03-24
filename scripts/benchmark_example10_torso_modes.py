#!/usr/bin/env python3
"""Benchmark responsiveness impact of example-10 torso modes.

Compares floating-base Panda solve_position behavior across:
1) EE-only (torso secondary off)
2) Torso secondary only (upright)
3) Torso secondary + pose bounds
4) Torso secondary + pose bounds with epsilon-lock (trans/rot locked)

Reports status distribution, solve timing, and tracking-error stats.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    v = sorted(values)
    p95_idx = max(0, min(len(v) - 1, int(math.ceil(0.95 * len(v)) - 1)))
    p99_idx = max(0, min(len(v) - 1, int(math.ceil(0.99 * len(v)) - 1)))
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p95": float(v[p95_idx]),
        "p99": float(v[p99_idx]),
        "max": float(v[-1]),
    }


def _parse_int_list(csv: str) -> list[int]:
    out: list[int] = []
    for token in csv.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def _resolve_torso_frame(robot: eik.RobotModel, ee_frame: str) -> str:
    candidates = ("torso", "torso_link", "base_link", "panda_link0", "pelvis", "root_link")
    frame_names = set(robot.get_frame_names())
    for name in candidates:
        if name in frame_names and name != ee_frame:
            return name
    for name in robot.get_frame_names():
        lname = name.lower()
        if ("torso" in lname or "base" in lname or "pelvis" in lname) and name != ee_frame:
            return name
    for name in robot.get_frame_names():
        if name != ee_frame:
            return name
    return ee_frame


def _init_floating_panda() -> tuple[eik.RobotModel, eik.KinematicsSolver, np.ndarray]:
    ensure_ros_package_path(Path(PANDA_URDF_PATH))
    robot = eik.RobotModel(str(PANDA_URDF_PATH), floating_base=True)
    q = np.asarray(robot.get_current_configuration(), dtype=float)
    q[:7] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
    q[-9:] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.05, 0.05], dtype=float)
    robot.update_configuration(q)
    solver = eik.KinematicsSolver(robot)
    return robot, solver, q


@dataclass
class ModeCfg:
    name: str
    torso_secondary: bool
    pose_box: bool
    lock_trans: bool
    lock_rot: bool
    softening_enabled: bool = False
    softening_fraction: float = 0.10
    use_base_joint_lock_optimization: bool = False
    disable_torso_constraint_when_locked: bool = False


MODES: list[ModeCfg] = [
    ModeCfg("ee_only", torso_secondary=False, pose_box=False, lock_trans=False, lock_rot=False),
    ModeCfg("torso_secondary_only", torso_secondary=True, pose_box=False, lock_trans=False, lock_rot=False),
    ModeCfg("torso_secondary_plus_box", torso_secondary=True, pose_box=True, lock_trans=False, lock_rot=False),
    ModeCfg(
        "torso_secondary_plus_box_softened",
        torso_secondary=True,
        pose_box=True,
        lock_trans=False,
        lock_rot=False,
        softening_enabled=True,
        softening_fraction=0.10,
    ),
    ModeCfg("torso_secondary_box_eps_locked", torso_secondary=True, pose_box=True, lock_trans=True, lock_rot=True),
    ModeCfg(
        "torso_secondary_box_eps_locked_base_lock",
        torso_secondary=True,
        pose_box=True,
        lock_trans=True,
        lock_rot=True,
        use_base_joint_lock_optimization=True,
        disable_torso_constraint_when_locked=True,
    ),
]


def _target_pose_homogeneous(step: int, base_h: np.ndarray, jump_every: int) -> np.ndarray:
    t = 0.025 * float(step)
    out = np.array(base_h, dtype=float, order="F")
    # Aggressive but continuous Lissajous-like motion.
    out[0, 3] += 0.17 * math.cos(t)
    out[1, 3] += 0.12 * math.sin(0.7 * t)
    out[2, 3] += 0.06 * math.sin(0.45 * t)
    # Optional deterministic jump to emulate fast target dragging.
    if jump_every > 0 and step > 0 and (step % jump_every == 0):
        out[0, 3] += 0.10
        out[1, 3] -= 0.06
    return out


def run_mode(
    mode: ModeCfg, warmup: int, steps: int, dt: float, jump_every: int, max_iterations: int
) -> dict[str, Any]:
    robot, solver, q = _init_floating_panda()
    solver.dt = dt
    solver.set_damping(0.01)

    ee_frame = "panda_hand"
    torso_frame = _resolve_torso_frame(robot, ee_frame)
    torso_ref_pose = robot.get_frame_pose(torso_frame)
    torso_ref_h = np.asarray(torso_ref_pose.homogeneous(), dtype=float)
    base_target_h = np.asarray(robot.get_frame_pose(ee_frame).homogeneous(), dtype=float, order="F")

    q_lower, q_upper = robot.get_joint_limits()
    nullspace_active = list(range(6, robot.nv)) if robot.nv > 6 else list(range(robot.nv))
    last_feasible_q = q.copy()
    prev_position_error = float("inf")

    statuses: dict[str, int] = {}
    timing_ms: list[float] = []
    pos_err_mm: list[float] = []
    iterations_used: list[float] = []
    timing_by_status: dict[str, list[float]] = {}
    pos_err_by_status: dict[str, list[float]] = {}
    iterations_by_status: dict[str, list[float]] = {}
    accepted_updates = 0
    accepted_improved_non_success = 0
    held_non_success = 0

    total = warmup + steps
    for i in range(total):
        target_h = _target_pose_homogeneous(i, base_target_h, jump_every=jump_every)

        opts = eik.PositionIKOptions()
        opts.max_iterations = int(max_iterations)
        opts.dt = dt
        opts.position_gain = 10.0
        opts.orientation_gain = 10.0
        opts.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
        opts.primary_allow_min_error_fallback = True
        opts.classify_stagnation_as_no_progress = False
        opts.limit_change_from_seed = False
        opts.stagnation_iterations = 10
        opts.nullspace_bias = q.copy()
        opts.nullspace_gain = 0.002
        opts.nullspace_active_joints = nullspace_active
        opts.nullspace_joint_weights = np.ones(len(nullspace_active), dtype=float)

        full_eps_lock = bool(mode.lock_trans and mode.lock_rot)
        bypass_torso_constraint = (
            mode.use_base_joint_lock_optimization
            and mode.disable_torso_constraint_when_locked
            and full_eps_lock
            and robot.nv >= 6
        )
        if mode.torso_secondary and (not bypass_torso_constraint):
            opts.torso_constraint.enabled = True
            opts.torso_constraint.frame_name = torso_frame
            disable_torso_secondary = (
                mode.use_base_joint_lock_optimization and full_eps_lock and robot.nv >= 6
            )
            if disable_torso_secondary:
                opts.torso_constraint.orientation_mask = np.zeros(3, dtype=float)
                opts.torso_constraint.orientation_gain = 0.0
            else:
                opts.torso_constraint.target_orientation = np.asarray(
                    torso_ref_pose.rotation, dtype=float
                )
                opts.torso_constraint.orientation_mask = np.array([1.0, 1.0, 0.0], dtype=float)
                opts.torso_constraint.orientation_gain = 0.05

        if mode.torso_secondary and mode.pose_box and (not bypass_torso_constraint):
            half = np.array(
                [0.10, 0.10, 0.10, np.deg2rad(15.0), np.deg2rad(15.0), np.deg2rad(15.0)],
                dtype=float,
            )
            if mode.lock_trans:
                half[:3] = 1e-4
            if mode.lock_rot:
                half[3:] = 1e-3
            if mode.use_base_joint_lock_optimization and full_eps_lock and robot.nv >= 6:
                opts.excluded_joint_indices = list(range(6))
            else:
                opts.torso_constraint.pose_lower_bounds = -half
                opts.torso_constraint.pose_upper_bounds = half
                opts.torso_constraint.pose_axis_mask = np.ones(6, dtype=float)
                opts.torso_constraint.velocity_limits = np.full(6, 0.8, dtype=float)
                opts.torso_constraint.acceleration_limits = np.full(6, 1.0, dtype=float)
                opts.torso_constraint.pose_bound_softening_enabled = bool(mode.softening_enabled)
                opts.torso_constraint.pose_bound_softening_fraction = float(mode.softening_fraction)
                opts.torso_constraint.pose_bounds_reference_pose = torso_ref_h
        elif bypass_torso_constraint and robot.nv >= 6:
            opts.excluded_joint_indices = list(range(6))

        t0 = time.perf_counter()
        out = solver.solve_position(q, target_h, ee_frame, opts)
        ms = (time.perf_counter() - t0) * 1000.0

        st = out.status.name
        statuses[st] = statuses.get(st, 0) + 1

        if out.status == eik.SolverStatus.SUCCESS:
            q_next = np.asarray(out.q_solution, dtype=float).copy()
            quat = q_next[3:7]
            n = float(np.linalg.norm(quat))
            if n > 1e-12:
                q_next[3:7] = quat / n
            q_next[7:] = np.clip(q_next[7:], q_lower[7:], q_upper[7:])
            q[:] = q_next
            last_feasible_q[:] = q
            prev_position_error = float(out.position_error)
            accepted_updates += 1
        elif out.status in (eik.SolverStatus.INFEASIBLE, eik.SolverStatus.NO_PROGRESS):
            if mode.use_base_joint_lock_optimization:
                q_next = np.asarray(out.q_solution, dtype=float).copy()
                quat = q_next[3:7]
                n = float(np.linalg.norm(quat))
                if n > 1e-12:
                    q_next[3:7] = quat / n
                q_next[7:] = np.clip(q_next[7:], q_lower[7:], q_upper[7:])
                q[:] = q_next
                last_feasible_q[:] = q
                prev_position_error = float(out.position_error)
                accepted_updates += 1
                accepted_improved_non_success += 1
            else:
                improved = float(out.position_error) < (prev_position_error - 1e-6)
                if improved:
                    q_next = np.asarray(out.q_solution, dtype=float).copy()
                    quat = q_next[3:7]
                    n = float(np.linalg.norm(quat))
                    if n > 1e-12:
                        q_next[3:7] = quat / n
                    q_next[7:] = np.clip(q_next[7:], q_lower[7:], q_upper[7:])
                    q[:] = q_next
                    last_feasible_q[:] = q
                    prev_position_error = float(out.position_error)
                    accepted_updates += 1
                    accepted_improved_non_success += 1
                else:
                    q[:] = last_feasible_q
                    held_non_success += 1
        else:
            q[:] = last_feasible_q
            held_non_success += 1

        robot.update_configuration(q)

        if i >= warmup:
            timing_ms.append(float(ms))
            pos_err_mm.append(float(out.position_error * 1e3))
            iterations_used.append(float(out.iterations_used))
            timing_by_status.setdefault(st, []).append(float(ms))
            pos_err_by_status.setdefault(st, []).append(float(out.position_error * 1e3))
            iterations_by_status.setdefault(st, []).append(float(out.iterations_used))

    samples = max(1, len(timing_ms))
    all_statuses = ("SUCCESS", "INFEASIBLE", "NO_PROGRESS", "NUMERICAL_ERROR")
    status_counts = {name: int(statuses.get(name, 0)) for name in all_statuses}
    status_timing_ms = {name: _stats(timing_by_status.get(name, [])) for name in all_statuses}
    status_position_error_mm = {
        name: _stats(pos_err_by_status.get(name, [])) for name in all_statuses
    }
    status_iterations = {
        name: _stats(iterations_by_status.get(name, [])) for name in all_statuses
    }
    iteration_histogram: dict[str, int] = {}
    for it in iterations_used:
        key = str(int(round(it)))
        iteration_histogram[key] = iteration_histogram.get(key, 0) + 1
    return {
        "mode": mode.name,
        "samples": samples,
        "status_counts": status_counts,
        "timing_ms": _stats(timing_ms),
        "position_error_mm": _stats(pos_err_mm),
        "iterations_used": _stats(iterations_used),
        "iterations_histogram": iteration_histogram,
        "status_timing_ms": status_timing_ms,
        "status_position_error_mm": status_position_error_mm,
        "status_iterations_used": status_iterations,
        "acceptance": {
            "accepted_updates": accepted_updates,
            "accepted_rate": float(accepted_updates / total),
            "accepted_improved_non_success": accepted_improved_non_success,
            "held_non_success": held_non_success,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=int, default=80)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--jump-every", type=int, default=120, help="Inject target jump every N steps (0 disables).")
    p.add_argument(
        "--jump-values",
        type=str,
        default="0,120",
        help="CSV of jump periods for matrix sweep (e.g. '0,120').",
    )
    p.add_argument(
        "--max-iterations-values",
        type=str,
        default="1,3,5,10",
        help="CSV of max_iterations values for matrix sweep.",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Baseline max_iterations value (kept for backward-compatible top-level modes output).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(".editor/reports/perf_runs/example10_torso_modes.json"),
    )
    args = p.parse_args()

    report: dict[str, Any] = {
        "timestamp_unix": time.time(),
        "warmup": args.warmup,
        "steps": args.steps,
        "dt": args.dt,
        "jump_every": args.jump_every,
        "max_iterations": args.max_iterations,
        "modes": {},
    }
    jump_values = _parse_int_list(args.jump_values)
    max_iter_values = _parse_int_list(args.max_iterations_values)
    if not jump_values:
        jump_values = [args.jump_every]
    if not max_iter_values:
        max_iter_values = [args.max_iterations]
    for mode in MODES:
        report["modes"][mode.name] = run_mode(
            mode,
            warmup=args.warmup,
            steps=args.steps,
            dt=args.dt,
            jump_every=args.jump_every,
            max_iterations=args.max_iterations,
        )

    # simple overhead table vs ee_only
    base = report["modes"]["ee_only"]["timing_ms"]["mean"]
    overhead = {}
    for name, data in report["modes"].items():
        m = data["timing_ms"]["mean"]
        overhead[name] = 0.0 if base <= 1e-12 else 100.0 * (m - base) / base
    report["mean_timing_overhead_percent_vs_ee_only"] = overhead

    # Matrix sweep for jump-vs-no-jump and max_iterations.
    matrix: dict[str, Any] = {}
    for jump_every in jump_values:
        jump_key = f"jump_{jump_every}"
        matrix[jump_key] = {}
        for max_iter in max_iter_values:
            iter_key = f"iter_{max_iter}"
            run_entry: dict[str, Any] = {"modes": {}, "mean_timing_overhead_percent_vs_ee_only": {}}
            for mode in MODES:
                run_entry["modes"][mode.name] = run_mode(
                    mode,
                    warmup=args.warmup,
                    steps=args.steps,
                    dt=args.dt,
                    jump_every=jump_every,
                    max_iterations=max_iter,
                )
            sweep_base = run_entry["modes"]["ee_only"]["timing_ms"]["mean"]
            for mode_name, mode_data in run_entry["modes"].items():
                m = mode_data["timing_ms"]["mean"]
                run_entry["mean_timing_overhead_percent_vs_ee_only"][mode_name] = (
                    0.0 if sweep_base <= 1e-12 else 100.0 * (m - sweep_base) / sweep_base
                )
            matrix[jump_key][iter_key] = run_entry
    report["matrix"] = matrix

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Wrote report: {args.output}")
    for name, data in report["modes"].items():
        t = data["timing_ms"]["mean"]
        e = data["position_error_mm"]["mean"]
        s = data["status_counts"]
        print(f"{name:32s} mean_ms={t:7.4f}  mean_pos_err_mm={e:8.3f}  statuses={s}")


if __name__ == "__main__":
    main()

