#!/usr/bin/env python3
"""Local-only alpha wheelbase stall reproducer.

This is intentionally a scratch harness for investigation and should not be
committed. It mirrors the hmnd alpha teleop call pattern closely:
  - solve_position_step with PositionStepOptions(stall_recovery=...)
  - optionally update q only when status == SUCCESS
  - curated collision pairs from collisions.json
  - configurable collision tuning mode + max_constraints
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import embodik as eik


RIGHT_ARM_SEED = {
    "right_shoulder_pitch_joint": 0.269,
    "right_shoulder_roll_joint": -0.180,
    "right_shoulder_yaw_joint": 0.441,
    "right_elbow_pitch_joint": -0.326,
    "right_elbow_yaw_joint": 0.053,
    "right_wrist_pitch_joint": -0.238,
    "right_wrist_roll_joint": -0.436,
}


@dataclass
class CaseConfig:
    name: str
    stall_recovery: bool
    max_constraints: int
    tuning_mode: str
    success_only_update: bool


def _geom_to_link_and_idx(geom_name: str) -> tuple[str, int]:
    match = re.match(r"^(.+?)_(\d+)$", geom_name)
    if match:
        return match.group(1), int(match.group(2))
    return geom_name, 0


def _canonical(a: str, b: str) -> str:
    return f"{a}|{b}" if a <= b else f"{b}|{a}"


def _load_link_pairs(path: Path) -> list[tuple[str, str]]:
    raw = json.loads(path.read_text())
    return [(str(item[0]), str(item[1])) for item in raw if len(item) >= 2]


def _map_link_pairs_to_geometry_pairs(
    robot: eik.RobotModel, link_pairs: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    key_to_candidates: dict[str, list[tuple[str, str, int]]] = {}
    for ga, gb in robot.get_collision_pair_names():
        la, ia = _geom_to_link_and_idx(ga)
        lb, ib = _geom_to_link_and_idx(gb)
        key = _canonical(la, lb)
        key_to_candidates.setdefault(key, []).append((ga, gb, ia + ib))

    result: list[tuple[str, str]] = []
    for la, lb in link_pairs:
        key = _canonical(la, lb)
        candidates = key_to_candidates.get(key)
        if not candidates:
            continue
        best = min(candidates, key=lambda x: x[2])
        result.append((best[0], best[1]))
    return result


def _set_tuning_mode(solver: eik.KinematicsSolver, label: str) -> None:
    mode = label.strip().lower()
    if not hasattr(solver, "set_collision_tuning_mode"):
        return
    enum_map = {
        "precision": eik.CollisionTuningMode.PRECISE,
        "precise": eik.CollisionTuningMode.PRECISE,
        "balanced": eik.CollisionTuningMode.BALANCED,
        "speed": eik.CollisionTuningMode.SPEED,
    }
    solver.set_collision_tuning_mode(enum_map.get(mode, eik.CollisionTuningMode.BALANCED))


def _seed_right_arm_q(robot: eik.RobotModel, q: np.ndarray) -> np.ndarray:
    q_seed = np.array(q, dtype=float, copy=True)
    for name, value in RIGHT_ARM_SEED.items():
        if not robot.has_joint(name):
            continue
        idx = int(robot.get_joint_config_index(name))
        q_seed[idx] = float(value)
    return q_seed


def _pick_right_ee_frame(robot: eik.RobotModel) -> str:
    candidates = [
        "right_finger_frame",
        "right_hand_pad_link",
        "right_gripper_tip_link",
        "right_wrist_roll_link",
    ]
    for name in candidates:
        if robot.has_frame(name):
            return name
    raise RuntimeError("Could not find a right EE frame candidate")


def _status_name(status: object) -> str:
    try:
        return str(status).split(".")[-1]
    except Exception:
        return str(status)


def run_case(
    *,
    urdf_path: Path,
    collisions_json: Path,
    min_distance: float,
    dt: float,
    damping: float,
    tolerance: float,
    steps: int,
    offset_xyz: np.ndarray,
    case: CaseConfig,
) -> dict[str, float]:
    robot = eik.RobotModel(str(urdf_path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = float(dt)
    solver.set_damping(float(damping))
    solver.set_tolerance(float(tolerance))
    _set_tuning_mode(solver, case.tuning_mode)

    link_pairs = _load_link_pairs(collisions_json)
    geom_pairs = _map_link_pairs_to_geometry_pairs(robot, link_pairs)
    print(f"[{case.name}] mapped {len(geom_pairs)} / {len(link_pairs)} collision pairs")
    solver.configure_collision_constraint(
        min_distance=float(min_distance),
        include_pairs=geom_pairs,
        exclude_pairs=[],
        max_constraints=int(case.max_constraints),
    )
    if case.stall_recovery:
        solver.enable_stall_handler(float(min_distance))
    else:
        solver.disable_stall_handler()

    ee_frame = _pick_right_ee_frame(robot)
    task_name = "right_ee_task"
    task = solver.add_frame_task(task_name, ee_frame)
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = eik.TaskSolveMode.SCALE

    q = np.asarray(robot.get_current_configuration(), dtype=float)
    q = _seed_right_arm_q(robot, q)
    robot.update_configuration(q)

    pose = robot.get_frame_pose(ee_frame)
    target_pos = np.array(pose.translation, dtype=float) + offset_xyz
    target_rot = np.array(pose.rotation, dtype=float)
    target = eik.TaskTarget.from_se3(task_name, eik.Rt(R=target_rot, t=target_pos), 20.0, 20.0)

    opts = eik.PositionStepOptions()
    opts.stall_recovery = bool(case.stall_recovery)
    opts.max_steps = 1
    opts.dt = float(dt)

    q_min, q_max = robot.get_joint_limits()
    clamp_margin = 5e-4

    worst_collision = np.inf
    min_stall_margin = np.inf
    success_count = 0
    infeasible_count = 0
    stall_count = 0

    print(
        f"[{case.name}] start: stall={case.stall_recovery}, K={case.max_constraints}, "
        f"mode={case.tuning_mode}, success_only={case.success_only_update}"
    )
    for step in range(steps):
        result = solver.solve_position_step(q, [target], opts)
        dq_norm = float(np.linalg.norm(np.asarray(result.joint_velocities, dtype=float)))
        status = result.status
        status_name = _status_name(status)

        if "SUCCESS" in status_name:
            success_count += 1
            q_next = np.asarray(result.q_solution, dtype=float)
            if case.success_only_update:
                q = q_next
            else:
                q = q_next
        else:
            if "INFEASIBLE" in status_name:
                infeasible_count += 1
            if not case.success_only_update:
                q = np.asarray(result.q_solution, dtype=float)

        if dq_norm < 1e-5:
            stall_count += 1

        q = np.clip(q, q_min, q_max)
        robot.update_configuration(q)

        dbg = solver.evaluate_collision_debug(q)
        d = float(dbg.distance) if dbg is not None else float("inf")
        worst_collision = min(worst_collision, d)
        cur_stall_margin = float(solver.stall_handler_current_min_distance())
        min_stall_margin = min(min_stall_margin, cur_stall_margin)

        active_rows = solver.get_last_collision_debug_list()
        row_ds = [float(row.distance) for row in active_rows]
        tight_rows = sum(1 for x in row_ds if x <= cur_stall_margin + 1e-9)

        near_limit = int(
            np.count_nonzero(
                ((q - q_min) > 0.0) & ((q - q_min) < clamp_margin)
                | ((q_max - q) > 0.0) & ((q_max - q) < clamp_margin)
            )
        )
        if step % 5 == 0 or step == steps - 1:
            print(
                f"[{case.name}] step={step:03d} status={status_name:<10} "
                f"dq={dq_norm:.3e} d={d:+.4f} stall_min={cur_stall_margin:.4f} "
                f"rows={len(row_ds)} tight={tight_rows} near_lim={near_limit}"
            )

    print(
        f"[{case.name}] done: success={success_count}, infeasible={infeasible_count}, "
        f"stall_steps={stall_count}, worst_collision={worst_collision:+.4f}, "
        f"min_stall_margin={min_stall_margin:.4f}"
    )
    return {
        "success_count": float(success_count),
        "infeasible_count": float(infeasible_count),
        "stall_count": float(stall_count),
        "worst_collision": float(worst_collision),
        "min_stall_margin": float(min_stall_margin),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alpha wheelbase stall reproducer (scratch)")
    p.add_argument(
        "--urdf",
        type=Path,
        default=Path(
            "/home/andypark/Projects/hmnd-repos/hmnd/hmnd_sim/assets/sim_robots/official/humanoid/alpha/wheelbase/v2.1/urdf/alpha_wheelbase_full.urdf"
        ),
    )
    p.add_argument(
        "--collisions-json",
        type=Path,
        default=Path(
            "/home/andypark/Projects/hmnd-repos/hmnd/hmnd_robot/ros/platforms/alpha_wheelbase_description/urdf/collisions.json"
        ),
    )
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--min-distance", type=float, default=0.04)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--damping", type=float, default=0.1)
    p.add_argument("--tolerance", type=float, default=0.1)
    p.add_argument("--offset-x", type=float, default=-0.22)
    p.add_argument("--offset-y", type=float, default=0.20)
    p.add_argument("--offset-z", type=float, default=-0.08)
    p.add_argument("--matrix", action="store_true", help="Run A/B/C/D matrix from plan")
    p.add_argument("--stall-recovery", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-constraints", type=int, default=2)
    p.add_argument("--tuning-mode", type=str, default="balanced")
    p.add_argument("--success-only-update", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    offset_xyz = np.array([args.offset_x, args.offset_y, args.offset_z], dtype=float)
    if args.matrix:
        cases = [
            CaseConfig("A", True, 2, "balanced", True),
            CaseConfig("B", False, 2, "balanced", True),
            CaseConfig("C", True, 1, "balanced", True),
            CaseConfig("D", True, 2, "precision", True),
        ]
    else:
        cases = [
            CaseConfig(
                "single",
                bool(args.stall_recovery),
                int(args.max_constraints),
                str(args.tuning_mode),
                bool(args.success_only_update),
            )
        ]

    for case in cases:
        run_case(
            urdf_path=args.urdf,
            collisions_json=args.collisions_json,
            min_distance=float(args.min_distance),
            dt=float(args.dt),
            damping=float(args.damping),
            tolerance=float(args.tolerance),
            steps=int(args.steps),
            offset_xyz=offset_xyz,
            case=case,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

