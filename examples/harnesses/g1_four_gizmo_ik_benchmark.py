#!/usr/bin/env python3
"""Headless performance harness for the G1 four-gizmo IK path.

The harness mirrors the four transform controls in
``examples/13_unitree_g1_retargeting_ik.py``: right hand, left hand, right foot,
and left foot are all driven as FRAME_POSE targets. Collision and CoM
constraints are intentionally left disabled so the measured path isolates
multi-target IK solve cost and tracking quality.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import embodik
import numpy as np

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (EXAMPLES_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from example_helpers.g1_model_utils import (  # noqa: E402
        create_g1_robot_model,
        g1_collision_pair_preset_options,
        g1_collision_pairs_for_preset,
        ground_floating_base_from_feet_center,
        resolve_frames_for_g1_base_mode,
        resolve_g1_collision_urdf_path,
        resolve_g1_urdf_path,
    )
except ModuleNotFoundError as exc:
    if exc.name != "example_helpers" and not str(exc.name).startswith("example_helpers."):
        raise
    from examples.example_helpers.g1_model_utils import (  # noqa: E402
        create_g1_robot_model,
        g1_collision_pair_preset_options,
        g1_collision_pairs_for_preset,
        ground_floating_base_from_feet_center,
        resolve_frames_for_g1_base_mode,
        resolve_g1_collision_urdf_path,
        resolve_g1_urdf_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120, help="Measured IK steps.")
    parser.add_argument("--warmup", type=int, default=12, help="Warmup steps excluded from timing.")
    parser.add_argument(
        "--max-steps", type=int, default=2, help="Internal solver iterations per sample."
    )
    parser.add_argument(
        "--reset-interval",
        type=int,
        default=0,
        help="Reset to the initial robot state every N measured samples; 0 disables resets.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Scale for synthetic gizmo offsets; use 1.0 for a harder stress sweep.",
    )
    parser.add_argument("--hand-position-gain", type=float, default=12.0)
    parser.add_argument("--hand-orientation-gain", type=float, default=8.0)
    parser.add_argument("--foot-position-gain", type=float, default=40.0)
    parser.add_argument("--foot-orientation-gain", type=float, default=14.0)
    parser.add_argument(
        "--full-ik-model",
        action="store_true",
        help="Use the full G1 model including finger joints instead of the reduced IK model.",
    )
    parser.add_argument(
        "--disable-posture",
        action="store_true",
        help="Disable the posture/nullspace task for latency ablations.",
    )
    parser.add_argument(
        "--include-pelvis",
        action="store_true",
        help="Include the torso/pelvis frame as a fifth direct IK target.",
    )
    parser.add_argument(
        "--collision-preset",
        choices=("none", *g1_collision_pair_preset_options()),
        default="none",
        help="Enable collision constraints with a curated G1 include-pair preset.",
    )
    parser.add_argument("--collision-min-distance-mm", type=float, default=20.0)
    parser.add_argument("--collision-max-rows", type=int, default=3)
    parser.add_argument(
        "--collision-tuning",
        choices=("speed", "balanced", "precise"),
        default="balanced",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None, help="Optional metrics JSON path."
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the final JSON summary.")
    return parser.parse_args()


def _clip_q(robot, q: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).copy()
    if getattr(robot, "is_floating_base", False) and q.size >= 7:
        q[7:] = np.clip(q[7:], q_lo[7:], q_hi[7:])
        quat = q[3:7]
        n = float(np.linalg.norm(quat))
        q[3:7] = quat / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return q
    return np.clip(q, q_lo, q_hi)


def _apply_g1_soft_knee_seed(robot, q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).copy()
    seed = {
        "left_hip_pitch_joint": -0.18,
        "left_knee_joint": 0.36,
        "left_ankle_pitch_joint": -0.18,
        "right_hip_pitch_joint": -0.18,
        "right_knee_joint": 0.36,
        "right_ankle_pitch_joint": -0.18,
    }
    for joint_name, value in seed.items():
        try:
            idx = int(robot.get_joint_config_index(joint_name))
        except Exception:
            continue
        if 0 <= idx < q.size:
            q[idx] = float(value)
    return q


def _pose_matrix(robot, frame_name: str) -> np.ndarray:
    pose = robot.get_frame_pose(frame_name)
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = np.asarray(pose.rotation, dtype=float)
    mat[:3, 3] = np.asarray(pose.translation, dtype=float)
    return mat


def _target_poses(
    base_poses: dict[str, np.ndarray],
    phase: float,
    *,
    scale: float,
) -> dict[str, np.ndarray]:
    targets = {name: pose.copy() for name, pose in base_poses.items()}
    targets["right_palm"][:3, 3] += scale * np.array(
        [0.055 * np.sin(phase), 0.030 * np.sin(1.3 * phase), 0.025 * np.sin(2.0 * phase)]
    )
    targets["left_palm"][:3, 3] += scale * np.array(
        [
            -0.050 * np.sin(phase),
            -0.030 * np.sin(1.1 * phase),
            0.022 * np.sin(1.7 * phase),
        ]
    )
    targets["right_ankle"][:3, 3] += scale * np.array(
        [0.025 * np.sin(0.8 * phase), 0.020 * np.sin(phase), 0.012 * np.sin(1.6 * phase)]
    )
    targets["left_ankle"][:3, 3] += scale * np.array(
        [
            0.022 * np.sin(0.9 * phase + np.pi),
            -0.018 * np.sin(phase),
            0.010 * np.sin(1.4 * phase),
        ]
    )
    if "pelvis" in targets:
        targets["pelvis"][:3, 3] += scale * np.array(
            [0.018 * np.sin(0.7 * phase), 0.010 * np.sin(1.2 * phase), 0.012 * np.sin(phase)]
        )
    return targets


def _zeroed_target_poses(
    base_poses: dict[str, np.ndarray],
    phase: float,
    *,
    scale: float,
) -> dict[str, np.ndarray]:
    targets = _target_poses(base_poses, phase, scale=scale)
    origin = _target_poses(base_poses, 0.0, scale=scale)
    for key, pose in targets.items():
        pose[:3, 3] -= origin[key][:3, 3] - base_poses[key][:3, 3]
    return targets


def _frame_errors(
    robot, frame_map: dict[str, str], targets: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    errors: dict[str, dict[str, float]] = {}
    for key, target in targets.items():
        frame_name = frame_map["imu_in_torso"] if key == "pelvis" else frame_map[key]
        current = _pose_matrix(robot, frame_name)
        pos_error = float(np.linalg.norm(current[:3, 3] - target[:3, 3]))
        rot_error = float(np.linalg.norm(embodik.log3(target[:3, :3].T @ current[:3, :3])))
        errors[key] = {"position_m": pos_error, "orientation_rad": rot_error}
    return errors


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), pct))


def _g1_posture_controlled_joint_indices(robot) -> list[int]:
    if getattr(robot, "is_floating_base", False) and int(robot.nv) > 6:
        return list(range(6, int(robot.nv)))
    return list(range(int(robot.nv)))


def _configure_g1_posture_task(posture, robot, q_target: np.ndarray) -> None:
    posture.set_target_configuration(np.asarray(q_target, dtype=float).copy())
    if hasattr(posture, "set_controlled_joint_indices"):
        posture.set_controlled_joint_indices(_g1_posture_controlled_joint_indices(robot))


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    robot = create_g1_robot_model(floating_base=True, reduced_ik=not bool(args.full_ik_model))
    q = _apply_g1_soft_knee_seed(robot, robot.neutral_configuration())
    q_lo, q_hi = robot.get_joint_limits()
    robot.update_configuration(q)
    frame_map = resolve_frames_for_g1_base_mode(robot.get_frame_names())

    r0 = np.asarray(robot.get_frame_pose(frame_map["right_ankle"]).translation, dtype=float)
    l0 = np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).translation, dtype=float)
    q = ground_floating_base_from_feet_center(q, r0, l0)
    q = _clip_q(robot, q, q_lo, q_hi)
    robot.update_configuration(q)
    q_initial = q.copy()

    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)
    solver.set_tolerance(0.1)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    if hasattr(solver, "clear_collision_constraint"):
        solver.clear_collision_constraint()
    if hasattr(solver, "clear_com_constraint"):
        solver.clear_com_constraint()
    collision_include_pairs: list[tuple[str, str]] = []
    collision_enabled = str(args.collision_preset) != "none"
    if collision_enabled:
        collision_include_pairs = g1_collision_pairs_for_preset(robot, str(args.collision_preset))
        if hasattr(solver, "set_collision_tuning_mode") and hasattr(embodik, "CollisionTuningMode"):
            mode_map = {
                "speed": embodik.CollisionTuningMode.SPEED,
                "balanced": embodik.CollisionTuningMode.BALANCED,
                "precise": embodik.CollisionTuningMode.PRECISE,
            }
            solver.set_collision_tuning_mode(mode_map[str(args.collision_tuning)])
        solver.configure_collision_constraint(
            min_distance=float(args.collision_min_distance_mm) * 1e-3,
            include_pairs=list(collision_include_pairs),
            exclude_pairs=[],
            nearest_points_all_pairs=False,
            max_constraints=int(args.collision_max_rows),
        )

    task_names = {
        "right_palm": "right_palm_pose",
        "left_palm": "left_palm_pose",
        "right_ankle": "right_ankle_pose",
        "left_ankle": "left_ankle_pose",
    }
    if bool(args.include_pelvis):
        task_names["pelvis"] = "pelvis_pose"
    tasks = []
    for key, task_name in task_names.items():
        frame_name = frame_map["imu_in_torso"] if key == "pelvis" else frame_map[key]
        task = solver.add_frame_task(task_name, frame_name, embodik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 1.0
        task.solve_mode = getattr(
            embodik.TaskSolveMode,
            "SCALE_ELASTIC",
            embodik.TaskSolveMode.MIN_ERROR,
        )
        task.allow_min_error_fallback = True
        tasks.append(task)

    posture = None
    if not bool(args.disable_posture):
        posture = solver.add_posture_task("posture")
        posture.priority = 1
        posture.weight = 0.002
        _configure_g1_posture_task(posture, robot, q)

    opts = embodik.PositionStepOptions()
    opts.max_steps = int(args.max_steps)
    opts.position_gain = float(args.hand_position_gain)
    opts.orientation_gain = float(args.hand_orientation_gain)
    if hasattr(opts, "stall_recovery"):
        opts.stall_recovery = False
    if hasattr(opts, "adaptive_dt"):
        opts.adaptive_dt = False

    base_poses = {
        key: _pose_matrix(robot, frame_map["imu_in_torso"] if key == "pelvis" else frame_map[key])
        for key in task_names
    }
    total_steps = max(1, int(args.warmup)) + max(1, int(args.steps))
    measured_from = max(0, int(args.warmup))
    phases = np.linspace(0.0, 2.0 * np.pi, total_steps, endpoint=False)

    wall_ms: list[float] = []
    max_position_errors: list[float] = []
    max_orientation_errors: list[float] = []
    q_step_norms: list[float] = []
    q_step_max_components: list[float] = []
    q_accel_norms: list[float] = []
    collision_ms: list[float] = []
    collision_pairs_considered: list[int] = []
    collision_exact_queries: list[int] = []
    collision_budget_exhausted = 0
    status_counts: Counter[str] = Counter()
    accepted = 0
    previous_dq: np.ndarray | None = None
    reset_count = 0
    reset_max_recapture_error = 0.0
    reset_first_move_min_dq = float("inf")
    pending_reset_move = False
    segment_phase_origin = 0.0

    per_frame_max = {key: {"position_m": 0.0, "orientation_rad": 0.0} for key in task_names}

    for sample_idx, phase in enumerate(phases):
        measured_idx = sample_idx - measured_from
        if (
            int(args.reset_interval) > 0
            and measured_idx >= 0
            and measured_idx > 0
            and measured_idx % int(args.reset_interval) == 0
        ):
            q = q_initial.copy()
            robot.update_configuration(q)
            if posture is not None:
                _configure_g1_posture_task(posture, robot, q)
            previous_dq = None
            pending_reset_move = True
            reset_count += 1
            segment_phase_origin = float(phase)
            reset_targets = {
                key: _pose_matrix(
                    robot, frame_map["imu_in_torso"] if key == "pelvis" else frame_map[key]
                )
                for key in task_names
            }
            reset_max_recapture_error = max(
                reset_max_recapture_error,
                max(
                    float(
                        np.linalg.norm(
                            _pose_matrix(
                                robot,
                                frame_map["imu_in_torso"] if key == "pelvis" else frame_map[key],
                            )[:3, 3]
                            - pose[:3, 3]
                        )
                    )
                    for key, pose in reset_targets.items()
                ),
            )
            continue

        targets_by_frame = _zeroed_target_poses(
            base_poses, float(phase) - segment_phase_origin, scale=float(args.scale)
        )
        targets = [
            embodik.TaskTarget(
                task_names["right_palm"],
                targets_by_frame["right_palm"],
                float(args.hand_position_gain),
                float(args.hand_orientation_gain),
            ),
            embodik.TaskTarget(
                task_names["left_palm"],
                targets_by_frame["left_palm"],
                float(args.hand_position_gain),
                float(args.hand_orientation_gain),
            ),
            embodik.TaskTarget(
                task_names["right_ankle"],
                targets_by_frame["right_ankle"],
                float(args.foot_position_gain),
                float(args.foot_orientation_gain),
            ),
            embodik.TaskTarget(
                task_names["left_ankle"],
                targets_by_frame["left_ankle"],
                float(args.foot_position_gain),
                float(args.foot_orientation_gain),
            ),
        ]
        if "pelvis" in task_names:
            targets.append(
                embodik.TaskTarget(
                    task_names["pelvis"],
                    targets_by_frame["pelvis"],
                    float(args.foot_position_gain),
                    float(args.foot_orientation_gain),
                )
            )

        q_before = q.copy()
        t0 = time.perf_counter()
        result = solver.solve_position_step(q, targets, opts)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        status_name = getattr(result.status, "name", str(result.status))

        if hasattr(result, "q_solution") and status_name != "COLLISION_VIOLATED":
            q_candidate = _clip_q(robot, np.asarray(result.q_solution, dtype=float), q_lo, q_hi)
            if np.all(np.isfinite(q_candidate)):
                q = q_candidate
                robot.update_configuration(q)
                accepted += 1

        errors = _frame_errors(robot, frame_map, targets_by_frame)
        max_pos = max(item["position_m"] for item in errors.values())
        max_rot = max(item["orientation_rad"] for item in errors.values())
        dq = np.asarray(robot.difference(q_before, q), dtype=float)
        dq_norm = float(np.linalg.norm(dq))
        dq_max_component = float(np.max(np.abs(dq))) if dq.size else 0.0
        accel_norm = 0.0 if previous_dq is None else float(np.linalg.norm(dq - previous_dq))
        previous_dq = dq.copy()
        productive_status_name = status_name
        if status_name == "NO_PROGRESS" and max_pos <= 0.01 and max_rot <= 0.01:
            productive_status_name = "SUCCESS"
        if pending_reset_move:
            reset_first_move_min_dq = min(reset_first_move_min_dq, dq_norm)
            pending_reset_move = False

        if sample_idx >= measured_from:
            wall_ms.append(float(elapsed_ms))
            max_position_errors.append(max_pos)
            max_orientation_errors.append(max_rot)
            q_step_norms.append(dq_norm)
            q_step_max_components.append(dq_max_component)
            q_accel_norms.append(accel_norm)
            collision_ms.append(float(getattr(result, "collision_constraint_time_ms", 0.0)))
            collision_pairs_considered.append(int(getattr(result, "collision_pairs_considered", 0)))
            collision_exact_queries.append(
                int(getattr(result, "collision_exact_distance_queries", 0))
            )
            collision_budget_exhausted += int(
                bool(getattr(result, "collision_budget_exhausted", False))
            )
            status_counts[productive_status_name] += 1
            for key, item in errors.items():
                per_frame_max[key]["position_m"] = max(
                    per_frame_max[key]["position_m"], item["position_m"]
                )
                per_frame_max[key]["orientation_rad"] = max(
                    per_frame_max[key]["orientation_rad"],
                    item["orientation_rad"],
                )

    measured_steps = len(wall_ms)
    return {
        "harness": "g1_four_gizmo_ik_benchmark",
        "urdf": str(resolve_g1_urdf_path()),
        "collision_urdf": str(resolve_g1_collision_urdf_path()),
        "config": {
            "steps": int(args.steps),
            "warmup": int(args.warmup),
            "max_steps": int(args.max_steps),
            "scale": float(args.scale),
            "reset_interval": int(args.reset_interval),
            "collision_enabled": bool(collision_enabled),
            "collision_preset": str(args.collision_preset),
            "collision_include_pair_count": len(collision_include_pairs),
            "collision_min_distance_m": float(args.collision_min_distance_mm) * 1e-3,
            "collision_max_rows": int(args.collision_max_rows),
            "collision_tuning": str(args.collision_tuning),
            "com_enabled": False,
            "reduced_ik_model": not bool(args.full_ik_model),
            "posture_enabled": not bool(args.disable_posture),
            "pelvis_target_enabled": bool(args.include_pelvis),
            "nq": int(robot.nq),
            "nv": int(robot.nv),
        },
        "frames": frame_map,
        "measured_steps": measured_steps,
        "accepted_steps_total": accepted,
        "status_counts": dict(sorted(status_counts.items())),
        "reset": {
            "count": reset_count,
            "max_recapture_error_m": reset_max_recapture_error,
            "first_move_min_dq_norm": (
                reset_first_move_min_dq if np.isfinite(reset_first_move_min_dq) else 0.0
            ),
        },
        "wall_time_ms": {
            "mean": float(np.mean(wall_ms)) if wall_ms else 0.0,
            "median": float(np.median(wall_ms)) if wall_ms else 0.0,
            "p95": _percentile(wall_ms, 95.0),
            "max": max(wall_ms, default=0.0),
        },
        "collision_time_ms": {
            "mean": float(np.mean(collision_ms)) if collision_ms else 0.0,
            "median": float(np.median(collision_ms)) if collision_ms else 0.0,
            "p95": _percentile(collision_ms, 95.0),
            "max": max(collision_ms, default=0.0),
            "pairs_considered_max": max(collision_pairs_considered, default=0),
            "exact_queries_max": max(collision_exact_queries, default=0),
            "budget_exhausted_count": collision_budget_exhausted,
        },
        "target_error": {
            "max_position_m": max(max_position_errors, default=0.0),
            "rms_max_position_m": (
                float(np.sqrt(np.mean(np.square(max_position_errors))))
                if max_position_errors
                else 0.0
            ),
            "max_orientation_rad": max(max_orientation_errors, default=0.0),
            "rms_max_orientation_rad": (
                float(np.sqrt(np.mean(np.square(max_orientation_errors))))
                if max_orientation_errors
                else 0.0
            ),
            "per_frame_max": per_frame_max,
        },
        "smoothness": {
            "mean_q_step_norm": float(np.mean(q_step_norms)) if q_step_norms else 0.0,
            "max_q_step_norm": max(q_step_norms, default=0.0),
            "max_q_step_component": max(q_step_max_components, default=0.0),
            "mean_q_step_delta_norm": float(np.mean(q_accel_norms)) if q_accel_norms else 0.0,
            "max_q_step_delta_norm": max(q_accel_norms, default=0.0),
        },
    }


def main() -> None:
    args = parse_args()
    metrics = run_benchmark(args)
    encoded = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    if args.quiet:
        print(encoded)
        return

    status_counts = metrics["status_counts"]
    wall = metrics["wall_time_ms"]
    collision = metrics["collision_time_ms"]
    error = metrics["target_error"]
    smoothness = metrics["smoothness"]
    reset = metrics["reset"]
    print("[g1-four-gizmo-benchmark]")
    print(f"  measured_steps={metrics['measured_steps']} status_counts={status_counts}")
    print(
        "  reset="
        f"count={reset['count']} recapture={reset['max_recapture_error_m']:.3e}m "
        f"first_move_min_dq={reset['first_move_min_dq_norm']:.3e}"
    )
    print(
        "  wall_time_ms="
        f"mean={wall['mean']:.3f} median={wall['median']:.3f} "
        f"p95={wall['p95']:.3f} max={wall['max']:.3f}"
    )
    if metrics["config"]["collision_enabled"]:
        print(
            "  collision="
            f"preset={metrics['config']['collision_preset']} "
            f"pairs={metrics['config']['collision_include_pair_count']} "
            f"mean={collision['mean']:.3f}ms p95={collision['p95']:.3f}ms "
            f"max={collision['max']:.3f}ms exact_max={collision['exact_queries_max']}"
        )
    print(
        "  target_error="
        f"max_pos={error['max_position_m']:.5f}m "
        f"rms_max_pos={error['rms_max_position_m']:.5f}m "
        f"max_rot={error['max_orientation_rad']:.5f}rad"
    )
    print(
        "  smoothness="
        f"mean_q_step={smoothness['mean_q_step_norm']:.5f} "
        f"max_q_step={smoothness['max_q_step_norm']:.5f} "
        f"max_q_step_delta={smoothness['max_q_step_delta_norm']:.5f}"
    )
    if args.output_json is not None:
        print(f"  output_json={args.output_json}")


if __name__ == "__main__":
    main()
