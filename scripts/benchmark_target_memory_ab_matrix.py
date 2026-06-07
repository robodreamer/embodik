#!/usr/bin/env python3
"""A/B matrix benchmark for target-return memory in the shared bimanual app.

The target-return memory feature lives in the Python app layer, so this harness
drives the same public bimanual fixture directly instead of reusing the C++
runtime-only nullspace matrix.  Each case runs an A -> B -> A trajectory twice:
with memory disabled and enabled.  During hold phases, memory may store/recall a
healthy solution; during target-motion phases it is disabled to match the GUI
integration.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import embodik  # noqa: E402
from embodik.interactive_ik import (  # noqa: E402
    clear_all_target_velocities_if_available,
    configure_primary_solve_mode,
    robust_solve_position_step,
)
from examples.example_helpers.common_bimanual_solver_fixture import (  # noqa: E402
    build_common_bimanual_solver_fixture,
    compute_inner_polygon,
)
from examples.example_helpers.common_bimanual_teleop_app import (  # noqa: E402
    DEFAULT_ARM_NULLSPACE_WEIGHT,
    DEFAULT_MAX_ANGULAR_SPEED,
    DEFAULT_MAX_LINEAR_SPEED,
    DEFAULT_POSTURE_WEIGHT,
    EE_POSITION_DEADBAND,
    _apply_position_step_speed_caps,
    _polygon_slack,
    _TargetReturnMemory,
)
from examples.harnesses.kinematic_health_metrics import (  # noqa: E402
    kinematic_health_sample,
    summarize_continuity_series,
    summarize_health_series,
)

HOLD_PHASES = {"hold_a_initial", "hold_b", "hold_a_return"}
SCENARIO_ALIASES = {
    "all": (
        "both_outward_return",
        "both_cross_return",
        "both_vertical_return",
        "both_down_return",
        "both_down_far_return",
        "right_only_cross_return",
        "right_only_down_return",
        "right_only_down_far_return",
    ),
    "stress": (
        "both_cross_return",
        "both_down_return",
        "both_down_far_return",
        "right_only_cross_return",
        "right_only_down_return",
        "right_only_down_far_return",
    ),
}
CONTEXT_ALIASES = {
    "all": ("no_collision", "collision", "collision_com"),
}
TORSO_POLICY_ALIASES = {
    "all": ("unlocked", "locked", "prefer_locked"),
}


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    active_sides: tuple[str, ...]
    right_delta: tuple[float, float, float]
    left_delta: tuple[float, float, float]


@dataclass(frozen=True)
class ContextSpec:
    name: str
    collision_enabled: bool
    com_enabled: bool


@dataclass(frozen=True)
class ModelSpec:
    name: str
    fixture_variant: str
    urdf_path: Path | None = None
    collision_urdf_path: Path | None = None
    support_contact_frames: tuple[str, ...] | None = None
    lock_joint_names: tuple[str, ...] | None = None
    posture_joint_names: tuple[str, ...] | None = None
    collision_link_pairs: tuple[tuple[str, str], ...] | None = None
    com_frame_name: str = "base_link"
    collision_min_distance_m: float | None = None
    max_collision_constraints: int | None = None


@dataclass
class CaseMetrics:
    variant: str
    scenario: str
    context: str
    torso_policy: str
    memory_enabled: bool
    steps: int
    status_counts: dict[str, int]
    memory_hits: int
    memory_stores: int
    memory_rejects: int
    memory_entry_count: int
    max_memory_entry_count: int
    prefer_locked_attempts: int
    prefer_locked_used: int
    avg_solve_attempts_per_step: float
    avg_solver_ms: float
    p95_solver_ms: float
    avg_wall_ms: float
    p95_wall_ms: float
    avg_memory_ms: float
    p95_memory_ms: float
    final_max_error_m: float
    return_mean_max_error_m: float
    return_p95_max_error_m: float
    return_final_q_distance_to_initial_a: float
    return_mean_q_distance_to_initial_a: float
    return_joint_limit_health_mean: float
    return_joint_limit_health_p05: float
    return_limit_weighted_dexterity_mean: float
    return_limit_weighted_dexterity_p05: float
    full_joint_limit_health_mean: float
    full_limit_weighted_dexterity_mean: float
    min_collision_distance_m: float | None
    min_com_inner_slack_m: float | None
    torso_motion_mean: float
    torso_motion_p95: float
    torso_motion_max: float
    arm_motion_mean: float
    arm_motion_p95: float
    arm_motion_max: float
    torso_arm_motion_ratio_mean: float
    locked_chain_velocity_max: float
    continuity: dict[str, float | int]


def _stats(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def _pose_to_matrix(pose) -> np.ndarray:
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = np.asarray(pose.rotation, dtype=float)
    mat[:3, 3] = np.asarray(pose.translation, dtype=float)
    return mat


def _pose_from_position(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = np.asarray(rotation, dtype=float)
    pose[:3, 3] = np.asarray(position, dtype=float)
    return pose


def _make_pose_target(task_name: str, pose: np.ndarray, pos_gain: float, rot_gain: float):
    return embodik.TaskTarget(
        task_name, np.asarray(pose, dtype=float), float(pos_gain), float(rot_gain)
    )


def _smoothstep(values: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _phase_lengths(steps: int) -> dict[str, int]:
    n = max(5, int(steps))
    initial_hold = max(5, int(round(n * 0.15)))
    move_to_b = max(5, int(round(n * 0.25)))
    hold_b = max(5, int(round(n * 0.20)))
    move_to_a = max(5, int(round(n * 0.25)))
    hold_a_return = max(5, n - initial_hold - move_to_b - hold_b - move_to_a)
    total = initial_hold + move_to_b + hold_b + move_to_a + hold_a_return
    hold_a_return += n - total
    return {
        "hold_a_initial": initial_hold,
        "move_to_b": move_to_b,
        "hold_b": hold_b,
        "move_to_a": move_to_a,
        "hold_a_return": max(1, hold_a_return),
    }


def _aba_positions(
    start: np.ndarray,
    delta: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, list[str]]:
    phases = _phase_lengths(steps)
    positions: list[np.ndarray] = []
    labels: list[str] = []
    a = np.asarray(start, dtype=float)
    b = a + np.asarray(delta, dtype=float)
    for phase, count in phases.items():
        count = int(count)
        if phase == "hold_a_initial":
            values = np.repeat(a[None, :], count, axis=0)
        elif phase == "move_to_b":
            alpha = _smoothstep(np.linspace(0.0, 1.0, count))
            values = a[None, :] + alpha[:, None] * (b - a)[None, :]
        elif phase == "hold_b":
            values = np.repeat(b[None, :], count, axis=0)
        elif phase == "move_to_a":
            alpha = _smoothstep(np.linspace(0.0, 1.0, count))
            values = b[None, :] + alpha[:, None] * (a - b)[None, :]
        else:
            values = np.repeat(a[None, :], count, axis=0)
        positions.extend(np.asarray(values, dtype=float))
        labels.extend([phase] * count)
    return np.asarray(positions[:steps], dtype=float), labels[:steps]


def _scenario_specs() -> dict[str, ScenarioSpec]:
    return {
        "both_outward_return": ScenarioSpec(
            "both_outward_return",
            ("right", "left"),
            (0.12, -0.10, 0.06),
            (0.12, 0.10, 0.06),
        ),
        "both_cross_return": ScenarioSpec(
            "both_cross_return",
            ("right", "left"),
            (0.10, 0.16, 0.02),
            (0.10, -0.16, 0.02),
        ),
        "both_vertical_return": ScenarioSpec(
            "both_vertical_return",
            ("right", "left"),
            (0.04, -0.04, 0.16),
            (0.04, 0.04, 0.16),
        ),
        "both_down_return": ScenarioSpec(
            "both_down_return",
            ("right", "left"),
            (0.03, -0.03, -0.14),
            (0.03, 0.03, -0.14),
        ),
        "both_down_far_return": ScenarioSpec(
            "both_down_far_return",
            ("right", "left"),
            (0.08, -0.05, -0.28),
            (0.08, 0.05, -0.28),
        ),
        "right_only_cross_return": ScenarioSpec(
            "right_only_cross_return",
            ("right",),
            (0.12, 0.18, 0.03),
            (0.0, 0.0, 0.0),
        ),
        "right_only_down_return": ScenarioSpec(
            "right_only_down_return",
            ("right",),
            (0.03, 0.02, -0.14),
            (0.0, 0.0, 0.0),
        ),
        "right_only_down_far_return": ScenarioSpec(
            "right_only_down_far_return",
            ("right",),
            (0.08, 0.04, -0.28),
            (0.0, 0.0, 0.0),
        ),
    }


def _context_specs() -> dict[str, ContextSpec]:
    return {
        "no_collision": ContextSpec("no_collision", False, False),
        "collision": ContextSpec("collision", True, False),
        "collision_com": ContextSpec("collision_com", True, True),
    }


def _load_example06_module():
    module_path = _REPO_ROOT / "examples" / "06_bimanual_whole_body_ik.py"
    spec = importlib.util.spec_from_file_location("_embodik_example06_bimanual", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_collision_link_pairs_json(paths: Iterable[str | Path]) -> tuple[tuple[str, str], ...]:
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        for item in raw:
            if len(item) < 2:
                continue
            pair = tuple(sorted((str(item[0]), str(item[1]))))
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
    return tuple(pairs)


def _resolve_model_specs(
    args: argparse.Namespace,
    contexts: list[ContextSpec],
) -> tuple[list[ModelSpec], list[dict[str, str]]]:
    specs: list[ModelSpec] = []
    skipped: list[dict[str, str]] = []
    needs_collision = any(context.collision_enabled for context in contexts)
    for raw_variant in args.variants:
        variant = str(raw_variant)
        if variant in {"sg2", "bg2"}:
            specs.append(ModelSpec(name=variant, fixture_variant=variant))
            continue
        if variant == "rby1":
            try:
                example06 = _load_example06_module()
                base_urdf = (
                    Path(args.rby1_urdf).expanduser().resolve()
                    if args.rby1_urdf
                    else example06._resolve_rby1_urdf_path()
                )
                urdf_path = example06._prepare_rby1_tip_frame_urdf(base_urdf)
                collision_urdf_path = (
                    example06._prepare_rby1_tip_frame_urdf(
                        Path(args.rby1_collision_urdf).expanduser().resolve()
                    )
                    if args.rby1_collision_urdf
                    else example06._prepare_rby1_collision_urdf(urdf_path)
                )
                torso_joints = tuple(f"torso_{idx}" for idx in range(6))
                specs.append(
                    ModelSpec(
                        name="rby1",
                        fixture_variant="rby1",
                        urdf_path=Path(urdf_path),
                        collision_urdf_path=Path(collision_urdf_path),
                        support_contact_frames=("wheel_l", "wheel_r", "base"),
                        lock_joint_names=torso_joints,
                        posture_joint_names=torso_joints,
                        com_frame_name="base",
                        collision_min_distance_m=0.020,
                        max_collision_constraints=8,
                    )
                )
            except Exception as exc:
                skipped.append(
                    {"variant": "rby1", "scenario": "*", "context": "*", "reason": repr(exc)}
                )
            continue
        if variant == "alpha_wheelbase":
            alpha_urdf = Path(args.alpha_urdf).expanduser().resolve() if args.alpha_urdf else None
            alpha_collision_jsons = [
                Path(p).expanduser().resolve() for p in args.alpha_collisions_json
            ]
            if alpha_urdf is None:
                skipped.append(
                    {
                        "variant": "alpha_wheelbase",
                        "scenario": "*",
                        "context": "*",
                        "reason": "alpha_wheelbase requires --alpha-urdf",
                    }
                )
                continue
            if needs_collision and not alpha_collision_jsons:
                skipped.append(
                    {
                        "variant": "alpha_wheelbase",
                        "scenario": "*",
                        "context": "*",
                        "reason": "alpha_wheelbase collision contexts require --alpha-collisions-json",
                    }
                )
                continue
            collision_link_pairs = (
                _load_collision_link_pairs_json(alpha_collision_jsons)
                if alpha_collision_jsons
                else None
            )
            lock_joints = (
                "base_yaw_joint",
                "base_pitch_joint",
                "knee_pitch_joint",
                "hip_pitch_joint",
                "torso_yaw_joint",
            )
            specs.append(
                ModelSpec(
                    name="alpha_wheelbase",
                    fixture_variant="alpha_wheelbase",
                    urdf_path=alpha_urdf,
                    collision_urdf_path=(
                        Path(args.alpha_collision_urdf).expanduser().resolve()
                        if args.alpha_collision_urdf
                        else alpha_urdf
                    ),
                    support_contact_frames=(
                        "wheel_front_left_link",
                        "wheel_front_right_link",
                        "wheel_back_left_link",
                        "wheel_back_right_link",
                    ),
                    lock_joint_names=lock_joints,
                    posture_joint_names=lock_joints,
                    collision_link_pairs=collision_link_pairs,
                    collision_min_distance_m=0.035,
                    max_collision_constraints=6,
                )
            )
            continue
        skipped.append(
            {
                "variant": variant,
                "scenario": "*",
                "context": "*",
                "reason": "unknown variant/model",
            }
        )
    return specs, skipped


def _build_position_step_options(
    solve_mode_label: str,
    allow_fallback: bool,
    pos_gain: float,
    rot_gain: float,
    stall_recovery: bool,
    max_linear_speed: float,
    max_angular_speed: float,
) -> object:
    opts = embodik.PositionStepOptions()
    opts.dt = 0.01
    opts.max_steps = 1
    if hasattr(opts, "position_gain"):
        opts.position_gain = float(pos_gain)
    if hasattr(opts, "orientation_gain"):
        opts.orientation_gain = float(rot_gain)
    if hasattr(opts, "stall_recovery"):
        opts.stall_recovery = bool(stall_recovery)
    if hasattr(opts, "no_progress_max_steps"):
        opts.no_progress_max_steps = 5
    if hasattr(opts, "no_progress_error_tolerance"):
        opts.no_progress_error_tolerance = 1e-5
    if hasattr(opts, "no_progress_dq_norm_tolerance"):
        opts.no_progress_dq_norm_tolerance = 1e-6
    _apply_position_step_speed_caps(
        opts,
        max_linear_speed=float(max_linear_speed),
        max_angular_speed=float(max_angular_speed),
    )
    solve_mode = getattr(embodik.TaskSolveMode, solve_mode_label, embodik.TaskSolveMode.SCALE)
    configure_primary_solve_mode(opts, solve_mode, bool(allow_fallback))
    return opts


def _status_name(result: object) -> str:
    status = getattr(result, "status", None)
    return str(getattr(status, "name", status))


def _memory_health_score(result: object) -> float:
    values: list[float] = []
    for source in (getattr(result, "diagnostics", None), result):
        if source is None:
            continue
        for name in ("health_sampling_best_score", "health_sampling_base_score"):
            value = float(getattr(source, name, float("nan")))
            if np.isfinite(value):
                values.append(value)
    return max(values) if values else float("nan")


def _memory_result_is_clean(result: object, status_name: str) -> bool:
    if status_name not in {"SUCCESS", "NO_PROGRESS"}:
        return False
    for source in (getattr(result, "diagnostics", None), result):
        if source is None:
            continue
        if int(getattr(source, "collision_rejection_count", 0) or 0) > 0:
            return False
        if int(getattr(source, "stall_escape_count", 0) or 0) > 0:
            return False
        if bool(getattr(source, "weighted_fallback_used", False)):
            return False
        raw_task_scales = getattr(source, "task_scales", ())
        task_scales = [] if raw_task_scales is None else list(raw_task_scales)
        if task_scales and min(float(v) for v in task_scales) < 0.05:
            return False
    return True


def _evaluate_collision_min_distance(solver, q: np.ndarray) -> float:
    if not hasattr(solver, "evaluate_collision_debug"):
        return float("inf")
    try:
        dbg = solver.evaluate_collision_debug(np.asarray(q, dtype=float))
    except Exception:
        return float("inf")
    if dbg is None:
        return float("inf")
    try:
        return float(dbg.distance)
    except Exception:
        return float("inf")


def _evaluate_com_inner_slack(robot, inner_polygon: np.ndarray) -> float:
    if not hasattr(robot, "get_com_position"):
        return float("inf")
    com_xy = np.asarray(robot.get_com_position(), dtype=float)[:2]
    slacks = _polygon_slack(np.asarray(inner_polygon, dtype=float), com_xy)
    return float(slacks.min()) if slacks.size else float("inf")


def _active_pose_errors(
    robot,
    frame_map: dict[str, str],
    active_sides: tuple[str, ...],
    target_poses: dict[str, np.ndarray],
) -> tuple[float, dict[str, float]]:
    errors: dict[str, float] = {}
    for side in active_sides:
        frame = frame_map["right_tool" if side == "right" else "left_tool"]
        current = np.asarray(robot.get_frame_pose(frame).translation, dtype=float)
        target = np.asarray(target_poses[side], dtype=float)[:3, 3]
        errors[side] = float(np.linalg.norm(target - current))
    return max(errors.values()) if errors else 0.0, errors


def _combined_active_jacobian(fx, active_sides: tuple[str, ...]) -> np.ndarray | None:
    rows: list[np.ndarray] = []
    try:
        if "right" in active_sides:
            fx.right_task.update(fx.robot)
            rows.append(np.asarray(fx.right_task.get_jacobian(), dtype=float))
        if "left" in active_sides:
            fx.left_task.update(fx.robot)
            rows.append(np.asarray(fx.left_task.get_jacobian(), dtype=float))
    except Exception:
        return None
    return np.vstack(rows) if rows else None


def _configure_active_tasks(
    fx,
    active_sides: tuple[str, ...],
    solve_mode_label: str,
    *,
    extra_excluded_indices: Iterable[int] = (),
) -> None:
    right_active = "right" in active_sides
    left_active = "left" in active_sides
    solve_mode = getattr(embodik.TaskSolveMode, solve_mode_label, embodik.TaskSolveMode.SCALE)
    fx.right_task.weight = 1.0 if right_active else 0.0
    fx.left_task.weight = 1.0 if left_active else 0.0
    for task in (fx.right_task, fx.left_task):
        task.priority = 0
        task.solve_mode = solve_mode

    extra = list(extra_excluded_indices)
    right_excluded = list(fx.left_arm_velocity_indices) + extra
    left_excluded = list(fx.right_arm_velocity_indices) + extra
    if hasattr(fx.right_task, "set_excluded_joint_indices"):
        if right_active and right_excluded:
            fx.right_task.set_excluded_joint_indices(sorted(set(right_excluded)))
        elif hasattr(fx.right_task, "clear_excluded_joint_indices"):
            fx.right_task.clear_excluded_joint_indices()
    if hasattr(fx.left_task, "set_excluded_joint_indices"):
        if left_active and left_excluded:
            fx.left_task.set_excluded_joint_indices(sorted(set(left_excluded)))
        elif hasattr(fx.left_task, "clear_excluded_joint_indices"):
            fx.left_task.clear_excluded_joint_indices()


def _dynamic_excluded_indices(
    fx, active_sides: tuple[str, ...], *, lock_passive: bool
) -> list[int]:
    dynamic: list[int] = []
    right_active = "right" in active_sides
    left_active = "left" in active_sides
    if right_active and not left_active:
        dynamic.extend(fx.left_arm_velocity_indices)
    elif left_active and not right_active:
        dynamic.extend(fx.right_arm_velocity_indices)
    if lock_passive:
        dynamic.extend(fx.locked_velocity_indices)
    return sorted(set(dynamic))


def _active_arm_indices(fx, active_sides: tuple[str, ...]) -> list[int]:
    indices: list[int] = []
    if "left" in active_sides:
        indices.extend(fx.left_arm_velocity_indices)
    if "right" in active_sides:
        indices.extend(fx.right_arm_velocity_indices)
    return sorted(set(indices))


def _indexed_norm(values: np.ndarray, indices: Iterable[int]) -> float:
    arr = np.asarray(values, dtype=float)
    valid = [int(idx) for idx in indices if 0 <= int(idx) < arr.size]
    if not valid:
        return 0.0
    return float(np.linalg.norm(arr[valid]))


def _target_memory_context_key(
    *,
    active_sides: tuple[str, ...],
    solve_mode_label: str,
    lock_passive: bool,
    lock_lift: bool,
    collision_enabled: bool,
    collision_min_distance_m: float,
    com_enabled: bool,
    torso_policy: str,
    torso_contribution: float = 1.0,
) -> tuple[object, ...]:
    return (
        str(solve_mode_label),
        tuple(active_sides),
        bool(lock_passive),
        bool(lock_lift),
        bool(collision_enabled),
        round(float(collision_min_distance_m), 4),
        bool(com_enabled),
        str(torso_policy),
        round(float(torso_contribution), 2),
    )


def run_case(
    *,
    model: ModelSpec,
    scenario: ScenarioSpec,
    context: ContextSpec,
    torso_policy: str,
    memory_enabled: bool,
    steps: int,
    solve_mode_label: str,
    locked_solve_mode_label: str,
    allow_fallback: bool,
    pos_gain: float,
    rot_gain: float,
    collision_min_distance_m: float,
    max_collision_constraints: int,
    collision_tuning_mode: str,
    com_margin_frac: float,
    max_linear_speed: float,
    max_angular_speed: float,
    prefer_locked_tracking_slack_m: float,
    prefer_locked_step_slack: float,
) -> CaseMetrics:
    effective_collision_min_distance_m = (
        float(model.collision_min_distance_m)
        if model.collision_min_distance_m is not None
        else float(collision_min_distance_m)
    )
    effective_max_collision_constraints = (
        int(model.max_collision_constraints)
        if model.max_collision_constraints is not None
        else int(max_collision_constraints)
    )
    fx = build_common_bimanual_solver_fixture(
        model.fixture_variant,
        urdf_path=model.urdf_path,
        collision_urdf_path=model.collision_urdf_path,
        support_contact_frames=model.support_contact_frames,
        lock_joint_names=model.lock_joint_names,
        posture_joint_names=model.posture_joint_names,
        collision_link_pairs=model.collision_link_pairs,
        com_frame_name=model.com_frame_name,
        collision_enabled=context.collision_enabled,
        collision_min_distance_m=effective_collision_min_distance_m,
        max_collision_constraints=effective_max_collision_constraints,
        collision_tuning_mode=collision_tuning_mode,
        configure_com=context.com_enabled,
        com_margin_frac=com_margin_frac,
    )
    robot = fx.robot
    solver = fx.solver
    q = np.asarray(fx.q0, dtype=float).copy()
    robot.update_configuration(q)

    lock_chain_indices = list(fx.lift_velocity_indices)
    chain_locked = torso_policy == "locked"
    dynamic_excluded = _dynamic_excluded_indices(fx, scenario.active_sides, lock_passive=True)
    unlocked_excluded = list(dynamic_excluded)
    locked_excluded = sorted(set(dynamic_excluded + lock_chain_indices))
    _configure_active_tasks(
        fx,
        scenario.active_sides,
        locked_solve_mode_label if chain_locked else solve_mode_label,
        extra_excluded_indices=lock_chain_indices if chain_locked else (),
    )
    active_arm_indices = _active_arm_indices(fx, scenario.active_sides)
    if hasattr(fx.arm_nullspace_task, "set_controlled_joint_indices"):
        fx.arm_nullspace_task.set_controlled_joint_indices(active_arm_indices)
    fx.arm_nullspace_task.weight = float(DEFAULT_ARM_NULLSPACE_WEIGHT)
    fx.posture_task.weight = float(DEFAULT_POSTURE_WEIGHT)

    right_pose0 = _pose_to_matrix(robot.get_frame_pose(fx.frame_map["right_tool"]))
    left_pose0 = _pose_to_matrix(robot.get_frame_pose(fx.frame_map["left_tool"]))
    right_positions, phases = _aba_positions(
        right_pose0[:3, 3], np.asarray(scenario.right_delta, dtype=float), int(steps)
    )
    left_positions, left_phases = _aba_positions(
        left_pose0[:3, 3], np.asarray(scenario.left_delta, dtype=float), int(steps)
    )
    if left_phases != phases:
        raise RuntimeError("internal phase generation mismatch")
    right_rot = right_pose0[:3, :3]
    left_rot = left_pose0[:3, :3]

    inner_polygon = compute_inner_polygon(fx.support_polygon, com_margin_frac)
    memory = _TargetReturnMemory()
    context_key = _target_memory_context_key(
        active_sides=scenario.active_sides,
        solve_mode_label=locked_solve_mode_label if chain_locked else solve_mode_label,
        lock_passive=True,
        lock_lift=chain_locked,
        collision_enabled=context.collision_enabled,
        collision_min_distance_m=effective_collision_min_distance_m,
        com_enabled=context.com_enabled,
        torso_policy=torso_policy,
    )

    unlocked_opts = _build_position_step_options(
        solve_mode_label,
        allow_fallback,
        pos_gain=pos_gain,
        rot_gain=rot_gain,
        stall_recovery=(context.collision_enabled or context.com_enabled),
        max_linear_speed=max_linear_speed,
        max_angular_speed=max_angular_speed,
    )
    unlocked_opts.excluded_joint_indices = list(unlocked_excluded)
    unlocked_opts.integration_zero_velocity_indices = list(unlocked_excluded)
    core_preferred_lock_enabled = (
        torso_policy == "prefer_locked"
        and bool(lock_chain_indices)
        and hasattr(unlocked_opts, "preferred_locked_joint_indices")
    )
    if core_preferred_lock_enabled:
        unlocked_opts.preferred_locked_joint_indices = list(lock_chain_indices)
        unlocked_opts.preferred_lock_tracking_tolerance = float(prefer_locked_tracking_slack_m)
        unlocked_opts.preferred_lock_max_step_norm = float(prefer_locked_step_slack)
        if hasattr(unlocked_opts, "preferred_lock_orientation_tolerance"):
            unlocked_opts.preferred_lock_orientation_tolerance = 0.35
        if hasattr(unlocked_opts, "preferred_lock_solve_mode"):
            unlocked_opts.preferred_lock_solve_mode = getattr(
                embodik.TaskSolveMode,
                locked_solve_mode_label,
                embodik.TaskSolveMode.MIN_ERROR,
            )
    locked_opts = _build_position_step_options(
        locked_solve_mode_label,
        allow_fallback=False,
        pos_gain=pos_gain,
        rot_gain=rot_gain,
        stall_recovery=(context.collision_enabled or context.com_enabled),
        max_linear_speed=max_linear_speed,
        max_angular_speed=max_angular_speed,
    )
    locked_opts.excluded_joint_indices = list(locked_excluded)
    locked_opts.integration_zero_velocity_indices = list(locked_excluded)
    opts = locked_opts if chain_locked else unlocked_opts

    def _solve_candidate(q_seed: np.ndarray, *, locked: bool):
        _configure_active_tasks(
            fx,
            scenario.active_sides,
            locked_solve_mode_label if locked else solve_mode_label,
            extra_excluded_indices=lock_chain_indices if locked else (),
        )
        return robust_solve_position_step(
            solver=solver,
            q_current=q_seed,
            targets=targets,
            options=locked_opts if locked else unlocked_opts,
        )

    q_series: list[np.ndarray] = []
    health_samples = []
    return_health_samples = []
    return_errors: list[float] = []
    return_q_distances: list[float] = []
    solver_ms: list[float] = []
    wall_ms: list[float] = []
    memory_ms: list[float] = []
    collision_distances: list[float] = []
    com_inner_slacks: list[float] = []
    torso_motion_norms: list[float] = []
    arm_motion_norms: list[float] = []
    locked_chain_velocity_norms: list[float] = []
    status_counts: Counter[str] = Counter()
    memory_hits = 0
    memory_stores = 0
    memory_rejects = 0
    max_memory_entry_count = 0
    prefer_locked_attempts = 0
    prefer_locked_used = 0
    solve_attempt_total = 0
    initial_a_reference_q: np.ndarray | None = None
    final_max_error = 0.0

    for step_idx in range(int(steps)):
        phase = phases[step_idx]
        in_hold_phase = phase in HOLD_PHASES
        right_target_pose = _pose_from_position(right_positions[step_idx], right_rot)
        left_target_pose = _pose_from_position(left_positions[step_idx], left_rot)
        active_target_poses = {
            "right": right_target_pose.copy(),
            "left": left_target_pose.copy(),
        }

        remembered_bias_q = np.asarray(fx.q0, dtype=float).copy()
        mem_start = time.perf_counter()
        if memory_enabled and in_hold_phase:
            entry = memory.lookup(
                context_key=context_key,
                active_sides=scenario.active_sides,
                target_poses=active_target_poses,
                q_current=q,
                q_lower=fx.q_lo,
                q_upper=fx.q_hi,
            )
            if entry is not None:
                memory_hits += 1
                remembered_bias_q = memory.bias_configuration(entry, q)
        memory_elapsed = (time.perf_counter() - mem_start) * 1e3

        fx.posture_task.set_target_configuration(remembered_bias_q)
        fx.arm_nullspace_task.set_target_configuration(remembered_bias_q)

        targets = []
        if "right" in scenario.active_sides:
            targets.append(
                _make_pose_target("right_tool_pose", right_target_pose, pos_gain, rot_gain)
            )
        if "left" in scenario.active_sides:
            targets.append(
                _make_pose_target("left_tool_pose", left_target_pose, pos_gain, rot_gain)
            )

        t0 = time.perf_counter()
        attempts_counted = False
        if (
            torso_policy == "prefer_locked"
            and lock_chain_indices
            and not core_preferred_lock_enabled
        ):
            prefer_locked_attempts += 1
            solve_attempt_total += 1
            attempts_counted = True
            locked_step = _solve_candidate(q, locked=True)
            locked_result = locked_step.solver_result
            locked_status = _status_name(locked_result)
            q_locked = np.asarray(locked_step.q_next, dtype=float)
            locked_usable = bool(np.all(np.isfinite(q_locked)))
            locked_error = float("inf")
            locked_step_norm = float("inf")
            if locked_usable and locked_status in {"SUCCESS", "NO_PROGRESS"}:
                robot.update_configuration(q_locked)
                locked_error, _ = _active_pose_errors(
                    robot, fx.frame_map, scenario.active_sides, active_target_poses
                )
                locked_step_norm = float(np.linalg.norm(q_locked - q))
                robot.update_configuration(q)
            if (
                locked_usable
                and locked_status in {"SUCCESS", "NO_PROGRESS"}
                and locked_error <= float(prefer_locked_tracking_slack_m)
                and (
                    float(prefer_locked_step_slack) <= 0.0
                    or locked_step_norm <= float(prefer_locked_step_slack)
                )
            ):
                step = locked_step
                prefer_locked_used += 1
            else:
                solve_attempt_total += 1
                step = _solve_candidate(q, locked=False)
        else:
            step = _solve_candidate(q, locked=chain_locked)
        wall_elapsed = (time.perf_counter() - t0) * 1e3
        result = step.solver_result
        status = _status_name(result)
        status_counts[status] += 1
        if not attempts_counted:
            diagnostic_source = getattr(result, "diagnostics", None) or result
            if bool(getattr(diagnostic_source, "preferred_lock_attempted", False)):
                prefer_locked_attempts += 1
                if bool(getattr(diagnostic_source, "preferred_lock_used", False)):
                    prefer_locked_used += 1
                    solve_attempt_total += 1
                else:
                    solve_attempt_total += 2
            else:
                solve_attempt_total += 1

        q_prev = np.asarray(q, dtype=float).copy()
        q = np.asarray(step.q_next, dtype=float)
        if not np.all(np.isfinite(q)):
            q = q_prev
        if status in {"INFEASIBLE", "NO_PROGRESS"}:
            q = q_prev
        robot.update_configuration(q)
        q_delta = q - q_prev
        torso_motion_norms.append(_indexed_norm(q_delta, lock_chain_indices))
        arm_motion_norms.append(_indexed_norm(q_delta, active_arm_indices))
        locked_chain_velocity_norms.append(
            _indexed_norm(np.asarray(result.joint_velocities, dtype=float), lock_chain_indices)
        )

        max_error, _side_errors = _active_pose_errors(
            robot, fx.frame_map, scenario.active_sides, active_target_poses
        )
        final_max_error = max_error

        mem_start = time.perf_counter()
        if memory_enabled and in_hold_phase and _memory_result_is_clean(result, status):
            stored = memory.observe(
                context_key=context_key,
                active_sides=scenario.active_sides,
                target_poses=active_target_poses,
                q_solution=q,
                q_lower=fx.q_lo,
                q_upper=fx.q_hi,
                health_score=_memory_health_score(result),
                max_position_error_m=max_error,
                max_rotation_error_rad=0.0,
            )
            if stored:
                memory_stores += 1
            else:
                memory_rejects += 1
        memory_elapsed += (time.perf_counter() - mem_start) * 1e3
        max_memory_entry_count = max(max_memory_entry_count, int(memory.entry_count))

        if phase == "hold_a_initial":
            initial_a_reference_q = q.copy()
        if initial_a_reference_q is None and phase != "hold_a_initial":
            initial_a_reference_q = q.copy()

        jacobian = _combined_active_jacobian(fx, scenario.active_sides)
        sample = kinematic_health_sample(robot, q, jacobian)
        health_samples.append(sample)
        if phase == "hold_a_return":
            return_health_samples.append(sample)
            return_errors.append(max_error)
            return_q_distances.append(float(np.linalg.norm(q - initial_a_reference_q)))

        if context.collision_enabled:
            collision_distances.append(_evaluate_collision_min_distance(solver, q))
        if context.com_enabled:
            com_inner_slacks.append(_evaluate_com_inner_slack(robot, inner_polygon))

        q_series.append(q.copy())
        solver_ms.append(float(getattr(step, "elapsed_ms", wall_elapsed)))
        wall_ms.append(wall_elapsed)
        memory_ms.append(memory_elapsed)

        dq_norm = float(np.linalg.norm(np.asarray(result.joint_velocities, dtype=float)))
        if dq_norm < 1e-5 and max_error > EE_POSITION_DEADBAND and step_idx % 30 == 0:
            clear_all_target_velocities_if_available(solver)

    solver_stats = _stats(solver_ms)
    wall_stats = _stats(wall_ms)
    memory_stats = _stats(memory_ms)
    return_error_stats = _stats(return_errors)
    return_distance_stats = _stats(return_q_distances)
    return_health = summarize_health_series(return_health_samples)
    full_health = summarize_health_series(health_samples)
    continuity = summarize_continuity_series(
        q_series,
        dt=0.01,
        velocity_limits=np.asarray(robot.get_velocity_limits(), dtype=float),
        q_lower=fx.q_lo,
        q_upper=fx.q_hi,
    )
    torso_motion_stats = _stats(torso_motion_norms)
    arm_motion_stats = _stats(arm_motion_norms)
    ratios = [
        float(torso / max(arm, 1e-12))
        for torso, arm in zip(torso_motion_norms, arm_motion_norms, strict=False)
    ]
    ratio_stats = _stats(ratios)
    locked_velocity_stats = _stats(locked_chain_velocity_norms)

    min_collision = None
    finite_collision = [v for v in collision_distances if np.isfinite(v)]
    if finite_collision:
        min_collision = float(min(finite_collision))
    min_com = None
    finite_com = [v for v in com_inner_slacks if np.isfinite(v)]
    if finite_com:
        min_com = float(min(finite_com))

    return CaseMetrics(
        variant=str(model.name),
        scenario=scenario.name,
        context=context.name,
        torso_policy=str(torso_policy),
        memory_enabled=bool(memory_enabled),
        steps=int(steps),
        status_counts=dict(status_counts),
        memory_hits=int(memory_hits),
        memory_stores=int(memory_stores),
        memory_rejects=int(memory_rejects),
        memory_entry_count=int(memory.entry_count),
        max_memory_entry_count=int(max_memory_entry_count),
        prefer_locked_attempts=int(prefer_locked_attempts),
        prefer_locked_used=int(prefer_locked_used),
        avg_solve_attempts_per_step=float(solve_attempt_total / max(1, int(steps))),
        avg_solver_ms=solver_stats["mean"],
        p95_solver_ms=solver_stats["p95"],
        avg_wall_ms=wall_stats["mean"],
        p95_wall_ms=wall_stats["p95"],
        avg_memory_ms=memory_stats["mean"],
        p95_memory_ms=memory_stats["p95"],
        final_max_error_m=float(final_max_error),
        return_mean_max_error_m=return_error_stats["mean"],
        return_p95_max_error_m=return_error_stats["p95"],
        return_final_q_distance_to_initial_a=(
            float(return_q_distances[-1]) if return_q_distances else 0.0
        ),
        return_mean_q_distance_to_initial_a=return_distance_stats["mean"],
        return_joint_limit_health_mean=float(return_health.get("joint_limit_health_mean", 0.0)),
        return_joint_limit_health_p05=float(return_health.get("joint_limit_health_p05", 0.0)),
        return_limit_weighted_dexterity_mean=float(
            return_health.get("limit_weighted_dexterity_mean", 0.0)
        ),
        return_limit_weighted_dexterity_p05=float(
            return_health.get("limit_weighted_dexterity_p05", 0.0)
        ),
        full_joint_limit_health_mean=float(full_health.get("joint_limit_health_mean", 0.0)),
        full_limit_weighted_dexterity_mean=float(
            full_health.get("limit_weighted_dexterity_mean", 0.0)
        ),
        min_collision_distance_m=min_collision,
        min_com_inner_slack_m=min_com,
        torso_motion_mean=torso_motion_stats["mean"],
        torso_motion_p95=torso_motion_stats["p95"],
        torso_motion_max=torso_motion_stats["max"],
        arm_motion_mean=arm_motion_stats["mean"],
        arm_motion_p95=arm_motion_stats["p95"],
        arm_motion_max=arm_motion_stats["max"],
        torso_arm_motion_ratio_mean=ratio_stats["mean"],
        locked_chain_velocity_max=locked_velocity_stats["max"],
        continuity=continuity,
    )


def _expand_selection(values: list[str], aliases: dict[str, tuple[str, ...]]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value in aliases:
            out.extend(aliases[value])
        else:
            out.append(value)
    return list(dict.fromkeys(out))


def _build_pairs(rows: list[CaseMetrics]) -> list[dict[str, object]]:
    by_key: dict[tuple[str, str, str, str, bool], CaseMetrics] = {}
    for row in rows:
        key = (row.variant, row.scenario, row.context, row.torso_policy, row.memory_enabled)
        by_key[key] = row
    pairs: list[dict[str, object]] = []
    for row in rows:
        if row.memory_enabled:
            continue
        key = (row.variant, row.scenario, row.context, row.torso_policy)
        on = by_key.get((*key, True))
        if on is None:
            continue
        off = row
        tracking_regressed = on.return_mean_max_error_m > off.return_mean_max_error_m + 0.005
        helped_q_return = (
            on.return_final_q_distance_to_initial_a
            < off.return_final_q_distance_to_initial_a - 1e-5
        )
        helped_health = (
            on.return_joint_limit_health_mean > off.return_joint_limit_health_mean + 1e-6
            or on.return_limit_weighted_dexterity_mean
            > off.return_limit_weighted_dexterity_mean + 1e-9
        )
        pairs.append(
            {
                "variant": key[0],
                "scenario": key[1],
                "context": key[2],
                "torso_policy": key[3],
                "memory_hits": on.memory_hits,
                "memory_stores": on.memory_stores,
                "memory_entry_count": on.memory_entry_count,
                "max_memory_entry_count": on.max_memory_entry_count,
                "avg_solve_attempts_per_step": on.avg_solve_attempts_per_step,
                "prefer_locked_attempts": on.prefer_locked_attempts,
                "prefer_locked_used": on.prefer_locked_used,
                "return_final_q_distance_delta": (
                    on.return_final_q_distance_to_initial_a
                    - off.return_final_q_distance_to_initial_a
                ),
                "return_mean_error_delta_m": on.return_mean_max_error_m
                - off.return_mean_max_error_m,
                "return_joint_limit_health_delta": on.return_joint_limit_health_mean
                - off.return_joint_limit_health_mean,
                "return_limit_weighted_dexterity_delta": (
                    on.return_limit_weighted_dexterity_mean
                    - off.return_limit_weighted_dexterity_mean
                ),
                "avg_wall_ms_delta": on.avg_wall_ms - off.avg_wall_ms,
                "avg_memory_ms_on": on.avg_memory_ms,
                "tracking_regressed": tracking_regressed,
                "helped": (not tracking_regressed) and (helped_q_return or helped_health),
            }
        )
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["sg2", "bg2", "rby1", "alpha_wheelbase"])
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["all"],
        choices=tuple(SCENARIO_ALIASES["all"]) + tuple(SCENARIO_ALIASES.keys()),
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        default=["all"],
        choices=tuple(CONTEXT_ALIASES["all"]) + ("all",),
    )
    parser.add_argument(
        "--torso-policies",
        nargs="+",
        default=["all"],
        choices=tuple(TORSO_POLICY_ALIASES["all"]) + ("all",),
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument(
        "--solve-mode",
        default="SCALE_ELASTIC",
        choices=("SCALE", "SCALE_ELASTIC", "MIN_ERROR"),
    )
    parser.add_argument(
        "--locked-solve-mode",
        default="MIN_ERROR",
        choices=("SCALE", "SCALE_ELASTIC", "MIN_ERROR"),
    )
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--pos-gain", type=float, default=10.0)
    parser.add_argument("--rot-gain", type=float, default=10.0)
    parser.add_argument("--collision-min-distance", type=float, default=0.035)
    parser.add_argument("--max-collision-constraints", type=int, default=3)
    parser.add_argument(
        "--collision-tuning-mode", default="balanced", choices=("speed", "balanced", "precise")
    )
    parser.add_argument("--com-margin-pct", type=float, default=10.0)
    parser.add_argument("--max-linear-speed", type=float, default=DEFAULT_MAX_LINEAR_SPEED)
    parser.add_argument("--max-angular-speed", type=float, default=DEFAULT_MAX_ANGULAR_SPEED)
    parser.add_argument(
        "--prefer-locked-tracking-slack",
        type=float,
        default=0.035,
        help="Use the locked candidate in prefer_locked mode when max active EEF error is below this value.",
    )
    parser.add_argument(
        "--prefer-locked-step-slack",
        type=float,
        default=0.35,
        help="Reject prefer_locked candidates whose configuration step norm exceeds this value; <=0 disables this gate.",
    )
    parser.add_argument("--rby1-urdf", type=str, default=None)
    parser.add_argument("--rby1-collision-urdf", type=str, default=None)
    parser.add_argument("--alpha-urdf", type=str, default=None)
    parser.add_argument("--alpha-collision-urdf", type=str, default=None)
    parser.add_argument(
        "--alpha-collisions-json",
        nargs="*",
        default=[],
        help="One or more alpha-wheelbase collision link-pair JSON files; entries are unioned.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".editor/reports/target_memory_ab_matrix.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenario_map = _scenario_specs()
    context_map = _context_specs()
    scenarios = [scenario_map[name] for name in _expand_selection(args.scenarios, SCENARIO_ALIASES)]
    contexts = [context_map[name] for name in _expand_selection(args.contexts, CONTEXT_ALIASES)]
    torso_policies = _expand_selection(args.torso_policies, TORSO_POLICY_ALIASES)
    model_specs, model_skips = _resolve_model_specs(args, contexts)

    rows: list[CaseMetrics] = []
    skipped: list[dict[str, str]] = list(model_skips)
    total = len(model_specs) * len(scenarios) * len(contexts) * len(torso_policies) * 2
    case_index = 0
    for model in model_specs:
        for scenario in scenarios:
            for context in contexts:
                for torso_policy in torso_policies:
                    for memory_enabled in (False, True):
                        case_index += 1
                        label = (
                            f"{model.name}/{scenario.name}/{context.name}/{torso_policy}/"
                            f"memory={'on' if memory_enabled else 'off'}"
                        )
                        print(f"[{case_index:03d}/{total:03d}] {label}", flush=True)
                        try:
                            row = run_case(
                                model=model,
                                scenario=scenario,
                                context=context,
                                torso_policy=str(torso_policy),
                                memory_enabled=memory_enabled,
                                steps=int(args.steps),
                                solve_mode_label=str(args.solve_mode),
                                locked_solve_mode_label=str(args.locked_solve_mode),
                                allow_fallback=bool(args.allow_fallback),
                                pos_gain=float(args.pos_gain),
                                rot_gain=float(args.rot_gain),
                                collision_min_distance_m=float(args.collision_min_distance),
                                max_collision_constraints=int(args.max_collision_constraints),
                                collision_tuning_mode=str(args.collision_tuning_mode),
                                com_margin_frac=float(args.com_margin_pct) / 100.0,
                                max_linear_speed=float(args.max_linear_speed),
                                max_angular_speed=float(args.max_angular_speed),
                                prefer_locked_tracking_slack_m=float(
                                    args.prefer_locked_tracking_slack
                                ),
                                prefer_locked_step_slack=float(args.prefer_locked_step_slack),
                            )
                            rows.append(row)
                            print(
                                "  "
                                f"hits={row.memory_hits} stores={row.memory_stores} "
                                f"entries={row.memory_entry_count}/{row.max_memory_entry_count} "
                                f"prefer={row.prefer_locked_used}/{row.prefer_locked_attempts} "
                                f"return_err={row.return_mean_max_error_m:.5f} "
                                f"qA={row.return_final_q_distance_to_initial_a:.5f} "
                                f"jl={row.return_joint_limit_health_mean:.6f} "
                                f"dex={row.return_limit_weighted_dexterity_mean:.6g} "
                                f"torso={row.torso_motion_mean:.5f} "
                                f"arm={row.arm_motion_mean:.5f} "
                                f"wall={row.avg_wall_ms:.3f}ms mem={row.avg_memory_ms:.4f}ms",
                                flush=True,
                            )
                        except Exception as exc:
                            skipped.append(
                                {
                                    "variant": str(model.name),
                                    "scenario": scenario.name,
                                    "context": context.name,
                                    "torso_policy": str(torso_policy),
                                    "memory_enabled": str(memory_enabled),
                                    "reason": repr(exc),
                                }
                            )
                            print(f"  skipped: {exc!r}", flush=True)

    pairs = _build_pairs(rows)
    helped = sum(1 for pair in pairs if bool(pair["helped"]))
    tracking_regressed = sum(1 for pair in pairs if bool(pair["tracking_regressed"]))
    report = {
        "config": {
            "variants": list(args.variants),
            "resolved_models": [model.name for model in model_specs],
            "scenarios": [scenario.name for scenario in scenarios],
            "contexts": [context.name for context in contexts],
            "torso_policies": list(torso_policies),
            "steps": int(args.steps),
            "solve_mode": str(args.solve_mode),
            "locked_solve_mode": str(args.locked_solve_mode),
            "allow_fallback": bool(args.allow_fallback),
            "pos_gain": float(args.pos_gain),
            "rot_gain": float(args.rot_gain),
            "collision_min_distance": float(args.collision_min_distance),
            "max_collision_constraints": int(args.max_collision_constraints),
            "collision_tuning_mode": str(args.collision_tuning_mode),
            "com_margin_pct": float(args.com_margin_pct),
            "max_linear_speed": float(args.max_linear_speed),
            "max_angular_speed": float(args.max_angular_speed),
            "prefer_locked_tracking_slack": float(args.prefer_locked_tracking_slack),
            "prefer_locked_step_slack": float(args.prefer_locked_step_slack),
        },
        "rows": [asdict(row) for row in rows],
        "pairs": pairs,
        "summary": {
            "row_count": len(rows),
            "pair_count": len(pairs),
            "helped_pair_count": helped,
            "tracking_regressed_pair_count": tracking_regressed,
            "skipped_count": len(skipped),
            "mean_avg_memory_ms_on": (
                float(np.mean([row.avg_memory_ms for row in rows if row.memory_enabled]))
                if any(row.memory_enabled for row in rows)
                else 0.0
            ),
            "mean_avg_wall_ms_delta": (
                float(np.mean([pair["avg_wall_ms_delta"] for pair in pairs])) if pairs else 0.0
            ),
        },
        "skipped": skipped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("\nA/B pair summary:")
    for pair in pairs:
        print(
            f"{pair['variant']}/{pair['scenario']}/{pair['context']}/{pair['torso_policy']}: "
            f"helped={pair['helped']} hits={pair['memory_hits']} stores={pair['memory_stores']} "
            f"entries={pair['memory_entry_count']}/{pair['max_memory_entry_count']} "
            f"prefer={pair['prefer_locked_used']}/{pair['prefer_locked_attempts']} "
            f"attempts={pair['avg_solve_attempts_per_step']:.2f} "
            f"dqA={pair['return_final_q_distance_delta']:+.6f} "
            f"derr={pair['return_mean_error_delta_m']:+.6f} "
            f"djl={pair['return_joint_limit_health_delta']:+.6f} "
            f"ddex={pair['return_limit_weighted_dexterity_delta']:+.6g} "
            f"dwall={pair['avg_wall_ms_delta']:+.4f}ms "
            f"mem={pair['avg_memory_ms_on']:.4f}ms"
        )
    print(f"\nWrote {args.output}")
    print(
        "Summary: "
        f"rows={report['summary']['row_count']} pairs={report['summary']['pair_count']} "
        f"helped={helped} tracking_regressed={tracking_regressed} "
        f"skipped={len(skipped)} mean_mem_ms={report['summary']['mean_avg_memory_ms_on']:.4f} "
        f"mean_wall_delta_ms={report['summary']['mean_avg_wall_ms_delta']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
