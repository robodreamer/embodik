#!/usr/bin/env python3
"""Headless robustness harness for the ROBOTIS AI worker example.

Runs scripted target trajectories through the same solver configuration the
interactive example uses and reports stall / infeasible / constraint-breach
counts. Designed to be driven by ``autoresearch:fix`` so every fix in
``robotis_ai_worker_ik.py`` has a numeric signal.

Usage::

    pixi run python scripts/verify_ai_worker_robustness.py \\
        --variant sg2 --scenario all --output-json /tmp/ai_worker_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import embodik  # noqa: E402

from examples.example_helpers.ai_worker_solver_fixture import (  # noqa: E402
    build_worker_solver_fixture,
    compute_inner_polygon,
)
from examples.example_helpers.ai_worker_constraint_teleop_app import (  # noqa: E402
    EE_POSITION_DEADBAND,
    _polygon_slack,
)
from examples.example_helpers.robust_ik_runtime import (  # noqa: E402
    clear_all_target_velocities_if_available,
    configure_primary_solve_mode,
    robust_solve_position_step,
)

SCENARIO_NAMES = (
    "collision_sweep",
    "com_edge",
    "joint_limit_chase",
    "collision_zigzag",
    "combined_stress",
    "target_in_torso",
    "target_in_torso_release",
)
DQ_STALL_EPS = 1e-5
UNRECOVERABLE_WINDOW = 30
UNRECOVERABLE_STALL_FRAMES = 25
ZERO_MOTION_EPS = 1e-8


@dataclass
class StepRecord:
    step: int
    status_name: str
    dq_norm: float
    right_pos_err: float
    left_pos_err: float
    min_collision_distance: float
    com_slack_outer: float
    com_slack_inner: float
    relaxed_margin: float
    elapsed_ms: float


@dataclass
class ScenarioMetrics:
    scenario: str
    total_steps: int = 0
    success_count: int = 0
    stall_count: int = 0
    zero_motion_count: int = 0
    infeasible_count: int = 0
    no_progress_count: int = 0
    collision_violated_count: int = 0
    collision_breach_count: int = 0
    com_breach_count: int = 0
    unrecoverable_stall_count: int = 0
    escape_events: int = 0
    avg_relaxed_collision_margin_mm: float = 0.0
    avg_solve_ms: float = 0.0
    final_right_position_error_m: float = 0.0
    final_left_position_error_m: float = 0.0
    peak_right_position_error_m: float = 0.0
    peak_left_position_error_m: float = 0.0
    late_stall_count: int = 0  # stalls during the second half of the scenario (recovery phase)
    min_collision_distance_m: float = float("inf")
    min_com_inner_slack_m: float = float("inf")
    step_records: list[StepRecord] = field(default_factory=list)

    def as_dict(self) -> dict:
        record = {
            "scenario": self.scenario,
            "total_steps": self.total_steps,
            "success_count": self.success_count,
            "stall_count": self.stall_count,
            "zero_motion_count": self.zero_motion_count,
            "infeasible_count": self.infeasible_count,
            "no_progress_count": self.no_progress_count,
            "collision_violated_count": self.collision_violated_count,
            "collision_breach_count": self.collision_breach_count,
            "com_breach_count": self.com_breach_count,
            "unrecoverable_stall_count": self.unrecoverable_stall_count,
            "escape_events": self.escape_events,
            "avg_relaxed_collision_margin_mm": round(self.avg_relaxed_collision_margin_mm, 4),
            "avg_solve_ms": round(self.avg_solve_ms, 4),
            "final_right_position_error_m": round(self.final_right_position_error_m, 6),
            "final_left_position_error_m": round(self.final_left_position_error_m, 6),
            "peak_right_position_error_m": round(self.peak_right_position_error_m, 6),
            "peak_left_position_error_m": round(self.peak_left_position_error_m, 6),
            "late_stall_count": self.late_stall_count,
            "min_collision_distance_m": (
                round(self.min_collision_distance_m, 6)
                if np.isfinite(self.min_collision_distance_m)
                else None
            ),
            "min_com_inner_slack_m": (
                round(self.min_com_inner_slack_m, 6)
                if np.isfinite(self.min_com_inner_slack_m)
                else None
            ),
        }
        record["hard_failures"] = (
            self.unrecoverable_stall_count
            + self.collision_breach_count
            + self.com_breach_count
            + self.late_stall_count
        )
        return record


def _status_name(result) -> str:
    status = getattr(result, "status", None)
    name = getattr(status, "name", None)
    if name is None:
        return str(status)
    return str(name)


def _pose_to_matrix(pose) -> np.ndarray:
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = np.asarray(pose.rotation, dtype=float)
    mat[:3, 3] = np.asarray(pose.translation, dtype=float)
    return mat


def _make_pose_target(
    task_name: str,
    position: np.ndarray,
    rotation: np.ndarray,
    pos_gain: float,
    rot_gain: float,
):
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = np.asarray(rotation, dtype=float)
    pose[:3, 3] = np.asarray(position, dtype=float)
    return embodik.TaskTarget(task_name, pose, float(pos_gain), float(rot_gain))


def _build_position_step_options(
    solve_mode_label: str,
    allow_fallback: bool,
    pos_gain: float,
    rot_gain: float,
    stall_recovery: bool,
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
    solve_mode = getattr(embodik.TaskSolveMode, solve_mode_label, embodik.TaskSolveMode.SCALE)
    configure_primary_solve_mode(opts, solve_mode, bool(allow_fallback))
    return opts


def _trajectory_for_scenario(
    scenario: str,
    *,
    right_start_pose: np.ndarray,
    left_start_pose: np.ndarray,
    duration_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Ramp targets to an impossible extreme within the first 20% of steps, then
    HOLD there — mimicking a user dragging a target into the body and leaving it.
    This is what reproduces the "unrecoverable stall" the solver falls into.
    """
    n = int(duration_steps)
    ramp_steps = max(1, n // 5)
    t = np.arange(n, dtype=float)
    ramp = np.clip(t / ramp_steps, 0.0, 1.0)
    ramp = 0.5 - 0.5 * np.cos(np.pi * ramp)  # smoothstep, then holds at 1.0

    right_pos = np.tile(right_start_pose[:3, 3].copy(), (n, 1))
    left_pos = np.tile(left_start_pose[:3, 3].copy(), (n, 1))

    if scenario == "collision_sweep":
        # Drive right EE 40cm PAST the left EE (deep into/through the left arm).
        direction = left_start_pose[:3, 3] - right_start_pose[:3, 3]
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > 1e-9:
            unit = direction / direction_norm
        else:
            unit = np.array([0.0, -1.0, 0.0], dtype=float)
        overshoot = direction_norm + 0.40
        right_pos = right_start_pose[:3, 3][None, :] + (ramp * overshoot)[:, None] * unit[None, :]
    elif scenario == "com_edge":
        # Both EEs reach 1.0 m forward — torso must lean far past the polygon.
        forward = np.array([1.0, 0.0, 0.0], dtype=float)
        reach = ramp * 1.0
        right_pos = right_start_pose[:3, 3][None, :] + reach[:, None] * forward[None, :]
        left_pos = left_start_pose[:3, 3][None, :] + reach[:, None] * forward[None, :]
    elif scenario == "joint_limit_chase":
        # Drive right EE 1.2 m straight up — well past reachable workspace.
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        right_pos = right_start_pose[:3, 3][None, :] + (ramp * 1.2)[:, None] * up[None, :]
    elif scenario == "collision_zigzag":
        # Zigzag toward left EE then back — mimics a user dragging the target
        # across the self-collision boundary repeatedly. This is the pattern
        # most likely to leave the solver in an oscillating stall.
        direction = left_start_pose[:3, 3] - right_start_pose[:3, 3]
        direction_norm = float(np.linalg.norm(direction))
        unit = direction / direction_norm if direction_norm > 1e-9 else np.array([0.0, -1.0, 0.0], dtype=float)
        # Zigzag amplitude alternates between 0 and direction_norm + 30cm every 40 steps.
        period = 40
        phase = np.floor(t / period) % 2.0  # 0, 1, 0, 1, ...
        zigzag = phase * (direction_norm + 0.30)
        right_pos = right_start_pose[:3, 3][None, :] + zigzag[:, None] * unit[None, :]
    elif scenario == "combined_stress":
        # Push-then-release: both constraints engage at extreme, then target
        # returns to start. Tests both saturation AND recovery. The solver must
        # resume motion promptly once the target becomes feasible again.
        forward = np.array([1.0, 0.0, 0.0], dtype=float)
        direction = left_start_pose[:3, 3] - right_start_pose[:3, 3]
        direction_norm = float(np.linalg.norm(direction))
        unit = direction / direction_norm if direction_norm > 1e-9 else np.array([0.0, -1.0, 0.0], dtype=float)
        # push during first half, release to 0 during second half
        profile = np.where(
            t < (n * 0.5),
            np.clip(t / (n * 0.1), 0.0, 1.0),
            np.clip(1.0 - (t - n * 0.5) / (n * 0.15), 0.0, 1.0),
        )
        profile = 0.5 - 0.5 * np.cos(np.pi * profile)
        right_pos = (
            right_start_pose[:3, 3][None, :]
            + (profile * 0.6)[:, None] * forward[None, :]
            + (profile * (direction_norm + 0.25))[:, None] * unit[None, :]
        )
        left_pos = left_start_pose[:3, 3][None, :] + (profile * 0.6)[:, None] * forward[None, :]
    elif scenario == "target_in_torso":
        # Mirror the observed GUI bug: user drags right EE target INTO the
        # torso/neck volume. Target sits inside the robot's self-collision
        # envelope, so any motion toward it is immediately blocked. Held there
        # for the full run to exercise sustained unreachable target behavior.
        midline = np.array([left_start_pose[0, 3], 0.0, right_start_pose[2, 3]], dtype=float)
        offset_into_body = np.array([-0.15, 0.0, 0.05], dtype=float)
        torso_target = midline + offset_into_body
        delta = torso_target - right_start_pose[:3, 3]
        right_pos = right_start_pose[:3, 3][None, :] + ramp[:, None] * delta[None, :]
    elif scenario == "target_in_torso_release":
        # Same as target_in_torso, but release the target back to start after
        # 250 steps so we can verify the solver RESUMES after the stall.
        midline = np.array([left_start_pose[0, 3], 0.0, right_start_pose[2, 3]], dtype=float)
        offset_into_body = np.array([-0.15, 0.0, 0.05], dtype=float)
        torso_target = midline + offset_into_body
        delta = torso_target - right_start_pose[:3, 3]
        profile = np.where(
            t < (n * 0.5),
            np.clip(t / (n * 0.1), 0.0, 1.0),
            np.clip(1.0 - (t - n * 0.5) / (n * 0.15), 0.0, 1.0),
        )
        profile = 0.5 - 0.5 * np.cos(np.pi * profile)
        right_pos = right_start_pose[:3, 3][None, :] + profile[:, None] * delta[None, :]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return right_pos, left_pos


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


def _evaluate_com_slacks(
    robot, support_polygon: np.ndarray, inner_polygon: np.ndarray
) -> tuple[float, float]:
    com_xy = np.asarray(robot.get_com_position(), dtype=float)[:2]
    slacks_outer = _polygon_slack(support_polygon, com_xy)
    slacks_inner = _polygon_slack(inner_polygon, com_xy)
    return (
        float(slacks_outer.min()) if slacks_outer.size else float("inf"),
        float(slacks_inner.min()) if slacks_inner.size else float("inf"),
    )


def _relaxed_margin_mm(solver) -> float:
    if not hasattr(solver, "stall_handler_current_min_distance"):
        return float("nan")
    try:
        return float(solver.stall_handler_current_min_distance()) * 1e3
    except Exception:
        return float("nan")


def run_scenario(
    scenario: str,
    *,
    variant: str,
    duration_steps: int,
    collision_min_distance_m: float,
    max_collision_constraints: int,
    com_margin_frac: float,
    collision_tuning_mode: str,
    solve_mode_label: str,
    allow_fallback: bool,
    pos_gain: float,
    rot_gain: float,
    enable_collision: bool,
    enable_com: bool,
    fallback_status_names: tuple[str, ...],
    verbose: bool = False,
) -> ScenarioMetrics:
    fx = build_worker_solver_fixture(
        variant,
        collision_enabled=enable_collision,
        collision_min_distance_m=collision_min_distance_m,
        max_collision_constraints=max_collision_constraints,
        collision_tuning_mode=collision_tuning_mode,
        configure_com=enable_com,
        com_margin_frac=com_margin_frac,
    )

    robot = fx.robot
    solver = fx.solver
    q = fx.q0.copy()
    robot.update_configuration(q)

    right_pose0 = _pose_to_matrix(robot.get_frame_pose(fx.frame_map["right_tool"]))
    left_pose0 = _pose_to_matrix(robot.get_frame_pose(fx.frame_map["left_tool"]))
    right_positions, left_positions = _trajectory_for_scenario(
        scenario,
        right_start_pose=right_pose0,
        left_start_pose=left_pose0,
        duration_steps=duration_steps,
    )
    right_rot = right_pose0[:3, :3]
    left_rot = left_pose0[:3, :3]

    inner_polygon = compute_inner_polygon(fx.support_polygon, com_margin_frac)

    opts = _build_position_step_options(
        solve_mode_label,
        allow_fallback,
        pos_gain=pos_gain,
        rot_gain=rot_gain,
        stall_recovery=(enable_collision or enable_com),
    )
    active_solve_mode = getattr(embodik.TaskSolveMode, solve_mode_label, embodik.TaskSolveMode.SCALE)
    for task in (fx.right_task, fx.left_task):
        if hasattr(task, "solve_mode"):
            task.solve_mode = active_solve_mode
        if hasattr(task, "allow_min_error_fallback"):
            task.allow_min_error_fallback = bool(allow_fallback)

    metrics = ScenarioMetrics(scenario=scenario)
    stall_window: deque[bool] = deque(maxlen=UNRECOVERABLE_WINDOW)
    cumulative_relaxed_mm = 0.0
    relaxed_mm_count = 0
    cumulative_solve_ms = 0.0

    for step_idx in range(int(duration_steps)):
        right_target = _make_pose_target(
            "right_tool_pose", right_positions[step_idx], right_rot, pos_gain, rot_gain
        )
        left_target = _make_pose_target(
            "left_tool_pose", left_positions[step_idx], left_rot, pos_gain, rot_gain
        )
        fx.posture_task.set_target_configuration(np.asarray(fx.q0, dtype=float))

        targets = [right_target, left_target]

        t0 = time.perf_counter()
        step = robust_solve_position_step(
            robot=robot,
            solver=solver,
            q_current=q,
            targets=targets,
            options=opts,
            q_lo=fx.q_lo,
            q_hi=fx.q_hi,
            zero_velocity_indices=fx.locked_velocity_indices,
            fallback_status_names=fallback_status_names,
            allow_solver_intervention=True,
            apply_collision_violated_q_solution=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        cumulative_solve_ms += elapsed_ms

        result = step.solver_result
        q_prev = np.asarray(q, dtype=float).copy()
        q = np.asarray(step.q_next, dtype=float)
        if not np.all(np.isfinite(q)):
            q = q_prev
        status_name = _status_name(result)
        # Mirror the app: hold last-known-good q on unresolvable statuses.
        if status_name in {"INFEASIBLE", "NO_PROGRESS"}:
            q = q_prev
        robot.update_configuration(q)
        dq_norm = float(np.linalg.norm(np.asarray(result.joint_velocities, dtype=float)))

        right_now = np.asarray(robot.get_frame_pose(fx.frame_map["right_tool"]).translation, dtype=float)
        left_now = np.asarray(robot.get_frame_pose(fx.frame_map["left_tool"]).translation, dtype=float)
        right_err = float(np.linalg.norm(right_positions[step_idx] - right_now))
        left_err = float(np.linalg.norm(left_positions[step_idx] - left_now))

        collision_min = (
            _evaluate_collision_min_distance(solver, q) if enable_collision else float("inf")
        )
        if enable_com:
            com_outer, com_inner = _evaluate_com_slacks(robot, fx.support_polygon, inner_polygon)
        else:
            com_outer, com_inner = float("inf"), float("inf")
        relaxed_mm = _relaxed_margin_mm(solver) if enable_collision else float("nan")
        if np.isfinite(relaxed_mm):
            cumulative_relaxed_mm += relaxed_mm
            relaxed_mm_count += 1

        metrics.total_steps += 1
        if status_name == "SUCCESS":
            metrics.success_count += 1
        elif status_name == "INFEASIBLE":
            metrics.infeasible_count += 1
        elif status_name == "NO_PROGRESS":
            metrics.no_progress_count += 1
        elif status_name == "COLLISION_VIOLATED":
            metrics.collision_violated_count += 1

        zero_motion = dq_norm <= ZERO_MOTION_EPS
        if zero_motion:
            metrics.zero_motion_count += 1

        above_deadband = max(right_err, left_err) > EE_POSITION_DEADBAND
        stalled = (dq_norm < DQ_STALL_EPS) and above_deadband
        if stalled:
            metrics.stall_count += 1
            if step_idx >= duration_steps // 2:
                metrics.late_stall_count += 1
        stall_window.append(stalled)
        if len(stall_window) == UNRECOVERABLE_WINDOW and sum(stall_window) >= UNRECOVERABLE_STALL_FRAMES:
            metrics.unrecoverable_stall_count += 1

        # Breach thresholds: only count meaningful (>1mm) violations, since the
        # solver may leave micrometer-scale slack at the boundary even when the
        # constraint is correctly active.
        breach_slack_m = 1e-3
        if enable_collision and np.isfinite(collision_min) and collision_min < (collision_min_distance_m - breach_slack_m):
            metrics.collision_breach_count += 1
        if enable_com and np.isfinite(com_inner) and com_inner < -breach_slack_m:
            metrics.com_breach_count += 1

        metrics.escape_events += int(getattr(result, "stall_escape_count", 0) or 0)
        if np.isfinite(collision_min):
            metrics.min_collision_distance_m = min(metrics.min_collision_distance_m, collision_min)
        if np.isfinite(com_inner):
            metrics.min_com_inner_slack_m = min(metrics.min_com_inner_slack_m, com_inner)
        metrics.peak_right_position_error_m = max(metrics.peak_right_position_error_m, right_err)
        metrics.peak_left_position_error_m = max(metrics.peak_left_position_error_m, left_err)
        metrics.final_right_position_error_m = right_err
        metrics.final_left_position_error_m = left_err

        metrics.step_records.append(
            StepRecord(
                step=step_idx,
                status_name=status_name,
                dq_norm=dq_norm,
                right_pos_err=right_err,
                left_pos_err=left_err,
                min_collision_distance=collision_min if np.isfinite(collision_min) else -1.0,
                com_slack_outer=com_outer if np.isfinite(com_outer) else -1.0,
                com_slack_inner=com_inner if np.isfinite(com_inner) else -1.0,
                relaxed_margin=relaxed_mm if np.isfinite(relaxed_mm) else -1.0,
                elapsed_ms=elapsed_ms,
            )
        )

        # Clear solver target velocities when persistent zero motion, mirroring
        # what the GUI loop does; keeps harness comparable to the live demo.
        if stalled and (step_idx % 30 == 0):
            clear_all_target_velocities_if_available(solver)

        if verbose and (step_idx % 50 == 0 or stalled):
            print(
                f"[{scenario}] step={step_idx:4d} status={status_name} dq={dq_norm:.2e} "
                f"right_err={right_err:.4f} left_err={left_err:.4f} "
                f"collision_min={collision_min if np.isfinite(collision_min) else 'inf'} "
                f"com_inner={com_inner if np.isfinite(com_inner) else 'inf'} "
                f"relaxed_mm={relaxed_mm}"
            )

    if metrics.total_steps:
        metrics.avg_solve_ms = cumulative_solve_ms / metrics.total_steps
    if relaxed_mm_count:
        metrics.avg_relaxed_collision_margin_mm = cumulative_relaxed_mm / relaxed_mm_count
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("sg2", "bg2"), default="sg2")
    parser.add_argument(
        "--scenario",
        choices=("all", *SCENARIO_NAMES),
        default="all",
    )
    parser.add_argument("--duration-steps", type=int, default=500)
    parser.add_argument("--collision-min-distance", type=float, default=0.035)
    parser.add_argument("--max-collision-constraints", type=int, default=2)
    parser.add_argument("--com-margin-pct", type=float, default=10.0, help="Percent of support polygon (0-100)")
    parser.add_argument("--collision-tuning-mode", default="balanced", choices=("speed", "balanced", "precise"))
    parser.add_argument("--solve-mode", default="SCALE_ELASTIC", choices=("SCALE", "SCALE_ELASTIC", "MIN_ERROR"))
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--pos-gain", type=float, default=10.0)
    parser.add_argument("--rot-gain", type=float, default=10.0)
    parser.add_argument("--no-collision", action="store_true")
    parser.add_argument("--no-com", action="store_true")
    parser.add_argument(
        "--fallback-status-names",
        nargs="*",
        default=("INVALID_INPUT", "INFEASIBLE", "NUMERICAL_ERROR", "NO_PROGRESS"),
        help="Status names that trigger velocity-solve fallback.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    scenarios = SCENARIO_NAMES if args.scenario == "all" else (args.scenario,)
    results: dict[str, dict] = {}
    total_hard = 0

    for scenario in scenarios:
        metrics = run_scenario(
            scenario,
            variant=args.variant,
            duration_steps=args.duration_steps,
            collision_min_distance_m=float(args.collision_min_distance),
            max_collision_constraints=int(args.max_collision_constraints),
            com_margin_frac=float(args.com_margin_pct) / 100.0,
            collision_tuning_mode=args.collision_tuning_mode,
            solve_mode_label=args.solve_mode,
            allow_fallback=bool(args.allow_fallback),
            pos_gain=float(args.pos_gain),
            rot_gain=float(args.rot_gain),
            enable_collision=not args.no_collision,
            enable_com=not args.no_com,
            fallback_status_names=tuple(args.fallback_status_names),
            verbose=bool(args.verbose),
        )
        record = metrics.as_dict()
        results[scenario] = record
        total_hard += record["hard_failures"]
        print(
            f"[{scenario}] steps={record['total_steps']} success={record['success_count']} "
            f"stall={record['stall_count']} zero_motion={record['zero_motion_count']} "
            f"infeasible={record['infeasible_count']} no_progress={record['no_progress_count']} "
            f"collision_violated={record['collision_violated_count']} "
            f"collision_breach={record['collision_breach_count']} "
            f"com_breach={record['com_breach_count']} "
            f"unrecoverable={record['unrecoverable_stall_count']} "
            f"late_stall={record['late_stall_count']} "
            f"hard={record['hard_failures']} "
            f"escape_events={record['escape_events']} "
            f"avg_solve_ms={record['avg_solve_ms']} "
            f"final_right={record['final_right_position_error_m']} "
            f"final_left={record['final_left_position_error_m']} "
            f"min_col={record['min_collision_distance_m']} "
            f"min_com={record['min_com_inner_slack_m']}"
        )

    aggregate = {
        "variant": args.variant,
        "duration_steps": args.duration_steps,
        "scenarios": results,
        "total_hard_failures": total_hard,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(aggregate, indent=2))
        print(f"[harness] wrote metrics to {args.output_json}")
    return 1 if total_hard > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
