#!/usr/bin/env python3
"""Matrix benchmark for nullspace health sampling experiments.

This harness compares current runtime behavior, prior workaround modes, and
health-sampling variants using public EmbodiK scenarios and metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

import embodik as eik

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.harnesses.kinematic_health_metrics import (  # noqa: E402
    collision_distance_stats_dict,
    kinematic_health_sample,
    summarize_continuity_series,
    summarize_health_series,
)
from examples.utils.robot_models import ensure_ros_package_path  # noqa: E402


@dataclass(frozen=True)
class Scenario:
    name: str
    builder: Callable[[], tuple[eik.RobotModel, eik.KinematicsSolver, np.ndarray, np.ndarray, str]]


_HEALTH_SAMPLING_MODES = {
    "health_sampling",
    "health_sampling_cache_only",
    "health_sampling_collision",
    "health_sampling_parallel_seed",
    "ramped_parallel_seed_cache",
}
_PARALLEL_SEED_MODES = {
    "parallel_seed_recovery",
    "health_sampling_parallel_seed",
    "ramped_parallel_seed",
    "ramped_parallel_seed_cache",
    "ramped_parallel_seed_strict",
}
_RAMPED_PARALLEL_SEED_MODES = {
    "ramped_parallel_seed",
    "ramped_parallel_seed_cache",
    "ramped_parallel_seed_strict",
}
_PROJECTED_GRADIENT_MODES = {
    "projected_health_gradient",
}
_HYBRID_MIN_ERROR_MODES = {
    "projected_min_error_hybrid",
    "projected_min_error_hybrid_smooth",
    "projected_min_error_hybrid_smooth_stall_only",
    "projected_min_error_hybrid_smooth_strict",
    "projected_min_error_hybrid_smooth_strict_throttled",
}


@dataclass(frozen=True)
class ParallelSeedExperimentConfig:
    sample_radius: float = 0.035
    blend: float = 0.2
    min_score_improvement: float = 1e-4
    singularity_normalization_scale: float = 1e-6
    joint_limit_weight: float = 1.0
    singularity_weight: float = 0.25
    collision_weight: float = 0.5
    max_step_norm: float = 0.08
    max_step_component: float = 0.06
    max_primary_deviation_norm: float = 0.008
    max_primary_deviation_component: float = 0.004
    max_step_delta_norm: float = 0.08
    max_step_jerk_norm: float = 0.16
    task_error_abs_tolerance: float = 1e-4
    task_error_rel_tolerance: float = 0.05
    collision_worsen_tolerance: float = 1e-4
    activation_joint_limit_cost: float = 50.0
    zero_motion_step_norm: float = 1e-8
    ramp_gain: float = 0.35
    ramp_decay: float = 0.55
    ramp_min_score_improvement: float = 1e-6
    ramp_max_correction_norm: float = 0.004
    ramp_max_correction_component: float = 0.002
    ramp_max_correction_delta_norm: float = 0.0005
    ramp_max_correction_delta_component: float = 0.00025
    projected_gradient_correction_norm: float = 0.0025
    projected_gradient_correction_component: float = 0.00125
    projected_gradient_min_score_improvement: float = 1e-6
    hybrid_min_error_tracking_gain_abs: float = 1e-7
    hybrid_min_error_tracking_gain_rel: float = 0.0
    hybrid_health_debt_tolerance: float = 0.05
    hybrid_min_error_ramp_gain: float = 0.45
    hybrid_min_error_ramp_decay: float = 0.55
    hybrid_min_error_max_correction_norm: float = 0.008
    hybrid_min_error_max_correction_component: float = 0.004
    hybrid_min_error_max_correction_delta_norm: float = 0.0005
    hybrid_min_error_max_correction_delta_component: float = 0.00025
    hybrid_min_error_max_step_delta_norm: float = 0.0005
    hybrid_min_error_max_step_jerk_norm: float = 0.002
    hybrid_min_error_exit_decay: float = 0.82
    hybrid_min_error_hold_ticks: int = 6
    hybrid_min_error_refresh_interval: int = 1


@dataclass
class ParallelSeedExperimentStats:
    attempted_steps: int = 0
    candidate_solves: int = 0
    accepted_steps: int = 0
    rejected_limit: int = 0
    rejected_collision: int = 0
    rejected_tracking: int = 0
    rejected_continuity: int = 0
    rejected_health: int = 0
    best_score_delta: float = float("-inf")
    time_ms_total: float = 0.0
    ramped_target_updates: int = 0
    ramped_applied_steps: int = 0
    ramped_decayed_steps: int = 0
    projected_gradient_applied_steps: int = 0
    hybrid_min_error_attempted_steps: int = 0
    hybrid_min_error_applied_steps: int = 0
    hybrid_min_error_health_debt_steps: int = 0
    hybrid_min_error_time_ms_total: float = 0.0
    hybrid_min_error_smoothed_steps: int = 0


@dataclass
class RampedSeedRecoveryState:
    correction: np.ndarray | None = None
    target_q: np.ndarray | None = None
    active_ticks: int = 0
    hold_ticks_remaining: int = 0
    refresh_cooldown: int = 0


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def _make_runtime(
    mode: str, sample_budget: int, collision_scoring: bool
) -> eik.SolverRuntimeConfig:
    cfg = eik.SolverRuntimeConfig()
    cfg.weighted_fallback_enabled = mode in {
        "baseline",
        "weighted_fallback",
        "health_sampling",
        "health_sampling_cache_only",
        "health_sampling_collision",
        "parallel_seed_recovery",
        "health_sampling_parallel_seed",
        "ramped_parallel_seed",
        "ramped_parallel_seed_cache",
        "ramped_parallel_seed_strict",
        "projected_health_gradient",
        "projected_min_error_hybrid",
        "projected_min_error_hybrid_smooth",
        "projected_min_error_hybrid_smooth_stall_only",
        "projected_min_error_hybrid_smooth_strict",
        "projected_min_error_hybrid_smooth_strict_throttled",
    }
    cfg.weighted_advisor_enabled = mode == "weighted_fallback"
    if mode in _HEALTH_SAMPLING_MODES:
        cfg.health_sampling.enabled = True
        cfg.health_sampling.seed = 20260606
        cfg.health_sampling.sample_count = (
            0
            if mode in {"health_sampling_cache_only", "ramped_parallel_seed_cache"}
            else int(sample_budget)
        )
        cfg.health_sampling.sample_radius = 0.02
        cfg.health_sampling.gain = 0.2
        cfg.health_sampling.collision_scoring = bool(collision_scoring)
    return cfg


def _apply_mode(
    solver: eik.KinematicsSolver,
    task: eik.Task,
    mode: str,
    sample_budget: int,
    collision_scoring: bool,
) -> None:
    solver.configure_runtime(_make_runtime(mode, sample_budget, collision_scoring))
    task.solve_mode = eik.TaskSolveMode.SCALE
    task.allow_min_error_fallback = False
    if mode == "min_error":
        task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    elif mode == "elastic_scale":
        task.solve_mode = eik.TaskSolveMode.SCALE_ELASTIC


def _panda_limit_scenario() -> (
    tuple[eik.RobotModel, eik.KinematicsSolver, np.ndarray, np.ndarray, str]
):
    from robot_descriptions.panda_description import URDF_PATH

    ensure_ros_package_path(Path(URDF_PATH))
    robot = eik.RobotModel(str(URDF_PATH), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)
    solver.enable_timing_breakdown(True)
    q = np.array([0.0, -0.785, 0.0, -2.85, 0.0, 0.08, 0.785, 0.04, 0.04], dtype=float)
    robot.update_configuration(q)
    task = solver.add_frame_task("ee", "panda_hand")
    task.priority = 0
    task.weight = 1.0
    pose = robot.get_frame_pose("panda_hand")
    target = np.eye(4, dtype=float)
    target[:3, :3] = np.asarray(pose.rotation, dtype=float)
    target[:3, 3] = np.asarray(pose.translation, dtype=float) + np.array([-0.12, 0.0, 0.08])
    return robot, solver, q, target, "ee"


def _panda_collision_scenario() -> (
    tuple[eik.RobotModel, eik.KinematicsSolver, np.ndarray, np.ndarray, str]
):
    robot, solver, q, target, task_name = _panda_limit_scenario()
    solver.configure_collision_constraint(
        min_distance=0.03,
        include_pairs=[],
        exclude_pairs=[],
        nearest_points_all_pairs=False,
        max_constraints=3,
    )
    return robot, solver, q, target, task_name


def _dual_iiwa_scenario() -> (
    tuple[eik.RobotModel, eik.KinematicsSolver, np.ndarray, np.ndarray, str]
):
    import os
    import tempfile

    from examples.utils.dual_iiwa_urdf import build_dual_iiwa_urdf, get_dual_iiwa_frame_names

    with tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False) as f:
        f.write(build_dual_iiwa_urdf())
        urdf_path = f.name
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
    finally:
        os.unlink(urdf_path)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)
    solver.enable_timing_breakdown(True)
    frame, _right = get_dual_iiwa_frame_names()
    q = np.asarray(robot.get_current_configuration(), dtype=float)
    robot.update_configuration(q)
    task = solver.add_frame_task("left_ee", frame)
    task.priority = 0
    task.weight = 1.0
    pose = robot.get_frame_pose(frame)
    target = np.eye(4, dtype=float)
    target[:3, :3] = np.asarray(pose.rotation, dtype=float)
    target[:3, 3] = np.asarray(pose.translation, dtype=float) + np.array([0.05, -0.10, 0.03])
    return robot, solver, q, target, "left_ee"


def _health_score(
    sample,
    cfg: ParallelSeedExperimentConfig,
    collision_distance: float | None = None,
) -> float:
    singularity = 0.0
    if (
        math.isfinite(float(sample.limit_weighted_dexterity))
        and sample.limit_weighted_dexterity > 0.0
    ):
        scale = max(float(cfg.singularity_normalization_scale), 1e-12)
        singularity = float(sample.limit_weighted_dexterity) / (
            float(sample.limit_weighted_dexterity) + scale
        )
    score = (
        cfg.joint_limit_weight * float(sample.joint_limit_health)
        + cfg.singularity_weight * singularity
    )
    if collision_distance is not None and math.isfinite(float(collision_distance)):
        score += cfg.collision_weight * float(collision_distance)
    return float(score)


def _task_error_at(robot: eik.RobotModel, task: eik.Task, q: np.ndarray) -> float:
    robot.update_configuration(np.asarray(q, dtype=float))
    task.update(robot)
    err = np.asarray(task.get_error(), dtype=float)
    if err.size >= 6:
        return float(np.linalg.norm(err[:3]) + np.linalg.norm(err[3:]))
    return float(np.linalg.norm(err))


def _collision_distance_at(
    solver: eik.KinematicsSolver,
    robot: eik.RobotModel,
    q: np.ndarray,
    q_restore: np.ndarray,
) -> float:
    try:
        d = solver.evaluate_min_collision_distance(np.asarray(q, dtype=float))
        return float(d) if math.isfinite(float(d)) else float("inf")
    finally:
        robot.update_configuration(np.asarray(q_restore, dtype=float))


def _finite_limited_indices(q_lower: np.ndarray, q_upper: np.ndarray, q_size: int) -> list[int]:
    indices: list[int] = []
    n = min(int(q_size), q_lower.size, q_upper.size)
    for i in range(n):
        if (
            math.isfinite(float(q_lower[i]))
            and math.isfinite(float(q_upper[i]))
            and float(q_upper[i] - q_lower[i]) > 1e-9
        ):
            indices.append(i)
    return indices


def _project_onto_task_nullspace(delta: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
    if jacobian.size == 0 or jacobian.shape[1] != delta.size:
        return delta
    try:
        projector = np.eye(delta.size) - np.linalg.pinv(jacobian, rcond=1e-8) @ jacobian
    except np.linalg.LinAlgError:
        return delta
    return np.asarray(projector @ delta, dtype=float)


def _limit_projected_seed(
    q: np.ndarray,
    delta: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
) -> np.ndarray:
    q_seed = np.asarray(q, dtype=float) + np.asarray(delta, dtype=float)
    n = min(q_seed.size, q_lower.size, q_upper.size)
    if n > 0:
        q_seed[:n] = np.clip(q_seed[:n], q_lower[:n], q_upper[:n])
    return q_seed


def _generate_parallel_seed_candidates(
    *,
    robot: eik.RobotModel,
    task: eik.Task,
    q: np.ndarray,
    jacobian: np.ndarray,
    sample_budget: int,
    step_index: int,
    cfg: ParallelSeedExperimentConfig,
) -> list[np.ndarray]:
    q_arr = np.asarray(q, dtype=float)
    q_lower, q_upper = robot.get_joint_limits()
    lower = np.asarray(q_lower, dtype=float)
    upper = np.asarray(q_upper, dtype=float)
    active = _finite_limited_indices(lower, upper, q_arr.size)
    if not active or sample_budget <= 0:
        return []

    base_sample = kinematic_health_sample(robot, q_arr, jacobian)
    base_score = _health_score(base_sample, cfg)
    rng = np.random.default_rng(20260606 + 7919 * int(step_index) + 104729 * q_arr.size)
    candidates: list[tuple[float, np.ndarray]] = []

    def add_delta(delta: np.ndarray) -> None:
        delta = np.asarray(delta, dtype=float)
        if delta.size != q_arr.size or not np.all(np.isfinite(delta)):
            return
        delta = _project_onto_task_nullspace(delta, jacobian)
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-12:
            return
        delta = (cfg.sample_radius / norm) * delta
        q_seed = _limit_projected_seed(q_arr, delta, lower, upper)
        if np.linalg.norm(q_seed - q_arr) <= 1e-12:
            return
        robot.update_configuration(q_seed)
        task.update(robot)
        seed_jac = np.asarray(task.get_jacobian(), dtype=float)
        seed_sample = kinematic_health_sample(robot, q_seed, seed_jac)
        seed_score = _health_score(seed_sample, cfg)
        robot.update_configuration(q_arr)
        task.update(robot)
        if seed_score > base_score + cfg.min_score_improvement:
            candidates.append((seed_score - base_score, q_seed))

    try:
        gradient = np.asarray(
            eik.joint_limit_distance_gradient(q_arr, lower, upper, 1e-6),
            dtype=float,
        )
        grad_delta = np.zeros_like(q_arr)
        for idx in active:
            grad_delta[idx] = gradient[idx]
        add_delta(grad_delta)
        add_delta(-grad_delta)
    except Exception:
        pass

    for _ in range(max(0, int(sample_budget) * 2)):
        delta = np.zeros_like(q_arr)
        values = rng.uniform(-1.0, 1.0, size=len(active))
        for idx, value in zip(active, values, strict=False):
            delta[idx] = value
        add_delta(delta)
        if len(candidates) >= int(sample_budget):
            break

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [q_seed for _score, q_seed in candidates[: int(sample_budget)]]


def _build_parallel_seed_workers(
    scenario: Scenario,
    mode: str,
    *,
    sample_budget: int,
    collision_scoring: bool,
) -> list[tuple[eik.RobotModel, eik.KinematicsSolver, eik.Task]]:
    workers = []
    for _ in range(max(0, int(sample_budget))):
        robot, solver, _q, _target, task_name = scenario.builder()
        task = solver.get_task(task_name)
        if task is None:
            raise RuntimeError(f"task {task_name!r} was not created")
        _apply_mode(solver, task, mode, sample_budget, collision_scoring)
        workers.append((robot, solver, task))
    return workers


def _parallel_seed_recovery_step(
    *,
    scenario: Scenario,
    mode: str,
    workers: list[tuple[eik.RobotModel, eik.KinematicsSolver, eik.Task]],
    main_solver: eik.KinematicsSolver,
    main_robot: eik.RobotModel,
    main_task: eik.Task,
    q_current: np.ndarray,
    q_primary: np.ndarray,
    target: np.ndarray,
    opts: eik.PositionStepOptions,
    primary_error: float,
    base_health,
    base_jacobian: np.ndarray,
    sample_budget: int,
    step_index: int,
    prev_step: np.ndarray,
    prev_step_delta: np.ndarray,
    collision_scoring: bool,
    seed_worker_mode: str,
    executor: ThreadPoolExecutor | None,
    stats: ParallelSeedExperimentStats,
    cfg: ParallelSeedExperimentConfig,
) -> tuple[np.ndarray, bool]:
    del scenario
    q_current = np.asarray(q_current, dtype=float)
    q_primary = np.asarray(q_primary, dtype=float)
    primary_step = q_primary - q_current
    primary_step_norm = float(np.linalg.norm(primary_step))
    status_trigger = primary_step_norm <= cfg.zero_motion_step_norm and primary_error > 1e-4
    health_trigger = float(base_health.joint_limit_cost) >= cfg.activation_joint_limit_cost
    if not (status_trigger or health_trigger):
        return q_primary, False

    t0 = time.perf_counter()
    stats.attempted_steps += 1
    seeds = _generate_parallel_seed_candidates(
        robot=main_robot,
        task=main_task,
        q=q_current,
        jacobian=base_jacobian,
        sample_budget=sample_budget,
        step_index=step_index,
        cfg=cfg,
    )
    if not seeds:
        stats.time_ms_total += (time.perf_counter() - t0) * 1000.0
        return q_primary, False

    q_lower, q_upper = main_robot.get_joint_limits()
    lower = np.asarray(q_lower, dtype=float)
    upper = np.asarray(q_upper, dtype=float)
    vel_limits = np.asarray(main_robot.get_velocity_limits(), dtype=float)
    dt_safe = max(float(opts.dt if opts.dt > 0.0 else main_solver.dt), 1e-12)
    current_collision = (
        _collision_distance_at(main_solver, main_robot, q_current, q_current)
        if collision_scoring
        else float("inf")
    )
    best_q: np.ndarray | None = None
    best_objective = -float("inf")

    task_name = str(main_task.name)

    def solve_candidate(seed_index: int, q_seed: np.ndarray) -> tuple[int, np.ndarray | None]:
        if seed_index >= len(workers):
            return seed_index, None
        worker_robot, worker_solver, _worker_task = workers[seed_index]
        worker_robot.update_configuration(q_seed)
        try:
            result = worker_solver.solve_position_step(q_seed, target, task_name, opts)
            return seed_index, np.asarray(result.q_solution, dtype=float)
        except Exception:
            return seed_index, None

    indexed_seeds = list(enumerate(seeds[: len(workers)]))
    stats.candidate_solves += len(indexed_seeds)
    if seed_worker_mode == "thread" and executor is not None and len(indexed_seeds) > 1:
        futures = [
            executor.submit(solve_candidate, i, q_seed.copy()) for i, q_seed in indexed_seeds
        ]
        candidate_outputs = [future.result() for future in futures]
    else:
        candidate_outputs = [solve_candidate(i, q_seed) for i, q_seed in indexed_seeds]

    for _seed_index, q_candidate in candidate_outputs:
        if q_candidate is None:
            stats.rejected_limit += 1
            continue
        if q_candidate.size != q_current.size or not np.all(np.isfinite(q_candidate)):
            stats.rejected_limit += 1
            continue

        step = q_candidate - q_current
        m = min(step.size, vel_limits.size)
        if m > 0:
            step_limit = np.maximum(0.0, vel_limits[:m]) * dt_safe
            finite_limits = np.isfinite(step_limit)
            limited_step = step[:m].copy()
            limited_step[finite_limits] = np.clip(
                limited_step[finite_limits],
                -step_limit[finite_limits],
                step_limit[finite_limits],
            )
            step[:m] = limited_step
        max_component = float(np.max(np.abs(step))) if step.size else 0.0
        if max_component > cfg.max_step_component:
            step *= cfg.max_step_component / max_component
        step_norm = float(np.linalg.norm(step))
        if step_norm > cfg.max_step_norm:
            step *= cfg.max_step_norm / step_norm
        blend = min(1.0, max(0.0, float(cfg.blend)))
        step = primary_step + blend * (step - primary_step)
        q_candidate = q_current + step

        n = min(q_candidate.size, lower.size, upper.size)
        if n > 0:
            q_candidate[:n] = np.clip(q_candidate[:n], lower[:n], upper[:n])
            step = q_candidate - q_current
        if n > 0 and (
            np.any(q_candidate[:n] < lower[:n] - 1e-9) or np.any(q_candidate[:n] > upper[:n] + 1e-9)
        ):
            stats.rejected_limit += 1
            continue

        step_norm = float(np.linalg.norm(step))
        primary_deviation = step - primary_step
        primary_delta_norm = float(np.linalg.norm(primary_step - prev_step))
        primary_jerk_norm = float(np.linalg.norm((primary_step - prev_step) - prev_step_delta))
        if (
            step_norm > cfg.max_step_norm
            or float(np.max(np.abs(step))) > cfg.max_step_component
            or float(np.linalg.norm(primary_deviation)) > cfg.max_primary_deviation_norm
            or float(np.max(np.abs(primary_deviation))) > cfg.max_primary_deviation_component
            or float(np.linalg.norm(step - prev_step))
            > max(cfg.max_step_delta_norm, primary_delta_norm + cfg.max_primary_deviation_norm)
            or float(np.linalg.norm((step - prev_step) - prev_step_delta))
            > max(cfg.max_step_jerk_norm, primary_jerk_norm + cfg.max_primary_deviation_norm)
        ):
            stats.rejected_continuity += 1
            continue

        m = min(step.size, vel_limits.size)
        if m > 0 and np.any(np.abs(step[:m] / dt_safe) > vel_limits[:m] + 1e-9):
            stats.rejected_continuity += 1
            continue

        candidate_error = _task_error_at(main_robot, main_task, q_candidate)
        main_robot.update_configuration(q_current)
        main_task.update(main_robot)
        allowed_error = primary_error + max(
            cfg.task_error_abs_tolerance,
            cfg.task_error_rel_tolerance * max(primary_error, 1e-12),
        )
        if candidate_error > allowed_error:
            stats.rejected_tracking += 1
            continue

        candidate_collision = float("inf")
        collision_delta = 0.0
        if collision_scoring:
            candidate_collision = _collision_distance_at(
                main_solver,
                main_robot,
                q_candidate,
                q_current,
            )
            collision_delta = candidate_collision - current_collision
            if collision_delta < -cfg.collision_worsen_tolerance:
                stats.rejected_collision += 1
                continue

        main_robot.update_configuration(q_candidate)
        main_task.update(main_robot)
        candidate_jac = np.asarray(main_task.get_jacobian(), dtype=float)
        candidate_health = kinematic_health_sample(main_robot, q_candidate, candidate_jac)
        candidate_score = _health_score(
            candidate_health,
            cfg,
            candidate_collision if collision_scoring else None,
        )
        base_score_with_collision = _health_score(
            base_health,
            cfg,
            current_collision if collision_scoring else None,
        )
        score_delta = candidate_score - base_score_with_collision
        main_robot.update_configuration(q_current)
        main_task.update(main_robot)
        if score_delta <= cfg.min_score_improvement:
            stats.rejected_health += 1
            continue

        objective = (
            score_delta
            + 0.25 * max(0.0, primary_error - candidate_error)
            + 0.1 * collision_delta
            - 0.01 * step_norm
        )
        if objective > best_objective:
            best_objective = objective
            best_q = q_candidate
            stats.best_score_delta = max(stats.best_score_delta, score_delta)

    stats.time_ms_total += (time.perf_counter() - t0) * 1000.0
    if best_q is None:
        return q_primary, False
    stats.accepted_steps += 1
    return best_q, True


def _limit_vector(
    values: np.ndarray,
    *,
    max_norm: float,
    max_component: float,
) -> np.ndarray:
    limited = np.asarray(values, dtype=float).copy()
    if limited.size == 0:
        return limited
    component = float(np.max(np.abs(limited)))
    if max_component > 0.0 and component > max_component:
        limited *= max_component / component
    norm = float(np.linalg.norm(limited))
    if max_norm > 0.0 and norm > max_norm:
        limited *= max_norm / norm
    return limited


def _ramped_parallel_seed_step(
    *,
    main_solver: eik.KinematicsSolver,
    main_robot: eik.RobotModel,
    main_task: eik.Task,
    q_current: np.ndarray,
    q_primary: np.ndarray,
    q_target: np.ndarray | None,
    primary_error: float,
    base_health,
    base_jacobian: np.ndarray,
    prev_step: np.ndarray,
    collision_scoring: bool,
    state: RampedSeedRecoveryState,
    stats: ParallelSeedExperimentStats,
    cfg: ParallelSeedExperimentConfig,
    opts: eik.PositionStepOptions,
) -> tuple[np.ndarray, bool]:
    q_current = np.asarray(q_current, dtype=float)
    q_primary = np.asarray(q_primary, dtype=float)
    primary_step = q_primary - q_current
    if state.correction is None or state.correction.size != q_current.size:
        state.correction = np.zeros_like(q_current)

    if q_target is not None:
        state.target_q = np.asarray(q_target, dtype=float).copy()
        stats.ramped_target_updates += 1

    desired = np.zeros_like(q_current)
    if state.target_q is not None and state.target_q.size == q_current.size:
        desired = np.asarray(state.target_q, dtype=float) - q_primary

    if q_target is None and float(np.linalg.norm(desired)) <= 1e-12:
        next_correction = state.correction * max(0.0, min(1.0, float(cfg.ramp_decay)))
        stats.ramped_decayed_steps += int(float(np.linalg.norm(next_correction)) > 1e-12)
    else:
        gain = max(0.0, min(1.0, float(cfg.ramp_gain)))
        next_correction = state.correction + gain * (desired - state.correction)

    next_correction = _limit_vector(
        next_correction,
        max_norm=cfg.ramp_max_correction_norm,
        max_component=cfg.ramp_max_correction_component,
    )
    correction_delta = _limit_vector(
        next_correction - state.correction,
        max_norm=cfg.ramp_max_correction_delta_norm,
        max_component=cfg.ramp_max_correction_delta_component,
    )
    next_correction = state.correction + correction_delta
    if float(np.linalg.norm(next_correction)) <= 1e-12:
        state.correction = next_correction
        return q_primary, False

    q_lower, q_upper = main_robot.get_joint_limits()
    lower = np.asarray(q_lower, dtype=float)
    upper = np.asarray(q_upper, dtype=float)
    vel_limits = np.asarray(main_robot.get_velocity_limits(), dtype=float)
    dt_safe = max(float(opts.dt if opts.dt > 0.0 else main_solver.dt), 1e-12)
    q_candidate = q_primary + next_correction
    n = min(q_candidate.size, lower.size, upper.size)
    if n > 0:
        q_candidate[:n] = np.clip(q_candidate[:n], lower[:n], upper[:n])
    step = q_candidate - q_current

    m = min(step.size, vel_limits.size)
    if m > 0 and np.any(np.abs(step[:m] / dt_safe) > vel_limits[:m] + 1e-9):
        stats.rejected_continuity += 1
        state.correction *= max(0.0, min(1.0, float(cfg.ramp_decay)))
        return q_primary, False

    if float(np.linalg.norm(step - prev_step)) > float(np.linalg.norm(primary_step - prev_step)) + (
        cfg.ramp_max_correction_delta_norm + 1e-12
    ):
        stats.rejected_continuity += 1
        state.correction *= max(0.0, min(1.0, float(cfg.ramp_decay)))
        return q_primary, False

    candidate_error = _task_error_at(main_robot, main_task, q_candidate)
    main_robot.update_configuration(q_current)
    main_task.update(main_robot)
    allowed_error = primary_error + max(
        cfg.task_error_abs_tolerance,
        cfg.task_error_rel_tolerance * max(primary_error, 1e-12),
    )
    if candidate_error > allowed_error:
        stats.rejected_tracking += 1
        state.correction *= max(0.0, min(1.0, float(cfg.ramp_decay)))
        return q_primary, False

    current_collision = float("inf")
    candidate_collision = float("inf")
    if collision_scoring:
        current_collision = _collision_distance_at(main_solver, main_robot, q_current, q_current)
        candidate_collision = _collision_distance_at(
            main_solver, main_robot, q_candidate, q_current
        )
        if candidate_collision - current_collision < -cfg.collision_worsen_tolerance:
            stats.rejected_collision += 1
            state.correction *= max(0.0, min(1.0, float(cfg.ramp_decay)))
            return q_primary, False

    main_robot.update_configuration(q_candidate)
    main_task.update(main_robot)
    candidate_jac = np.asarray(main_task.get_jacobian(), dtype=float)
    candidate_health = kinematic_health_sample(main_robot, q_candidate, candidate_jac)
    candidate_score = _health_score(
        candidate_health,
        cfg,
        candidate_collision if collision_scoring else None,
    )
    base_score = _health_score(
        base_health,
        cfg,
        current_collision if collision_scoring else None,
    )
    main_robot.update_configuration(q_current)
    main_task.update(main_robot)
    score_delta = candidate_score - base_score
    if score_delta <= cfg.ramp_min_score_improvement:
        stats.rejected_health += 1
        state.correction *= max(0.0, min(1.0, float(cfg.ramp_decay)))
        return q_primary, False

    state.correction = next_correction
    state.active_ticks += 1
    stats.ramped_applied_steps += 1
    stats.best_score_delta = max(stats.best_score_delta, score_delta)
    return q_candidate, True


def _projected_health_gradient_step(
    *,
    main_solver: eik.KinematicsSolver,
    main_robot: eik.RobotModel,
    main_task: eik.Task,
    q_current: np.ndarray,
    q_primary: np.ndarray,
    primary_error: float,
    base_health,
    base_jacobian: np.ndarray,
    prev_step: np.ndarray,
    collision_scoring: bool,
    stats: ParallelSeedExperimentStats,
    cfg: ParallelSeedExperimentConfig,
    opts: eik.PositionStepOptions,
) -> tuple[np.ndarray, bool]:
    q_current = np.asarray(q_current, dtype=float)
    q_primary = np.asarray(q_primary, dtype=float)
    primary_step = q_primary - q_current
    primary_step_norm = float(np.linalg.norm(primary_step))
    status_trigger = primary_step_norm <= cfg.zero_motion_step_norm and primary_error > 1e-4
    health_trigger = float(base_health.joint_limit_cost) >= cfg.activation_joint_limit_cost
    if not (status_trigger or health_trigger):
        return q_primary, False

    stats.attempted_steps += 1
    q_lower, q_upper = main_robot.get_joint_limits()
    lower = np.asarray(q_lower, dtype=float)
    upper = np.asarray(q_upper, dtype=float)
    active = _finite_limited_indices(lower, upper, q_current.size)
    if not active:
        return q_primary, False

    try:
        gradient = np.asarray(
            eik.joint_limit_distance_gradient(q_current, lower, upper, 1e-6),
            dtype=float,
        )
    except Exception:
        stats.rejected_limit += 1
        return q_primary, False

    correction = np.zeros_like(q_current)
    for idx in active:
        correction[idx] = -gradient[idx]
    correction = _project_onto_task_nullspace(correction, base_jacobian)
    norm = float(np.linalg.norm(correction))
    if norm <= 1e-12 or not np.all(np.isfinite(correction)):
        stats.rejected_health += 1
        return q_primary, False

    correction = correction / norm
    correction = _limit_vector(
        correction,
        max_norm=cfg.projected_gradient_correction_norm,
        max_component=cfg.projected_gradient_correction_component,
    )
    if float(np.linalg.norm(correction)) <= 1e-12:
        stats.rejected_health += 1
        return q_primary, False

    q_candidate = q_primary + correction
    n = min(q_candidate.size, lower.size, upper.size)
    if n > 0:
        q_candidate[:n] = np.clip(q_candidate[:n], lower[:n], upper[:n])
    step = q_candidate - q_current
    if n > 0 and (
        np.any(q_candidate[:n] < lower[:n] - 1e-9) or np.any(q_candidate[:n] > upper[:n] + 1e-9)
    ):
        stats.rejected_limit += 1
        return q_primary, False

    vel_limits = np.asarray(main_robot.get_velocity_limits(), dtype=float)
    dt_safe = max(float(opts.dt if opts.dt > 0.0 else main_solver.dt), 1e-12)
    m = min(step.size, vel_limits.size)
    if m > 0 and np.any(np.abs(step[:m] / dt_safe) > vel_limits[:m] + 1e-9):
        stats.rejected_continuity += 1
        return q_primary, False

    if float(np.linalg.norm(step - prev_step)) > float(
        np.linalg.norm(primary_step - prev_step)
    ) + max(cfg.ramp_max_correction_delta_norm, cfg.projected_gradient_correction_norm):
        stats.rejected_continuity += 1
        return q_primary, False

    candidate_error = _task_error_at(main_robot, main_task, q_candidate)
    main_robot.update_configuration(q_current)
    main_task.update(main_robot)
    allowed_error = primary_error + max(
        cfg.task_error_abs_tolerance,
        cfg.task_error_rel_tolerance * max(primary_error, 1e-12),
    )
    if candidate_error > allowed_error:
        stats.rejected_tracking += 1
        return q_primary, False

    current_collision = float("inf")
    candidate_collision = float("inf")
    if collision_scoring:
        current_collision = _collision_distance_at(main_solver, main_robot, q_current, q_current)
        candidate_collision = _collision_distance_at(
            main_solver, main_robot, q_candidate, q_current
        )
        if candidate_collision - current_collision < -cfg.collision_worsen_tolerance:
            stats.rejected_collision += 1
            return q_primary, False

    main_robot.update_configuration(q_candidate)
    main_task.update(main_robot)
    candidate_jac = np.asarray(main_task.get_jacobian(), dtype=float)
    candidate_health = kinematic_health_sample(main_robot, q_candidate, candidate_jac)
    candidate_score = _health_score(
        candidate_health,
        cfg,
        candidate_collision if collision_scoring else None,
    )
    base_score = _health_score(
        base_health,
        cfg,
        current_collision if collision_scoring else None,
    )
    main_robot.update_configuration(q_current)
    main_task.update(main_robot)
    score_delta = candidate_score - base_score
    if score_delta <= cfg.projected_gradient_min_score_improvement:
        stats.rejected_health += 1
        return q_primary, False

    stats.accepted_steps += 1
    stats.projected_gradient_applied_steps += 1
    stats.best_score_delta = max(stats.best_score_delta, score_delta)
    return q_candidate, True


def _evaluate_hybrid_correction(
    *,
    main_solver: eik.KinematicsSolver,
    main_robot: eik.RobotModel,
    main_task: eik.Task,
    q_current: np.ndarray,
    q_primary: np.ndarray,
    correction: np.ndarray,
    primary_error: float,
    base_health,
    prev_step: np.ndarray,
    prev_step_delta: np.ndarray,
    collision_scoring: bool,
    stats: ParallelSeedExperimentStats,
    cfg: ParallelSeedExperimentConfig,
    opts: eik.PositionStepOptions,
    require_tracking_gain: bool,
) -> tuple[np.ndarray | None, float]:
    q_candidate = q_primary + correction
    q_lower, q_upper = main_robot.get_joint_limits()
    lower = np.asarray(q_lower, dtype=float)
    upper = np.asarray(q_upper, dtype=float)
    n = min(q_candidate.size, lower.size, upper.size)
    if n > 0:
        q_candidate[:n] = np.clip(q_candidate[:n], lower[:n], upper[:n])
        if np.any(q_candidate[:n] < lower[:n] - 1e-9) or np.any(q_candidate[:n] > upper[:n] + 1e-9):
            stats.rejected_limit += 1
            return None, 0.0

    step = q_candidate - q_current
    if (
        float(np.linalg.norm(step)) > cfg.max_step_norm
        or float(np.max(np.abs(step))) > cfg.max_step_component
    ):
        stats.rejected_continuity += 1
        return None, 0.0

    vel_limits = np.asarray(main_robot.get_velocity_limits(), dtype=float)
    dt_safe = max(float(opts.dt if opts.dt > 0.0 else main_solver.dt), 1e-12)
    m = min(step.size, vel_limits.size)
    if m > 0 and np.any(np.abs(step[:m] / dt_safe) > vel_limits[:m] + 1e-9):
        stats.rejected_continuity += 1
        return None, 0.0

    step_delta = step - prev_step
    step_jerk = step_delta - prev_step_delta
    if (
        float(np.linalg.norm(step_delta)) > cfg.hybrid_min_error_max_step_delta_norm
        or float(np.linalg.norm(step_jerk)) > cfg.hybrid_min_error_max_step_jerk_norm
    ):
        stats.rejected_continuity += 1
        return None, 0.0

    candidate_error = _task_error_at(main_robot, main_task, q_candidate)
    main_robot.update_configuration(q_current)
    main_task.update(main_robot)
    if require_tracking_gain:
        required_gain = max(
            cfg.hybrid_min_error_tracking_gain_abs,
            cfg.hybrid_min_error_tracking_gain_rel * max(primary_error, 1e-12),
        )
        if candidate_error > primary_error - required_gain:
            stats.rejected_tracking += 1
            return None, 0.0
    else:
        allowed_error = primary_error + max(
            cfg.task_error_abs_tolerance,
            cfg.task_error_rel_tolerance * max(primary_error, 1e-12),
        )
        if candidate_error > allowed_error:
            stats.rejected_tracking += 1
            return None, 0.0

    current_collision = float("inf")
    candidate_collision = float("inf")
    if collision_scoring:
        current_collision = _collision_distance_at(main_solver, main_robot, q_current, q_current)
        candidate_collision = _collision_distance_at(
            main_solver,
            main_robot,
            q_candidate,
            q_current,
        )
        if candidate_collision - current_collision < -cfg.collision_worsen_tolerance:
            stats.rejected_collision += 1
            return None, 0.0

    main_robot.update_configuration(q_candidate)
    main_task.update(main_robot)
    candidate_jac = np.asarray(main_task.get_jacobian(), dtype=float)
    candidate_health = kinematic_health_sample(main_robot, q_candidate, candidate_jac)
    candidate_score = _health_score(
        candidate_health,
        cfg,
        candidate_collision if collision_scoring else None,
    )
    base_score = _health_score(
        base_health,
        cfg,
        current_collision if collision_scoring else None,
    )
    main_robot.update_configuration(q_current)
    main_task.update(main_robot)
    score_delta = candidate_score - base_score
    if score_delta < -cfg.hybrid_health_debt_tolerance:
        stats.rejected_health += 1
        return None, score_delta

    return q_candidate, score_delta


def _bounded_min_error_hybrid_step(
    *,
    main_solver: eik.KinematicsSolver,
    main_robot: eik.RobotModel,
    main_task: eik.Task,
    q_current: np.ndarray,
    q_primary: np.ndarray,
    target: np.ndarray,
    task_name: str,
    primary_error: float,
    base_health,
    base_jacobian: np.ndarray,
    prev_step: np.ndarray,
    prev_step_delta: np.ndarray,
    collision_scoring: bool,
    state: RampedSeedRecoveryState,
    stats: ParallelSeedExperimentStats,
    cfg: ParallelSeedExperimentConfig,
    opts: eik.PositionStepOptions,
    smooth_on_reject: bool,
) -> tuple[np.ndarray, bool, float]:
    del base_jacobian
    q_current = np.asarray(q_current, dtype=float)
    q_primary = np.asarray(q_primary, dtype=float)
    primary_step = q_primary - q_current
    if state.correction is None or state.correction.size != q_current.size:
        state.correction = np.zeros_like(q_current)

    def smooth_exit(elapsed_ms: float) -> tuple[np.ndarray, bool, float]:
        if (
            not smooth_on_reject
            or state.correction is None
            or state.hold_ticks_remaining <= 0
            or float(np.linalg.norm(state.correction)) <= 1e-12
        ):
            state.correction *= max(0.0, min(1.0, float(cfg.hybrid_min_error_ramp_decay)))
            return q_primary, False, elapsed_ms

        state.hold_ticks_remaining -= 1
        decayed = state.correction * max(0.0, min(1.0, float(cfg.hybrid_min_error_exit_decay)))
        if float(np.linalg.norm(decayed)) <= 1e-12:
            state.correction = decayed
            return q_primary, False, elapsed_ms

        q_smooth, score_delta = _evaluate_hybrid_correction(
            main_solver=main_solver,
            main_robot=main_robot,
            main_task=main_task,
            q_current=q_current,
            q_primary=q_primary,
            correction=decayed,
            primary_error=primary_error,
            base_health=base_health,
            prev_step=prev_step,
            prev_step_delta=prev_step_delta,
            collision_scoring=collision_scoring,
            stats=stats,
            cfg=cfg,
            opts=opts,
            require_tracking_gain=False,
        )
        if q_smooth is None:
            state.correction *= max(0.0, min(1.0, float(cfg.hybrid_min_error_ramp_decay)))
            return q_primary, False, elapsed_ms

        state.correction = decayed
        state.active_ticks += 1
        stats.accepted_steps += 1
        stats.hybrid_min_error_applied_steps += 1
        stats.hybrid_min_error_smoothed_steps += 1
        stats.hybrid_min_error_health_debt_steps += int(score_delta < 0.0)
        stats.best_score_delta = max(stats.best_score_delta, score_delta)
        return q_smooth, True, elapsed_ms

    if float(np.linalg.norm(primary_step)) > cfg.zero_motion_step_norm:
        return smooth_exit(0.0)
    if primary_error <= 1e-4:
        return smooth_exit(0.0)
    if smooth_on_reject and state.refresh_cooldown > 0:
        state.refresh_cooldown -= 1
        return smooth_exit(0.0)

    old_mode = main_task.solve_mode
    old_fallback = main_task.allow_min_error_fallback
    stats.hybrid_min_error_attempted_steps += 1
    t0 = time.perf_counter()
    try:
        main_task.solve_mode = eik.TaskSolveMode.MIN_ERROR
        main_task.allow_min_error_fallback = True
        result = main_solver.solve_position_step(q_current, target, task_name, opts)
    finally:
        main_task.solve_mode = old_mode
        main_task.allow_min_error_fallback = old_fallback
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    stats.hybrid_min_error_time_ms_total += elapsed_ms

    q_candidate = np.asarray(result.q_solution, dtype=float)
    if q_candidate.size != q_current.size or not np.all(np.isfinite(q_candidate)):
        stats.rejected_limit += 1
        return smooth_exit(elapsed_ms)

    desired = q_candidate - q_primary
    gain = max(0.0, min(1.0, float(cfg.hybrid_min_error_ramp_gain)))
    next_correction = state.correction + gain * (desired - state.correction)
    next_correction = _limit_vector(
        next_correction,
        max_norm=cfg.hybrid_min_error_max_correction_norm,
        max_component=cfg.hybrid_min_error_max_correction_component,
    )
    correction_delta = _limit_vector(
        next_correction - state.correction,
        max_norm=cfg.hybrid_min_error_max_correction_delta_norm,
        max_component=cfg.hybrid_min_error_max_correction_delta_component,
    )
    next_correction = state.correction + correction_delta
    if float(np.linalg.norm(next_correction)) <= 1e-12:
        stats.rejected_continuity += 1
        return smooth_exit(elapsed_ms)

    q_candidate, score_delta = _evaluate_hybrid_correction(
        main_solver=main_solver,
        main_robot=main_robot,
        main_task=main_task,
        q_current=q_current,
        q_primary=q_primary,
        correction=next_correction,
        primary_error=primary_error,
        base_health=base_health,
        prev_step=prev_step,
        prev_step_delta=prev_step_delta,
        collision_scoring=collision_scoring,
        stats=stats,
        cfg=cfg,
        opts=opts,
        require_tracking_gain=True,
    )
    if q_candidate is None:
        return smooth_exit(elapsed_ms)

    state.correction = next_correction
    state.target_q = q_candidate.copy()
    state.active_ticks += 1
    state.hold_ticks_remaining = max(0, int(cfg.hybrid_min_error_hold_ticks))
    state.refresh_cooldown = max(0, int(cfg.hybrid_min_error_refresh_interval) - 1)
    stats.accepted_steps += 1
    stats.hybrid_min_error_applied_steps += 1
    stats.hybrid_min_error_health_debt_steps += int(score_delta < 0.0)
    stats.best_score_delta = max(stats.best_score_delta, score_delta)
    return q_candidate, True, elapsed_ms


def _run_case(
    scenario: Scenario,
    mode: str,
    *,
    steps: int,
    sample_budget: int,
    collision_scoring: bool,
    seed_worker_mode: str,
) -> dict[str, object]:
    robot, solver, q, target, task_name = scenario.builder()
    task = solver.get_task(task_name)
    if task is None:
        raise RuntimeError(f"task {task_name!r} was not created")
    _apply_mode(solver, task, mode, sample_budget, collision_scoring)

    health_samples = []
    collision_distances: list[float] = []
    timings: list[float] = []
    statuses: dict[str, int] = {}
    q_history: list[np.ndarray] = [np.asarray(q, dtype=float).copy()]
    health_applied = 0
    health_cache_available = 0
    health_cache_used = 0
    health_activation_allowed = 0
    accepted_samples = 0
    health_rejected_invalid = 0
    health_rejected_limit = 0
    health_rejected_collision = 0
    health_rejected_score = 0
    health_base_scores: list[float] = []
    health_best_scores: list[float] = []
    health_best_source_counts: dict[str, int] = {}
    exact_collision_queries = 0
    start_error = None
    final_error = None
    zero_progress = 0

    opts = eik.PositionStepOptions()
    opts.max_steps = 1
    opts.dt = solver.dt
    opts.position_gain = 10.0
    opts.orientation_gain = 10.0

    parallel_cfg = ParallelSeedExperimentConfig()
    if (
        mode in _RAMPED_PARALLEL_SEED_MODES
        or mode == "projected_min_error_hybrid_smooth_stall_only"
    ):
        parallel_cfg = replace(parallel_cfg, activation_joint_limit_cost=float("inf"))
    if mode == "ramped_parallel_seed_strict":
        parallel_cfg = replace(
            parallel_cfg,
            ramp_gain=0.15,
            ramp_max_correction_norm=0.0015,
            ramp_max_correction_component=0.00075,
            ramp_max_correction_delta_norm=0.0001,
            ramp_max_correction_delta_component=0.00005,
            max_primary_deviation_norm=0.004,
            max_primary_deviation_component=0.002,
        )
    if mode in {
        "projected_min_error_hybrid_smooth_strict",
        "projected_min_error_hybrid_smooth_strict_throttled",
    }:
        parallel_cfg = replace(
            parallel_cfg,
            activation_joint_limit_cost=float("inf"),
            hybrid_min_error_ramp_gain=0.2,
            hybrid_min_error_max_correction_norm=0.003,
            hybrid_min_error_max_correction_component=0.0015,
            hybrid_min_error_max_correction_delta_norm=0.0001,
            hybrid_min_error_max_correction_delta_component=0.00005,
            hybrid_min_error_max_step_delta_norm=0.0002,
            hybrid_min_error_max_step_jerk_norm=0.0005,
            hybrid_min_error_exit_decay=0.9,
            hybrid_min_error_hold_ticks=10,
            hybrid_min_error_refresh_interval=(
                3 if mode == "projected_min_error_hybrid_smooth_strict_throttled" else 1
            ),
        )
    parallel_stats = ParallelSeedExperimentStats()
    ramp_state = RampedSeedRecoveryState()
    hybrid_state = RampedSeedRecoveryState()
    parallel_workers = (
        _build_parallel_seed_workers(
            scenario,
            mode,
            sample_budget=sample_budget,
            collision_scoring=collision_scoring,
        )
        if mode in _PARALLEL_SEED_MODES and sample_budget > 0
        else []
    )
    prev_q = np.asarray(q, dtype=float)
    prev_step = np.zeros_like(prev_q)
    prev_step_delta = np.zeros_like(prev_q)
    executor = (
        ThreadPoolExecutor(max_workers=max(1, len(parallel_workers)))
        if mode in _PARALLEL_SEED_MODES
        and seed_worker_mode == "thread"
        and len(parallel_workers) > 1
        else None
    )
    try:
        for step_index in range(steps):
            robot.update_configuration(q)
            task.update(robot)
            jac = np.asarray(task.get_jacobian(), dtype=float)
            base_health = kinematic_health_sample(robot, q, jac)
            health_samples.append(base_health)
            d = solver.evaluate_min_collision_distance(q)
            if math.isfinite(float(d)):
                collision_distances.append(float(d))

            q_before_step = np.asarray(q, dtype=float).copy()
            t0 = time.perf_counter()
            result = solver.solve_position_step(q, target, task_name, opts)
            timings.append((time.perf_counter() - t0) * 1000.0)
            statuses[result.status.name] = statuses.get(result.status.name, 0) + 1
            diag = result.diagnostics
            health_applied += int(bool(diag.health_sampling_applied))
            health_cache_available += int(
                bool(getattr(diag, "health_sampling_cache_available", False))
            )
            health_cache_used += int(bool(getattr(diag, "health_sampling_cache_used", False)))
            health_activation_allowed += int(
                bool(getattr(diag, "health_sampling_activation_allowed", False))
            )
            accepted_samples += int(diag.health_sampling_accepted)
            health_rejected_invalid += int(getattr(diag, "health_sampling_rejected_invalid", 0))
            health_rejected_limit += int(getattr(diag, "health_sampling_rejected_limit", 0))
            health_rejected_collision += int(getattr(diag, "health_sampling_rejected_collision", 0))
            health_rejected_score += int(getattr(diag, "health_sampling_rejected_score", 0))
            base_score = float(getattr(diag, "health_sampling_base_score", float("nan")))
            if math.isfinite(base_score):
                health_base_scores.append(base_score)
            best_score = float(getattr(diag, "health_sampling_best_score", float("nan")))
            if math.isfinite(best_score):
                health_best_scores.append(best_score)
            best_source = getattr(diag, "health_sampling_best_source", None)
            best_source_name = getattr(best_source, "name", None)
            if best_source_name is None and best_source is not None:
                best_source_name = str(best_source).split(".")[-1]
            if best_source_name and best_source_name != "NONE":
                health_best_source_counts[best_source_name] = (
                    health_best_source_counts.get(best_source_name, 0) + 1
                )
            exact_collision_queries += int(getattr(result, "collision_exact_distance_queries", 0))
            final_error = float(result.position_error + result.orientation_error)
            if start_error is None:
                start_error = final_error
            q_primary = np.asarray(result.q_solution, dtype=float)
            if mode in _PARALLEL_SEED_MODES and parallel_workers:
                q_recovered, recovered_target = _parallel_seed_recovery_step(
                    scenario=scenario,
                    mode=mode,
                    workers=parallel_workers,
                    main_solver=solver,
                    main_robot=robot,
                    main_task=task,
                    q_current=q_before_step,
                    q_primary=q_primary,
                    target=target,
                    opts=opts,
                    primary_error=final_error,
                    base_health=base_health,
                    base_jacobian=jac,
                    sample_budget=sample_budget,
                    step_index=step_index,
                    prev_step=prev_step,
                    prev_step_delta=prev_step_delta,
                    collision_scoring=solver.get_collision_min_distance() > 0.0,
                    seed_worker_mode=seed_worker_mode,
                    executor=executor,
                    stats=parallel_stats,
                    cfg=parallel_cfg,
                )
                if mode in _RAMPED_PARALLEL_SEED_MODES:
                    q_ramped, recovered = _ramped_parallel_seed_step(
                        main_solver=solver,
                        main_robot=robot,
                        main_task=task,
                        q_current=q_before_step,
                        q_primary=q_primary,
                        q_target=q_recovered if recovered_target else None,
                        primary_error=final_error,
                        base_health=base_health,
                        base_jacobian=jac,
                        prev_step=prev_step,
                        collision_scoring=solver.get_collision_min_distance() > 0.0,
                        state=ramp_state,
                        stats=parallel_stats,
                        cfg=parallel_cfg,
                        opts=opts,
                    )
                    if recovered:
                        q_primary = q_ramped
                        final_error = _task_error_at(robot, task, q_primary)
                        robot.update_configuration(q_before_step)
                        task.update(robot)
                elif recovered_target:
                    q_primary = q_recovered
                    final_error = _task_error_at(robot, task, q_primary)
                    robot.update_configuration(q_before_step)
                    task.update(robot)
            elif mode in _HYBRID_MIN_ERROR_MODES:
                q_projected, recovered = _projected_health_gradient_step(
                    main_solver=solver,
                    main_robot=robot,
                    main_task=task,
                    q_current=q_before_step,
                    q_primary=q_primary,
                    primary_error=final_error,
                    base_health=base_health,
                    base_jacobian=jac,
                    prev_step=prev_step,
                    collision_scoring=solver.get_collision_min_distance() > 0.0,
                    stats=parallel_stats,
                    cfg=parallel_cfg,
                    opts=opts,
                )
                if recovered:
                    q_primary = q_projected
                    final_error = _task_error_at(robot, task, q_primary)
                    robot.update_configuration(q_before_step)
                    task.update(robot)

                q_hybrid, hybrid_recovered, extra_ms = _bounded_min_error_hybrid_step(
                    main_solver=solver,
                    main_robot=robot,
                    main_task=task,
                    q_current=q_before_step,
                    q_primary=q_primary,
                    target=target,
                    task_name=task_name,
                    primary_error=final_error,
                    base_health=base_health,
                    base_jacobian=jac,
                    prev_step=prev_step,
                    prev_step_delta=prev_step_delta,
                    collision_scoring=solver.get_collision_min_distance() > 0.0,
                    state=hybrid_state,
                    stats=parallel_stats,
                    cfg=parallel_cfg,
                    opts=opts,
                    smooth_on_reject=mode
                    in {
                        "projected_min_error_hybrid_smooth",
                        "projected_min_error_hybrid_smooth_stall_only",
                        "projected_min_error_hybrid_smooth_strict",
                        "projected_min_error_hybrid_smooth_strict_throttled",
                    },
                )
                if timings:
                    timings[-1] += extra_ms
                if hybrid_recovered:
                    q_primary = q_hybrid
                    final_error = _task_error_at(robot, task, q_primary)
                    robot.update_configuration(q_before_step)
                    task.update(robot)
            elif mode in _PROJECTED_GRADIENT_MODES:
                q_projected, recovered = _projected_health_gradient_step(
                    main_solver=solver,
                    main_robot=robot,
                    main_task=task,
                    q_current=q_before_step,
                    q_primary=q_primary,
                    primary_error=final_error,
                    base_health=base_health,
                    base_jacobian=jac,
                    prev_step=prev_step,
                    collision_scoring=solver.get_collision_min_distance() > 0.0,
                    stats=parallel_stats,
                    cfg=parallel_cfg,
                    opts=opts,
                )
                if recovered:
                    q_primary = q_projected
                    final_error = _task_error_at(robot, task, q_primary)
                    robot.update_configuration(q_before_step)
                    task.update(robot)
            q = q_primary
            if float(np.linalg.norm(q - prev_q)) < 1e-8:
                zero_progress += 1
            current_step = q - q_before_step
            current_step_delta = current_step - prev_step
            prev_step = current_step
            prev_step_delta = current_step_delta
            prev_q = q.copy()
            q_history.append(prev_q.copy())
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    summary = {
        "scenario": scenario.name,
        "mode": mode,
        "sample_budget": int(sample_budget),
        "collision_scoring": bool(collision_scoring),
        "status_counts": statuses,
        "start_error": float(start_error if start_error is not None else 0.0),
        "final_error": float(final_error if final_error is not None else 0.0),
        "zero_progress_steps": int(zero_progress),
        "health_sampling_applied_steps": int(health_applied),
        "health_sampling_cache_available_steps": int(health_cache_available),
        "health_sampling_cache_used_steps": int(health_cache_used),
        "health_sampling_activation_allowed_steps": int(health_activation_allowed),
        "health_sampling_accepted_samples": int(accepted_samples),
        "health_sampling_rejected_invalid": int(health_rejected_invalid),
        "health_sampling_rejected_limit": int(health_rejected_limit),
        "health_sampling_rejected_collision": int(health_rejected_collision),
        "health_sampling_rejected_score": int(health_rejected_score),
        "health_sampling_base_score": _stats(health_base_scores),
        "health_sampling_best_score": _stats(health_best_scores),
        "health_sampling_best_source_counts": health_best_source_counts,
        "collision_exact_distance_queries": int(exact_collision_queries),
        "parallel_seed_worker_mode": (
            seed_worker_mode if mode in _PARALLEL_SEED_MODES else "disabled"
        ),
        "parallel_seed_worker_count": int(len(parallel_workers)),
        "parallel_seed_collision_gate_enabled": bool(
            mode in _PARALLEL_SEED_MODES and solver.get_collision_min_distance() > 0.0
        ),
        "parallel_seed_attempted_steps": int(parallel_stats.attempted_steps),
        "parallel_seed_candidate_solves": int(parallel_stats.candidate_solves),
        "parallel_seed_accepted_steps": int(parallel_stats.accepted_steps),
        "parallel_seed_rejected_limit": int(parallel_stats.rejected_limit),
        "parallel_seed_rejected_collision": int(parallel_stats.rejected_collision),
        "parallel_seed_rejected_tracking": int(parallel_stats.rejected_tracking),
        "parallel_seed_rejected_continuity": int(parallel_stats.rejected_continuity),
        "parallel_seed_rejected_health": int(parallel_stats.rejected_health),
        "parallel_seed_best_score_delta": (
            float(parallel_stats.best_score_delta)
            if math.isfinite(parallel_stats.best_score_delta)
            else float("nan")
        ),
        "parallel_seed_avg_time_ms": (
            float(parallel_stats.time_ms_total / parallel_stats.attempted_steps)
            if parallel_stats.attempted_steps
            else 0.0
        ),
        "parallel_seed_ramped_target_updates": int(parallel_stats.ramped_target_updates),
        "parallel_seed_ramped_applied_steps": int(parallel_stats.ramped_applied_steps),
        "parallel_seed_ramped_decayed_steps": int(parallel_stats.ramped_decayed_steps),
        "parallel_seed_ramped_active_ticks": int(ramp_state.active_ticks),
        "projected_gradient_applied_steps": int(parallel_stats.projected_gradient_applied_steps),
        "hybrid_min_error_attempted_steps": int(parallel_stats.hybrid_min_error_attempted_steps),
        "hybrid_min_error_applied_steps": int(parallel_stats.hybrid_min_error_applied_steps),
        "hybrid_min_error_health_debt_steps": int(
            parallel_stats.hybrid_min_error_health_debt_steps
        ),
        "hybrid_min_error_active_ticks": int(hybrid_state.active_ticks),
        "hybrid_min_error_smoothed_steps": int(parallel_stats.hybrid_min_error_smoothed_steps),
        "hybrid_min_error_avg_time_ms": (
            float(
                parallel_stats.hybrid_min_error_time_ms_total
                / parallel_stats.hybrid_min_error_attempted_steps
            )
            if parallel_stats.hybrid_min_error_attempted_steps
            else 0.0
        ),
        "wall_time_ms": _stats(timings),
    }
    summary.update(summarize_health_series(health_samples))
    summary.update(collision_distance_stats_dict(collision_distances, threshold=0.0))
    q_lower, q_upper = robot.get_joint_limits()
    summary.update(
        summarize_continuity_series(
            q_history,
            dt=float(opts.dt),
            velocity_limits=np.asarray(robot.get_velocity_limits(), dtype=float),
            q_lower=np.asarray(q_lower, dtype=float),
            q_upper=np.asarray(q_upper, dtype=float),
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sample-budget", type=int, default=8)
    parser.add_argument(
        "--seed-worker-mode",
        choices=("sequential", "thread"),
        default="sequential",
        help="Execution mode for Python-side multi-seed recovery workers.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "baseline",
            "weighted_fallback",
            "min_error",
            "elastic_scale",
            "health_sampling",
            "health_sampling_cache_only",
            "health_sampling_collision",
            "parallel_seed_recovery",
            "health_sampling_parallel_seed",
            "ramped_parallel_seed",
            "ramped_parallel_seed_cache",
            "ramped_parallel_seed_strict",
            "projected_health_gradient",
            "projected_min_error_hybrid",
            "projected_min_error_hybrid_smooth",
            "projected_min_error_hybrid_smooth_stall_only",
            "projected_min_error_hybrid_smooth_strict",
            "projected_min_error_hybrid_smooth_strict_throttled",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".editor/reports/nullspace_health_matrix.json"),
    )
    args = parser.parse_args()

    scenarios = [
        Scenario("franka_panda_limit", _panda_limit_scenario),
        Scenario("franka_panda_collision", _panda_collision_scenario),
        Scenario("bimanual_dual_iiwa", _dual_iiwa_scenario),
    ]
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for scenario in scenarios:
        for mode in args.modes:
            try:
                rows.append(
                    _run_case(
                        scenario,
                        mode,
                        steps=args.steps,
                        sample_budget=args.sample_budget,
                        collision_scoring=mode == "health_sampling_collision",
                        seed_worker_mode=args.seed_worker_mode,
                    )
                )
            except Exception as exc:
                skipped.append({"scenario": scenario.name, "mode": mode, "reason": str(exc)})

    report = {
        "rows": rows,
        "skipped": skipped,
        "acceptance_notes": {
            "hard_failures": "status_counts should not shift toward failures",
            "tracking": "final_error should not regress materially vs baseline",
            "health": "joint_limit_health and limit_weighted_dexterity higher is better",
            "collision": "collision_count should not increase; min_self_collision_dist higher is better",
            "overhead": "compare wall_time_ms.p95 against baseline",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
