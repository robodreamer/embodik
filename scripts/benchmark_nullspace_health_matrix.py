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
from dataclasses import dataclass
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
    summarize_health_series,
)
from examples.utils.robot_models import ensure_ros_package_path  # noqa: E402


@dataclass(frozen=True)
class Scenario:
    name: str
    builder: Callable[[], tuple[eik.RobotModel, eik.KinematicsSolver, np.ndarray, np.ndarray, str]]


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
        "health_sampling_collision",
    }
    cfg.weighted_advisor_enabled = mode == "weighted_fallback"
    if mode in {"health_sampling", "health_sampling_collision"}:
        cfg.health_sampling.enabled = True
        cfg.health_sampling.seed = 20260606
        cfg.health_sampling.sample_count = int(sample_budget)
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


def _run_case(
    scenario: Scenario,
    mode: str,
    *,
    steps: int,
    sample_budget: int,
    collision_scoring: bool,
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
    health_applied = 0
    health_cache_available = 0
    health_cache_used = 0
    accepted_samples = 0
    exact_collision_queries = 0
    start_error = None
    final_error = None
    zero_progress = 0

    opts = eik.PositionStepOptions()
    opts.max_steps = 1
    opts.dt = solver.dt
    opts.position_gain = 10.0
    opts.orientation_gain = 10.0

    prev_q = np.asarray(q, dtype=float)
    for _ in range(steps):
        robot.update_configuration(q)
        task.update(robot)
        jac = np.asarray(task.get_jacobian(), dtype=float)
        health_samples.append(kinematic_health_sample(robot, q, jac))
        d = solver.evaluate_min_collision_distance(q)
        if math.isfinite(float(d)):
            collision_distances.append(float(d))

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
        accepted_samples += int(diag.health_sampling_accepted)
        exact_collision_queries += int(getattr(result, "collision_exact_distance_queries", 0))
        final_error = float(result.position_error + result.orientation_error)
        if start_error is None:
            start_error = final_error
        q = np.asarray(result.q_solution, dtype=float)
        if float(np.linalg.norm(q - prev_q)) < 1e-8:
            zero_progress += 1
        prev_q = q.copy()

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
        "health_sampling_accepted_samples": int(accepted_samples),
        "collision_exact_distance_queries": int(exact_collision_queries),
        "wall_time_ms": _stats(timings),
    }
    summary.update(summarize_health_series(health_samples))
    summary.update(collision_distance_stats_dict(collision_distances, threshold=0.0))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sample-budget", type=int, default=8)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "baseline",
            "weighted_fallback",
            "min_error",
            "elastic_scale",
            "health_sampling",
            "health_sampling_collision",
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
