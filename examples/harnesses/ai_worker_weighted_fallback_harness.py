#!/usr/bin/env python3
"""Headless AI Worker comparison for constrained weighted fallback.

The scenario intentionally blocks a high-priority lift command at its upper
joint limit while leaving a secondary arm posture row feasible. The strict
prioritized solve should classify the step as infeasible; the opt-in
constrained weighted fallback may accept the feasible weighted candidate.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import embodik
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from example_helpers.common_bimanual_model_utils import (
        default_common_bimanual_ik_joint_names,
    )
except ModuleNotFoundError as exc:
    if exc.name != "example_helpers" and not str(exc.name).startswith("example_helpers."):
        raise
    from examples.example_helpers.common_bimanual_model_utils import (
        default_common_bimanual_ik_joint_names,
    )

_AI_WORKER_COLLISION_URDF = (
    _REPO_ROOT
    / "examples"
    / "assets"
    / "ai_worker"
    / "generated"
    / "sg2"
    / "ffw_sg2_mobile_robot_teleop_collision.urdf"
)
_BLOCKED_JOINT = "lift_joint"
_SECONDARY_JOINT = "arm_r_joint3"


@dataclass(frozen=True)
class AIWorkerWeightedFallbackComparison:
    """Result bundle for one AI Worker weighted-fallback comparison."""

    prioritized_status: object
    weighted_status: object
    weighted_fallback_used: bool
    weighted_advisory_available: bool
    prioritized_solution: np.ndarray
    weighted_solution: np.ndarray
    q_seed: np.ndarray
    q_next: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    blocked_index: int
    secondary_index: int
    weighted_status_message: str


def _load_reduced_ai_worker_robot() -> embodik.RobotModel:
    if not _AI_WORKER_COLLISION_URDF.is_file():
        raise FileNotFoundError(
            f"AI Worker collision URDF not found: {_AI_WORKER_COLLISION_URDF}"
        )
    full = embodik.RobotModel(str(_AI_WORKER_COLLISION_URDF), floating_base=False)
    return embodik.RobotModel(
        str(_AI_WORKER_COLLISION_URDF),
        actuated_joint_names=default_common_bimanual_ik_joint_names(full.get_joint_names()),
        floating_base=False,
    )


def _make_solver(weighted_fallback: bool):
    robot = _load_reduced_ai_worker_robot()
    lower, upper = robot.get_joint_limits()
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    q = np.clip(
        np.asarray(robot.neutral_configuration(), dtype=float),
        lower + 0.02,
        upper - 0.02,
    )

    blocked_index = int(robot.get_joint_config_index(_BLOCKED_JOINT))
    secondary_index = int(robot.get_joint_config_index(_SECONDARY_JOINT))
    q[blocked_index] = upper[blocked_index]
    robot.update_configuration(q)

    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.0)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    blocked = solver.add_joint_task(
        "blocked_lift", _BLOCKED_JOINT, target_value=upper[blocked_index] + 0.4
    )
    blocked.priority = 0
    blocked.weight = 1.0
    blocked.allow_min_error_fallback = False

    secondary = solver.add_posture_task("useful_arm", [secondary_index])
    secondary.priority = 1
    secondary.weight = 1.0
    secondary.allow_min_error_fallback = False
    secondary.set_controlled_joint_targets(np.array([q[secondary_index] + 0.4]))

    runtime = embodik.SolverRuntimeConfig()
    runtime.weighted_fallback_enabled = bool(weighted_fallback)
    solver.configure_runtime(runtime)

    return solver, q, lower, upper, blocked_index, secondary_index


def run_ai_worker_weighted_fallback_comparison() -> AIWorkerWeightedFallbackComparison:
    """Compare strict priority and constrained weighted fallback on AI Worker."""
    prioritized_solver, q_seed, lower, upper, blocked_index, secondary_index = _make_solver(
        weighted_fallback=False
    )
    prioritized = prioritized_solver.solve_velocity(q_seed, apply_limits=True)
    prioritized_solution = np.asarray(prioritized.solution, dtype=float)

    weighted_solver, q_seed_weighted, lower_weighted, upper_weighted, _, _ = _make_solver(
        weighted_fallback=True
    )
    weighted = weighted_solver.solve_velocity(q_seed_weighted, apply_limits=True)
    weighted_solution = np.asarray(weighted.solution, dtype=float)
    q_next = q_seed_weighted + weighted_solver.dt * weighted_solution

    return AIWorkerWeightedFallbackComparison(
        prioritized_status=prioritized.status,
        weighted_status=weighted.status,
        weighted_fallback_used=bool(weighted.weighted_fallback_used),
        weighted_advisory_available=bool(weighted.weighted_advisory_available),
        prioritized_solution=prioritized_solution,
        weighted_solution=weighted_solution,
        q_seed=q_seed_weighted,
        q_next=q_next,
        lower=lower_weighted,
        upper=upper_weighted,
        blocked_index=blocked_index,
        secondary_index=secondary_index,
        weighted_status_message=str(weighted.status_message),
    )


def main() -> int:
    comparison = run_ai_worker_weighted_fallback_comparison()
    print(
        "[ai-worker-weighted-fallback] "
        f"prioritized={comparison.prioritized_status} "
        f"weighted={comparison.weighted_status} "
        f"fallback_used={comparison.weighted_fallback_used} "
        f"blocked_v={comparison.weighted_solution[comparison.blocked_index]:.6g} "
        f"secondary_v={comparison.weighted_solution[comparison.secondary_index]:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
