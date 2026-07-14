#!/usr/bin/env python3
"""Continuous escape and return coverage for fully extended public arms."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

import embodik as eik

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from utils.robot_models import resolve_robot_configuration  # noqa: E402


@dataclass(frozen=True)
class SingularityCase:
    case_id: str
    preset: str
    base_frame: str
    arm_joint_prefix: str
    bend_joint: str
    limit_joint: str | None
    min_extension_ratio: float
    max_initial_sigma: float
    min_initial_condition: float
    min_tick6_sigma_improvement: float
    min_tick6_error_reduction: float
    min_return_error_reduction: float
    min_return_sigma_improvement: float
    add_nominal_posture: bool


CASES = (
    SingularityCase(
        case_id="panda-upper-limit-extension",
        preset="panda",
        base_frame="panda_link0",
        arm_joint_prefix="panda_joint",
        bend_joint="panda_joint4",
        limit_joint="panda_joint4",
        min_extension_ratio=0.99,
        max_initial_sigma=0.03,
        min_initial_condition=30.0,
        min_tick6_sigma_improvement=0.01,
        min_tick6_error_reduction=1e-3,
        min_return_error_reduction=1e-3,
        min_return_sigma_improvement=0.01,
        add_nominal_posture=False,
    ),
    SingularityCase(
        case_id="iiwa-exact-extension",
        preset="iiwa",
        base_frame="iiwa_link_0",
        arm_joint_prefix="iiwa_joint_",
        bend_joint="iiwa_joint_4",
        limit_joint=None,
        min_extension_ratio=0.99,
        max_initial_sigma=1e-8,
        min_initial_condition=1e8,
        min_tick6_sigma_improvement=0.006,
        min_tick6_error_reduction=5e-5,
        min_return_error_reduction=1e-4,
        min_return_sigma_improvement=0.005,
        add_nominal_posture=True,
    ),
)


def _frame_position(robot: eik.RobotModel, frame_name: str) -> np.ndarray:
    return np.asarray(robot.get_frame_pose(frame_name).translation, dtype=float)


def _position_condition(robot: eik.RobotModel, frame_name: str) -> tuple[float, float]:
    jacobian = np.asarray(robot.get_frame_jacobian(frame_name), dtype=float)[:3, :]
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1])
    normalized_sigma_min = sigma_min / max(sigma_max, 1e-12)
    condition_number = sigma_max / sigma_min if sigma_min > 1e-12 else float("inf")
    return normalized_sigma_min, condition_number


def _max_true_streak(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _seed_configuration(
    case: SingularityCase,
    robot: eik.RobotModel,
    upper: np.ndarray,
) -> np.ndarray:
    if case.preset == "panda":
        elbow_index = int(robot.get_joint_config_index("panda_joint4"))
        return np.array(
            [
                0.0,
                -0.017618,
                0.0,
                upper[elbow_index] - 1e-6,
                0.0,
                2.59284,
                0.0,
                0.02,
                0.02,
            ],
            dtype=float,
        )
    return np.zeros(robot.nq, dtype=float)


def _run_case(case: SingularityCase) -> dict[str, float | int | list[str]]:
    config = resolve_robot_configuration(case.preset)
    robot = config["robot"]
    frame_name = str(config["target_link"])
    joint_names = list(robot.get_joint_names())
    lower, upper = (np.asarray(value, dtype=float) for value in robot.get_joint_limits())
    q = _seed_configuration(case, robot, upper)
    robot.update_configuration(q)

    base_position = _frame_position(robot, case.base_frame)
    start_position = _frame_position(robot, frame_name)
    start_reach = float(np.linalg.norm(start_position - base_position))
    bend_index = int(robot.get_joint_config_index(case.bend_joint))
    bent_q = q.copy()
    bent_q[bend_index] -= 0.25
    robot.update_configuration(bent_q)
    bent_reach = float(np.linalg.norm(_frame_position(robot, frame_name) - base_position))
    robot.update_configuration(q)
    extension_ratio = start_reach / max(start_reach, bent_reach)

    initial_sigma, initial_condition = _position_condition(robot, frame_name)
    radial_direction = (start_position - base_position) / start_reach
    start_pose = np.eye(4, dtype=float)
    start_pose[:3, :3] = np.asarray(robot.get_frame_pose(frame_name).rotation, dtype=float)
    start_pose[:3, 3] = start_position
    escape_target = start_pose.copy()
    escape_target[:3, 3] -= 0.10 * radial_direction

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    runtime = eik.SolverRuntimeConfig()
    runtime.joint_limit_non_worsening_enabled = True
    runtime.joint_limit_non_worsening_margin = 0.04
    solver.configure_runtime(runtime)

    frame_task = solver.add_frame_task("ee", frame_name, eik.TaskType.FRAME_POSITION)
    frame_task.priority = 0
    frame_task.weight = 1.0
    frame_task.solve_mode = eik.TaskSolveMode.SCALE_ELASTIC
    frame_task.allow_min_error_fallback = True

    arm_velocity_indices = [
        int(robot.get_joint_velocity_index(name))
        for name in joint_names
        if name.startswith(case.arm_joint_prefix)
    ]
    conditioning = solver.add_manipulability_task(
        "conditioning", frame_name, eik.TaskType.FRAME_POSITION
    )
    conditioning.priority = 1
    conditioning.weight = 10.0
    conditioning.solve_mode = eik.TaskSolveMode.MIN_ERROR
    conditioning.allow_min_error_fallback = False
    conditioning.set_controlled_joint_indices(arm_velocity_indices)
    conditioning.set_regularization(0.03)

    if case.add_nominal_posture:
        posture = solver.add_posture_task("symmetry_selector")
        posture.priority = 2
        posture.weight = 1.0
        posture.solve_mode = eik.TaskSolveMode.MIN_ERROR
        posture.set_target_configuration(np.asarray(config["default_configuration"], dtype=float))

    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.dt = 0.02
    options.position_gain = 10.0
    options.orientation_gain = 0.0
    options.primary_solve_mode = eik.TaskSolveMode.SCALE_ELASTIC
    options.primary_allow_min_error_fallback = True
    options.max_configuration_step_norm = 0.08

    q_trace = [q.copy()]
    solve_inputs: list[np.ndarray] = []
    normalized_sigma = [initial_sigma]
    condition_numbers = [initial_condition]
    statuses: list[str] = []
    escape_errors = [float(np.linalg.norm(escape_target[:3, 3] - start_position))]

    def step(target: np.ndarray) -> None:
        nonlocal q
        solve_inputs.append(q.copy())
        result = solver.solve_position_step(q, target, "ee", options)
        statuses.append(getattr(result.status, "name", str(result.status)))
        q_next = np.asarray(result.q_solution, dtype=float)
        assert q_next.shape == q.shape
        assert np.all(np.isfinite(q_next))
        q = q_next
        robot.update_configuration(q)
        q_trace.append(q.copy())
        sigma, condition = _position_condition(robot, frame_name)
        normalized_sigma.append(sigma)
        condition_numbers.append(condition)

    for _ in range(8):
        step(escape_target)
        escape_errors.append(
            float(np.linalg.norm(escape_target[:3, 3] - _frame_position(robot, frame_name)))
        )

    return_start_index = len(q_trace) - 1
    return_errors = [float(np.linalg.norm(start_position - _frame_position(robot, frame_name)))]
    for _ in range(16):
        step(start_pose)
        return_errors.append(
            float(np.linalg.norm(start_position - _frame_position(robot, frame_name)))
        )

    stationary_run_steps = 240
    settling_window_steps = 120
    for _ in range(stationary_run_steps):
        step(start_pose)
        return_errors.append(
            float(np.linalg.norm(start_position - _frame_position(robot, frame_name)))
        )

    q_array = np.vstack(q_trace)
    input_array = np.vstack(solve_inputs)
    q_steps = np.linalg.norm(np.diff(q_array, axis=0), axis=1)
    q_acceleration = np.linalg.norm(np.diff(q_array, n=2, axis=0), axis=1)
    q_jerk = np.linalg.norm(np.diff(q_array, n=3, axis=0), axis=1)
    joint_slack = np.minimum(q_array - lower, upper - q_array)
    active_errors = np.concatenate([np.asarray(escape_errors[1:]), np.asarray(return_errors[1:])])
    zero_motion = (q_steps < 1e-5) & (active_errors > 2e-2)
    hold_q = q_array[-(settling_window_steps + 1) :]
    hold_deltas = np.diff(hold_q, axis=0)
    hold_velocities = hold_deltas / options.dt
    hold_speeds = np.linalg.norm(hold_velocities, axis=1)
    hold_acceleration = np.diff(hold_velocities, axis=0) / options.dt
    hold_jerk = np.diff(hold_acceleration, axis=0) / options.dt
    hold_products = np.einsum("ij,ij->i", hold_velocities[:-1], hold_velocities[1:])
    hold_active = (hold_speeds[:-1] > 1e-3) & (hold_speeds[1:] > 1e-3)
    hold_errors = np.asarray(return_errors[-settling_window_steps:], dtype=float)

    return_recovery_lag = 16
    for index in range(6):
        step_index = return_start_index + index
        if q_steps[step_index] > 1e-5 and return_errors[index + 1] <= (return_errors[index] + 1e-6):
            return_recovery_lag = index
            break

    limit_exit = float("nan")
    if case.limit_joint is not None:
        limit_index = int(robot.get_joint_config_index(case.limit_joint))
        limit_exit = float(upper[limit_index] - q_array[6, limit_index])

    return {
        "extension_ratio": extension_ratio,
        "initial_normalized_sigma_min": normalized_sigma[0],
        "initial_condition_number": condition_numbers[0],
        "tick6_normalized_sigma_min": normalized_sigma[6],
        "tick6_condition_number": condition_numbers[6],
        "tick6_error_reduction_m": escape_errors[0] - escape_errors[6],
        "tick6_limit_exit_rad": limit_exit,
        "return_tick6_error_reduction_m": return_errors[0] - return_errors[6],
        "return_tick6_normalized_sigma_min": normalized_sigma[return_start_index + 6],
        "hold_start_error_m": return_errors[-(settling_window_steps + 1)],
        "final_return_error_m": return_errors[-1],
        "return_switch_step_norm": q_steps[return_start_index],
        "return_recovery_lag_steps": return_recovery_lag,
        "max_step_norm": float(q_steps.max(initial=0.0)),
        "max_acceleration_norm": float(q_acceleration.max(initial=0.0)),
        "max_jerk_norm": float(q_jerk.max(initial=0.0)),
        "minimum_joint_limit_slack": float(np.min(joint_slack)),
        "max_zero_motion_streak": _max_true_streak(zero_motion),
        "post_settle_rms_joint_velocity": float(np.sqrt(np.mean(hold_speeds**2))),
        "post_settle_peak_joint_velocity": float(hold_speeds.max(initial=0.0)),
        "post_settle_velocity_total_variation": float(
            np.sum(np.linalg.norm(np.diff(hold_velocities, axis=0), axis=1))
        ),
        "post_settle_configuration_drift": float(np.linalg.norm(hold_q[-1] - hold_q[0])),
        "post_settle_alternating_steps": int(np.sum((hold_products < 0.0) & hold_active)),
        "post_settle_error_increases": int(np.sum(np.diff(hold_errors) > 1e-4)),
        "post_settle_peak_acceleration": float(
            np.linalg.norm(hold_acceleration, axis=1).max(initial=0.0)
        ),
        "post_settle_peak_jerk": float(np.linalg.norm(hold_jerk, axis=1).max(initial=0.0)),
        "no_reset_input_mismatches": int(
            np.count_nonzero(np.any(input_array != q_array[:-1], axis=1))
        ),
        "statuses": sorted(set(statuses)),
    }


@pytest.mark.benchmark
@pytest.mark.parametrize("case", CASES, ids=[case.case_id for case in CASES])
def test_fully_extended_arm_escapes_and_returns_without_reset(
    case: SingularityCase,
    record_property: Callable[[str, object], None],
) -> None:
    metrics = _run_case(case)
    for key, value in metrics.items():
        record_property(key, value)
    failure_context = "\n".join(f"{key}={value}" for key, value in metrics.items())

    assert metrics["extension_ratio"] >= case.min_extension_ratio, failure_context
    assert metrics["initial_normalized_sigma_min"] <= case.max_initial_sigma, failure_context
    assert metrics["initial_condition_number"] >= case.min_initial_condition, failure_context
    assert "NUMERICAL_ERROR" not in metrics["statuses"], failure_context
    assert metrics["minimum_joint_limit_slack"] >= -1e-8, failure_context
    assert metrics["max_step_norm"] <= 0.08 + 1e-9, failure_context
    assert metrics["max_acceleration_norm"] <= 0.16, failure_context
    assert metrics["max_jerk_norm"] <= 0.24, failure_context
    assert metrics["max_zero_motion_streak"] <= 2, failure_context
    assert metrics["post_settle_rms_joint_velocity"] <= 0.005, failure_context
    assert metrics["post_settle_peak_joint_velocity"] <= 0.02, failure_context
    assert metrics["post_settle_velocity_total_variation"] <= 0.05, failure_context
    assert metrics["post_settle_configuration_drift"] <= 0.005, failure_context
    assert metrics["post_settle_alternating_steps"] <= 2, failure_context
    assert metrics["post_settle_error_increases"] <= 1, failure_context
    assert metrics["post_settle_peak_acceleration"] <= 2.5, failure_context
    assert metrics["post_settle_peak_jerk"] <= 250.0, failure_context
    assert metrics["return_recovery_lag_steps"] <= 6, failure_context
    assert metrics["no_reset_input_mismatches"] == 0, failure_context

    assert metrics["tick6_error_reduction_m"] >= case.min_tick6_error_reduction, failure_context
    assert (
        metrics["tick6_normalized_sigma_min"]
        >= metrics["initial_normalized_sigma_min"] + case.min_tick6_sigma_improvement
    ), failure_context
    assert (
        metrics["return_tick6_error_reduction_m"] >= case.min_return_error_reduction
    ), failure_context
    assert (
        metrics["return_tick6_normalized_sigma_min"]
        >= metrics["initial_normalized_sigma_min"] + case.min_return_sigma_improvement
    ), failure_context
    assert metrics["return_switch_step_norm"] <= 0.08 + 1e-9, failure_context
    if case.limit_joint is not None:
        assert metrics["tick6_limit_exit_rad"] >= 1e-3, failure_context
