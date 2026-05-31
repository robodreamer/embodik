"""Headless scenarios comparing fixed vs auto-tuned IK gains (panda, no Viser)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

import embodik

from example_helpers.adaptive_gain_tuning import (
    AdaptiveGainTuningConfig,
    AdaptiveGainTuningState,
    compute_effective_gains,
)
from example_helpers.ik_common import configure_solver_runtime_policy

try:
    from robot_descriptions.panda_description import URDF_PATH as PANDA_URDF_PATH
except ImportError as exc:  # pragma: no cover - optional in minimal envs
    raise ImportError("robot_descriptions.panda_description is required") from exc


CONTROL_HZ = 200
DEFAULT_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.035, 0.035])
TARGET_JUMP_M = 0.30
CONVERGE_ERROR_M = 0.01
STEP_JUMP_MAX_CYCLES = 400
SINE_STREAM_SECONDS = 2.0
SINE_AMPLITUDE_M = 0.08
SINE_FREQ_HZ = 0.5


@dataclass(frozen=True, slots=True)
class ScenarioMetrics:
    name: str
    mode: str
    ticks_to_converge: int | None
    peak_position_error_m: float
    final_position_error_m: float
    mean_position_error_m: float
    p95_position_error_m: float
    mean_position_gain: float
    std_position_gain: float
    mean_orientation_gain: float
    max_position_gain: float
    steps: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "ticks_to_converge": self.ticks_to_converge,
            "peak_position_error_m": self.peak_position_error_m,
            "final_position_error_m": self.final_position_error_m,
            "mean_position_error_m": self.mean_position_error_m,
            "p95_position_error_m": self.p95_position_error_m,
            "mean_position_gain": self.mean_position_gain,
            "std_position_gain": self.std_position_gain,
            "mean_orientation_gain": self.mean_orientation_gain,
            "max_position_gain": self.max_position_gain,
            "steps": self.steps,
        }


def _setup_ros_package_path(urdf_path: str) -> None:
    import os
    from pathlib import Path

    existing = os.environ.get("ROS_PACKAGE_PATH", "")
    paths: set[str] = set()
    p = Path(urdf_path).parent
    while p != p.parent:
        paths.add(str(p))
        p = p.parent
    new_part = ":".join(paths)
    os.environ["ROS_PACKAGE_PATH"] = f"{existing}:{new_part}" if existing else new_part


def make_panda_solver(*, with_collision: bool = False) -> tuple[embodik.RobotModel, embodik.KinematicsSolver]:
    _setup_ros_package_path(PANDA_URDF_PATH)
    robot = embodik.RobotModel(PANDA_URDF_PATH)
    solver = embodik.KinematicsSolver(robot)
    configure_solver_runtime_policy(solver)
    solver.dt = 1.0 / CONTROL_HZ
    solver.set_damping(0.01)
    ee_task = solver.add_frame_task("ee_task", "panda_hand")
    ee_task.weight = 10.0
    ee_task.priority = 0
    if with_collision:
        solver.configure_collision_constraint(min_distance=0.03)
        solver.set_collision_tuning_mode(embodik.CollisionTuningMode.BALANCED)
    return robot, solver


def _random_jump_target(
    robot: embodik.RobotModel, q: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    robot.update_configuration(q)
    ee_pose = robot.get_frame_pose("panda_hand")
    direction = rng.standard_normal(3)
    direction /= max(np.linalg.norm(direction), 1e-12)
    target_pos = ee_pose.translation + TARGET_JUMP_M * direction
    target = np.eye(4)
    target[:3, :3] = ee_pose.rotation
    target[:3, 3] = target_pos
    return target


def _position_error(robot: embodik.RobotModel, target: np.ndarray) -> float:
    ee = robot.get_frame_pose("panda_hand")
    return float(np.linalg.norm(np.asarray(target[:3, 3], dtype=float) - ee.translation))


def _run_tracking_loop(
    *,
    robot: embodik.RobotModel,
    solver: embodik.KinematicsSolver,
    q0: np.ndarray,
    target_pose_fn: Any,
    n_steps: int,
    base_position_gain: float,
    base_orientation_gain: float,
    max_steps: int,
    auto_tune: bool,
    tuning_config: AdaptiveGainTuningConfig,
    disable_adaptive_dt_when_auto: bool = True,
) -> ScenarioMetrics:
    q = np.asarray(q0, dtype=float).copy()
    robot.update_configuration(q)
    state = AdaptiveGainTuningState()
    pos_errors: list[float] = []
    pos_gains: list[float] = []
    ori_gains: list[float] = []
    ticks_to_converge: int | None = None
    peak_error = 0.0

    for step_idx in range(n_steps):
        target = target_pose_fn(step_idx, q, robot)
        pos_err = _position_error(robot, target)
        peak_error = max(peak_error, pos_err)
        pos_errors.append(pos_err)
        if ticks_to_converge is None and pos_err <= CONVERGE_ERROR_M:
            ticks_to_converge = step_idx + 1

        eff_pos, eff_ori, _, _ = compute_effective_gains(
            base_position_gain=base_position_gain,
            base_orientation_gain=base_orientation_gain,
            position_error_m=pos_err,
            orientation_error_rad=0.0,
            config=tuning_config,
            state=state,
            enabled=auto_tune,
        )
        pos_gains.append(eff_pos)
        ori_gains.append(eff_ori)

        opts = embodik.PositionStepOptions()
        opts.max_steps = max(1, int(max_steps))
        opts.position_gain = eff_pos
        opts.orientation_gain = eff_ori
        if auto_tune:
            opts.adaptive_dt = not disable_adaptive_dt_when_auto
        else:
            opts.adaptive_dt = True
        if opts.adaptive_dt:
            opts.adaptive_dt_max_scale = 5.0
            opts.adaptive_dt_reference_distance = 0.05

        result = solver.solve_position_step(q, target, "ee_task", opts)
        q_sol = np.asarray(getattr(result, "q_solution", q), dtype=float)
        if q_sol.shape == q.shape and np.all(np.isfinite(q_sol)):
            q = q_sol
        robot.update_configuration(q)

    pos_arr = np.asarray(pos_errors, dtype=float)
    gain_arr = np.asarray(pos_gains, dtype=float)
    mode = "auto" if auto_tune else "fixed"
    return ScenarioMetrics(
        name="tracking",
        mode=mode,
        ticks_to_converge=ticks_to_converge,
        peak_position_error_m=float(peak_error),
        final_position_error_m=float(pos_arr[-1]) if pos_arr.size else float("inf"),
        mean_position_error_m=float(np.mean(pos_arr)) if pos_arr.size else float("inf"),
        p95_position_error_m=float(np.percentile(pos_arr, 95)) if pos_arr.size else float("inf"),
        mean_position_gain=float(np.mean(gain_arr)) if gain_arr.size else base_position_gain,
        std_position_gain=float(np.std(gain_arr)) if gain_arr.size else 0.0,
        mean_orientation_gain=float(np.mean(ori_gains)) if ori_gains else base_orientation_gain,
        max_position_gain=float(np.max(gain_arr)) if gain_arr.size else base_position_gain,
        steps=n_steps,
    )


def run_step_jump_scenario(
    *,
    base_position_gain: float = 10.0,
    base_orientation_gain: float = 10.0,
    max_steps: int = 2,
    seed: int = 0,
    tuning_config: AdaptiveGainTuningConfig | None = None,
) -> dict[str, ScenarioMetrics]:
    """Single random 0.3 m target jump; compare fixed vs auto gains."""
    cfg = tuning_config or AdaptiveGainTuningConfig()
    robot, solver = make_panda_solver(with_collision=False)
    rng = np.random.default_rng(seed)
    q0 = DEFAULT_Q.copy()
    target = _random_jump_target(robot, q0, rng)

    def target_fn(_step: int, _q: np.ndarray, _robot: embodik.RobotModel) -> np.ndarray:
        return target

    fixed = _run_tracking_loop(
        robot=robot,
        solver=solver,
        q0=q0,
        target_pose_fn=target_fn,
        n_steps=STEP_JUMP_MAX_CYCLES,
        base_position_gain=base_position_gain,
        base_orientation_gain=base_orientation_gain,
        max_steps=max_steps,
        auto_tune=False,
        tuning_config=cfg,
        disable_adaptive_dt_when_auto=False,
    )
    robot, solver = make_panda_solver(with_collision=False)
    auto = _run_tracking_loop(
        robot=robot,
        solver=solver,
        q0=q0,
        target_pose_fn=target_fn,
        n_steps=STEP_JUMP_MAX_CYCLES,
        base_position_gain=base_position_gain,
        base_orientation_gain=base_orientation_gain,
        max_steps=max_steps,
        auto_tune=True,
        tuning_config=cfg,
    )
    return {"step_jump": fixed, "step_jump_auto": auto}


def run_sine_stream_scenario(
    *,
    base_position_gain: float = 10.0,
    base_orientation_gain: float = 10.0,
    max_steps: int = 2,
    seed: int = 0,
    tuning_config: AdaptiveGainTuningConfig | None = None,
) -> dict[str, ScenarioMetrics]:
    """Sinusoidal EE target stream at CONTROL_HZ."""
    cfg = tuning_config or AdaptiveGainTuningConfig()
    robot, solver = make_panda_solver(with_collision=False)
    q0 = DEFAULT_Q.copy()
    robot.update_configuration(q0)
    start_pose = robot.get_frame_pose("panda_hand")
    origin = np.array(start_pose.translation, dtype=float, copy=True)
    start_rotation = np.array(start_pose.rotation, dtype=float, copy=True)
    axis = np.array([1.0, 0.0, 0.0])
    n_steps = max(1, int(SINE_STREAM_SECONDS * CONTROL_HZ))
    phase0 = float(seed) * 0.1

    def target_fn(step: int, _q: np.ndarray, _robot: embodik.RobotModel) -> np.ndarray:
        t = step / CONTROL_HZ
        offset = SINE_AMPLITUDE_M * math.sin(2.0 * math.pi * SINE_FREQ_HZ * t + phase0)
        target = np.eye(4)
        target[:3, :3] = start_rotation
        target[:3, 3] = origin + axis * offset
        return target

    fixed = _run_tracking_loop(
        robot=robot,
        solver=solver,
        q0=q0,
        target_pose_fn=target_fn,
        n_steps=n_steps,
        base_position_gain=base_position_gain,
        base_orientation_gain=base_orientation_gain,
        max_steps=max_steps,
        auto_tune=False,
        tuning_config=cfg,
    )
    robot, solver = make_panda_solver(with_collision=False)
    auto = _run_tracking_loop(
        robot=robot,
        solver=solver,
        q0=q0,
        target_pose_fn=target_fn,
        n_steps=n_steps,
        base_position_gain=base_position_gain,
        base_orientation_gain=base_orientation_gain,
        max_steps=max_steps,
        auto_tune=True,
        tuning_config=cfg,
    )
    return {"sine_stream": fixed, "sine_stream_auto": auto}


def compare_step_jump_modes(
    fixed: ScenarioMetrics,
    auto: ScenarioMetrics,
    *,
    base_position_gain: float,
    max_scale: float = 3.0,
) -> dict[str, Any]:
    """Regression checks for a large target jump (fixed uses adaptive dt; auto does not)."""
    ticks_ok = auto.ticks_to_converge is not None and auto.ticks_to_converge < STEP_JUMP_MAX_CYCLES
    final_error_ok = auto.final_position_error_m <= CONVERGE_ERROR_M * 5.0
    stable_ok = auto.peak_position_error_m <= TARGET_JUMP_M * 1.35 + 1e-6
    gain_cap = float(base_position_gain) * float(max_scale) * 1.01
    gain_bounded = auto.max_position_gain <= gain_cap
    faster_than_fixed = (
        fixed.ticks_to_converge is not None
        and auto.ticks_to_converge is not None
        and auto.ticks_to_converge <= fixed.ticks_to_converge
    )
    return {
        "ticks_ok": ticks_ok,
        "final_error_ok": final_error_ok,
        "stable_ok": stable_ok,
        "gain_bounded": gain_bounded,
        "gain_cap": gain_cap,
        "faster_than_fixed": faster_than_fixed,
        "passed": bool(ticks_ok and final_error_ok and stable_ok and gain_bounded),
    }


def compare_sine_stream_modes(
    fixed: ScenarioMetrics,
    auto: ScenarioMetrics,
    *,
    base_position_gain: float,
    max_scale: float = 3.0,
) -> dict[str, Any]:
    error_ok = auto.p95_position_error_m <= max(
        fixed.p95_position_error_m * 1.15 + 1e-6, SINE_AMPLITUDE_M * 0.35
    )
    stable_ok = auto.peak_position_error_m <= max(
        fixed.peak_position_error_m * 2.0 + 1e-6, SINE_AMPLITUDE_M * 5.0
    )
    gain_cap = float(base_position_gain) * float(max_scale) * 1.01
    gain_bounded = auto.max_position_gain <= gain_cap
    return {
        "p95_error_ok": error_ok,
        "stable_ok": stable_ok,
        "gain_bounded": gain_bounded,
        "gain_cap": gain_cap,
        "passed": bool(error_ok and stable_ok and gain_bounded),
    }


def compare_modes(
    fixed: ScenarioMetrics,
    auto: ScenarioMetrics,
    *,
    base_position_gain: float,
    max_scale: float = 3.0,
) -> dict[str, Any]:
    """Generic compare — prefer scenario-specific helpers."""
    if fixed.ticks_to_converge is not None and auto.ticks_to_converge is not None:
        ticks_ok = auto.ticks_to_converge <= int(math.ceil(fixed.ticks_to_converge * 1.15))
    elif auto.ticks_to_converge is not None:
        ticks_ok = True
    else:
        ticks_ok = auto.peak_position_error_m <= fixed.peak_position_error_m * 1.05 + 1e-9

    error_ok = auto.p95_position_error_m <= fixed.p95_position_error_m * 1.10 + 1e-6
    stable_ok = auto.peak_position_error_m <= max(
        TARGET_JUMP_M * 1.05, fixed.peak_position_error_m * 1.5
    )
    gain_cap = float(base_position_gain) * float(max_scale) * 1.01
    gain_bounded = auto.max_position_gain <= gain_cap

    return {
        "ticks_ok": ticks_ok,
        "p95_error_ok": error_ok,
        "stable_ok": stable_ok,
        "gain_bounded": gain_bounded,
        "gain_cap": gain_cap,
        "passed": bool(ticks_ok and error_ok and stable_ok and gain_bounded),
    }


def run_all_scenarios(
    *,
    base_position_gain: float = 10.0,
    base_orientation_gain: float = 10.0,
    max_steps: int = 2,
    seed: int = 0,
    tuning_config: AdaptiveGainTuningConfig | None = None,
) -> dict[str, Any]:
    cfg = tuning_config or AdaptiveGainTuningConfig()
    kwargs = {
        "base_position_gain": base_position_gain,
        "base_orientation_gain": base_orientation_gain,
        "max_steps": max_steps,
        "seed": seed,
        "tuning_config": cfg,
    }
    jump = run_step_jump_scenario(**kwargs)
    sine = run_sine_stream_scenario(**kwargs)
    jump_checks = compare_step_jump_modes(
        jump["step_jump"],
        jump["step_jump_auto"],
        base_position_gain=base_position_gain,
        max_scale=cfg.max_scale,
    )
    sine_checks = compare_sine_stream_modes(
        sine["sine_stream"],
        sine["sine_stream_auto"],
        base_position_gain=base_position_gain,
        max_scale=cfg.max_scale,
    )
    return {
        "config": {
            "base_position_gain": base_position_gain,
            "base_orientation_gain": base_orientation_gain,
            "max_steps": max_steps,
            "seed": seed,
            "tuning": {
                "position_reference_m": cfg.position_reference_m,
                "orientation_reference_rad": cfg.orientation_reference_rad,
                "min_scale": cfg.min_scale,
                "max_scale": cfg.max_scale,
                "smoothing": cfg.smoothing,
            },
        },
        "step_jump": {
            "fixed": jump["step_jump"].as_dict(),
            "auto": jump["step_jump_auto"].as_dict(),
            "checks": jump_checks,
        },
        "sine_stream": {
            "fixed": sine["sine_stream"].as_dict(),
            "auto": sine["sine_stream_auto"].as_dict(),
            "checks": sine_checks,
        },
        "passed": bool(jump_checks["passed"] and sine_checks["passed"]),
    }
