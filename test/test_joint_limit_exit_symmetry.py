#!/usr/bin/env python3
"""Regression tests for joint-limit saturation exit symmetry."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np

import embodik as eik


def create_test_urdf() -> str:
    """Create a simple 1-DoF URDF with position and velocity limits."""
    urdf_content = """<?xml version="1.0"?>
<robot name="saturation_symmetry_robot">
  <link name="base_link"/>
  <link name="link1"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0.0" upper="1.0" velocity="1.0" effort="10.0"/>
  </joint>
</robot>"""
    fd, path = tempfile.mkstemp(suffix=".urdf")
    with os.fdopen(fd, "w") as file_obj:
        file_obj.write(urdf_content)
    return path


@dataclass
class SymmetryMetrics:
    entry_steps_to_saturation: int
    exit_steps_from_saturation: int
    saturation_dwell_steps: int
    stall_count_reverse: int
    oscillation_count_near_limit: int
    return_error: float

    def score(
        self,
        w_entry_exit: float = 1.0,
        w_return_error: float = 150.0,
        w_stall: float = 3.0,
        w_oscillation: float = 1.0,
    ) -> float:
        return (
            w_entry_exit * abs(self.entry_steps_to_saturation - self.exit_steps_from_saturation)
            + w_return_error * self.return_error
            + w_stall * self.stall_count_reverse
            + w_oscillation * self.oscillation_count_near_limit
        )


def _run_round_trip(
    solver: eik.KinematicsSolver,
    *,
    q_start: float = 0.70,
    q_forward_target: float = 1.30,
    q_reverse_target: float | None = None,
    per_phase_steps: int = 90,
    margin_limit: float = 1e-3,
) -> SymmetryMetrics:
    """Run a deterministic forward-then-reverse command profile."""
    if q_reverse_target is None:
        q_reverse_target = q_start

    velocity_limit = 1.0
    acceleration_limit = 10.0
    dt = solver.dt
    q = q_start

    entry_step_idx: int | None = None
    exit_step_idx: int | None = None
    saturation_dwell = 0
    stall_reverse = 0
    oscillations = 0
    prev_near_vel: float | None = None

    def step_once(target: float, in_reverse_phase: bool, step_idx: int) -> None:
        nonlocal q
        nonlocal entry_step_idx
        nonlocal exit_step_idx
        nonlocal saturation_dwell
        nonlocal stall_reverse
        nonlocal oscillations
        nonlocal prev_near_vel

        lower_margin = q - 0.0 - margin_limit
        upper_margin = 1.0 - q - margin_limit
        lower_bound, upper_bound = solver.calculate_velocity_box_constraint(
            lower_margin,
            upper_margin,
            velocity_limit,
            acceleration_limit,
            dt,
        )

        desired_velocity = np.clip((target - q) / dt, -velocity_limit, velocity_limit)
        commanded_velocity = float(np.clip(desired_velocity, lower_bound, upper_bound))
        q = float(np.clip(q + commanded_velocity * dt, 0.0, 1.0))

        near_upper = q >= (1.0 - 2.0 * margin_limit)
        if near_upper:
            saturation_dwell += 1
            if entry_step_idx is None and not in_reverse_phase:
                entry_step_idx = step_idx
            if abs(q - 1.0) <= 5.0 * margin_limit:
                if prev_near_vel is not None and np.sign(prev_near_vel) != np.sign(
                    commanded_velocity
                ):
                    oscillations += 1
                prev_near_vel = commanded_velocity
        elif in_reverse_phase and exit_step_idx is None:
            exit_step_idx = step_idx

        if in_reverse_phase and desired_velocity < -1e-6 and commanded_velocity > -1e-6:
            stall_reverse += 1

    for i in range(per_phase_steps):
        step_once(q_forward_target, False, i)

    for i in range(per_phase_steps):
        step_once(q_reverse_target, True, i)

    entry = entry_step_idx if entry_step_idx is not None else per_phase_steps
    if exit_step_idx is None:
        exit = per_phase_steps
    else:
        exit = exit_step_idx

    return SymmetryMetrics(
        entry_steps_to_saturation=entry,
        exit_steps_from_saturation=exit,
        saturation_dwell_steps=saturation_dwell,
        stall_count_reverse=stall_reverse,
        oscillation_count_near_limit=oscillations,
        return_error=abs(q - q_start),
    )


def _build_solver() -> tuple[str, eik.KinematicsSolver]:
    urdf_path = create_test_urdf()
    robot = eik.RobotModel(urdf_path, floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    return urdf_path, solver


def test_forward_reverse_saturation_symmetry_metric_guardrail():
    urdf_path, solver = _build_solver()
    try:
        metrics = _run_round_trip(solver)
        assert metrics.stall_count_reverse <= 2
        assert metrics.return_error <= 0.08
        assert metrics.score() <= 60.0
    finally:
        os.remove(urdf_path)


def test_release_margin_and_hysteresis_improve_exit_symmetry():
    urdf_path, solver = _build_solver()
    try:
        baseline = _run_round_trip(solver)

        solver.set_limit_exit_release_margin(0.003)
        solver.set_limit_recovery_hysteresis(1e-4, 3e-4)
        improved = _run_round_trip(solver)

        assert improved.stall_count_reverse <= baseline.stall_count_reverse
        assert improved.oscillation_count_near_limit <= baseline.oscillation_count_near_limit
        assert improved.score() <= baseline.score()
    finally:
        os.remove(urdf_path)


def test_optional_self_correcting_tuner_improves_score():
    if os.environ.get("EMBODIK_SATURATION_TUNING", "") != "1":
        return

    urdf_path, solver = _build_solver()
    try:
        baseline = _run_round_trip(solver)
        baseline_score = baseline.score()

        gains = [0.2, 0.4, 0.6, 0.8]
        enter_eps = [1e-4, 2e-4]
        exit_eps = [2e-4, 4e-4]
        release_margin = [0.001, 0.002, 0.003]

        best_score = float("inf")
        best_params = None
        for gain in gains:
            solver.set_limit_recovery_gain(gain)
            for e_eps in enter_eps:
                for x_eps in exit_eps:
                    solver.set_limit_recovery_hysteresis(e_eps, x_eps)
                    for margin in release_margin:
                        solver.set_limit_exit_release_margin(margin)
                        score = _run_round_trip(solver).score()
                        if score < best_score:
                            best_score = score
                            best_params = (gain, e_eps, x_eps, margin)

        assert best_params is not None
        assert best_score < baseline_score
    finally:
        os.remove(urdf_path)
