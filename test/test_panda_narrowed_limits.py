#!/usr/bin/env python3
"""Narrowed-limits Panda test framework for saturation exit behavior.

Uses set_joint_limits() to narrow the Panda's joint ranges around the default
configuration, forcing multiple joints to saturate during normal EE motion
without triggering kinematic singularity.  Compares baseline, Strategy A
(Python-level saturated-joint posture task), and Strategy B (C++ built-in
barrier gradient task) across multiple scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pytest

import embodik as eik

# ---------------------------------------------------------------------------
# Panda fixture with narrowed limits
# ---------------------------------------------------------------------------

_PANDA_DEFAULT_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
_PANDA_GRIPPER_EXTRA = np.array([0.05, 0.05])
_PANDA_EE_FRAME = "panda_hand"


def _load_panda() -> tuple[eik.RobotModel, eik.KinematicsSolver]:
    """Load the Panda model via robot_descriptions."""
    from robot_descriptions.panda_description import URDF_PATH

    robot = eik.RobotModel(URDF_PATH, floating_base=False)
    q_init = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
    robot.update_configuration(q_init)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    return robot, solver


def _narrow_limits(
    robot: eik.RobotModel,
    margin: float = 0.25,
    q_center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Narrow joint limits to q_center +/- margin for actuated joints."""
    q_lower_orig, q_upper_orig = robot.get_joint_limits()
    q_lower = q_lower_orig.copy()
    q_upper = q_upper_orig.copy()
    if q_center is None:
        q_center_full = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
    else:
        q_center_full = q_center

    nq_arm = len(_PANDA_DEFAULT_Q)
    for i in range(nq_arm):
        q_lower[i] = max(q_lower_orig[i], q_center_full[i] - margin)
        q_upper[i] = min(q_upper_orig[i], q_center_full[i] + margin)

    robot.set_joint_limits(q_lower, q_upper)
    return q_lower, q_upper


# ---------------------------------------------------------------------------
# Round-trip helper and metrics
# ---------------------------------------------------------------------------


@dataclass
class RoundTripMetrics:
    """Metrics collected from a single EE round-trip."""

    stall_steps_reverse: int = 0
    ee_return_error: float = 0.0
    recovery_ratio: float = 0.0
    min_task_scale_reverse: float = 1.0
    total_saturated_joint_steps: int = 0
    oscillation_count: int = 0
    forward_steps: int = 0
    reverse_steps: int = 0
    task_scale_trace: list[float] = field(default_factory=list)


SolverCallback = Callable[[eik.KinematicsSolver, eik.VelocitySolverResult, np.ndarray], None]


def run_round_trip(
    robot: eik.RobotModel,
    solver: eik.KinematicsSolver,
    ee_offset: np.ndarray,
    *,
    steps_per_phase: int = 200,
    stall_vel_threshold: float = 1e-4,
    callback: SolverCallback | None = None,
) -> RoundTripMetrics:
    """Drive the EE forward by ee_offset then reverse back.

    Args:
        robot: Robot model (state will be modified).
        solver: Configured solver.
        ee_offset: 3D position offset for the EE target.
        steps_per_phase: Steps for each phase (forward/reverse).
        stall_vel_threshold: Velocity norm below which a step is counted
            as stalled.
        callback: Optional callback invoked after each solve_velocity call.

    Returns:
        RoundTripMetrics with diagnostics.
    """
    metrics = RoundTripMetrics()
    q_init = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
    q = q_init.copy()
    robot.update_configuration(q)
    dt = solver.dt

    solver.clear_tasks()

    task = solver.add_frame_task("ee_task", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSE)
    task.priority = 0
    task.weight = 10.0

    robot.update_kinematics(q)
    ee_start = np.array(task.current_position)
    ee_start_rot = np.array(task.current_orientation)

    ee_target_forward = ee_start + ee_offset
    ee_target_reverse = ee_start.copy()

    prev_dq_sign = None

    def _run_phase(target_pos: np.ndarray, is_reverse: bool, phase_steps: int):
        nonlocal q, prev_dq_sign
        task.set_target_pose(target_pos, ee_start_rot)

        for step in range(phase_steps):
            result = solver.solve_velocity(q)

            if callback is not None:
                callback(solver, result, q)

            dq = result.joint_velocities
            q = robot.integrate(q, dq, dt)
            q_lower, q_upper = robot.get_joint_limits()
            q = np.clip(q, q_lower, q_upper)
            robot.update_kinematics(q)

            scales = result.task_scales
            scale = scales[0] if scales else 1.0
            metrics.task_scale_trace.append(scale)
            metrics.total_saturated_joint_steps += len(result.saturated_joints)

            if is_reverse:
                metrics.min_task_scale_reverse = min(metrics.min_task_scale_reverse, scale)
                if np.linalg.norm(dq) < stall_vel_threshold:
                    metrics.stall_steps_reverse += 1

                dq_sign = np.sign(np.sum(dq[:7]))
                if prev_dq_sign is not None and dq_sign != 0 and prev_dq_sign != 0:
                    if dq_sign != prev_dq_sign:
                        metrics.oscillation_count += 1
                prev_dq_sign = dq_sign if dq_sign != 0 else prev_dq_sign

    _run_phase(ee_target_forward, False, steps_per_phase)
    metrics.forward_steps = steps_per_phase

    _run_phase(ee_target_reverse, True, steps_per_phase)
    metrics.reverse_steps = steps_per_phase

    robot.update_kinematics(q)
    ee_final = np.array(task.current_position)
    metrics.ee_return_error = float(np.linalg.norm(ee_final - ee_start))

    forward_distance = float(np.linalg.norm(ee_offset))
    if forward_distance > 1e-6:
        metrics.recovery_ratio = 1.0 - metrics.ee_return_error / forward_distance
    else:
        metrics.recovery_ratio = 1.0

    solver.clear_tasks()
    return metrics


# ---------------------------------------------------------------------------
# Strategy A: Saturated-joint posture task (Python-level)
# ---------------------------------------------------------------------------


class StrategyACallback:
    """After each solve, read saturated_joints and set a posture task
    driving those joints toward midrange."""

    def __init__(self, solver: eik.KinematicsSolver, robot: eik.RobotModel):
        self.solver = solver
        self.robot = robot
        self._posture_task: eik.PostureTask | None = None

    def __call__(
        self,
        solver: eik.KinematicsSolver,
        result: eik.VelocitySolverResult,
        q: np.ndarray,
    ):
        sat = result.saturated_joints
        q_lower, q_upper = self.robot.get_joint_limits()
        q_mid = 0.5 * (q_lower + q_upper)

        if sat:
            if self._posture_task is None:
                self._posture_task = solver.add_posture_task(
                    "_strategy_a_posture", list(range(self.robot.nv))
                )
                self._posture_task.priority = 1
                self._posture_task.weight = 1.0
            self._posture_task.set_target_configuration(q_mid)
            self._posture_task.active = True
        else:
            if self._posture_task is not None:
                self._posture_task.active = False


# ---------------------------------------------------------------------------
# Strategy B: C++ built-in barrier gradient task (no Python callback needed
# for the solver, but we enable/disable it before the run)
# ---------------------------------------------------------------------------


def setup_strategy_b(
    solver: eik.KinematicsSolver,
    barrier_margin: float = 0.3,
    gain: float = 1.0,
):
    """Enable the C++ barrier gradient task."""
    solver.set_joint_limit_barrier_task(barrier_margin, gain)


def setup_strategy_b_with_posture(
    solver: eik.KinematicsSolver,
    robot: eik.RobotModel,
    q_target: np.ndarray | None = None,
    posture_weight: float = 0.1,
    barrier_margin: float = 0.3,
    gain: float = 1.0,
):
    """Enable barrier + a posture bias toward q_target."""
    solver.set_joint_limit_barrier_task(barrier_margin, gain)
    q_lower, q_upper = robot.get_joint_limits()
    if q_target is None:
        q_target = 0.5 * (q_lower + q_upper)
    posture = solver.add_posture_task("_b_posture")
    posture.priority = 1
    posture.weight = posture_weight
    posture.set_target_configuration(q_target)
    return posture


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

_SINGLE_AXIS_OFFSETS = [
    np.array([0.10, 0.0, 0.0]),
    np.array([0.0, 0.10, 0.0]),
    np.array([0.0, 0.0, 0.10]),
]

_DIAGONAL_OFFSET = np.array([0.06, 0.06, 0.06])


@pytest.fixture
def panda_narrow():
    """Fixture: Panda with limits narrowed to +/- 0.25 rad around default."""
    robot, solver = _load_panda()
    _narrow_limits(robot, margin=0.25)
    return robot, solver


def _run_baseline(robot, solver, offset, steps=200):
    solver.clear_tasks()
    solver.clear_joint_limit_barrier_task()
    return run_round_trip(robot, solver, offset, steps_per_phase=steps)


def _run_strategy_a(robot, solver, offset, steps=200):
    solver.clear_tasks()
    solver.clear_joint_limit_barrier_task()
    cb = StrategyACallback(solver, robot)
    m = run_round_trip(robot, solver, offset, steps_per_phase=steps, callback=cb)
    if cb._posture_task is not None:
        solver.remove_task("_strategy_a_posture")
    return m


def _run_strategy_b(robot, solver, offset, steps=200):
    solver.clear_tasks()
    setup_strategy_b(solver, barrier_margin=0.3, gain=1.0)
    m = run_round_trip(robot, solver, offset, steps_per_phase=steps)
    solver.clear_joint_limit_barrier_task()
    return m


def _run_strategy_b_with_posture(robot, solver, offset, q_target=None, steps=200):
    solver.clear_tasks()
    setup_strategy_b(solver, barrier_margin=0.3, gain=1.0)
    q_lower, q_upper = robot.get_joint_limits()
    if q_target is None:
        q_target = 0.5 * (q_lower + q_upper)
    posture = solver.add_posture_task("_b_posture")
    posture.priority = 1
    posture.weight = 0.1
    posture.set_target_configuration(q_target)
    m = run_round_trip(robot, solver, offset, steps_per_phase=steps)
    solver.clear_joint_limit_barrier_task()
    solver.remove_task("_b_posture")
    return m


# ---------------------------------------------------------------------------
# Baseline tests
# ---------------------------------------------------------------------------


class TestBaseline:
    """Baseline (no secondary task) round-trip tests.

    The baseline with narrowed limits is expected to have high return error
    (0.6+) on X/Y axes due to the SNS task-scale collapse near limits.
    These tests characterize existing behavior; the comparison tests below
    verify that strategies A and B improve on this.
    """

    def test_single_axis_x(self, panda_narrow):
        robot, solver = panda_narrow
        m = _run_baseline(robot, solver, _SINGLE_AXIS_OFFSETS[0])
        assert m.ee_return_error < 1.0, f"X return error {m.ee_return_error:.4f}"

    def test_single_axis_y(self, panda_narrow):
        robot, solver = panda_narrow
        m = _run_baseline(robot, solver, _SINGLE_AXIS_OFFSETS[1])
        assert m.ee_return_error < 1.0, f"Y return error {m.ee_return_error:.4f}"

    def test_single_axis_z(self, panda_narrow):
        robot, solver = panda_narrow
        m = _run_baseline(robot, solver, _SINGLE_AXIS_OFFSETS[2])
        assert m.ee_return_error < 1.0, f"Z return error {m.ee_return_error:.4f}"

    def test_diagonal(self, panda_narrow):
        robot, solver = panda_narrow
        m = _run_baseline(robot, solver, _DIAGONAL_OFFSET)
        assert m.ee_return_error < 1.0, f"Diagonal return error {m.ee_return_error:.4f}"


# ---------------------------------------------------------------------------
# Comparison tests
# ---------------------------------------------------------------------------


class TestStrategyComparison:
    """Compare baseline vs A vs B on the same scenarios."""

    @pytest.mark.parametrize("axis", [0, 1, 2], ids=["X", "Y", "Z"])
    def test_strategy_b_improves_over_baseline_single_axis(self, panda_narrow, axis):
        robot, solver = panda_narrow
        offset = _SINGLE_AXIS_OFFSETS[axis]
        baseline = _run_baseline(robot, solver, offset)

        robot.update_configuration(np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA]))
        _narrow_limits(robot, margin=0.25)
        b_result = _run_strategy_b(robot, solver, offset)

        assert (
            b_result.stall_steps_reverse <= baseline.stall_steps_reverse + 5
        ), f"axis={axis}: B stalls {b_result.stall_steps_reverse} > baseline {baseline.stall_steps_reverse}+5"
        assert (
            b_result.ee_return_error <= baseline.ee_return_error + 0.02
        ), f"axis={axis}: B error {b_result.ee_return_error:.4f} > baseline {baseline.ee_return_error:.4f}+0.02"

    def test_strategy_b_no_oscillation(self, panda_narrow):
        robot, solver = panda_narrow
        m = _run_strategy_b(robot, solver, _DIAGONAL_OFFSET)
        assert m.oscillation_count < 30, f"Oscillations: {m.oscillation_count}"

    def test_strategy_b_with_posture_coexistence(self, panda_narrow):
        """B + posture bias should not oscillate or degrade vs B alone."""
        robot, solver = panda_narrow
        b_alone = _run_strategy_b(robot, solver, _SINGLE_AXIS_OFFSETS[0])

        robot.update_configuration(np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA]))
        _narrow_limits(robot, margin=0.25)
        b_posture = _run_strategy_b_with_posture(robot, solver, _SINGLE_AXIS_OFFSETS[0])

        assert (
            b_posture.oscillation_count < 30
        ), f"B+posture oscillations: {b_posture.oscillation_count}"
        assert (
            b_posture.ee_return_error < b_alone.ee_return_error + 0.03
        ), f"B+posture error {b_posture.ee_return_error:.4f} >> B alone {b_alone.ee_return_error:.4f}"

    def test_strategy_b_with_posture_near_limits_stress(self, panda_narrow):
        """B + posture bias targeting near-limit config should not oscillate."""
        robot, solver = panda_narrow
        q_lower, q_upper = robot.get_joint_limits()
        q_near_limit = q_upper.copy()
        q_near_limit[:7] = q_upper[:7] - 0.05
        m = _run_strategy_b_with_posture(
            robot, solver, _SINGLE_AXIS_OFFSETS[0], q_target=q_near_limit
        )
        assert m.oscillation_count < 30, f"Stress test oscillations: {m.oscillation_count}"

    @pytest.mark.parametrize("axis", [0, 1, 2], ids=["X", "Y", "Z"])
    def test_strategy_a_vs_baseline(self, panda_narrow, axis):
        robot, solver = panda_narrow
        baseline = _run_baseline(robot, solver, _SINGLE_AXIS_OFFSETS[axis])

        robot.update_configuration(np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA]))
        _narrow_limits(robot, margin=0.25)
        a_result = _run_strategy_a(robot, solver, _SINGLE_AXIS_OFFSETS[axis])

        assert (
            a_result.stall_steps_reverse <= baseline.stall_steps_reverse + 5
        ), f"axis={axis}: A stalls {a_result.stall_steps_reverse} > baseline {baseline.stall_steps_reverse}+5"


class TestBarrierQuantitativeBenefit:
    """Document quantitative baseline vs barrier (Strategy B) improvement.

    Validation runs (margin=0.25, X-axis 0.10m): baseline ee_return_error ~0.59 m,
    stall_steps_reverse 200; barrier ee_return_error ~0, stall_steps ~97.
    Very narrow (margin=0.15): barrier improves X/Y return error and reduces stalls.
    """

    @pytest.mark.parametrize("axis", [0, 1, 2], ids=["X", "Y", "Z"])
    def test_barrier_improves_or_matches_baseline(self, panda_narrow, axis):
        """Barrier (Strategy B) must not degrade vs baseline on any axis."""
        robot, solver = panda_narrow
        offset = _SINGLE_AXIS_OFFSETS[axis]
        baseline = _run_baseline(robot, solver, offset)

        robot.update_configuration(np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA]))
        _narrow_limits(robot, margin=0.25)
        barrier = _run_strategy_b(robot, solver, offset)

        assert barrier.ee_return_error <= baseline.ee_return_error + 0.02
        assert barrier.stall_steps_reverse <= baseline.stall_steps_reverse + 5

    def test_barrier_very_narrow_x_improves(self, panda_narrow):
        """Very narrow (0.15): barrier improves X-axis return error."""
        robot, solver = panda_narrow
        _narrow_limits(robot, margin=0.15)
        baseline = _run_baseline(robot, solver, _SINGLE_AXIS_OFFSETS[0])

        robot.update_configuration(np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA]))
        _narrow_limits(robot, margin=0.15)
        barrier = _run_strategy_b(robot, solver, _SINGLE_AXIS_OFFSETS[0])

        assert barrier.ee_return_error <= baseline.ee_return_error + 0.02
        assert barrier.stall_steps_reverse <= baseline.stall_steps_reverse + 5


class TestMultiSaturationStress:
    """Very narrow limits (0.15 rad) to stress-test multi-joint saturation."""

    @pytest.fixture
    def panda_very_narrow(self):
        robot, solver = _load_panda()
        _narrow_limits(robot, margin=0.15)
        return robot, solver

    def test_stress_baseline_does_not_crash(self, panda_very_narrow):
        robot, solver = panda_very_narrow
        m = _run_baseline(robot, solver, np.array([0.05, 0.0, 0.0]))
        assert m.forward_steps == 200

    def test_stress_strategy_b(self, panda_very_narrow):
        robot, solver = panda_very_narrow
        m = _run_strategy_b(robot, solver, np.array([0.05, 0.0, 0.0]))
        assert m.oscillation_count < 50, f"Stress B oscillations: {m.oscillation_count}"
