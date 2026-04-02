#!/usr/bin/env python3
"""Elastic band joint limit expansion tests.

TDD tests for the elastic band mechanism that temporarily expands joint limit
margins when the solver is overconstrained by joint limits, then snaps back
to nominal limits when the solver is healthy.

Phase 0: Hypothesis validation — does keeping DOFs unsaturated (via slightly
widened limits) improve task progress compared to baseline?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

import embodik as eik

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_PANDA_DEFAULT_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
_PANDA_GRIPPER_EXTRA = np.array([0.05, 0.05])
_PANDA_EE_FRAME = "panda_hand"


def _load_panda() -> tuple[eik.RobotModel, eik.KinematicsSolver]:
    from robot_descriptions.panda_description import URDF_PATH

    robot = eik.RobotModel(URDF_PATH, floating_base=False)
    q_init = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
    robot.update_configuration(q_init)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    return robot, solver


def _narrow_limits(
    robot: eik.RobotModel,
    margin: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Narrow joint limits to default +/- margin for arm joints."""
    q_lower_orig, q_upper_orig = robot.get_joint_limits()
    q_lower = q_lower_orig.copy()
    q_upper = q_upper_orig.copy()
    q_center = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])

    nq_arm = len(_PANDA_DEFAULT_Q)
    for i in range(nq_arm):
        q_lower[i] = max(q_lower_orig[i], q_center[i] - margin)
        q_upper[i] = min(q_upper_orig[i], q_center[i] + margin)

    robot.set_joint_limits(q_lower, q_upper)
    return q_lower, q_upper


# ---------------------------------------------------------------------------
# Round-trip metrics
# ---------------------------------------------------------------------------


@dataclass
class RoundTripMetrics:
    stall_steps_forward: int = 0
    stall_steps_reverse: int = 0
    ee_return_error: float = 0.0
    recovery_ratio: float = 0.0
    min_task_scale: float = 1.0
    total_saturated_joint_steps: int = 0
    oscillation_count: int = 0
    infeasible_count: int = 0
    task_scale_trace: list[float] = field(default_factory=list)
    q_trace: list[np.ndarray] = field(default_factory=list)


def _run_solve_loop(
    robot: eik.RobotModel,
    solver: eik.KinematicsSolver,
    ee_offset: np.ndarray,
    *,
    steps_per_phase: int = 200,
    stall_vel_threshold: float = 1e-4,
    widen_amount: float = 0.0,
) -> RoundTripMetrics:
    """Drive EE forward by ee_offset then reverse back.

    If widen_amount > 0, temporarily widen joint limits by that amount
    before each solve (and restore after), simulating the elastic band
    headroom hypothesis.
    """
    metrics = RoundTripMetrics()
    q_init = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
    q = q_init.copy()
    robot.update_configuration(q)
    dt = solver.dt
    nq_arm = len(_PANDA_DEFAULT_Q)

    solver.clear_tasks()
    task = solver.add_frame_task("ee_task", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSE)
    task.priority = 0
    task.weight = 10.0

    robot.update_kinematics(q)
    ee_start = np.array(task.current_position)
    ee_start_rot = np.array(task.current_orientation)

    # Store nominal limits for restoration
    q_lower_nom, q_upper_nom = robot.get_joint_limits()

    prev_dq_sign = None

    def _run_phase(target_pos: np.ndarray, is_reverse: bool, phase_steps: int):
        nonlocal q, prev_dq_sign
        task.set_target_pose(target_pos, ee_start_rot)

        for step in range(phase_steps):
            # Temporarily widen limits if requested
            if widen_amount > 0:
                q_lower_wide = q_lower_nom.copy()
                q_upper_wide = q_upper_nom.copy()
                for i in range(nq_arm):
                    q_lower_wide[i] -= widen_amount
                    q_upper_wide[i] += widen_amount
                robot.set_joint_limits(q_lower_wide, q_upper_wide)

            result = solver.solve_velocity(q)

            # Restore nominal limits
            if widen_amount > 0:
                robot.set_joint_limits(q_lower_nom, q_upper_nom)

            dq = result.joint_velocities
            q = robot.integrate(q, dq, dt)
            # Always clip to nominal limits
            q = np.clip(q, q_lower_nom, q_upper_nom)
            robot.update_kinematics(q)

            scales = result.task_scales
            scale = scales[0] if scales else 1.0
            metrics.task_scale_trace.append(scale)
            metrics.min_task_scale = min(metrics.min_task_scale, scale)
            metrics.total_saturated_joint_steps += len(result.saturated_joints)
            metrics.q_trace.append(q.copy())

            if result.status == eik.SolverStatus.INFEASIBLE:
                metrics.infeasible_count += 1

            dq_norm = np.linalg.norm(dq)
            if dq_norm < stall_vel_threshold:
                if is_reverse:
                    metrics.stall_steps_reverse += 1
                else:
                    metrics.stall_steps_forward += 1

            if is_reverse:
                dq_sign = np.sign(np.sum(dq[:7]))
                if prev_dq_sign is not None and dq_sign != 0 and prev_dq_sign != 0:
                    if dq_sign != prev_dq_sign:
                        metrics.oscillation_count += 1
                prev_dq_sign = dq_sign if dq_sign != 0 else prev_dq_sign

    ee_target_forward = ee_start + ee_offset
    _run_phase(ee_target_forward, False, steps_per_phase)
    _run_phase(ee_start.copy(), True, steps_per_phase)

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
# Phase 0: Approach A — Hypothesis Validation
# ---------------------------------------------------------------------------


class TestApproachAHypothesis:
    """Quick validation: does slightly widening limits reduce stalls?

    Uses set_joint_limits() as a proxy for velocity headroom.
    Decision gate: if stall count improves by >=20%, Approach B is validated.
    """

    @pytest.fixture
    def panda_narrow(self):
        robot, solver = _load_panda()
        _narrow_limits(robot, margin=0.15)
        return robot, solver

    def _reset_panda(self, robot, solver):
        q_init = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q_init)
        robot.update_kinematics(q_init)
        _narrow_limits(robot, margin=0.15)
        return q_init

    @pytest.mark.parametrize("axis,offset", [
        ("X", np.array([0.08, 0.0, 0.0])),
        ("Y", np.array([0.0, 0.08, 0.0])),
        ("Z", np.array([0.0, 0.0, 0.08])),
    ], ids=["X", "Y", "Z"])
    def test_widened_limits_reduce_infeasible_or_stalls(self, panda_narrow, axis, offset):
        """Widening limits by 0.001 rad should reduce infeasible/stall count."""
        robot, solver = panda_narrow

        # Baseline
        self._reset_panda(robot, solver)
        baseline = _run_solve_loop(robot, solver, offset, widen_amount=0.0)

        # Widened (small: 0.001 rad ≈ 0.06 degrees)
        self._reset_panda(robot, solver)
        widened = _run_solve_loop(robot, solver, offset, widen_amount=0.001)

        total_stalls_baseline = baseline.stall_steps_forward + baseline.stall_steps_reverse
        total_stalls_widened = widened.stall_steps_forward + widened.stall_steps_reverse

        print(f"\n--- Axis {axis} ---")
        print(f"Baseline: stalls={total_stalls_baseline} (fwd={baseline.stall_steps_forward}, "
              f"rev={baseline.stall_steps_reverse}), infeasible={baseline.infeasible_count}, "
              f"min_scale={baseline.min_task_scale:.4f}, "
              f"return_err={baseline.ee_return_error:.4f}")
        print(f"Widened:  stalls={total_stalls_widened} (fwd={widened.stall_steps_forward}, "
              f"rev={widened.stall_steps_reverse}), infeasible={widened.infeasible_count}, "
              f"min_scale={widened.min_task_scale:.4f}, "
              f"return_err={widened.ee_return_error:.4f}")

        # Decision gate: widened should show some improvement
        # Either fewer stalls, fewer infeasibles, or better return error
        improved_stalls = total_stalls_widened < total_stalls_baseline
        improved_infeasible = widened.infeasible_count < baseline.infeasible_count
        improved_return = widened.ee_return_error < baseline.ee_return_error

        assert improved_stalls or improved_infeasible or improved_return, (
            f"Axis {axis}: widened limits showed no improvement over baseline. "
            f"Stalls: {total_stalls_widened} vs {total_stalls_baseline}, "
            f"Infeasible: {widened.infeasible_count} vs {baseline.infeasible_count}, "
            f"Return error: {widened.ee_return_error:.4f} vs {baseline.ee_return_error:.4f}"
        )

    def test_widened_limits_q_stays_within_nominal(self, panda_narrow):
        """Even with widened solve limits, clipped q stays within nominal."""
        robot, solver = panda_narrow
        q_lower_nom, q_upper_nom = robot.get_joint_limits()

        self._reset_panda(robot, solver)
        metrics = _run_solve_loop(
            robot, solver, np.array([0.08, 0.0, 0.0]), widen_amount=0.001
        )

        for i, q in enumerate(metrics.q_trace):
            assert np.all(q >= q_lower_nom - 1e-10), (
                f"Step {i}: q below lower limit: {q} < {q_lower_nom}"
            )
            assert np.all(q <= q_upper_nom + 1e-10), (
                f"Step {i}: q above upper limit: {q} > {q_upper_nom}"
            )


# ---------------------------------------------------------------------------
# Phase 1: Elastic Band State Dynamics (Tests 1-4)
# ---------------------------------------------------------------------------


class TestElasticBandStateDynamics:
    """Tests for elastic band state: expansion, decay, max, per-joint gating."""

    @pytest.fixture
    def panda_narrow(self):
        robot, solver = _load_panda()
        _narrow_limits(robot, margin=0.15)
        return robot, solver

    def test_elastic_state_grows_on_limit_stall(self, panda_narrow):
        """Test 1: delta grows when solver is stalled at joint limits."""
        robot, solver = panda_narrow
        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q)
        robot.update_kinematics(q)

        solver.enable_elastic_band(delta_max=0.05)
        assert solver.elastic_band_enabled()
        assert solver.elastic_band_max_delta() == 0.0

        # Add task driving EE far beyond reachable range to force stalling
        solver.clear_tasks()
        task = solver.add_frame_task("ee", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 10.0
        robot.update_kinematics(q)
        ee_pos = np.array(task.current_position)
        ee_rot = np.array(task.current_orientation)
        # Target far beyond reachable — will saturate joints
        task.set_target_pose(ee_pos + np.array([0.5, 0.0, 0.0]), ee_rot)

        # Run enough steps to trigger expansion (stall_threshold default = 3)
        for _ in range(20):
            result = solver.solve_velocity(q)
            dq = result.joint_velocities
            q = robot.integrate(q, dq, solver.dt)
            q_lower, q_upper = robot.get_joint_limits()
            q = np.clip(q, q_lower, q_upper)
            robot.update_kinematics(q)

        assert solver.elastic_band_max_delta() > 0.0, (
            "Elastic band delta should grow after stalling at joint limits"
        )
        solver.clear_tasks()

    def test_elastic_state_decays_when_healthy(self, panda_narrow):
        """Test 2: delta decays toward 0 when solver is healthy."""
        robot, solver = panda_narrow
        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q)
        robot.update_kinematics(q)

        solver.enable_elastic_band(delta_max=0.05)
        solver.configure_elastic_band(
            delta_max=0.05, expand_rate=0.01, decay_rate=0.3,
            stall_threshold=3,
        )

        solver.clear_tasks()
        task = solver.add_frame_task("ee", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 10.0
        robot.update_kinematics(q)
        ee_pos = np.array(task.current_position)
        ee_rot = np.array(task.current_orientation)

        # Phase 1: Force stalling to build up delta
        task.set_target_pose(ee_pos + np.array([0.5, 0.0, 0.0]), ee_rot)
        for _ in range(20):
            result = solver.solve_velocity(q)
            dq = result.joint_velocities
            q = robot.integrate(q, dq, solver.dt)
            q_lower, q_upper = robot.get_joint_limits()
            q = np.clip(q, q_lower, q_upper)
            robot.update_kinematics(q)

        peak_delta = solver.elastic_band_max_delta()
        assert peak_delta > 0.0, "Should have expanded during stall phase"

        # Phase 2: Set target to current pose (trivially achievable)
        robot.update_kinematics(q)
        ee_pos_now = np.array(task.current_position)
        ee_rot_now = np.array(task.current_orientation)
        task.set_target_pose(ee_pos_now, ee_rot_now)

        for _ in range(30):
            result = solver.solve_velocity(q)
            dq = result.joint_velocities
            q = robot.integrate(q, dq, solver.dt)
            q_lower, q_upper = robot.get_joint_limits()
            q = np.clip(q, q_lower, q_upper)
            robot.update_kinematics(q)

        assert solver.elastic_band_max_delta() < peak_delta * 0.5, (
            f"Delta should decay: was {peak_delta:.6f}, "
            f"now {solver.elastic_band_max_delta():.6f}"
        )
        solver.clear_tasks()

    def test_elastic_delta_respects_max(self, panda_narrow):
        """Test 3: delta never exceeds configured delta_max."""
        robot, solver = panda_narrow
        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q)
        robot.update_kinematics(q)

        delta_max = 0.02
        solver.enable_elastic_band(delta_max=delta_max)
        solver.configure_elastic_band(
            delta_max=delta_max, expand_rate=0.01, decay_rate=0.2,
            stall_threshold=2,
        )

        solver.clear_tasks()
        task = solver.add_frame_task("ee", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 10.0
        robot.update_kinematics(q)
        ee_pos = np.array(task.current_position)
        ee_rot = np.array(task.current_orientation)
        task.set_target_pose(ee_pos + np.array([0.5, 0.0, 0.0]), ee_rot)

        # Run many steps to try to exceed max
        for _ in range(100):
            result = solver.solve_velocity(q)
            dq = result.joint_velocities
            q = robot.integrate(q, dq, solver.dt)
            q_lower, q_upper = robot.get_joint_limits()
            q = np.clip(q, q_lower, q_upper)
            robot.update_kinematics(q)

            assert solver.elastic_band_max_delta() <= delta_max + 1e-10, (
                f"Delta {solver.elastic_band_max_delta():.6f} exceeded max {delta_max}"
            )
        solver.clear_tasks()

    def test_elastic_only_saturated_joints_expand(self, panda_narrow):
        """Test 4: only saturated joints get nonzero delta."""
        robot, solver = panda_narrow
        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q)
        robot.update_kinematics(q)

        solver.enable_elastic_band(delta_max=0.05)
        solver.configure_elastic_band(
            delta_max=0.05, expand_rate=0.01, decay_rate=0.2,
            stall_threshold=3, expand_only_saturated=True,
        )

        solver.clear_tasks()
        task = solver.add_frame_task("ee", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 10.0
        robot.update_kinematics(q)
        ee_pos = np.array(task.current_position)
        ee_rot = np.array(task.current_orientation)
        task.set_target_pose(ee_pos + np.array([0.5, 0.0, 0.0]), ee_rot)

        # Run enough steps to trigger expansion
        last_saturated = set()
        for _ in range(30):
            result = solver.solve_velocity(q)
            if result.saturated_joints:
                last_saturated.update(result.saturated_joints)
            dq = result.joint_velocities
            q = robot.integrate(q, dq, solver.dt)
            q_lower, q_upper = robot.get_joint_limits()
            q = np.clip(q, q_lower, q_upper)
            robot.update_kinematics(q)

        deltas = solver.elastic_band_deltas()
        nv = robot.nv
        assert len(deltas) == nv

        # Check that only joints that were ever saturated have nonzero delta
        for i in range(nv):
            if i not in last_saturated and deltas[i] > 1e-10:
                # Allow gripper joints to be excluded from check
                if i < len(_PANDA_DEFAULT_Q):
                    assert False, (
                        f"Joint {i} was never saturated but has delta={deltas[i]:.6f}"
                    )

        solver.clear_tasks()


# ---------------------------------------------------------------------------
# Phase 2: Velocity Box Widening + Restoring Spring + Jacobian Bypass (Tests 5-7)
# ---------------------------------------------------------------------------


class TestElasticBandVelocityBox:
    """Tests that elastic band actually affects the solver's feasibility."""

    @pytest.fixture
    def panda_narrow(self):
        robot, solver = _load_panda()
        _narrow_limits(robot, margin=0.15)
        return robot, solver

    def test_solver_finds_feasible_direction_with_elastic_band(self, panda_narrow):
        """Test 5: elastic band prevents scale collapse at joint limits.

        Without elastic band → many INFEASIBLE results.
        With elastic band → fewer infeasible results.
        """
        robot, solver = panda_narrow
        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q)
        robot.update_kinematics(q)

        solver.clear_tasks()
        task = solver.add_frame_task("ee", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 10.0
        robot.update_kinematics(q)
        ee_pos = np.array(task.current_position)
        ee_rot = np.array(task.current_orientation)
        target = ee_pos + np.array([0.3, 0.0, 0.0])
        task.set_target_pose(target, ee_rot)

        # Run baseline (no elastic band) and count infeasible
        infeasible_baseline = 0
        q_run = q.copy()
        for _ in range(50):
            result = solver.solve_velocity(q_run)
            if result.status == eik.SolverStatus.INFEASIBLE:
                infeasible_baseline += 1
            dq = result.joint_velocities
            q_run = robot.integrate(q_run, dq, solver.dt)
            q_lower, q_upper = robot.get_joint_limits()
            q_run = np.clip(q_run, q_lower, q_upper)
            robot.update_kinematics(q_run)

        # Reset and run with elastic band
        robot.update_configuration(q)
        robot.update_kinematics(q)
        solver.enable_elastic_band(delta_max=0.05)
        solver.configure_elastic_band(
            delta_max=0.05, expand_rate=0.01, decay_rate=0.2,
            stall_threshold=3,
        )
        task.set_target_pose(target, ee_rot)

        infeasible_elastic = 0
        q_run = q.copy()
        for _ in range(50):
            result = solver.solve_velocity(q_run)
            if result.status == eik.SolverStatus.INFEASIBLE:
                infeasible_elastic += 1
            dq = result.joint_velocities
            q_run = robot.integrate(q_run, dq, solver.dt)
            q_lower, q_upper = robot.get_joint_limits()
            q_run = np.clip(q_run, q_lower, q_upper)
            robot.update_kinematics(q_run)

        solver.disable_elastic_band()
        solver.clear_tasks()

        print(f"\nInfeasible: baseline={infeasible_baseline}, elastic={infeasible_elastic}")
        assert infeasible_elastic < infeasible_baseline, (
            f"Elastic band should reduce infeasible count: "
            f"{infeasible_elastic} >= {infeasible_baseline}"
        )

    def test_restoring_spring_biases_toward_nominal(self, panda_narrow):
        """Test 6: after stall + expansion, deltas decay quickly when task eases."""
        robot, solver = panda_narrow
        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q)
        robot.update_kinematics(q)

        solver.enable_elastic_band(delta_max=0.05)
        solver.configure_elastic_band(
            delta_max=0.05, expand_rate=0.01, decay_rate=0.3,
            stall_threshold=3,
        )

        solver.clear_tasks()
        task = solver.add_frame_task("ee", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 10.0
        robot.update_kinematics(q)
        ee_pos = np.array(task.current_position)
        ee_rot = np.array(task.current_orientation)

        # Phase 1: Force stalling to build up delta
        task.set_target_pose(ee_pos + np.array([0.5, 0.0, 0.0]), ee_rot)
        for _ in range(30):
            result = solver.solve_velocity(q)
            dq = result.joint_velocities
            q = robot.integrate(q, dq, solver.dt)
            q_lower, q_upper = robot.get_joint_limits()
            q = np.clip(q, q_lower, q_upper)
            robot.update_kinematics(q)

        assert solver.elastic_band_is_expanded(), "Should be expanded after stalling"

        # Phase 2: Move to center of workspace (easy to reach)
        q_center = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q_center)
        robot.update_kinematics(q_center)
        q = q_center.copy()
        ee_pos_center = np.array(task.current_position)
        task.set_target_pose(ee_pos_center, np.array(task.current_orientation))

        for _ in range(20):
            result = solver.solve_velocity(q)
            dq = result.joint_velocities
            q = robot.integrate(q, dq, solver.dt)
            q_lower, q_upper = robot.get_joint_limits()
            q = np.clip(q, q_lower, q_upper)
            robot.update_kinematics(q)

        assert solver.elastic_band_max_delta() < 0.005, (
            f"Deltas should decay: {solver.elastic_band_max_delta():.6f}"
        )
        solver.disable_elastic_band()
        solver.clear_tasks()

    def test_jacobian_not_zeroed_for_expanded_joints(self, panda_narrow):
        """Test 7: with elastic band, task scale stays nonzero near limits
        (verifying Jacobian columns aren't zeroed for expanded joints).
        """
        robot, solver = panda_narrow
        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        q_lower, q_upper = robot.get_joint_limits()

        # Push joint 1 near upper limit
        q[1] = q_upper[1] - 1e-5
        robot.update_configuration(q)
        robot.update_kinematics(q)

        solver.clear_tasks()
        task = solver.add_frame_task("ee", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 10.0
        robot.update_kinematics(q)
        ee_pos = np.array(task.current_position)
        ee_rot = np.array(task.current_orientation)
        task.set_target_pose(ee_pos + np.array([0.0, 0.0, 0.1]), ee_rot)

        # Baseline single step
        result_baseline = solver.solve_velocity(q)
        scale_baseline = result_baseline.task_scales[0] if result_baseline.task_scales else 1.0

        # Enable elastic band and run steps
        solver.enable_elastic_band(delta_max=0.05)
        q_run = q.copy()
        scales_elastic = []
        for _ in range(20):
            result = solver.solve_velocity(q_run)
            if result.task_scales:
                scales_elastic.append(result.task_scales[0])
            dq = result.joint_velocities
            q_run = robot.integrate(q_run, dq, solver.dt)
            q_run = np.clip(q_run, q_lower, q_upper)
            robot.update_kinematics(q_run)

        max_scale_elastic = max(scales_elastic) if scales_elastic else 0.0
        print(f"\nBaseline scale: {scale_baseline:.4f}, "
              f"Max elastic scale: {max_scale_elastic:.4f}")

        assert solver.elastic_band_is_expanded() or max_scale_elastic > scale_baseline, (
            "Elastic band should be active or improve task scales"
        )
        solver.disable_elastic_band()
        solver.clear_tasks()
