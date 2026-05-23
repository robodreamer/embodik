#!/usr/bin/env python3
"""Regression tests for constrained weighted-advisor diagnostics."""

from __future__ import annotations

import math

import numpy as np
import pytest

import embodik as eik

pytest.importorskip("robot_descriptions")

_PANDA_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.05, 0.05])
_PANDA_NEAR_SINGULAR_Q = np.array([0.0, -0.785, 0.0, -0.05, 0.0, 1.571, 0.785, 0.05, 0.05])
_PANDA_EE_FRAME = "panda_hand"
_PANDA_JOINT1 = "panda_joint1"
_PANDA_JOINT2_VEL_INDEX = 1


def _make_split_panda(offset: np.ndarray, q_seed: np.ndarray = _PANDA_Q):
    from robot_descriptions.panda_description import URDF_PATH

    robot = eik.RobotModel(URDF_PATH, floating_base=False)
    robot.update_configuration(q_seed)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01

    pos_task = solver.add_frame_task("ee_pos", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSITION)
    pos_task.priority = 0
    pos_task.weight = 10.0

    ori_task = solver.add_frame_task("ee_ori", _PANDA_EE_FRAME, eik.TaskType.FRAME_ORIENTATION)
    ori_task.priority = 1
    ori_task.weight = 1.0

    pose = robot.get_frame_pose(_PANDA_EE_FRAME)
    target = np.eye(4, dtype=float)
    target[:3, :3] = np.asarray(pose.rotation, dtype=float)
    target[:3, 3] = np.asarray(pose.translation, dtype=float) + offset
    pos_task.set_target_position(target[:3, 3])
    ori_task.set_target_orientation(target[:3, :3])

    return robot, solver, np.asarray(q_seed, dtype=float).copy(), target


def _make_panda_limit_conflict_solver(*, weighted_fallback: bool | None):
    from robot_descriptions.panda_description import URDF_PATH

    robot = eik.RobotModel(URDF_PATH, floating_base=False)
    lower, upper = robot.get_joint_limits()
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    q = np.clip(_PANDA_Q, lower + 0.02, upper - 0.02)
    q[0] = upper[0]
    robot.update_configuration(q)

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.0)

    blocked = solver.add_joint_task("blocked_joint1", _PANDA_JOINT1, target_value=upper[0] + 1.0)
    blocked.priority = 0
    blocked.weight = 1.0
    blocked.allow_min_error_fallback = False

    useful = solver.add_posture_task("useful_joint2", [_PANDA_JOINT2_VEL_INDEX])
    useful.priority = 1
    useful.weight = 1.0
    useful.allow_min_error_fallback = False
    useful.set_controlled_joint_targets(np.array([q[_PANDA_JOINT2_VEL_INDEX] + 0.3]))

    if weighted_fallback is not None:
        cfg = eik.SolverRuntimeConfig()
        cfg.weighted_fallback_enabled = bool(weighted_fallback)
        solver.configure_runtime(cfg)

    return solver, q, lower, upper


def _fixed_dls_inverse(jacobian: np.ndarray, damping: float) -> np.ndarray:
    gram = jacobian @ jacobian.T
    return jacobian.T @ np.linalg.inv(gram + damping * np.eye(gram.shape[0]))


def _srinv_reference(jacobian: np.ndarray, *, epsilon: float, damping: float) -> np.ndarray:
    gram = jacobian @ jacobian.T
    threshold_squared = epsilon**2
    det_value = float(np.linalg.det(gram))
    global_regularization = (
        (1.0 - (det_value / threshold_squared) ** 2) * threshold_squared
        if det_value < threshold_squared
        else 0.0
    )
    regularized = gram.copy()
    regularized[np.diag_indices_from(regularized)] += global_regularization

    u, sigma_values, _vh = np.linalg.svd(jacobian, full_matrices=True)
    per_sv_damping = damping * (1.0 - (sigma_values / epsilon).clip(max=1.0) ** 2)
    per_sv_damping = np.maximum(per_sv_damping, 0.0)
    if np.any(per_sv_damping > 0.0):
        regularized += u @ np.diag(per_sv_damping) @ u.T
    return jacobian.T @ np.linalg.inv(regularized)


def _weighted_srinv_solution(
    goals: list[np.ndarray],
    jacobians: list[np.ndarray],
    weights: list[float] | None = None,
    *,
    epsilon: float = 1e-6,
    damping: float = 0.0,
) -> np.ndarray:
    if weights is None:
        weights = [1.0] * len(goals)
    rows = []
    target_rows = []
    for goal, jacobian, weight in zip(goals, jacobians, weights):
        sqrt_w = math.sqrt(max(float(weight), 0.0))
        rows.append(sqrt_w * np.asarray(jacobian, dtype=float))
        target_rows.append(sqrt_w * np.asarray(goal, dtype=float))
    stacked_jacobian = np.vstack(rows)
    stacked_goal = np.concatenate(target_rows)
    return _srinv_reference(stacked_jacobian, epsilon=epsilon, damping=damping) @ stacked_goal


def _prioritized_solution(
    goals: list[np.ndarray],
    jacobians: list[np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        goals,
        jacobians,
        np.eye(lower.size),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
        sr_tolerance=1e-6,
        sr_damping=0.0,
    )
    assert result.status == eik.SolverStatus.SUCCESS
    return np.asarray(result.solution, dtype=float)


def test_srinv_kernel_preserves_well_conditioned_directions_better_than_fixed_dls():
    epsilon = 1e-6
    damping = 1e-1
    jacobian = np.diag([1.0, 0.3, 1e-9])
    target = np.array([1.0, 1.0, 1.0], dtype=float)

    fixed_dls = _fixed_dls_inverse(jacobian, damping) @ target
    srinv = _srinv_reference(jacobian, epsilon=epsilon, damping=damping) @ target

    assert abs(srinv[0] - 1.0) < abs(fixed_dls[0] - 1.0)
    assert abs(srinv[1] - (1.0 / 0.3)) < abs(fixed_dls[1] - (1.0 / 0.3))
    assert np.isfinite(srinv[2])
    assert abs(srinv[2]) < 1e-6


def test_weighted_fallback_enabled_by_default_but_lazy_on_success():
    cfg = eik.SolverRuntimeConfig()
    assert cfg.weighted_advisor_enabled is False
    assert cfg.weighted_fallback_enabled is True

    _robot, solver, q, _target = _make_split_panda(np.array([0.04, 0.0, 0.0]))
    result = solver.solve_velocity(q, apply_limits=True)
    assert result.weighted_advisory_available is False
    assert result.weighted_fallback_used is False
    assert result.recovery_stage == eik.SolverRecoveryStage.PRIORITIZED
    assert math.isnan(result.weighted_advisory_v_norm)


def test_default_weighted_fallback_accepts_useful_panda_limit_solution():
    solver, q, lower, upper = _make_panda_limit_conflict_solver(weighted_fallback=None)
    weighted = solver.solve_velocity(q, apply_limits=True)
    weighted_solution = np.asarray(weighted.solution, dtype=float)
    q_next = q + solver.dt * weighted_solution

    assert weighted.status == eik.SolverStatus.SUCCESS
    assert weighted.weighted_advisory_available is True
    assert weighted.weighted_fallback_used is True
    assert weighted.active_task_layout == eik.TaskLayout.SPLIT
    assert weighted.recovery_stage == eik.SolverRecoveryStage.WEIGHTED_FALLBACK
    assert abs(weighted_solution[0]) <= 1e-9
    assert weighted_solution[_PANDA_JOINT2_VEL_INDEX] > 0.25
    assert np.all(q_next <= upper + 1e-9)
    assert np.all(q_next >= lower - 1e-9)


def test_weighted_fallback_accepts_useful_limit_solution_with_active_com_constraint():
    solver, q, lower, upper = _make_panda_limit_conflict_solver(weighted_fallback=True)
    robot = solver.robot
    robot.update_configuration(q)
    com_xy = np.asarray(robot.get_com_position(), dtype=float)[:2]
    support_polygon = np.array(
        [
            com_xy + [-0.5, -0.5],
            com_xy + [0.5, -0.5],
            com_xy + [0.5, 0.5],
            com_xy + [-0.5, 0.5],
        ],
        dtype=float,
    )
    solver.configure_com_constraint(support_polygon, margin=0.0)

    weighted = solver.solve_velocity(q, apply_limits=True)
    weighted_solution = np.asarray(weighted.solution, dtype=float)
    q_next = q + solver.dt * weighted_solution
    robot.update_configuration(q_next)
    com_next = np.asarray(robot.get_com_position(), dtype=float)[:2]

    assert weighted.status == eik.SolverStatus.SUCCESS
    assert weighted.weighted_fallback_used is True
    assert abs(weighted_solution[0]) <= 1e-9
    assert weighted_solution[_PANDA_JOINT2_VEL_INDEX] > 0.25
    assert np.all(q_next <= upper + 1e-9)
    assert np.all(q_next >= lower - 1e-9)
    assert np.all(com_next >= support_polygon.min(axis=0) - 1e-9)
    assert np.all(com_next <= support_polygon.max(axis=0) + 1e-9)


def _axis_aligned_polygon_violation(point_xy: np.ndarray, polygon: np.ndarray) -> float:
    lower = polygon.min(axis=0)
    upper = polygon.max(axis=0)
    return float(
        max(
            lower[0] - point_xy[0],
            point_xy[0] - upper[0],
            lower[1] - point_xy[1],
            point_xy[1] - upper[1],
            0.0,
        )
    )


def test_weighted_fallback_enabled_with_violated_com_constraint_does_not_worsen_violation():
    solver, q, _lower, _upper = _make_panda_limit_conflict_solver(weighted_fallback=True)
    robot = solver.robot
    robot.update_configuration(q)
    com_xy = np.asarray(robot.get_com_position(), dtype=float)[:2]
    shifted_polygon = np.array(
        [
            [com_xy[0] + 0.001, com_xy[1] - 0.50],
            [com_xy[0] + 0.50, com_xy[1] - 0.50],
            [com_xy[0] + 0.50, com_xy[1] + 0.50],
            [com_xy[0] + 0.001, com_xy[1] + 0.50],
        ],
        dtype=float,
    )
    solver.configure_com_constraint(shifted_polygon, margin=0.0)

    current_violation = _axis_aligned_polygon_violation(com_xy, shifted_polygon)
    result = solver.solve_velocity(q, apply_limits=True)
    solution = np.asarray(result.solution, dtype=float)
    q_next = q + solver.dt * solution
    robot.update_configuration(q_next)
    next_com_xy = np.asarray(robot.get_com_position(), dtype=float)[:2]
    next_violation = _axis_aligned_polygon_violation(next_com_xy, shifted_polygon)

    assert result.status == eik.SolverStatus.SUCCESS
    assert next_violation < current_violation


def test_position_step_weighted_fallback_validates_adaptive_dt_against_com_constraint():
    solver, q, _lower, _upper = _make_panda_limit_conflict_solver(weighted_fallback=True)
    robot = solver.robot

    probe = solver.solve_velocity(q, apply_limits=True)
    assert probe.weighted_fallback_used is True
    dq = np.asarray(probe.solution, dtype=float)

    robot.update_configuration(q)
    com0 = np.asarray(robot.get_com_position(), dtype=float)[:2]
    robot.update_configuration(robot.integrate(q, dq, solver.dt))
    com_base = np.asarray(robot.get_com_position(), dtype=float)[:2]
    adaptive_scale = 10.0
    robot.update_configuration(robot.integrate(q, dq, solver.dt * adaptive_scale))
    com_scaled = np.asarray(robot.get_com_position(), dtype=float)[:2]

    axis = int(np.argmax(np.abs(com_scaled - com0)))
    if abs(com_scaled[axis] - com0[axis]) < 1e-9:
        pytest.skip("weighted fallback candidate does not move CoM enough for adaptive-dt gate")

    lower = com0 - 1.0
    upper = com0 + 1.0
    if com_scaled[axis] > com0[axis]:
        upper[axis] = 0.5 * (com_base[axis] + com_scaled[axis])
    else:
        lower[axis] = 0.5 * (com_base[axis] + com_scaled[axis])
    support_polygon = np.array(
        [
            [lower[0], lower[1]],
            [upper[0], lower[1]],
            [upper[0], upper[1]],
            [lower[0], upper[1]],
        ],
        dtype=float,
    )
    assert _axis_aligned_polygon_violation(com_base, support_polygon) <= 1e-10
    assert _axis_aligned_polygon_violation(com_scaled, support_polygon) > 1e-10
    solver.configure_com_constraint(support_polygon, margin=0.0)

    ee_task = solver.add_frame_task("ee_probe", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSITION)
    ee_task.priority = 2
    ee_task.weight = 0.0
    pose = robot.get_frame_pose(_PANDA_EE_FRAME)
    target = np.eye(4, dtype=float)
    target[:3, :3] = np.asarray(pose.rotation, dtype=float)
    target[:3, 3] = np.asarray(pose.translation, dtype=float) + np.array([0.5, 0.0, 0.0])

    opts = eik.PositionStepOptions()
    opts.max_steps = 1
    opts.adaptive_dt = True
    opts.adaptive_dt_max_scale = adaptive_scale
    opts.adaptive_dt_reference_distance = 0.01

    result = solver.solve_position_step(q, [eik.TaskTarget("ee_probe", target)], opts)
    q_next = np.asarray(result.q_solution, dtype=float)
    robot.update_configuration(q_next)
    com_next = np.asarray(robot.get_com_position(), dtype=float)[:2]

    assert _axis_aligned_polygon_violation(com_next, support_polygon) <= 1e-10


def test_weighted_advisor_enabled_does_not_change_prioritized_solution():
    offset = np.array([0.04, 0.02, 0.0])
    _r1, solver1, q1, _target1 = _make_split_panda(offset)
    baseline = solver1.solve_velocity(q1, apply_limits=True)

    _r2, solver2, q2, _target2 = _make_split_panda(offset)
    cfg = eik.SolverRuntimeConfig()
    cfg.damping = 0.1
    cfg.weighted_advisor_enabled = True
    solver2.configure_runtime(cfg)
    advised = solver2.solve_velocity(q2, apply_limits=True)

    assert advised.weighted_advisory_available is True
    assert baseline.status == advised.status
    np.testing.assert_allclose(
        np.asarray(advised.solution, dtype=float),
        np.asarray(baseline.solution, dtype=float),
        atol=1e-9,
        rtol=1e-9,
    )


def test_weighted_advisor_weight_scale_does_not_change_prioritized_solution():
    offset = np.array([0.04, 0.02, 0.0])
    _r1, solver1, q1, _target1 = _make_split_panda(offset)
    baseline = solver1.solve_velocity(q1, apply_limits=True)

    _r2, solver2, q2, _target2 = _make_split_panda(offset)
    cfg = eik.SolverRuntimeConfig()
    cfg.damping = 0.1
    cfg.weighted_advisor_enabled = True
    cfg.advisor_position_weight_scale = 1e6
    solver2.configure_runtime(cfg)
    advised = solver2.solve_velocity(q2, apply_limits=True)

    assert advised.weighted_advisory_available is True
    assert baseline.status == advised.status
    np.testing.assert_allclose(
        np.asarray(advised.solution, dtype=float),
        np.asarray(baseline.solution, dtype=float),
        atol=1e-9,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(advised.task_scales, dtype=float),
        np.asarray(baseline.task_scales, dtype=float),
        atol=1e-9,
        rtol=1e-9,
    )


def test_unconstrained_weighted_srinv_is_not_a_constraint_safe_fallback():
    jacobian = np.array([[1.0]], dtype=float)
    target = np.array([2.0], dtype=float)
    lower_bound = -1.0
    upper_bound = 1.0

    advisory_velocity = _srinv_reference(jacobian, epsilon=1e-6, damping=0.1) @ target

    assert advisory_velocity[0] > upper_bound
    assert not (lower_bound <= advisory_velocity[0] <= upper_bound)


def test_weighted_and_prioritized_agree_on_unconstrained_compatible_tasks():
    goals = [np.array([0.25]), np.array([-0.4])]
    jacobians = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])]
    lower = np.array([-10.0, -10.0])
    upper = np.array([10.0, 10.0])

    weighted = _weighted_srinv_solution(goals, jacobians)
    prioritized = _prioritized_solution(goals, jacobians, lower, upper)

    np.testing.assert_allclose(weighted, prioritized, atol=1e-12, rtol=1e-12)


def test_weighted_solution_trades_conflicting_priorities():
    goals = [np.array([1.0]), np.array([-1.0])]
    jacobians = [np.array([[1.0]]), np.array([[1.0]])]
    lower = np.array([-10.0])
    upper = np.array([10.0])

    weighted = _weighted_srinv_solution(goals, jacobians)
    prioritized = _prioritized_solution(goals, jacobians, lower, upper)

    assert prioritized[0] > 0.9
    assert abs(weighted[0]) < 1e-12


def test_weighted_solution_can_violate_bounds_that_prioritized_solver_respects():
    goals = [np.array([2.0])]
    jacobians = [np.array([[1.0]])]
    lower = np.array([-1.0])
    upper = np.array([1.0])

    weighted = _weighted_srinv_solution(goals, jacobians)
    prioritized = _prioritized_solution(goals, jacobians, lower, upper)

    assert weighted[0] > upper[0]
    assert lower[0] <= prioritized[0] <= upper[0]
    assert prioritized[0] == pytest.approx(upper[0])


def test_weighted_solution_ignores_coupled_linear_constraint():
    goals = [np.array([1.0]), np.array([1.0])]
    jacobians = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])]
    weighted = _weighted_srinv_solution(goals, jacobians)

    C = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    lower = np.array([-10.0, -10.0, -10.0])
    upper = np.array([10.0, 10.0, 1.0])
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        goals,
        jacobians,
        C,
        lower,
        upper,
        sr_tolerance=1e-6,
        sr_damping=0.0,
    )
    prioritized = np.asarray(result.solution, dtype=float)

    assert result.status == eik.SolverStatus.SUCCESS
    assert weighted.sum() > upper[2]
    assert prioritized.sum() <= upper[2] + 1e-9


def test_constrained_weighted_solution_respects_coupled_linear_constraint():
    goals = [np.array([1.0]), np.array([1.0])]
    jacobians = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])]
    C = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    lower = np.array([-10.0, -10.0, -10.0])
    upper = np.array([10.0, 10.0, 1.0])

    result = eik.computeConstrainedWeightedVelocitySolutionEigen(
        goals,
        jacobians,
        C,
        lower,
        upper,
        sr_tolerance=1e-6,
        sr_damping=0.0,
    )
    solution = np.asarray(result.solution, dtype=float)

    assert result.status == eik.SolverStatus.SUCCESS
    assert solution.sum() <= upper[2] + 1e-9
    np.testing.assert_allclose(solution, np.array([0.5, 0.5]), atol=1e-9)


def test_constrained_weighted_solution_respects_scalar_bound():
    result = eik.computeConstrainedWeightedVelocitySolutionEigen(
        [np.array([2.0])],
        [np.array([[1.0]])],
        np.eye(1),
        np.array([-1.0]),
        np.array([1.0]),
        sr_tolerance=1e-6,
        sr_damping=0.0,
    )
    solution = np.asarray(result.solution, dtype=float)

    assert result.status == eik.SolverStatus.SUCCESS
    np.testing.assert_allclose(solution, np.array([1.0]), atol=1e-9)
    assert result.task_modes_effective == [eik.TaskSolveMode.MIN_ERROR]


def test_constrained_weighted_fallback_accepts_useful_panda_limit_solution():
    _solver_base, q_base, _lower, _upper = _make_panda_limit_conflict_solver(
        weighted_fallback=False
    )
    prioritized = _solver_base.solve_velocity(q_base, apply_limits=True)
    prioritized_solution = np.asarray(prioritized.solution, dtype=float)

    solver_fallback, q, lower, upper = _make_panda_limit_conflict_solver(weighted_fallback=True)
    weighted = solver_fallback.solve_velocity(q, apply_limits=True)
    weighted_solution = np.asarray(weighted.solution, dtype=float)
    q_next = q + solver_fallback.dt * weighted_solution

    assert prioritized.status == eik.SolverStatus.INFEASIBLE
    assert prioritized.weighted_fallback_used is False
    assert prioritized_solution[_PANDA_JOINT2_VEL_INDEX] > 0.25

    assert weighted.status == eik.SolverStatus.SUCCESS
    assert weighted.weighted_advisory_available is True
    assert weighted.weighted_fallback_used is True
    assert weighted.active_task_layout == eik.TaskLayout.SPLIT
    assert weighted.recovery_stage == eik.SolverRecoveryStage.WEIGHTED_FALLBACK
    assert abs(weighted_solution[0]) <= 1e-9
    assert weighted_solution[_PANDA_JOINT2_VEL_INDEX] > 0.25
    assert np.all(q_next <= upper + 1e-9)
    assert np.all(q_next >= lower - 1e-9)


def test_weighted_advisor_diagnostics_are_finite():
    _robot, solver, q, _target = _make_split_panda(np.array([0.04, 0.0, 0.0]))
    cfg = eik.SolverRuntimeConfig()
    cfg.weighted_advisor_enabled = True
    solver.configure_runtime(cfg)

    result = solver.solve_velocity(q, apply_limits=True)
    assert result.weighted_advisory_available is True
    assert math.isfinite(result.weighted_advisory_v_norm)
    assert result.weighted_advisory_v_norm >= 0.0
    assert math.isfinite(result.weighted_advisory_pos_task_error_norm)
    assert math.isfinite(result.weighted_advisory_ori_task_error_norm)
    assert math.isfinite(result.weighted_advisory_condition_number)
    assert result.weighted_advisory_condition_number >= 1.0


def test_weighted_advisor_propagates_to_position_step_diagnostics():
    _robot, solver, q, target = _make_split_panda(np.array([0.04, 0.0, 0.0]))
    cfg = eik.SolverRuntimeConfig()
    cfg.damping = 0.1
    cfg.weighted_advisor_enabled = True
    solver.configure_runtime(cfg)

    opts = eik.PositionStepOptions()
    opts.max_steps = 1
    result = solver.solve_position_step(
        q,
        [
            eik.TaskTarget("ee_pos", target, 1.0, 0.0),
            eik.TaskTarget("ee_ori", target, 0.0, 1.0),
        ],
        opts,
    )

    diag = result.diagnostics
    assert diag.weighted_advisory_available is True
    assert math.isfinite(diag.weighted_advisory_v_norm)
    assert math.isfinite(diag.weighted_advisory_pos_task_error_norm)
    assert math.isfinite(diag.weighted_advisory_ori_task_error_norm)


def test_weighted_advisor_uses_srinv_kernel_near_singularity():
    _robot, solver, q, _target = _make_split_panda(
        np.array([0.0, 0.0, 0.08]), _PANDA_NEAR_SINGULAR_Q
    )
    cfg = eik.SolverRuntimeConfig()
    cfg.weighted_advisor_enabled = True
    solver.configure_runtime(cfg)

    result = solver.solve_velocity(q, apply_limits=True)

    assert result.weighted_advisory_available is True
    assert math.isfinite(result.weighted_advisory_v_norm)
    assert result.weighted_advisory_v_norm < 1e3
    assert result.weighted_advisory_condition_number > 10.0
