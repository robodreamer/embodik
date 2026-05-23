#!/usr/bin/env python3
"""Randomized constraint-invariant checks for accepted velocity solves."""

from __future__ import annotations

import os

import numpy as np
import pytest
from test_contact_root_projection import _create_floating_contact_urdf, _rx
from test_ects import create_dual_arm_urdf
from test_embodik import _create_three_link_collision_urdf
from test_relative_constraint import get_relative_position

import embodik as eik

_PANDA_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.05, 0.05])
_PANDA_JOINT1 = "panda_joint1"
_PANDA_JOINT2_VEL_INDEX = 1


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


def _finite_delta(
    q: np.ndarray, lower: np.ndarray, upper: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    delta = np.zeros_like(q, dtype=float)
    finite = np.isfinite(lower) & np.isfinite(upper)
    delta[finite] = rng.uniform(-0.015, 0.015, size=int(np.count_nonzero(finite)))
    return delta


def _configure_weighted_runtime(solver: eik.KinematicsSolver) -> None:
    runtime = eik.SolverRuntimeConfig()
    runtime.weighted_fallback_enabled = True
    solver.configure_runtime(runtime)


def test_randomized_constrained_weighted_solver_satisfies_linear_rows() -> None:
    rng = np.random.default_rng(20260522)
    successes = 0

    for seed_index in range(96):
        n_dof = int(rng.integers(2, 8))
        n_rows = int(rng.integers(n_dof, n_dof + 6))
        n_tasks = int(rng.integers(1, 4))
        v_reference = rng.uniform(-0.4, 0.4, size=n_dof)
        C = rng.normal(size=(n_rows, n_dof))
        row_values = C @ v_reference
        slack = rng.uniform(0.01, 0.5, size=n_rows)
        lower = row_values - slack
        upper = row_values + slack
        goals = []
        jacobians = []
        weights = []
        for _ in range(n_tasks):
            task_dim = int(rng.integers(1, min(4, n_dof) + 1))
            J = rng.normal(size=(task_dim, n_dof))
            jacobians.append(J)
            goals.append(J @ v_reference + rng.normal(scale=0.2, size=task_dim))
            weights.append(float(rng.uniform(0.1, 5.0)))

        result = eik.computeConstrainedWeightedVelocitySolutionEigen(
            goals,
            jacobians,
            C,
            lower,
            upper,
            objective_weights=weights,
            sr_tolerance=1e-6,
            sr_damping=0.0,
        )
        if result.status != eik.SolverStatus.SUCCESS:
            continue
        successes += 1
        solution = np.asarray(result.solution, dtype=float)
        assert np.all(np.isfinite(solution)), seed_index
        values = C @ solution
        assert np.all(values >= lower - 1e-7), seed_index
        assert np.all(values <= upper + 1e-7), seed_index
    assert successes >= 80


def _assert_joint_step_inside_limits(
    q_next: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> None:
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    assert np.all(q_next[finite_lower] >= lower[finite_lower] - 1e-8)
    assert np.all(q_next[finite_upper] <= upper[finite_upper] + 1e-8)


def test_randomized_weighted_fallback_preserves_joint_limits_and_active_com() -> None:
    pytest.importorskip("robot_descriptions")
    from robot_descriptions.panda_description import URDF_PATH

    rng = np.random.default_rng(20260523)

    for seed_index in range(20):
        robot = eik.RobotModel(URDF_PATH, floating_base=False)
        lower, upper = robot.get_joint_limits()
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        q = np.clip(
            _PANDA_Q + _finite_delta(_PANDA_Q, lower, upper, rng), lower + 0.02, upper - 0.02
        )
        q[0] = upper[0]
        robot.update_configuration(q)

        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01
        solver.enable_position_limits(True)
        solver.enable_velocity_limits(True)
        _configure_weighted_runtime(solver)

        blocked = solver.add_joint_task(
            f"blocked_joint1_{seed_index}", _PANDA_JOINT1, target_value=upper[0] + 1.0
        )
        blocked.priority = 0
        blocked.weight = 1.0
        blocked.allow_min_error_fallback = False

        useful = solver.add_posture_task(f"useful_joint2_{seed_index}", [_PANDA_JOINT2_VEL_INDEX])
        useful.priority = 1
        useful.weight = 1.0
        useful.allow_min_error_fallback = False
        useful.set_controlled_joint_targets(
            np.array([q[_PANDA_JOINT2_VEL_INDEX] + rng.uniform(0.1, 0.35)])
        )

        com_xy = np.asarray(robot.get_com_position(), dtype=float)[:2]
        half_width = rng.uniform(0.25, 0.6)
        support_polygon = np.array(
            [
                com_xy + [-half_width, -half_width],
                com_xy + [half_width, -half_width],
                com_xy + [half_width, half_width],
                com_xy + [-half_width, half_width],
            ],
            dtype=float,
        )
        solver.configure_com_constraint(support_polygon, margin=0.0)

        result = solver.solve_velocity(q, apply_limits=True)
        assert result.status == eik.SolverStatus.SUCCESS
        solution = np.asarray(result.solution, dtype=float)
        q_next = robot.integrate(q, solution, solver.dt)
        _assert_joint_step_inside_limits(q_next, lower, upper)

        robot.update_configuration(q_next)
        next_com_xy = np.asarray(robot.get_com_position(), dtype=float)[:2]
        assert _axis_aligned_polygon_violation(next_com_xy, support_polygon) <= 1e-8


def test_randomized_relative_pose_recovery_does_not_worsen_violation() -> None:
    rng = np.random.default_rng(20260524)
    path = create_dual_arm_urdf()
    try:
        for seed_index in range(24):
            robot = eik.RobotModel(path)
            lower, upper = robot.get_joint_limits()
            lower = np.asarray(lower, dtype=float)
            upper = np.asarray(upper, dtype=float)
            q_base = np.array([0.3, 0.5, 0.1, -0.3, 0.5, -0.1], dtype=float)
            q = np.clip(
                q_base + rng.uniform(-0.025, 0.025, size=q_base.shape), lower + 0.02, upper - 0.02
            )
            q[0] = upper[0]
            robot.update_configuration(q)

            solver = eik.KinematicsSolver(robot)
            solver.dt = 0.01
            solver.enable_position_limits(True)
            solver.enable_velocity_limits(True)
            _configure_weighted_runtime(solver)

            blocked = solver.add_joint_task(
                f"blocked_left_j1_{seed_index}", "left_j1", target_value=upper[0] + 1.0
            )
            blocked.priority = 0
            blocked.weight = 1.0
            blocked.allow_min_error_fallback = False

            useful = solver.add_posture_task(f"useful_left_j2_{seed_index}", [1])
            useful.priority = 1
            useful.weight = 1.0
            useful.allow_min_error_fallback = False
            useful.set_controlled_joint_targets(np.array([q[1] + rng.uniform(0.15, 0.35)]))

            initial_rel = get_relative_position(robot, "left_ee", "right_ee")
            rel_lower = np.array(
                [
                    initial_rel[0] - 1.0,
                    initial_rel[1] - 1.0,
                    initial_rel[2] - 1.0,
                    -10.0,
                    -10.0,
                    -10.0,
                ]
            )
            rel_upper = np.array(
                [
                    initial_rel[0] + 1.0,
                    initial_rel[1] + 1.0,
                    initial_rel[2] - 0.005,
                    10.0,
                    10.0,
                    10.0,
                ]
            )
            mask = np.array([1, 1, 1, 0, 0, 0], dtype=np.float64)
            solver.configure_relative_pose_constraint(
                "left_ee", "right_ee", rel_lower, rel_upper, mask
            )

            current_violation = _relative_position_violation(initial_rel, rel_lower, rel_upper)
            result = solver.solve_velocity(q, apply_limits=True)
            assert result.status == eik.SolverStatus.SUCCESS
            solution = np.asarray(result.solution, dtype=float)
            q_next = robot.integrate(q, solution, solver.dt)
            _assert_joint_step_inside_limits(q_next, lower, upper)

            robot.update_configuration(q_next)
            next_rel = get_relative_position(robot, "left_ee", "right_ee")
            next_violation = _relative_position_violation(next_rel, rel_lower, rel_upper)
            assert next_violation <= current_violation + 1e-9
    finally:
        os.unlink(path)


def _relative_position_violation(
    rel_pos: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    return float(np.max(np.maximum(np.maximum(lower[:3] - rel_pos, rel_pos - upper[:3]), 0.0)))


def test_randomized_collision_fallback_does_not_reduce_clearance(tmp_path) -> None:
    rng = np.random.default_rng(20260525)
    urdf_path = _create_three_link_collision_urdf(tmp_path)
    successes = 0

    for seed_index in range(30):
        robot = eik.RobotModel(str(urdf_path), floating_base=False)
        lower, upper = robot.get_joint_limits()
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        q = np.zeros(robot.nq, dtype=float)
        q[0] = upper[0]
        q[1] = rng.uniform(lower[1] + 0.1, upper[1] - 0.1)
        robot.update_configuration(q)

        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01
        solver.enable_position_limits(True)
        solver.enable_velocity_limits(True)
        _configure_weighted_runtime(solver)

        blocked = solver.add_joint_task(
            f"blocked_joint1_{seed_index}", "joint1", target_value=upper[0] + 1.0
        )
        blocked.priority = 0
        blocked.weight = 1.0
        blocked.allow_min_error_fallback = False

        useful = solver.add_posture_task(f"useful_joint2_{seed_index}", [1])
        useful.priority = 1
        useful.weight = 1.0
        useful.allow_min_error_fallback = False
        useful.set_controlled_joint_targets(np.array([q[1] + rng.uniform(0.1, 0.35)]))

        current_distance = solver.evaluate_min_collision_distance(q)
        assert current_distance is not None
        solver.configure_collision_constraint(
            min_distance=float(current_distance) + rng.uniform(1e-5, 1e-4),
            max_constraints=2,
        )
        if hasattr(solver, "set_proximity_gated_collision_activation_enabled"):
            solver.set_proximity_gated_collision_activation_enabled(False)

        result = solver.solve_velocity(q, apply_limits=True)
        if result.status != eik.SolverStatus.SUCCESS:
            continue
        successes += 1
        solution = np.asarray(result.solution, dtype=float)
        q_next = robot.integrate(q, solution, solver.dt)
        _assert_joint_step_inside_limits(q_next, lower, upper)
        next_distance = solver.evaluate_min_collision_distance(q_next)
        assert next_distance is not None
        assert float(next_distance) >= float(current_distance) - 1e-9
    assert successes >= 20


def test_randomized_contact_projection_preserves_contact_velocity() -> None:
    rng = np.random.default_rng(20260526)
    urdf = _create_floating_contact_urdf()
    try:
        for seed_index in range(24):
            robot = eik.RobotModel(urdf, floating_base=True)
            solver = eik.KinematicsSolver(robot)
            solver.dt = 0.01
            solver.enable_position_limits(True)
            solver.enable_velocity_limits(True)
            _configure_weighted_runtime(solver)

            task = solver.add_frame_task(
                f"hand_pose_{seed_index}", "hand_link", eik.TaskType.FRAME_POSE
            )
            task.priority = 0
            task.weight = 1.0

            q = robot.neutral_configuration()
            robot.update_configuration(q)
            pose = robot.get_frame_pose("hand_link")
            target_pos = np.asarray(pose.translation, dtype=float) + rng.uniform(
                -0.08, 0.08, size=3
            )
            target_rot = _rx(rng.uniform(-0.35, 0.35)) @ np.asarray(pose.rotation, dtype=float)
            task.set_target_pose(target_pos, target_rot)

            left_type = (
                eik.ContactType.RIGID_CONTACT
                if seed_index % 2 == 0
                else eik.ContactType.POINT_CONTACT
            )
            solver.add_contact_frame("left_contact", left_type)
            if seed_index % 3 == 0:
                solver.add_contact_frame("right_contact", eik.ContactType.POINT_CONTACT)

            result = solver.solve_velocity(q, apply_limits=True)
            assert result.status in (eik.SolverStatus.SUCCESS, eik.SolverStatus.NO_PROGRESS)
            solution = np.asarray(result.solution, dtype=float)
            q_next = robot.integrate(q, solution, solver.dt)
            assert abs(np.linalg.norm(q_next[3:7]) - 1.0) <= 1e-10

            left_jac = np.asarray(robot.get_frame_jacobian("left_contact"), dtype=float)
            left_vel = (
                left_jac @ solution
                if left_type == eik.ContactType.RIGID_CONTACT
                else left_jac[:3, :] @ solution
            )
            assert np.linalg.norm(left_vel) <= 1e-8
            if seed_index % 3 == 0:
                right_jac = np.asarray(robot.get_frame_jacobian("right_contact"), dtype=float)
                assert np.linalg.norm(right_jac[:3, :] @ solution) <= 1e-8
    finally:
        os.remove(urdf)
