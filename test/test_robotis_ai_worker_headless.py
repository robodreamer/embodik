#!/usr/bin/env python3
"""Headless motion regressions for the public ROBOTIS AI Worker example."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


embodik = pytest.importorskip("embodik")

from examples.example_helpers.ai_worker_model_utils import (  # noqa: E402
    default_ai_worker_ik_joint_names,
    resolve_ai_worker_frames,
)
from examples.example_helpers.public_ai_worker_paths import resolve_public_ai_worker_urdf_paths  # noqa: E402
from examples.example_helpers.ai_worker_constraint_teleop_app import (  # noqa: E402
    DEFAULT_WORKER_SEED,
    _attempt_deep_penetration_escape_burst,
    _apply_named_joint_seed,
    _apply_soft_lift_margin,
    _configure_collision_constraint,
    _is_com_boundary_stall,
    _is_collision_boundary_stall,
    _generate_consecutive_collision_exclusions,
    _generate_worker_collision_include_pairs,
)
from examples.example_helpers.robust_ik_runtime import robust_solve_position_step  # noqa: E402


def _resolve_worker_urdf() -> Path:
    urdf, _collision_urdf = resolve_public_ai_worker_urdf_paths(variant="sg2")
    return Path(urdf)


def _resolve_worker_collision_urdf() -> Path | None:
    _urdf, collision_urdf = resolve_public_ai_worker_urdf_paths(variant="sg2")
    return Path(collision_urdf) if collision_urdf is not None else None


def _load_worker_robot():
    urdf = _resolve_worker_urdf()
    if not Path(urdf).is_file():
        pytest.skip("local FFW SG2 URDF not available")
    robot = embodik.RobotModel(str(urdf), floating_base=False)
    frames = resolve_ai_worker_frames(robot.get_frame_names())
    return robot, frames


def _load_worker_collision_robot():
    urdf = _resolve_worker_collision_urdf()
    if urdf is None or not Path(urdf).is_file():
        pytest.skip("generated worker collision URDF not available")
    full = embodik.RobotModel(str(urdf), floating_base=False)
    robot = embodik.RobotModel(
        str(urdf),
        actuated_joint_names=default_ai_worker_ik_joint_names(full.get_joint_names()),
        floating_base=False,
    )
    frames = resolve_ai_worker_frames(robot.get_frame_names())
    return robot, frames, Path(urdf)


def _frame_pose_matrix(robot, frame_name: str) -> np.ndarray:
    pose = robot.get_frame_pose(frame_name)
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(pose.rotation, dtype=float)
    T[:3, 3] = np.asarray(pose.translation, dtype=float)
    return T


def _build_locked_velocity_indices(robot) -> list[int]:
    joint_names = list(robot.get_joint_names())
    allowed_joint_names = set(default_ai_worker_ik_joint_names(joint_names))
    locked: list[int] = []
    for joint_name in joint_names:
        if joint_name in allowed_joint_names:
            continue
        idx_v = int(robot.get_joint_velocity_index(joint_name))
        nv_joint = int(robot.get_joint_velocity_size(joint_name)) if hasattr(robot, "get_joint_velocity_size") else 1
        for offset in range(max(nv_joint, 1)):
            locked.append(idx_v + offset)
    return sorted(set(locked))


def test_reduced_worker_ik_joint_set_excludes_wheels_and_head() -> None:
    urdf = _resolve_worker_urdf()
    reduced = embodik.RobotModel(
        str(urdf),
        actuated_joint_names=default_ai_worker_ik_joint_names(embodik.RobotModel(str(urdf), floating_base=False).get_joint_names()),
        floating_base=False,
    )
    joint_names = list(reduced.get_joint_names())
    assert "lift_joint" in joint_names
    assert any(name.startswith("arm_l_") for name in joint_names)
    assert any(name.startswith("arm_r_") for name in joint_names)
    assert not any("wheel_" in name for name in joint_names)
    assert not any(name.startswith("head_") for name in joint_names)
    assert not any(name.startswith("gripper_") for name in joint_names)


def test_worker_collision_exclusions_include_shoulder_root_pairs() -> None:
    robot, _frames, _collision_urdf = _load_worker_collision_robot()
    urdf = _resolve_worker_urdf()
    excl = set(tuple(p) for p in _generate_consecutive_collision_exclusions(robot, urdf))
    assert ("arm_base_link_0", "arm_l_link1_0") in excl or ("arm_l_link1_0", "arm_base_link_0") in excl
    assert ("arm_base_link_0", "arm_r_link1_0") in excl or ("arm_r_link1_0", "arm_base_link_0") in excl


def test_worker_collision_include_pairs_are_curated_for_teleop() -> None:
    robot, _frames, _collision_urdf = _load_worker_collision_robot()
    urdf = _resolve_worker_urdf()
    excl = _generate_consecutive_collision_exclusions(robot, urdf)
    include_pairs = _generate_worker_collision_include_pairs(robot, urdf, excl)
    assert 0 < len(include_pairs) <= 100
    for a, b in include_pairs:
        assert "camera_" not in a
        assert "camera_" not in b
        if a.startswith("gripper_"):
            assert a.endswith(("_base_0", "_l2_0", "_r2_0"))
        if b.startswith("gripper_"):
            assert b.endswith(("_base_0", "_l2_0", "_r2_0"))


def test_worker_self_collision_constraint_stays_feasible_for_inward_reach() -> None:
    urdf = _resolve_worker_collision_urdf()
    assert urdf is not None
    visual_urdf = _resolve_worker_urdf()
    full = embodik.RobotModel(str(urdf), floating_base=False)
    robot = embodik.RobotModel(
        str(urdf),
        actuated_joint_names=default_ai_worker_ik_joint_names(full.get_joint_names()),
        floating_base=False,
    )
    frames = resolve_ai_worker_frames(robot.get_frame_names())
    q_lo, q_hi = robot.get_joint_limits()
    joint_name_to_cfg = {name: int(robot.get_joint_config_index(name)) for name in robot.get_joint_names()}
    q0 = _apply_named_joint_seed(robot.neutral_configuration(), joint_name_to_cfg, q_lo, q_hi, DEFAULT_WORKER_SEED)
    robot.update_configuration(q0)

    def _frame_pose_matrix_local(frame_name: str) -> np.ndarray:
        pose = robot.get_frame_pose(frame_name)
        T = np.eye(4, dtype=float)
        T[:3, :3] = np.asarray(pose.rotation, dtype=float)
        T[:3, 3] = np.asarray(pose.translation, dtype=float)
        return T

    right_target = _frame_pose_matrix_local(frames["right_tool"])
    left_target = _frame_pose_matrix_local(frames["left_tool"])
    right_target[:3, 3] += np.array([-0.18, 0.12, 0.02], dtype=float)
    left_target[:3, 3] += np.array([-0.18, -0.12, 0.02], dtype=float)

    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)
    solver.set_tolerance(0.1)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    right_task = solver.add_frame_task("right_tool_pose", frames["right_tool"], embodik.TaskType.FRAME_POSE)
    left_task = solver.add_frame_task("left_tool_pose", frames["left_tool"], embodik.TaskType.FRAME_POSE)
    for task in (right_task, left_task):
        task.priority = 0
        task.weight = 1.0
        task.solve_mode = embodik.TaskSolveMode.SCALE_ELASTIC

    exclusions = _generate_consecutive_collision_exclusions(robot, visual_urdf)
    include_pairs = _generate_worker_collision_include_pairs(robot, visual_urdf, exclusions)
    solver.configure_collision_constraint(
        min_distance=0.03, include_pairs=include_pairs, exclude_pairs=exclusions, max_constraints=3
    )

    opts = embodik.PositionStepOptions()
    opts.max_steps = 20
    opts.position_gain = 10.0
    opts.orientation_gain = 10.0
    result = solver.solve_position_step(
        q0,
        [
            embodik.TaskTarget("right_tool_pose", right_target, 10.0, 10.0),
            embodik.TaskTarget("left_tool_pose", left_target, 10.0, 10.0),
        ],
        opts,
    )
    assert result.status.name == "SUCCESS"

    min_dist = solver.evaluate_min_collision_distance(np.asarray(result.q_solution, dtype=float))
    assert min_dist is not None
    assert float(min_dist) >= 0.03 - 1e-3


def _solve_single_step(
    robot,
    frames: dict[str, str],
    *,
    right_target_offset: np.ndarray,
    exclude_right: list[int] | None = None,
    exclude_left: list[int] | None = None,
) -> np.ndarray:
    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)
    solver.set_tolerance(0.1)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    right_task = solver.add_frame_task("right_tool_pose", frames["right_tool"], embodik.TaskType.FRAME_POSE)
    left_task = solver.add_frame_task("left_tool_pose", frames["left_tool"], embodik.TaskType.FRAME_POSE)
    for task in (right_task, left_task):
        task.priority = 0
        task.weight = 1.0
        task.solve_mode = embodik.TaskSolveMode.SCALE_ELASTIC

    if exclude_right:
        right_task.set_excluded_joint_indices(list(exclude_right))
    if exclude_left:
        left_task.set_excluded_joint_indices(list(exclude_left))

    q0 = robot.neutral_configuration()
    robot.update_configuration(q0)
    right0 = _frame_pose_matrix(robot, frames["right_tool"])
    left0 = _frame_pose_matrix(robot, frames["left_tool"])

    right_target = right0.copy()
    right_target[:3, 3] += np.asarray(right_target_offset, dtype=float)

    opts = embodik.PositionStepOptions()
    opts.max_steps = 1
    opts.position_gain = 10.0
    opts.orientation_gain = 10.0
    locked = _build_locked_velocity_indices(robot)
    opts.excluded_joint_indices = list(locked)
    opts.integration_zero_velocity_indices = list(locked)

    result = solver.solve_position_step(
        q0,
        [
            embodik.TaskTarget("right_tool_pose", right_target, 10.0, 10.0),
            embodik.TaskTarget("left_tool_pose", left0, 10.0, 10.0),
        ],
        opts,
    )
    return np.asarray(result.q_solution, dtype=float)


def test_direct_joint_configuration_moves_right_tool() -> None:
    robot, frames = _load_worker_robot()
    q0 = robot.neutral_configuration()
    robot.update_configuration(q0)
    p0 = _frame_pose_matrix(robot, frames["right_tool"])[:3, 3].copy()

    idx = int(robot.get_joint_config_index("arm_r_joint4"))
    q1 = q0.copy()
    q1[idx] += 0.2
    q_lo, q_hi = robot.get_joint_limits()
    q1 = np.clip(q1, q_lo, q_hi)

    robot.update_configuration(q1)
    p1 = _frame_pose_matrix(robot, frames["right_tool"])[:3, 3].copy()
    assert np.linalg.norm(p1 - p0) > 1e-4


def test_single_step_ik_moves_right_tool_toward_target() -> None:
    robot, frames = _load_worker_robot()
    q0 = robot.neutral_configuration()
    robot.update_configuration(q0)
    right0 = _frame_pose_matrix(robot, frames["right_tool"])
    q1 = _solve_single_step(robot, frames, right_target_offset=np.array([0.10, 0.0, 0.0], dtype=float))
    assert np.linalg.norm(q1 - q0) > 1e-6

    robot.update_configuration(q1)
    right1 = _frame_pose_matrix(robot, frames["right_tool"])
    right_target = right0.copy()
    right_target[:3, 3] += np.array([0.10, 0.0, 0.0], dtype=float)
    moved = np.linalg.norm(right1[:3, 3] - right0[:3, 3])
    residual0 = np.linalg.norm(right_target[:3, 3] - right0[:3, 3])
    residual1 = np.linalg.norm(right_target[:3, 3] - right1[:3, 3])

    assert moved > 1e-4
    assert residual1 < residual0


def test_unlocked_lift_can_participate_in_single_arm_solve() -> None:
    robot, frames = _load_worker_robot()
    joint_names = list(robot.get_joint_names())
    left_arm_vel = []
    lift_vel = []
    for joint_name in joint_names:
        idx_v = int(robot.get_joint_velocity_index(joint_name))
        nv_joint = int(robot.get_joint_velocity_size(joint_name)) if hasattr(robot, "get_joint_velocity_size") else 1
        expanded = [idx_v + offset for offset in range(max(nv_joint, 1))]
        if joint_name.startswith(("arm_l_", "gripper_l_")):
            left_arm_vel.extend(expanded)
        if joint_name.startswith("lift_"):
            lift_vel.extend(expanded)

    q_unlocked = _solve_single_step(
        robot,
        frames,
        right_target_offset=np.array([0.0, 0.0, 0.08], dtype=float),
        exclude_right=left_arm_vel,
        exclude_left=[],
    )
    q_locked = _solve_single_step(
        robot,
        frames,
        right_target_offset=np.array([0.0, 0.0, 0.08], dtype=float),
        exclude_right=left_arm_vel + lift_vel,
        exclude_left=lift_vel,
    )

    lift_idx = int(robot.get_joint_config_index("lift_joint"))
    lift_unlocked = float(q_unlocked[lift_idx])
    lift_locked = float(q_locked[lift_idx])
    assert abs(lift_unlocked) > 1e-5
    assert abs(lift_locked) <= 1e-9


def test_worker_one_arm_collision_plateau_classifies_as_boundary_stall() -> None:
    robot, frames, collision_urdf = _load_worker_collision_robot()
    visual_urdf = _resolve_worker_urdf()
    q_lo, q_hi = robot.get_joint_limits()
    joint_name_to_cfg = {name: int(robot.get_joint_config_index(name)) for name in robot.get_joint_names()}
    q = _apply_named_joint_seed(robot.neutral_configuration(), joint_name_to_cfg, q_lo, q_hi, DEFAULT_WORKER_SEED)
    q = _apply_soft_lift_margin(q, joint_name_to_cfg=joint_name_to_cfg, q_lo=q_lo, q_hi=q_hi)
    robot.update_configuration(q)

    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)
    solver.set_tolerance(0.1)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    right_task = solver.add_frame_task("right_tool_pose", frames["right_tool"], embodik.TaskType.FRAME_POSE)
    left_task = solver.add_frame_task("left_tool_pose", frames["left_tool"], embodik.TaskType.FRAME_POSE)
    for task in (right_task, left_task):
        task.priority = 0
        task.weight = 1.0
        task.solve_mode = embodik.TaskSolveMode.SCALE_ELASTIC

    left_arm_velocity_indices: list[int] = []
    locked_velocity_indices: list[int] = []
    allowed_joint_names = set(default_ai_worker_ik_joint_names(robot.get_joint_names()))
    for joint_name in robot.get_joint_names():
        idx_v = int(robot.get_joint_velocity_index(joint_name))
        nv_joint = int(robot.get_joint_velocity_size(joint_name)) if hasattr(robot, "get_joint_velocity_size") else 1
        for offset in range(max(nv_joint, 1)):
            vi = idx_v + offset
            if joint_name.startswith(("arm_l_", "gripper_l_")):
                left_arm_velocity_indices.append(vi)
            if joint_name not in allowed_joint_names:
                locked_velocity_indices.append(vi)
    left_arm_velocity_indices = sorted(set(left_arm_velocity_indices))
    locked_velocity_indices = sorted(set(locked_velocity_indices))

    right_task.set_excluded_joint_indices(left_arm_velocity_indices)
    if hasattr(left_task, "clear_excluded_joint_indices"):
        left_task.clear_excluded_joint_indices()

    exclusions = _generate_consecutive_collision_exclusions(robot, visual_urdf)
    include_pairs = _generate_worker_collision_include_pairs(robot, visual_urdf, exclusions)
    _configure_collision_constraint(
        solver,
        enabled=True,
        min_distance_m=0.035,
        max_constraints=3,
        tuning_mode="balanced",
        include_pairs=include_pairs,
        exclude_pairs=exclusions,
    )

    opts = embodik.PositionStepOptions()
    opts.max_steps = 1
    opts.position_gain = 10.0
    opts.orientation_gain = 10.0
    opts.stall_recovery = True
    opts.excluded_joint_indices = sorted(set(locked_velocity_indices + left_arm_velocity_indices))
    opts.integration_zero_velocity_indices = sorted(set(locked_velocity_indices + left_arm_velocity_indices))

    start_pose = _frame_pose_matrix(robot, frames["right_tool"])
    target_pose = start_pose.copy()
    target_pose[:3, 3] += np.array([-0.25, 0.25, 0.0], dtype=float)

    boundary_stall_detected = False
    snapped_target = None
    for _step_idx in range(50):
        step = robust_solve_position_step(
            robot=robot,
            solver=solver,
            q_current=q,
            targets=[embodik.TaskTarget("right_tool_pose", target_pose, 10.0, 10.0)],
            options=opts,
            q_lo=q_lo,
            q_hi=q_hi,
            zero_velocity_indices=locked_velocity_indices + left_arm_velocity_indices,
            fallback_status_names=("INVALID_INPUT", "NUMERICAL_ERROR"),
            allow_solver_intervention=True,
            apply_collision_violated_q_solution=True,
        )
        q = step.q_next
        robot.update_configuration(q)

        current_collision_min = None
        if hasattr(solver, "get_last_collision_debug_list"):
            debug_rows = list(solver.get_last_collision_debug_list())
            if debug_rows:
                current_collision_min = min(float(row.distance) for row in debug_rows)
        if current_collision_min is None and hasattr(solver, "get_last_collision_debug"):
            dbg = solver.get_last_collision_debug()
            if dbg is not None:
                current_collision_min = float(dbg.distance)

        if _is_collision_boundary_stall(
            result=step.solver_result,
            collision_enabled=True,
            current_collision_min=current_collision_min,
            collision_min_distance_m=0.035,
        ):
            boundary_stall_detected = True
            snapped_target = _frame_pose_matrix(robot, frames["right_tool"])
            break

    assert boundary_stall_detected, "Expected repeated inward solve to reach a collision-boundary plateau"
    assert snapped_target is not None

    recovery_step = robust_solve_position_step(
        robot=robot,
        solver=solver,
        q_current=q,
        targets=[embodik.TaskTarget("right_tool_pose", snapped_target, 10.0, 10.0)],
        options=opts,
        q_lo=q_lo,
        q_hi=q_hi,
        zero_velocity_indices=locked_velocity_indices + left_arm_velocity_indices,
        fallback_status_names=("INVALID_INPUT", "NUMERICAL_ERROR"),
        allow_solver_intervention=True,
        apply_collision_violated_q_solution=True,
    )
    np.testing.assert_allclose(recovery_step.q_next, q, atol=1e-9)


class _DummyStatus:
    def __init__(self, name: str):
        self.name = name


class _DummyResult:
    def __init__(self, status_name: str, dq_norm: float = 0.0):
        self.status = _DummyStatus(status_name)
        self.joint_velocities = np.array([dq_norm], dtype=float)


def test_worker_com_boundary_stall_classifier_detects_zero_motion_margin_plateau() -> None:
    result = _DummyResult("INFEASIBLE", dq_norm=0.0)
    assert _is_com_boundary_stall(
        result=result,
        com_enabled=True,
        current_com_min_slack=1e-4,
    )
    assert not _is_com_boundary_stall(
        result=result,
        com_enabled=True,
        current_com_min_slack=2e-2,
    )
    # SCALE_ELASTIC routinely converges at the boundary with SUCCESS + tiny dq
    # — that IS the stall plateau we want the UI to snap out of.
    assert _is_com_boundary_stall(
        result=_DummyResult("SUCCESS", dq_norm=0.0),
        com_enabled=True,
        current_com_min_slack=1e-4,
    )
    # Genuine motion (dq well above the stall threshold) is not a stall.
    assert not _is_com_boundary_stall(
        result=_DummyResult("INFEASIBLE", dq_norm=1e-3),
        com_enabled=True,
        current_com_min_slack=1e-4,
    )
    assert not _is_com_boundary_stall(
        result=_DummyResult("SUCCESS", dq_norm=1e-3),
        com_enabled=True,
        current_com_min_slack=1e-4,
    )


def test_worker_deep_penetration_escape_burst_recovers_stuck_pull_away() -> None:
    robot, frames, collision_urdf = _load_worker_collision_robot()
    visual_urdf = _resolve_worker_urdf()
    q_lo, q_hi = robot.get_joint_limits()
    joint_name_to_cfg = {name: int(robot.get_joint_config_index(name)) for name in robot.get_joint_names()}
    q = _apply_named_joint_seed(robot.neutral_configuration(), joint_name_to_cfg, q_lo, q_hi, DEFAULT_WORKER_SEED)
    q = _apply_soft_lift_margin(q, joint_name_to_cfg=joint_name_to_cfg, q_lo=q_lo, q_hi=q_hi)
    robot.update_configuration(q)

    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)
    solver.set_tolerance(0.1)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    right_task = solver.add_frame_task("right_tool_pose", frames["right_tool"], embodik.TaskType.FRAME_POSE)
    left_task = solver.add_frame_task("left_tool_pose", frames["left_tool"], embodik.TaskType.FRAME_POSE)
    for task in (right_task, left_task):
        task.priority = 0
        task.weight = 1.0
        task.solve_mode = embodik.TaskSolveMode.SCALE_ELASTIC

    left_arm_velocity_indices: list[int] = []
    locked_velocity_indices: list[int] = []
    allowed_joint_names = set(default_ai_worker_ik_joint_names(robot.get_joint_names()))
    for joint_name in robot.get_joint_names():
        idx_v = int(robot.get_joint_velocity_index(joint_name))
        nv_joint = int(robot.get_joint_velocity_size(joint_name)) if hasattr(robot, "get_joint_velocity_size") else 1
        for offset in range(max(nv_joint, 1)):
            vi = idx_v + offset
            if joint_name.startswith(("arm_l_", "gripper_l_")):
                left_arm_velocity_indices.append(vi)
            if joint_name not in allowed_joint_names:
                locked_velocity_indices.append(vi)
    left_arm_velocity_indices = sorted(set(left_arm_velocity_indices))
    locked_velocity_indices = sorted(set(locked_velocity_indices))
    right_task.set_excluded_joint_indices(left_arm_velocity_indices)

    opts = embodik.PositionStepOptions()
    opts.max_steps = 1
    opts.position_gain = 10.0
    opts.orientation_gain = 10.0
    opts.stall_recovery = True
    opts.excluded_joint_indices = sorted(set(locked_velocity_indices + left_arm_velocity_indices))
    opts.integration_zero_velocity_indices = sorted(set(locked_velocity_indices + left_arm_velocity_indices))

    start_pose = _frame_pose_matrix(robot, frames["right_tool"])
    into_target = start_pose.copy()
    into_target[:3, 3] += np.array([-0.32, 0.30, 0.0], dtype=float)

    for _step_idx in range(60):
        step = robust_solve_position_step(
            robot=robot,
            solver=solver,
            q_current=q,
            targets=[embodik.TaskTarget("right_tool_pose", into_target, 10.0, 10.0)],
            options=opts,
            q_lo=q_lo,
            q_hi=q_hi,
            zero_velocity_indices=opts.integration_zero_velocity_indices,
            fallback_status_names=("INVALID_INPUT", "NUMERICAL_ERROR"),
        )
        q = step.q_next
        robot.update_configuration(q)

    assert float(solver.evaluate_collision_debug(q).distance) <= 0.0

    exclusions = _generate_consecutive_collision_exclusions(robot, visual_urdf)
    include_pairs = _generate_worker_collision_include_pairs(robot, visual_urdf, exclusions)
    _configure_collision_constraint(
        solver,
        enabled=True,
        min_distance_m=0.035,
        max_constraints=3,
        tuning_mode="balanced",
        include_pairs=include_pairs,
        exclude_pairs=exclusions,
    )

    away_target = start_pose.copy()
    away_target[:3, 3] += np.array([0.10, -0.10, 0.0], dtype=float)
    frozen_step = robust_solve_position_step(
        robot=robot,
        solver=solver,
        q_current=q,
        targets=[embodik.TaskTarget("right_tool_pose", away_target, 10.0, 10.0)],
        options=opts,
        q_lo=q_lo,
        q_hi=q_hi,
        zero_velocity_indices=opts.integration_zero_velocity_indices,
        fallback_status_names=("INVALID_INPUT", "NUMERICAL_ERROR"),
        allow_solver_intervention=True,
        apply_collision_violated_q_solution=True,
    )
    assert frozen_step.solver_result.status.name == "INFEASIBLE"
    assert np.linalg.norm(np.asarray(frozen_step.solver_result.joint_velocities, dtype=float)) == pytest.approx(0.0)

    escaped_q = _attempt_deep_penetration_escape_burst(
        robot=robot,
        solver=solver,
        q_current=q,
        targets=[embodik.TaskTarget("right_tool_pose", away_target, 10.0, 10.0)],
        options=opts,
        q_lo=q_lo,
        q_hi=q_hi,
        zero_velocity_indices=opts.integration_zero_velocity_indices,
        min_distance_m=0.035,
        max_constraints=3,
        tuning_mode="balanced",
        include_pairs=include_pairs,
        exclude_pairs=exclusions,
    )
    assert escaped_q is not None, "Expected deep-penetration escape burst to find a better configuration"

    q = np.asarray(escaped_q, dtype=float)
    robot.update_configuration(q)
    assert float(solver.evaluate_collision_debug(q).distance) > 0.0

    recovery_step = robust_solve_position_step(
        robot=robot,
        solver=solver,
        q_current=q,
        targets=[embodik.TaskTarget("right_tool_pose", away_target, 10.0, 10.0)],
        options=opts,
        q_lo=q_lo,
        q_hi=q_hi,
        zero_velocity_indices=opts.integration_zero_velocity_indices,
        fallback_status_names=("INVALID_INPUT", "NUMERICAL_ERROR"),
        allow_solver_intervention=True,
        apply_collision_violated_q_solution=True,
    )
    assert recovery_step.solver_result.status.name == "SUCCESS"


def test_worker_target_in_torso_release_preserves_collision_margin() -> None:
    """Balanced collision tuning should recover from an impossible torso target
    without drifting through the configured self-collision clearance shell.
    """
    from scripts.verify_ai_worker_robustness import run_scenario

    _load_worker_collision_robot()
    metrics = run_scenario(
        "target_in_torso_release",
        variant="sg2",
        duration_steps=140,
        collision_min_distance_m=0.035,
        max_collision_constraints=2,
        com_margin_frac=0.10,
        collision_tuning_mode="balanced",
        solve_mode_label="SCALE_ELASTIC",
        allow_fallback=False,
        pos_gain=10.0,
        rot_gain=10.0,
        enable_collision=True,
        enable_com=True,
        fallback_status_names=("INVALID_INPUT", "INFEASIBLE", "NUMERICAL_ERROR", "NO_PROGRESS"),
    )
    record = metrics.as_dict()
    assert record["collision_breach_count"] == 0
    assert record["unrecoverable_stall_count"] == 0
    assert record["min_collision_distance_m"] >= 0.034
