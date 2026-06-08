#!/usr/bin/env python3
"""Headless motion regressions for the public ROBOTIS AI Worker example."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

embodik = pytest.importorskip("embodik")

from embodik.interactive_ik import (  # noqa: E402
    is_collision_boundary_stall as _is_collision_boundary_stall,
)
from embodik.interactive_ik import is_com_boundary_stall as _is_com_boundary_stall
from embodik.interactive_ik import (
    robust_solve_position_step,
)
from examples.example_helpers.common_bimanual_model_utils import (  # noqa: E402
    default_common_bimanual_ik_joint_names,
    resolve_common_bimanual_frames,
)
from examples.example_helpers.common_bimanual_teleop_app import (  # noqa: E402
    COMMON_BIMANUAL_SUPPORT_CONTACT_FRAMES,
    DEFAULT_ADAPTIVE_DT_MAX_SCALE,
    DEFAULT_COMMON_BIMANUAL_SEED,
    DEFAULT_MAX_ANGULAR_SPEED,
    DEFAULT_MAX_LINEAR_SPEED,
    _apply_named_joint_seed,
    _apply_position_step_speed_caps,
    _apply_soft_lift_margin,
    _compute_support_polygon_from_contacts,
    _configure_collision_constraint,
    _generate_common_bimanual_collision_include_pairs,
    _generate_consecutive_collision_exclusions,
)
from examples.example_helpers.ik_common import (  # noqa: E402
    configure_solver_runtime_policy,
)
from examples.example_helpers.public_ai_worker_paths import (  # noqa: E402
    resolve_public_ai_worker_urdf_paths,
)
from examples.harnesses.ai_worker_weighted_fallback_harness import (  # noqa: E402
    run_ai_worker_weighted_fallback_comparison,
)


def _disable_weighted_fallback(solver: embodik.KinematicsSolver) -> None:
    cfg = embodik.SolverRuntimeConfig()
    cfg.weighted_fallback_enabled = False
    solver.configure_runtime(cfg)


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
    frames = resolve_common_bimanual_frames(robot.get_frame_names())
    return robot, frames


def _load_worker_collision_robot():
    urdf = _resolve_worker_collision_urdf()
    if urdf is None or not Path(urdf).is_file():
        pytest.skip("generated worker collision URDF not available")
    full = embodik.RobotModel(str(urdf), floating_base=False)
    robot = embodik.RobotModel(
        str(urdf),
        actuated_joint_names=default_common_bimanual_ik_joint_names(full.get_joint_names()),
        floating_base=False,
    )
    frames = resolve_common_bimanual_frames(robot.get_frame_names())
    return robot, frames, Path(urdf)


def _load_reduced_worker_robot():
    urdf = _resolve_worker_urdf()
    if not Path(urdf).is_file():
        pytest.skip("local FFW SG2 URDF not available")
    full = embodik.RobotModel(str(urdf), floating_base=False)
    robot = embodik.RobotModel(
        str(urdf),
        actuated_joint_names=default_common_bimanual_ik_joint_names(full.get_joint_names()),
        floating_base=False,
    )
    frames = resolve_common_bimanual_frames(robot.get_frame_names())
    q_lo, q_hi = robot.get_joint_limits()
    joint_name_to_cfg = {
        name: int(robot.get_joint_config_index(name)) for name in robot.get_joint_names()
    }
    q0 = _apply_named_joint_seed(
        robot.neutral_configuration(), joint_name_to_cfg, q_lo, q_hi, DEFAULT_COMMON_BIMANUAL_SEED
    )
    q0 = _apply_soft_lift_margin(q0, joint_name_to_cfg=joint_name_to_cfg, q_lo=q_lo, q_hi=q_hi)
    return robot, frames, np.asarray(q0, dtype=float)


def _frame_pose_matrix(robot, frame_name: str) -> np.ndarray:
    pose = robot.get_frame_pose(frame_name)
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(pose.rotation, dtype=float)
    T[:3, 3] = np.asarray(pose.translation, dtype=float)
    return T


def _build_locked_velocity_indices(robot) -> list[int]:
    joint_names = list(robot.get_joint_names())
    allowed_joint_names = set(default_common_bimanual_ik_joint_names(joint_names))
    locked: list[int] = []
    for joint_name in joint_names:
        if joint_name in allowed_joint_names:
            continue
        idx_v = int(robot.get_joint_velocity_index(joint_name))
        nv_joint = (
            int(robot.get_joint_velocity_size(joint_name))
            if hasattr(robot, "get_joint_velocity_size")
            else 1
        )
        for offset in range(max(nv_joint, 1)):
            locked.append(idx_v + offset)
    return sorted(set(locked))


def _run_limited_bimanual_pose_group_case(
    *,
    mode: str,
    limit_width: float,
    target_offset: np.ndarray,
    steps: int = 40,
):
    robot, frames, q0 = _load_reduced_worker_robot()
    robot.update_configuration(q0)
    lower, upper = robot.get_joint_limits()
    lower = lower.copy()
    upper = upper.copy()
    for idx in range(len(q0)):
        lower[idx] = max(lower[idx], q0[idx] - limit_width)
        upper[idx] = min(upper[idx], q0[idx] + limit_width)
    robot.set_joint_limits(lower, upper)

    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    _disable_weighted_fallback(solver)
    cfg = embodik.SolverRuntimeConfig()
    cfg.weighted_fallback_enabled = False
    cfg.enable_auto_task_layout = mode == "auto"
    solver.configure_runtime(cfg)

    groups = [
        solver.add_pose_task_group(
            "right_tool_pose",
            frames["right_tool"],
            merged_pose=mode == "merged",
            auto_switch=mode == "auto",
        ),
        solver.add_pose_task_group(
            "left_tool_pose",
            frames["left_tool"],
            merged_pose=mode == "merged",
            auto_switch=mode == "auto",
        ),
    ]
    right_target = _frame_pose_matrix(robot, frames["right_tool"])
    left_target = _frame_pose_matrix(robot, frames["left_tool"])
    right_target[:3, 3] += np.asarray(target_offset, dtype=float)
    left_target[:3, 3] += np.asarray(target_offset, dtype=float) * np.array(
        [1.0, -1.0, 1.0], dtype=float
    )

    opts = embodik.PositionStepOptions()
    opts.max_steps = 1
    statuses = []
    layouts = []
    errors = []
    moved_frames = 0
    q = q0.copy()
    for _ in range(steps):
        targets = []
        for group, target in zip(groups, (right_target, left_target)):
            group.set_target(target, position_gain=10.0, rotation_gain=3.0)
            targets.extend(group.task_targets())
        result = solver.solve_position_step(q, targets, opts)
        q_next = np.asarray(result.q_solution, dtype=float)
        if float(np.linalg.norm(q_next - q)) > 1e-9:
            moved_frames += 1
        q = q_next
        statuses.append(result.status)
        layouts.append(result.diagnostics.active_task_layout)
        errors.append(float(result.position_error))
    return statuses, layouts, errors, moved_frames


def test_reduced_worker_ik_joint_set_excludes_wheels_and_head() -> None:
    urdf = _resolve_worker_urdf()
    reduced = embodik.RobotModel(
        str(urdf),
        actuated_joint_names=default_common_bimanual_ik_joint_names(
            embodik.RobotModel(str(urdf), floating_base=False).get_joint_names()
        ),
        floating_base=False,
    )
    joint_names = list(reduced.get_joint_names())
    assert "lift_joint" in joint_names
    assert any(name.startswith("arm_l_") for name in joint_names)
    assert any(name.startswith("arm_r_") for name in joint_names)
    assert not any("wheel_" in name for name in joint_names)
    assert not any(name.startswith("head_") for name in joint_names)
    assert not any(name.startswith("gripper_") for name in joint_names)


def test_ai_worker_weighted_fallback_accepts_useful_limit_solution() -> None:
    comparison = run_ai_worker_weighted_fallback_comparison()

    assert comparison.prioritized_status == embodik.SolverStatus.INFEASIBLE
    assert comparison.weighted_status == embodik.SolverStatus.SUCCESS
    assert comparison.weighted_fallback_used is True
    assert comparison.weighted_advisory_available is True
    assert abs(comparison.weighted_solution[comparison.blocked_index]) <= 1e-9
    assert comparison.weighted_solution[comparison.secondary_index] > 0.35
    assert comparison.prioritized_solution[comparison.secondary_index] > 0.35
    assert np.all(comparison.q_next <= comparison.upper + 1e-9)
    assert np.all(comparison.q_next >= comparison.lower - 1e-9)
    assert "constrained weighted fallback accepted" in comparison.weighted_status_message


def test_ai_worker_auto_pose_layout_keeps_bimanual_case_productive() -> None:
    offset = np.array([-0.18, 0.12, 0.03], dtype=float)

    merged_statuses, merged_layouts, merged_errors, merged_moved = (
        _run_limited_bimanual_pose_group_case(mode="merged", limit_width=0.04, target_offset=offset)
    )
    split_statuses, split_layouts, split_errors, split_moved = (
        _run_limited_bimanual_pose_group_case(mode="split", limit_width=0.04, target_offset=offset)
    )
    auto_statuses, auto_layouts, auto_errors, auto_moved = _run_limited_bimanual_pose_group_case(
        mode="auto", limit_width=0.04, target_offset=offset
    )

    assert all(layout == embodik.TaskLayout.MERGED for layout in merged_layouts)
    assert all(layout == embodik.TaskLayout.SPLIT for layout in split_layouts)
    assert auto_layouts[0] == embodik.TaskLayout.SPLIT
    assert all(layout == embodik.TaskLayout.SPLIT for layout in auto_layouts)
    assert merged_statuses[-1] != embodik.SolverStatus.SUCCESS
    assert split_statuses[-1] == embodik.SolverStatus.SUCCESS
    assert auto_statuses[-1] == embodik.SolverStatus.SUCCESS
    assert merged_moved < 15
    assert split_moved >= 35
    assert auto_moved >= 35
    assert auto_errors[-1] <= merged_errors[-1]


def test_worker_collision_exclusions_include_shoulder_root_pairs() -> None:
    robot, _frames, _collision_urdf = _load_worker_collision_robot()
    urdf = _resolve_worker_urdf()
    excl = set(tuple(p) for p in _generate_consecutive_collision_exclusions(robot, urdf))
    assert ("arm_base_link_0", "arm_l_link1_0") in excl or (
        "arm_l_link1_0",
        "arm_base_link_0",
    ) in excl
    assert ("arm_base_link_0", "arm_r_link1_0") in excl or (
        "arm_r_link1_0",
        "arm_base_link_0",
    ) in excl


def test_worker_collision_include_pairs_are_curated_for_teleop() -> None:
    robot, _frames, _collision_urdf = _load_worker_collision_robot()
    urdf = _resolve_worker_urdf()
    excl = _generate_consecutive_collision_exclusions(robot, urdf)
    include_pairs = _generate_common_bimanual_collision_include_pairs(robot, urdf, excl)
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
        actuated_joint_names=default_common_bimanual_ik_joint_names(full.get_joint_names()),
        floating_base=False,
    )
    frames = resolve_common_bimanual_frames(robot.get_frame_names())
    q_lo, q_hi = robot.get_joint_limits()
    joint_name_to_cfg = {
        name: int(robot.get_joint_config_index(name)) for name in robot.get_joint_names()
    }
    q0 = _apply_named_joint_seed(
        robot.neutral_configuration(), joint_name_to_cfg, q_lo, q_hi, DEFAULT_COMMON_BIMANUAL_SEED
    )
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
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    _disable_weighted_fallback(solver)
    right_task = solver.add_frame_task(
        "right_tool_pose", frames["right_tool"], embodik.TaskType.FRAME_POSE
    )
    left_task = solver.add_frame_task(
        "left_tool_pose", frames["left_tool"], embodik.TaskType.FRAME_POSE
    )
    for task in (right_task, left_task):
        task.priority = 0
        task.weight = 1.0
        task.solve_mode = embodik.TaskSolveMode.SCALE_ELASTIC

    exclusions = _generate_consecutive_collision_exclusions(robot, visual_urdf)
    include_pairs = _generate_common_bimanual_collision_include_pairs(
        robot, visual_urdf, exclusions
    )
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
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    right_task = solver.add_frame_task(
        "right_tool_pose", frames["right_tool"], embodik.TaskType.FRAME_POSE
    )
    left_task = solver.add_frame_task(
        "left_tool_pose", frames["left_tool"], embodik.TaskType.FRAME_POSE
    )
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
    q1 = _solve_single_step(
        robot, frames, right_target_offset=np.array([0.10, 0.0, 0.0], dtype=float)
    )
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
        nv_joint = (
            int(robot.get_joint_velocity_size(joint_name))
            if hasattr(robot, "get_joint_velocity_size")
            else 1
        )
        expanded = [idx_v + offset for offset in range(max(nv_joint, 1))]
        if joint_name.startswith(("arm_l_", "gripper_l_")):
            left_arm_vel.extend(expanded)
        if joint_name.startswith("lift_"):
            lift_vel.extend(expanded)

    q_up = _solve_single_step(
        robot,
        frames,
        right_target_offset=np.array([0.0, 0.0, 0.08], dtype=float),
        exclude_right=left_arm_vel,
        exclude_left=[],
    )
    q_unlocked = _solve_single_step(
        robot,
        frames,
        right_target_offset=np.array([0.0, 0.0, -0.08], dtype=float),
        exclude_right=left_arm_vel,
        exclude_left=[],
    )
    q_locked = _solve_single_step(
        robot,
        frames,
        right_target_offset=np.array([0.0, 0.0, -0.08], dtype=float),
        exclude_right=left_arm_vel + lift_vel,
        exclude_left=lift_vel,
    )

    lift_idx = int(robot.get_joint_config_index("lift_joint"))
    q_lo, q_hi = robot.get_joint_limits()
    lift_up = float(q_up[lift_idx])
    lift_unlocked = float(q_unlocked[lift_idx])
    lift_locked = float(q_locked[lift_idx])
    assert q_lo[lift_idx] <= lift_up <= q_hi[lift_idx]
    assert abs(lift_unlocked) > 1e-5
    assert abs(lift_locked) <= 1e-9


def test_worker_one_arm_collision_push_remains_productive_without_boundary_stall() -> None:
    robot, frames, collision_urdf = _load_worker_collision_robot()
    visual_urdf = _resolve_worker_urdf()
    q_lo, q_hi = robot.get_joint_limits()
    joint_name_to_cfg = {
        name: int(robot.get_joint_config_index(name)) for name in robot.get_joint_names()
    }
    q = _apply_named_joint_seed(
        robot.neutral_configuration(), joint_name_to_cfg, q_lo, q_hi, DEFAULT_COMMON_BIMANUAL_SEED
    )
    q = _apply_soft_lift_margin(q, joint_name_to_cfg=joint_name_to_cfg, q_lo=q_lo, q_hi=q_hi)
    robot.update_configuration(q)

    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    right_task = solver.add_frame_task(
        "right_tool_pose", frames["right_tool"], embodik.TaskType.FRAME_POSE
    )
    left_task = solver.add_frame_task(
        "left_tool_pose", frames["left_tool"], embodik.TaskType.FRAME_POSE
    )
    for task in (right_task, left_task):
        task.priority = 0
        task.weight = 1.0
        task.solve_mode = embodik.TaskSolveMode.SCALE_ELASTIC

    left_arm_velocity_indices: list[int] = []
    locked_velocity_indices: list[int] = []
    allowed_joint_names = set(default_common_bimanual_ik_joint_names(robot.get_joint_names()))
    for joint_name in robot.get_joint_names():
        idx_v = int(robot.get_joint_velocity_index(joint_name))
        nv_joint = (
            int(robot.get_joint_velocity_size(joint_name))
            if hasattr(robot, "get_joint_velocity_size")
            else 1
        )
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
    include_pairs = _generate_common_bimanual_collision_include_pairs(
        robot, visual_urdf, exclusions
    )
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
    opts.integration_zero_velocity_indices = sorted(
        set(locked_velocity_indices + left_arm_velocity_indices)
    )

    start_pose = _frame_pose_matrix(robot, frames["right_tool"])
    target_pose = start_pose.copy()
    target_pose[:3, 3] += np.array([-0.25, 0.25, 0.0], dtype=float)

    boundary_stall_detected = False
    statuses = []
    for _step_idx in range(50):
        step = robust_solve_position_step(
            solver=solver,
            q_current=q,
            targets=[embodik.TaskTarget("right_tool_pose", target_pose, 10.0, 10.0)],
            options=opts,
        )
        statuses.append(step.solver_result.status.name)
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
            break

    assert not boundary_stall_detected
    assert statuses
    assert statuses[-1] == "SUCCESS"
    assert np.all(np.isfinite(q))


def _drive_unreachable_left_reach(*, allow_fallback: bool) -> tuple[float, int, str]:
    """Drive the left tool toward a far, unreachable target with both arm pose
    tasks at priority 0 (mirrors the bimanual teleop app). Returns
    (extension_m, terminal_zero_motion_steps, final_status)."""
    robot, frames, collision_urdf = _load_worker_collision_robot()
    q = robot.neutral_configuration()
    robot.update_configuration(q)

    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    configure_solver_runtime_policy(solver)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    if hasattr(solver, "set_non_worsening_collision_floor_enabled"):
        solver.set_non_worsening_collision_floor_enabled(True)
    right_task = solver.add_frame_task(
        "right_tool_pose", frames["right_tool"], embodik.TaskType.FRAME_POSE
    )
    left_task = solver.add_frame_task(
        "left_tool_pose", frames["left_tool"], embodik.TaskType.FRAME_POSE
    )
    for task in (right_task, left_task):
        task.priority = 0
        task.weight = 1.0
        task.solve_mode = embodik.TaskSolveMode.SCALE
        task.allow_min_error_fallback = bool(allow_fallback)
    posture = solver.add_posture_task("bimanual_posture")
    posture.priority = 1
    posture.weight = 1e-2
    posture.set_target_configuration(q.copy())

    exclusions = _generate_consecutive_collision_exclusions(robot, collision_urdf)
    include_pairs = _generate_common_bimanual_collision_include_pairs(
        robot, collision_urdf, exclusions
    )
    _configure_collision_constraint(
        solver,
        enabled=True,
        min_distance_m=0.035,
        max_constraints=3,
        tuning_mode="balanced",
        include_pairs=include_pairs,
        exclude_pairs=exclusions,
    )
    # The CoM balance constraint is the trigger: as the arm reaches far out, the
    # support-polygon constraint opposes the reach and the prioritized SCALE solve
    # collapses the task scale (freeze) unless the min-error fallback engages.
    if hasattr(solver, "configure_com_constraint") and "base" in frames:
        polygon = _compute_support_polygon_from_contacts(
            robot, COMMON_BIMANUAL_SUPPORT_CONTACT_FRAMES
        )
        solver.configure_com_constraint(
            support_polygon=polygon,
            margin=0.10,
            frame_name=frames["base"],
            com_vel_max=1.0,
            com_acc_max=10.0,
            use_acceleration_limits=False,
            proximity_fraction=0.05,
        )

    left0 = _frame_pose_matrix(robot, frames["left_tool"])
    right0 = _frame_pose_matrix(robot, frames["right_tool"])
    left_start = left0[:3, 3].copy()
    # A far, kinematically unreachable target: SCALE collapses the primary task
    # scale to zero here (frozen) unless the min-error fallback engages.
    target = left_start + np.array([1.2, 0.0, 0.8], dtype=float)

    opts = embodik.PositionStepOptions()
    opts.max_steps = 4
    opts.adaptive_dt = True
    opts.adaptive_dt_max_scale = 10.0
    opts.adaptive_dt_reference_distance = 0.05

    ramp, hold = 25, 15
    terminal_zero_motion = 0
    status = "UNKNOWN"
    for k in range(ramp + hold):
        frac = (k + 1) / ramp if k < ramp else 1.0
        tgt = left_start + (target - left_start) * min(frac, 1.0)
        left_pose = left0.copy()
        left_pose[:3, 3] = tgt
        targets = [
            embodik.TaskTarget("right_tool_pose", right0, 12.0, 12.0),
            embodik.TaskTarget("left_tool_pose", left_pose, 12.0, 12.0),
        ]
        result = solver.solve_position_step(q, targets, opts)
        q_next = np.asarray(getattr(result, "q_solution", q), dtype=float)
        dq = float(np.linalg.norm(q_next - q)) if q_next.shape == q.shape else -1.0
        status = result.status.name
        if k >= ramp:
            if dq < 1e-5:
                terminal_zero_motion += 1
        if q_next.shape == q.shape and np.all(np.isfinite(q_next)):
            q = q_next
        robot.update_configuration(q)

    extension = float(
        np.linalg.norm(_frame_pose_matrix(robot, frames["left_tool"])[:3, 3] - left_start)
    )
    return extension, terminal_zero_motion, status


def test_worker_unreachable_reach_does_not_freeze_with_min_error_fallback() -> None:
    # Dragging a gizmo to a far, unreachable pose drives the prioritized SCALE
    # solve to a zero task scale (the arm freezes mid-extension). With the
    # min-error fallback enabled (the bimanual teleop app's default), the step
    # degrades to MIN_ERROR and the arm keeps extending toward the target instead
    # of freezing. Guards against regressing that hardening.
    ext_off, stalls_off, _ = _drive_unreachable_left_reach(allow_fallback=False)
    ext_on, stalls_on, status_on = _drive_unreachable_left_reach(allow_fallback=True)

    assert ext_on > ext_off + 0.03, (
        f"min-error fallback should reach further toward an unreachable target "
        f"(on={ext_on*1000:.0f} mm vs off={ext_off*1000:.0f} mm)"
    )
    assert stalls_on < stalls_off, (
        f"min-error fallback should reduce zero-motion freezing "
        f"(on={stalls_on} vs off={stalls_off} terminal stalled steps)"
    )
    assert status_on == "SUCCESS"


def _drive_far_left_reach_smoothness(
    *,
    max_linear_speed: float,
    max_angular_speed: float,
    adaptive_dt_max_scale: float,
) -> dict[str, float | int | str]:
    """Drive a discontinuous far target and return smoothness metrics.

    This mirrors the common bimanual app's default SCALE_ELASTIC loop on the
    public AI Worker model. The target is intentionally unreachable; the test is
    about damping reversals and large jumps, not making the final error zero.
    """
    robot, frames, collision_urdf = _load_worker_collision_robot()
    q_lo, q_hi = robot.get_joint_limits()
    joint_name_to_cfg = {
        name: int(robot.get_joint_config_index(name)) for name in robot.get_joint_names()
    }
    q = _apply_named_joint_seed(
        robot.neutral_configuration(), joint_name_to_cfg, q_lo, q_hi, DEFAULT_COMMON_BIMANUAL_SEED
    )
    q = _apply_soft_lift_margin(q, joint_name_to_cfg=joint_name_to_cfg, q_lo=q_lo, q_hi=q_hi)
    robot.update_configuration(q)

    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    configure_solver_runtime_policy(solver)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    if hasattr(solver, "set_non_worsening_collision_floor_enabled"):
        solver.set_non_worsening_collision_floor_enabled(True)

    right_task = solver.add_frame_task(
        "right_tool_pose", frames["right_tool"], embodik.TaskType.FRAME_POSE
    )
    left_task = solver.add_frame_task(
        "left_tool_pose", frames["left_tool"], embodik.TaskType.FRAME_POSE
    )
    for task in (right_task, left_task):
        task.priority = 0
        task.weight = 1.0
        task.solve_mode = embodik.TaskSolveMode.SCALE_ELASTIC
        task.allow_min_error_fallback = True

    left_arm_v = sorted(
        int(robot.get_joint_velocity_index(n))
        for n in robot.get_joint_names()
        if n.startswith(("arm_l_", "gripper_l_"))
    )
    right_arm_v = sorted(
        int(robot.get_joint_velocity_index(n))
        for n in robot.get_joint_names()
        if n.startswith(("arm_r_", "gripper_r_"))
    )
    right_task.set_excluded_joint_indices(left_arm_v)
    left_task.set_excluded_joint_indices(right_arm_v)

    posture = solver.add_posture_task("bimanual_posture")
    posture.priority = 1
    posture.weight = 1e-2
    posture.set_target_configuration(q.copy())

    exclusions = _generate_consecutive_collision_exclusions(robot, collision_urdf)
    include_pairs = _generate_common_bimanual_collision_include_pairs(
        robot, collision_urdf, exclusions
    )
    _configure_collision_constraint(
        solver,
        enabled=True,
        min_distance_m=0.035,
        max_constraints=3,
        tuning_mode="balanced",
        include_pairs=include_pairs,
        exclude_pairs=exclusions,
    )
    if hasattr(solver, "configure_com_constraint") and "base" in frames:
        polygon = _compute_support_polygon_from_contacts(
            robot, COMMON_BIMANUAL_SUPPORT_CONTACT_FRAMES
        )
        solver.configure_com_constraint(
            support_polygon=polygon,
            margin=0.10,
            frame_name=frames["base"],
            com_vel_max=1.0,
            com_acc_max=10.0,
            use_acceleration_limits=False,
            proximity_fraction=0.05,
        )

    right0 = _frame_pose_matrix(robot, frames["right_tool"])
    left0 = _frame_pose_matrix(robot, frames["left_tool"])
    left_start = left0[:3, 3].copy()
    target = left_start + np.array([1.2, 0.0, 0.8], dtype=float)
    target_direction = (target - left_start) / float(np.linalg.norm(target - left_start))

    opts = embodik.PositionStepOptions()
    opts.max_steps = 2
    opts.position_gain = 10.0
    opts.orientation_gain = 10.0
    opts.adaptive_dt = True
    opts.adaptive_dt_max_scale = float(adaptive_dt_max_scale)
    opts.adaptive_dt_reference_distance = 0.02
    if hasattr(opts, "primary_solve_mode"):
        opts.primary_solve_mode = embodik.TaskSolveMode.SCALE_ELASTIC
        opts.primary_allow_min_error_fallback = True
    if hasattr(opts, "stall_recovery"):
        opts.stall_recovery = True
    _apply_position_step_speed_caps(
        opts,
        max_linear_speed=float(max_linear_speed),
        max_angular_speed=float(max_angular_speed),
    )

    q_trace = []
    errors = []
    progress = []
    status = "UNKNOWN"
    fallback_steps = 0
    left_pose = left0.copy()
    left_pose[:3, 3] = target
    for _ in range(80):
        result = solver.solve_position_step(
            q,
            [
                embodik.TaskTarget("right_tool_pose", right0, 10.0, 10.0),
                embodik.TaskTarget("left_tool_pose", left_pose, 10.0, 10.0),
            ],
            opts,
        )
        q_next = np.asarray(getattr(result, "q_solution", q), dtype=float)
        if q_next.shape == q.shape and np.all(np.isfinite(q_next)):
            q = q_next
        robot.update_configuration(q)
        ee_pos = _frame_pose_matrix(robot, frames["left_tool"])[:3, 3]
        errors.append(float(np.linalg.norm(target - ee_pos)))
        progress.append(float(np.dot(ee_pos - left_start, target_direction)))
        q_trace.append(q.copy())
        status = result.status.name
        diagnostics = getattr(result, "diagnostics", None)
        fallback_steps += int(bool(getattr(diagnostics, "weighted_fallback_used", False)))

    q_arr = np.vstack(q_trace)
    q_steps = np.linalg.norm(np.diff(q_arr, axis=0), axis=1)
    q_accel = np.linalg.norm(np.diff(q_arr, n=2, axis=0), axis=1)
    error_arr = np.asarray(errors, dtype=float)
    progress_arr = np.asarray(progress, dtype=float)
    return {
        "final_error": float(error_arr[-1]),
        "min_error": float(error_arr.min()),
        "error_increases": int(np.sum(np.diff(error_arr) > 1e-4)),
        "backsteps": int(np.sum(np.diff(progress_arr) < -1e-4)),
        "max_step_norm": float(q_steps.max()) if q_steps.size else 0.0,
        "p95_step_norm": float(np.percentile(q_steps, 95)) if q_steps.size else 0.0,
        "max_accel_norm": float(q_accel.max()) if q_accel.size else 0.0,
        "fallback_steps": int(fallback_steps),
        "final_status": status,
    }


def test_worker_far_target_default_speed_caps_dampen_oscillation() -> None:
    uncapped = _drive_far_left_reach_smoothness(
        max_linear_speed=0.0,
        max_angular_speed=0.0,
        adaptive_dt_max_scale=10.0,
    )
    capped = _drive_far_left_reach_smoothness(
        max_linear_speed=DEFAULT_MAX_LINEAR_SPEED,
        max_angular_speed=DEFAULT_MAX_ANGULAR_SPEED,
        adaptive_dt_max_scale=DEFAULT_ADAPTIVE_DT_MAX_SCALE,
    )

    assert uncapped["error_increases"] >= 5
    assert uncapped["backsteps"] >= 5
    assert capped["error_increases"] <= 1, capped
    assert capped["backsteps"] <= 1, capped
    assert capped["max_step_norm"] < uncapped["max_step_norm"] * 0.6
    assert capped["max_accel_norm"] < uncapped["max_accel_norm"] * 0.6
    assert capped["final_error"] <= uncapped["final_error"] + 1e-3
    assert capped["fallback_steps"] == 0
    assert capped["final_status"] == "SUCCESS"


def _held_arm_drift_when_dragging_other(*, solve_mode, protect_held: bool = False) -> float:
    """Hold the right tool target fixed while ramping the left tool through a
    CoM-shifting reach; return the max drift (m) of the held right EEF.

    When ``protect_held`` is set, the held (right) arm is given priority over the
    moving (left) arm, mirroring the app's held-arm protection."""
    robot, frames, collision_urdf = _load_worker_collision_robot()
    q = robot.neutral_configuration()
    robot.update_configuration(q)

    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    configure_solver_runtime_policy(solver)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    if hasattr(solver, "set_non_worsening_collision_floor_enabled"):
        solver.set_non_worsening_collision_floor_enabled(True)
    right_task = solver.add_frame_task(
        "right_tool_pose", frames["right_tool"], embodik.TaskType.FRAME_POSE
    )
    left_task = solver.add_frame_task(
        "left_tool_pose", frames["left_tool"], embodik.TaskType.FRAME_POSE
    )
    for task in (right_task, left_task):
        task.priority = 0
        task.weight = 1.0
        task.solve_mode = solve_mode
        task.allow_min_error_fallback = True
    if protect_held:
        # We drag the left arm below, so the right arm is the held one: give it the
        # higher priority (0) and demote the moving left arm (1) -- the app's
        # held-arm protection.
        right_task.priority = 0
        left_task.priority = 1
    posture = solver.add_posture_task("bimanual_posture")
    posture.priority = 2
    posture.weight = 1e-2
    posture.set_target_configuration(q.copy())

    # Mutual arm exclusion (as the app does): each arm task ignores the other
    # arm's joints but shares the torso/base chain -- the coupling path.
    left_arm_v = sorted(
        int(robot.get_joint_velocity_index(n))
        for n in robot.get_joint_names()
        if n.startswith(("arm_l_", "gripper_l_"))
    )
    right_arm_v = sorted(
        int(robot.get_joint_velocity_index(n))
        for n in robot.get_joint_names()
        if n.startswith(("arm_r_", "gripper_r_"))
    )
    right_task.set_excluded_joint_indices(left_arm_v)
    left_task.set_excluded_joint_indices(right_arm_v)

    exclusions = _generate_consecutive_collision_exclusions(robot, collision_urdf)
    include_pairs = _generate_common_bimanual_collision_include_pairs(
        robot, collision_urdf, exclusions
    )
    _configure_collision_constraint(
        solver,
        enabled=True,
        min_distance_m=0.035,
        max_constraints=3,
        tuning_mode="balanced",
        include_pairs=include_pairs,
        exclude_pairs=exclusions,
    )
    if hasattr(solver, "configure_com_constraint") and "base" in frames:
        polygon = _compute_support_polygon_from_contacts(
            robot, COMMON_BIMANUAL_SUPPORT_CONTACT_FRAMES
        )
        solver.configure_com_constraint(
            support_polygon=polygon,
            margin=0.10,
            frame_name=frames["base"],
            com_vel_max=1.0,
            com_acc_max=10.0,
            use_acceleration_limits=False,
            proximity_fraction=0.05,
        )

    left0 = _frame_pose_matrix(robot, frames["left_tool"])
    right0 = _frame_pose_matrix(robot, frames["right_tool"])
    left_start = left0[:3, 3].copy()
    right_held = right0[:3, 3].copy()
    delta = np.array([0.30, 0.0, -0.25], dtype=float)  # CoM-shifting reach

    opts = embodik.PositionStepOptions()
    opts.max_steps = 4
    opts.adaptive_dt = True
    opts.adaptive_dt_max_scale = 10.0
    opts.adaptive_dt_reference_distance = 0.05

    max_drift = 0.0
    for k in range(40):
        tgt = left_start + delta * min((k + 1) / 30.0, 1.0)
        left_pose = left0.copy()
        left_pose[:3, 3] = tgt
        targets = [
            embodik.TaskTarget("right_tool_pose", right0, 12.0, 12.0),  # held fixed
            embodik.TaskTarget("left_tool_pose", left_pose, 12.0, 12.0),
        ]
        result = solver.solve_position_step(q, targets, opts)
        q_next = np.asarray(getattr(result, "q_solution", q), dtype=float)
        if q_next.shape == q.shape and np.all(np.isfinite(q_next)):
            q = q_next
        robot.update_configuration(q)
        drift = float(
            np.linalg.norm(_frame_pose_matrix(robot, frames["right_tool"])[:3, 3] - right_held)
        )
        max_drift = max(max_drift, drift)
    return max_drift


def test_worker_held_arm_stays_put_in_scale_mode() -> None:
    # Cross-coupling: dragging one arm through a CoM-shifting reach must not pull
    # the held arm's EEF off its fixed target. SCALE keeps the held arm planted;
    # SCALE_ELASTIC lets it drift (the elastic relaxation trades off the held
    # task), which is why the app defaults to SCALE.
    scale_drift = _held_arm_drift_when_dragging_other(solve_mode=embodik.TaskSolveMode.SCALE)
    elastic_drift = _held_arm_drift_when_dragging_other(
        solve_mode=embodik.TaskSolveMode.SCALE_ELASTIC
    )
    assert (
        scale_drift < 5e-3
    ), f"held arm should stay put in SCALE, drifted {scale_drift*1000:.1f} mm"
    assert elastic_drift > scale_drift, (
        "SCALE should couple less than SCALE_ELASTIC "
        f"(scale {scale_drift*1000:.1f} mm vs elastic {elastic_drift*1000:.1f} mm)"
    )


def test_worker_held_arm_protection_plants_held_arm_under_scale_elastic() -> None:
    # Cross-coupling fix: SCALE_ELASTIC keeps its limit-saturation robustness but
    # couples the held arm (it drifts when the other arm moves). Giving the held
    # arm priority over the actively-dragged arm plants it -- the app's held-arm
    # protection -- without leaving SCALE_ELASTIC.
    coupled = _held_arm_drift_when_dragging_other(
        solve_mode=embodik.TaskSolveMode.SCALE_ELASTIC, protect_held=False
    )
    protected = _held_arm_drift_when_dragging_other(
        solve_mode=embodik.TaskSolveMode.SCALE_ELASTIC, protect_held=True
    )
    assert protected < 5e-3, (
        f"held-arm protection should plant the held EEF under SCALE_ELASTIC, "
        f"drifted {protected*1000:.1f} mm"
    )
    assert protected < coupled, (
        f"protection should cut coupling (protected {protected*1000:.1f} mm vs "
        f"coupled {coupled*1000:.1f} mm)"
    )


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


def test_worker_deep_penetration_stall_is_solver_visible_without_example_escape() -> None:
    robot, frames, collision_urdf = _load_worker_collision_robot()
    visual_urdf = _resolve_worker_urdf()
    q_lo, q_hi = robot.get_joint_limits()
    joint_name_to_cfg = {
        name: int(robot.get_joint_config_index(name)) for name in robot.get_joint_names()
    }
    q = _apply_named_joint_seed(
        robot.neutral_configuration(), joint_name_to_cfg, q_lo, q_hi, DEFAULT_COMMON_BIMANUAL_SEED
    )
    q = _apply_soft_lift_margin(q, joint_name_to_cfg=joint_name_to_cfg, q_lo=q_lo, q_hi=q_hi)
    robot.update_configuration(q)

    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    right_task = solver.add_frame_task(
        "right_tool_pose", frames["right_tool"], embodik.TaskType.FRAME_POSE
    )
    left_task = solver.add_frame_task(
        "left_tool_pose", frames["left_tool"], embodik.TaskType.FRAME_POSE
    )
    for task in (right_task, left_task):
        task.priority = 0
        task.weight = 1.0
        task.solve_mode = embodik.TaskSolveMode.SCALE_ELASTIC

    left_arm_velocity_indices: list[int] = []
    locked_velocity_indices: list[int] = []
    allowed_joint_names = set(default_common_bimanual_ik_joint_names(robot.get_joint_names()))
    for joint_name in robot.get_joint_names():
        idx_v = int(robot.get_joint_velocity_index(joint_name))
        nv_joint = (
            int(robot.get_joint_velocity_size(joint_name))
            if hasattr(robot, "get_joint_velocity_size")
            else 1
        )
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
    opts.integration_zero_velocity_indices = sorted(
        set(locked_velocity_indices + left_arm_velocity_indices)
    )

    start_pose = _frame_pose_matrix(robot, frames["right_tool"])
    into_target = start_pose.copy()
    into_target[:3, 3] += np.array([-0.32, 0.30, 0.0], dtype=float)

    for _step_idx in range(60):
        step = robust_solve_position_step(
            solver=solver,
            q_current=q,
            targets=[embodik.TaskTarget("right_tool_pose", into_target, 10.0, 10.0)],
            options=opts,
        )
        q = step.q_next
        robot.update_configuration(q)

    assert float(solver.evaluate_collision_debug(q).distance) <= 0.0

    exclusions = _generate_consecutive_collision_exclusions(robot, visual_urdf)
    include_pairs = _generate_common_bimanual_collision_include_pairs(
        robot, visual_urdf, exclusions
    )
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
    before_distance = float(solver.evaluate_collision_debug(q).distance)
    recovery_step = robust_solve_position_step(
        solver=solver,
        q_current=q,
        targets=[embodik.TaskTarget("right_tool_pose", away_target, 10.0, 10.0)],
        options=opts,
    )
    q_recovered = np.asarray(recovery_step.q_next, dtype=float)
    robot.update_configuration(q_recovered)
    after_distance = float(solver.evaluate_collision_debug(q_recovered).distance)

    assert recovery_step.solver_result.status.name == "SUCCESS"
    assert np.all(np.isfinite(q_recovered))
    assert after_distance >= before_distance - 1e-6


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
    )
    record = metrics.as_dict()
    assert record["collision_breach_count"] == 0
    assert record["unrecoverable_stall_count"] == 0
    assert record["min_collision_distance_m"] >= 0.034
