#!/usr/bin/env python3
"""Tests for ROBOTIS AI worker example helpers."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import embodik
from embodik.interactive_ik import configure_primary_solve_mode, robust_solve_position_step
from examples.example_helpers.common_bimanual_model_utils import (
    default_common_bimanual_allowed_joint_names,
    default_common_bimanual_ik_joint_names,
    resolve_common_bimanual_frames,
)
from examples.example_helpers.common_bimanual_teleop_app import (
    DEFAULT_RBY1_SEED,
    _apply_named_joint_seed,
    _collision_object_to_link_name,
    _configure_acceleration_limit_constraint,
    _configure_collision_constraint,
    _contains_point_in_polygon,
    _default_collision_max_constraints,
    _default_collision_min_distance_mm,
    _default_seed_for_joint_names,
    _generate_common_bimanual_collision_include_pairs,
    _generate_consecutive_collision_exclusions,
    _posture_control_joint_names,
    _resolve_torso_marker_frame,
    _resolve_torso_marker_z_bounds,
)
from examples.example_helpers.public_ai_worker_paths import resolve_public_ai_worker_urdf_paths


def _load_bimanual_example_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "06_bimanual_whole_body_ik.py"
    spec = importlib.util.spec_from_file_location("bimanual_whole_body_ik_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_rby1_description() -> None:
    try:
        importlib.import_module("robot_descriptions.rby1_description")
    except Exception as exc:
        pytest.skip(
            "robot_descriptions.rby1_description could not be resolved; "
            f"optional external asset fetch failed: {exc}"
        )


def _pose_matrix(robot, frame_name: str) -> np.ndarray:
    pose = robot.get_frame_pose(frame_name)
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = np.asarray(pose.rotation, dtype=float)
    mat[:3, 3] = np.asarray(pose.translation, dtype=float)
    return mat


def test_resolve_torso_marker_frame_prefers_ai_worker_arm_base() -> None:
    frame_map = {
        "base": "base_link",
        "arm_base": "arm_base_link",
        "head": "head_link2",
        "right_tool": "gripper_r_rh_p12_rn_base",
        "left_tool": "gripper_l_rh_p12_rn_base",
    }

    assert (
        _resolve_torso_marker_frame(
            frame_map,
            [
                "base_link",
                "lift_link",
                "lift_joint",
                "arm_base_link",
                "head_link2",
            ],
        )
        == "arm_base_link"
    )


def test_resolve_torso_marker_frame_prefers_explicit_torso_frame() -> None:
    frame_map = {
        "base": "base_link",
        "arm_base": "arm_base_link",
    }

    assert (
        _resolve_torso_marker_frame(
            frame_map,
            [
                "base_link",
                "arm_base_link",
                "torso_yaw_joint",
            ],
        )
        == "torso_yaw_joint"
    )


def test_resolve_ai_worker_torso_marker_z_bounds_from_lift_joint() -> None:
    urdf_path, collision_urdf_path = resolve_public_ai_worker_urdf_paths(variant="sg2")
    model_path = collision_urdf_path or urdf_path
    full_robot = embodik.RobotModel(str(model_path), floating_base=False)
    robot = embodik.RobotModel(
        str(model_path),
        actuated_joint_names=default_common_bimanual_ik_joint_names(full_robot.get_joint_names()),
        floating_base=False,
    )
    q0 = robot.neutral_configuration()
    frames = resolve_common_bimanual_frames(robot.get_frame_names())
    frame = _resolve_torso_marker_frame(frames, robot.get_frame_names())
    q_lo, q_hi = robot.get_joint_limits()
    joint_name_to_cfg = {
        name: int(robot.get_joint_config_index(name)) for name in robot.get_joint_names()
    }

    z_bounds = _resolve_torso_marker_z_bounds(
        robot,
        frame,
        q_ref=q0,
        joint_name_to_cfg=joint_name_to_cfg,
        q_lo=np.asarray(q_lo, dtype=float),
        q_hi=np.asarray(q_hi, dtype=float),
    )

    assert frame == "arm_base_link"
    assert z_bounds == pytest.approx((0.9316, 1.4316), abs=1e-4)


def _load_reduced_rby1_visual_robot():
    _require_rby1_description()
    mod = _load_bimanual_example_module()
    source_urdf = mod._resolve_rby1_urdf_path()
    visual_urdf = mod._prepare_rby1_tip_frame_urdf(source_urdf)
    full_robot = embodik.RobotModel(str(visual_urdf), floating_base=False)
    ik_joint_names = default_common_bimanual_ik_joint_names(full_robot.get_joint_names())
    robot = embodik.RobotModel(
        str(visual_urdf),
        actuated_joint_names=ik_joint_names,
        floating_base=False,
    )
    frames = resolve_common_bimanual_frames(robot.get_frame_names())
    q_lo, q_hi = robot.get_joint_limits()
    joint_name_to_cfg = {
        name: int(robot.get_joint_config_index(name)) for name in robot.get_joint_names()
    }
    q0 = _apply_named_joint_seed(
        robot.neutral_configuration(), joint_name_to_cfg, q_lo, q_hi, DEFAULT_RBY1_SEED
    )
    return robot, frames, np.asarray(q0, dtype=float)


def _run_limited_rby1_pose_group_case(
    *,
    mode: str,
    limit_width: float,
    target_offset: np.ndarray,
    steps: int = 40,
):
    robot, frames, q0 = _load_reduced_rby1_visual_robot()
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
    right_target = _pose_matrix(robot, frames["right_tool"])
    left_target = _pose_matrix(robot, frames["left_tool"])
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


def test_resolve_common_bimanual_frames_prefers_gripper_base_frames() -> None:
    frame_map = resolve_common_bimanual_frames(
        [
            "base_link",
            "arm_base_link",
            "head_link2",
            "arm_r_link7",
            "arm_l_link7",
            "gripper_r_rh_p12_rn_base",
            "gripper_l_rh_p12_rn_base",
        ]
    )
    assert frame_map["base"] == "base_link"
    assert frame_map["arm_base"] == "arm_base_link"
    assert frame_map["head"] == "head_link2"
    assert frame_map["right_tool"] == "gripper_r_rh_p12_rn_base"
    assert frame_map["left_tool"] == "gripper_l_rh_p12_rn_base"


def test_resolve_common_bimanual_frames_falls_back_to_link7_and_camera() -> None:
    frame_map = resolve_common_bimanual_frames(
        [
            "base_link",
            "lift_link",
            "head_link1",
            "arm_r_link7",
            "camera_l_link",
        ]
    )
    assert frame_map["arm_base"] == "lift_link"
    assert frame_map["head"] == "head_link1"
    assert frame_map["right_tool"] == "arm_r_link7"
    assert frame_map["left_tool"] == "camera_l_link"


def test_default_common_bimanual_allowed_joint_names_excludes_wheels() -> None:
    allowed = default_common_bimanual_allowed_joint_names(
        [
            "lift_joint",
            "head_joint1",
            "arm_r_joint3",
            "gripper_l_joint2",
            "left_wheel_steer",
            "rear_wheel_drive",
        ]
    )
    assert "lift_joint" in allowed
    assert "arm_r_joint3" in allowed
    assert "head_joint1" not in allowed
    assert "gripper_l_joint2" not in allowed
    assert "left_wheel_steer" not in allowed
    assert "rear_wheel_drive" not in allowed


def test_resolve_rby1_frames_for_ai_worker_style_app() -> None:
    frame_map = resolve_common_bimanual_frames(
        [
            "base",
            "link_torso_5",
            "link_head_1",
            "ee_left_tip",
            "ee_right_tip",
            "tool_left",
            "tool_right",
        ]
    )
    assert frame_map["base"] == "base"
    assert frame_map["arm_base"] == "link_torso_5"
    assert frame_map["head"] == "link_head_1"
    assert frame_map["right_tool"] == "ee_right_tip"
    assert frame_map["left_tool"] == "ee_left_tip"


def test_default_common_bimanual_allowed_joint_names_supports_rby1_reduced_ik_set() -> None:
    allowed = default_common_bimanual_allowed_joint_names(
        [
            "left_wheel",
            "right_wheel",
            "torso_0",
            "torso_5",
            "left_arm_0",
            "right_arm_6",
            "head_0",
            "gripper_finger_l1",
        ]
    )
    assert "torso_0" in allowed
    assert "torso_5" in allowed
    assert "left_arm_0" in allowed
    assert "right_arm_6" in allowed
    assert "left_wheel" not in allowed
    assert "right_wheel" not in allowed
    assert "head_0" not in allowed
    assert "gripper_finger_l1" not in allowed


def test_posture_controls_are_model_aware() -> None:
    assert _posture_control_joint_names(["lift_joint", "arm_l_joint1"]) == ["lift_joint"]
    assert _posture_control_joint_names(["torso_0", "torso_2", "left_arm_0", "right_arm_0"]) == [
        "torso_0",
        "torso_2",
    ]


def test_default_seed_is_model_aware() -> None:
    ai_worker_seed = _default_seed_for_joint_names(["lift_joint", "arm_l_joint1", "arm_r_joint1"])
    assert ai_worker_seed["lift_joint"] == -0.1

    rby1_seed = _default_seed_for_joint_names(["torso_0", "left_arm_0", "right_arm_0"])
    assert rby1_seed["torso_1"] == 0.5236
    assert rby1_seed["torso_2"] == -1.0472
    assert rby1_seed["left_arm_1"] == 0.5236
    assert rby1_seed["right_arm_1"] == -0.5236
    assert rby1_seed["left_arm_3"] == rby1_seed["right_arm_3"] == -1.5708


def test_default_collision_clearance_is_model_aware() -> None:
    assert _default_collision_min_distance_mm("sg2") == pytest.approx(35.0)
    assert _default_collision_min_distance_mm("bg2") == pytest.approx(35.0)
    assert _default_collision_min_distance_mm("rby1") == pytest.approx(20.0)


def test_default_collision_active_rows_are_model_aware() -> None:
    assert _default_collision_max_constraints("sg2") == 3
    assert _default_collision_max_constraints("bg2") == 3
    assert _default_collision_max_constraints("rby1") == 8


class _DummyCollisionConfigSolver:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def disable_stall_handler(self) -> None:
        self.calls.append("disable_stall_handler")

    def clear_collision_constraint(self) -> None:
        self.calls.append("clear_collision_constraint")

    def configure_collision_constraint(
        self,
        *,
        min_distance: float,
        max_constraints: int,
        include_pairs: list[tuple[str, str]],
        exclude_pairs: list[tuple[str, str]],
    ) -> None:
        self.calls.append(f"configure:{min_distance}:{max_constraints}")

    def enable_stall_handler(self, _min_distance: float) -> None:
        self.calls.append("enable_stall_handler")

    def configure_stall_handler(
        self, *, stall_threshold: int, restore_rate: float, floor_fraction: float
    ) -> None:
        self.calls.append("configure_stall_handler")


def test_collision_constraint_reset_mimics_toggle_before_reconfigure() -> None:
    solver = _DummyCollisionConfigSolver()
    _configure_collision_constraint(
        solver,
        enabled=True,
        min_distance_m=0.02,
        max_constraints=8,
        tuning_mode="balanced",
        include_pairs=[],
        exclude_pairs=[],
        reset_stall_state=True,
    )
    assert solver.calls[:3] == [
        "disable_stall_handler",
        "clear_collision_constraint",
        "configure:0.02:8",
    ]


class _DummyAccelerationLimitSolver:
    def __init__(self) -> None:
        self.enabled: list[bool] = []
        self.limits: list[np.ndarray] = []

    def enable_acceleration_limits(self, enabled: bool) -> None:
        self.enabled.append(bool(enabled))

    def set_acceleration_limits(self, limits: np.ndarray) -> None:
        self.limits.append(np.asarray(limits, dtype=float).copy())


def test_joint_acceleration_limit_constraint_configures_uniform_bound() -> None:
    solver = _DummyAccelerationLimitSolver()
    assert _configure_acceleration_limit_constraint(
        solver,
        enabled=True,
        nv=4,
        max_acceleration=12.0,
    )
    assert solver.enabled == [True]
    assert len(solver.limits) == 1
    assert np.allclose(solver.limits[0], np.full(4, 12.0))

    assert _configure_acceleration_limit_constraint(
        solver,
        enabled=False,
        nv=4,
        max_acceleration=12.0,
    )
    assert solver.enabled == [True, False]
    assert len(solver.limits) == 1


def test_support_polygon_gate_detects_initial_com_outside_rby1_triangle() -> None:
    poly = np.array([[0.0, 0.0], [0.228, -0.265], [0.228, 0.265]])
    assert _contains_point_in_polygon(poly, np.array([0.1, 0.0]))
    assert not _contains_point_in_polygon(poly, np.array([-0.009, -0.009]))


def test_rby1_generated_collision_urdf_has_bounded_curated_pairs() -> None:
    _require_rby1_description()
    mod = _load_bimanual_example_module()
    source_urdf = mod._resolve_rby1_urdf_path()
    visual_urdf = mod._prepare_rby1_tip_frame_urdf(source_urdf)
    collision_urdf = mod._prepare_rby1_collision_urdf(visual_urdf)

    import xml.etree.ElementTree as ET

    root = ET.parse(collision_urdf).getroot()
    collision_count = len(root.findall(".//collision"))
    box_count = len(root.findall(".//collision/geometry/box"))
    assert 40 <= collision_count <= 80
    assert box_count == collision_count
    assert len(root.findall(".//collision/geometry/mesh")) == 0

    full_robot = embodik.RobotModel(str(collision_urdf), floating_base=False)
    robot = embodik.RobotModel(
        str(collision_urdf),
        actuated_joint_names=default_common_bimanual_ik_joint_names(full_robot.get_joint_names()),
        floating_base=False,
    )
    assert robot.has_collision_geometry()
    assert len(list(robot.get_collision_pair_names())) > 0

    exclusions = _generate_consecutive_collision_exclusions(robot, collision_urdf)
    include_pairs = _generate_common_bimanual_collision_include_pairs(
        robot, collision_urdf, exclusions
    )
    assert 220 <= len(include_pairs) <= 320
    link_names = {str(link.get("name", "")) for link in root.findall("link")}
    included_link_pairs = {
        tuple(
            sorted(
                (
                    _collision_object_to_link_name(pair[0], link_names) or str(pair[0]),
                    _collision_object_to_link_name(pair[1], link_names) or str(pair[1]),
                )
            )
        )
        for pair in include_pairs
    }
    assert ("link_right_arm_6", "link_torso_5") in included_link_pairs
    assert ("link_left_arm_6", "link_torso_5") in included_link_pairs


def test_rby1_generated_collision_constraint_configures_and_evaluates() -> None:
    _require_rby1_description()
    mod = _load_bimanual_example_module()
    source_urdf = mod._resolve_rby1_urdf_path()
    visual_urdf = mod._prepare_rby1_tip_frame_urdf(source_urdf)
    collision_urdf = mod._prepare_rby1_collision_urdf(visual_urdf)

    full_robot = embodik.RobotModel(str(collision_urdf), floating_base=False)
    robot = embodik.RobotModel(
        str(collision_urdf),
        actuated_joint_names=default_common_bimanual_ik_joint_names(full_robot.get_joint_names()),
        floating_base=False,
    )
    solver = embodik.KinematicsSolver(robot)
    exclusions = _generate_consecutive_collision_exclusions(robot, collision_urdf)
    include_pairs = _generate_common_bimanual_collision_include_pairs(
        robot, collision_urdf, exclusions
    )

    solver.configure_collision_constraint(
        min_distance=0.035,
        include_pairs=include_pairs,
        exclude_pairs=exclusions,
        max_constraints=3,
    )
    q = np.asarray(robot.neutral_configuration(), dtype=float)
    debug = solver.evaluate_collision_debug(q)
    assert debug is not None
    assert np.isfinite(float(debug.distance))


def test_rby1_auto_pose_layout_keeps_bimanual_case_productive() -> None:
    offset = np.array([-0.18, 0.12, 0.03], dtype=float)

    merged_statuses, merged_layouts, merged_errors, merged_moved = (
        _run_limited_rby1_pose_group_case(mode="merged", limit_width=0.02, target_offset=offset)
    )
    split_statuses, split_layouts, _split_errors, split_moved = _run_limited_rby1_pose_group_case(
        mode="split", limit_width=0.02, target_offset=offset
    )
    auto_statuses, auto_layouts, auto_errors, auto_moved = _run_limited_rby1_pose_group_case(
        mode="auto", limit_width=0.02, target_offset=offset
    )

    assert all(layout == embodik.TaskLayout.MERGED for layout in merged_layouts)
    assert all(layout == embodik.TaskLayout.SPLIT for layout in split_layouts)
    assert auto_layouts[0] == embodik.TaskLayout.SPLIT
    assert all(layout == embodik.TaskLayout.SPLIT for layout in auto_layouts)
    assert split_statuses[-1] == embodik.SolverStatus.SUCCESS
    assert auto_statuses[-1] == embodik.SolverStatus.SUCCESS
    if merged_statuses[-1] != embodik.SolverStatus.SUCCESS:
        assert merged_moved < 25
    assert split_moved >= 35
    assert auto_moved >= 35
    assert auto_moved >= split_moved
    assert auto_errors[-1] <= merged_errors[-1] + 0.02


def test_rby1_collision_push_release_uses_hardening_policy_without_stall_lock() -> None:
    _require_rby1_description()
    mod = _load_bimanual_example_module()
    source_urdf = mod._resolve_rby1_urdf_path()
    visual_urdf = mod._prepare_rby1_tip_frame_urdf(source_urdf)
    collision_urdf = mod._prepare_rby1_collision_urdf(visual_urdf)

    full_robot = embodik.RobotModel(str(collision_urdf), floating_base=False)
    ik_joint_names = default_common_bimanual_ik_joint_names(full_robot.get_joint_names())
    robot = embodik.RobotModel(
        str(collision_urdf),
        actuated_joint_names=ik_joint_names,
        floating_base=False,
    )
    frames = resolve_common_bimanual_frames(robot.get_frame_names())
    q_lo, q_hi = robot.get_joint_limits()
    joint_name_to_cfg = {
        name: int(robot.get_joint_config_index(name)) for name in robot.get_joint_names()
    }
    q0 = _apply_named_joint_seed(
        robot.neutral_configuration(), joint_name_to_cfg, q_lo, q_hi, DEFAULT_RBY1_SEED
    )
    robot.update_configuration(q0)

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
    posture = solver.add_posture_task("posture")
    posture.priority = 1
    posture.weight = 1e-2
    posture.set_target_configuration(q0.copy())
    arm_nullspace = solver.add_posture_task("arm_nullspace")
    arm_nullspace.priority = 1
    arm_nullspace.weight = 1.0
    arm_nullspace.set_target_configuration(q0.copy())

    exclusions = _generate_consecutive_collision_exclusions(robot, collision_urdf)
    include_pairs = _generate_common_bimanual_collision_include_pairs(
        robot, collision_urdf, exclusions
    )
    min_distance_m = _default_collision_min_distance_mm("rby1") * 1e-3
    _configure_collision_constraint(
        solver,
        enabled=True,
        min_distance_m=min_distance_m,
        max_constraints=_default_collision_max_constraints("rby1"),
        tuning_mode="balanced",
        include_pairs=include_pairs,
        exclude_pairs=exclusions,
    )
    assert solver.stall_handler_enabled()

    locked_velocity_indices: list[int] = []
    left_arm_velocity_indices: list[int] = []
    right_arm_velocity_indices: list[int] = []
    for joint_name in robot.get_joint_names():
        idx_v = int(robot.get_joint_velocity_index(joint_name))
        nv_joint = (
            int(robot.get_joint_velocity_size(joint_name))
            if hasattr(robot, "get_joint_velocity_size")
            else 1
        )
        for offset in range(max(nv_joint, 1)):
            vi = idx_v + offset
            if joint_name.startswith(("left_arm_", "gripper_finger_l")):
                left_arm_velocity_indices.append(vi)
            if joint_name.startswith(("right_arm_", "gripper_finger_r")):
                right_arm_velocity_indices.append(vi)
            if joint_name not in ik_joint_names:
                locked_velocity_indices.append(vi)
    right_task.set_excluded_joint_indices(sorted(set(left_arm_velocity_indices)))
    left_task.set_excluded_joint_indices(sorted(set(right_arm_velocity_indices)))

    opts = embodik.PositionStepOptions()
    opts.max_steps = 1
    opts.position_gain = 10.0
    opts.orientation_gain = 10.0
    opts.stall_recovery = True
    opts.excluded_joint_indices = list(locked_velocity_indices)
    opts.integration_zero_velocity_indices = list(locked_velocity_indices)
    configure_primary_solve_mode(opts, embodik.TaskSolveMode.SCALE_ELASTIC, False)

    def _pose(frame_name: str) -> np.ndarray:
        pose = robot.get_frame_pose(frame_name)
        mat = np.eye(4, dtype=float)
        mat[:3, :3] = np.asarray(pose.rotation, dtype=float)
        mat[:3, 3] = np.asarray(pose.translation, dtype=float)
        return mat

    right_start = _pose(frames["right_tool"])
    left_start = _pose(frames["left_tool"])
    right_push = right_start.copy()
    right_push[:3, 3] = right_start[:3, 3] + 1.25 * (left_start[:3, 3] - right_start[:3, 3])

    q = q0.copy()
    stalled_frames = 0
    late_stalled_frames = 0
    moved_frames = 0
    release_moved_frames = 0
    for step_idx in range(80):
        right_target = right_push if step_idx < 40 else right_start
        step = robust_solve_position_step(
            solver=solver,
            q_current=q,
            targets=[
                embodik.TaskTarget("right_tool_pose", right_target, 10.0, 10.0),
                embodik.TaskTarget("left_tool_pose", left_start, 10.0, 10.0),
            ],
            options=opts,
        )
        dq = float(
            np.linalg.norm(np.asarray(step.q_next, dtype=float) - np.asarray(q, dtype=float))
        )
        q = np.asarray(step.q_next, dtype=float)
        robot.update_configuration(q)
        current_min = float(solver.evaluate_collision_debug(q).distance)
        right_err = float(
            np.linalg.norm(
                right_target[:3, 3]
                - np.asarray(robot.get_frame_pose(frames["right_tool"]).translation)
            )
        )
        if dq <= 1e-8 and right_err > 1e-4:
            stalled_frames += 1
            if step_idx >= 40:
                late_stalled_frames += 1
        if dq > 1e-9:
            moved_frames += 1
            if step_idx >= 40:
                release_moved_frames += 1

    assert stalled_frames < 45
    assert late_stalled_frames <= 3
    assert moved_frames >= 35
    assert release_moved_frames >= 35
    assert float(solver.evaluate_collision_debug(q).distance) > min_distance_m


def test_rby1_repeated_collision_entry_release_uses_solver_owned_recovery() -> None:
    """Repeated push/release cycles should stay bounded without app-side escape bursts.

    This intentionally exercises the raw solver integration path used by the
    example. Release motion should come from solver-owned constrained recovery,
    not from clearing collision constraints in the UI loop.
    """
    _require_rby1_description()
    mod = _load_bimanual_example_module()
    source_urdf = mod._resolve_rby1_urdf_path()
    visual_urdf = mod._prepare_rby1_tip_frame_urdf(source_urdf)
    collision_urdf = mod._prepare_rby1_collision_urdf(visual_urdf)

    full_robot = embodik.RobotModel(str(collision_urdf), floating_base=False)
    ik_joint_names = default_common_bimanual_ik_joint_names(full_robot.get_joint_names())
    robot = embodik.RobotModel(
        str(collision_urdf),
        actuated_joint_names=ik_joint_names,
        floating_base=False,
    )
    frames = resolve_common_bimanual_frames(robot.get_frame_names())
    q_lo, q_hi = robot.get_joint_limits()
    joint_name_to_cfg = {
        name: int(robot.get_joint_config_index(name)) for name in robot.get_joint_names()
    }
    q0 = _apply_named_joint_seed(
        robot.neutral_configuration(), joint_name_to_cfg, q_lo, q_hi, DEFAULT_RBY1_SEED
    )
    robot.update_configuration(q0)

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
    posture = solver.add_posture_task("posture")
    posture.priority = 1
    posture.weight = 1e-2
    posture.set_target_configuration(q0.copy())
    arm_nullspace = solver.add_posture_task("arm_nullspace")
    arm_nullspace.priority = 1
    arm_nullspace.weight = 1.0
    arm_nullspace.set_target_configuration(q0.copy())

    exclusions = _generate_consecutive_collision_exclusions(robot, collision_urdf)
    include_pairs = _generate_common_bimanual_collision_include_pairs(
        robot, collision_urdf, exclusions
    )
    min_distance_m = _default_collision_min_distance_mm("rby1") * 1e-3
    _configure_collision_constraint(
        solver,
        enabled=True,
        min_distance_m=min_distance_m,
        max_constraints=_default_collision_max_constraints("rby1"),
        tuning_mode="balanced",
        include_pairs=include_pairs,
        exclude_pairs=exclusions,
    )

    locked_velocity_indices: list[int] = []
    left_arm_velocity_indices: list[int] = []
    right_arm_velocity_indices: list[int] = []
    for joint_name in robot.get_joint_names():
        idx_v = int(robot.get_joint_velocity_index(joint_name))
        nv_joint = (
            int(robot.get_joint_velocity_size(joint_name))
            if hasattr(robot, "get_joint_velocity_size")
            else 1
        )
        for offset in range(max(nv_joint, 1)):
            vi = idx_v + offset
            if joint_name.startswith(("left_arm_", "gripper_finger_l")):
                left_arm_velocity_indices.append(vi)
            if joint_name.startswith(("right_arm_", "gripper_finger_r")):
                right_arm_velocity_indices.append(vi)
            if joint_name not in ik_joint_names:
                locked_velocity_indices.append(vi)
    right_task.set_excluded_joint_indices(sorted(set(left_arm_velocity_indices)))
    left_task.set_excluded_joint_indices(sorted(set(right_arm_velocity_indices)))

    opts = embodik.PositionStepOptions()
    opts.max_steps = 1
    opts.position_gain = 10.0
    opts.orientation_gain = 10.0
    opts.stall_recovery = True
    opts.excluded_joint_indices = list(locked_velocity_indices)
    opts.integration_zero_velocity_indices = list(locked_velocity_indices)
    configure_primary_solve_mode(opts, embodik.TaskSolveMode.SCALE_ELASTIC, False)

    def _pose(frame_name: str) -> np.ndarray:
        pose = robot.get_frame_pose(frame_name)
        mat = np.eye(4, dtype=float)
        mat[:3, :3] = np.asarray(pose.rotation, dtype=float)
        mat[:3, 3] = np.asarray(pose.translation, dtype=float)
        return mat

    right_start = _pose(frames["right_tool"])
    left_start = _pose(frames["left_tool"])
    right_push = right_start.copy()
    right_push[:3, 3] = right_start[:3, 3] + 1.25 * (left_start[:3, 3] - right_start[:3, 3])

    q = q0.copy()
    release_stalled_frames = 0
    release_moved_frames = 0
    release_window_moves = [0, 0, 0]
    reached_collision_boundary = False
    min_distance_seen = float("inf")
    for step_idx in range(72):
        cycle_idx = step_idx // 24
        phase_idx = step_idx % 24
        releasing = phase_idx >= 12
        right_target = right_start if releasing else right_push
        step = robust_solve_position_step(
            solver=solver,
            q_current=q,
            targets=[
                embodik.TaskTarget("right_tool_pose", right_target, 10.0, 10.0),
                embodik.TaskTarget("left_tool_pose", left_start, 10.0, 10.0),
            ],
            options=opts,
        )
        dq = float(
            np.linalg.norm(np.asarray(step.q_next, dtype=float) - np.asarray(q, dtype=float))
        )
        q = np.asarray(step.q_next, dtype=float)
        robot.update_configuration(q)
        current_min = float(solver.evaluate_collision_debug(q).distance)
        min_distance_seen = min(min_distance_seen, current_min)
        reached_collision_boundary = (
            reached_collision_boundary or current_min <= min_distance_m + 5e-4
        )

        right_err = float(
            np.linalg.norm(
                right_target[:3, 3]
                - np.asarray(robot.get_frame_pose(frames["right_tool"]).translation)
            )
        )
        if releasing:
            if dq > 1e-9:
                release_moved_frames += 1
                if phase_idx < 18:
                    release_window_moves[cycle_idx] += 1
            elif right_err > 1e-4:
                release_stalled_frames += 1

    assert (
        reached_collision_boundary
    ), f"test did not reach the collision boundary; min={min_distance_seen:.4f}"
    assert min_distance_seen >= min_distance_m - 1e-4
    assert release_stalled_frames <= 18
    assert release_moved_frames >= 20
    assert release_window_moves[0] >= 3
    assert release_window_moves[1] >= 1
    assert release_window_moves[2] >= 1
    assert float(solver.evaluate_collision_debug(q).distance) > min_distance_m


def test_rby1_compact_dual_target_drag_stays_productive_near_collision() -> None:
    """Both active targets near the torso should not collapse into zero motion."""
    _require_rby1_description()
    mod = _load_bimanual_example_module()
    source_urdf = mod._resolve_rby1_urdf_path()
    visual_urdf = mod._prepare_rby1_tip_frame_urdf(source_urdf)
    collision_urdf = mod._prepare_rby1_collision_urdf(visual_urdf)

    full_robot = embodik.RobotModel(str(collision_urdf), floating_base=False)
    ik_joint_names = default_common_bimanual_ik_joint_names(full_robot.get_joint_names())
    robot = embodik.RobotModel(
        str(collision_urdf),
        actuated_joint_names=ik_joint_names,
        floating_base=False,
    )
    frames = resolve_common_bimanual_frames(robot.get_frame_names())
    q_lo, q_hi = robot.get_joint_limits()
    joint_name_to_cfg = {
        name: int(robot.get_joint_config_index(name)) for name in robot.get_joint_names()
    }
    q0 = _apply_named_joint_seed(
        robot.neutral_configuration(), joint_name_to_cfg, q_lo, q_hi, DEFAULT_RBY1_SEED
    )
    robot.update_configuration(q0)

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
    posture = solver.add_posture_task("posture")
    posture.priority = 1
    posture.weight = 1e-2
    posture.set_target_configuration(q0.copy())
    arm_nullspace = solver.add_posture_task("arm_nullspace")
    arm_nullspace.priority = 1
    arm_nullspace.weight = 1.0
    arm_nullspace.set_target_configuration(q0.copy())

    exclusions = _generate_consecutive_collision_exclusions(robot, collision_urdf)
    include_pairs = _generate_common_bimanual_collision_include_pairs(
        robot, collision_urdf, exclusions
    )
    min_distance_m = _default_collision_min_distance_mm("rby1") * 1e-3
    _configure_collision_constraint(
        solver,
        enabled=True,
        min_distance_m=min_distance_m,
        max_constraints=_default_collision_max_constraints("rby1"),
        tuning_mode="balanced",
        include_pairs=include_pairs,
        exclude_pairs=exclusions,
        reset_stall_state=True,
    )

    locked_velocity_indices: list[int] = []
    left_arm_velocity_indices: list[int] = []
    right_arm_velocity_indices: list[int] = []
    for joint_name in robot.get_joint_names():
        idx_v = int(robot.get_joint_velocity_index(joint_name))
        nv_joint = (
            int(robot.get_joint_velocity_size(joint_name))
            if hasattr(robot, "get_joint_velocity_size")
            else 1
        )
        for offset in range(max(nv_joint, 1)):
            vi = idx_v + offset
            if joint_name.startswith(("left_arm_", "gripper_finger_l")):
                left_arm_velocity_indices.append(vi)
            if joint_name.startswith(("right_arm_", "gripper_finger_r")):
                right_arm_velocity_indices.append(vi)
            if joint_name not in ik_joint_names:
                locked_velocity_indices.append(vi)
    right_task.set_excluded_joint_indices(sorted(set(left_arm_velocity_indices)))
    left_task.set_excluded_joint_indices(sorted(set(right_arm_velocity_indices)))

    opts = embodik.PositionStepOptions()
    opts.max_steps = 1
    opts.position_gain = 10.0
    opts.orientation_gain = 10.0
    opts.stall_recovery = True
    opts.excluded_joint_indices = list(locked_velocity_indices)
    opts.integration_zero_velocity_indices = list(locked_velocity_indices)
    configure_primary_solve_mode(opts, embodik.TaskSolveMode.SCALE_ELASTIC, False)

    def _pose(frame_name: str) -> np.ndarray:
        pose = robot.get_frame_pose(frame_name)
        mat = np.eye(4, dtype=float)
        mat[:3, :3] = np.asarray(pose.rotation, dtype=float)
        mat[:3, 3] = np.asarray(pose.translation, dtype=float)
        return mat

    torso = _pose(frames["arm_base"])
    compact_center = torso[:3, 3] + np.array([0.20, 0.0, 0.03], dtype=float)
    right_target = _pose(frames["right_tool"])
    left_target = _pose(frames["left_tool"])
    right_target[:3, 3] = compact_center + np.array([0.0, -0.04, 0.0], dtype=float)
    left_target[:3, 3] = compact_center + np.array([0.0, 0.04, 0.0], dtype=float)
    assert np.linalg.norm(right_target[:3, 3] - left_target[:3, 3]) == pytest.approx(0.08)

    q = q0.copy()
    moved_frames = 0
    unresolved_zero_motion_frames = 0
    last20_dq: list[float] = []
    min_distance_seen = float("inf")
    status_counts: dict[str, int] = {}
    for _step_idx in range(120):
        step = robust_solve_position_step(
            solver=solver,
            q_current=q,
            targets=[
                embodik.TaskTarget("right_tool_pose", right_target, 10.0, 10.0),
                embodik.TaskTarget("left_tool_pose", left_target, 10.0, 10.0),
            ],
            options=opts,
        )
        q_prev = q
        q = np.asarray(step.q_next, dtype=float)
        robot.update_configuration(q)

        dq = float(np.linalg.norm(q - q_prev))
        if dq > 1e-9:
            moved_frames += 1
        last20_dq.append(dq)
        last20_dq = last20_dq[-20:]

        status_name = getattr(getattr(step.solver_result, "status", None), "name", "")
        status_counts[status_name] = status_counts.get(status_name, 0) + 1

        current_min = float(solver.evaluate_collision_debug(q).distance)
        min_distance_seen = min(min_distance_seen, current_min)
        max_target_error = max(
            float(
                np.linalg.norm(
                    right_target[:3, 3]
                    - np.asarray(robot.get_frame_pose(frames["right_tool"]).translation)
                )
            ),
            float(
                np.linalg.norm(
                    left_target[:3, 3]
                    - np.asarray(robot.get_frame_pose(frames["left_tool"]).translation)
                )
            ),
        )
        if dq <= 1e-8 and max_target_error > 0.02:
            unresolved_zero_motion_frames += 1

    assert min_distance_seen < min_distance_m + 0.005
    assert unresolved_zero_motion_frames <= 25, status_counts
    assert moved_frames >= 60, status_counts
    assert sum(dq > 1e-9 for dq in last20_dq) >= 8, status_counts
    assert float(solver.evaluate_collision_debug(q).distance) >= -1e-3
