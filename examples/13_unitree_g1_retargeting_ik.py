#!/usr/bin/env python3
"""Unitree G1 retargeting and collision-free IK demo.

This example mirrors the cuRobo G1 framing at an EmbodiK scale: hands and
feet are explicit target frames, collision and CoM are opt-in constraints,
and synthetic keyframe clips drive the same whole-body IK loop as the
interactive gizmos.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import embodik
import numpy as np
from embodik.interactive_ik import (
    ConstrainedStepGuard,
    ConstraintBoundary,
    clear_all_target_velocities_if_available,
    joint_velocity_norm,
)
from embodik.utils import r2q

EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

try:
    from example_helpers.g1_ik_runtime import (
        _apply_g1_soft_knee_seed,
        _apply_zero_based_task_hierarchy,
        _attempt_g1_penetration_escape_burst,
        _clear_interactive_solver_transients,
        _clip_q,
        _configure_g1_collision_constraint,
        _configure_g1_posture_task,
        _configure_interactive_elastic_band,
        _current_collision_min_distance,
        _enum_names,
        _frame_delta6,
        _g1_interactive_hold_error_threshold,
        _limit_tangent_step,
        _max_frame_target_position_error,
        _pose_from_ctrl,
        _pose_signature,
        _post_step_collision_distance,
        _set_ctrl_from_pose,
        _sleep_for_loop_rate,
        _solve_quality_step,
        _target_solve_mode,
    )
    from example_helpers.g1_model_utils import (
        build_retargeting_delta_target_poses,
        com_min_slack,
        com_slack_color,
        compute_feet_center,
        compute_support_polygon_from_foot_poses,
        create_g1_robot_and_visual,
        g1_collision_pair_preset_options,
        g1_collision_pairs_for_preset,
        get_retargeting_clip_duration,
        get_retargeting_clips,
        get_retargeting_presets,
        ground_floating_base_from_feet_center,
        make_visual_config_mapper,
        polygon_segments_xy,
        prepare_g1_viewer_urdf_path,
        resolve_frames_for_g1_base_mode,
        resolve_g1_collision_urdf_path,
        sample_retargeting_clip,
        shrink_polygon_xy,
    )
except ModuleNotFoundError as exc:
    if exc.name != "example_helpers" and not str(exc.name).startswith("example_helpers."):
        raise
    from examples.example_helpers.g1_ik_runtime import (
        _apply_g1_soft_knee_seed,
        _apply_zero_based_task_hierarchy,
        _attempt_g1_penetration_escape_burst,
        _clear_interactive_solver_transients,
        _clip_q,
        _configure_g1_collision_constraint,
        _configure_g1_posture_task,
        _configure_interactive_elastic_band,
        _current_collision_min_distance,
        _enum_names,
        _frame_delta6,
        _g1_interactive_hold_error_threshold,
        _limit_tangent_step,
        _max_frame_target_position_error,
        _pose_from_ctrl,
        _pose_signature,
        _post_step_collision_distance,
        _set_ctrl_from_pose,
        _sleep_for_loop_rate,
        _solve_quality_step,
        _target_solve_mode,
    )
    from examples.example_helpers.g1_model_utils import (
        build_retargeting_delta_target_poses,
        com_min_slack,
        com_slack_color,
        compute_feet_center,
        compute_support_polygon_from_foot_poses,
        create_g1_robot_and_visual,
        g1_collision_pair_preset_options,
        g1_collision_pairs_for_preset,
        get_retargeting_clip_duration,
        get_retargeting_clips,
        get_retargeting_presets,
        ground_floating_base_from_feet_center,
        make_visual_config_mapper,
        polygon_segments_xy,
        prepare_g1_viewer_urdf_path,
        resolve_frames_for_g1_base_mode,
        resolve_g1_collision_urdf_path,
        sample_retargeting_clip,
        shrink_polygon_xy,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8087)
    p.add_argument("--headless-smoke-steps", type=int, default=0)
    p.add_argument("--headless-ik-smoke-steps", type=int, default=0)
    p.add_argument("--headless-gizmo-stress-steps", type=int, default=0)
    p.add_argument("--headless-reset-interval", type=int, default=0)
    p.add_argument("--headless-reset-smoke", action="store_true")
    p.add_argument("--headless-com-smoke", action="store_true")
    p.add_argument("--headless-left-ee-oscillation-steps", type=int, default=0)
    p.add_argument("--headless-single-target-oscillation-steps", type=int, default=0)
    p.add_argument(
        "--headless-single-target",
        choices=("right_palm", "left_palm", "right_ankle", "left_ankle", "pelvis", "all"),
        default="all",
    )
    p.add_argument(
        "--full-ik-model",
        action="store_true",
        help="Use the full G1 model including finger joints instead of the reduced IK model.",
    )
    p.add_argument(
        "--performance-mode",
        action="store_true",
        help="Use latency-oriented IK settings: one step, direct solve, light posture.",
    )
    p.add_argument(
        "--quality-mode",
        action="store_true",
        help="Start the UI in quality mode instead of the default one-solve performance path.",
    )
    return p.parse_args()


def _import_g1_harnesses():
    try:
        return importlib.import_module("harnesses.g1_headless_harnesses")
    except ModuleNotFoundError as exc:
        if exc.name != "harnesses" and not str(exc.name).startswith("harnesses."):
            raise
        return importlib.import_module("examples.harnesses.g1_headless_harnesses")


def main() -> None:
    args = parse_args()
    if args.headless_smoke_steps > 0:
        _import_g1_harnesses().run_headless_smoke(args.headless_smoke_steps)
        return
    if args.headless_ik_smoke_steps > 0:
        _import_g1_harnesses().run_headless_ik_smoke(
            args.headless_ik_smoke_steps, args.headless_reset_interval
        )
        return

    if args.headless_gizmo_stress_steps > 0:
        _import_g1_harnesses().run_headless_ik_smoke(
            args.headless_gizmo_stress_steps, args.headless_reset_interval
        )
        return
    if args.headless_reset_smoke:
        _import_g1_harnesses().run_headless_reset_smoke()
        return
    if args.headless_com_smoke:
        _import_g1_harnesses().run_headless_com_smoke()
        return
    if args.headless_left_ee_oscillation_steps > 0:
        _import_g1_harnesses().run_headless_left_ee_oscillation(
            args.headless_left_ee_oscillation_steps
        )
        return
    if args.headless_single_target_oscillation_steps > 0:
        harnesses = _import_g1_harnesses()

        target_keys = (
            harnesses._SINGLE_TARGET_KEYS
            if args.headless_single_target == "all"
            else (args.headless_single_target,)
        )
        for target_key in target_keys:
            harnesses.run_headless_single_target_oscillation(
                target_key, args.headless_single_target_oscillation_steps
            )
        return

    robot, server, urdf_vis = create_g1_robot_and_visual(
        floating_base=True,
        port=args.port,
        root_node_name="/floating_base/robot",
        reduced_ik=not bool(args.full_ik_model),
    )
    import yourdfpy
    from viser.extras import ViserUrdf

    collision_urdf_path = resolve_g1_collision_urdf_path()
    collision_viewer_urdf_path = prepare_g1_viewer_urdf_path(collision_urdf_path)
    urdf_collision_vis = ViserUrdf(
        server,
        yourdfpy.URDF.load(
            str(collision_viewer_urdf_path),
            build_scene_graph=False,
            build_collision_scene_graph=True,
            load_meshes=False,
            load_collision_meshes=True,
        ),
        root_node_name="/floating_base/robot_collision",
        load_meshes=False,
        load_collision_meshes=True,
    )
    base_node = server.scene.add_frame("/floating_base", show_axes=False)

    q = _apply_g1_soft_knee_seed(robot, robot.neutral_configuration())
    q_lo, q_hi = robot.get_joint_limits()
    robot.update_configuration(q)
    frame_map = resolve_frames_for_g1_base_mode(robot.get_frame_names())
    upright_frame = "pelvis" if "pelvis" in robot.get_frame_names() else frame_map["imu_in_torso"]

    r0 = np.asarray(robot.get_frame_pose(frame_map["right_ankle"]).translation, dtype=float)
    l0 = np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).translation, dtype=float)
    q = ground_floating_base_from_feet_center(q, r0, l0)
    q = _clip_q(robot, q, q_lo, q_hi)
    robot.update_configuration(q)
    r1 = np.asarray(robot.get_frame_pose(frame_map["right_ankle"]).translation, dtype=float)
    l1 = np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).translation, dtype=float)
    q[2] += 0.03 - float(min(r1[2], l1[2]))
    q = _clip_q(robot, q, q_lo, q_hi)
    robot.update_configuration(q)

    map_q = make_visual_config_mapper(robot, urdf_vis)
    map_q_collision = make_visual_config_mapper(robot, urdf_collision_vis)
    urdf_vis.update_cfg(map_q(q))
    urdf_collision_vis.update_cfg(map_q_collision(q))

    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)
    solver.set_tolerance(0.1)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    collision_available = (
        hasattr(solver, "configure_collision_constraint")
        and hasattr(robot, "has_collision_geometry")
        and bool(robot.has_collision_geometry())
    )

    right_task = solver.add_frame_task(
        "right_palm_pose", frame_map["right_palm"], embodik.TaskType.FRAME_POSE
    )
    left_task = solver.add_frame_task(
        "left_palm_pose", frame_map["left_palm"], embodik.TaskType.FRAME_POSE
    )
    right_foot_task = solver.add_frame_task(
        "right_ankle_pose", frame_map["right_ankle"], embodik.TaskType.FRAME_POSE
    )
    left_foot_task = solver.add_frame_task(
        "left_ankle_pose", frame_map["left_ankle"], embodik.TaskType.FRAME_POSE
    )
    pelvis_task = solver.add_frame_task("pelvis_pose", upright_frame, embodik.TaskType.FRAME_POSE)
    posture = solver.add_posture_task("posture")
    torso_ori = solver.add_frame_task(
        "torso_upright_ori", frame_map["imu_in_torso"], embodik.TaskType.FRAME_ORIENTATION
    )
    _apply_zero_based_task_hierarchy(
        primary_tasks=[right_task, left_task, right_foot_task, left_foot_task, pelvis_task],
        secondary_tasks=[torso_ori],
        tertiary_tasks=[posture],
    )
    posture.weight = 0.02
    _configure_g1_posture_task(posture, robot, q)

    torso_target = np.asarray(
        robot.get_frame_pose(frame_map["imu_in_torso"]).homogeneous(), dtype=float
    )
    retarget_anchor = np.asarray(
        robot.get_frame_pose(frame_map["imu_in_torso"]).homogeneous(), dtype=float
    )
    retarget_reference_poses = {
        "right_palm": np.asarray(
            robot.get_frame_pose(frame_map["right_palm"]).homogeneous(), dtype=float
        ),
        "left_palm": np.asarray(
            robot.get_frame_pose(frame_map["left_palm"]).homogeneous(), dtype=float
        ),
        "right_ankle": np.asarray(
            robot.get_frame_pose(frame_map["right_ankle"]).homogeneous(), dtype=float
        ),
        "left_ankle": np.asarray(
            robot.get_frame_pose(frame_map["left_ankle"]).homogeneous(), dtype=float
        ),
        "imu_in_torso": retarget_anchor.copy(),
    }
    retarget_neutral_offsets = get_retargeting_presets()["neutral"]
    q_initial = np.asarray(q, dtype=float).copy()
    retarget_clips = get_retargeting_clips()
    retarget_clip_names = tuple(retarget_clips.keys())

    def _mk_gizmo(key: str, scale: float):
        pose = robot.get_frame_pose(frame_map[key])
        q_xyzw = r2q(np.asarray(pose.rotation, dtype=float), order="xyzs")
        return server.scene.add_transform_controls(
            f"/target/{key}",
            scale=scale,
            position=tuple(np.asarray(pose.translation, dtype=float)),
            wxyz=(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]),
        )

    right_ctrl = _mk_gizmo("right_palm", 0.16)
    left_ctrl = _mk_gizmo("left_palm", 0.16)
    right_foot_ctrl = _mk_gizmo("right_ankle", 0.11)
    left_foot_ctrl = _mk_gizmo("left_ankle", 0.11)
    pelvis_pose = robot.get_frame_pose(upright_frame)
    pelvis_q_xyzw = r2q(np.asarray(pelvis_pose.rotation, dtype=float), order="xyzs")
    pelvis_ctrl = server.scene.add_transform_controls(
        "/target/pelvis",
        scale=0.14,
        position=tuple(np.asarray(pelvis_pose.translation, dtype=float)),
        wxyz=(pelvis_q_xyzw[3], pelvis_q_xyzw[0], pelvis_q_xyzw[1], pelvis_q_xyzw[2]),
    )
    target_control_by_key = {
        "right_palm": right_ctrl,
        "left_palm": left_ctrl,
        "right_ankle": right_foot_ctrl,
        "left_ankle": left_foot_ctrl,
        "pelvis": pelvis_ctrl,
    }
    target_frame_by_key = {
        "right_palm": frame_map["right_palm"],
        "left_palm": frame_map["left_palm"],
        "right_ankle": frame_map["right_ankle"],
        "left_ankle": frame_map["left_ankle"],
        "pelvis": upright_frame,
    }
    task_by_key = {
        "right_palm": right_task,
        "left_palm": left_task,
        "right_ankle": right_foot_task,
        "left_ankle": left_foot_task,
        "pelvis": pelvis_task,
    }
    task_name_by_key = {
        "right_palm": "right_palm_pose",
        "left_palm": "left_palm_pose",
        "right_ankle": "right_ankle_pose",
        "left_ankle": "left_ankle_pose",
        "pelvis": "pelvis_pose",
    }
    ordered_target_keys = ("right_palm", "left_palm", "right_ankle", "left_ankle", "pelvis")
    with server.gui.add_folder("IK Controls"):
        target_enabled = {
            "right_palm": server.gui.add_checkbox("Enable right palm target", initial_value=True),
            "left_palm": server.gui.add_checkbox("Enable left palm target", initial_value=True),
            "right_ankle": server.gui.add_checkbox("Enable right foot target", initial_value=True),
            "left_ankle": server.gui.add_checkbox("Enable left foot target", initial_value=True),
            "pelvis": server.gui.add_checkbox("Enable pelvis target", initial_value=True),
        }
        steps = server.gui.add_slider("IK Steps", 1, 12, 1, 4)
        target_pos_gain = server.gui.add_slider("Target Position Gain", 1.0, 80.0, 0.5, 16.0)
        target_ori_gain = server.gui.add_slider("Target Orientation Gain", 0.1, 80.0, 0.1, 8.0)
        target_solve_mode = server.gui.add_dropdown(
            "Target solve mode",
            options=("SCALE_ELASTIC", "MIN_ERROR", "SCALE"),
            initial_value="SCALE_ELASTIC",
        )
        enable_torso_upright = server.gui.add_checkbox("Enable torso upright", initial_value=True)
        torso_upright_gain = server.gui.add_slider("Torso upright gain", 0.0, 30.0, 0.1, 2.0)
        posture_bias_weight = server.gui.add_slider(
            "Nullspace posture bias weight", 0.0, 0.2, 0.001, 0.002
        )
        posture_update_period = server.gui.add_slider("Posture update period", 1, 30, 1, 8)
        recapture_bias = server.gui.add_button("Recapture posture bias target (current q)")
        reset_all = server.gui.add_button("Reset to initial configuration")

    with server.gui.add_folder("Geometry Visualization"):
        geometry_view = server.gui.add_dropdown(
            "Robot Geometry",
            options=("Visual", "Collision", "Both"),
            initial_value="Visual",
        )
        server.gui.add_text(
            "Collision URDF",
            initial_value=collision_urdf_path.name,
        )

    with server.gui.add_folder("Synthetic Retargeting"):
        retarget_enable = server.gui.add_checkbox("Playback enabled", initial_value=False)
        retarget_clip = server.gui.add_dropdown(
            "Clip",
            options=retarget_clip_names,
            initial_value=retarget_clip_names[0],
        )
        retarget_loop = server.gui.add_checkbox("Loop", initial_value=True)
        retarget_speed = server.gui.add_slider("Speed", 0.05, 3.0, 0.05, 1.0)
        retarget_time = server.gui.add_slider(
            "Frame time (s)",
            0.0,
            float(get_retargeting_clip_duration(retarget_clip_names[0])),
            0.01,
            0.0,
        )
        retarget_scale = server.gui.add_slider("Retarget scale", 0.25, 1.5, 0.01, 1.0)
        apply_retarget_once = server.gui.add_button("Apply sampled targets once")

    with server.gui.add_folder("CoM Constraint"):
        enable_com = server.gui.add_checkbox("Enable CoM Constraint", initial_value=False)
        com_margin_pct = server.gui.add_slider("Safety margin (%)", 0.0, 40.0, 1.0, 5.0)
        com_use_proximity = server.gui.add_checkbox("Use proximity activation", initial_value=True)
        com_prox_display = server.gui.add_number(
            "Proximity threshold (m)", initial_value=0.0, disabled=True
        )
        vel_max = server.gui.add_slider("CoM vel max (m/s)", 0.05, 1.0, 0.01, 0.3)
        acc_max = server.gui.add_slider("CoM acc max (m/s^2)", 0.01, 2.0, 0.01, 0.3)
        com_use_acc_limits = server.gui.add_checkbox("Acceleration limits", initial_value=True)
        show_com_visualization = server.gui.add_checkbox(
            "Show CoM visualization", initial_value=False
        )
        com_slack_diag = server.gui.add_text("CoM min slack", initial_value="--")
        foot_length = server.gui.add_slider("Foot contact length (m)", 0.12, 0.35, 0.001, 0.22)
        foot_width = server.gui.add_slider("Foot contact width (m)", 0.05, 0.20, 0.001, 0.10)
        toe_pad = server.gui.add_slider("Polygon toe pad (m)", 0.0, 0.2, 0.001, 0.06)
        side_pad = server.gui.add_slider("Polygon side pad (m)", 0.0, 0.2, 0.001, 0.07)

    with server.gui.add_folder("Collision Constraint"):
        enable_collision = server.gui.add_checkbox(
            "Enable self-collision constraint",
            initial_value=False,
            disabled=not collision_available,
        )
        collision_min_dist_mm = server.gui.add_slider(
            "Collision min distance (mm)", 0.0, 120.0, 1.0, 20.0
        )
        collision_max_rows = server.gui.add_slider("Collision max rows", 1, 16, 1, 3)
        collision_tuning = server.gui.add_dropdown(
            "Collision tuning mode",
            options=("speed", "balanced", "precise"),
            initial_value="balanced",
        )
        collision_pair_preset = server.gui.add_dropdown(
            "Collision pair preset",
            options=g1_collision_pair_preset_options(),
            initial_value="core",
        )
        show_collision_debug = server.gui.add_checkbox(
            "Show collision debug",
            initial_value=True,
            disabled=not hasattr(solver, "get_last_collision_debug"),
        )
        collision_log_mode = server.gui.add_checkbox("Collision debug logging", initial_value=False)
        collision_debug_text = server.gui.add_text(
            "Collision debug",
            initial_value="collision disabled",
        )
        collision_pairs_stats = server.gui.add_text(
            "Collision pairs stats",
            initial_value=f"available={collision_available}, disabled",
        )

    with server.gui.add_folder("Diagnostics"):
        status = server.gui.add_text("Status", initial_value="running")
        solve_ms = server.gui.add_text("Solver time (ms)", initial_value="--")
        target_mode_diag = server.gui.add_text("Target solve effective", initial_value="--")
        step_diag = server.gui.add_text("Step acceptance", initial_value="--")
        contact_residual = server.gui.add_text("Contact residual ||Jc*dq||", initial_value="--")
        bias_diag = server.gui.add_text("Posture bias", initial_value="--")
        feet_bounds = server.gui.add_text("Feet drift", initial_value="--")

    com_outer_poly = server.scene.add_line_segments(
        "/com_viz/outer_polygon",
        points=np.zeros((1, 2, 3), dtype=float),
        colors=np.array([[[0.12, 0.47, 0.86], [0.12, 0.47, 0.86]]], dtype=float),
        line_width=5.0,
        visible=False,
    )
    com_inner_poly = server.scene.add_line_segments(
        "/com_viz/inner_polygon",
        points=np.zeros((1, 2, 3), dtype=float),
        colors=np.array([[[0.18, 0.80, 0.30], [0.18, 0.80, 0.30]]], dtype=float),
        line_width=3.0,
        visible=False,
    )
    com_sphere = server.scene.add_icosphere(
        "/com_viz/sphere",
        radius=0.03,
        color=(0.12, 0.78, 0.32),
        position=(0.0, 0.0, 0.5),
        visible=False,
    )
    com_floor_disk = server.scene.add_icosphere(
        "/com_viz/floor_disk",
        radius=0.02,
        color=(0.12, 0.78, 0.32),
        position=(0.0, 0.0, 0.001),
        visible=False,
    )
    com_drop_line = server.scene.add_line_segments(
        "/com_viz/drop_line",
        points=np.zeros((1, 2, 3), dtype=float),
        colors=np.array([[[0.65, 0.65, 0.65], [0.65, 0.65, 0.65]]], dtype=float),
        line_width=2.0,
        visible=False,
    )
    collision_debug_colors = (
        ((1.0, 0.2, 0.2), (0.2, 0.8, 0.2)),
        ((1.0, 0.5, 0.0), (0.3, 0.7, 1.0)),
        ((0.9, 0.2, 0.9), (0.2, 0.9, 0.9)),
        ((1.0, 0.9, 0.2), (0.2, 0.6, 1.0)),
        ((0.9, 0.4, 0.1), (0.1, 0.9, 0.4)),
        ((0.7, 0.2, 1.0), (0.2, 1.0, 0.7)),
    )
    collision_debug_points_a = [
        server.scene.add_icosphere(
            f"/collision_debug/point_a_{i}", radius=0.012, color=color_a, visible=False
        )
        for i, (color_a, _color_b) in enumerate(collision_debug_colors)
    ]
    collision_debug_points_b = [
        server.scene.add_icosphere(
            f"/collision_debug/point_b_{i}", radius=0.012, color=color_b, visible=False
        )
        for i, (_color_a, color_b) in enumerate(collision_debug_colors)
    ]
    collision_debug_lines = [None for _ in collision_debug_colors]
    opts = embodik.PositionStepOptions()
    constraint_guard = ConstrainedStepGuard(
        q,
        zero_motion_resync_frames=4,
        zero_motion_snap_frames=12,
    )
    no_progress_streak = 0
    collision_include_pair_cache: dict[str, list[tuple[str, str]]] = {}
    collision_total_pairs: int | None = None
    last_collision_cfg = None
    last_com_cfg = None
    last_collision_pair = None
    last_collision_log_ts = 0.0
    retarget_elapsed = 0.0
    last_target_signature = None
    last_control_signature = None
    last_target_positions: dict[str, np.ndarray] | None = None
    settle_steps_remaining = 0
    stability_ticks_remaining = 0
    reset_recapture_ticks_remaining = 0
    performance_mode_enabled = bool(args.performance_mode) or not bool(args.quality_mode)

    def _recapture_upright_targets() -> None:
        torso_target[:, :] = np.asarray(
            robot.get_frame_pose(frame_map["imu_in_torso"]).homogeneous(), dtype=float
        )

    def _recapture_target_controls_from_robot() -> None:
        for frame_name, ctrl in (
            (frame_map["right_palm"], right_ctrl),
            (frame_map["left_palm"], left_ctrl),
            (frame_map["right_ankle"], right_foot_ctrl),
            (frame_map["left_ankle"], left_foot_ctrl),
            (upright_frame, pelvis_ctrl),
        ):
            _set_ctrl_from_pose(
                ctrl,
                np.asarray(robot.get_frame_pose(frame_name).homogeneous(), dtype=float),
            )

    def _apply_robot_visual_state() -> None:
        robot.update_configuration(q)
        base_node.position = (float(q[0]), float(q[1]), float(q[2]))
        base_node.wxyz = (float(q[6]), float(q[3]), float(q[4]), float(q[5]))
        _set_geometry_view(str(geometry_view.value))
        urdf_vis.update_cfg(map_q(q))
        urdf_collision_vis.update_cfg(map_q_collision(q))

    def _set_geometry_view(mode: str) -> None:
        show_visual = str(mode) in {"Visual", "Both"}
        show_collision = str(mode) in {"Collision", "Both"}
        if getattr(urdf_vis, "_visual_root_frame", None) is not None:
            urdf_vis.show_visual = show_visual
        if getattr(urdf_vis, "_collision_root_frame", None) is not None:
            urdf_vis.show_collision = False
        if getattr(urdf_collision_vis, "_visual_root_frame", None) is not None:
            urdf_collision_vis.show_visual = False
        if getattr(urdf_collision_vis, "_collision_root_frame", None) is not None:
            urdf_collision_vis.show_collision = show_collision

    @geometry_view.on_update
    def _(_event) -> None:
        _set_geometry_view(str(geometry_view.value))

    _set_geometry_view(str(geometry_view.value))

    def _apply_retarget_sample(sample_time: float) -> None:
        local_offsets = sample_retargeting_clip(str(retarget_clip.value), float(sample_time))
        target_poses = build_retargeting_delta_target_poses(
            retarget_anchor,
            retarget_reference_poses,
            local_offsets,
            retarget_neutral_offsets,
            scale=float(retarget_scale.value),
        )
        if "right_palm" in target_poses:
            _set_ctrl_from_pose(right_ctrl, target_poses["right_palm"])
        if "left_palm" in target_poses:
            _set_ctrl_from_pose(left_ctrl, target_poses["left_palm"])
        if "right_ankle" in target_poses:
            _set_ctrl_from_pose(right_foot_ctrl, target_poses["right_ankle"])
        if "left_ankle" in target_poses:
            _set_ctrl_from_pose(left_foot_ctrl, target_poses["left_ankle"])
        if "imu_in_torso" in target_poses:
            _set_ctrl_from_pose(pelvis_ctrl, target_poses["imu_in_torso"])
            torso_target[:, :] = target_poses["imu_in_torso"]

    def _ensure_collision_metadata(preset: str) -> tuple[int, list[tuple[str, str]]]:
        nonlocal collision_total_pairs
        if preset not in collision_include_pair_cache:
            collision_include_pair_cache[preset] = g1_collision_pairs_for_preset(robot, preset)
        if collision_total_pairs is None:
            if hasattr(robot, "get_collision_pair_names"):
                try:
                    collision_total_pairs = len(list(robot.get_collision_pair_names()))
                except Exception:
                    collision_total_pairs = 0
            else:
                collision_total_pairs = 0
        return collision_total_pairs, collision_include_pair_cache[preset]

    def _clear_collision_debug() -> None:
        for point in collision_debug_points_a + collision_debug_points_b:
            point.visible = False
        for line in collision_debug_lines:
            if line is not None:
                line.visible = False
        collision_debug_text.value = "Collision: --"

    def _update_collision_debug() -> None:
        nonlocal collision_debug_lines, last_collision_pair, last_collision_log_ts
        if not (
            enable_collision.value
            and show_collision_debug.value
            and (
                hasattr(solver, "get_last_collision_debug_list")
                or hasattr(solver, "get_last_collision_debug")
            )
        ):
            _clear_collision_debug()
            return

        debug_rows = []
        if hasattr(solver, "get_last_collision_debug_list"):
            try:
                debug_rows = list(solver.get_last_collision_debug_list())
            except Exception:
                debug_rows = []
        if not debug_rows and hasattr(solver, "get_last_collision_debug"):
            dbg = solver.get_last_collision_debug()
            debug_rows = [] if dbg is None else [dbg]
        if not debug_rows:
            _clear_collision_debug()
            return

        debug_summaries: list[str] = []
        visible_rows = debug_rows[: len(collision_debug_colors)]
        for i, row in enumerate(visible_rows):
            p_a = np.asarray(row.point_a_world, dtype=float)
            p_b = np.asarray(row.point_b_world, dtype=float)
            collision_debug_points_a[i].position = tuple(p_a)
            collision_debug_points_b[i].position = tuple(p_b)
            collision_debug_points_a[i].visible = True
            collision_debug_points_b[i].visible = True
            if collision_debug_lines[i] is not None:
                collision_debug_lines[i].remove()
            seg = np.zeros((1, 2, 3), dtype=float)
            seg[0, 0, :] = p_a
            seg[0, 1, :] = p_b
            color_a, color_b = collision_debug_colors[i]
            collision_debug_lines[i] = server.scene.add_line_segments(
                f"/collision_debug/segment_{i}",
                points=seg,
                colors=np.array([[color_a, color_b]], dtype=float),
                line_width=3.0,
                visible=True,
            )
            debug_summaries.append(
                f"{row.object_a} <-> {row.object_b} | d={float(row.distance):.4f} m"
            )

        for i in range(len(visible_rows), len(collision_debug_colors)):
            collision_debug_points_a[i].visible = False
            collision_debug_points_b[i].visible = False
            if collision_debug_lines[i] is not None:
                collision_debug_lines[i].visible = False

        collision_debug_text.value = " || ".join(debug_summaries)

        first = visible_rows[0]
        now = time.time()
        pair_key = (
            str(first.object_a),
            str(first.object_b),
            round(float(first.distance), 4),
            len(debug_rows),
        )
        if collision_log_mode.value and (
            pair_key != last_collision_pair or now - last_collision_log_ts > 1.0
        ):
            print(
                "[13][collision]",
                f"rows={len(debug_rows)}",
                f"{first.object_a} <-> {first.object_b}",
                f"d={float(first.distance):.4f} m",
            )
            last_collision_pair = pair_key
            last_collision_log_ts = now

    def _margin_frac() -> float:
        return float(com_margin_pct.value) / 100.0

    def _support_polygon_from_targets(
        right_foot_pose: np.ndarray,
        left_foot_pose: np.ndarray,
    ) -> np.ndarray:
        return compute_support_polygon_from_foot_poses(
            right_foot_pose,
            left_foot_pose,
            foot_length=float(foot_length.value),
            foot_width=float(foot_width.value),
            toe_pad=float(toe_pad.value),
            side_pad=float(side_pad.value),
        )

    def _configure_com_constraint_if_needed(
        support_polygon: np.ndarray,
        *,
        force: bool = False,
    ) -> None:
        nonlocal last_com_cfg
        enabled = bool(enable_com.value) and hasattr(solver, "configure_com_constraint")
        next_cfg = (
            enabled,
            round(_margin_frac(), 6),
            bool(com_use_proximity.value),
            round(float(vel_max.value), 6),
            round(float(acc_max.value), 6),
            bool(com_use_acc_limits.value),
            tuple(np.round(np.asarray(support_polygon, dtype=float).reshape(-1), 6)),
        )
        if not force and next_cfg == last_com_cfg:
            return
        if not enabled:
            if hasattr(solver, "clear_com_constraint"):
                solver.clear_com_constraint()
            com_prox_display.value = 0.0
            last_com_cfg = next_cfg
            return
        solver.configure_com_constraint(
            support_polygon=support_polygon,
            margin=_margin_frac(),
            frame_name="world",
            com_vel_max=float(vel_max.value),
            com_acc_max=float(acc_max.value),
            use_acceleration_limits=bool(com_use_acc_limits.value),
            proximity_fraction=0.05 if bool(com_use_proximity.value) else 0.0,
        )
        if hasattr(solver, "get_com_proximity_threshold"):
            com_prox_display.value = round(float(solver.get_com_proximity_threshold()), 4)
        last_com_cfg = next_cfg

    def _current_com_min_slack(support_polygon: np.ndarray) -> float | None:
        if not bool(enable_com.value):
            return None
        return com_min_slack(
            support_polygon,
            np.asarray(robot.get_com_position(), dtype=float)[:2],
            margin_fraction=_margin_frac(),
        )

    def _update_com_visualization(support_polygon: np.ndarray) -> None:
        inner_polygon = shrink_polygon_xy(support_polygon, _margin_frac())
        outer_seg = polygon_segments_xy(support_polygon, z=0.002)
        inner_seg = polygon_segments_xy(inner_polygon, z=0.003)
        outer_color = np.array([[[0.12, 0.47, 0.86], [0.12, 0.47, 0.86]]], dtype=float)
        inner_color = np.array([[[0.18, 0.80, 0.30], [0.18, 0.80, 0.30]]], dtype=float)

        com_pos = np.asarray(robot.get_com_position(), dtype=float)
        com_xy = com_pos[:2]
        min_slack = com_min_slack(support_polygon, com_xy, margin_fraction=_margin_frac())
        color = com_slack_color(min_slack)
        visible = bool(show_com_visualization.value)

        com_outer_poly.points = outer_seg
        com_outer_poly.colors = np.repeat(outer_color, max(outer_seg.shape[0], 1), axis=0)
        com_outer_poly.visible = visible
        com_inner_poly.points = inner_seg
        com_inner_poly.colors = np.repeat(inner_color, max(inner_seg.shape[0], 1), axis=0)
        com_inner_poly.visible = visible and _margin_frac() > 0.0

        com_sphere.position = tuple(float(v) for v in com_pos)
        com_sphere.color = color
        com_sphere.visible = visible

        com_floor_disk.position = (float(com_xy[0]), float(com_xy[1]), 0.001)
        com_floor_disk.color = color
        com_floor_disk.visible = visible

        com_drop_line.points = np.array(
            [
                [
                    [float(com_xy[0]), float(com_xy[1]), float(com_pos[2])],
                    [float(com_xy[0]), float(com_xy[1]), 0.001],
                ]
            ],
            dtype=float,
        )
        com_drop_line.visible = visible
        com_slack_diag.value = f"{min_slack:.4f} m"

    @recapture_bias.on_click
    def _(_event) -> None:
        _configure_g1_posture_task(posture, robot, q)

    @reset_all.on_click
    def _(_event) -> None:
        nonlocal q, no_progress_streak, retarget_elapsed
        nonlocal last_target_signature, last_control_signature
        nonlocal last_target_positions
        nonlocal settle_steps_remaining, stability_ticks_remaining, reset_recapture_ticks_remaining
        nonlocal last_collision_cfg, last_com_cfg
        q = q_initial.copy()
        no_progress_streak = 0
        retarget_elapsed = 0.0
        last_target_signature = None
        last_control_signature = None
        last_target_positions = None
        settle_steps_remaining = 12
        stability_ticks_remaining = 12
        reset_recapture_ticks_remaining = 3
        last_collision_cfg = None
        last_com_cfg = None
        retarget_enable.value = False
        retarget_time.value = 0.0
        collision_pairs_stats.value = "disabled"
        constraint_guard.reset(q)
        _clear_collision_debug()
        _clear_interactive_solver_transients(solver)
        _recapture_upright_targets()
        _apply_robot_visual_state()
        _configure_g1_posture_task(posture, robot, q)
        _recapture_target_controls_from_robot()

        status.value = "RESET_TO_INITIAL_CONFIGURATION"
        contact_residual.value = "--"
        bias_diag.value = "base=--, eff=--, no_progress_streak=0"

    @apply_retarget_once.on_click
    def _(_event) -> None:
        _apply_retarget_sample(float(retarget_time.value))

    def _sync_target_control_visibility() -> None:
        for key, ctrl in target_control_by_key.items():
            ctrl.visible = bool(target_enabled[key].value)

    for handle in target_enabled.values():

        @handle.on_update
        def _(_event) -> None:
            _sync_target_control_visibility()

    _sync_target_control_visibility()

    def _enabled_target_keys() -> tuple[str, ...]:
        keys = tuple(key for key in ordered_target_keys if bool(target_enabled[key].value))
        return keys if keys else ordered_target_keys

    prev_loop_time = time.perf_counter()
    loop_count = 0
    while True:
        loop_count += 1
        loop_t0 = time.perf_counter()
        dt_loop = max(0.0, loop_t0 - prev_loop_time)
        prev_loop_time = loop_t0
        q_prev = np.asarray(q, dtype=float).copy()
        duration = get_retargeting_clip_duration(str(retarget_clip.value))
        if retarget_enable.value:
            retarget_elapsed += dt_loop * float(retarget_speed.value)
            if retarget_loop.value and duration > 0.0:
                retarget_elapsed = retarget_elapsed % duration
            else:
                retarget_elapsed = min(retarget_elapsed, duration)
            retarget_time.value = float(retarget_elapsed)
            _apply_retarget_sample(retarget_elapsed)
        else:
            retarget_elapsed = float(np.clip(float(retarget_time.value), 0.0, duration))
        if reset_recapture_ticks_remaining > 0:
            _apply_robot_visual_state()
            _recapture_upright_targets()
            _recapture_target_controls_from_robot()
            _configure_g1_posture_task(posture, robot, q)
            reset_recapture_ticks_remaining -= 1
        rh_pose = _pose_from_ctrl(right_ctrl)
        lh_pose = _pose_from_ctrl(left_ctrl)
        rf_target_pose = _pose_from_ctrl(right_foot_ctrl)
        lf_target_pose = _pose_from_ctrl(left_foot_ctrl)
        pelvis_target_pose = _pose_from_ctrl(pelvis_ctrl)
        target_pose_by_key = {
            "right_palm": rh_pose,
            "left_palm": lh_pose,
            "right_ankle": rf_target_pose,
            "left_ankle": lf_target_pose,
            "pelvis": pelvis_target_pose,
        }
        enabled_target_keys = _enabled_target_keys()
        current_target_positions = {
            key: target_pose_by_key[key][:3, 3].copy() for key in ordered_target_keys
        }
        target_deltas = {
            key: (
                0.0
                if last_target_positions is None
                else float(np.linalg.norm(pos - last_target_positions[key]))
            )
            for key, pos in current_target_positions.items()
        }
        frame_targets = [
            (target_frame_by_key[key], target_pose_by_key[key]) for key in enabled_target_keys
        ]
        active_support_polygon = _support_polygon_from_targets(rf_target_pose, lf_target_pose)
        live_target_error = _max_frame_target_position_error(robot, frame_targets)
        moving_targets = {key for key in enabled_target_keys if target_deltas.get(key, 0.0) > 1e-5}
        target_signature = (
            _pose_signature(rh_pose),
            _pose_signature(lh_pose),
            _pose_signature(rf_target_pose),
            _pose_signature(lf_target_pose),
            _pose_signature(pelvis_target_pose),
        )
        control_signature = (
            tuple(enabled_target_keys),
            str(target_solve_mode.value),
            int(steps.value),
            round(float(target_pos_gain.value), 3),
            round(float(target_ori_gain.value), 3),
            bool(enable_torso_upright.value),
            round(float(torso_upright_gain.value), 3),
            round(float(posture_bias_weight.value), 4),
            bool(performance_mode_enabled),
            int(posture_update_period.value),
            bool(enable_collision.value),
            round(float(collision_min_dist_mm.value), 3),
            int(collision_max_rows.value),
            str(collision_tuning.value),
            str(collision_pair_preset.value),
            bool(enable_com.value),
            round(_margin_frac(), 4),
            bool(com_use_proximity.value),
            round(float(vel_max.value), 3),
            round(float(acc_max.value), 3),
            bool(com_use_acc_limits.value),
            round(float(foot_length.value), 4),
            round(float(foot_width.value), 4),
            round(float(toe_pad.value), 4),
            round(float(side_pad.value), 4),
        )
        if retarget_enable.value:
            settle_steps_remaining = max(settle_steps_remaining, 2)
        elif (
            target_signature != last_target_signature or control_signature != last_control_signature
        ):
            target_changed = target_signature != last_target_signature
            control_changed = control_signature != last_control_signature
            last_target_signature = target_signature
            last_control_signature = control_signature
            settle_steps_remaining = 12 if control_changed else 4
            if control_changed and not target_changed:
                stability_ticks_remaining = max(stability_ticks_remaining, 8)
        elif (
            settle_steps_remaining <= 0
            and live_target_error <= _g1_interactive_hold_error_threshold()
        ):
            _apply_robot_visual_state()
            _configure_com_constraint_if_needed(active_support_polygon)
            _update_com_visualization(active_support_polygon)
            _update_collision_debug()
            _sleep_for_loop_rate(loop_t0)
            continue
        last_target_positions = current_target_positions

        target_mode, target_mode_label, target_can_fallback = _target_solve_mode(
            str(target_solve_mode.value)
        )
        for key, task in task_by_key.items():
            task.active = key in enabled_target_keys
            task.solve_mode = target_mode
            task.allow_min_error_fallback = bool(target_can_fallback)
        perf_mode = bool(performance_mode_enabled)
        torso_ori.active = bool(enable_torso_upright.value) and not perf_mode
        torso_ori.solve_mode = embodik.TaskSolveMode.MIN_ERROR
        torso_ori.allow_min_error_fallback = True
        base_bias = float(posture_bias_weight.value)
        posture_period = max(1, int(posture_update_period.value))
        posture.active = (
            not perf_mode or reset_recapture_ticks_remaining > 0 or loop_count % posture_period == 0
        )
        posture.weight = base_bias if posture.active else 0.0

        solver.clear_contact_frames()
        solver.clear_tight_frame_pose_constraints()

        cfg_tuple = (
            bool(enable_collision.value),
            float(collision_min_dist_mm.value),
            int(collision_max_rows.value),
            str(collision_tuning.value),
            str(collision_pair_preset.value),
        )
        active_collision_include_pairs: list[tuple[str, str]] = []
        if bool(enable_collision.value):
            total_pairs, active_collision_include_pairs = _ensure_collision_metadata(
                str(collision_pair_preset.value)
            )
            collision_pairs_stats.value = (
                f"preset={collision_pair_preset.value}, total_pairs={total_pairs}, "
                f"included={len(active_collision_include_pairs)}"
            )
        if cfg_tuple != last_collision_cfg:
            if enable_collision.value:
                _configure_g1_collision_constraint(
                    solver,
                    enabled=True,
                    min_distance_m=float(collision_min_dist_mm.value) * 1e-3,
                    max_constraints=int(collision_max_rows.value),
                    tuning_mode=str(collision_tuning.value),
                    include_pairs=active_collision_include_pairs,
                )
            else:
                _configure_g1_collision_constraint(
                    solver,
                    enabled=False,
                    min_distance_m=float(collision_min_dist_mm.value) * 1e-3,
                    max_constraints=int(collision_max_rows.value),
                    tuning_mode=str(collision_tuning.value),
                    include_pairs=[],
                )
                constraint_guard.reset(q)
                collision_pairs_stats.value = "disabled"
                collision_debug_text.value = "collision disabled"
            last_collision_cfg = cfg_tuple

        _configure_com_constraint_if_needed(active_support_polygon)

        base_pos_gain = float(target_pos_gain.value)
        base_ori_gain = float(target_ori_gain.value)
        single_moving_target = next(iter(moving_targets)) if len(moving_targets) == 1 else None
        right_hand_pos_gain = base_pos_gain
        left_hand_pos_gain = base_pos_gain
        right_hand_ori_gain = base_ori_gain
        left_hand_ori_gain = base_ori_gain
        right_foot_pos_gain_eff = base_pos_gain
        left_foot_pos_gain_eff = base_pos_gain
        pelvis_pos_gain_eff = base_pos_gain
        pelvis_ori_gain_eff = base_ori_gain
        if single_moving_target == "right_palm":
            right_hand_pos_gain = min(80.0, base_pos_gain * 2.0)
            if "left_palm" in enabled_target_keys:
                left_hand_pos_gain = base_pos_gain * 0.2
                left_hand_ori_gain = base_ori_gain * 0.2
            if "right_ankle" in enabled_target_keys:
                right_foot_pos_gain_eff = base_pos_gain * 0.4
            if "left_ankle" in enabled_target_keys:
                left_foot_pos_gain_eff = base_pos_gain * 0.4
            if "pelvis" in enabled_target_keys:
                pelvis_pos_gain_eff = base_pos_gain * 0.25
                pelvis_ori_gain_eff = base_ori_gain * 0.25
        elif single_moving_target == "left_palm":
            left_hand_pos_gain = min(80.0, base_pos_gain * 2.0)
            if "right_palm" in enabled_target_keys:
                right_hand_pos_gain = base_pos_gain * 0.2
                right_hand_ori_gain = base_ori_gain * 0.2
            if "right_ankle" in enabled_target_keys:
                right_foot_pos_gain_eff = base_pos_gain * 0.4
            if "left_ankle" in enabled_target_keys:
                left_foot_pos_gain_eff = base_pos_gain * 0.4
            if "pelvis" in enabled_target_keys:
                pelvis_pos_gain_eff = base_pos_gain * 0.25
                pelvis_ori_gain_eff = base_ori_gain * 0.25
        elif single_moving_target == "right_ankle":
            right_foot_pos_gain_eff = min(120.0, base_pos_gain * 1.5)
            if "right_palm" in enabled_target_keys:
                right_hand_pos_gain = base_pos_gain * 0.35
                right_hand_ori_gain = base_ori_gain * 0.35
            if "left_palm" in enabled_target_keys:
                left_hand_pos_gain = base_pos_gain * 0.35
                left_hand_ori_gain = base_ori_gain * 0.35
            if "left_ankle" in enabled_target_keys:
                left_foot_pos_gain_eff = base_pos_gain * 0.25
            if "pelvis" in enabled_target_keys:
                pelvis_pos_gain_eff = base_pos_gain * 0.25
                pelvis_ori_gain_eff = base_ori_gain * 0.25
        elif single_moving_target == "left_ankle":
            left_foot_pos_gain_eff = min(120.0, base_pos_gain * 1.5)
            if "right_palm" in enabled_target_keys:
                right_hand_pos_gain = base_pos_gain * 0.35
                right_hand_ori_gain = base_ori_gain * 0.35
            if "left_palm" in enabled_target_keys:
                left_hand_pos_gain = base_pos_gain * 0.35
                left_hand_ori_gain = base_ori_gain * 0.35
            if "right_ankle" in enabled_target_keys:
                right_foot_pos_gain_eff = base_pos_gain * 0.25
            if "pelvis" in enabled_target_keys:
                pelvis_pos_gain_eff = base_pos_gain * 0.25
                pelvis_ori_gain_eff = base_ori_gain * 0.25
        elif single_moving_target == "pelvis":
            pelvis_pos_gain_eff = min(80.0, max(18.0, base_pos_gain * 2.0))
            pelvis_ori_gain_eff = min(60.0, base_ori_gain * 1.4)
            if "right_palm" in enabled_target_keys:
                right_hand_pos_gain = base_pos_gain * 0.25
                right_hand_ori_gain = base_ori_gain * 0.25
            if "left_palm" in enabled_target_keys:
                left_hand_pos_gain = base_pos_gain * 0.25
                left_hand_ori_gain = base_ori_gain * 0.25
            if "right_ankle" in enabled_target_keys:
                right_foot_pos_gain_eff = base_pos_gain * 0.35
            if "left_ankle" in enabled_target_keys:
                left_foot_pos_gain_eff = base_pos_gain * 0.35

        gain_by_key = {
            "right_palm": (right_hand_pos_gain, right_hand_ori_gain),
            "left_palm": (left_hand_pos_gain, left_hand_ori_gain),
            "right_ankle": (right_foot_pos_gain_eff, base_ori_gain),
            "left_ankle": (left_foot_pos_gain_eff, base_ori_gain),
            "pelvis": (pelvis_pos_gain_eff, pelvis_ori_gain_eff),
        }
        targets = [
            embodik.TaskTarget(
                task_name_by_key[key],
                target_pose_by_key[key],
                float(gain_by_key[key][0]),
                float(gain_by_key[key][1]),
            )
            for key in enabled_target_keys
        ]
        if bool(enable_torso_upright.value):
            targets.append(
                embodik.TaskTarget(
                    "torso_upright_ori", torso_target, 0.0, float(torso_upright_gain.value)
                )
            )
        quality_frame_targets = (
            [
                (
                    target_frame_by_key[single_moving_target],
                    target_pose_by_key[single_moving_target],
                )
            ]
            if single_moving_target in enabled_target_keys
            else frame_targets
        )

        opts.max_steps = 1 if perf_mode else int(steps.value)
        if not perf_mode and single_moving_target in {
            "right_palm",
            "left_palm",
            "right_ankle",
            "left_ankle",
            "pelvis",
        }:
            opts.max_steps = max(opts.max_steps, 6)
        opts.position_gain = float(target_pos_gain.value)
        opts.orientation_gain = float(target_ori_gain.value)
        opts.max_linear_speed = 1.8
        opts.max_angular_speed = 2.5
        opts.stall_recovery = not perf_mode
        opts.adaptive_dt = not perf_mode
        opts.adaptive_dt_reference_distance = 0.04
        opts.adaptive_dt_max_scale = 3.0
        t0 = time.perf_counter()
        tracked_tasks = [task_by_key[key] for key in enabled_target_keys]
        if perf_mode:
            result = solver.solve_position_step(q, targets, opts)
            recovered_step = False
        else:
            result, recovered_step = _solve_quality_step(
                solver,
                robot,
                q,
                q_lo,
                q_hi,
                targets,
                opts,
                tracked_tasks,
                quality_frame_targets,
                prefer_min_error=stability_ticks_remaining > 0,
            )
        solve_ms.value = f"{(time.perf_counter() - t0) * 1e3:.2f}"
        stability_active = stability_ticks_remaining > 0
        if recovered_step:
            stability_ticks_remaining = 12
        else:
            stability_ticks_remaining = max(0, stability_ticks_remaining - 1)

        solver_intervened = (
            int(getattr(result, "collision_rejection_count", 0)) > 0
            or int(getattr(result, "stall_escape_count", 0)) > 0
        )
        accept_non_success = result.status == embodik.SolverStatus.NO_PROGRESS
        has_finite_solution = hasattr(result, "q_solution") and np.all(
            np.isfinite(np.asarray(result.q_solution, dtype=float))
        )
        accepted_step = (
            result.status == embodik.SolverStatus.SUCCESS
            or (accept_non_success and has_finite_solution and not bool(enable_collision.value))
            or (bool(enable_collision.value) and solver_intervened and has_finite_solution)
        )
        status_override: str | None = None
        q_step_component = 0.0
        if accepted_step:
            q_candidate = _clip_q(robot, np.asarray(result.q_solution, dtype=float), q_lo, q_hi)
            q_limited, q_step_component = _limit_tangent_step(robot, q_prev, q_candidate, 0.8)
            q = _clip_q(robot, q_limited, q_lo, q_hi)
        else:
            q = q_prev
        if (
            accepted_step
            and enable_collision.value
            and (
                hasattr(solver, "evaluate_post_step_collision_distance")
                or hasattr(solver, "evaluate_min_collision_distance")
            )
        ):
            post_distance = _post_step_collision_distance(solver, q)
            if post_distance < float(collision_min_dist_mm.value) * 1e-3 - 1e-4:
                q = q_prev
                no_progress_streak = max(no_progress_streak, 1)
                status_override = f"HELD_LAST_SAFE_COLLISION d={post_distance:.4f} m"
        robot.update_configuration(q)
        current_target_error = _max_frame_target_position_error(robot, frame_targets)
        collision_min_distance_m = float(collision_min_dist_mm.value) * 1e-3
        current_collision_min = (
            _current_collision_min_distance(solver) if bool(enable_collision.value) else None
        )
        current_com_min_slack = _current_com_min_slack(active_support_polygon)
        boundaries = [
            ConstraintBoundary(
                "collision",
                current_collision_min,
                collision_min_distance_m,
                enabled=bool(enable_collision.value),
                violation_tolerance=1e-5,
            ),
            ConstraintBoundary(
                "CoM",
                current_com_min_slack,
                0.0,
                enabled=bool(enable_com.value),
                violation_tolerance=1e-4,
            ),
        ]
        guard_decision = constraint_guard.evaluate(
            q_candidate=q,
            result=result,
            max_task_error=current_target_error,
            boundaries=boundaries,
            task_deadband=_g1_interactive_hold_error_threshold(),
            constraints_enabled=bool(enable_collision.value or enable_com.value),
        )
        if guard_decision.restored_last_safe:
            q = _clip_q(robot, guard_decision.q_next, q_lo, q_hi)
            robot.update_configuration(q)
            current_target_error = _max_frame_target_position_error(robot, frame_targets)
            current_collision_min = _current_collision_min_distance(solver, q)
            current_com_min_slack = _current_com_min_slack(active_support_polygon)
            status_override = (
                f"{' + '.join(guard_decision.restore_labels)} guard restored last safe pose"
            )

        if (
            bool(enable_collision.value)
            and current_collision_min is not None
            and float(current_collision_min) < collision_min_distance_m
            and result.status
            in {
                embodik.SolverStatus.INFEASIBLE,
                embodik.SolverStatus.NUMERICAL_ERROR,
                embodik.SolverStatus.NO_PROGRESS,
            }
            and joint_velocity_norm(result) <= 1e-8
        ):
            escaped_q = _attempt_g1_penetration_escape_burst(
                robot=robot,
                solver=solver,
                q_current=q,
                targets=targets,
                options=opts,
                target_tasks=tracked_tasks,
                frame_targets=quality_frame_targets,
                q_lo=q_lo,
                q_hi=q_hi,
                min_distance_m=collision_min_distance_m,
                max_constraints=int(collision_max_rows.value),
                tuning_mode=str(collision_tuning.value),
                include_pairs=active_collision_include_pairs,
            )
            if escaped_q is not None:
                q = _clip_q(robot, escaped_q, q_lo, q_hi)
                robot.update_configuration(q)
                current_target_error = _max_frame_target_position_error(robot, frame_targets)
                current_collision_min = _current_collision_min_distance(solver, q)
                current_com_min_slack = _current_com_min_slack(active_support_polygon)
                constraint_guard.reset(q)
                status_override = "penetration escape burst accepted"

        boundaries = [
            ConstraintBoundary(
                "collision",
                current_collision_min,
                collision_min_distance_m,
                enabled=bool(enable_collision.value),
                violation_tolerance=1e-5,
            ),
            ConstraintBoundary(
                "CoM",
                current_com_min_slack,
                0.0,
                enabled=bool(enable_com.value),
                violation_tolerance=1e-4,
            ),
        ]
        constraint_guard.remember_if_clear(q, boundaries)
        active_constraint_labels = list(guard_decision.boundary_stall_labels)
        if guard_decision.zero_motion_resync:
            clear_all_target_velocities_if_available(solver)
            status_override = status_override or "constrained zero-motion; solver state re-synced"
        if guard_decision.zero_motion_snap:
            _recapture_target_controls_from_robot()
            status_override = (
                status_override
                or "constrained zero-motion persisted; targets snapped to current frames"
            )
        elif active_constraint_labels:
            _recapture_target_controls_from_robot()
            status_override = status_override or (
                f"{' + '.join(active_constraint_labels)} limited; targets snapped to current frames"
            )
        productive_no_progress = (
            result.status == embodik.SolverStatus.NO_PROGRESS
            and accepted_step
            and current_target_error <= 0.03
            and (q_step_component > 1e-8 or not moving_targets)
        )
        if result.status == embodik.SolverStatus.NO_PROGRESS and not productive_no_progress:
            no_progress_streak += 1
        elif result.status == embodik.SolverStatus.SUCCESS or productive_no_progress:
            no_progress_streak = max(0, no_progress_streak - 1)
        else:
            no_progress_streak = max(0, no_progress_streak - 1)
        settle_steps_remaining = max(0, settle_steps_remaining - 1)
        _apply_robot_visual_state()
        _update_com_visualization(active_support_polygon)

        r_now = np.asarray(
            robot.get_frame_pose(frame_map["right_ankle"]).homogeneous(), dtype=float
        )
        l_now = np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).homogeneous(), dtype=float)
        r_err = _frame_delta6(rf_target_pose, r_now)
        l_err = _frame_delta6(lf_target_pose, l_now)
        dq = np.asarray(result.joint_velocities, dtype=float)
        task_modes = _enum_names(getattr(result, "task_modes_effective", []))
        task_scales = getattr(result, "task_scales", [])
        task_fallback = getattr(result, "task_used_fallback", [])
        target_mode_diag.value = (
            f"requested={target_solve_mode.value}, active={target_mode_label}, "
            f"upright=MIN_ERROR, stability={stability_active}, modes=[{task_modes}]"
        )
        step_diag.value = (
            f"accepted={accepted_step}, dq_norm={np.linalg.norm(dq):.3e}, "
            f"recovered={recovered_step}, scales={list(task_scales)[:4]}, "
            f"fallback={list(task_fallback)[:4]}"
        )
        foot_task_err = np.linalg.norm(r_err[:3]) + np.linalg.norm(l_err[:3])
        contact_residual.value = f"soft-foot pos sum={foot_task_err:.3e}"
        feet_bounds.value = (
            f"R {'on' if 'right_ankle' in enabled_target_keys else 'off'} "
            f"pos={np.max(np.abs(r_err[:3])):.4e}, rot={np.rad2deg(np.max(np.abs(r_err[3:]))):.3f} deg | "
            f"L {'on' if 'left_ankle' in enabled_target_keys else 'off'} "
            f"pos={np.max(np.abs(l_err[:3])):.4e}, rot={np.rad2deg(np.max(np.abs(l_err[3:]))):.3f} deg"
        )
        feet_center = compute_feet_center(r_now[:3, 3], l_now[:3, 3])
        if status_override:
            status.value = f"{status_override}" + (
                f" | {getattr(result, 'status_message', '')}"
                if getattr(result, "status_message", "")
                else ""
            )
        else:
            status.value = (
                f"{result.status.name} | targets={','.join(enabled_target_keys)} | "
                f"clip={retarget_clip.value}@{retarget_elapsed:.2f}s | "
                f"feet_center=({feet_center[0]:.3f},{feet_center[1]:.3f},{feet_center[2]:.3f})"
            )
        bias_diag.value = f"base={base_bias:.3f}, eff={posture.weight:.3f}, no_progress_streak={no_progress_streak}"

        _update_collision_debug()

        _sleep_for_loop_rate(loop_t0)
        _ = loop_t0
        time.sleep(0.002)


if __name__ == "__main__":
    main()
