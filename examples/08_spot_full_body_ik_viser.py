#!/usr/bin/env python3
"""Spot full-body IK in regular Viser.

This example is independent from the MuJoCo policy rollout. It loads a Spot
whole-body URDF directly into EmbodiK and demonstrates four interactive solve
modes: arm+torso, torso-only through the legs, single-stage full-body, and
two-stage full-body IK.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_EXAMPLES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLES_DIR.parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from embodik.gpu.wbc import gpu_wbc_collision_control_availability  # noqa: E402
from example_helpers.ik_common import (  # noqa: E402
    COLLISION_TUNING_OPTIONS,
    DEFAULT_VISER_PORT,
    quiet_websocket_handshake_logs,
)
from example_helpers.seer_teleop import (  # noqa: E402
    DEFAULT_TELEOP_SCALE_FACTOR,
    SeerController,
    apply_controller_delta,
    gripper_command_from_trigger_fraction,
    pose_from_transform_control,
    set_transform_control_pose,
)
from example_helpers.spot_locomanip_policy import (  # noqa: E402
    DEFAULT_ARM_COMMAND,
    INITIAL_ARM_COMMAND,
)
from example_helpers.spot_whole_body_ik import (  # noqa: E402
    SPOT_COLLISION_MIN_DISTANCE_M,
    STANDARD_FULL_BODY_NULLSPACE_GAIN,
    STANDARD_FULL_BODY_TORSO_POSE_HALF_RANGE,
    SpotFullBodyIK,
    SpotFullBodyIKConfig,
    SpotFullBodyIKMode,
    resolve_spot_ik_urdf,
    target_pose_from_wxyz,
)


def _rotation_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    from embodik import r2q

    xyzw = np.asarray(r2q(np.asarray(rotation, dtype=float), order="xyzs"), dtype=float)
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)


def _pose_to_viser(pose) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(pose.translation, dtype=float),
        _rotation_to_wxyz(np.asarray(pose.rotation, dtype=float)),
    )


def _base_pose_to_viser(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quat_xyzw = np.asarray(q[3:7], dtype=float)
    return (
        np.asarray(q[:3], dtype=float),
        np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float),
    )


def _vis_cfg_from_q(backend: SpotFullBodyIK, actuated_names: list[str]) -> np.ndarray:
    cfg = np.zeros(len(actuated_names), dtype=float)
    q = np.asarray(backend.q, dtype=float)
    for i, joint_name in enumerate(actuated_names):
        if not backend.robot.has_joint(joint_name):
            continue
        idx = int(backend.robot.get_joint_config_index(joint_name))
        size = int(backend.robot.get_joint_config_size(joint_name))
        if size == 1 and idx < q.size:
            cfg[i] = float(q[idx])
    return cfg


def _status_text(result) -> str:
    if isinstance(result, tuple):
        return " | ".join(_status_text(item) for item in result)
    if isinstance(getattr(result, "status", None), str):
        position = max(getattr(result, "position_errors", (0.0,)))
        orientation = max(getattr(result, "rotation_errors", (0.0,)))
        return (
            f"GPU {result.status}: pos={float(position) * 1e3:.1f} mm, "
            f"rot={float(orientation):.3f} rad"
        )
    return (
        f"{result.status.name}: pos={float(result.position_error) * 1e3:.1f} mm, "
        f"rot={float(result.orientation_error):.3f} rad"
    )


def _make_gpu_solver(
    args: argparse.Namespace, backend: SpotFullBodyIK, urdf_path: Path
):
    if not getattr(args, "gpu_wbc_warp_info", False):
        import warp as wp

        # Keep actionable warnings visible while hiding routine cached-module
        # load timing from the public example's control-loop output.
        wp.config.log_level = wp.LOG_WARNING
    from example_helpers.gpu_spot_full_body_ik import GpuSpotFullBodyIK

    return GpuSpotFullBodyIK(args, backend, urdf_path)


def run_headless(args: argparse.Namespace) -> None:
    urdf_path = resolve_spot_ik_urdf(args.urdf)
    if urdf_path is None:
        raise SystemExit(
            "Spot whole-body URDF not found; pass --urdf or set EMBODIK_SPOT_IK_URDF"
        )
    backend = SpotFullBodyIK(urdf_path, config=SpotFullBodyIKConfig(dt=args.dt))
    current = backend.current_tool_pose()
    tool_target = target_pose_from_wxyz(
        np.asarray(current.translation, dtype=float)
        + np.array([args.target_dx, 0.0, 0.0]),
        _rotation_to_wxyz(np.asarray(current.rotation, dtype=float)),
    )
    torso_current = backend.current_torso_pose()
    torso_target = target_pose_from_wxyz(
        np.asarray(torso_current.translation, dtype=float)
        + np.array([args.target_dx, 0.0, 0.0]),
        _rotation_to_wxyz(np.asarray(torso_current.rotation, dtype=float)),
    )
    mode = SpotFullBodyIKMode(args.mode)
    result = None
    if args.gpu_wbc:
        gpu_solver = _make_gpu_solver(args, backend, urdf_path)
        for _ in range(args.steps):
            result = gpu_solver.solve(mode, tool_target, torso_target)
    else:
        for _ in range(args.steps):
            backend.solve(mode, tool_target, torso_target)
    final = backend.current_tool_pose()
    pos_error = float(
        np.linalg.norm(
            np.asarray(final.translation) - np.asarray(tool_target.translation)
        )
    )
    print(
        f"Ran {args.steps} headless Spot IK steps with "
        f"backend={'gpu-newton' if args.gpu_wbc else 'cpu'}, mode={mode.value}, "
        f"tool_position_error={pos_error * 1e3:.1f} mm, "
        f"foot_anchor_error={backend.foot_anchor_error() * 1e3:.1f} mm"
    )
    if args.gpu_wbc and result is not None:
        print(_status_text(result))


def run_viser(args: argparse.Namespace) -> None:
    quiet_websocket_handshake_logs()

    try:
        import viser
        from viser.extras import ViserUrdf
        from yourdfpy import URDF
    except ImportError as exc:
        raise SystemExit("Interactive mode requires viser and yourdfpy") from exc

    urdf_path = resolve_spot_ik_urdf(args.urdf)
    if urdf_path is None:
        raise SystemExit(
            "Spot whole-body URDF not found; pass --urdf or set EMBODIK_SPOT_IK_URDF"
        )

    backend = SpotFullBodyIK(urdf_path, config=SpotFullBodyIKConfig(dt=args.dt))
    gpu_solver = _make_gpu_solver(args, backend, urdf_path) if args.gpu_wbc else None
    teleop_controller = SeerController(
        args.controller_port if args.enable_teleop else None
    )
    teleop_connected = teleop_controller.connect()
    urdf = URDF.load(str(urdf_path))
    actuated_names = list(getattr(urdf, "actuated_joint_names", []))

    server = viser.ViserServer(port=args.port, label="EmbodiK Spot full-body IK")
    server.scene.add_grid("/ground", width=5.0, height=5.0)
    base_node = server.scene.add_frame(
        "/spot_base",
        show_axes=True,
        axes_length=0.15,
        axes_radius=0.006,
        position=(0.0, 0.0, 0.0),
        wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    urdf_vis = ViserUrdf(
        server,
        urdf_path,
        root_node_name="/spot_base/robot",
        collision_mesh_color_override=(0.1, 0.65, 1.0, 0.35),
        load_collision_meshes=True,
    )
    collision_geometry_available = (
        getattr(urdf_vis, "_collision_root_frame", None) is not None
    )
    if collision_geometry_available:
        urdf_vis.show_collision = False

    tool_pos, tool_wxyz = _pose_to_viser(backend.current_tool_pose())
    torso_pos, torso_wxyz = _pose_to_viser(backend.current_torso_pose())
    tool_target = server.scene.add_transform_controls(
        "/targets/gripper",
        scale=0.18,
        position=tool_pos,
        wxyz=tool_wxyz,
    )
    torso_target = server.scene.add_transform_controls(
        "/targets/torso",
        scale=0.36,
        position=torso_pos,
        wxyz=torso_wxyz,
    )

    with server.gui.add_folder("Spot IK"):
        backend_dropdown = server.gui.add_dropdown(
            "Solver backend",
            (
                ("CPU EmbodiK", "GPU Newton")
                if gpu_solver is not None
                else ("CPU EmbodiK",)
            ),
            initial_value=(
                "GPU Newton"
                if gpu_solver is not None
                else "CPU EmbodiK"
            ),
        )
        mode_dropdown = server.gui.add_dropdown(
            "Mode",
            tuple(mode.value for mode in SpotFullBodyIKMode),
            initial_value=args.mode,
        )
        standard_bounds = STANDARD_FULL_BODY_TORSO_POSE_HALF_RANGE
        half_x = server.gui.add_slider(
            "Torso +/- x (m)",
            0.0,
            0.3,
            initial_value=float(standard_bounds[0]),
            step=0.005,
        )
        half_y = server.gui.add_slider(
            "Torso +/- y (m)",
            0.0,
            0.3,
            initial_value=float(standard_bounds[1]),
            step=0.005,
        )
        half_z = server.gui.add_slider(
            "Torso +/- z (m)",
            0.0,
            0.3,
            initial_value=float(standard_bounds[2]),
            step=0.005,
        )
        half_r = server.gui.add_slider(
            "Torso +/- rpy (deg)", 0.0, 45.0, initial_value=15.0, step=0.5
        )
        pos_gain = server.gui.add_slider(
            "Position gain", 1.0, 120.0, initial_value=60.0, step=1.0
        )
        rot_gain = server.gui.add_slider(
            "Orientation gain", 1.0, 120.0, initial_value=60.0, step=1.0
        )
        nullspace_gain = server.gui.add_slider(
            "Nullspace gain",
            0.0,
            5.0,
            initial_value=STANDARD_FULL_BODY_NULLSPACE_GAIN,
            step=0.01,
        )

    with server.gui.add_folder("Collision"):
        collision_available = hasattr(
            backend.solver, "configure_collision_constraint"
        ) and bool(backend._collision_include_pairs)
        collision_debug_available = hasattr(
            backend.solver, "get_last_collision_debug_list"
        ) or hasattr(backend.solver, "get_last_collision_debug")
        collision_enable = server.gui.add_checkbox(
            "Enable collision constraint",
            initial_value=collision_available,
            disabled=not collision_available,
        )
        collision_min_dist_mm = server.gui.add_slider(
            "Collision min dist (mm)",
            0.0,
            100.0,
            initial_value=SPOT_COLLISION_MIN_DISTANCE_M * 1e3,
            step=1.0,
        )
        collision_max_rows = server.gui.add_slider(
            "Collision rows",
            1,
            8,
            initial_value=3,
            step=1,
        )
        collision_tuning = server.gui.add_dropdown(
            "Collision tuning",
            options=COLLISION_TUNING_OPTIONS,
            initial_value="balanced",
        )
        robot_geometry = server.gui.add_dropdown(
            "Robot geometry",
            options=("Visual", "Collision", "Visual + collision"),
            initial_value="Visual",
            disabled=not collision_geometry_available,
        )
        show_collision_debug = server.gui.add_checkbox(
            "Show collision debug",
            initial_value=False,
            disabled=not collision_debug_available,
        )
        collision_log_mode = server.gui.add_checkbox(
            "Collision debug logging",
            initial_value=False,
            disabled=not collision_debug_available,
        )
        collision_min_dist_mm.disabled = not collision_available
        collision_max_rows.disabled = not collision_available
        collision_tuning.disabled = not collision_available

    def sync_robot_geometry_visibility() -> None:
        geometry_mode = str(robot_geometry.value)
        urdf_vis.show_visual = geometry_mode in {"Visual", "Visual + collision"}
        if collision_geometry_available:
            urdf_vis.show_collision = geometry_mode in {
                "Collision",
                "Visual + collision",
            }

    @robot_geometry.on_update
    def _(_) -> None:
        sync_robot_geometry_visibility()

    with server.gui.add_folder("Solver"):
        solve_mode = server.gui.add_dropdown(
            "Target solve mode",
            options=("SCALE_ELASTIC", "MIN_ERROR", "SCALE"),
            initial_value="SCALE_ELASTIC",
        )
        dt = server.gui.add_slider(
            "dt (s)", 0.002, 0.03, initial_value=args.dt, step=0.001
        )
        max_steps = server.gui.add_slider(
            "Steps per frame", 1, 10, initial_value=1, step=1
        )
        adaptive_dt = server.gui.add_checkbox("Adaptive dt", initial_value=True)
        adaptive_dt_max_scale = server.gui.add_slider(
            "Adaptive dt max scale",
            1.0,
            10.0,
            initial_value=3.0,
            step=0.5,
        )
        adaptive_dt_ref_dist = server.gui.add_slider(
            "Adaptive dt ref dist (m)",
            0.01,
            0.20,
            initial_value=0.04,
            step=0.01,
        )
        snap_tool = server.gui.add_button("Snap gripper target")
        snap_torso = server.gui.add_button("Snap torso target")
        stow_btn = server.gui.add_button("Stow arm")
        unstow_btn = server.gui.add_button("Unstow arm")
        reset_btn = server.gui.add_button("Reset robot")
        reanchor_btn = server.gui.add_button("Re-anchor feet/torso")

    with server.gui.add_folder("Seer teleop"):
        teleop_enabled_checkbox = server.gui.add_checkbox(
            "Enable teleop",
            initial_value=False,
            disabled=not teleop_connected,
        )
        teleop_connected_text = server.gui.add_text(
            "Controller",
            initial_value="connected" if teleop_connected else "browser only",
            disabled=True,
        )
        teleop_streaming_text = server.gui.add_text(
            "Streaming", initial_value="OFF", disabled=True
        )
        teleop_gripper_text = server.gui.add_text(
            "Gripper", initial_value="OPEN", disabled=True
        )
        teleop_reset_text = server.gui.add_text(
            "Reset", initial_value="Button B", disabled=True
        )
        teleop_scale = server.gui.add_slider(
            "Position Scale",
            0.5,
            3.0,
            initial_value=float(args.teleop_scale),
            step=0.1,
            disabled=not teleop_connected,
        )
        teleop_scale.disabled = not (
            teleop_connected and bool(teleop_enabled_checkbox.value)
        )

    with server.gui.add_folder("Status"):
        status = server.gui.add_text("IK status", initial_value="ready")
        foot_error = server.gui.add_text("Foot anchor error", initial_value="0.0 mm")
        nullspace_error = server.gui.add_text("Nullspace error", initial_value="--")
        solve_ms = server.gui.add_text("Last solve (ms)", initial_value="--")
        collision_debug_text = server.gui.add_text(
            "Minimum collision vector",
            initial_value="Collision: --",
        )
        collision_status = server.gui.add_text(
            "Collision pairs",
            initial_value=str(len(backend._collision_include_pairs)),
        )

    last_collision_log_key = None
    last_collision_log_time = 0.0
    collision_debug_colors = (
        ((1.0, 0.2, 0.2), (0.2, 0.8, 0.2)),
        ((1.0, 0.5, 0.0), (0.3, 0.7, 1.0)),
        ((0.9, 0.2, 0.9), (0.2, 0.9, 0.9)),
        ((1.0, 0.9, 0.2), (0.2, 0.6, 1.0)),
    )
    collision_debug_points_a = [
        server.scene.add_icosphere(
            f"/collision_debug/point_a_{i}",
            radius=0.012,
            color=color_a,
            visible=False,
        )
        for i, (color_a, _color_b) in enumerate(collision_debug_colors)
    ]
    collision_debug_points_b = [
        server.scene.add_icosphere(
            f"/collision_debug/point_b_{i}",
            radius=0.012,
            color=color_b,
            visible=False,
        )
        for i, (_color_a, color_b) in enumerate(collision_debug_colors)
    ]
    collision_debug_lines = [None for _ in collision_debug_colors]

    def clear_collision_debug() -> None:
        for point in collision_debug_points_a + collision_debug_points_b:
            point.visible = False
        for line in collision_debug_lines:
            if line is not None:
                line.visible = False
        collision_debug_text.value = "Collision: --"

    last_collision_debug_update_time = 0.0

    def update_collision_debug(*, gpu_debug=None, force: bool = False) -> None:
        nonlocal collision_debug_lines
        nonlocal last_collision_debug_update_time
        nonlocal last_collision_log_key, last_collision_log_time
        gpu_rows = None if gpu_debug is None else [gpu_debug]
        cpu_debug_ready = backend.config.enable_collision and (
            hasattr(backend.solver, "get_last_collision_debug_list")
            or hasattr(backend.solver, "get_last_collision_debug")
        )
        if not show_collision_debug.value or (gpu_rows is None and not cpu_debug_ready):
            clear_collision_debug()
            return
        now = time.monotonic()
        if not force and now - last_collision_debug_update_time < 0.1:
            return
        last_collision_debug_update_time = now
        debug_rows = [] if gpu_rows is None else gpu_rows
        if gpu_rows is None and hasattr(
            backend.solver, "get_last_collision_debug_list"
        ):
            try:
                debug_rows = list(backend.solver.get_last_collision_debug_list())
            except Exception:
                debug_rows = []
        if (
            gpu_rows is None
            and not debug_rows
            and hasattr(backend.solver, "get_last_collision_debug")
        ):
            debug = backend.solver.get_last_collision_debug()
            debug_rows = [] if debug is None else [debug]
        if not debug_rows:
            clear_collision_debug()
            return
        summaries: list[str] = []
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
            segment = np.zeros((1, 2, 3), dtype=float)
            segment[0, 0, :] = p_a
            segment[0, 1, :] = p_b
            color_a, color_b = collision_debug_colors[i]
            collision_debug_lines[i] = server.scene.add_line_segments(
                f"/collision_debug/segment_{i}",
                points=segment,
                colors=np.array([[color_a, color_b]], dtype=float),
                line_width=3.0,
                visible=True,
            )
            vector = p_b - p_a
            summaries.append(
                f"{row.object_a} <-> {row.object_b} | "
                f"d={float(row.distance):.4f} m, "
                f"v=[{vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f}]"
            )
        for i in range(len(visible_rows), len(collision_debug_colors)):
            collision_debug_points_a[i].visible = False
            collision_debug_points_b[i].visible = False
            if collision_debug_lines[i] is not None:
                collision_debug_lines[i].visible = False
        collision_debug_text.value = " || ".join(summaries)
        first = visible_rows[0]
        now = time.time()
        log_key = (
            str(first.object_a),
            str(first.object_b),
            round(float(first.distance), 4),
            len(debug_rows),
        )
        if collision_log_mode.value and (
            log_key != last_collision_log_key or now - last_collision_log_time > 1.0
        ):
            print(
                "[spot-full-body][collision]",
                f"rows={len(debug_rows)}",
                f"{first.object_a} <-> {first.object_b}",
                f"d={float(first.distance):.4f} m",
                f"vector={np.asarray(first.point_b_world, dtype=float) - np.asarray(first.point_a_world, dtype=float)}",
            )
            last_collision_log_key = log_key
            last_collision_log_time = now

    def sync_visual(update_targets: bool = False) -> None:
        backend.robot.update_configuration(backend.q)
        base_pos, base_wxyz = _base_pose_to_viser(backend.q)
        base_node.position = base_pos
        base_node.wxyz = base_wxyz
        urdf_vis.update_cfg(_vis_cfg_from_q(backend, actuated_names))
        if update_targets:
            pos, wxyz = _pose_to_viser(backend.current_tool_pose())
            tool_target.position = pos
            tool_target.wxyz = wxyz
            pos, wxyz = _pose_to_viser(backend.current_torso_pose())
            torso_target.position = pos
            torso_target.wxyz = wxyz

    def sync_target_visibility() -> None:
        mode = SpotFullBodyIKMode(str(mode_dropdown.value))
        tool_visible = mode in {
            SpotFullBodyIKMode.ARM_TORSO,
            SpotFullBodyIKMode.FULL_BODY,
            SpotFullBodyIKMode.TWO_STAGE,
        }
        torso_visible = mode == SpotFullBodyIKMode.TORSO_ONLY
        tool_target.visible = tool_visible
        torso_target.visible = torso_visible
        snap_tool.disabled = not tool_visible
        snap_torso.disabled = not torso_visible

    @snap_tool.on_click
    def _(_) -> None:
        pos, wxyz = _pose_to_viser(backend.current_tool_pose())
        tool_target.position = pos
        tool_target.wxyz = wxyz

    @snap_torso.on_click
    def _(_) -> None:
        pos, wxyz = _pose_to_viser(backend.current_torso_pose())
        torso_target.position = pos
        torso_target.wxyz = wxyz

    def stow_arm() -> None:
        backend.set_arm_configuration(DEFAULT_ARM_COMMAND)
        backend.reanchor_feet_and_torso()
        sync_visual(update_targets=True)
        status.value = "ARM STOWED"

    def unstow_arm() -> None:
        backend.set_arm_configuration(INITIAL_ARM_COMMAND)
        backend.reanchor_feet_and_torso()
        sync_visual(update_targets=True)
        status.value = "ARM UNSTOWED"

    def reset_robot() -> None:
        backend.reset()
        if gpu_solver is not None:
            gpu_solver.reset_state()
        teleop_controller.reset_reference()
        sync_visual(update_targets=True)
        status.value = "RESET"

    @stow_btn.on_click
    def _(_) -> None:
        stow_arm()

    @unstow_btn.on_click
    def _(_) -> None:
        unstow_arm()

    @reset_btn.on_click
    def _(_) -> None:
        reset_robot()

    @reanchor_btn.on_click
    def _(_) -> None:
        backend.reanchor_feet_and_torso()
        status.value = "RE-ANCHORED"

    teleop_tool_start_pose = pose_from_transform_control(tool_target)

    def start_teleop_streaming() -> None:
        nonlocal teleop_tool_start_pose
        teleop_controller.reset_reference()
        teleop_tool_start_pose = pose_from_transform_control(tool_target)
        teleop_streaming_text.value = "ON"

    def stop_teleop_streaming() -> None:
        teleop_streaming_text.value = "OFF"

    def set_teleop_enabled() -> None:
        enabled = bool(teleop_enabled_checkbox.value) and teleop_controller.connected
        teleop_scale.disabled = not enabled
        teleop_controller.set_enabled(enabled)
        if enabled:
            teleop_controller.reset_reference()
        else:
            teleop_streaming_text.value = "OFF"

    teleop_controller.on_stream_start = start_teleop_streaming
    teleop_controller.on_stream_stop = stop_teleop_streaming
    teleop_controller.on_reset = reset_robot

    def update_teleop_target() -> None:
        teleop_connected_text.value = (
            "connected" if teleop_controller.connected else "browser only"
        )
        if not (teleop_controller.connected and bool(teleop_enabled_checkbox.value)):
            teleop_controller.set_enabled(False)
            teleop_gripper_text.value = "disabled"
            teleop_reset_text.value = "enable teleop"
            return
        teleop_controller.set_enabled(True)
        teleop_controller.process_buttons()
        trigger_fraction = float(getattr(teleop_controller, "trigger_fraction", 0.0))
        gripper_command = gripper_command_from_trigger_fraction(
            trigger_fraction,
            open_command=float(INITIAL_ARM_COMMAND[6]),
            closed_command=0.0,
        )
        backend.set_gripper_configuration(gripper_command)
        teleop_gripper_text.value = f"{100.0 * trigger_fraction:.0f}% closed"
        teleop_reset_text.value = "Button B"
        sync_visual(update_targets=False)
        if not (teleop_controller.streaming and tool_target.visible):
            return
        delta = teleop_controller.relative_pose()
        if delta is None:
            return
        delta_pos, delta_wxyz = delta
        pose = apply_controller_delta(
            teleop_tool_start_pose,
            delta_pos,
            delta_wxyz,
            float(teleop_scale.value),
        )
        set_transform_control_pose(tool_target, pose)

    teleop_controller.set_enabled(False)
    teleop_enabled_checkbox.on_update(lambda _: set_teleop_enabled())

    print(f"Viser: open the URL shown above. Spot URDF: {urdf_path}")
    if teleop_connected:
        print(
            "Seer teleop connected. Check 'Enable teleop', hold A to stream controller motion, "
            "use trigger travel for gripper command, and press B to reset."
        )
    sync_visual(update_targets=True)
    sync_target_visibility()
    loop_period_s = 1.0 / max(float(args.rate_hz), 1.0)
    last_gpu_active = None
    cpu_collision_enabled = bool(collision_enable.value)
    gpu_collision_enabled = bool(args.gpu_wbc_collision)
    cpu_show_collision_debug = bool(show_collision_debug.value)
    cpu_collision_log_mode = bool(collision_log_mode.value)
    last_cpu_debug_controls_enabled = None
    gpu_fixed_controls = (solve_mode,)
    try:
        while True:
            loop_start = time.perf_counter()
            update_teleop_target()
            gpu_active = (
                gpu_solver is not None
                and str(backend_dropdown.value) == "GPU Newton"
            )
            if gpu_active != last_gpu_active:
                constraint_available, debug_available = (
                    gpu_wbc_collision_control_availability(
                        gpu_active=gpu_active,
                        cpu_constraint_available=collision_available,
                        cpu_debug_available=collision_debug_available,
                        gpu_constraint_available=bool(
                            gpu_solver is not None and gpu_solver.collision_supported
                        ),
                        gpu_debug_available=bool(
                            gpu_solver is not None and gpu_solver.collision_supported
                        ),
                    )
                )
                if gpu_active:
                    gpu_solver.reset_state()
                    cpu_collision_enabled = bool(collision_enable.value)
                    cpu_show_collision_debug = bool(show_collision_debug.value)
                    cpu_collision_log_mode = bool(collision_log_mode.value)
                    collision_enable.value = bool(
                        constraint_available and gpu_collision_enabled
                    )
                    show_collision_debug.value = False
                    collision_log_mode.value = False
                    clear_collision_debug()
                    collision_debug_text.value = (
                        "GPU collision: clear"
                        if constraint_available
                        else "GPU collision debug unavailable"
                    )
                else:
                    collision_enable.value = bool(
                        constraint_available and cpu_collision_enabled
                    )
                    show_collision_debug.value = bool(
                        debug_available and cpu_show_collision_debug
                    )
                    collision_log_mode.value = bool(
                        debug_available and cpu_collision_log_mode
                    )
                mode_dropdown.disabled = False
                for control in gpu_fixed_controls:
                    control.disabled = gpu_active
                collision_enable.disabled = not constraint_available
                collision_min_dist_mm.disabled = not constraint_available
                collision_max_rows.disabled = gpu_active or not constraint_available
                collision_tuning.disabled = gpu_active or not constraint_available
                show_collision_debug.disabled = not debug_available
                collision_log_mode.disabled = gpu_active or not debug_available
                last_cpu_debug_controls_enabled = (
                    bool(debug_available and collision_enable.value)
                    if not gpu_active
                    else None
                )
                if not gpu_active:
                    show_collision_debug.disabled = not last_cpu_debug_controls_enabled
                    collision_log_mode.disabled = not last_cpu_debug_controls_enabled
                last_gpu_active = gpu_active
            elif not gpu_active:
                cpu_collision_enabled = bool(collision_enable.value)
                if not show_collision_debug.disabled:
                    cpu_show_collision_debug = bool(show_collision_debug.value)
                if not collision_log_mode.disabled:
                    cpu_collision_log_mode = bool(collision_log_mode.value)
                debug_enabled = bool(
                    collision_debug_available and collision_enable.value
                )
                if debug_enabled != last_cpu_debug_controls_enabled:
                    show_collision_debug.disabled = not debug_enabled
                    collision_log_mode.disabled = not debug_enabled
                    if debug_enabled:
                        show_collision_debug.value = cpu_show_collision_debug
                        collision_log_mode.value = cpu_collision_log_mode
                    else:
                        show_collision_debug.value = False
                        collision_log_mode.value = False
                        clear_collision_debug()
                    last_cpu_debug_controls_enabled = debug_enabled
            else:
                gpu_collision_enabled = bool(
                    collision_available and collision_enable.value
                )
                show_collision_debug.disabled = not (
                    collision_debug_available and gpu_collision_enabled
                )
                if not gpu_collision_enabled:
                    show_collision_debug.value = False
                    clear_collision_debug()

            backend.config.torso_pose_half_range = np.array(
                [
                    half_x.value,
                    half_y.value,
                    half_z.value,
                    np.deg2rad(float(half_r.value)),
                    np.deg2rad(float(half_r.value)),
                    np.deg2rad(float(half_r.value)),
                ],
                dtype=float,
            )
            backend.config.position_gain = float(pos_gain.value)
            backend.config.orientation_gain = float(rot_gain.value)
            backend.config.nullspace_gain = float(nullspace_gain.value)
            backend.config.use_contact_projection = True
            backend.config.enable_collision = bool(
                not gpu_active and collision_available and collision_enable.value
            )
            backend.config.collision_min_distance = (
                float(collision_min_dist_mm.value) * 1e-3
            )
            backend.config.collision_max_constraints = int(collision_max_rows.value)
            backend.config.collision_tuning_mode = str(collision_tuning.value)
            backend.config.target_solve_mode = str(solve_mode.value)
            backend.config.dt = float(dt.value)
            backend.config.max_steps = int(max_steps.value)
            backend.config.adaptive_dt = bool(adaptive_dt.value)
            backend.config.adaptive_dt_max_scale = float(adaptive_dt_max_scale.value)
            backend.config.adaptive_dt_reference_distance = float(
                adaptive_dt_ref_dist.value
            )
            backend.solver.dt = float(dt.value)

            tool_pose = target_pose_from_wxyz(
                np.asarray(tool_target.position, dtype=float),
                np.asarray(tool_target.wxyz, dtype=float),
            )
            torso_pose = target_pose_from_wxyz(
                np.asarray(torso_target.position, dtype=float),
                np.asarray(torso_target.wxyz, dtype=float),
            )
            mode = SpotFullBodyIKMode(str(mode_dropdown.value))
            sync_target_visibility()
            start = time.perf_counter()
            if gpu_active:
                gpu_solver.collision_enabled = gpu_collision_enabled
                result = gpu_solver.solve(
                    mode, tool_pose, torso_pose,
                    include_collision_debug=bool(show_collision_debug.value),
                )
            else:
                result = backend.solve(mode, tool_pose, torso_pose)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            solve_ms.value = f"{elapsed_ms:.3f} ms"
            status.value = _status_text(result)
            foot_error.value = f"{backend.foot_anchor_error() * 1e3:.1f} mm"
            posture_error = backend.posture_bias_error()
            nullspace_error.value = (
                f"base={posture_error['base_position'] * 1e3:.1f} mm, "
                f"arm={posture_error['arm']:.3f} rad, leg={posture_error['leg']:.3f} rad"
            )
            if gpu_active:
                # Keep both stage diagnostics in status; display the final endpoint.
                stages = result if isinstance(result, tuple) else (result,)
                result = stages[-1]
                if result.minimum_collision_distance_m is None:
                    collision_status.value = "GPU collision unavailable"
                elif any(stage.collision_overflow for stage in stages):
                    collision_status.value = "GPU overflow: safe hold"
                else:
                    state = "active" if result.collision_active else "clear"
                    collision_status.value = f"GPU {state}: {result.minimum_collision_distance_m * 1e3:.1f} mm"
                update_collision_debug(gpu_debug=result.collision_debug)
            else:
                collision_status.value = (
                    f"{len(backend._collision_include_pairs)}"
                    if backend.config.enable_collision
                    else f"off ({len(backend._collision_include_pairs)} available)"
                )
                update_collision_debug()
            sync_visual(update_targets=False)
            remaining = loop_period_s - (time.perf_counter() - loop_start)
            if remaining > 0.0:
                time.sleep(remaining)
    finally:
        teleop_controller.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf", type=Path, default=None, help="Spot whole-body URDF path."
    )
    parser.add_argument("--port", type=int, default=DEFAULT_VISER_PORT)
    parser.add_argument(
        "--enable-teleop",
        action="store_true",
        help="Enable Seer/xvisio controller teleoperation. Without this flag, xvisio is not imported.",
    )
    parser.add_argument(
        "--controller-port",
        default="/dev/ttyUSB0",
        help="Seer/xvisio controller serial port used with --enable-teleop.",
    )
    parser.add_argument(
        "--scale",
        "--teleop-scale",
        dest="teleop_scale",
        type=float,
        default=DEFAULT_TELEOP_SCALE_FACTOR,
        help="Position scale for Seer controller relative motion.",
    )
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--rate-hz", type=float, default=60.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--target-dx", type=float, default=0.04)
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in SpotFullBodyIKMode),
        default=SpotFullBodyIKMode.TWO_STAGE.value,
    )
    parser.add_argument(
        "--gpu-wbc",
        action="store_true",
        help="Enable experimental GPU solving for all Spot IK modes (layouts warm lazily).",
    )
    parser.add_argument(
        "--gpu-wbc-collision",
        action="store_true",
        help="Enable experimental pairwise-validated Newton collision recovery.",
    )
    parser.add_argument("--gpu-wbc-manifest", type=Path)
    parser.add_argument(
        "--gpu-wbc-solver-backend",
        choices=("torch_srinv", "cusolver_srinv", "warp_srinv"),
        default="torch_srinv",
        help="Select the experimental GPU spectral backend.",
    )
    parser.add_argument(
        "--gpu-wbc-collision-contacts-per-world",
        type=int,
        default=4096,
        help="Preallocated Newton contact capacity; overflow fails closed.",
    )
    parser.add_argument(
        "--gpu-wbc-collision-triangle-pairs-per-world",
        type=int,
        default=1024,
        help="Preallocated Newton mesh-pair capacity; overflow fails closed.",
    )
    parser.add_argument(
        "--gpu-wbc-collision-graph-repair-iterations",
        type=int,
        default=2,
        help="Fixed graph-captured collision repair passes (0 keeps safe rejection).",
    )
    parser.add_argument(
        "--gpu-wbc-certified-nominal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use a zero-repair certified graph with persistent full recovery for "
            "the experimental cuSOLVER backend."
        ),
    )
    parser.add_argument(
        "--gpu-wbc-candidate-convex-certificate",
        dest="gpu_wbc_candidate_convex_certificate_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the model-derived convex-envelope certificate to skip exact "
            "candidate mesh queries; uncertified candidates rerun exact recovery."
        ),
    )
    parser.add_argument(
        "--gpu-wbc-current-convex-certificate",
        dest="gpu_wbc_current_convex_certificate_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the model-derived convex envelope to certify exact-empty current "
            "collision queries; uncertified states rerun exact recovery."
        ),
    )
    parser.add_argument(
        "--gpu-wbc-cusolver-specialized-outputs",
        dest="gpu_wbc_cusolver_specialized_outputs_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip cuSOLVER inverse outputs unused by each GPU priority stage.",
    )
    parser.add_argument(
        "--gpu-wbc-reuse-locked-primary-inverse",
        dest="gpu_wbc_reuse_locked_primary_inverse_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Embed and reuse the primary cuSOLVER inverse for the first locked "
            "priority projector instead of refactorizing the same Jacobian."
        ),
    )
    parser.add_argument(
        "--gpu-wbc-warp-info",
        action="store_true",
        help="Show Warp initialization and cached-module load diagnostics.",
    )
    parser.add_argument(
        "--gpu-wbc-device-two-stage",
        action="store_true",
        help=(
            "Experimental cuSOLVER-only two-stage device handoff using one flattened "
            "CUDA graph and one host publication; disabled by default pending exact "
            "CPU-FK diagnostic parity."
        ),
    )
    parser.add_argument(
        "--gpu-wbc-cache-dir",
        type=Path,
        default=Path("build/gpu-wbc-newton-cache"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.headless:
        run_headless(args)
    else:
        run_viser(args)


if __name__ == "__main__":
    main()
