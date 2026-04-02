#!/usr/bin/env python3
"""Floating-base torso hierarchy: EE (primary) → torso upright (secondary) → nullspace (tertiary).

Default: **Viser** UI — sliders for torso pose box half-ranges, vel/acc limits, gains, and live
IK status while you drag the EE target.

Headless: ``--headless`` runs the original scripted benchmark loop (timing stats only).

Requirements: robot_descriptions, viser, yourdfpy (via robot_descriptions loader), pinocchio.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

import embodik as eik
from embodik import r2q

try:
    from robot_descriptions.loaders.yourdfpy import load_robot_description
except ImportError as exc:
    raise SystemExit(
        "robot_descriptions (with yourdfpy) is required: pip install robot_descriptions"
    ) from exc

try:
    import viser
    from viser.extras import ViserUrdf
except ImportError as exc:
    viser = None  # type: ignore[assignment]
    ViserUrdf = None  # type: ignore[assignment]
    _VISER_IMPORT_ERROR = exc
else:
    _VISER_IMPORT_ERROR = None

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from utils.robot_models import ensure_ros_package_path

try:
    from robot_descriptions.panda_description import URDF_PATH as PANDA_URDF_PATH
except ImportError as exc:
    raise SystemExit("robot_descriptions.panda_description is required") from exc

EE_FRAME = "panda_hand"
BASE_LINK = "panda_link0"  # Fixed URDF root link; floating-base world pose applied here in Viser.


PRESET_VALUES: dict[str, dict[str, float | bool | str]] = {
    "Responsive": {
        "enable_box": True,
        "enable_trans_bounds": True,
        "enable_rot_bounds": True,
        "half_txyz": 0.10,
        "half_rxyz_deg": 15.0,
        "vlim": 0.9,
        "alim": 1.0,
        "soften_bounds": True,
        "soften_fraction": 0.10,
        "torso_task_enabled": True,
        "primary_mode": "MIN_ERROR",
        "pos_gain": 12.0,
        "rot_gain": 12.0,
        "torso_ori_gain": 0.05,
        "ns_gain": 0.002,
        "max_iter": 5.0,
        "dt": 0.01,
        "allow_fallback": True,
        "slew_limit": False,
        "max_target_trans_step_mm": 25.0,
        "max_target_rot_step_deg": 10.0,
    },
    "Stable": {
        "enable_box": True,
        "enable_trans_bounds": True,
        "enable_rot_bounds": True,
        "half_txyz": 0.10,
        "half_rxyz_deg": 15.0,
        "vlim": 0.7,
        "alim": 0.8,
        "soften_bounds": True,
        "soften_fraction": 0.10,
        "torso_task_enabled": True,
        "primary_mode": "MIN_ERROR",
        "pos_gain": 10.0,
        "rot_gain": 10.0,
        "torso_ori_gain": 0.05,
        "ns_gain": 0.002,
        "max_iter": 7.0,
        "dt": 0.01,
        "allow_fallback": True,
        "slew_limit": True,
        "max_target_trans_step_mm": 12.0,
        "max_target_rot_step_deg": 4.0,
    },
    "StrictBounds": {
        "enable_box": True,
        "enable_trans_bounds": True,
        "enable_rot_bounds": True,
        "half_txyz": 0.08,
        "half_rxyz_deg": 10.0,
        "vlim": 0.5,
        "alim": 0.6,
        "soften_bounds": False,
        "soften_fraction": 0.10,
        "torso_task_enabled": True,
        "primary_mode": "MIN_ERROR",
        "pos_gain": 10.0,
        "rot_gain": 10.0,
        "torso_ori_gain": 0.06,
        "ns_gain": 0.0015,
        "max_iter": 10.0,
        "dt": 0.01,
        "allow_fallback": True,
        "slew_limit": True,
        "max_target_trans_step_mm": 8.0,
        "max_target_rot_step_deg": 3.0,
    },
}


def _resolve_torso_frame(robot: eik.RobotModel, ee_frame: str) -> str:
    candidates = ("torso", "torso_link", "base_link", "panda_link0", "pelvis", "root_link")
    frame_names = set(robot.get_frame_names())
    for name in candidates:
        if name in frame_names and name != ee_frame:
            return name
    for name in robot.get_frame_names():
        lname = name.lower()
        if ("torso" in lname or "base" in lname or "pelvis" in lname) and name != ee_frame:
            return name
    for name in robot.get_frame_names():
        if name != ee_frame:
            return name
    return ee_frame


def _init_floating_panda() -> tuple[eik.RobotModel, np.ndarray]:
    ensure_ros_package_path(Path(PANDA_URDF_PATH))
    robot = eik.RobotModel(str(PANDA_URDF_PATH), floating_base=True)
    q = np.asarray(robot.get_current_configuration(), dtype=float)
    if q.size >= 7:
        q[:7] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
    tail = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.05, 0.05], dtype=float)
    q[-tail.size :] = tail
    robot.update_configuration(q)
    return robot, q


def _parse_csv6(raw: str, flag_name: str) -> np.ndarray:
    arr = np.array([float(v.strip()) for v in raw.split(",") if v.strip()], dtype=float)
    if arr.size != 6:
        raise ValueError(
            f"{flag_name} must provide 6 comma-separated values "
            "[x,y,z,rx,ry,rz] with units [m,m,m,rad,rad,rad]"
        )
    return arr


def _vis_cfg_from_actuated(
    robot: eik.RobotModel, actuated_names: list[str], q_full: np.ndarray
) -> np.ndarray:
    """Map full configuration ``q_full`` to URDF actuated joint vector for ViserUrdf.

    Pinocchio ``q`` is **not** indexed like ``enumerate(get_joint_names())``: the floating
    base (e.g. ``root_joint``) occupies the first 7 scalars; arm and gripper joints follow.
    Using joint order as ``q`` indices shifts every value and breaks the gripper / meshes.
    """
    if not actuated_names:
        return np.array([], dtype=float)
    cfg_vec = np.zeros(len(actuated_names), dtype=float)
    for i, joint_name in enumerate(actuated_names):
        if not robot.has_joint(joint_name):
            continue
        jidx = robot.get_joint_config_index(joint_name)
        jsize = robot.get_joint_config_size(joint_name)
        if jsize == 1 and jidx < q_full.size:
            cfg_vec[i] = float(q_full[jidx])
        elif jsize > 1:
            # Prismatic / multi-DoF joints: URDF vis expects one entry per actuated row; copy first.
            end = min(jidx + jsize, q_full.size)
            if jidx < end:
                cfg_vec[i] = float(q_full[jidx])
    return cfg_vec


def _se3_wxyz_pos(pose: object) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    R = np.asarray(pose.rotation, dtype=float)
    t = np.asarray(pose.translation, dtype=float)
    quat_xyzw = r2q(R, order="xyzs")
    wxyz = (float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]))
    return (float(t[0]), float(t[1]), float(t[2])), wxyz


def _torso_rel6(ref_pose: object, cur_pose: object) -> np.ndarray:
    """6D delta vs reference: Δt world; log(R_ref^T R_cur) (matches solver box)."""
    from embodik import _embodik_impl as _impl

    Rr = np.asarray(ref_pose.rotation, dtype=float)
    tr = np.asarray(ref_pose.translation, dtype=float)
    Rc = np.asarray(cur_pose.rotation, dtype=float)
    tc = np.asarray(cur_pose.translation, dtype=float)
    rel = np.zeros(6, dtype=float)
    rel[:3] = tc - tr
    rel[3:] = _impl.log3(Rr.T @ Rc)
    return rel


def run_headless(args: argparse.Namespace) -> None:
    """Original scripted timing loop."""
    robot, q = _init_floating_panda()
    solver = eik.KinematicsSolver(robot)
    solver.dt = args.dt
    solver.set_damping(0.1)

    torso_frame = _resolve_torso_frame(robot, EE_FRAME)
    torso_pose = robot.get_frame_pose(torso_frame)
    torso_pose_bounds_ref = np.asarray(torso_pose.homogeneous(), dtype=float)
    ee_pose = robot.get_frame_pose(EE_FRAME)
    target_rot = np.asarray(ee_pose.rotation, dtype=float)
    center = np.asarray(ee_pose.translation, dtype=float)
    start_t = time.perf_counter()

    times_ms: list[float] = []
    for step in range(args.steps):
        phase = 0.06 * float(step)
        target_pos = center + np.array(
            [0.04 * math.cos(phase), 0.03 * math.sin(phase), 0.015 * math.sin(0.5 * phase)],
            dtype=float,
        )
        target = pin.SE3(target_rot, target_pos).homogeneous

        opts = eik.PositionIKOptions()
        opts.max_iterations = 1
        opts.dt = args.dt
        opts.position_gain = 40.0
        opts.orientation_gain = 40.0
        opts.primary_solve_mode = eik.TaskSolveMode.SCALE
        opts.primary_allow_min_error_fallback = False
        opts.nullspace_bias = q.copy()
        opts.nullspace_gain = 2e-3
        active = list(range(robot.nv))
        opts.nullspace_active_joints = active
        opts.nullspace_joint_weights = np.ones(len(active), dtype=float)

        opts.torso_constraint.enabled = True
        opts.torso_constraint.frame_name = torso_frame
        opts.torso_constraint.target_orientation = np.asarray(torso_pose.rotation, dtype=float)
        opts.torso_constraint.orientation_mask = np.array([1.0, 1.0, 0.0], dtype=float)
        opts.torso_constraint.orientation_gain = 8e-2
        if args.enable_torso_pose_constraints:
            lower = (
                _parse_csv6(args.torso_pose_lower_bounds, "--torso-pose-lower-bounds")
                if args.torso_pose_lower_bounds
                else None
            )
            upper = (
                _parse_csv6(args.torso_pose_upper_bounds, "--torso-pose-upper-bounds")
                if args.torso_pose_upper_bounds
                else None
            )
            if (lower is None) != (upper is None):
                raise ValueError(
                    "Set both --torso-pose-lower-bounds and --torso-pose-upper-bounds "
                    "to use asymmetric limits"
                )
            if lower is not None and upper is not None:
                opts.torso_constraint.pose_lower_bounds = lower
                opts.torso_constraint.pose_upper_bounds = upper
            else:
                half = _parse_csv6(args.torso_pose_half_range, "--torso-pose-half-range")
                opts.torso_constraint.pose_lower_bounds = -half
                opts.torso_constraint.pose_upper_bounds = half
            opts.torso_constraint.pose_axis_mask = np.ones(6, dtype=float)
            opts.torso_constraint.velocity_limits = np.full(6, 0.5, dtype=float)
            opts.torso_constraint.acceleration_limits = np.full(6, 1.0, dtype=float)
            opts.torso_constraint.pose_bounds_reference_pose = torso_pose_bounds_ref

        t0 = time.perf_counter()
        out = solver.solve_position(q, target, EE_FRAME, opts)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        if out.status in (
            eik.SolverStatus.SUCCESS,
            eik.SolverStatus.INFEASIBLE,
            eik.SolverStatus.NO_PROGRESS,
        ):
            q = np.asarray(out.q_solution, dtype=float)
            robot.update_configuration(q)

    elapsed = time.perf_counter() - start_t
    samples = np.asarray(times_ms, dtype=float)
    print(f"Torso frame: {torso_frame}")
    print(
        "solve_position timing "
        f"mean={samples.mean():.4f}ms p95={np.percentile(samples, 95):.4f}ms "
        f"p99={np.percentile(samples, 99):.4f}ms max={samples.max():.4f}ms"
    )
    print(f"Completed {args.steps} steps in {elapsed:.3f}s")


def run_viser(args: argparse.Namespace) -> None:
    if viser is None or ViserUrdf is None:
        raise SystemExit(
            "Viser mode requires viser. Install with: pip install viser\n"
            f"Import error: {_VISER_IMPORT_ERROR}"
        )

    urdf = load_robot_description("panda_description")

    robot, q_init = _init_floating_panda()
    q = q_init.copy()
    robot.update_configuration(q)
    solver = eik.KinematicsSolver(robot)
    solver.dt = args.dt
    solver.set_damping(float(args.damping))

    torso_frame = _resolve_torso_frame(robot, EE_FRAME)
    torso_pose_ref_handle = robot.get_frame_pose(torso_frame)
    torso_pose_bounds_ref = np.asarray(torso_pose_ref_handle.homogeneous(), dtype=float)

    ee_pose = robot.get_frame_pose(EE_FRAME)
    pos0, wxyz0 = _se3_wxyz_pos(ee_pose)

    server = viser.ViserServer(port=args.port, label="Floating-base torso hierarchy")
    server.scene.add_grid("/ground", width=4, height=4)

    base_node = server.scene.add_frame(
        "/floating_base",
        position=(0.0, 0.0, 0.0),
        wxyz=(1.0, 0.0, 0.0, 0.0),
        show_axes=True,
        axes_length=0.15,
    )
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/floating_base/robot")

    actuated_names = list(getattr(urdf_vis._urdf, "actuated_joint_names", []))

    def vis_cfg_from_q(q_full: np.ndarray) -> np.ndarray:
        return _vis_cfg_from_actuated(robot, actuated_names, q_full)

    def sync_base_visual() -> None:
        plk = robot.get_frame_pose(BASE_LINK)
        pos, wxyz = _se3_wxyz_pos(plk)
        base_node.position = pos
        base_node.wxyz = wxyz

    ik_target = server.scene.add_transform_controls(
        "/ik_target",
        scale=0.18,
        position=pos0,
        wxyz=wxyz0,
    )

    # --- GUI: bounds & limits (half-range per axis) ---
    with server.gui.add_folder("Torso pose box (vs anchored reference)"):
        enable_box = server.gui.add_checkbox("Enable torso pose bounds", initial_value=True)
        enable_trans_bounds = server.gui.add_checkbox("Enable translation bounds (x,y,z)", initial_value=True)
        enable_rot_bounds = server.gui.add_checkbox("Enable rotation bounds (rx,ry,rz)", initial_value=True)
        lock_opt_checkbox = server.gui.add_checkbox(
            "Optimize full lock with base joint lock (fixed-base emulation)", initial_value=False
        )
        reanchor_btn = server.gui.add_button("Re-anchor bounds to current torso")
        half_tx = server.gui.add_slider("±x half (m)", 0.0, 0.2, initial_value=0.10, step=0.005)
        half_ty = server.gui.add_slider("±y half (m)", 0.0, 0.2, initial_value=0.10, step=0.005)
        half_tz = server.gui.add_slider("±z half (m)", 0.0, 0.2, initial_value=0.10, step=0.005)
        half_rrx = server.gui.add_slider("±rx half (deg)", 0.0, 45.0, initial_value=15.0, step=0.5)
        half_rry = server.gui.add_slider("±ry half (deg)", 0.0, 45.0, initial_value=15.0, step=0.5)
        half_rrz = server.gui.add_slider("±rz half (deg)", 0.0, 45.0, initial_value=15.0, step=0.5)
        vlim = server.gui.add_slider("Vel limit (all axes)", 0.05, 1.0, initial_value=0.8, step=0.05)
        alim = server.gui.add_slider("Accel limit (all axes)", 0.1, 1.0, initial_value=1.0, step=0.05)
        soften_bounds = server.gui.add_checkbox(
            "Enable torso bound softening", initial_value=True
        )
        soften_frac = server.gui.add_slider(
            "Bound softening fraction", 0.0, 0.4, initial_value=0.10, step=0.01
        )

    with server.gui.add_folder("IK / torso task"):
        preset_dropdown = server.gui.add_dropdown(
            "Behavior preset",
            tuple(PRESET_VALUES.keys()),
            initial_value="Stable",
        )
        apply_preset_btn = server.gui.add_button("Apply preset")
        torso_task_enabled = server.gui.add_checkbox(
            "Enable torso secondary orientation task", initial_value=True
        )
        primary_mode = server.gui.add_dropdown(
            "Primary solve mode", ("MIN_ERROR", "SCALE", "SCALE_ELASTIC"), initial_value="MIN_ERROR"
        )
        pos_gain_s = server.gui.add_slider("EE position gain", 1.0, 80.0, initial_value=10.0, step=1.0)
        rot_gain_s = server.gui.add_slider("EE orientation gain", 1.0, 80.0, initial_value=10.0, step=1.0)
        torso_ori_gain_s = server.gui.add_slider(
            "Torso orientation gain", 0.0, 0.3, initial_value=0.05, step=0.005
        )
        ns_gain_s = server.gui.add_slider("Nullspace gain", 0.0, 0.02, initial_value=0.002, step=0.0005)
        max_iter_s = server.gui.add_slider("max_iterations", 1, 25, initial_value=10, step=1)
        dt_s = server.gui.add_slider("dt (s)", 0.005, 0.03, initial_value=args.dt, step=0.001)
        allow_fallback = server.gui.add_checkbox(
            "Allow min-error fallback (partial EE fit)", initial_value=True
        )
        slew_limit_checkbox = server.gui.add_checkbox(
            "Limit target step per frame", initial_value=False
        )
        max_target_trans_step_mm = server.gui.add_slider(
            "Max target translation step (mm/frame)", 1.0, 80.0, initial_value=25.0, step=1.0
        )
        max_target_rot_step_deg = server.gui.add_slider(
            "Max target rotation step (deg/frame)", 0.5, 20.0, initial_value=10.0, step=0.5
        )
        snap_target_btn = server.gui.add_button("Snap Target to Current EE")
        reset_cfg_btn = server.gui.add_button("Reset configuration")

    with server.gui.add_folder("Status"):
        status_txt = server.gui.add_text("IK status", initial_value="—")
        err_txt = server.gui.add_text("EE errors", initial_value="—")
        torso_txt = server.gui.add_text("Torso Δ vs anchor (6D)", initial_value="—")
        torso_slack_txt = server.gui.add_text("Torso box min slack", initial_value="—")
        infeasible_txt = server.gui.add_text("Infeasible streak", initial_value="0")
        pressure_txt = server.gui.add_text("Constraint pressure", initial_value="—")
        timing_txt = server.gui.add_number("Last solve (ms)", 0.001, disabled=True)

    def reanchor_callback(_: object) -> None:
        nonlocal torso_pose_bounds_ref, torso_pose_ref_handle
        robot.update_configuration(q)
        torso_pose_ref_handle = robot.get_frame_pose(torso_frame)
        torso_pose_bounds_ref = np.asarray(torso_pose_ref_handle.homogeneous(), dtype=float)

    def sync_from_state(update_target: bool = False) -> None:
        """Keep visualization/controls consistent with current q (same pattern as examples 01/02)."""
        robot.update_configuration(q)
        sync_base_visual()
        urdf_vis.update_cfg(vis_cfg_from_q(q))
        if update_target:
            ee_pose_cur = robot.get_frame_pose(EE_FRAME)
            ee_pos, ee_wxyz = _se3_wxyz_pos(ee_pose_cur)
            ik_target.position = ee_pos
            ik_target.wxyz = ee_wxyz

    reanchor_btn.on_click(reanchor_callback)
    prev_enable_box = bool(enable_box.value)
    last_torso_min_slack = float("nan")

    def sync_bounds_toggle_enabled_state() -> None:
        bounds_on = bool(enable_box.value)
        enable_trans_bounds.disabled = not bounds_on
        enable_rot_bounds.disabled = not bounds_on
        lock_opt_checkbox.disabled = not bounds_on

    def apply_preset(preset_name: str) -> None:
        preset = PRESET_VALUES[preset_name]
        enable_box.value = bool(preset["enable_box"])
        enable_trans_bounds.value = bool(preset["enable_trans_bounds"])
        enable_rot_bounds.value = bool(preset["enable_rot_bounds"])
        half_tx.value = float(preset["half_txyz"])
        half_ty.value = float(preset["half_txyz"])
        half_tz.value = float(preset["half_txyz"])
        half_rrx.value = float(preset["half_rxyz_deg"])
        half_rry.value = float(preset["half_rxyz_deg"])
        half_rrz.value = float(preset["half_rxyz_deg"])
        vlim.value = float(preset["vlim"])
        alim.value = float(preset["alim"])
        soften_bounds.value = bool(preset["soften_bounds"])
        soften_frac.value = float(preset["soften_fraction"])
        torso_task_enabled.value = bool(preset["torso_task_enabled"])
        primary_mode.value = str(preset["primary_mode"])
        pos_gain_s.value = float(preset["pos_gain"])
        rot_gain_s.value = float(preset["rot_gain"])
        torso_ori_gain_s.value = float(preset["torso_ori_gain"])
        ns_gain_s.value = float(preset["ns_gain"])
        max_iter_s.value = int(round(float(preset["max_iter"])))
        dt_s.value = float(preset["dt"])
        allow_fallback.value = bool(preset["allow_fallback"])
        slew_limit_checkbox.value = bool(preset["slew_limit"])
        max_target_trans_step_mm.value = float(preset["max_target_trans_step_mm"])
        max_target_rot_step_deg.value = float(preset["max_target_rot_step_deg"])
        status_txt.value = f"PRESET | applied {preset_name}"

    def apply_preset_callback(_: object) -> None:
        apply_preset(str(preset_dropdown.value))

    apply_preset_btn.on_click(apply_preset_callback)
    apply_preset("Stable")
    sync_bounds_toggle_enabled_state()

    @server.on_client_connect
    def _(_client: object) -> None:
        sync_from_state(update_target=True)

    print(f"Viser: open the URL shown above. Torso frame: {torso_frame}")
    print("Drag the IK target. Adjust sliders to change pose bounds and limits.")
    if actuated_names:
        miss = [n for n in actuated_names if not robot.has_joint(n)]
        if miss:
            print(f"[warn] URDF actuated joints not in embodik model: {miss}")

    target_homog = np.eye(4, dtype=float, order="F")
    filtered_target_pos = np.asarray(robot.get_frame_pose(EE_FRAME).translation, dtype=float).copy()
    filtered_target_R = np.asarray(robot.get_frame_pose(EE_FRAME).rotation, dtype=float).copy()
    q_lower, q_upper = robot.get_joint_limits()
    # For floating-base systems, keep nullspace posture on articulated joints by default.
    nullspace_active = list(range(6, robot.nv)) if robot.nv > 6 else list(range(robot.nv))
    last_feasible_q = q.copy()
    infeasible_streak = 0
    prev_position_error = float("inf")
    near_target_pos_m = 5e-3
    near_target_rot_rad = 3e-2
    target_hold_homog = np.eye(4, dtype=float, order="F")
    target_hold_frames = 0
    post_reset_freeze_frames = 0

    def snap_target_callback(_: object) -> None:
        nonlocal target_hold_homog, target_hold_frames, filtered_target_pos, filtered_target_R
        sync_from_state(update_target=True)
        ee_pose_cur = robot.get_frame_pose(EE_FRAME)
        filtered_target_pos = np.asarray(ee_pose_cur.translation, dtype=float).copy()
        filtered_target_R = np.asarray(ee_pose_cur.rotation, dtype=float).copy()
        target_hold_homog[:, :] = np.asarray(ee_pose_cur.homogeneous(), dtype=float)
        target_hold_frames = 3
        status_txt.value = "SNAP | target set to current EE"

    snap_target_btn.on_click(snap_target_callback)

    def reset_config_callback(_: object) -> None:
        nonlocal q, last_feasible_q, infeasible_streak, target_hold_homog, target_hold_frames, post_reset_freeze_frames, filtered_target_pos, filtered_target_R
        q[:] = q_init.copy()
        last_feasible_q[:] = q
        infeasible_streak = 0

        # Same reset semantics as examples 01/02: reset state, then sync visuals/target.
        sync_from_state(update_target=True)
        reanchor_callback(None)
        ee_pose_cur = robot.get_frame_pose(EE_FRAME)
        filtered_target_pos = np.asarray(ee_pose_cur.translation, dtype=float).copy()
        filtered_target_R = np.asarray(ee_pose_cur.rotation, dtype=float).copy()
        target_hold_homog[:, :] = np.asarray(ee_pose_cur.homogeneous(), dtype=float)
        # Hold target briefly so reset cannot be immediately undone by stale control state.
        target_hold_frames = 30
        # Also freeze IK briefly so reset is clearly visible.
        post_reset_freeze_frames = 30
        infeasible_txt.value = "0"
        status_txt.value = "RESET | robot and target synced"
        err_txt.value = "pos=— | rot=—"
        timing_txt.value = 0.0
        print("[info] Reset configuration to initial pose.")

    reset_cfg_btn.on_click(reset_config_callback)
    while True:
        # Keep loop responsive like examples 01/02.
        sleep_dt = 1e-3
        nonlocal_enable = bool(enable_box.value)
        sync_bounds_toggle_enabled_state()
        if nonlocal_enable and not prev_enable_box:
            reanchor_callback(None)
            print("[info] Torso pose bounds enabled: anchor refreshed to current torso pose.")
        prev_enable_box = nonlocal_enable

        sync_from_state(update_target=False)
        if post_reset_freeze_frames > 0:
            post_reset_freeze_frames -= 1
            sync_from_state(update_target=True)
            status_txt.value = "RESETTING | holding reset pose"
            err_txt.value = "pos=— | rot=—"
            time.sleep(sleep_dt)
            continue

        if target_hold_frames > 0:
            target_homog[:, :] = target_hold_homog
            target_hold_frames -= 1
        else:
            R_tgt = ik_target.wxyz
            w, x, y, z = R_tgt[0], R_tgt[1], R_tgt[2], R_tgt[3]
            quat_xyzw = np.array([x, y, z, w], dtype=float)
            desired_R = pin.Quaternion(quat_xyzw).toRotationMatrix()
            desired_pos = np.array(ik_target.position, dtype=float)

            if slew_limit_checkbox.value:
                max_dp = float(max_target_trans_step_mm.value) * 1e-3
                max_dr = np.deg2rad(float(max_target_rot_step_deg.value))

                # Translation slew-rate limiting.
                dp = desired_pos - filtered_target_pos
                dp_norm = float(np.linalg.norm(dp))
                if dp_norm > max_dp and dp_norm > 1e-12:
                    filtered_target_pos += (max_dp / dp_norm) * dp
                else:
                    filtered_target_pos = desired_pos

                # Rotation slew-rate limiting using log/exp on SO(3).
                R_rel = filtered_target_R.T @ desired_R
                rot_vec = eik.log3(R_rel)
                ang = float(np.linalg.norm(rot_vec))
                if ang > max_dr and ang > 1e-12:
                    rot_vec = (max_dr / ang) * rot_vec
                filtered_target_R = filtered_target_R @ eik.exp3(rot_vec)
            else:
                filtered_target_pos = desired_pos
                filtered_target_R = desired_R

            target_homog[:3, :3] = filtered_target_R
            target_homog[:3, 3] = filtered_target_pos

        half = np.array(
            [
                half_tx.value,
                half_ty.value,
                half_tz.value,
                np.deg2rad(half_rrx.value),
                np.deg2rad(half_rry.value),
                np.deg2rad(half_rrz.value),
            ],
            dtype=float,
        )
        # If translation/rotation toggles are unchecked, keep those axes constrained
        # but with epsilon bounds so they are effectively locked in place.
        eps_trans = 1e-4  # meters
        eps_rot = 1e-3  # radians
        effective_half = half.copy()
        if not enable_trans_bounds.value:
            effective_half[:3] = eps_trans
        if not enable_rot_bounds.value:
            effective_half[3:] = eps_rot

        opts = eik.PositionIKOptions()
        opts.max_iterations = int(max_iter_s.value)
        opts.dt = float(dt_s.value)
        solver.dt = opts.dt
        opts.position_gain = float(pos_gain_s.value)
        opts.orientation_gain = float(rot_gain_s.value)
        opts.primary_solve_mode = getattr(
            eik.TaskSolveMode, primary_mode.value, eik.TaskSolveMode.SCALE
        )
        opts.primary_allow_min_error_fallback = bool(allow_fallback.value)
        # Keep behavior closer to example 01: avoid early NO_PROGRESS exits on large target jumps.
        opts.classify_stagnation_as_no_progress = False
        opts.limit_change_from_seed = False
        opts.stagnation_iterations = max(int(max_iter_s.value), 8)
        opts.nullspace_bias = q.copy()
        opts.nullspace_gain = float(ns_gain_s.value)
        opts.nullspace_active_joints = nullspace_active
        opts.nullspace_joint_weights = np.ones(len(nullspace_active), dtype=float)

        torso_secondary_enabled = bool(torso_task_enabled.value)
        torso_bounds_enabled = bool(enable_box.value)
        full_lock_active = (
            torso_bounds_enabled and (not enable_trans_bounds.value) and (not enable_rot_bounds.value)
        )
        use_base_joint_lock_optimization = bool(lock_opt_checkbox.value) and full_lock_active and robot.nv >= 6
        torso_secondary_effective = torso_secondary_enabled and (not use_base_joint_lock_optimization)
        torso_bounds_effective = torso_bounds_enabled and (not use_base_joint_lock_optimization)

        if use_base_joint_lock_optimization:
            # Full epsilon lock on both translation+rotation is equivalent to a
            # floating-base velocity lock for this example; use direct joint locking
            # to avoid overly stiff torso box rows that can stall arm updates.
            opts.excluded_joint_indices = list(range(6))

        if torso_secondary_effective or torso_bounds_effective:
            opts.torso_constraint.enabled = True
            opts.torso_constraint.frame_name = torso_frame
            if torso_secondary_effective:
                opts.torso_constraint.target_orientation = np.asarray(
                    torso_pose_ref_handle.rotation, dtype=float
                )
                opts.torso_constraint.orientation_mask = np.array([1.0, 1.0, 0.0], dtype=float)
                opts.torso_constraint.orientation_gain = float(torso_ori_gain_s.value)
            else:
                opts.torso_constraint.orientation_mask = np.zeros(3, dtype=float)
                opts.torso_constraint.orientation_gain = 0.0

            if torso_bounds_effective:
                opts.torso_constraint.pose_lower_bounds = -effective_half
                opts.torso_constraint.pose_upper_bounds = effective_half
                opts.torso_constraint.pose_axis_mask = np.ones(6, dtype=float)
                vl = float(vlim.value)
                al = float(alim.value)
                opts.torso_constraint.velocity_limits = np.full(6, vl, dtype=float)
                opts.torso_constraint.acceleration_limits = np.full(6, al, dtype=float)
                opts.torso_constraint.pose_bound_softening_enabled = bool(soften_bounds.value)
                opts.torso_constraint.pose_bound_softening_fraction = float(soften_frac.value)
                opts.torso_constraint.pose_bounds_reference_pose = torso_pose_bounds_ref

        t0 = time.perf_counter()
        out = solver.solve_position(q, target_homog, EE_FRAME, opts)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if out.status == eik.SolverStatus.SUCCESS:
            # Use embodiK/pinocchio-integrated configuration directly.
            # Avoid blanket clipping on floating base, which can corrupt quaternion consistency.
            q_next = np.asarray(out.q_solution, dtype=float).copy()
            if q_next.size >= 7:
                quat = q_next[3:7]
                quat_norm = float(np.linalg.norm(quat))
                if quat_norm > 1e-12:
                    q_next[3:7] = quat / quat_norm
            # Clamp articulated joints only; keep free-flyer part untouched.
            if q_next.size > 7:
                q_next[7:] = np.clip(q_next[7:], q_lower[7:], q_upper[7:])
            q[:] = q_next
            last_feasible_q[:] = q
            infeasible_streak = 0
            prev_position_error = float(out.position_error)
        elif out.status in (eik.SolverStatus.INFEASIBLE, eik.SolverStatus.NO_PROGRESS):
            # In full base-lock optimization mode, prefer fixed-base semantics:
            # apply non-success candidate directly to avoid visible hold/stall.
            if use_base_joint_lock_optimization:
                q_next = np.asarray(out.q_solution, dtype=float).copy()
                if q_next.size >= 7:
                    quat = q_next[3:7]
                    quat_norm = float(np.linalg.norm(quat))
                    if quat_norm > 1e-12:
                        q_next[3:7] = quat / quat_norm
                if q_next.size > 7:
                    q_next[7:] = np.clip(q_next[7:], q_lower[7:], q_upper[7:])
                q[:] = q_next
                last_feasible_q[:] = q
                prev_position_error = float(out.position_error)
                infeasible_streak = 0
            else:
                # Default behavior: accept non-success candidate only when it
                # still improves EE position error, otherwise hold state.
                improved = float(out.position_error) < (prev_position_error - 1e-6)
                if improved:
                    q_next = np.asarray(out.q_solution, dtype=float).copy()
                    if q_next.size >= 7:
                        quat = q_next[3:7]
                        quat_norm = float(np.linalg.norm(quat))
                        if quat_norm > 1e-12:
                            q_next[3:7] = quat / quat_norm
                    if q_next.size > 7:
                        q_next[7:] = np.clip(q_next[7:], q_lower[7:], q_upper[7:])
                    q[:] = q_next
                    last_feasible_q[:] = q
                    prev_position_error = float(out.position_error)
                    infeasible_streak = 0
                else:
                    q[:] = last_feasible_q
                    near_target = (
                        float(out.position_error) <= near_target_pos_m
                        and float(out.orientation_error) <= near_target_rot_rad
                    )
                    if near_target:
                        # Avoid runaway "infeasible streak" when the target is already
                        # effectively reached but constraints/tolerances block exact success.
                        infeasible_streak = max(0, infeasible_streak - 1)
                    else:
                        infeasible_streak += 1
        elif out.status == eik.SolverStatus.NUMERICAL_ERROR:
            # Keep the last known-feasible configuration so one bad solve does not
            # poison subsequent frames.
            q[:] = last_feasible_q
            infeasible_streak += 1

        robot.update_configuration(q)
        ref_se3 = pin.SE3(torso_pose_bounds_ref)
        rel6 = _torso_rel6(ref_se3, robot.get_frame_pose(torso_frame))
        if torso_bounds_effective:
            lower_slack = rel6 - (-effective_half)
            upper_slack = effective_half - rel6
            min_slack = float(np.minimum(lower_slack, upper_slack).min())
            last_torso_min_slack = min_slack
            torso_slack_txt.value = f"{min_slack:.4f} ({'inside' if min_slack >= 0.0 else 'outside'})"
        elif enable_box.value and use_base_joint_lock_optimization:
            last_torso_min_slack = float("nan")
            torso_slack_txt.value = "joint-lock optimization active"
        else:
            last_torso_min_slack = float("nan")
            torso_slack_txt.value = "bounds disabled"

        status_txt.value = f"{out.status.name} | iters={out.iterations_used}"
        if use_base_joint_lock_optimization and torso_secondary_enabled:
            status_txt.value += " | torso-secondary auto-off"
        if (
            out.status in (eik.SolverStatus.INFEASIBLE, eik.SolverStatus.NO_PROGRESS)
            and float(out.position_error) <= near_target_pos_m
            and float(out.orientation_error) <= near_target_rot_rad
        ):
            status_txt.value += " | near-target constrained"
        err_txt.value = f"pos={out.position_error*1e3:.2f} mm | rot={out.orientation_error:.4f} rad"
        infeasible_txt.value = str(infeasible_streak)
        torso_txt.value = (
            f"Δxyz=[{rel6[0]:.3f},{rel6[1]:.3f},{rel6[2]:.3f}] m "
            f"Δrpy~[{rel6[3]:.3f},{rel6[4]:.3f},{rel6[5]:.3f}] rad"
        )
        if not enable_box.value:
            pressure_txt.value = "Low (no torso box)"
        elif use_base_joint_lock_optimization:
            pressure_txt.value = "Low (joint-lock optimization)"
        else:
            if not np.isfinite(last_torso_min_slack):
                pressure_txt.value = "N/A"
            else:
                if last_torso_min_slack < 0.0 or infeasible_streak >= 4:
                    level = "High"
                elif last_torso_min_slack < 0.01 or infeasible_streak >= 2:
                    level = "Medium"
                else:
                    level = "Low"
                pressure_txt.value = (
                    f"{level} | slack={last_torso_min_slack:.4f} | streak={infeasible_streak:d}"
                )
        timing_txt.value = float(elapsed_ms)

        time.sleep(sleep_dt)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--headless",
        action="store_true",
        help="Run scripted timing benchmark (no Viser).",
    )
    p.add_argument("--port", type=int, default=8080, help="Viser server port (default: 8080).")
    p.add_argument("--steps", type=int, default=5000, help="Headless: number of IK steps.")
    p.add_argument("--dt", type=float, default=0.01, help="Solver / integration timestep (s).")
    p.add_argument("--damping", type=float, default=0.01, help="Viser: solver damping.")
    p.add_argument(
        "--enable-torso-pose-constraints",
        action="store_true",
        help="Headless: enable 6D torso pose box (symmetric unless lower/upper set).",
    )
    p.add_argument(
        "--torso-pose-half-range",
        type=str,
        default="0.08,0.08,0.08,0.21,0.21,0.21",
        help="Headless: 6 CSV half-ranges [m,m,m,rad,rad,rad] when asymmetric flags unset.",
    )
    p.add_argument(
        "--torso-pose-lower-bounds",
        type=str,
        default=None,
        help="Headless: optional 6 CSV lower bounds (requires upper bounds too).",
    )
    p.add_argument(
        "--torso-pose-upper-bounds",
        type=str,
        default=None,
        help="Headless: optional 6 CSV upper bounds (requires lower bounds too).",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.headless:
        run_headless(args)
    else:
        run_viser(args)


if __name__ == "__main__":
    main()
