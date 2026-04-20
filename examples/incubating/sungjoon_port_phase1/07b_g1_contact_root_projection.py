#!/usr/bin/env python3
"""G1 dual-hand demo comparing contact-root projection vs tight constraints."""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import embodik
from embodik.utils import q2r, r2q

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.incubating.sungjoon_port_phase1.g1_viser_utils import (
    compute_feet_center,
    compute_support_polygon_from_foot_poses,
    create_g1_robot_and_visual,
    ground_floating_base_from_feet_center,
    make_visual_config_mapper,
    polygon_segments_xy,
    resolve_frames_for_g1_base_mode,
    validate_robot_collision_model,
)
from examples.incubating.sungjoon_port_phase1.robust_ik_runtime import configure_primary_solve_mode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8088)
    return p.parse_args()


def _clip_q(robot, q: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).copy()
    if getattr(robot, "is_floating_base", False) and q.size >= 7:
        q[7:] = np.clip(q[7:], q_lo[7:], q_hi[7:])
        quat = q[3:7]
        n = np.linalg.norm(quat)
        q[3:7] = quat / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return q
    return np.clip(q, q_lo, q_hi)


def _pose_from_ctrl(ctrl) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = np.asarray(ctrl.position, dtype=float)
    wxyz = np.asarray(ctrl.wxyz, dtype=float)
    n = np.linalg.norm(wxyz)
    if not np.isfinite(n) or n < 1e-12:
        wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        wxyz = wxyz / n
    pose[:3, :3] = q2r(np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=float), order="xyzs")
    return pose


def _frame_delta6(anchor_pose: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
    delta = np.zeros(6, dtype=float)
    delta[:3] = np.asarray(current_pose[:3, 3] - anchor_pose[:3, 3], dtype=float)
    delta[3:] = np.asarray(
        embodik.log3(np.asarray(anchor_pose[:3, :3], dtype=float).T @ np.asarray(current_pose[:3, :3], dtype=float)),
        dtype=float,
    )
    return delta


def _generate_consecutive_collision_exclusions(robot) -> list[tuple[str, str]]:
    """Exclude likely adjacent/consecutive link collision pairs."""
    if not hasattr(robot, "get_collision_pair_names") or not hasattr(robot, "get_collision_geometries"):
        return []
    try:
        pair_names = list(robot.get_collision_pair_names())
        geoms = list(robot.get_collision_geometries())
    except Exception:
        return []

    parent_joint_by_geom: dict[str, int] = {}
    parent_frame_by_geom: dict[str, str] = {}
    for g in geoms:
        name = str(g.get("name", ""))
        if not name:
            continue
        parent_frame_by_geom[name] = str(g.get("parent_frame", ""))
        try:
            parent_joint_by_geom[name] = int(g.get("parent_joint", -10_000))
        except Exception:
            parent_joint_by_geom[name] = -10_000

    exclusions: list[tuple[str, str]] = []
    for a, b in pair_names:
        a = str(a)
        b = str(b)
        ja = parent_joint_by_geom.get(a, -10_000)
        jb = parent_joint_by_geom.get(b, -10_000)
        fa = parent_frame_by_geom.get(a, "")
        fb = parent_frame_by_geom.get(b, "")
        # Same parent frame or immediately neighboring joints in the kinematic tree.
        if (fa and fb and fa == fb) or (ja > -9999 and jb > -9999 and abs(ja - jb) <= 1):
            exclusions.append((a, b))
    return exclusions


def _apply_zero_based_task_hierarchy(
    primary_tasks: list,
    secondary_tasks: list,
    tertiary_tasks: list,
) -> None:
    """Assign solver priorities with an explicit zero-based hierarchy."""
    for t in primary_tasks:
        t.priority = 0
    for t in secondary_tasks:
        t.priority = 1
    for t in tertiary_tasks:
        t.priority = 2


def main() -> None:
    args = parse_args()
    robot, server, urdf_vis = create_g1_robot_and_visual(
        floating_base=True,
        port=args.port,
        root_node_name="/floating_base/robot",
    )
    collision_model_report = validate_robot_collision_model(robot)
    base_node = server.scene.add_frame("/floating_base", show_axes=False)

    q = robot.neutral_configuration()
    q_lo, q_hi = robot.get_joint_limits()
    robot.update_configuration(q)
    frame_map = resolve_frames_for_g1_base_mode(robot.get_frame_names())
    upright_frame = "pelvis" if "pelvis" in robot.get_frame_names() else frame_map["imu_in_torso"]
    torso_upright_frame = frame_map["imu_in_torso"] if "imu_in_torso" in frame_map else upright_frame

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
    urdf_vis.update_cfg(map_q(q))

    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    right_task = solver.add_frame_task("right_palm_pose", frame_map["right_palm"], embodik.TaskType.FRAME_POSE)
    left_task = solver.add_frame_task("left_palm_pose", frame_map["left_palm"], embodik.TaskType.FRAME_POSE)
    posture = solver.add_posture_task("posture")
    pelvis_ori = solver.add_frame_task("pelvis_upright_ori", upright_frame, embodik.TaskType.FRAME_ORIENTATION)
    torso_ori = solver.add_frame_task("torso_upright_ori", torso_upright_frame, embodik.TaskType.FRAME_ORIENTATION)
    _apply_zero_based_task_hierarchy(
        primary_tasks=[right_task, left_task],
        secondary_tasks=[pelvis_ori, torso_ori],
        tertiary_tasks=[posture],
    )
    posture.weight = 0.02
    posture.set_target_configuration(q.copy())

    right_foot_anchor = np.asarray(robot.get_frame_pose(frame_map["right_ankle"]).homogeneous(), dtype=float)
    left_foot_anchor = np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).homogeneous(), dtype=float)
    pelvis_init = np.asarray(robot.get_frame_pose(upright_frame).homogeneous(), dtype=float)
    pelvis_target = np.eye(4, dtype=float)
    pelvis_target[:3, :] = pelvis_init[:3, :]
    torso_init = np.asarray(robot.get_frame_pose(torso_upright_frame).homogeneous(), dtype=float)
    torso_target = np.eye(4, dtype=float)
    torso_target[:3, :] = torso_init[:3, :]
    q_initial = np.asarray(q, dtype=float).copy()

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

    with server.gui.add_folder("IK Controls"):
        mode = server.gui.add_dropdown(
            "Contact handling mode",
            options=("Contact Root (rigid 6D)", "Contact Root (point 3D)", "Tight Epsilon (baseline)"),
            initial_value="Contact Root (rigid 6D)",
        )
        steps = server.gui.add_slider("IK Steps", 1, 20, 1, 4)
        hand_pos_gain = server.gui.add_slider("Hand Position Gain", 1.0, 60.0, 0.5, 12.0)
        hand_ori_gain = server.gui.add_slider("Hand Orientation Gain", 0.1, 60.0, 0.1, 8.0)
        hand_solve_mode = server.gui.add_dropdown(
            "Hand solve mode",
            options=("SCALE", "SCALE_ELASTIC", "MIN_ERROR"),
            initial_value="SCALE_ELASTIC",
        )
        allow_hand_fallback = server.gui.add_checkbox("Allow SCALE fallback", initial_value=False)
        enable_pelvis_upright = server.gui.add_checkbox("Enable pelvis upright", initial_value=False)
        pelvis_upright_gain = server.gui.add_slider("Pelvis upright gain", 0.0, 30.0, 0.1, 3.0)
        enable_torso_upright = server.gui.add_checkbox("Enable torso upright", initial_value=False)
        torso_upright_gain = server.gui.add_slider("Torso upright gain", 0.0, 30.0, 0.1, 2.0)
        posture_bias_weight = server.gui.add_slider("Nullspace posture bias weight", 0.0, 0.2, 0.001, 0.03)
        auto_boost_bias = server.gui.add_checkbox("Auto-boost bias on NO_PROGRESS", initial_value=True)
        max_bias_boost = server.gui.add_slider("Max bias boost x", 1.0, 8.0, 0.1, 3.0)
        strict_contact_guard = server.gui.add_checkbox("Strict rigid-contact drift guard", initial_value=True)
        guard_pos_mm = server.gui.add_slider("Guard max foot drift (mm)", 0.01, 10.0, 0.01, 0.5)
        guard_rot_deg = server.gui.add_slider("Guard max foot rot drift (deg)", 0.01, 5.0, 0.01, 0.2)
        recapture_bias = server.gui.add_button("Recapture posture bias target (current q)")
        reset_all = server.gui.add_button("Reset to initial configuration")
        foot_pos_eps = server.gui.add_slider("Foot position epsilon (m)", 1e-5, 0.01, 1e-5, 5e-4)
        foot_rot_eps_deg = server.gui.add_slider("Foot orientation epsilon (deg)", 0.01, 3.0, 0.01, 0.2)

    with server.gui.add_folder("CoM Constraint"):
        enable_com = server.gui.add_checkbox("Enable CoM Constraint", initial_value=True)
        margin = server.gui.add_slider("Margin (m)", 0.0, 0.08, 0.001, 0.01)
        vel_max = server.gui.add_slider("CoM vel max (m/s)", 0.05, 1.0, 0.01, 0.3)
        acc_max = server.gui.add_slider("CoM acc max (m/s^2)", 0.01, 2.0, 0.01, 0.3)
        foot_length = server.gui.add_slider("Foot contact length (m)", 0.12, 0.35, 0.001, 0.22)
        foot_width = server.gui.add_slider("Foot contact width (m)", 0.05, 0.20, 0.001, 0.10)
        toe_pad = server.gui.add_slider("Polygon toe pad (m)", 0.0, 0.2, 0.001, 0.06)
        side_pad = server.gui.add_slider("Polygon side pad (m)", 0.0, 0.2, 0.001, 0.07)

    with server.gui.add_folder("Collision Constraint"):
        enable_collision = server.gui.add_checkbox(
            "Enable self-collision constraint",
            initial_value=bool(collision_model_report.get("pair_count", 0) > 0),
        )
        collision_min_dist_mm = server.gui.add_slider("Collision min distance (mm)", 0.0, 120.0, 1.0, 35.0)
        collision_tuning = server.gui.add_dropdown(
            "Collision tuning mode",
            options=("speed", "balanced", "precise"),
            initial_value="speed",
        )
        exclude_consecutive = server.gui.add_checkbox("Exclude consecutive links", initial_value=True)
        show_collision_debug = server.gui.add_checkbox("Show collision debug", initial_value=True)
        collision_log_mode = server.gui.add_checkbox("Collision debug logging", initial_value=False)
        collision_debug_text = server.gui.add_text(
            "Collision debug",
            initial_value=(
                f"model_pairs={int(collision_model_report.get('pair_count', 0))}, "
                f"model_geoms={int(collision_model_report.get('geometry_count', 0))}"
            ),
        )
        collision_pairs_stats = server.gui.add_text("Collision pairs stats", initial_value="--")

    with server.gui.add_folder("Diagnostics"):
        status = server.gui.add_text("Status", initial_value="running")
        solve_ms = server.gui.add_text("Solver time (ms)", initial_value="--")
        contact_residual = server.gui.add_text("Contact residual ||Jc*dq||", initial_value="--")
        bias_diag = server.gui.add_text("Posture bias", initial_value="--")
        feet_bounds = server.gui.add_text("Feet drift", initial_value="--")

    poly_handle = server.scene.add_line_segments(
        "/com/support_polygon",
        points=np.zeros((1, 2, 3), dtype=float),
        colors=np.array([[[0.1, 0.5, 1.0], [0.1, 0.5, 1.0]]], dtype=float),
        line_width=3.0,
        visible=False,
    )
    col_a = server.scene.add_icosphere(
        "/collision_debug/point_a", radius=0.015, color=(1.0, 0.2, 0.2), visible=False
    )
    col_b = server.scene.add_icosphere(
        "/collision_debug/point_b", radius=0.015, color=(0.2, 0.8, 0.2), visible=False
    )
    col_line = None

    opts = embodik.PositionStepOptions()
    no_progress_streak = 0
    collision_exclusions = _generate_consecutive_collision_exclusions(robot)
    last_collision_cfg = None
    last_collision_pair = None
    last_collision_log_ts = 0.0

    @recapture_bias.on_click
    def _(_event) -> None:
        posture.set_target_configuration(np.asarray(q, dtype=float).copy())

    @reset_all.on_click
    def _(_event) -> None:
        nonlocal q, no_progress_streak
        q = q_initial.copy()
        no_progress_streak = 0
        robot.update_configuration(q)
        posture.set_target_configuration(q.copy())
        base_node.position = (float(q[0]), float(q[1]), float(q[2]))
        base_node.wxyz = (float(q[6]), float(q[3]), float(q[4]), float(q[5]))
        urdf_vis.update_cfg(map_q(q))

        for key, ctrl in (("right_palm", right_ctrl), ("left_palm", left_ctrl)):
            pose = robot.get_frame_pose(frame_map[key])
            ctrl.position = tuple(np.asarray(pose.translation, dtype=float))
            q_xyzw = r2q(np.asarray(pose.rotation, dtype=float), order="xyzs")
            ctrl.wxyz = (float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))

        status.value = "RESET_TO_INITIAL_CONFIGURATION"
        contact_residual.value = "--"
        bias_diag.value = "base=--, eff=--, no_progress_streak=0"

    while True:
        loop_t0 = time.perf_counter()
        q_prev = np.asarray(q, dtype=float).copy()
        rh_pose = _pose_from_ctrl(right_ctrl)
        lh_pose = _pose_from_ctrl(left_ctrl)
        is_contact_root_mode = mode.value != "Tight Epsilon (baseline)"

        # Contact-root mode typically behaves best with min-error hand solve mode.
        hand_mode = getattr(
            embodik.TaskSolveMode,
            hand_solve_mode.value,
            embodik.TaskSolveMode.SCALE,
        )
        right_task.solve_mode = hand_mode
        left_task.solve_mode = hand_mode
        right_task.allow_min_error_fallback = bool(allow_hand_fallback.value or is_contact_root_mode)
        left_task.allow_min_error_fallback = bool(allow_hand_fallback.value or is_contact_root_mode)
        pelvis_ori.active = bool(enable_pelvis_upright.value)
        torso_ori.active = bool(enable_torso_upright.value)
        base_bias = float(posture_bias_weight.value)
        boost = 1.0
        if auto_boost_bias.value and no_progress_streak > 0:
            boost = min(float(max_bias_boost.value), 1.0 + 0.4 * float(no_progress_streak))
        posture.weight = base_bias * boost

        solver.clear_contact_frames()
        solver.clear_tight_frame_pose_constraints()
        if mode.value == "Contact Root (rigid 6D)":
            solver.add_contact_frame(frame_map["right_ankle"], embodik.ContactType.RIGID_CONTACT)
            solver.add_contact_frame(frame_map["left_ankle"], embodik.ContactType.RIGID_CONTACT)
        elif mode.value == "Contact Root (point 3D)":
            solver.add_contact_frame(frame_map["right_ankle"], embodik.ContactType.POINT_CONTACT)
            solver.add_contact_frame(frame_map["left_ankle"], embodik.ContactType.POINT_CONTACT)
        else:
            pos_eps = float(foot_pos_eps.value)
            rot_eps = np.deg2rad(float(foot_rot_eps_deg.value))
            solver.add_tight_frame_pose_constraint(frame_map["right_ankle"], right_foot_anchor, pos_eps, rot_eps)
            solver.add_tight_frame_pose_constraint(frame_map["left_ankle"], left_foot_anchor, pos_eps, rot_eps)

        # Collision constraints with optional adjacent-link exclusions.
        exclusion_pairs = collision_exclusions if exclude_consecutive.value else []
        total_pairs = int(collision_model_report.get("pair_count", 0))
        collision_pairs_stats.value = (
            f"total_pairs={total_pairs}, excluded={len(exclusion_pairs)}, "
            f"effective={max(0, total_pairs - len(exclusion_pairs))}"
        )
        cfg_tuple = (
            bool(enable_collision.value),
            float(collision_min_dist_mm.value),
            str(collision_tuning.value),
            bool(exclude_consecutive.value),
        )
        if cfg_tuple != last_collision_cfg:
            if enable_collision.value:
                if hasattr(solver, "set_collision_tuning_mode") and hasattr(embodik, "CollisionTuningMode"):
                    mode_map = {
                        "speed": embodik.CollisionTuningMode.SPEED,
                        "balanced": embodik.CollisionTuningMode.BALANCED,
                        "precise": embodik.CollisionTuningMode.PRECISE,
                    }
                    solver.set_collision_tuning_mode(mode_map[str(collision_tuning.value)])
                try:
                    solver.configure_collision_constraint(
                        min_distance=float(collision_min_dist_mm.value) * 1e-3,
                        include_pairs=[],
                        exclude_pairs=list(exclusion_pairs),
                    )
                except RuntimeError:
                    solver.clear_collision_constraint()
            else:
                solver.clear_collision_constraint()
            last_collision_cfg = cfg_tuple

        poly_xy = compute_support_polygon_from_foot_poses(
            right_foot_anchor,
            left_foot_anchor,
            foot_length=float(foot_length.value),
            foot_width=float(foot_width.value),
            toe_pad=float(toe_pad.value),
            side_pad=float(side_pad.value),
        )
        if enable_com.value:
            solver.configure_com_constraint(
                poly_xy,
                margin=float(margin.value),
                com_vel_max=float(vel_max.value),
                com_acc_max=float(acc_max.value),
                use_acceleration_limits=True,
            )
            seg = polygon_segments_xy(poly_xy, z=0.002)
            poly_handle.remove()
            poly_handle = server.scene.add_line_segments(
                "/com/support_polygon",
                points=seg,
                colors=np.array([[[0.1, 0.5, 1.0], [0.1, 0.5, 1.0]]] * seg.shape[0], dtype=float),
                line_width=3.0,
                visible=True,
            )
        else:
            solver.clear_com_constraint()
            poly_handle.visible = False

        targets = [
            embodik.TaskTarget("right_palm_pose", rh_pose, float(hand_pos_gain.value), float(hand_ori_gain.value)),
            embodik.TaskTarget("left_palm_pose", lh_pose, float(hand_pos_gain.value), float(hand_ori_gain.value)),
            embodik.TaskTarget("pelvis_upright_ori", pelvis_target, 0.0, float(pelvis_upright_gain.value)),
            embodik.TaskTarget("torso_upright_ori", torso_target, 0.0, float(torso_upright_gain.value)),
        ]

        opts.max_steps = int(steps.value)
        opts.position_gain = float(hand_pos_gain.value)
        opts.orientation_gain = float(hand_ori_gain.value)
        configure_primary_solve_mode(
            opts,
            hand_mode,
            bool(allow_hand_fallback.value or is_contact_root_mode),
        )
        opts.max_linear_speed = 0.7
        opts.max_angular_speed = 1.5
        t0 = time.perf_counter()
        result = solver.solve_position_step(q, targets, opts)
        solve_ms.value = f"{(time.perf_counter() - t0) * 1e3:.2f}"

        accept_non_success = result.status in (
            embodik.SolverStatus.NO_PROGRESS,
            embodik.SolverStatus.INFEASIBLE,
        )
        if result.status == embodik.SolverStatus.SUCCESS or (
            is_contact_root_mode and accept_non_success and hasattr(result, "q_solution")
        ):
            q = _clip_q(robot, np.asarray(result.q_solution, dtype=float), q_lo, q_hi)
        else:
            q = q_prev
        if result.status == embodik.SolverStatus.NO_PROGRESS:
            no_progress_streak += 1
        elif result.status == embodik.SolverStatus.SUCCESS:
            no_progress_streak = max(0, no_progress_streak - 1)
        else:
            no_progress_streak = max(0, no_progress_streak - 1)
        robot.update_configuration(q)
        base_node.position = (float(q[0]), float(q[1]), float(q[2]))
        base_node.wxyz = (float(q[6]), float(q[3]), float(q[4]), float(q[5]))
        urdf_vis.update_cfg(map_q(q))

        r_now = np.asarray(robot.get_frame_pose(frame_map["right_ankle"]).homogeneous(), dtype=float)
        l_now = np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).homogeneous(), dtype=float)
        r_err = _frame_delta6(right_foot_anchor, r_now)
        l_err = _frame_delta6(left_foot_anchor, l_now)
        if strict_contact_guard.value and mode.value == "Contact Root (rigid 6D)":
            max_pos_err = max(float(np.max(np.abs(r_err[:3]))), float(np.max(np.abs(l_err[:3]))))
            max_rot_err = max(float(np.max(np.abs(r_err[3:]))), float(np.max(np.abs(l_err[3:]))))
            pos_thr = float(guard_pos_mm.value) * 1e-3
            rot_thr = np.deg2rad(float(guard_rot_deg.value))
            if max_pos_err > pos_thr or max_rot_err > rot_thr:
                q = q_prev.copy()
                robot.update_configuration(q)
                r_now = np.asarray(robot.get_frame_pose(frame_map["right_ankle"]).homogeneous(), dtype=float)
                l_now = np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).homogeneous(), dtype=float)
                r_err = _frame_delta6(right_foot_anchor, r_now)
                l_err = _frame_delta6(left_foot_anchor, l_now)
                no_progress_streak = max(no_progress_streak, 1)
        dq = np.asarray(result.joint_velocities, dtype=float)
        nv = int(robot.nv if hasattr(robot, "nv") else robot.nv())
        if is_contact_root_mode and dq.size == nv:
            J_rows = []
            J_r = np.asarray(robot.get_frame_jacobian(frame_map["right_ankle"]), dtype=float)
            J_l = np.asarray(robot.get_frame_jacobian(frame_map["left_ankle"]), dtype=float)
            if mode.value == "Contact Root (point 3D)":
                J_rows.extend([J_r[:3, :], J_l[:3, :]])
            else:
                J_rows.extend([J_r, J_l])
            J_c = np.vstack(J_rows) if J_rows else np.zeros((0, nv), dtype=float)
            residual = float(np.linalg.norm(J_c @ dq)) if J_c.size else 0.0
            contact_residual.value = f"{residual:.3e}"
        else:
            contact_residual.value = "--"
        feet_bounds.value = (
            f"R pos={np.max(np.abs(r_err[:3])):.4e}, rot={np.rad2deg(np.max(np.abs(r_err[3:]))):.3f} deg | "
            f"L pos={np.max(np.abs(l_err[:3])):.4e}, rot={np.rad2deg(np.max(np.abs(l_err[3:]))):.3f} deg"
        )
        feet_center = compute_feet_center(r_now[:3, 3], l_now[:3, 3])
        status.value = (
            f"{result.status.name} | mode={mode.value} | "
            f"feet_center=({feet_center[0]:.3f},{feet_center[1]:.3f},{feet_center[2]:.3f})"
        )
        bias_diag.value = f"base={base_bias:.3f}, eff={posture.weight:.3f}, no_progress_streak={no_progress_streak}"

        # Collision debug visualization/logging (closest active pair).
        if (
            enable_collision.value
            and show_collision_debug.value
            and hasattr(solver, "get_last_collision_debug")
        ):
            dbg = solver.get_last_collision_debug()
            if dbg is not None:
                p_a = np.asarray(dbg.point_a_world, dtype=float)
                p_b = np.asarray(dbg.point_b_world, dtype=float)
                col_a.position = tuple(p_a)
                col_b.position = tuple(p_b)
                col_a.visible = True
                col_b.visible = True
                if col_line is not None:
                    col_line.remove()
                seg = np.zeros((1, 2, 3), dtype=float)
                seg[0, 0, :] = p_a
                seg[0, 1, :] = p_b
                col_line = server.scene.add_line_segments(
                    "/collision_debug/segment",
                    points=seg,
                    colors=np.array([[[1.0, 0.2, 0.2], [0.2, 0.8, 0.2]]], dtype=float),
                    line_width=3.0,
                    visible=True,
                )
                collision_debug_text.value = f"{dbg.object_a} ↔ {dbg.object_b} | d={float(dbg.distance):.4f} m"

                now = time.time()
                pair_key = (str(dbg.object_a), str(dbg.object_b), round(float(dbg.distance), 4))
                if collision_log_mode.value and (pair_key != last_collision_pair or now - last_collision_log_ts > 1.0):
                    print(
                        "[07b][collision]",
                        f"{dbg.object_a} <-> {dbg.object_b}",
                        f"d={float(dbg.distance):.4f} m",
                    )
                    last_collision_pair = pair_key
                    last_collision_log_ts = now
            else:
                collision_debug_text.value = "--"
                col_a.visible = False
                col_b.visible = False
                if col_line is not None:
                    col_line.visible = False
        else:
            collision_debug_text.value = "--"
            col_a.visible = False
            col_b.visible = False
            if col_line is not None:
                col_line.visible = False
        _ = loop_t0
        time.sleep(0.002)


if __name__ == "__main__":
    main()
