#!/usr/bin/env python3
"""G1 dual-hand grounded floating-base IK with CoM constraints.

Reimagines the G1 flow by combining:
- EmbodiK interactive target controls (02 style)
- CoM support polygon visualization + constraint controls (08 style)

Key behavior:
- Floating base is shifted by negative feet-center so feet rest around ground.
- Both hands are controlled by independent 6D gizmos.
- Both feet are constrained with 6D pose tasks (anchored at startup).
"""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import embodik
from embodik.utils import r2q, q2r

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
)
from examples.incubating.sungjoon_port_phase1.robust_ik_runtime import (
    clear_all_target_velocities_if_available,
    clip_configuration,
    configure_primary_solve_mode,
    robust_solve_position_step,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8087)
    return p.parse_args()


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


def _is_finite_pose(pose: np.ndarray) -> bool:
    pose = np.asarray(pose, dtype=float)
    return bool(np.all(np.isfinite(pose)))


def _q_root_pos_wxyz(q: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Extract floating-base root pose from q=[tx,ty,tz,qx,qy,qz,qw,...]."""
    q = np.asarray(q, dtype=float)
    pos = (float(q[0]), float(q[1]), float(q[2]))
    wxyz = (float(q[6]), float(q[3]), float(q[4]), float(q[5]))
    return pos, wxyz


def _frame_delta6(anchor_pose: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
    """6D delta [dx,dy,dz,rx,ry,rz] from anchor to current frame pose."""
    delta = np.zeros(6, dtype=float)
    delta[:3] = np.asarray(current_pose[:3, 3] - anchor_pose[:3, 3], dtype=float)
    rot_ref = np.asarray(anchor_pose[:3, :3], dtype=float)
    rot_cur = np.asarray(current_pose[:3, :3], dtype=float)
    delta[3:] = np.asarray(embodik.log3(rot_ref.T @ rot_cur), dtype=float)
    return delta


def main() -> None:
    args = parse_args()
    robot, server, urdf_vis = create_g1_robot_and_visual(
        floating_base=True,
        port=args.port,
        root_node_name="/floating_base/robot",
    )
    base_node = server.scene.add_frame(
        "/floating_base",
        position=(0.0, 0.0, 0.0),
        wxyz=(1.0, 0.0, 0.0, 0.0),
        show_axes=False,
    )
    q = robot.neutral_configuration()
    q_lo, q_hi = robot.get_joint_limits()
    robot.update_configuration(q)

    frame_map = resolve_frames_for_g1_base_mode(robot.get_frame_names())
    print("Frame mapping:", frame_map)

    base_frame = "base_link" if "base_link" in robot.get_frame_names() else frame_map["imu_in_torso"]
    upright_frame = "pelvis" if "pelvis" in robot.get_frame_names() else base_frame

    # Ground the floating base by translating base frame by negative feet center.
    # Step 1: evaluate feet FK at current q.
    r0 = np.asarray(robot.get_frame_pose(frame_map["right_ankle"]).translation, dtype=float)
    l0 = np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).translation, dtype=float)
    feet_center_pre = compute_feet_center(r0, l0)
    # Step 2: update base translation by -feet_center and recompute full FK.
    q = ground_floating_base_from_feet_center(q, r0, l0)
    q = clip_configuration(robot, q, q_lo, q_hi)
    robot.update_configuration(q)
    # Step 3: refresh all FK quantities after grounding.
    r1 = np.asarray(robot.get_frame_pose(frame_map["right_ankle"]).translation, dtype=float)
    l1 = np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).translation, dtype=float)
    # Lift slightly so ankle frames are not visually below the ground grid.
    desired_ankle_height = 0.03
    min_ankle_z = float(min(r1[2], l1[2]))
    q[2] += desired_ankle_height - min_ankle_z
    q = clip_configuration(robot, q, q_lo, q_hi)
    robot.update_configuration(q)
    r1 = np.asarray(robot.get_frame_pose(frame_map["right_ankle"]).translation, dtype=float)
    l1 = np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).translation, dtype=float)
    feet_center_post = compute_feet_center(r1, l1)

    base_pos, base_wxyz = _q_root_pos_wxyz(q)
    base_node.position = base_pos
    base_node.wxyz = base_wxyz

    map_q = make_visual_config_mapper(robot, urdf_vis)
    urdf_vis.update_cfg(map_q(q))

    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    # 6D hands (single FRAME_POSE task per hand for robust pose tracking)
    hand_tasks = []
    for hand_key in ("right_palm", "left_palm"):
        t_pose = solver.add_frame_task(f"{hand_key}_pose", frame_map[hand_key], embodik.TaskType.FRAME_POSE)
        # Hands are secondary to feet so stance stays grounded.
        t_pose.priority = 1
        t_pose.weight = 1.0
        hand_tasks.append(t_pose)

    posture = solver.add_posture_task("posture")
    torso_ori_task = solver.add_frame_task(
        "pelvis_upright_ori",
        upright_frame,
        embodik.TaskType.FRAME_ORIENTATION,
    )
    torso_ori_task.priority = 2
    torso_ori_task.weight = 1.0

    posture.priority = 3
    posture.weight = 0.02
    posture.set_target_configuration(q.copy())

    def _mk_gizmo(key: str, scale: float):
        pose = robot.get_frame_pose(frame_map[key])
        q_xyzw = r2q(np.asarray(pose.rotation, dtype=float), order="xyzs")
        return server.scene.add_transform_controls(
            f"/target/{key}",
            scale=scale,
            position=tuple(np.asarray(pose.translation, dtype=float)),
            wxyz=(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]),
        )

    right_hand_ctrl = _mk_gizmo("right_palm", 0.16)
    left_hand_ctrl = _mk_gizmo("left_palm", 0.16)
    rh_pose_prev = _pose_from_ctrl(right_hand_ctrl)
    lh_pose_prev = _pose_from_ctrl(left_hand_ctrl)

    # Feet remain anchored to startup grounded poses.
    right_foot_anchor = np.asarray(robot.get_frame_pose(frame_map["right_ankle"]).homogeneous(), dtype=float)
    left_foot_anchor = np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).homogeneous(), dtype=float)
    pelvis_pose_init = np.asarray(robot.get_frame_pose(upright_frame).homogeneous(), dtype=float)
    pelvis_upright_target = np.eye(4, dtype=float)
    pelvis_upright_target[:3, :3] = pelvis_pose_init[:3, :3]
    pelvis_upright_target[:3, 3] = pelvis_pose_init[:3, 3]
    feet_center0 = compute_feet_center(right_foot_anchor[:3, 3], left_foot_anchor[:3, 3])
    print("Grounding feet center pre:", feet_center_pre)
    print("Grounding feet center post:", feet_center_post)
    print("Anchored feet center:", feet_center0)

    with server.gui.add_folder("IK Controls"):
        steps = server.gui.add_slider("IK Steps", 1, 20, 1, 5)
        hand_pos_gain = server.gui.add_slider("Hand Position Gain", 1.0, 60.0, 0.5, 12.0)
        hand_ori_gain = server.gui.add_slider("Hand Orientation Gain", 0.1, 60.0, 0.1, 8.0)
        enable_torso_upright = server.gui.add_checkbox("Enable pelvis upright", initial_value=True)
        torso_ori_gain = server.gui.add_slider("Pelvis upright gain", 0.0, 30.0, 0.1, 3.0)
        hand_solve_mode = server.gui.add_dropdown(
            "Hand solve mode",
            options=("SCALE", "SCALE_ELASTIC", "MIN_ERROR"),
            initial_value="SCALE_ELASTIC",
        )
        allow_hand_fallback = server.gui.add_checkbox("Allow SCALE fallback", initial_value=False)
        max_lin_speed = server.gui.add_slider("Max linear speed (m/s)", 0.05, 2.0, 0.01, 0.7)
        max_ang_speed = server.gui.add_slider("Max angular speed (rad/s)", 0.05, 4.0, 0.01, 1.5)
        foot_pos_eps = server.gui.add_slider("Foot position epsilon (m)", 1e-5, 0.01, 1e-5, 5e-4)
        foot_rot_eps_deg = server.gui.add_slider(
            "Foot orientation epsilon (deg)", 0.01, 3.0, 0.01, 0.2
        )
        auto_relax_stall = server.gui.add_checkbox("Auto relax when stalled", initial_value=True)
        stall_trigger_steps = server.gui.add_slider("Stall trigger steps", 1, 80, 1, 8)
        max_relax_mult = server.gui.add_slider("Max epsilon relax x", 1.0, 50.0, 0.5, 8.0)
        posture_w = server.gui.add_slider("Posture Weight", 0.0, 0.2, 0.001, 0.02)

    with server.gui.add_folder("CoM Constraint"):
        enable_com = server.gui.add_checkbox("Enable CoM Constraint", initial_value=True)
        margin = server.gui.add_slider("Margin (m)", 0.0, 0.08, 0.001, 0.01)
        vel_max = server.gui.add_slider("CoM vel max (m/s)", 0.05, 1.0, 0.01, 0.3)
        acc_max = server.gui.add_slider("CoM acc max (m/s^2)", 0.01, 2.0, 0.01, 0.3)
        foot_length = server.gui.add_slider("Foot contact length (m)", 0.12, 0.35, 0.001, 0.22)
        foot_width = server.gui.add_slider("Foot contact width (m)", 0.05, 0.20, 0.001, 0.10)
        toe_pad = server.gui.add_slider("Polygon toe pad (m)", 0.0, 0.2, 0.001, 0.06)
        side_pad = server.gui.add_slider("Polygon side pad (m)", 0.0, 0.2, 0.001, 0.07)

    with server.gui.add_folder("Diagnostics"):
        status = server.gui.add_text("Status", initial_value="running")
        solve_ms = server.gui.add_text("Solver time (ms)", initial_value="--")
        loop_ms = server.gui.add_text("Loop time (ms)", initial_value="--")
        solver_calls = server.gui.add_text("Solver calls/frame", initial_value="1")
        stall_diag = server.gui.add_text("Stall/recovery", initial_value="--")
        com_status = server.gui.add_text("CoM status", initial_value="--")
        feet_bounds_status = server.gui.add_text("Feet epsilon bounds", initial_value="--")

    com_sphere = server.scene.add_icosphere("/com/current", radius=0.02, color=(0.2, 0.85, 0.2), visible=True)
    poly_handle = server.scene.add_line_segments(
        "/com/support_polygon",
        points=np.zeros((1, 2, 3), dtype=float),
        colors=np.array([[[0.1, 0.5, 1.0], [0.1, 0.5, 1.0]]], dtype=float),
        line_width=3.0,
        visible=False,
    )

    opts = embodik.PositionStepOptions()
    pelvis_upright_cooldown = 0
    stall_count = 0
    while True:
        loop_t0 = time.perf_counter()
        q_prev = np.asarray(q, dtype=float).copy()
        posture.weight = float(posture_w.value)
        in_stall_recovery = auto_relax_stall.value and (stall_count >= int(stall_trigger_steps.value))
        if pelvis_upright_cooldown > 0 or in_stall_recovery:
            torso_ori_task.active = False
            if pelvis_upright_cooldown > 0:
                pelvis_upright_cooldown -= 1
        else:
            torso_ori_task.active = bool(enable_torso_upright.value)
        # Avoid stale target velocities from prior exceptional frames.
        clear_all_target_velocities_if_available(solver)
        hand_mode = getattr(
            embodik.TaskSolveMode,
            hand_solve_mode.value,
            embodik.TaskSolveMode.SCALE,
        )
        for t in hand_tasks:
            t.solve_mode = hand_mode
            t.allow_min_error_fallback = bool(allow_hand_fallback.value)

        rh_pose = _pose_from_ctrl(right_hand_ctrl)
        lh_pose = _pose_from_ctrl(left_hand_ctrl)
        if not _is_finite_pose(rh_pose):
            rh_pose = rh_pose_prev.copy()
        if not _is_finite_pose(lh_pose):
            lh_pose = lh_pose_prev.copy()
        rh_pose_prev = rh_pose.copy()
        lh_pose_prev = lh_pose.copy()

        targets = [
            embodik.TaskTarget("right_palm_pose", rh_pose, float(hand_pos_gain.value), float(hand_ori_gain.value)),
            embodik.TaskTarget("left_palm_pose", lh_pose, float(hand_pos_gain.value), float(hand_ori_gain.value)),
            embodik.TaskTarget("pelvis_upright_ori", pelvis_upright_target, 0.0, float(torso_ori_gain.value)),
        ]

        poly_xy = compute_support_polygon_from_foot_poses(
            right_foot_anchor,
            left_foot_anchor,
            foot_length=float(foot_length.value),
            foot_width=float(foot_width.value),
            toe_pad=float(toe_pad.value),
            side_pad=float(side_pad.value),
        )
        if (poly_xy.shape[0] < 3) or (not np.all(np.isfinite(poly_xy))):
            solver.clear_com_constraint()
            poly_handle.visible = False
            status.value = "INVALID_POLYGON_INPUT"
            q = q_prev
            robot.update_configuration(q)
            continue
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

        # Solver-native tight feet constraints (epsilon-box equality style).
        solver.clear_tight_frame_pose_constraints()
        base_pos_eps = float(foot_pos_eps.value)
        base_rot_eps = np.deg2rad(float(foot_rot_eps_deg.value))
        relax_mult = 1.0
        if auto_relax_stall.value and stall_count >= int(stall_trigger_steps.value):
            relax_mult = min(
                float(max_relax_mult.value),
                1.0 + 0.5 * float(stall_count - int(stall_trigger_steps.value) + 1),
            )
        pos_eps = base_pos_eps * relax_mult
        rot_eps = base_rot_eps * relax_mult
        solver.add_tight_frame_pose_constraint(
            frame_map["right_ankle"], right_foot_anchor, pos_eps, rot_eps
        )
        solver.add_tight_frame_pose_constraint(
            frame_map["left_ankle"], left_foot_anchor, pos_eps, rot_eps
        )

        opts.max_steps = 1 if in_stall_recovery else int(steps.value)
        opts.position_gain = float(hand_pos_gain.value)
        opts.orientation_gain = float(hand_ori_gain.value)
        opts.max_linear_speed = float(max_lin_speed.value)
        opts.max_angular_speed = float(max_ang_speed.value)
        configure_primary_solve_mode(opts, hand_mode, bool(allow_hand_fallback.value))
        step = robust_solve_position_step(
            robot=robot,
            solver=solver,
            q_current=q,
            targets=targets,
            options=opts,
            q_lo=q_lo,
            q_hi=q_hi,
            fallback_status_names=("INVALID_INPUT", "NUMERICAL_ERROR"),
        )
        result = step.solver_result
        solver_elapsed_ms = step.elapsed_ms
        calls_this_frame = step.solver_calls
        q = step.q_next
        if not np.all(np.isfinite(q)):
            q = q_prev.copy()

        msg = getattr(result, "status_message", "") or ""
        if "non-finite" in msg.lower():
            pelvis_upright_cooldown = max(pelvis_upright_cooldown, 120)
        if result.status in (embodik.SolverStatus.NO_PROGRESS, embodik.SolverStatus.INFEASIBLE):
            stall_count += 1
        elif result.status == embodik.SolverStatus.SUCCESS:
            stall_count = max(0, stall_count - 2)
        else:
            stall_count = max(0, stall_count - 1)

        robot.update_configuration(q)
        base_pos, base_wxyz = _q_root_pos_wxyz(q)
        base_node.position = base_pos
        base_node.wxyz = base_wxyz
        urdf_vis.update_cfg(map_q(q))

        com = np.asarray(robot.get_com_position(), dtype=float)
        com_sphere.position = tuple(com)
        x_min, x_max = np.min(poly_xy[:, 0]), np.max(poly_xy[:, 0])
        y_min, y_max = np.min(poly_xy[:, 1]), np.max(poly_xy[:, 1])
        inside = (x_min <= com[0] <= x_max) and (y_min <= com[1] <= y_max)
        com_status.value = f"CoM=({com[0]:.3f},{com[1]:.3f},{com[2]:.3f}) {'inside' if inside else 'outside'}"
        right_ankle_now = np.asarray(robot.get_frame_pose(frame_map["right_ankle"]).translation, dtype=float)
        left_ankle_now = np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).translation, dtype=float)
        feet_center_now = compute_feet_center(right_ankle_now, left_ankle_now)
        right_pose_now = np.asarray(robot.get_frame_pose(frame_map["right_ankle"]).homogeneous(), dtype=float)
        left_pose_now = np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).homogeneous(), dtype=float)
        r_delta6 = _frame_delta6(right_foot_anchor, right_pose_now)
        l_delta6 = _frame_delta6(left_foot_anchor, left_pose_now)
        feet_ok = bool(
            np.all(np.abs(r_delta6[:3]) <= pos_eps)
            and np.all(np.abs(l_delta6[:3]) <= pos_eps)
            and np.all(np.abs(r_delta6[3:]) <= rot_eps)
            and np.all(np.abs(l_delta6[3:]) <= rot_eps)
        )
        result_status_name = result.status.name
        feet_bounds_status.value = (
            f"R pos|max|={np.max(np.abs(r_delta6[:3])):.4f} m, "
            f"L pos|max|={np.max(np.abs(l_delta6[:3])):.4f} m | "
            f"R rot|max|={np.rad2deg(np.max(np.abs(r_delta6[3:]))):.2f} deg, "
            f"L rot|max|={np.rad2deg(np.max(np.abs(l_delta6[3:]))):.2f} deg | "
            f"{'OK' if feet_ok else 'OUT_OF_BOUNDS'}"
        )
        status.value = (
            f"{result_status_name} | feet_center_now=({feet_center_now[0]:.3f},{feet_center_now[1]:.3f},{feet_center_now[2]:.3f})"
            + (f" | {result.status_message}" if getattr(result, "status_message", "") else "")
            + (
                f" | pelvis_upright_hold={pelvis_upright_cooldown}"
                if pelvis_upright_cooldown > 0
                else ""
            )
        )
        solve_ms.value = f"{solver_elapsed_ms:.2f}"
        loop_ms.value = f"{(time.perf_counter() - loop_t0) * 1e3:.2f}"
        solver_calls.value = str(calls_this_frame)
        stall_diag.value = (
            f"stall_count={stall_count}, relax_x={relax_mult:.2f}, "
            f"eff_steps={opts.max_steps}, pelvis_active={torso_ori_task.active}"
        )
        time.sleep(0.002)


if __name__ == "__main__":
    main()
