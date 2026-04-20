#!/usr/bin/env python3
"""Unitree G1 full-body IK counterpart with floating base + CoM constraints.

Highlights:
- Floating-base full-body control
- Position targets for hands + torso
- Selectable feet goal mode: position-only or 6D pose constraints
- CoM support-polygon constraint controls inspired by EmbodiK CoM example
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
    build_fullbody_target_payloads,
    compute_support_polygon_from_feet,
    create_g1_robot_and_visual,
    get_retargeting_presets,
    make_visual_config_mapper,
    polygon_segments_xy,
    retarget_position_from_anchor,
    resolve_frames_for_g1_base_mode,
)
from examples.incubating.sungjoon_port_phase1.robust_ik_runtime import robust_solve_position_step


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8085)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    robot, server, urdf_vis = create_g1_robot_and_visual(
        floating_base=True,  # full-body mode defaults to floating base
        port=args.port,
    )
    q = robot.neutral_configuration()
    q_lo, q_hi = robot.get_joint_limits()
    robot.update_configuration(q)

    frame_map = resolve_frames_for_g1_base_mode(robot.get_frame_names())
    print("Frame mapping:", frame_map)

    map_q = make_visual_config_mapper(robot, urdf_vis)
    urdf_vis.update_cfg(map_q(q))

    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    ordered_keys = ["right_palm", "left_palm", "imu_in_torso", "right_ankle", "left_ankle"]
    for key in ordered_keys:
        t = solver.add_frame_task(
            f"g1_base_{key}_pos",
            frame_map[key],
            embodik.TaskType.FRAME_POSITION,
        )
        t.priority = 0
        t.weight = 1.0

    right_foot_ori = solver.add_frame_task(
        "g1_base_right_ankle_ori",
        frame_map["right_ankle"],
        embodik.TaskType.FRAME_ORIENTATION,
    )
    right_foot_ori.priority = 0
    right_foot_ori.weight = 0.0
    left_foot_ori = solver.add_frame_task(
        "g1_base_left_ankle_ori",
        frame_map["left_ankle"],
        embodik.TaskType.FRAME_ORIENTATION,
    )
    left_foot_ori.priority = 0
    left_foot_ori.weight = 0.0

    targets = {}
    for key in ordered_keys:
        pose = robot.get_frame_pose(frame_map[key])
        R = np.asarray(pose.rotation, dtype=float)
        q_xyzw = r2q(R, order="xyzs")
        wxyz = (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])
        targets[key] = server.scene.add_transform_controls(
            f"/target/{key}",
            scale=0.12 if "ankle" in key else 0.16,
            position=tuple(np.asarray(pose.translation, dtype=float)),
            wxyz=wxyz,
        )

    with server.gui.add_folder("IK Controls"):
        foot_mode = server.gui.add_dropdown(
            "Feet Goal Mode",
            options=("Sungjoon Position-only", "EmbodiK 6D Feet"),
            initial_value="EmbodiK 6D Feet",
        )
        steps = server.gui.add_slider("IK Steps", 1, 20, 1, 5)
        gain = server.gui.add_slider("Position Gain", 1.0, 40.0, 0.5, 15.0)
        foot_ori_gain = server.gui.add_slider("Foot Orientation Gain", 0.1, 40.0, 0.1, 12.0)

    presets = get_retargeting_presets()
    with server.gui.add_folder("Retargeting Examples"):
        retarget_preset = server.gui.add_dropdown(
            "Preset",
            options=tuple(presets.keys()),
            initial_value="neutral",
        )
        retarget_scale = server.gui.add_slider("Retarget Scale", 0.5, 1.5, 0.01, 1.0)
        apply_retarget = server.gui.add_button("Apply retarget preset to targets")

    with server.gui.add_folder("CoM Constraint"):
        com_enable = server.gui.add_checkbox("Enable CoM polygon constraint", initial_value=True)
        com_margin = server.gui.add_slider("Margin (m)", 0.0, 0.08, 0.001, 0.01)
        com_vel = server.gui.add_slider("CoM vel max (m/s)", 0.05, 1.0, 0.01, 0.3)
        com_acc = server.gui.add_slider("CoM acc max (m/s^2)", 0.01, 2.0, 0.01, 0.3)
        toe_pad = server.gui.add_slider("Polygon toe pad (m)", 0.0, 0.2, 0.001, 0.06)
        side_pad = server.gui.add_slider("Polygon side pad (m)", 0.0, 0.2, 0.001, 0.07)

    with server.gui.add_folder("Diagnostics"):
        status = server.gui.add_text("Status", initial_value="running")
        com_status = server.gui.add_text("CoM status", initial_value="--")
        solve_ms = server.gui.add_text("Solve time (ms)", initial_value="--")

    @apply_retarget.on_click
    def _(_evt) -> None:
        preset = presets[retarget_preset.value]
        anchor_pose = robot.get_frame_pose(frame_map["imu_in_torso"])
        p_anchor = np.asarray(anchor_pose.translation, dtype=float)
        R_anchor = np.asarray(anchor_pose.rotation, dtype=float)
        for key, ctrl in targets.items():
            local = preset.get(key)
            if local is None:
                continue
            p_world = retarget_position_from_anchor(
                p_anchor,
                R_anchor,
                np.asarray(local, dtype=float),
                scale=float(retarget_scale.value),
            )
            ctrl.position = tuple(p_world)
        status.value = f"Applied retarget preset: {retarget_preset.value}"

    com_marker = server.scene.add_icosphere(
        "/com/current",
        radius=0.02,
        color=(0.2, 0.85, 0.2),
        visible=True,
    )
    polygon_handle = server.scene.add_line_segments(
        "/com/support_polygon",
        points=np.zeros((1, 2, 3), dtype=float),
        colors=np.array([[[0.1, 0.5, 1.0], [0.1, 0.5, 1.0]]], dtype=float),
        line_width=3.0,
        visible=False,
    )

    opts = embodik.PositionStepOptions()
    while True:
        step_targets = []
        foot_pose_targets = {}
        for key in ordered_keys:
            ctrl = targets[key]
            pose = np.eye(4, dtype=float)
            pose[:3, 3] = np.asarray(ctrl.position, dtype=float)
            wxyz = np.asarray(ctrl.wxyz, dtype=float)
            q_xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=float)
            pose[:3, :3] = q2r(q_xyzw, order="xyzs")
            foot_pose_targets[key] = pose
        payloads = build_fullbody_target_payloads(
            foot_mode.value,
            foot_pose_targets,
            position_gain=float(gain.value),
            foot_orientation_gain=float(foot_ori_gain.value),
        )
        right_foot_ori.weight = 1.0 if foot_mode.value == "EmbodiK 6D Feet" else 0.0
        left_foot_ori.weight = 1.0 if foot_mode.value == "EmbodiK 6D Feet" else 0.0
        for payload in payloads:
            step_targets.append(
                embodik.TaskTarget(
                    task_name=payload["task_name"],
                    target_pose=payload["target_pose"],
                    position_gain=float(payload["position_gain"]),
                    orientation_gain=float(payload["orientation_gain"]),
                )
            )

        # CoM support polygon from current foot targets.
        r_xy = foot_pose_targets["right_ankle"][:2, 3]
        l_xy = foot_pose_targets["left_ankle"][:2, 3]
        poly_xy = compute_support_polygon_from_feet(
            r_xy,
            l_xy,
            toe_pad=float(toe_pad.value),
            side_pad=float(side_pad.value),
        )
        if com_enable.value:
            solver.configure_com_constraint(
                poly_xy,
                margin=float(com_margin.value),
                com_vel_max=float(com_vel.value),
                com_acc_max=float(com_acc.value),
                use_acceleration_limits=True,
            )
            seg = polygon_segments_xy(poly_xy, z=0.001)
            poly_color = np.array([[[0.1, 0.5, 1.0], [0.1, 0.5, 1.0]]] * seg.shape[0], dtype=float)
            polygon_handle.remove()
            polygon_handle = server.scene.add_line_segments(
                "/com/support_polygon",
                points=seg,
                colors=poly_color,
                line_width=3.0,
                visible=True,
            )
        else:
            solver.clear_com_constraint()
            polygon_handle.visible = False

        opts.max_steps = int(steps.value)
        opts.position_gain = float(gain.value)
        opts.orientation_gain = float(foot_ori_gain.value)
        step = robust_solve_position_step(
            robot=robot,
            solver=solver,
            q_current=q,
            targets=step_targets,
            options=opts,
            q_lo=q_lo,
            q_hi=q_hi,
            fallback_status_names=("INVALID_INPUT",),
        )
        q = step.q_next
        result = step.solver_result

        robot.update_configuration(q)
        urdf_vis.update_cfg(map_q(q))

        errs = []
        for key in ordered_keys:
            p_target = np.asarray(targets[key].position, dtype=float)
            p_cur = np.asarray(robot.get_frame_pose(frame_map[key]).translation)
            errs.append(float(np.linalg.norm(p_target - p_cur)))
        com_pos = np.asarray(robot.get_com_position(), dtype=float)
        com_marker.position = tuple(com_pos)
        x_min, x_max = np.min(poly_xy[:, 0]), np.max(poly_xy[:, 0])
        y_min, y_max = np.min(poly_xy[:, 1]), np.max(poly_xy[:, 1])
        inside = (x_min <= com_pos[0] <= x_max) and (y_min <= com_pos[1] <= y_max)
        com_status.value = (
            f"CoM=({com_pos[0]:.3f},{com_pos[1]:.3f}) "
            f"{'inside' if inside else 'outside'} polygon"
        )
        status.value = (
            f"{result.status.name} | max_err={max(errs):.4f}"
            + (f" | {result.status_message}" if getattr(result, "status_message", "") else "")
        )
        solve_ms.value = f"{step.elapsed_ms:.2f}"
        time.sleep(0.002)


if __name__ == "__main__":
    main()
