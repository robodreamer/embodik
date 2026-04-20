#!/usr/bin/env python3
"""Unitree G1 counterpart of G1 02_ik_site_g1_inspire with Viser.

This version supports two interactive modes:
- G1 3-point: position-goal style with orientation surrogate points
- EmbodiK 6D: native position + orientation goal on the same end-effector frame
"""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rscipy
import embodik
from embodik.utils import r2q, q2r

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.incubating.g1_port_phase1.g1_viser_utils import (
    build_site_mode_target_payloads,
    create_g1_robot_and_visual,
    get_retargeting_presets,
    make_visual_config_mapper,
    retarget_position_from_anchor,
    resolve_frames_for_g1_site_mode,
)
from examples.incubating.g1_port_phase1.robust_ik_runtime import robust_solve_position_step


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8084)
    return p.parse_args()


def _rpy_to_rot(rpy: np.ndarray) -> np.ndarray:
    return Rscipy.from_euler("xyz", rpy).as_matrix()


def _rot_to_rpy(rot: np.ndarray) -> np.ndarray:
    # Use embodiK quaternion utilities to project noisy matrices onto SO(3)
    # before converting to Euler angles.
    q_xyzw = r2q(rot, order="xyzs")
    rot_ortho = q2r(q_xyzw, order="xyzs")
    return Rscipy.from_matrix(rot_ortho).as_euler("xyz")


def _safe_uv(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return v / n


def _estimate_hand_rot(robot: embodik.RobotModel, frames: list[str]) -> np.ndarray:
    p0 = np.asarray(robot.get_frame_pose(frames[0]).translation)
    p1 = np.asarray(robot.get_frame_pose(frames[1]).translation)
    p2 = np.asarray(robot.get_frame_pose(frames[2]).translation)
    p3 = np.asarray(robot.get_frame_pose(frames[3]).translation)
    # Build a guaranteed right-handed basis from two directions, then
    # orthonormalize with embodiK quaternion projection.
    ux = _safe_uv(p3 - p0)
    uy_raw = _safe_uv(p2 - p0)
    uz = np.cross(ux, uy_raw)
    if np.linalg.norm(uz) < 1e-9:
        uz = np.array([0.0, 0.0, 1.0], dtype=float)
    uz = _safe_uv(uz)
    uy = _safe_uv(np.cross(uz, ux))
    rot = np.column_stack((ux, uy, uz))
    q_xyzw = r2q(rot, order="xyzs")
    return q2r(q_xyzw, order="xyzs")


def main() -> None:
    args = parse_args()
    robot, server, urdf_vis = create_g1_robot_and_visual(
        floating_base=False, port=args.port
    )
    q = robot.neutral_configuration()
    q_lo, q_hi = robot.get_joint_limits()
    robot.update_configuration(q)

    frame_names = robot.get_frame_names()
    tracked = resolve_frames_for_g1_site_mode(frame_names)
    base_frame = tracked[0]
    print("Using frames:", tracked)

    map_q = make_visual_config_mapper(robot, urdf_vis)
    urdf_vis.update_cfg(map_q(q))

    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    tasks = []
    for i, frame in enumerate(tracked):
        t = solver.add_frame_task(f"g1_site_{i}", frame, embodik.TaskType.FRAME_POSITION)
        t.priority = 0
        t.weight = 1.0
        tasks.append(t)

    pose_pos_task = solver.add_frame_task(
        "g1_pose_pos",
        base_frame,
        embodik.TaskType.FRAME_POSITION,
    )
    pose_pos_task.priority = 0
    pose_pos_task.weight = 0.0

    pose_ori_task = solver.add_frame_task(
        "g1_pose_ori",
        base_frame,
        embodik.TaskType.FRAME_ORIENTATION,
    )
    pose_ori_task.priority = 0
    pose_ori_task.weight = 0.0

    base_pose = robot.get_frame_pose(base_frame)
    base_rot = np.asarray(base_pose.rotation, dtype=float)
    base_wxyz = r2q(base_rot)
    ik_target = server.scene.add_transform_controls(
        "/ik_target_site",
        scale=0.15,
        position=tuple(np.asarray(base_pose.translation, dtype=float)),
        wxyz=(base_wxyz[3], base_wxyz[0], base_wxyz[1], base_wxyz[2]),
    )

    with server.gui.add_folder("IK Controls"):
        mode = server.gui.add_dropdown(
            "IK Goal Mode",
            options=("G1 3-point", "EmbodiK 6D"),
            initial_value="G1 3-point",
        )
        base_p = np.asarray(robot.get_frame_pose(base_frame).translation)
        slider_x = server.gui.add_slider("X", base_p[0] - 1.0, base_p[0] + 1.0, 0.001, base_p[0])
        slider_y = server.gui.add_slider("Y", base_p[1] - 1.0, base_p[1] + 1.0, 0.001, base_p[1])
        slider_z = server.gui.add_slider("Z", base_p[2] - 1.0, base_p[2] + 1.0, 0.001, base_p[2])
        rot = np.asarray(robot.get_frame_pose(base_frame).rotation, dtype=float)
        rpy = _rot_to_rpy(rot)
        slider_r = server.gui.add_slider("Roll", -np.pi, np.pi, 0.001, rpy[0])
        slider_p = server.gui.add_slider("Pitch", -np.pi, np.pi, 0.001, rpy[1])
        slider_yaw = server.gui.add_slider("Yaw", -np.pi, np.pi, 0.001, rpy[2])
        ik_r = server.gui.add_checkbox("IK_R (strict orientation points)", initial_value=False)
        ori_weight = server.gui.add_slider("6D Orientation Weight", 0.01, 5.0, 0.01, 1.0)
        steps = server.gui.add_slider("IK Steps", 1, 20, 1, 5)
        gain = server.gui.add_slider("Position Gain", 1.0, 60.0, 0.5, 20.0)

    presets = get_retargeting_presets()
    with server.gui.add_folder("Retargeting Examples"):
        retarget_preset = server.gui.add_dropdown(
            "Preset",
            options=tuple(presets.keys()),
            initial_value="reach_forward",
        )
        retarget_scale = server.gui.add_slider("Retarget Scale", 0.5, 1.5, 0.01, 1.0)
        apply_retarget = server.gui.add_button("Apply retarget preset")

    with server.gui.add_folder("Diagnostics"):
        status = server.gui.add_text("Status", initial_value="running")
        solve_ms = server.gui.add_text("Solve time (ms)", initial_value="--")

    @apply_retarget.on_click
    def _(_evt) -> None:
        preset = presets[retarget_preset.value]
        base_pose_now = robot.get_frame_pose(base_frame)
        p_anchor = np.asarray(base_pose_now.translation, dtype=float)
        R_anchor = np.asarray(base_pose_now.rotation, dtype=float)
        offset = np.asarray(preset.get("right_palm", np.zeros(3, dtype=float)), dtype=float)
        p_world = retarget_position_from_anchor(
            p_anchor,
            R_anchor,
            offset,
            scale=float(retarget_scale.value),
        )
        ik_target.position = tuple(p_world)
        slider_x.value = float(p_world[0])
        slider_y.value = float(p_world[1])
        slider_z.value = float(p_world[2])
        status.value = f"Applied retarget preset: {retarget_preset.value}"

    # Preserve the current geometric layout of auxiliary points in the tracked
    # base frame coordinates. Using the actual base-frame orientation avoids
    # startup inconsistency for URDF variants that don't provide explicit
    # *_top/*_palmar/*_front hand frames.
    R_ref = np.asarray(robot.get_frame_pose(tracked[0]).rotation, dtype=float)
    p_ref = np.asarray(robot.get_frame_pose(tracked[0]).translation)
    local_offsets = [np.zeros(3)]
    for i in range(1, 4):
        p_i = np.asarray(robot.get_frame_pose(tracked[i]).translation)
        local_offsets.append(R_ref.T @ (p_i - p_ref))

    opts = embodik.PositionStepOptions()
    infeasible_streak = 0
    while True:
        p_target = np.array(ik_target.position, dtype=float)
        wxyz = np.asarray(ik_target.wxyz, dtype=float)
        q_xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=float)
        R_target = q2r(q_xyzw, order="xyzs")
        # Keep sliders synchronized for users who prefer numeric controls.
        slider_x.value = float(p_target[0])
        slider_y.value = float(p_target[1])
        slider_z.value = float(p_target[2])
        rpy_from_gizmo = _rot_to_rpy(R_target)
        slider_r.value = float(rpy_from_gizmo[0])
        slider_p.value = float(rpy_from_gizmo[1])
        slider_yaw.value = float(rpy_from_gizmo[2])
        targets = []
        points = []
        if mode.value == "G1 3-point":
            payloads = build_site_mode_target_payloads(
                "G1 3-point",
                p_target,
                R_target,
                local_offsets,
                position_gain=float(gain.value),
                orientation_gain=float(gain.value),
            )
            points = [pl["target_pose"][:3, 3].copy() for pl in payloads]
            aux_w = 1.0 if ik_r.value else 0.01
            tasks[0].weight = 1.0
            tasks[1].weight = aux_w
            tasks[2].weight = aux_w
            tasks[3].weight = aux_w
            pose_pos_task.weight = 0.0
            pose_ori_task.weight = 0.0
        else:
            payloads = build_site_mode_target_payloads(
                "EmbodiK 6D",
                p_target,
                R_target,
                local_offsets,
                position_gain=float(gain.value),
                orientation_gain=float(gain.value),
            )
            points = [p_target]
            tasks[0].weight = 0.0
            tasks[1].weight = 0.0
            tasks[2].weight = 0.0
            tasks[3].weight = 0.0
            pose_pos_task.weight = 1.0
            pose_ori_task.weight = float(ori_weight.value)
            payloads[1]["orientation_gain"] = float(ori_weight.value)

        for payload in payloads:
            targets.append(
                embodik.TaskTarget(
                    task_name=payload["task_name"],
                    target_pose=payload["target_pose"],
                    position_gain=float(payload["position_gain"]),
                    orientation_gain=float(payload["orientation_gain"]),
                )
            )

        opts.max_steps = int(steps.value)
        opts.position_gain = float(gain.value)
        opts.orientation_gain = float(gain.value)
        step = robust_solve_position_step(
            robot=robot,
            solver=solver,
            q_current=q,
            targets=targets,
            options=opts,
            q_lo=q_lo,
            q_hi=q_hi,
            fallback_status_names=("INVALID_INPUT", "INFEASIBLE", "NUMERICAL_ERROR", "NO_PROGRESS"),
        )
        q = step.q_next
        result = step.solver_result

        if result.status in (
            embodik.SolverStatus.INFEASIBLE,
            embodik.SolverStatus.NO_PROGRESS,
            embodik.SolverStatus.NUMERICAL_ERROR,
        ):
            infeasible_streak += 1
        else:
            infeasible_streak = 0

        if mode.value == "G1 3-point" and infeasible_streak >= 8 and ik_r.value:
            # Automatic recovery for hard hand-frame variants: relax auxiliary points.
            ik_r.value = False
            infeasible_streak = 0

        robot.update_configuration(q)
        urdf_vis.update_cfg(map_q(q))

        if mode.value == "G1 3-point" and (not ik_r.value):
            rot_now = _estimate_hand_rot(robot, tracked)
            rpy_now = _rot_to_rpy(rot_now)
            slider_r.value = float(rpy_now[0])
            slider_p.value = float(rpy_now[1])
            slider_yaw.value = float(rpy_now[2])

        errs = []
        if mode.value == "G1 3-point":
            for i, frame in enumerate(tracked):
                p_cur = np.asarray(robot.get_frame_pose(frame).translation)
                errs.append(float(np.linalg.norm(points[i] - p_cur)))
        else:
            p_cur = np.asarray(robot.get_frame_pose(base_frame).translation)
            errs.append(float(np.linalg.norm(points[0] - p_cur)))
        status.value = (
            f"{result.status.name} | max_err={max(errs):.4f}"
            + (f" | {result.status_message}" if getattr(result, "status_message", "") else "")
        )
        solve_ms.value = f"{step.elapsed_ms:.2f}"
        time.sleep(0.002)


if __name__ == "__main__":
    main()
