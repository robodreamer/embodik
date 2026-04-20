#!/usr/bin/env python3
"""ROBOTIS AI worker dual-arm IK demo using local FFW URDF assets.

This example targets the locally available FFW SG2/BG2 URDFs and mirrors the
interactive dual-tool workflow used in the G1 notebooks, adapted to an
embodiK + Viser loop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import embodik
from embodik.utils import q2r, r2q

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.incubating.example_helpers.robotis_ai_worker_utils import (
    default_worker_allowed_joint_names,
    resolve_ai_worker_frames,
    resolve_ffw_urdf_path,
)
from examples.incubating.g1_port_phase1.g1_viser_utils import make_visual_config_mapper
from examples.incubating.g1_port_phase1.robust_ik_runtime import (
    clear_all_target_velocities_if_available,
    clip_configuration,
    configure_primary_solve_mode,
    robust_solve_position_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("sg2", "bg2"), default="sg2")
    parser.add_argument("--port", type=int, default=8092)
    return parser.parse_args()


def _pose_from_ctrl(ctrl) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = np.asarray(ctrl.position, dtype=float)
    wxyz = np.asarray(ctrl.wxyz, dtype=float)
    n = float(np.linalg.norm(wxyz))
    if not np.isfinite(n) or n < 1e-12:
        wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        wxyz = wxyz / n
    pose[:3, :3] = q2r(np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=float), order="xyzs")
    return pose


def _ctrl_from_pose(ctrl, pose) -> None:
    pose = np.asarray(pose, dtype=float)
    q_xyzw = r2q(pose[:3, :3], order="xyzs")
    ctrl.position = tuple(np.asarray(pose[:3, 3], dtype=float))
    ctrl.wxyz = (float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))


def main() -> None:
    args = parse_args()
    urdf_path = resolve_ffw_urdf_path(args.variant)

    import viser
    from robot_descriptions.loaders.yourdfpy import load_robot_description
    from viser.extras import ViserUrdf

    robot = embodik.RobotModel(str(urdf_path), floating_base=False)
    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=4, height=4)

    urdf_vis = ViserUrdf(server, load_robot_description(str(urdf_path)), root_node_name="/robot")
    map_q = make_visual_config_mapper(robot, urdf_vis)

    q = robot.neutral_configuration()
    q_lo, q_hi = robot.get_joint_limits()
    robot.update_configuration(q)
    urdf_vis.update_cfg(map_q(q))

    frame_map = resolve_ai_worker_frames(robot.get_frame_names())
    print(f"[worker] variant={args.variant} urdf={urdf_path}")
    print(f"[worker] frames={frame_map}")

    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    right_task = solver.add_frame_task("right_tool_pose", frame_map["right_tool"], embodik.TaskType.FRAME_POSE)
    left_task = solver.add_frame_task("left_tool_pose", frame_map["left_tool"], embodik.TaskType.FRAME_POSE)
    right_task.priority = 0
    left_task.priority = 0
    right_task.weight = 1.0
    left_task.weight = 1.0

    posture = solver.add_posture_task("worker_posture")
    posture.priority = 1
    posture.weight = 0.02
    posture.set_target_configuration(q.copy())

    joint_names = list(robot.get_joint_names())
    joint_name_to_cfg = {}
    if hasattr(robot, "get_joint_config_index"):
        for name in joint_names:
            try:
                joint_name_to_cfg[name] = int(robot.get_joint_config_index(name))
            except Exception:
                pass
    allowed_joint_names = default_worker_allowed_joint_names(joint_names)
    locked_velocity_indices: list[int] = []
    for joint_name in joint_names:
        if joint_name in allowed_joint_names:
            continue
        if not hasattr(robot, "get_joint_velocity_index"):
            continue
        idx_v = int(robot.get_joint_velocity_index(joint_name))
        if hasattr(robot, "get_joint_velocity_size"):
            nv_joint = int(robot.get_joint_velocity_size(joint_name))
        else:
            nv_joint = 1
        for offset in range(max(nv_joint, 1)):
            locked_velocity_indices.append(idx_v + offset)
    locked_velocity_indices = sorted(set(locked_velocity_indices))

    def frame_pose(frame_name: str) -> np.ndarray:
        pose = robot.get_frame_pose(frame_name)
        T = np.eye(4, dtype=float)
        T[:3, :3] = np.asarray(pose.rotation, dtype=float)
        T[:3, 3] = np.asarray(pose.translation, dtype=float)
        return T

    right_ctrl = server.scene.add_transform_controls("/target/right_tool", scale=0.14)
    left_ctrl = server.scene.add_transform_controls("/target/left_tool", scale=0.14)
    _ctrl_from_pose(right_ctrl, frame_pose(frame_map["right_tool"]))
    _ctrl_from_pose(left_ctrl, frame_pose(frame_map["left_tool"]))

    posture_target = q.copy()

    def _joint_value(name: str, default: float = 0.0) -> float:
        idx = joint_name_to_cfg.get(name)
        if idx is None or idx >= q.size:
            return default
        return float(q[idx])

    with server.gui.add_folder("IK Controls"):
        ik_steps = server.gui.add_slider("IK Steps", 1, 20, 1, 5)
        pos_gain = server.gui.add_slider("Position Gain", 1.0, 60.0, 0.5, 14.0)
        ori_gain = server.gui.add_slider("Orientation Gain", 0.1, 60.0, 0.1, 10.0)
        solve_mode = server.gui.add_dropdown(
            "Solve Mode",
            options=("SCALE", "SCALE_ELASTIC", "MIN_ERROR"),
            initial_value="SCALE_ELASTIC",
        )
        allow_fallback = server.gui.add_checkbox("Allow SCALE fallback", initial_value=False)
        lock_passive = server.gui.add_checkbox("Lock passive joints", initial_value=True)
        posture_weight = server.gui.add_slider("Posture Weight", 0.0, 0.2, 0.001, 0.02)

    with server.gui.add_folder("Worker Posture"):
        lift_slider = server.gui.add_slider("Lift Joint", -0.5, 0.0, 0.001, _joint_value("lift_joint"))
        head_pitch = server.gui.add_slider("Head Pitch", -0.2317, 0.6951, 0.001, _joint_value("head_joint1"))
        head_yaw = server.gui.add_slider("Head Yaw", -0.35, 0.35, 0.001, _joint_value("head_joint2"))
        snap_targets = server.gui.add_button("Snap Targets to Current Tools")
        reset_pose = server.gui.add_button("Reset Robot + Targets")

    with server.gui.add_folder("Diagnostics"):
        status = server.gui.add_text("Status", initial_value="running")
        solve_ms = server.gui.add_text("Solve time (ms)", initial_value="--")
        right_err = server.gui.add_text("Right err", initial_value="--")
        left_err = server.gui.add_text("Left err", initial_value="--")

    @snap_targets.on_click
    def _(_evt) -> None:
        _ctrl_from_pose(right_ctrl, frame_pose(frame_map["right_tool"]))
        _ctrl_from_pose(left_ctrl, frame_pose(frame_map["left_tool"]))

    @reset_pose.on_click
    def _(_evt) -> None:
        nonlocal q, posture_target
        q = robot.neutral_configuration()
        q = clip_configuration(robot, q, q_lo, q_hi)
        posture_target = q.copy()
        robot.update_configuration(q)
        urdf_vis.update_cfg(map_q(q))
        _ctrl_from_pose(right_ctrl, frame_pose(frame_map["right_tool"]))
        _ctrl_from_pose(left_ctrl, frame_pose(frame_map["left_tool"]))

    opts = embodik.PositionStepOptions()

    while True:
        q_prev = np.asarray(q, dtype=float).copy()
        clear_all_target_velocities_if_available(solver)

        active_mode = getattr(
            embodik.TaskSolveMode,
            solve_mode.value,
            embodik.TaskSolveMode.SCALE,
        )
        for task in (right_task, left_task):
            task.solve_mode = active_mode
            task.allow_min_error_fallback = bool(allow_fallback.value)

        posture.weight = float(posture_weight.value)
        for joint_name, slider in (
            ("lift_joint", lift_slider),
            ("head_joint1", head_pitch),
            ("head_joint2", head_yaw),
        ):
            idx = joint_name_to_cfg.get(joint_name)
            if idx is not None and idx < posture_target.size:
                posture_target[idx] = float(slider.value)
        posture.set_target_configuration(posture_target)

        targets = [
            embodik.TaskTarget(
                "right_tool_pose",
                _pose_from_ctrl(right_ctrl),
                float(pos_gain.value),
                float(ori_gain.value),
            ),
            embodik.TaskTarget(
                "left_tool_pose",
                _pose_from_ctrl(left_ctrl),
                float(pos_gain.value),
                float(ori_gain.value),
            ),
        ]
        opts.max_steps = int(ik_steps.value)
        opts.position_gain = float(pos_gain.value)
        opts.orientation_gain = float(ori_gain.value)
        configure_primary_solve_mode(opts, active_mode, bool(allow_fallback.value))
        if lock_passive.value:
            opts.excluded_joint_indices = list(locked_velocity_indices)
            opts.integration_zero_velocity_indices = list(locked_velocity_indices)
        else:
            opts.excluded_joint_indices = []
            opts.integration_zero_velocity_indices = []

        step = robust_solve_position_step(
            robot=robot,
            solver=solver,
            q_current=q,
            targets=targets,
            options=opts,
            q_lo=q_lo,
            q_hi=q_hi,
            zero_velocity_indices=locked_velocity_indices if lock_passive.value else (),
            fallback_status_names=("INVALID_INPUT", "NUMERICAL_ERROR"),
        )
        q = step.q_next
        result = step.solver_result
        if not np.all(np.isfinite(q)):
            q = q_prev

        robot.update_configuration(q)
        urdf_vis.update_cfg(map_q(q))

        right_now = np.asarray(robot.get_frame_pose(frame_map["right_tool"]).translation, dtype=float)
        left_now = np.asarray(robot.get_frame_pose(frame_map["left_tool"]).translation, dtype=float)
        right_tgt = np.asarray(right_ctrl.position, dtype=float)
        left_tgt = np.asarray(left_ctrl.position, dtype=float)
        right_err.value = f"{np.linalg.norm(right_tgt - right_now):.4f} m"
        left_err.value = f"{np.linalg.norm(left_tgt - left_now):.4f} m"
        status.value = (
            f"{result.status.name}"
            + (f" | {result.status_message}" if getattr(result, "status_message", "") else "")
        )
        solve_ms.value = f"{step.elapsed_ms:.2f}"
        time.sleep(0.002)


if __name__ == "__main__":
    main()
