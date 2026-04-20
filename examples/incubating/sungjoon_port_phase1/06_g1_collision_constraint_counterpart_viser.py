#!/usr/bin/env python3
"""Unitree G1 collision-constraint counterpart with viser.

Covers G1-style collision/contact exploration with:
- collision constraint enable/disable
- min-distance margin tuning
- closest-pair debug visualization
"""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path
import numpy as np
import embodik

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.incubating.g1_port_phase1.g1_viser_utils import (
    collision_arm_only_option_values,
    compute_excluded_velocity_indices,
    create_g1_robot_and_visual,
    make_visual_config_mapper,
)
from examples.incubating.g1_port_phase1.robust_ik_runtime import robust_solve_position_step


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8086)
    p.add_argument("--floating-base", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    robot, server, urdf_vis = create_g1_robot_and_visual(
        floating_base=args.floating_base,
        port=args.port,
    )
    q = robot.neutral_configuration()
    q_lo, q_hi = robot.get_joint_limits()
    robot.update_configuration(q)
    map_q = make_visual_config_mapper(robot, urdf_vis)
    urdf_vis.update_cfg(map_q(q))

    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    task_frame = "right_rubber_hand"
    if task_frame not in robot.get_frame_names():
        task_frame = "right_wrist_yaw_link"
    task = solver.add_frame_task("primary", task_frame, embodik.TaskType.FRAME_POSITION)
    task.priority = 0
    task.weight = 1.0

    posture = solver.add_posture_task("posture")
    posture.priority = 1
    posture.weight = 0.01
    posture.set_target_configuration(q.copy())

    cur = np.asarray(robot.get_frame_pose(task_frame).translation)
    target = server.scene.add_transform_controls(
        "/target",
        scale=0.08,
        position=tuple(cur + np.array([0.08, -0.06, -0.02])),
        wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    with server.gui.add_folder("IK Controls"):
        ik_steps = server.gui.add_slider("IK Steps", 1, 20, 1, 5)
        pos_gain = server.gui.add_slider("Position Gain", 1.0, 60.0, 0.5, 20.0)
        hold_err_mm = server.gui.add_slider("Hold threshold (mm)", 1, 50, 1, 12)

    with server.gui.add_folder("Collision"):
        enable_collision = server.gui.add_checkbox("Enable", initial_value=True)
        min_dist_mm = server.gui.add_slider("Min distance (mm)", 1, 120, 1, 40)
        mode = server.gui.add_dropdown(
            "Tuning",
            options=("speed", "balanced", "precise"),
            initial_value="balanced",
        )
        show_debug = server.gui.add_checkbox("Show closest-pair debug", initial_value=True)
        arm_only = server.gui.add_checkbox("Arm-only (lock legs)", initial_value=True)

    with server.gui.add_folder("Diagnostics"):
        status = server.gui.add_text("Status", initial_value="running")
        closest_pair = server.gui.add_text("Closest pair", initial_value="--")
        solve_ms = server.gui.add_text("Solve time (ms)", initial_value="--")

    dbg_a = server.scene.add_icosphere(
        "/collision_debug/point_a",
        radius=0.012,
        color=(1.0, 0.2, 0.2),
        visible=False,
    )
    dbg_b = server.scene.add_icosphere(
        "/collision_debug/point_b",
        radius=0.012,
        color=(0.2, 0.9, 0.2),
        visible=False,
    )
    dbg_line = None

    def _apply_collision() -> None:
        if hasattr(embodik, "CollisionTuningMode"):
            mode_map = {
                "speed": embodik.CollisionTuningMode.SPEED,
                "balanced": embodik.CollisionTuningMode.BALANCED,
                "precise": embodik.CollisionTuningMode.PRECISE,
            }
            solver.set_collision_tuning_mode(mode_map[mode.value])
        if enable_collision.value:
            solver.configure_collision_constraint(
                min_distance=float(min_dist_mm.value) * 0.001,
                include_pairs=[],
                exclude_pairs=[],
            )
        else:
            solver.clear_collision_constraint()

    def _update_collision_debug() -> None:
        nonlocal dbg_line
        if not show_debug.value:
            dbg_a.visible = False
            dbg_b.visible = False
            if dbg_line is not None:
                dbg_line.visible = False
            closest_pair.value = "--"
            return

        info = solver.get_last_collision_debug() if hasattr(solver, "get_last_collision_debug") else None
        if info is None:
            dbg_a.visible = False
            dbg_b.visible = False
            if dbg_line is not None:
                dbg_line.visible = False
            closest_pair.value = "No active pair"
            return

        p_a = np.asarray(info.point_a_world, dtype=float)
        p_b = np.asarray(info.point_b_world, dtype=float)
        dbg_a.position = tuple(p_a)
        dbg_b.position = tuple(p_b)
        dbg_a.visible = True
        dbg_b.visible = True

        if dbg_line is not None:
            dbg_line.remove()
        seg = np.zeros((1, 2, 3), dtype=float)
        seg[0, 0] = p_a
        seg[0, 1] = p_b
        colors = np.array([[[1.0, 0.2, 0.2], [0.2, 0.9, 0.2]]], dtype=float)
        dbg_line = server.scene.add_line_segments(
            "/collision_debug/segment",
            points=seg,
            colors=colors,
            line_width=3.0,
            visible=True,
        )
        closest_pair.value = f"{info.object_a} <-> {info.object_b} | d={info.distance:.4f} m"

    _apply_collision()

    @enable_collision.on_update
    def _(_evt) -> None:
        _apply_collision()

    @min_dist_mm.on_update
    def _(_evt) -> None:
        _apply_collision()

    @mode.on_update
    def _(_evt) -> None:
        _apply_collision()

    allowed_joint_names = {
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    }
    excluded_joint_indices = compute_excluded_velocity_indices(robot, allowed_joint_names)
    if getattr(robot, "is_floating_base", False):
        # Freeze floating-base velocities in arm-only mode as well.
        excluded_joint_indices = sorted(set(excluded_joint_indices + list(range(6))))

    print("[arm-only] excluding", len(excluded_joint_indices), "velocity indices:", excluded_joint_indices)

    opts = embodik.PositionStepOptions()
    opts.orientation_gain = 0.0

    while True:
        p_now = np.asarray(robot.get_frame_pose(task_frame).translation)
        err_now = float(np.linalg.norm(np.asarray(target.position) - p_now))
        hold_err = float(hold_err_mm.value) * 0.001
        if err_now <= hold_err:
            _update_collision_debug()
            status.value = f"HOLD | err={err_now:.4f} m"
            time.sleep(0.002)
            continue

        pose = np.eye(4)
        pose[:3, 3] = np.asarray(target.position)
        opts.position_gain = float(pos_gain.value)
        opts.max_steps = int(ik_steps.value)
        opts.stall_recovery = True
        opts.no_progress_max_steps = 5
        opts.no_progress_error_tolerance = 1e-5
        opts.no_progress_dq_norm_tolerance = 1e-6
        excl, zero_idx = collision_arm_only_option_values(
            arm_only_enabled=bool(arm_only.value),
            floating_base=bool(getattr(robot, "is_floating_base", False)),
            excluded_joint_indices=excluded_joint_indices,
        )
        opts.excluded_joint_indices = excl
        opts.integration_zero_velocity_indices = zero_idx
        step = robust_solve_position_step(
            robot=robot,
            solver=solver,
            q_current=q,
            targets=[embodik.TaskTarget("primary", pose, float(pos_gain.value), 0.0)],
            options=opts,
            q_lo=q_lo,
            q_hi=q_hi,
            zero_velocity_indices=excluded_joint_indices if arm_only.value else (),
            fallback_status_names=("INVALID_INPUT", "INFEASIBLE", "NUMERICAL_ERROR", "NO_PROGRESS"),
        )
        q = step.q_next
        result = step.solver_result
        robot.update_configuration(q)
        urdf_vis.update_cfg(map_q(q))
        _update_collision_debug()

        p = np.asarray(robot.get_frame_pose(task_frame).translation)
        err = float(np.linalg.norm(np.asarray(target.position) - p))
        status.value = (
            f"{result.status.name} | err={err:.4f} m"
            + (f" | {result.status_message}" if getattr(result, "status_message", "") else "")
        )
        solve_ms.value = f"{step.elapsed_ms:.2f}"
        time.sleep(0.002)


if __name__ == "__main__":
    main()
