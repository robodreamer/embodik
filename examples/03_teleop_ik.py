#!/usr/bin/env python3
"""Minimal teleop-to-IK example.

This example shows the wiring pattern for connecting an input device to
embodiK:

1. Read a relative controller pose.
2. Convert that controller delta into a robot end-effector target pose.
3. Call ``backend.solve_step(goal_pose)``.
4. Send or visualize the returned joint positions.

Holding the A button enables teleop streaming. Trigger travel streams the
gripper command, and Button B resets the session. If no Seer controller is found,
the same IK path runs from the draggable transform control in the browser.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Any

_EXAMPLES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLES_DIR.parent
_PYTHON_DIR = _REPO_ROOT / "python"
for _path in (_PYTHON_DIR, _EXAMPLES_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import viser
from embodik import r2q
from example_helpers.ik_common import DEFAULT_VISER_PORT, quiet_websocket_handshake_logs
from embodik.gpu.wbc import (
    GpuWbcMultiFrameSolver,
)
from example_helpers.seer_teleop import (
    DEFAULT_TELEOP_SCALE_FACTOR,
    SeerController,
    apply_controller_delta,
    pose_from_transform_control,
    set_transform_control_pose,
)
from example_helpers.teleop_ik_backend import TeleopIKBackend
from robot_descriptions.loaders.yourdfpy import load_robot_description
from utils.robot_models import load_robot_presets
from viser.extras import ViserUrdf

quiet_websocket_handshake_logs()

DEFAULT_SCALE_FACTOR = DEFAULT_TELEOP_SCALE_FACTOR
_basic_gpu = importlib.import_module("01_basic_ik_simple")


def resolve_robot_configuration(robot_key: str) -> Any:
    """Reuse the richer collision-aware resolver without exposing it here."""

    examples_dir = Path(__file__).parent
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    collision_ik = importlib.import_module("02_collision_aware_IK")
    return collision_ik.resolve_robot_configuration(robot_key)


def pose_from_control(control: Any):
    return pose_from_transform_control(control)


def parse_args() -> argparse.Namespace:
    presets = load_robot_presets()
    parser = argparse.ArgumentParser(description="Minimal teleop input to embodiK IK demo.")
    parser.add_argument("--robot", choices=sorted(presets), default="panda")
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
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE_FACTOR)
    parser.add_argument("--no-collision", action="store_true")
    parser.add_argument("--gpu-wbc", action="store_true")
    parser.add_argument("--gpu-wbc-manifest", type=Path)
    parser.add_argument(
        "--gpu-wbc-cache-dir",
        type=Path,
        default=Path("build/gpu-wbc-newton-cache"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = resolve_robot_configuration(args.robot)

    backend = TeleopIKBackend(cfg, enable_collision=not args.no_collision)
    gpu_solver: GpuWbcMultiFrameSolver | None = None
    gpu_target_offset = None
    gpu_fault: str | None = None
    if args.gpu_wbc:
        gpu_solver, gpu_indices, gpu_collision_supported = _basic_gpu.create_gpu_solver(
            args,
            backend.robot,
            cfg.urdf_path,
            cfg.key,
            cfg.target_link,
            backend.default_full,
            exclusions=cfg.collision_exclusions,
            collision_enabled=not args.no_collision,
            min_distance=0.05,
        )
        visible_pose = backend.get_pose()
        solver_pose = backend.robot.get_frame_pose(gpu_solver.frames[0])
        gpu_target_offset = visible_pose.inverse() * solver_pose
        gpu_solver.warm_up(backend.q[list(gpu_indices)], (visible_pose * gpu_target_offset,))
    controller = SeerController(args.controller_port if args.enable_teleop else None)
    controller_connected = controller.connect()

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=2, height=2)
    urdf = load_robot_description(cfg.description_name)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    actuated_names = list(getattr(urdf_vis._urdf, "actuated_joint_names", []))
    name_to_index = {name: idx for idx, name in enumerate(cfg.joint_names)}

    def make_visual_config(q_arm: np.ndarray) -> np.ndarray:
        if not actuated_names:
            return q_arm
        cfg_vec = np.zeros(len(actuated_names), dtype=float)
        for i, joint_name in enumerate(actuated_names):
            idx = name_to_index.get(joint_name)
            if idx is not None and idx < q_arm.size:
                cfg_vec[i] = q_arm[idx]
        return cfg_vec

    current_pose = backend.get_pose()
    target_control = server.scene.add_transform_controls(
        "/ik_target",
        scale=0.2,
        position=tuple(current_pose.translation),
        wxyz=tuple(r2q(current_pose.rotation)),
    )

    with server.gui.add_folder("Teleop"):
        backend_select = server.gui.add_dropdown(
            "Solver Backend",
            options=("CPU EmbodiK",) + (("GPU Newton/Warp",) if gpu_solver is not None else ()),
            initial_value="CPU EmbodiK",
        )
        streaming_text = server.gui.add_text("Streaming", initial_value="OFF")
        gripper_text = server.gui.add_text("Gripper", initial_value="OPEN")
        reset_hint = server.gui.add_text("Reset", initial_value="Button B")
        status_text = server.gui.add_text("Status", initial_value="Ready")
        timing_ms = server.gui.add_number("IK Time (ms)", 0.001, disabled=True)
        scale_slider = server.gui.add_slider(
            "Position Scale", min=0.5, max=3.0, initial_value=args.scale, step=0.1
        )
        manual_mode = server.gui.add_checkbox(
            "Manual Target", initial_value=not controller_connected
        )
        reset_button = server.gui.add_button("Reset Robot & Controller")

    gpu_controls = (
        _basic_gpu.GpuControls(
            server.gui, server.scene, gpu_collision_supported, not args.no_collision, 0.05
        )
        if gpu_solver is not None
        else None
    )

    arm_stream_start_pose = backend.get_pose()
    goal_pose = current_pose

    def reset_session() -> None:
        nonlocal arm_stream_start_pose, goal_pose, gpu_fault
        gpu_fault = None
        pose = backend.reset()
        controller.reset_reference()
        arm_stream_start_pose = pose
        goal_pose = pose
        set_transform_control_pose(target_control, pose)
        urdf_vis.update_cfg(make_visual_config(backend.get_q()))
        status_text.value = "Reset"
        if gpu_solver is not None:
            gpu_solver.reset_state()
            gpu_controls.show_debug()

    def start_streaming() -> None:
        nonlocal arm_stream_start_pose
        controller.reset_reference()
        arm_stream_start_pose = backend.get_pose()
        streaming_text.value = "ON"

    def stop_streaming() -> None:
        streaming_text.value = "OFF"

    controller.on_stream_start = start_streaming
    controller.on_stream_stop = stop_streaming
    controller.on_reset = reset_session

    @reset_button.on_click
    def _(_) -> None:
        reset_session()

    @backend_select.on_update
    def _(_) -> None:
        nonlocal gpu_fault
        gpu_fault = None
        if gpu_solver is not None:
            gpu_solver.reset_state()
            gpu_controls.show_debug()

    urdf_vis.update_cfg(make_visual_config(backend.get_q()))

    print("\nTELEOP IK DEMO")
    print(f"Robot: {cfg.display_name}")
    print(f"Open http://localhost:{args.port} in your browser")
    if controller_connected:
        print("Hold A to stream controller motion into embodiK IK.")
        print("Use trigger travel to stream the gripper command. Press B to reset.")
    else:
        print("No controller connected; drag /ik_target in the browser.")

    frame_count = 0
    try:
        while True:
            controller.process_buttons()

            if manual_mode.value or not controller.connected:
                goal_pose = pose_from_control(target_control)
            elif controller.streaming:
                delta = controller.relative_pose()
                if delta is not None:
                    delta_pos, delta_wxyz = delta
                    goal_pose = apply_controller_delta(
                        arm_stream_start_pose,
                        delta_pos,
                        delta_wxyz,
                        float(scale_slider.value),
                    )
                    set_transform_control_pose(target_control, goal_pose)

            if controller.streaming or manual_mode.value:
                if backend_select.value == "GPU Newton/Warp":
                    assert gpu_solver is not None
                    assert gpu_target_offset is not None
                    if gpu_fault is None:
                        try:
                            gpu_result = gpu_controls.solve(
                                gpu_solver,
                                backend.q[list(gpu_indices)],
                                goal_pose * gpu_target_offset,
                            )
                        except Exception as exc:
                            gpu_fault = f"{type(exc).__name__}: {exc}"
                            status_text.value = f"GPU FAULT — SAFE HOLD: {gpu_fault}"
                            gpu_controls.show_debug()
                            time.sleep(0.001)
                            continue
                        else:
                            backend.q[list(gpu_indices)] = gpu_result.joints
                            backend.robot.update_configuration(backend.q)
                            from example_helpers.teleop_ik_backend import IKResult

                            result = IKResult(
                                joints=backend.get_q(),
                                status=gpu_result.status,
                                position_error=max(gpu_result.position_errors),
                                rotation_error=max(gpu_result.rotation_errors),
                                elapsed_ms=gpu_result.elapsed_ms,
                            )
                    else:
                        time.sleep(0.001)
                        continue
                else:
                    result = backend.solve_step(goal_pose)
                urdf_vis.update_cfg(make_visual_config(result.joints))
                timing_ms.value = 0.9 * timing_ms.value + 0.1 * result.elapsed_ms
                if frame_count % 50 == 0:
                    status_text.value = f"{result.status}: pos={result.position_error * 1e3:.1f} mm"

            gripper_text.value = "CLOSED" if controller.gripper_closed else "OPEN"
            reset_hint.value = "Button B"

            frame_count += 1
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nTeleop stopped")
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
