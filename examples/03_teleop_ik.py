#!/usr/bin/env python3
"""Minimal teleop-to-IK example.

This example shows the wiring pattern for connecting an input device to
embodiK:

1. Read a relative controller pose.
2. Convert that controller delta into a robot end-effector target pose.
3. Call ``backend.solve_step(goal_pose)``.
4. Send or visualize the returned joint positions.

Holding the trigger enables teleop streaming. If no Seer controller is found,
the same IK path runs from the draggable transform control in the browser.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from embodik import Rt, q2r, r2q
from example_helpers.ik_common import quiet_websocket_handshake_logs
from example_helpers.teleop_ik_backend import TeleopIKBackend
from utils.robot_models import load_robot_presets


quiet_websocket_handshake_logs()

try:
    import xvisio

    XVISIO_AVAILABLE = True
except ImportError:
    xvisio = None
    XVISIO_AVAILABLE = False


DEFAULT_SCALE_FACTOR = 1.5
BUTTON_A = 16
BUTTON_B = 32
TRIGGER_THRESHOLD = 30
SIDE_THRESHOLD = 30
BUTTON_DEBOUNCE_S = 2.0


def resolve_robot_configuration(robot_key: str) -> Any:
    """Reuse the richer collision-aware resolver without exposing it here."""

    examples_dir = Path(__file__).parent
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    collision_ik = importlib.import_module("02_collision_aware_IK")
    return collision_ik.resolve_robot_configuration(robot_key)


class SeerController:
    """Thin adapter from Seer controller events to teleop state."""

    def __init__(self, port: str):
        self.port = port
        self.device: Optional[Any] = None
        self.streaming = False
        self.gripper_closed = False
        self.data_collection = False
        self._prev_trigger = 0
        self._prev_side = 0
        self._prev_key = 0
        self._last_button_time = 0.0
        self.on_stream_start: Callable[[], None] = lambda: None
        self.on_stream_stop: Callable[[], None] = lambda: None
        self.on_reset: Callable[[], None] = lambda: None

    @property
    def connected(self) -> bool:
        return self.device is not None

    def connect(self) -> bool:
        if not XVISIO_AVAILABLE:
            print("xvisio not available; using browser transform controls.")
            return False

        try:
            controllers = xvisio.discover_controllers()
            if not controllers:
                print("No Seer controllers found; using browser transform controls.")
                return False

            print(f"Found controller: {controllers[0]}")
            self.device = xvisio.open_controller(port=self.port)
            time.sleep(0.5)
            self.reset_reference()
            return True
        except Exception as exc:
            print(f"Controller connection failed: {exc}")
            self.device = None
            return False

    def disconnect(self) -> None:
        if self.device is not None:
            self.device.close()
            self.device = None

    def reset_reference(self) -> None:
        if self.device is not None:
            self.device.reset_controller_reference()

    def relative_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.device is None:
            return None
        try:
            left, right = self.device.controller_relative()
            c = right if right is not None else left
            if c is None:
                return None
            return (
                np.asarray(c.position, dtype=float),
                np.asarray(c.quat_wxyz, dtype=float),
            )
        except Exception:
            return None

    def raw_buttons(self) -> Optional[Tuple[int, int, int]]:
        if self.device is None:
            return None
        try:
            left, right = self.device.controller()
            c = right if right is not None else left
            if c is None:
                return None
            return int(c.key_trigger), int(c.key_side), int(c.key)
        except Exception:
            return None

    def process_buttons(self) -> None:
        raw = self.raw_buttons()
        if raw is None:
            return

        trigger, side, key = raw
        now = time.time()

        if self._prev_trigger <= TRIGGER_THRESHOLD < trigger:
            self.streaming = True
            self.on_stream_start()
        elif self._prev_trigger > TRIGGER_THRESHOLD >= trigger:
            self.streaming = False
            self.on_stream_stop()

        if self._prev_side <= SIDE_THRESHOLD < side:
            self.gripper_closed = True
        elif self._prev_side > SIDE_THRESHOLD >= side:
            self.gripper_closed = False

        if key != self._prev_key and key != 0 and now - self._last_button_time > BUTTON_DEBOUNCE_S:
            if key == BUTTON_A:
                self.on_reset()
            elif key == BUTTON_B:
                self.data_collection = not self.data_collection
            self._last_button_time = now

        self._prev_trigger = trigger
        self._prev_side = side
        self._prev_key = key


def pose_from_control(control: Any) -> Rt:
    wxyz = np.asarray(control.wxyz, dtype=float)
    xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=float)
    return Rt(R=q2r(xyzw, order="xyzs"), t=np.asarray(control.position, dtype=float))


def apply_controller_delta(base_pose: Any, delta_pos: np.ndarray, delta_wxyz: np.ndarray, scale: float) -> Rt:
    delta_xyzw = np.array([delta_wxyz[1], delta_wxyz[2], delta_wxyz[3], delta_wxyz[0]], dtype=float)
    delta_R = q2r(delta_xyzw, order="xyzs")
    return Rt(
        R=delta_R @ base_pose.rotation,
        t=base_pose.translation + scale * delta_pos,
    )


def set_control_pose(control: Any, pose: Any) -> None:
    quat_xyzw = r2q(pose.rotation, order="xyzs")
    control.position = tuple(pose.translation)
    control.wxyz = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])


def parse_args() -> argparse.Namespace:
    presets = load_robot_presets()
    parser = argparse.ArgumentParser(description="Minimal teleop input to embodiK IK demo.")
    parser.add_argument("--robot", choices=sorted(presets), default="panda")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--controller-port", default="/dev/ttyUSB0")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE_FACTOR)
    parser.add_argument("--no-collision", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = resolve_robot_configuration(args.robot)

    backend = TeleopIKBackend(cfg, enable_collision=not args.no_collision)
    controller = SeerController(args.controller_port)
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
        streaming_text = server.gui.add_text("Streaming", initial_value="OFF")
        gripper_text = server.gui.add_text("Gripper", initial_value="OPEN")
        data_text = server.gui.add_text("Data Collection", initial_value="OFF")
        status_text = server.gui.add_text("Status", initial_value="Ready")
        timing_ms = server.gui.add_number("IK Time (ms)", 0.001, disabled=True)
        scale_slider = server.gui.add_slider("Position Scale", min=0.5, max=3.0, initial_value=args.scale, step=0.1)
        manual_mode = server.gui.add_checkbox("Manual Target", initial_value=not controller_connected)
        reset_button = server.gui.add_button("Reset Robot & Controller")

    arm_stream_start_pose = backend.get_pose()
    goal_pose = current_pose

    def reset_session() -> None:
        nonlocal arm_stream_start_pose, goal_pose
        pose = backend.reset()
        controller.reset_reference()
        arm_stream_start_pose = pose
        goal_pose = pose
        set_control_pose(target_control, pose)
        urdf_vis.update_cfg(make_visual_config(backend.get_q()))
        status_text.value = "Reset"

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

    urdf_vis.update_cfg(make_visual_config(backend.get_q()))

    print("\nTELEOP IK DEMO")
    print(f"Robot: {cfg.display_name}")
    print(f"Open http://localhost:{args.port} in your browser")
    if controller_connected:
        print("Hold trigger to stream controller motion into embodiK IK.")
        print("Hold side button to toggle gripper status.")
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
                    set_control_pose(target_control, goal_pose)

            if controller.streaming or manual_mode.value:
                result = backend.solve_step(goal_pose)
                urdf_vis.update_cfg(make_visual_config(result.joints))
                timing_ms.value = 0.9 * timing_ms.value + 0.1 * result.elapsed_ms
                if frame_count % 50 == 0:
                    status_text.value = (
                        f"{result.status}: pos={result.position_error * 1e3:.1f} mm"
                    )

            gripper_text.value = "CLOSED" if controller.gripper_closed else "OPEN"
            data_text.value = "ON" if controller.data_collection else "OFF"

            frame_count += 1
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nTeleop stopped")
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
