#!/usr/bin/env python3
"""Teleop IK Example using Seer Wireless Controller.

This example demonstrates teleoperation using the Seer wireless controller
with embodiK for collision-aware IK. The implementation follows the patterns
from teleop_stack/teleop_spot/teleop_devices/tracking_camera_device_handler.py.

Key patterns from teleop_stack:
- Button configuration with thresholds and debounce (BUTTON_CONFIG)
- Wireless controller button mappings (WIRELESS_BUTTON_MAPPINGS)
- Pose calculation: goal = arm_init_pose + pose_change (with scale factor)
- Separate process_input() and send_robot_command() phases

Button Mappings (thor_wireless):
- Side button (hold)  : keyside_above_threshold → toggle_stream_on
                       keyside_below_threshold → toggle_stream_off
- Trigger (hold)      : keytrigger_above_threshold → toggle_grasping_on
                       keytrigger_below_threshold → toggle_grasping_off
- Button A            : key_press buttonA → reset_robot_pose
- Button B            : key_press buttonB → toggle_data_collection

Usage:
    # Basic usage with panda robot
    pixi run python3 examples/03_teleop_ik.py --robot panda

    # With custom scale factor
    pixi run python3 examples/03_teleop_ik.py --robot iiwa --scale 2.0

    # Disable collision avoidance
    pixi run python3 examples/03_teleop_ik.py --no-collision

    # Custom controller port
    pixi run python3 examples/03_teleop_ik.py --controller-port /dev/ttyUSB1

Requirements:
    - xvisio package (pip install xvisio) for Seer controller support
    - If no controller is connected, runs in GUI-only mode with transform controls
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pinocchio as pin
import viser
from viser.extras import ViserUrdf

import embodik
from robot_descriptions.loaders.yourdfpy import load_robot_description
from embodik.utils import compute_pose_error, limit_task_velocity
from embodik import r2q, q2r, Rt
from utils.robot_models import load_robot_presets

# Try to import xvisio for controller support
try:
    import xvisio
    XVISIO_AVAILABLE = True
except ImportError:
    XVISIO_AVAILABLE = False
    print("Warning: xvisio not available. Controller input disabled.")
    print("Install with: pip install xvisio")

# -----------------------------------------------------------------------------
# Configuration (matching teleop_stack patterns from tracking_camera_device_handler.py)
# -----------------------------------------------------------------------------

# Button configuration matching TRACKING_CAMERA_BUTTON_CONFIG
BUTTON_CONFIG = {
    # Controller key values - map button names to numeric values
    "button_values": {
        "buttonA": 16,  # Key value that maps to button A press
        "buttonB": 32,  # Key value that maps to button B press
    },
    # Button press debounce delay in seconds
    "button_press_debounce_delay": 2.0,
    # Threshold for keyside changes (0-100 scale from controller)
    "keyside_threshold": 30,
    # Threshold for keytrigger changes (0-100 scale from controller)
    "keytrigger_threshold": 30,
}

# Button mappings for wireless controller (thor_wireless)
# Available actions:
# - toggle_stream_on/off: Enable/disable streaming
# - toggle_grasping_on/off: Close/open gripper
# - reset_robot_pose: Reset robot to initial pose
# - toggle_data_collection: Toggle data collection
WIRELESS_BUTTON_MAPPINGS = {
    "keyside_above_threshold": "toggle_stream_on",
    "keyside_below_threshold": "toggle_stream_off",
    "keytrigger_above_threshold": "toggle_grasping_on",
    "keytrigger_below_threshold": "toggle_grasping_off",
    "key_press": {
        "buttonA": "reset_robot_pose",
        "buttonB": "toggle_data_collection",
    },
}

# Default numeric constants
DEFAULT_SOLVER_DT = 0.01
DEFAULT_POS_GAIN = 1e2
DEFAULT_ROT_GAIN = 1e2
DEFAULT_NULLSPACE_GAIN = 1e-3
MAX_LINEAR_STEP = 2.0
MAX_ANGULAR_STEP = 2.0

# Scale factor for translational changes (from TRACKING_CAMERA_INPUT_DEVICE_CONFIG)
DEFAULT_SCALE_FACTOR = 1.5


@dataclass
class TeleopState:
    """State for teleop session (matching teleop_stack patterns)."""
    streaming: bool = False
    gripper_closed: bool = False
    data_collection_active: bool = False

    # Debounce timestamps
    last_reset_time: float = 0.0
    last_data_collection_toggle_time: float = 0.0

    # Previous controller data for change detection (matches _prev_controller_data in teleop_stack)
    prev_keyside: int = 0
    prev_keytrigger: int = 0
    prev_key: int = 0


@dataclass
class RobotConfig:
    key: str
    display_name: str
    urdf_path: Path
    description_name: str
    target_link: str
    joint_labels: List[str]
    joint_names: List[str]
    default_configuration: np.ndarray
    default_offset: np.ndarray
    collision_exclusions: List[Tuple[str, str]]


# -----------------------------------------------------------------------------
# Robot Configuration (shared with other examples)
# -----------------------------------------------------------------------------

ROBOT_PRESETS: Dict[str, Dict[str, object]] = load_robot_presets()


def resolve_robot_configuration(robot_key: str) -> RobotConfig:
    """Resolve robot configuration from presets."""
    # Import the full resolver from example 02
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    # Use the same configuration logic as collision-aware IK
    from importlib import import_module
    collision_ik = import_module("02_collision_aware_IK")
    return collision_ik.resolve_robot_configuration(robot_key)


def ensure_ros_package_path(urdf_path: Path) -> None:
    """Ensure ROS_PACKAGE_PATH includes ancestors that contain meshes."""
    import os
    resolved = urdf_path.resolve()
    candidate_roots: List[Path] = []
    for depth in range(1, 5):
        if len(resolved.parents) > depth:
            candidate_roots.append(resolved.parents[depth])

    current = os.environ.get("ROS_PACKAGE_PATH", "")
    paths = [Path(p) for p in current.split(":") if p]
    updated = False
    for root in candidate_roots:
        if root.is_dir() and root not in paths:
            paths.insert(0, root)
            updated = True

    if updated:
        os.environ["ROS_PACKAGE_PATH"] = ":".join(str(p) for p in paths)


# -----------------------------------------------------------------------------
# Controller Input Handler (matching teleop_stack/tracking_camera_device_handler.py)
# -----------------------------------------------------------------------------

class ControllerInputHandler:
    """Handle Seer controller input with button mapping.

    This class mirrors the TrackingCameraDeviceHandler from teleop_stack,
    implementing the same button handling logic for the wireless controller.
    """

    def __init__(self, port: str = "/dev/ttyUSB0", config: Optional[Dict] = None):
        self.port = port
        self.device: Optional[Any] = None
        self.state = TeleopState()
        self._initialized = False

        # Use provided config or defaults
        self.config = config or BUTTON_CONFIG

        # Store configuration values as member variables (like teleop_stack)
        self._button_press_debounce_delay = self.config["button_press_debounce_delay"]
        self._keyside_threshold = self.config["keyside_threshold"]
        self._keytrigger_threshold = self.config["keytrigger_threshold"]
        self._button_values = self.config["button_values"]

        # Action handlers (populated by set_action_handlers)
        self._action_handlers: Dict[str, Any] = {}

    def connect(self) -> bool:
        """Connect to the Seer controller."""
        if not XVISIO_AVAILABLE:
            print("xvisio not available - controller disabled")
            return False

        try:
            controllers = xvisio.discover_controllers()
            if not controllers:
                print("No Seer controllers found")
                return False

            print(f"Found controller: {controllers[0]}")
            self.device = xvisio.open_controller(port=self.port)
            time.sleep(0.5)
            self.device.reset_controller_reference()
            self._initialized = True
            print("Controller connected and initialized")
            return True
        except Exception as e:
            print(f"Failed to connect controller: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from the controller."""
        if self.device is not None:
            self.device.close()
            self.device = None
            self._initialized = False

    def reset_reference(self) -> None:
        """Reset the controller pose reference.

        Mirrors reset_pose_offset() from teleop_stack.
        """
        if self.device is not None:
            self.device.reset_controller_reference()
            print("Controller pose reference reset")

    def set_action_handlers(self, handlers: Dict[str, Any]) -> None:
        """Set action handlers for button events.

        Args:
            handlers: Dict mapping action names to callback functions.
                Available actions (from WIRELESS_BUTTON_MAPPINGS):
                - toggle_stream_on: Called when streaming should start
                - toggle_stream_off: Called when streaming should stop
                - toggle_grasping_on: Called when gripper should close
                - toggle_grasping_off: Called when gripper should open
                - reset_robot_pose: Called when pose should be reset
                - toggle_data_collection: Toggle data collection
        """
        self._action_handlers = handlers

    def get_relative_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get controller pose relative to reference.

        Returns:
            Tuple of (position, quaternion_wxyz) or None if no data.
        """
        if self.device is None:
            return None

        try:
            left, right = self.device.controller_relative()
            # Prefer right controller
            c = right if right is not None else left
            if c is None:
                return None

            pos = np.array(c.position, dtype=np.float64)
            quat_wxyz = np.array(c.quat_wxyz, dtype=np.float64)
            return pos, quat_wxyz
        except Exception:
            return None

    def get_raw_controller_data(self) -> Optional[Any]:
        """Get raw controller data for button state display."""
        if self.device is None:
            return None
        try:
            left, right = self.device.controller()
            return right if right is not None else left
        except Exception:
            return None

    def _call_action(self, action_name: str, *args) -> None:
        """Call an action handler if registered."""
        handler = self._action_handlers.get(action_name)
        if handler:
            handler(*args)

    def process_buttons(self) -> None:
        """Process button state changes and call appropriate action handlers.

        This mirrors _wireless_controller_data_callback() from teleop_stack.
        Button mappings are defined in WIRELESS_BUTTON_MAPPINGS.
        """
        if self.device is None:
            return

        try:
            left, right = self.device.controller()
            c = right if right is not None else left
            if c is None:
                return

            current_time = time.time()

            # Check for keyside (side button) changes
            # Matches the teleop_stack threshold crossing logic
            if c.key_side != self.state.prev_keyside:
                # Check if we crossed above threshold
                if (self.state.prev_keyside <= self._keyside_threshold and
                    c.key_side > self._keyside_threshold):
                    self._call_action("toggle_stream_on")
                    self.state.streaming = True
                # Check if we crossed below threshold
                elif (self.state.prev_keyside > self._keyside_threshold and
                      c.key_side <= self._keyside_threshold):
                    self._call_action("toggle_stream_off")
                    self.state.streaming = False

            # Check for keytrigger (trigger button) changes
            if c.key_trigger != self.state.prev_keytrigger:
                # Check if we crossed above threshold
                if (self.state.prev_keytrigger <= self._keytrigger_threshold and
                    c.key_trigger > self._keytrigger_threshold):
                    self._call_action("toggle_grasping_on")
                    self.state.gripper_closed = True
                # Check if we crossed below threshold
                elif (self.state.prev_keytrigger > self._keytrigger_threshold and
                      c.key_trigger <= self._keytrigger_threshold):
                    self._call_action("toggle_grasping_off")
                    self.state.gripper_closed = False

            # Check for key (A/B button) presses with debounce
            if c.key != self.state.prev_key and c.key != 0:
                # Find the button name that corresponds to this key value
                button_name = None
                for name, value in self._button_values.items():
                    if value == c.key:
                        button_name = name
                        break

                if button_name:
                    # Get the action for this button from mappings
                    action = WIRELESS_BUTTON_MAPPINGS.get("key_press", {}).get(button_name)

                    if action == "reset_robot_pose":
                        # Apply debounce for reset
                        if current_time - self.state.last_reset_time > self._button_press_debounce_delay:
                            self._call_action("reset_robot_pose")
                            self.state.last_reset_time = current_time

                    elif action == "toggle_data_collection":
                        # Apply debounce for data collection toggle
                        if current_time - self.state.last_data_collection_toggle_time > self._button_press_debounce_delay:
                            self._call_action("toggle_data_collection")
                            self.state.data_collection_active = not self.state.data_collection_active
                            self.state.last_data_collection_toggle_time = current_time

            # Update previous state
            self.state.prev_keyside = c.key_side
            self.state.prev_keytrigger = c.key_trigger
            self.state.prev_key = c.key

        except Exception as e:
            print(f"Button processing error: {e}")


# -----------------------------------------------------------------------------
# embodiK Backend (simplified from collision-aware IK)
# -----------------------------------------------------------------------------

@dataclass
class IKResult:
    joints: np.ndarray
    status: str
    position_error: float
    rotation_error: float
    elapsed_ms: float


class TeleopIKBackend:
    """IK backend for teleop with collision avoidance."""

    def __init__(self, cfg: RobotConfig, enable_collision: bool = True):
        self.cfg = cfg
        ensure_ros_package_path(cfg.urdf_path)
        self.robot = embodik.RobotModel(str(cfg.urdf_path), floating_base=False)
        self.solver = embodik.KinematicsSolver(self.robot)
        self.solver.dt = DEFAULT_SOLVER_DT
        self.solver.set_damping(0.1)
        self.solver.set_tolerance(0.1)

        self.arm_dofs = len(cfg.joint_names)
        self.full_dofs = self.robot.nq

        # Apply collision exclusions
        if enable_collision and cfg.collision_exclusions:
            try:
                self.robot.apply_collision_exclusions(cfg.collision_exclusions)
            except Exception as exc:
                print(f"Warning: failed to apply collision exclusions: {exc}")

        # Initialize configuration
        self.default_arm = cfg.default_configuration.copy()
        if len(self.default_arm) < self.arm_dofs:
            padding = np.zeros(self.arm_dofs - len(self.default_arm), dtype=float)
            self.default_arm = np.concatenate([self.default_arm, padding])
        elif len(self.default_arm) > self.arm_dofs:
            self.default_arm = self.default_arm[:self.arm_dofs]

        self.default_full = np.zeros(self.full_dofs, dtype=float)
        self.default_full[:self.arm_dofs] = self.default_arm
        self.q = self.default_full.copy()
        self.robot.update_configuration(self.q)

        # Joint limits
        lower, upper = self.robot.get_joint_limits()
        self.lower = lower.astype(float)
        self.upper = upper.astype(float)

        # Initial pose
        self.initial_pose = self.get_pose()

        # Frame task for IK
        self._zero_velocity = np.zeros(6, dtype=float)
        self.frame_task = self.solver.add_frame_task("ee_task", self.cfg.target_link)
        self.frame_task.priority = 0
        self.frame_task.weight = 0.0
        self.frame_task.set_target_velocity(self._zero_velocity)

        # Nullspace task
        self.nullspace_task = self.solver.add_posture_task("posture_task")
        self.nullspace_task.priority = 1
        self.nullspace_task.weight = 0.0
        self.nullspace_task.set_target_configuration(self.q.copy())
        self.nullspace_task.set_controlled_joint_indices([])

    def get_pose(self) -> pin.SE3:
        """Get current end-effector pose."""
        return self.robot.get_frame_pose(self.cfg.target_link)

    def get_q(self) -> np.ndarray:
        """Get current arm joint configuration."""
        return self.q[:self.arm_dofs].copy()

    def set_q(self, q_arm: np.ndarray) -> None:
        """Set arm joint configuration."""
        self.q[:self.arm_dofs] = np.clip(q_arm, self.lower[:self.arm_dofs], self.upper[:self.arm_dofs])
        self.robot.update_configuration(self.q)

    def solve_step(
        self,
        target: pin.SE3,
        pos_gain: float = DEFAULT_POS_GAIN,
        rot_gain: float = DEFAULT_ROT_GAIN,
        nullspace_gain: float = DEFAULT_NULLSPACE_GAIN,
    ) -> IKResult:
        """Solve one IK step toward target pose."""
        current = self.get_pose()
        pose_error = compute_pose_error(current, target)

        target_velocity = np.concatenate([pos_gain * pose_error[:3], rot_gain * pose_error[3:]])
        target_velocity = limit_task_velocity(target_velocity, MAX_LINEAR_STEP, MAX_ANGULAR_STEP)

        position_error = float(np.linalg.norm(pose_error[:3]))
        rotation_error = float(np.linalg.norm(pose_error[3:]))

        if position_error < 1e-4 and rotation_error < 1e-3:
            self.frame_task.weight = 0.0
            self.frame_task.set_target_velocity(self._zero_velocity)
        else:
            self.frame_task.weight = 1.0
            self.frame_task.set_target_velocity(target_velocity)

        # Nullspace bias toward default
        self.nullspace_task.set_controlled_joint_indices(list(range(self.arm_dofs)))
        self.nullspace_task.set_target_configuration(self.default_full)
        self.nullspace_task.weight = nullspace_gain

        ik_start = time.perf_counter()
        result = self.solver.solve_velocity(self.q, apply_limits=True)
        elapsed_ms = (time.perf_counter() - ik_start) * 1000.0

        if result.status == embodik.SolverStatus.SUCCESS:
            dq = np.array(result.joint_velocities) * self.solver.dt
            self.q = np.clip(self.q + dq, self.lower, self.upper)
            self.robot.update_configuration(self.q)

        updated_pose = self.get_pose()
        final_error = compute_pose_error(updated_pose, target)

        return IKResult(
            joints=self.get_q(),
            status=result.status.name,
            position_error=float(np.linalg.norm(final_error[:3])),
            rotation_error=float(np.linalg.norm(final_error[3:])),
            elapsed_ms=elapsed_ms,
        )

    def reset(self) -> pin.SE3:
        """Reset to default configuration."""
        self.q = self.default_full.copy()
        self.robot.update_configuration(self.q)
        return self.get_pose()


# -----------------------------------------------------------------------------
# Main Teleop Loop
# -----------------------------------------------------------------------------

def run_teleop(cfg: RobotConfig, args: argparse.Namespace) -> None:
    """Run the teleop loop.

    This follows the teleop_stack pattern with:
    - process_input(): Process controller input and update goal pose
    - send_robot_command(): Compute IK and send commands
    """

    # Initialize IK backend
    backend = TeleopIKBackend(cfg, enable_collision=not args.no_collision)
    print(f"Loaded robot: {cfg.display_name}")
    print(f"Target link: {cfg.target_link}")
    print(f"DOFs: {backend.arm_dofs}")

    # Initialize controller
    controller = ControllerInputHandler(port=args.controller_port)
    controller_connected = controller.connect()

    if not controller_connected:
        print("\nNo controller connected. Running in GUI-only mode.")
        print("Use the transform controls in the browser to move the target.")

    # Set up visualization
    urdf = load_robot_description(cfg.description_name)
    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # Joint name mapping for visualization
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

    # Get initial pose
    pose = backend.get_pose()
    initial_quat_xyzw = r2q(pose.rotation, order="xyzs")
    initial_wxyz = (initial_quat_xyzw[3], initial_quat_xyzw[0], initial_quat_xyzw[1], initial_quat_xyzw[2])

    # Current target (starts at initial EE pose)
    target_position = pose.translation.copy()
    target_rotation = pose.rotation.copy()

    # Add transform controls for visualization/manual control
    ik_target = server.scene.add_transform_controls(
        "/ik_target",
        scale=0.2,
        position=tuple(target_position),
        wxyz=initial_wxyz,
    )

    # GUI elements
    with server.gui.add_folder("Teleop Status"):
        streaming_text = server.gui.add_text("Streaming", initial_value="OFF")
        gripper_text = server.gui.add_text("Gripper", initial_value="OPEN")
        data_collection_text = server.gui.add_text("Data Collection", initial_value="OFF")
        status_text = server.gui.add_text("Status", initial_value="Ready")
        timing_handle = server.gui.add_number("IK Time (ms)", 0.001, disabled=True)

    with server.gui.add_folder("Teleop Controls"):
        scale_slider = server.gui.add_slider("Position Scale", min=0.5, max=3.0, initial_value=args.scale, step=0.1)
        pos_gain = server.gui.add_slider("Position Gain", min=10.0, max=200.0, initial_value=DEFAULT_POS_GAIN, step=10.0)
        rot_gain = server.gui.add_slider("Rotation Gain", min=10.0, max=200.0, initial_value=DEFAULT_ROT_GAIN, step=10.0)
        manual_mode = server.gui.add_checkbox("Manual Mode (use transform controls)", initial_value=not controller_connected)
        reset_button = server.gui.add_button("Reset Robot & Controller")

    with server.gui.add_folder("Controller Info", expand_by_default=False):
        controller_pos_text = server.gui.add_text("Position", initial_value="(0.000, 0.000, 0.000)")
        controller_buttons_text = server.gui.add_text("Buttons", initial_value="trigger=0 side=0 key=0")

    # Teleop state (matching teleop_stack patterns)
    # arm_init_pose: The arm pose when streaming started (updated on reset_pose_offset)
    # pose_offset: The controller pose when streaming started (for relative calculation)
    arm_init_pose = backend.get_pose()
    goal_pose = arm_init_pose  # Current goal pose

    # Action handlers (matching teleop_stack button behavior)
    def on_toggle_stream_on():
        """Handle streaming on - matches _handle_button_toggle_stream(True)."""
        streaming_text.value = "ON"
        print("Streaming: ON")
        # Reset pose offset when streaming is enabled (like teleop_stack)
        nonlocal arm_init_pose
        if controller._initialized:
            controller.reset_reference()
        arm_init_pose = backend.get_pose()

    def on_toggle_stream_off():
        """Handle streaming off - matches _handle_button_toggle_stream(False)."""
        streaming_text.value = "OFF"
        print("Streaming: OFF")

    def on_toggle_grasping_on():
        """Handle gripper close - matches _handle_button_toggle_grasping(True)."""
        gripper_text.value = "CLOSED"
        print("Gripper: CLOSED")
        # In a real implementation, this would send gripper close command

    def on_toggle_grasping_off():
        """Handle gripper open - matches _handle_button_toggle_grasping(False)."""
        gripper_text.value = "OPEN"
        print("Gripper: OPEN")
        # In a real implementation, this would send gripper open command

    def on_reset_robot_pose():
        """Handle reset - matches _handle_button_reset_robot_pose."""
        nonlocal arm_init_pose, goal_pose
        backend.reset()
        if controller._initialized:
            controller.reset_reference()
        arm_init_pose = backend.get_pose()
        goal_pose = arm_init_pose
        # Update target visualization
        pose_reset = backend.get_pose()
        quat_reset = r2q(pose_reset.rotation, order="xyzs")
        ik_target.position = tuple(pose_reset.translation)
        ik_target.wxyz = (quat_reset[3], quat_reset[0], quat_reset[1], quat_reset[2])
        status_text.value = "Robot reset to default"
        print("Reset: Robot and controller reference reset")

    def on_toggle_data_collection():
        """Handle data collection toggle - matches _handle_button_toggle_data_collection."""
        if controller.state.data_collection_active:
            data_collection_text.value = "ON"
            print("Data Collection: ON")
        else:
            data_collection_text.value = "OFF"
            print("Data Collection: OFF")

    # Register action handlers with controller
    controller.set_action_handlers({
        "toggle_stream_on": on_toggle_stream_on,
        "toggle_stream_off": on_toggle_stream_off,
        "toggle_grasping_on": on_toggle_grasping_on,
        "toggle_grasping_off": on_toggle_grasping_off,
        "reset_robot_pose": on_reset_robot_pose,
        "toggle_data_collection": on_toggle_data_collection,
    })

    @reset_button.on_click
    def _(_):
        on_reset_robot_pose()

    # Update visualization
    q_current = backend.get_q()
    urdf_vis.update_cfg(make_visual_config(q_current))

    # Display button mappings (like _display_button_mappings in teleop_stack)
    print("\n" + "=" * 60)
    print("TELEOP IK DEMO")
    print("=" * 60)
    print(f"Robot: {cfg.display_name}")
    print(f"URDF: {cfg.urdf_path}")
    print("=" * 60)

    if controller_connected:
        print("\nSEER WIRELESS CONTROLLER BUTTON MAPPINGS")
        print("-" * 40)
        print("  Side Button (Hold)  : Start/Stop Streaming")
        print("  Trigger (Hold)      : Close/Open Gripper")
        print("  Button A            : Reset Robot Pose")
        print("  Button B            : Toggle Data Collection")
        print("-" * 40)
    else:
        print("\nNo controller - use transform controls in browser")

    print(f"\nOpen http://localhost:{args.port} in your browser")
    print("Press Ctrl+C to exit")
    print("=" * 60 + "\n")

    frame_count = 0
    try:
        while True:
            # --- Process Input (matches process_input() pattern) ---

            # Process controller buttons
            if controller._initialized:
                controller.process_buttons()

            # Update goal pose based on controller input
            if manual_mode.value or not controller._initialized:
                # Manual mode: use transform controls directly
                target_position = np.array(ik_target.position, dtype=float)
                target_wxyz = np.array(ik_target.wxyz, dtype=float)
                target_xyzw = np.array([target_wxyz[1], target_wxyz[2], target_wxyz[3], target_wxyz[0]])
                target_rotation = q2r(target_xyzw, order="xyzs")
                goal_pose = Rt(R=target_rotation, t=target_position)

            elif controller.state.streaming:
                # Controller mode: compute goal pose from relative controller movement
                # This matches the process_input() logic from teleop_stack:
                #   pose_change = SE3.Rt(
                #       t=scale * pose_offset.R.T @ (pose.t - pose_offset.t),
                #       R=pose_offset.R.T @ pose.R,
                #   )
                #   goal_pose = SE3.Rt(
                #       t=arm_init_pose.t + pose_change.t,
                #       R=pose_change.R @ arm_init_pose.R,
                #   )

                pose_data = controller.get_relative_pose()
                if pose_data is not None:
                    delta_pos, delta_quat_wxyz = pose_data

                    # Convert quaternion to rotation matrix
                    delta_quat_xyzw = np.array([delta_quat_wxyz[1], delta_quat_wxyz[2], delta_quat_wxyz[3], delta_quat_wxyz[0]])
                    delta_R = q2r(delta_quat_xyzw, order="xyzs")

                    # Apply scale to position (like scale_factor_position_change in teleop_stack)
                    scaled_pos = delta_pos * scale_slider.value

                    # Compute goal pose (matching teleop_stack formula)
                    # goal_pose.t = arm_init_pose.t + pose_change.t
                    # goal_pose.R = pose_change.R @ arm_init_pose.R
                    target_position = arm_init_pose.translation + scaled_pos
                    target_rotation = delta_R @ arm_init_pose.rotation
                    goal_pose = Rt(R=target_rotation, t=target_position)

                    # Update transform controls to show current target
                    target_quat_xyzw = r2q(target_rotation, order="xyzs")
                    ik_target.position = tuple(target_position)
                    ik_target.wxyz = (target_quat_xyzw[3], target_quat_xyzw[0], target_quat_xyzw[1], target_quat_xyzw[2])

                    # Update controller info display
                    controller_pos_text.value = f"({delta_pos[0]:+.3f}, {delta_pos[1]:+.3f}, {delta_pos[2]:+.3f})"

                    # Get raw button data for display
                    c = controller.get_raw_controller_data()
                    if c is not None:
                        controller_buttons_text.value = f"trigger={c.key_trigger} side={c.key_side} key={c.key}"

            # --- Send Robot Command (matches send_robot_command() pattern) ---

            # Only send commands when streaming (like teleop_stack)
            if controller.state.streaming or manual_mode.value:
                # Solve IK toward goal pose
                result = backend.solve_step(
                    goal_pose,
                    pos_gain=pos_gain.value,
                    rot_gain=rot_gain.value,
                )

                # Update visualization
                q_current = result.joints
                urdf_vis.update_cfg(make_visual_config(q_current))

                # Update timing
                timing_handle.value = 0.9 * timing_handle.value + 0.1 * result.elapsed_ms

                # Update status periodically
                if frame_count % 100 == 0:
                    status_text.value = f"IK: {result.status} | pos_err={result.position_error*1e3:.1f}mm"

            frame_count += 1
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n\nTeleop stopped by user")
    finally:
        if controller._initialized:
            controller.disconnect()
            print("Controller disconnected")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teleop IK with Seer controller")
    parser.add_argument(
        "--robot",
        choices=sorted(ROBOT_PRESETS.keys()),
        default="panda",
        help="Robot model to load (default: panda)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Viser server port (default: 8080)",
    )
    parser.add_argument(
        "--controller-port",
        type=str,
        default="/dev/ttyUSB0",
        help="Controller serial port (default: /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE_FACTOR,
        help=f"Position scale factor (default: {DEFAULT_SCALE_FACTOR})",
    )
    parser.add_argument(
        "--no-collision",
        action="store_true",
        help="Disable collision avoidance",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = resolve_robot_configuration(args.robot)
    print(f"Teleop IK Demo ({cfg.display_name})")
    print(f"  - URDF path: {cfg.urdf_path}")
    print(f"  - Target link: {cfg.target_link}")

    run_teleop(cfg, args)


if __name__ == "__main__":
    main()
