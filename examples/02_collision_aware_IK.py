#!/usr/bin/env python3
"""Interactive collision-aware IK using embodiK and Viser."""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pinocchio as pin
import viser
from viser.extras import ViserUrdf

import embodik
from robot_descriptions.loaders.yourdfpy import load_robot_description
from embodik.utils import compute_pose_error, limit_task_velocity
from embodik import r2q, q2r, Rt
from utils.robot_models import load_robot_presets

# -----------------------------------------------------------------------------
# Default numeric constants
# -----------------------------------------------------------------------------

DEFAULT_SOLVER_DT = 0.01
DEFAULT_POS_GAIN = 1e2
DEFAULT_ROT_GAIN = 1e2
DEFAULT_NULLSPACE_GAIN = 1e-3
MAX_LINEAR_STEP = 2.0
MAX_ANGULAR_STEP = 2.0
DEFAULT_COLLISION_GAIN = 1.0

_LINK_INDEX_PATTERN = re.compile(r"link_?([0-9]+)")


def _extract_link_index(name: str) -> Optional[int]:
    match = _LINK_INDEX_PATTERN.search(name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:  # pragma: no cover
            return None
    return None


def _should_auto_exclude_pair(name_a: str, name_b: str, robot_key: str) -> bool:
    a_lower = name_a.lower()
    b_lower = name_b.lower()

    end_effector_tokens = ("finger", "hand")
    a_is_ee = any(token in a_lower for token in end_effector_tokens)
    b_is_ee = any(token in b_lower for token in end_effector_tokens)

    if a_is_ee and b_is_ee:
        return True

    idx_a = _extract_link_index(a_lower)
    idx_b = _extract_link_index(b_lower)

    if a_is_ee != b_is_ee:
        other_idx = idx_b if a_is_ee else idx_a
        if other_idx is not None and robot_key == "panda" and other_idx >= 5:
            return True
        if other_idx is not None and robot_key == "iiwa":
            return False
        # If we cannot determine the index (e.g., the hand entry itself), keep the pair
        return False

    if idx_a is None or idx_b is None:
        return False

    gap = abs(idx_a - idx_b)
    if robot_key == "panda":
        return gap <= 2
    if robot_key == "iiwa":
        return gap <= 3
    return gap <= 1


def generate_auto_collision_exclusions(robot: embodik.RobotModel, robot_key: str) -> List[Tuple[str, str]]:
    exclusions: List[Tuple[str, str]] = []
    for name_a, name_b in robot.get_collision_pair_names():
        if _should_auto_exclude_pair(name_a, name_b, robot_key):
            exclusions.append((name_a, name_b))
    return exclusions


# -----------------------------------------------------------------------------
# Robot presets shared with the other demos.
# Loaded from robot_presets.yaml to keep configurations in sync.
# -----------------------------------------------------------------------------

# Load presets from YAML file (shared with example 01)
ROBOT_PRESETS: Dict[str, Dict[str, object]] = load_robot_presets()


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


def resolve_robot_configuration(robot_key: str) -> RobotConfig:
    """Resolve robot configuration from presets and return RobotConfig dataclass.

    Supports both local URDF files (urdf_path) and robot_descriptions package (urdf_import + urdf_attr).
    """
    robot_key = robot_key.lower()
    if robot_key not in ROBOT_PRESETS:
        raise ValueError(f"Unsupported robot '{robot_key}'. Available options: {sorted(ROBOT_PRESETS)}")

    preset = ROBOT_PRESETS[robot_key]

    # Resolve URDF path - support both local files and robot_descriptions
    urdf_path = None

    # Priority 1: Check for local urdf_path (for backward compatibility and custom models)
    urdf_path_str = preset.get("urdf_path")
    if urdf_path_str:
        examples_dir = Path(__file__).parent
        urdf_path = examples_dir / urdf_path_str
        if not urdf_path.exists():
            raise FileNotFoundError(
                f"URDF file not found: {urdf_path}\n"
                f"Expected at: {urdf_path_str} (relative to examples/ directory)"
            )

    # Priority 2: Use robot_descriptions package
    if urdf_path is None:
        urdf_import = preset.get("urdf_import")
        urdf_attr = preset.get("urdf_attr", "URDF_PATH")

        if not urdf_import:
            raise ValueError(
                f"Robot preset '{robot_key}' must specify either 'urdf_path' (local file) "
                f"or 'urdf_import' (robot_descriptions package) in robot_presets.yaml"
            )

        try:
            module = __import__(urdf_import, fromlist=[urdf_attr])
            urdf_path = Path(getattr(module, urdf_attr))  # type: ignore[arg-type]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                f"Robot description package '{urdf_import}' is required for the '{robot_key}' model. "
                "Install the 'robot_descriptions' package to use this example:\n"
                "  pip install robot_descriptions\n"
                "Or install with examples dependencies:\n"
                "  pip install embodik[examples]"
            ) from exc
        except AttributeError as exc:
            raise ValueError(
                f"Robot description module '{urdf_import}' does not have attribute '{urdf_attr}'."
            ) from exc

        if not urdf_path.exists():
            raise FileNotFoundError(
                f"URDF file from robot_descriptions not found: {urdf_path}\n"
                f"This may indicate a caching issue. Try clearing ~/.cache/robot_descriptions/"
            )

    # Handle collision exclusions
    raw_exclusions = preset.get("collision_exclusions", [])
    auto_collision = False
    if raw_exclusions == "auto":
        auto_collision = True
        overrides = [tuple(pair) for pair in preset.get("collision_exclusion_overrides", [])]
        ensure_ros_package_path(urdf_path)
        temp_robot = embodik.RobotModel(str(urdf_path), floating_base=False)
        auto_list = generate_auto_collision_exclusions(temp_robot, robot_key)
        collision_exclusions = auto_list + overrides
    else:
        collision_exclusions = [tuple(pair) for pair in raw_exclusions]  # type: ignore[arg-type]
        ensure_ros_package_path(urdf_path)

    # Get joint names and labels (with fallback to auto-generation)
    joint_names = preset.get("joint_names", [])
    joint_labels = preset.get("joint_labels", [])

    # If not specified, try to extract from robot model
    if not joint_names or not joint_labels:
        ensure_ros_package_path(urdf_path)
        temp_robot = embodik.RobotModel(str(urdf_path), floating_base=False)
        if not joint_names:
            # Get all joint names, but for robots with grippers, we only want arm joints
            all_joint_names = temp_robot.get_joint_names()
            # For panda, default_configuration has 7 values (arm only), so use first 7 joints
            default_config = preset.get("default_configuration", [])
            if robot_key == "panda" and len(default_config) == 7 and len(all_joint_names) > 7:
                # Use only arm joints (first 7), exclude gripper joints
                joint_names = all_joint_names[:7]
            else:
                joint_names = all_joint_names
        if not joint_labels:
            # Auto-generate labels from joint names
            from utils.robot_models import generate_joint_labels_from_names
            joint_labels = generate_joint_labels_from_names(joint_names, robot_key)

    # Handle default configuration - ensure it matches the number of arm joints
    default_config = np.array(preset.get("default_configuration", []), dtype=float)

    # For panda with gripper, default_configuration should only have arm joints (7)
    # Handle gripper joints separately if robot has more DOF than default_configuration
    if robot_key == "panda" and len(default_config) == 7:
        # Ensure we have exactly 7 values for arm joints
        if len(joint_names) > len(default_config):
            # Robot has more joints than default_configuration - this is expected for panda with gripper
            # default_configuration should only contain arm joints
            pass
        elif len(joint_names) != len(default_config):
            # Mismatch - pad or truncate to match joint_names length
            if len(joint_names) > len(default_config):
                # Pad with zeros (for gripper joints)
                extra_gripper = preset.get("extra_gripper_default", np.array([0.05, 0.05]))
                if isinstance(extra_gripper, list):
                    extra_gripper = np.array(extra_gripper)
                default_config = np.concatenate([default_config, extra_gripper])
            else:
                # Truncate to match
                default_config = default_config[:len(joint_names)]

    return RobotConfig(
        key=robot_key,
        display_name=preset.get("display_name", robot_key),  # type: ignore[arg-type]
        urdf_path=urdf_path,
        description_name=preset.get("description_name", ""),  # type: ignore[arg-type]
        target_link=preset.get("target_link", "end_effector"),  # type: ignore[arg-type]
        joint_labels=list(joint_labels) if isinstance(joint_labels, (list, tuple)) else [],  # type: ignore[arg-type]
        joint_names=list(joint_names) if isinstance(joint_names, (list, tuple)) else [],  # type: ignore[arg-type]
        default_configuration=default_config,
        default_offset=np.array(preset.get("default_offset", [0.05, 0.0, 0.0]), dtype=float),
        collision_exclusions=collision_exclusions,
    )


def ensure_ros_package_path(urdf_path: Path) -> None:
    """Ensure ROS_PACKAGE_PATH includes ancestors that contain meshes."""

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
# embodiK backend
# -----------------------------------------------------------------------------


@dataclass
class embodiKResult:
    joints: np.ndarray
    status: str
    position_error: float
    rotation_error: float
    elapsed_ms: float


class embodiKBackend:
    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        ensure_ros_package_path(cfg.urdf_path)
        self.robot = embodik.RobotModel(str(cfg.urdf_path), floating_base=False)
        self.solver = embodik.KinematicsSolver(self.robot)
        self.solver.dt = DEFAULT_SOLVER_DT
        self.solver.set_damping(0.1)
        self.solver.set_tolerance(0.1)

        self.arm_dofs = len(cfg.joint_names)
        self.full_dofs = self.robot.nq

        exclusions: List[Tuple[str, str]] = list(cfg.collision_exclusions)
        self._collision_exclusions = exclusions
        if self._collision_exclusions:
            try:
                self.robot.apply_collision_exclusions(self._collision_exclusions)
            except Exception as exc:  # pragma: no cover
                print(f"[embodiK] Warning: failed to apply collision exclusions: {exc}")

        self.default_arm = cfg.default_configuration.copy()
        # Ensure default_arm matches arm_dofs
        if len(self.default_arm) != self.arm_dofs:
            if len(self.default_arm) < self.arm_dofs:
                # Pad with zeros if needed
                padding = np.zeros(self.arm_dofs - len(self.default_arm), dtype=float)
                self.default_arm = np.concatenate([self.default_arm, padding])
            else:
                # Truncate if needed
                self.default_arm = self.default_arm[:self.arm_dofs]

        self.default_full = np.zeros(self.full_dofs, dtype=float)
        self.default_full[: self.arm_dofs] = self.default_arm
        self.q = self.default_full.copy()
        self.robot.update_configuration(self.q)

        lower, upper = self.robot.get_joint_limits()
        self.lower = lower.astype(float)
        self.upper = upper.astype(float)
        self.initial_pose = self.get_pose()

        self._zero_velocity = np.zeros(6, dtype=float)
        self.frame_task = self.solver.add_frame_task("ee_task", self.cfg.target_link)
        self.frame_task.priority = 0
        self.frame_task.weight = 0.0
        self.frame_task.set_target_velocity(self._zero_velocity)

        self.nullspace_task = self.solver.add_posture_task("posture_task")
        self.nullspace_task.priority = 1
        self.nullspace_task.weight = 0.0
        self.nullspace_task.set_target_configuration(self.q.copy())
        self.nullspace_task.set_controlled_joint_indices([])

    def get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self.lower[: self.arm_dofs], self.upper[: self.arm_dofs]

    def get_q(self) -> np.ndarray:
        return self.q[: self.arm_dofs].copy()

    def set_q(self, q_arm: np.ndarray) -> None:
        self.q[: self.arm_dofs] = np.clip(q_arm, self.lower[: self.arm_dofs], self.upper[: self.arm_dofs])
        self.robot.update_configuration(self.q)

    def get_pose(self) -> pin.SE3:
        pose = self.robot.get_frame_pose(self.cfg.target_link)
        return pose  # Already SE3, no need to wrap

    def solve_step(
        self,
        target: pin.SE3,
        pos_gain: float,
        rot_gain: float,
        active_indices: List[int],
        nullspace_bias: np.ndarray,
        nullspace_gain: float,
        nullspace_enabled: bool,
    ) -> embodiKResult:
        self.frame_task.weight = 0.0
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

        if nullspace_enabled and active_indices and nullspace_gain > 0.0:
            bias_full = self.q.copy()
            for idx in active_indices:
                bias_full[idx] = nullspace_bias[idx]
            self.nullspace_task.set_controlled_joint_indices(active_indices)
            self.nullspace_task.set_target_configuration(bias_full)
            self.nullspace_task.weight = nullspace_gain
        else:
            self.nullspace_task.set_controlled_joint_indices([])
            self.nullspace_task.weight = 0.0

        ik_start = time.perf_counter()
        result = self.solver.solve_velocity(self.q, apply_limits=True)
        elapsed_ms = (time.perf_counter() - ik_start) * 1000.0
        if result.status == embodik.SolverStatus.SUCCESS:
            dq = np.array(result.joint_velocities) * self.solver.dt
            self.q = np.clip(self.q + dq, self.lower, self.upper)
            self.robot.update_configuration(self.q)

        updated_pose = self.get_pose()
        final_error = compute_pose_error(updated_pose, target)

        return embodiKResult(
            joints=self.get_q(),
            status=result.status.name,
            position_error=float(np.linalg.norm(final_error[:3])),
            rotation_error=float(np.linalg.norm(final_error[3:])),
            elapsed_ms=elapsed_ms,
        )

    def reset(self) -> pin.SE3:
        self.q = self.default_full.copy()
        self.robot.update_configuration(self.q)
        return self.get_pose()

    def enable_self_collision(self, enable: bool) -> None:
        if not hasattr(self, "_collision_enabled"):
            self._collision_enabled = False

        if enable and not self._collision_enabled:
            try:
                self.solver.configure_collision_constraint(
                    min_distance=0.05,
                    include_pairs=[],
                    exclude_pairs=list(self._collision_exclusions),
                )
                self._collision_enabled = True
            except RuntimeError as exc:
                print(f"[embodiK] Collision configuration failed: {exc}")
                self._collision_enabled = False
        elif not enable and self._collision_enabled:
            self.solver.clear_collision_constraint()
            self._collision_enabled = False


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Interactive GUI
# -----------------------------------------------------------------------------


def run_gui(cfg: RobotConfig) -> None:
    backend = embodiKBackend(cfg)

    if hasattr(backend, "robot"):
        try:
            obj_names = backend.robot.get_collision_geometry_names()
            pair_names = backend.robot.get_collision_pair_names()
            print(f"[embodiK] Loaded {len(obj_names)} collision geometries for {cfg.display_name}")
            print(f"[embodiK] Loaded {len(pair_names)} collision pairs")
            if obj_names:
                print("  sample objects:", ", ".join(obj_names[:8]))
            if pair_names:
                formatted = [f"{a}|{b}" for a, b in pair_names[:8]]
                print("  sample pairs:", ", ".join(formatted))
        except Exception as exc:  # pragma: no cover
            print(f"[embodiK] Unable to list collision data: {exc}")

    def default_bias_for_backend(b: object) -> np.ndarray:
        return b.default_arm.copy() if hasattr(b, "default_arm") else b.get_q().copy()  # type: ignore[attr-defined]

    q_current = backend.get_q()
    nullspace_bias = default_bias_for_backend(backend)

    urdf = load_robot_description(cfg.description_name)
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
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
            else:
                cfg_vec[i] = 0.0
        return cfg_vec

    pose = backend.get_pose()
    initial_quat_xyzw = r2q(pose.rotation, order="xyzs")
    initial_wxyz = (initial_quat_xyzw[3], initial_quat_xyzw[0], initial_quat_xyzw[1], initial_quat_xyzw[2])

    ik_target = server.scene.add_transform_controls(
        "/ik_target",
        scale=0.2,
        position=tuple(pose.translation),
        wxyz=initial_wxyz,
    )

    with server.gui.add_folder("IK Controls"):
        timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
        pos_gain = server.gui.add_slider("Position Gain", min=5.0, max=1e3, initial_value=DEFAULT_POS_GAIN, step=10.0)
        rot_gain = server.gui.add_slider("Rotation Gain", min=5.0, max=1e3, initial_value=DEFAULT_ROT_GAIN, step=10.0)
        nullspace_enabled_checkbox = server.gui.add_checkbox("Enable Nullspace Bias", initial_value=True)
        nullspace_gain = server.gui.add_slider("Nullspace Gain", min=0.0, max=2.0, initial_value=DEFAULT_NULLSPACE_GAIN, step=0.05)
        self_collision_checkbox = server.gui.add_checkbox(
            "Enable Self-Collision",
            initial_value=False,
            disabled=not hasattr(backend, "enable_self_collision"),
        )
        collision_debug_checkbox = server.gui.add_checkbox(
            "Show Collision Debug",
            initial_value=False,
            disabled=not (
                hasattr(backend, "solver")
                and hasattr(backend.solver, "get_last_collision_debug")
            ),
        )
        collision_debug_text = server.gui.add_text("Collision Debug", initial_value="Collision: --")
        manual_control = server.gui.add_checkbox("Manual Joint Control", initial_value=False)
        status_handle = server.gui.add_text("Status", initial_value="Status: Ready")
        snap_target_button = server.gui.add_button("Snap Target to Current EE")
        reset_robot_button = server.gui.add_button("Reset Robot & Target")

    joint_sliders: List[viser.GuiSliderHandle] = []
    with server.gui.add_folder("Joint Configuration", expand_by_default=False):
        lower, upper = backend.get_joint_limits()
        for idx, (label, lo, hi) in enumerate(zip(cfg.joint_labels, lower, upper)):
            slider = server.gui.add_slider(
                f"{label} (joint{idx + 1})",
                min=float(lo),
                max=float(hi),
                step=0.01,
                initial_value=float(q_current[idx]),
            )
            joint_sliders.append(slider)

    nullspace_checkboxes: List[viser.GuiCheckboxHandle] = []
    with server.gui.add_folder("Nullspace Joint Selection", expand_by_default=False):
        for idx, label in enumerate(cfg.joint_labels):
            checkbox = server.gui.add_checkbox(
                f"{label} (joint{idx + 1})",
                initial_value=True,
            )
            nullspace_checkboxes.append(checkbox)

    bias_to_initial = server.gui.add_button("Bias → Initial Configuration")
    bias_to_zero = server.gui.add_button("Bias → Zero Configuration")

    target_xyzw = np.zeros(4, dtype=float)

    collision_root = "/collision_debug"
    collision_point_a = server.scene.add_icosphere(
        f"{collision_root}/point_a",
        radius=0.015,
        color=(1.0, 0.2, 0.2),
        visible=False,
    )
    collision_point_b = server.scene.add_icosphere(
        f"{collision_root}/point_b",
        radius=0.015,
        color=(0.2, 0.8, 0.2),
        visible=False,
    )
    collision_line_handle = None
    last_collision_debug = None
    collision_log_timestamp = 0.0
    collision_state = "init"
    collision_debug_enabled = False

    urdf_vis.update_cfg(make_visual_config(q_current))

    def update_collision_visuals() -> None:
        nonlocal collision_line_handle, last_collision_debug, collision_log_timestamp, collision_state
        nonlocal collision_debug_enabled
        solver_obj = getattr(backend, "solver", None)
        now = time.time()

        debug_requested = (
            collision_debug_checkbox.value
            and self_collision_checkbox.value
            and solver_obj is not None
            and hasattr(solver_obj, "get_last_collision_debug")
        )

        if not debug_requested:
            if collision_debug_enabled:
                collision_point_a.visible = False
                collision_point_b.visible = False
                if collision_line_handle is not None:
                    collision_line_handle.visible = False
                collision_debug_text.value = "Collision: --"
                collision_state = "hidden"
                collision_debug_enabled = False
            return

        if solver_obj is None or not hasattr(solver_obj, "get_last_collision_debug"):
            collision_point_a.visible = False
            collision_point_b.visible = False
            if collision_line_handle is not None:
                collision_line_handle.visible = False
            collision_debug_text.value = "Collision: unsupported"
            if collision_state != "unsupported" or now - collision_log_timestamp > 1.0:
                print("[embodiK] Collision debug unavailable for current backend.")
                collision_log_timestamp = now
            last_collision_debug = None
            collision_state = "unsupported"
            collision_debug_enabled = True
            return

        debug_info = solver_obj.get_last_collision_debug()
        if debug_info is None:
            collision_point_a.visible = False
            collision_point_b.visible = False
            if collision_line_handle is not None:
                collision_line_handle.visible = False
            collision_debug_text.value = "Collision: --"
            if collision_state != "none" or now - collision_log_timestamp > 1.0:
                print("[embodiK] Collision debug: no active collision pairs.")
                collision_log_timestamp = now
            last_collision_debug = None
            collision_state = "none"
            collision_debug_enabled = True
            return

        point_a = np.array(debug_info.point_a_world, dtype=float)
        point_b = np.array(debug_info.point_b_world, dtype=float)

        collision_point_a.position = tuple(point_a)
        collision_point_b.position = tuple(point_b)
        collision_point_a.visible = True
        collision_point_b.visible = True

        if collision_line_handle is not None:
            collision_line_handle.remove()
        seg_points = np.zeros((1, 2, 3), dtype=float)
        seg_points[0, 0, :] = point_a
        seg_points[0, 1, :] = point_b
        colors = np.array([[[1.0, 0.2, 0.2], [0.2, 0.8, 0.2]]], dtype=float)
        collision_line_handle = server.scene.add_line_segments(
            f"{collision_root}/segment",
            points=seg_points,
            colors=colors,
            line_width=3.0,
            visible=True,
        )

        collision_debug_text.value = (
            f"Collision: {debug_info.object_a} ↔ {debug_info.object_b} | "
            f"d = {debug_info.distance:.3f} m"
        )
        if (
            last_collision_debug is None
            or debug_info.object_a != last_collision_debug.object_a
            or debug_info.object_b != last_collision_debug.object_b
            or abs(debug_info.distance - last_collision_debug.distance) > 1e-4
            or now - collision_log_timestamp > 1.0
        ):
            print(
                "[embodiK] Collision pair:",
                debug_info.object_a,
                "<->",
                debug_info.object_b,
                "| distance =",
                f"{debug_info.distance:.4f} m",
            )
            collision_log_timestamp = now
        collision_state = "active"
        last_collision_debug = debug_info
        collision_debug_enabled = True

    def sync_from_backend(update_target: bool = True) -> None:
        nonlocal q_current, nullspace_bias, target_xyzw, collision_line_handle
        lower, upper = backend.get_joint_limits()
        q_current = backend.get_q()
        nullspace_bias = default_bias_for_backend(backend)
        for slider, lo, hi, value in zip(joint_sliders, lower, upper, q_current):
            slider.min = float(lo)
            slider.max = float(hi)
            slider.value = float(value)
        if update_target:
            pose_local = backend.get_pose()
            quat_local = r2q(pose_local.rotation, order="xyzs")
            target_xyzw = np.array([quat_local[0], quat_local[1], quat_local[2], quat_local[3]])
            ik_target.position = tuple(pose_local.translation)
            ik_target.wxyz = (quat_local[3], quat_local[0], quat_local[1], quat_local[2])
        urdf_vis.update_cfg(make_visual_config(q_current))
        if hasattr(backend, "enable_self_collision"):
            backend.enable_self_collision(self_collision_checkbox.value and not self_collision_checkbox.disabled)
            solver_obj = getattr(backend, "solver", None)
            if (
                collision_debug_checkbox.value
                and self_collision_checkbox.value
                and solver_obj is not None
                and hasattr(solver_obj, "get_active_collision_pairs")
            ):
                active_pairs = solver_obj.get_active_collision_pairs()
                print(f"[embodiK] Active collision pairs: {len(active_pairs)}")
        if not collision_debug_checkbox.value:
            collision_point_a.visible = False
            collision_point_b.visible = False
            if collision_line_handle is not None:
                collision_line_handle.visible = False
        if collision_line_handle is not None:
            collision_line_handle.remove()
            collision_line_handle = None
        update_collision_visuals()

    sync_from_backend(update_target=True)

    prev_manual_state = False

    @bias_to_initial.on_click
    def _(_evt) -> None:
        nonlocal nullspace_bias
        nullspace_bias = default_bias_for_backend(backend)
        status_handle.value = "Status: Nullspace bias reset to initial configuration"

    @bias_to_zero.on_click
    def _(_evt) -> None:
        nonlocal nullspace_bias
        nullspace_bias = np.zeros_like(nullspace_bias)
        status_handle.value = "Status: Nullspace bias set to zero"

    @snap_target_button.on_click
    def _(_evt) -> None:
        current_pose = backend.get_pose()
        quat = r2q(current_pose.rotation, order="xyzs")
        ik_target.position = tuple(current_pose.translation)
        ik_target.wxyz = (quat[3], quat[0], quat[1], quat[2])
        status_handle.value = "Status: Target snapped to current end-effector pose"

    @reset_robot_button.on_click
    def _(_evt) -> None:
        backend.reset()
        sync_from_backend(update_target=True)
        status_handle.value = "Status: Robot reset to default configuration"

    @self_collision_checkbox.on_update
    def _(_evt) -> None:
        if hasattr(backend, "enable_self_collision"):
            backend.enable_self_collision(self_collision_checkbox.value and not self_collision_checkbox.disabled)
            solver_obj = getattr(backend, "solver", None)
            if (
                collision_debug_checkbox.value
                and self_collision_checkbox.value
                and solver_obj is not None
                and hasattr(solver_obj, "get_active_collision_pairs")
            ):
                active_pairs = solver_obj.get_active_collision_pairs()
                print(f"[embodiK] Active collision pairs: {len(active_pairs)}")
        if not collision_debug_checkbox.value:
            collision_point_a.visible = False
            collision_point_b.visible = False
            if collision_line_handle is not None:
                collision_line_handle.visible = False
        update_collision_visuals()

    @collision_debug_checkbox.on_update
    def _(_evt) -> None:
        update_collision_visuals()

    while True:
        solver_elapsed_ms = 0.0

        if manual_control.value:
            if not prev_manual_state:
                # entering manual mode, ensure sliders reflect current joint state
                q_current = backend.get_q()
                for slider, value in zip(joint_sliders, q_current):
                    slider.value = float(value)
            q_current = np.array([slider.value for slider in joint_sliders], dtype=float)
            backend.set_q(q_current)
            status_handle.value = "Status: Manual joint control active"
        else:
            target_position = np.array(ik_target.position, dtype=float)
            target_wxyz = np.array(ik_target.wxyz, dtype=float)
            target_xyzw = np.array([target_wxyz[1], target_wxyz[2], target_wxyz[3], target_wxyz[0]])
            target_rotation = q2r(target_xyzw, order="xyzs")
            target_pose = Rt(R=target_rotation, t=target_position)

            active_indices = [i for i, checkbox in enumerate(nullspace_checkboxes) if checkbox.value]
            if not nullspace_enabled_checkbox.value:
                active_indices = []

            result = backend.solve_step(
                target_pose,
                pos_gain.value,
                rot_gain.value,
                active_indices,
                nullspace_bias,
                nullspace_gain.value,
                nullspace_enabled_checkbox.value,
            )
            q_current = result.joints
            solver_elapsed_ms = result.elapsed_ms
            status_handle.value = (
                f"Status: embodiK {result.status} | "
                f"pos={result.position_error*1e3:.2f} mm, rot={result.rotation_error:.4f} rad"
            )

            for slider, value in zip(joint_sliders, q_current):
                slider.value = float(value)

        prev_manual_state = manual_control.value

        urdf_vis.update_cfg(make_visual_config(q_current))
        update_collision_visuals()

        timing_handle.value = 0.9 * timing_handle.value + 0.1 * solver_elapsed_ms

        time.sleep(0.001)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive collision-aware IK using embodiK.")
    parser.add_argument(
        "--robot",
        choices=sorted(ROBOT_PRESETS.keys()),
        default="panda",
        help="Robot model to load (default: panda).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = resolve_robot_configuration(args.robot)
    print(f"Interactive collision-aware IK demo ({cfg.display_name})")
    print(f"  - URDF path: {cfg.urdf_path}")
    print(f"  - Target link: {cfg.target_link}")

    run_gui(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
