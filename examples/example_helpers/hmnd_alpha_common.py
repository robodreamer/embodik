"""Shared helpers for the Alpha wheelbase and teststand demos and benchmarks."""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import xacro
from xacro import substitution_args

from . import limit_profiles

LOGGER = logging.getLogger("embodik.alpha_common")

__all__ = [
    "WorkspaceLayout",
    "PackagePathResolver",
    "detect_layout",
    "process_xacro_to_temp_urdf",
    "MOBILE_BASE_JOINTS",
    "TORSO_NECK_JOINTS",
    "BASE_LOCK_JOINTS",
    "LEFT_ARM",
    "RIGHT_ARM",
    "TORSO",
    "HEAD",
    "LEFT_ARM_TESTSTAND",
    "RIGHT_ARM_TESTSTAND",
    "HOME_JOINT_POSITIONS",
    "HOME_JOINT_POSITIONS_TESTSTAND",
    "load_home_configuration",
    "create_teststand_gripper_xacro",
    "create_wheelbase_gripper_xacro",
    "get_arm_config_for_reachability",
    "create_robot_config_for_ik",
    "inject_gripper_visuals",
    "harmonize_teststand_finger_frames",
    "generate_alpha_arm_reachability",
]


@dataclass(frozen=True)
class WorkspaceLayout:
    """Convenience bundle of important directories."""

    script_dir: Path
    repo_root: Path
    workspace_root: Path
    projects_root: Path
    hmnd_repos: Path


def detect_layout(reference_file: Path) -> WorkspaceLayout:
    """Infer workspace layout relative to a reference file."""

    script_dir = reference_file.parent.resolve()
    repo_root = script_dir.parent
    workspace_root = repo_root.parent
    projects_root = workspace_root.parent
    hmnd_repos = projects_root / "hmnd-repos"
    return WorkspaceLayout(
        script_dir=script_dir,
        repo_root=repo_root,
        workspace_root=workspace_root,
        projects_root=projects_root,
        hmnd_repos=hmnd_repos,
    )


class PackagePathResolver:
    """Resolves ROS-style package URIs without relying on ROS tooling."""

    def __init__(self, layout: WorkspaceLayout, *, verbose: bool = False) -> None:
        self._layout = layout
        self._verbose = verbose
        self._cache: Dict[str, Path] = {}
        self._urdf_dir: Optional[Path] = None
        self._package_hints = self._build_package_hints()
        self.package_directories = self._collect_package_dirs()

    def set_urdf_dir(self, directory: Path) -> None:
        self._urdf_dir = directory.resolve()

    def find_package(self, package: str) -> Path:
        if package in self._cache:
            return self._cache[package]

        env_override = os.environ.get(f"SWIFT_IK_PACKAGE_{package.upper()}")
        if env_override:
            candidate = Path(env_override).expanduser()
            if candidate.exists():
                self._cache[package] = candidate
                return candidate

        env_ros = os.environ.get(f"{package.upper()}_PATH")
        if env_ros:
            candidate = Path(env_ros).expanduser()
            if candidate.exists():
                self._cache[package] = candidate
                return candidate

        for candidate in self._package_hints.get(package, []):
            if candidate.exists():
                self._cache[package] = candidate
                if self._verbose:
                    LOGGER.info("Resolved package %s to %s", package, candidate)
                return candidate

        raise FileNotFoundError(f"Package '{package}' not found. Checked: {self._package_hints.get(package, [])}")

    def resolve_uri(self, uri: str) -> str:
        if uri.startswith("package://"):
            package_name, _, rel = uri[len("package://") :].partition("/")
            base = self.find_package(package_name)
            resolved = self._resolve_package_relative(base, rel)
            if resolved:
                return str(resolved)
            return uri

        if uri.startswith("file://"):
            return uri[len("file://") :]

        path = Path(uri)
        if path.is_absolute():
            return str(path)

        if self._urdf_dir is not None:
            candidate = (self._urdf_dir / path).resolve()
            if candidate.exists():
                return str(candidate)

        return str(path)

    def create_filename_handler(self) -> callable:
        def handler(*args, **kwargs) -> str:
            # yourdfpy may call the handler with either positional argument or
            # keyword (fname=...). Accept both to stay compatible.
            if args:
                resource = args[0]
            else:
                resource = kwargs.get("fname") or kwargs.get("name")
                if resource is None:
                    raise TypeError("Expected filename argument for handler")
            resolved = self.resolve_uri(resource)
            if self._verbose and resolved != resource:
                LOGGER.debug("Resolved resource %s -> %s", resource, resolved)
            return resolved

        return handler

    def _collect_package_dirs(self) -> List[str]:
        dirs = []
        for hints in self._package_hints.values():
            for candidate in hints:
                if candidate.exists():
                    dirs.append(str(candidate))
        return dirs

    def _build_package_hints(self) -> Dict[str, List[Path]]:
        layout = self._layout
        hmnd = layout.hmnd_repos
        models = layout.workspace_root / "hmnd_robot_models"

        def hints(*candidates: Path) -> List[Path]:
            return [candidate.resolve() for candidate in candidates]

        return {
            "alpha_wheelbase_description": hints(
                hmnd / "hmnd" / "hmnd_robot" / "ros" / "platforms" / "alpha_wheelbase_description",
                hmnd / "hmnd" / "hmnd_robot" / "install" / "alpha_wheelbase_description" / "share" / "alpha_wheelbase_description",
                hmnd / "hmnd" / "hmnd_sim" / "assets" / "sim_robots" / "official" / "humanoid" / "alpha" / "wheelbase" / "v1" / "alpha_wheelbase_description",
                models / "alpha_wheelbase_description",
            ),
            "alpha_teststand_description": hints(
                hmnd / "hmnd" / "hmnd_robot" / "ros" / "platforms" / "alpha_teststand_description",
                hmnd / "hmnd" / "hmnd_robot" / "install" / "alpha_teststand_description" / "share" / "alpha_teststand_description",
                models / "alpha_teststand_description",
            ),
            "alpha_description": hints(
                hmnd / "hmnd" / "hmnd_robot" / "ros" / "platforms" / "alpha_description",
                hmnd / "hmnd" / "hmnd_robot" / "install" / "alpha_description" / "share" / "alpha_description",
                models / "alpha_description",
            ),
            "alpha_biped_description": hints(
                hmnd / "hmnd" / "hmnd_robot" / "ros" / "platforms" / "alpha_biped_description",
                models / "alpha_biped_description",
            ),
            "alpha_ethercat": hints(
                hmnd / "hmnd" / "hmnd_robot" / "install" / "alpha_ethercat",
                hmnd / "hmnd" / "hmnd_robot" / "install" / "alpha_ethercat" / "share" / "alpha_ethercat",
                hmnd / "hmnd" / "hmnd_robot" / "ros" / "platforms" / "alpha_ethercat",
            ),
            "alpha_navigation_sim": hints(
                hmnd / "hmnd" / "hmnd_robot" / "install" / "alpha_navigation_sim",
                hmnd / "hmnd" / "hmnd_robot" / "install" / "alpha_navigation_sim" / "share" / "alpha_navigation_sim",
                hmnd / "hmnd" / "hmnd_sim" / "assets" / "sim_robots" / "official" / "humanoid" / "alpha" / "wheelbase" / "v1",
            ),
            "robotiq_description": hints(
                hmnd / "hmnd" / "hmnd_sim" / "assets" / "sim_robots" / "grippers" / "robotiq" / "robotiq_description",
                hmnd / "hmnd" / "hmnd_robot" / "install" / "robotiq_description",
                hmnd / "hmnd" / "hmnd_robot" / "install" / "robotiq_description" / "share" / "robotiq_description",
            ),
        }

    def _resolve_package_relative(self, base: Path, relative: str) -> Optional[Path]:
        candidate = (base / relative).resolve()
        if candidate.exists():
            return candidate

        rel_path = Path(relative)
        alt_candidates = [
            base / "urdf" / rel_path,
            base / "meshes" / rel_path,
            base.parent / rel_path,
        ]

        for option in alt_candidates:
            if option.exists():
                return option.resolve()

        rel_name = rel_path.name
        for search_root in (base, base / "urdf", base / "meshes"):
            if not search_root.exists():
                continue
            for match in search_root.rglob(rel_name):
                return match.resolve()

        return None


def process_xacro_to_temp_urdf(
    xacro_path: Path,
    resolver: PackagePathResolver,
    *,
    mappings: Optional[Dict[str, str]] = None,
    verbose: bool = False,
) -> Path:
    """Render xacro into a temporary URDF file using the provided resolver."""

    resolver.set_urdf_dir(xacro_path.parent)

    with _override_xacro_find(resolver):
        doc = xacro.process_file(str(xacro_path), mappings=mappings or {})

    urdf_xml = doc.toxml()
    # Determine prefix based on xacro path
    prefix = "alpha_teststand_" if "teststand" in str(xacro_path).lower() else "alpha_wheelbase_"
    temp_file = tempfile.NamedTemporaryFile("w", suffix=".urdf", prefix=prefix, delete=False)
    temp_file.write(urdf_xml)
    temp_file.flush()
    temp_file.close()

    temp_path = Path(temp_file.name)
    resolver.set_urdf_dir(temp_path.parent)

    if verbose:
        LOGGER.info("Temporary URDF written to %s", temp_path)

    atexit.register(lambda: _safe_unlink(temp_path))
    return temp_path


@contextlib.contextmanager
def _override_xacro_find(resolver: PackagePathResolver) -> Iterable[None]:
    """Override xacro's file resolution to use our custom resolver."""
    original_find = substitution_args._eval_find
    original_parse = xacro.parse
    original_process_include = xacro.process_include

    def _custom_eval_find(pkg: str) -> str:
        """Override xacro's $(find package) evaluation."""
        return str(resolver.find_package(pkg))

    def _custom_parse(root_dir: Optional[str], filename: str) -> xacro.xml:
        """Override xacro's parse function to use resolver for file paths."""
        # If filename is already resolved and exists, use it directly
        if Path(filename).exists():
            return original_parse(root_dir, filename)

        # Try to resolve using our resolver
        resolved = resolver.resolve_uri(filename)
        if Path(resolved).exists():
            return original_parse(root_dir, resolved)

        # If file doesn't exist, try to resolve relative to current URDF directory
        if resolver._urdf_dir is not None:
            candidate = (resolver._urdf_dir / filename).resolve()
            if candidate.exists():
                return original_parse(root_dir, str(candidate))

        # Try package:// resolution
        if filename.startswith("package://"):
            resolved = resolver.resolve_uri(filename)
            if Path(resolved).exists():
                return original_parse(root_dir, resolved)

        # If still not found, try original parse (may raise FileNotFoundError)
        # This allows xacro to handle the error appropriately
        return original_parse(root_dir, filename)

    def _custom_process_include(node, macros, symbols, eval_all):
        """Override xacro's process_include to handle missing files gracefully."""
        try:
            return original_process_include(node, macros, symbols, eval_all)
        except (FileNotFoundError, xacro.XacroException) as e:
            # If file doesn't exist, try to resolve it using our resolver
            filename = node.getAttribute("filename")
            if filename:
                resolved = resolver.resolve_uri(filename)
                if Path(resolved).exists():
                    # Update the node's filename attribute and try again
                    node.setAttribute("filename", resolved)
                    return original_process_include(node, macros, symbols, eval_all)

            # If still not found, check if it's an optional include (like cameras)
            # and skip it with a warning
            filename_str = filename or str(e)
            if "camera" in filename_str.lower() or "alpha_head_cameras" in filename_str:
                if resolver._verbose:
                    LOGGER.warning(f"Skipping optional include file that doesn't exist: {filename_str}")
                return

            # For other files, re-raise the error
            raise

    substitution_args._eval_find = _custom_eval_find
    xacro.parse = _custom_parse
    xacro.process_include = _custom_process_include

    try:
        yield
    finally:
        substitution_args._eval_find = original_find
        xacro.parse = original_parse
        xacro.process_include = original_process_include


def _safe_unlink(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


MOBILE_BASE_JOINTS: Set[str] = {
    "caster_back_left_joint",
    "wheel_back_left_joint",
    "caster_back_right_joint",
    "wheel_back_right_joint",
    "caster_front_left_joint",
    "wheel_front_left_joint",
    "caster_front_right_joint",
    "wheel_front_right_joint",

}

# Torso and neck joints (lockable via checkbox)
TORSO_NECK_JOINTS: Set[str] = {
    "torso_yaw_joint",
    "base_yaw_joint",
    "base_pitch_joint",
    "hip_pitch_joint",
    "knee_pitch_joint",
    "neck_yaw_joint",
    "neck_pitch_joint",
}

# Base lock joints: mobile base joints should always be locked (they're the floating base)
# Torso/neck joints can be optionally locked via checkbox
BASE_LOCK_JOINTS: Set[str] = MOBILE_BASE_JOINTS | TORSO_NECK_JOINTS

RIGHT_ARM: Tuple[str, ...] = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)

LEFT_ARM: Tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
)

TORSO: Tuple[str, ...] = (
    "base_yaw_joint",
    "base_pitch_joint",
    "knee_pitch_joint",
    "hip_pitch_joint",
    "torso_yaw_joint",
)

HEAD: Tuple[str, ...] = (
    "neck_yaw_joint",
    "neck_pitch_joint",
)

# Teststand-specific joint definitions (no head, no grippers, no torso)
LEFT_ARM_TESTSTAND: Tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
)

RIGHT_ARM_TESTSTAND: Tuple[str, ...] = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
)

HOME_JOINT_POSITIONS: Dict[str, float] = {
    "base_pitch_joint": 0.0,
    "base_yaw_joint": 0.0,
    "hip_pitch_joint": 0.0,
    "knee_pitch_joint": 0.0,
    "torso_yaw_joint": 0.0,
    "neck_yaw_joint": 0.0,
    "neck_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_pitch_joint": 0.0,
    "left_elbow_yaw_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_pitch_joint": 0.0,
    "right_elbow_yaw_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
}

# Teststand home configuration (L-shape pose from teststand example)
HOME_JOINT_POSITIONS_TESTSTAND: Dict[str, float] = {
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.3,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_pitch_joint": -1.571,
    "left_elbow_yaw_joint": 0.3,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": -0.3,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_pitch_joint": -1.571,
    "right_elbow_yaw_joint": -0.3,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
}


def load_home_configuration(joint_names: Sequence[str], *, is_teststand: bool = False) -> np.ndarray:
    """Load home configuration for joint names.

    Args:
        joint_names: List of joint names
        is_teststand: If True, use teststand home positions; otherwise use wheelbase positions
    """
    q = np.zeros(len(joint_names), dtype=float)
    home_positions = HOME_JOINT_POSITIONS_TESTSTAND if is_teststand else HOME_JOINT_POSITIONS
    for idx, name in enumerate(joint_names):
        if name in home_positions:
            q[idx] = home_positions[name]
    return q


def create_teststand_gripper_xacro(resolver: PackagePathResolver, verbose: bool = False) -> Path:
    """Create a temporary xacro that attaches Robotiq grippers to the teststand arms.

    Matches the wheelbase xacro pattern for consistency. Creates IK helper frames
    to ensure both arms have symmetric behavior for 6-DOF control.

    Args:
        resolver: Package path resolver for resolving URIs
        verbose: If True, log creation details

    Returns:
        Path to the temporary xacro file (automatically cleaned up on exit)
    """
    static_visual_xacro = (Path(__file__).resolve().parent / "robotiq_static_visual.xacro").as_posix()
    content = f"""<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="alpha_teststand_with_grippers">
  <xacro:property name="pi" value="3.141592653589793"/>
  <xacro:include filename="$(find alpha_teststand_description)/urdf/alpha_teststand.urdf"/>
  <xacro:include filename="$(find alpha_description)/alpha_left_arm/urdf/alpha_left_arm.urdf"/>
  <xacro:include filename="$(find alpha_description)/alpha_right_arm/urdf/alpha_right_arm.urdf"/>
  <xacro:include filename="{static_visual_xacro}"/>

  <joint name="left_shoulder_pitch_joint" type="revolute">
    <origin xyz="0 0.124108578064976 0.161430614180372" rpy="-1.3963 0 0" />
    <parent link="torso_link" />
    <child link="left_bicep_pitch_link" />
    <axis xyz="0 0 1" />
    <limit lower="-3.1416" upper="0.5236" effort="96" velocity="3.926991" />
  </joint>

  <joint name="right_shoulder_pitch_joint" type="revolute">
    <origin xyz="0 -0.124108578068307 0.161431" rpy="-1.7453 0 0" />
    <parent link="torso_link" />
    <child link="right_bicep_pitch_link" />
    <axis xyz="0 0 1" />
    <limit lower="-3.1416" upper="0.5236" effort="96" velocity="3.926991" />
  </joint>
  <xacro:swiftik_left_static_robotiq parent="left_hand_frame"/>
  <xacro:swiftik_right_static_robotiq parent="right_hand_frame"/>

  <!-- IK helper frames to ensure symmetric behavior for both arms -->
  <!-- Left finger frame already has 180deg Z rotation, so IK frame matches it -->
  <link name="left_finger_ik_frame"/>
  <joint name="left_finger_ik_frame_joint" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0" />
    <parent link="left_finger_frame" />
    <child link="left_finger_ik_frame" />
  </joint>
  <!-- Right finger frame has no rotation, so add 180deg Z rotation for symmetry -->
  <link name="right_finger_ik_frame"/>
  <joint name="right_finger_ik_frame_joint" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 3.1416" />
    <parent link="right_finger_frame" />
    <child link="right_finger_ik_frame" />
  </joint>
</robot>
"""
    temp_file = tempfile.NamedTemporaryFile("w", suffix=".xacro", prefix="alpha_teststand_gripper_", delete=False)
    temp_file.write(content)
    temp_file.flush()
    temp_file.close()
    temp_path = Path(temp_file.name)
    atexit.register(_safe_unlink, temp_path)
    if verbose:
        LOGGER.info("Created temporary teststand gripper xacro at %s", temp_path)
    return temp_path


def _safe_unlink(path: Path) -> None:
    """Safely remove a file, suppressing errors."""
    with contextlib.suppress(OSError):
        path.unlink()


def create_wheelbase_gripper_xacro(resolver: PackagePathResolver, verbose: bool = False) -> Path:
    """Create a temporary xacro that attaches Robotiq grippers to the wheelbase arms using static visual macros.

    Uses the same gripper macro approach as teststand for consistency. Creates IK helper frames
    to ensure both arms have symmetric behavior for 6-DOF control.

    This creates a wrapper xacro that includes the wheelbase xacro with gripper disabled,
    then adds the static visual gripper macros (same as teststand).

    Args:
        resolver: Package path resolver for resolving URIs
        verbose: If True, log creation details

    Returns:
        Path to the temporary xacro file (automatically cleaned up on exit)
    """
    wheelbase_package_uri = "package://alpha_wheelbase_description/urdf/alpha_wheelbase.urdf.xacro"
    wheelbase_xacro_path = Path(resolver.resolve_uri(wheelbase_package_uri))

    if not wheelbase_xacro_path.exists():
        raise FileNotFoundError(f"Wheelbase xacro not found: {wheelbase_xacro_path}")

    static_visual_xacro = (Path(__file__).resolve().parent / "robotiq_static_visual.xacro").as_posix()
    # Create wrapper xacro that includes wheelbase xacro with gripper disabled, then adds static visual grippers
    # Declare gripper arg so it can be passed via mappings, then set property before including wheelbase xacro
    content = f"""<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="alpha_wheelbase_with_grippers">
  <xacro:arg name="gripper" default="none"/>
  <xacro:property name="pi" value="3.141592653589793"/>

  <!-- Set gripper property to disable standard gripper macros in included wheelbase xacro -->
  <!-- The included xacro will use this property value via $(arg gripper) -->
  <xacro:property name="gripper" value="$(arg gripper)"/>

  <!-- Include wheelbase xacro (will use gripper="none" to skip standard gripper macros) -->
  <xacro:include filename="{wheelbase_xacro_path}"/>

  <!-- Include static visual gripper macros (same as teststand) -->
  <xacro:include filename="{static_visual_xacro}"/>

  <!-- Apply static visual gripper macros to hand frames (same as teststand) -->
  <xacro:swiftik_left_static_robotiq parent="left_hand_frame"/>
  <xacro:swiftik_right_static_robotiq parent="right_hand_frame"/>

  <!-- IK helper frames to ensure symmetric behavior for both arms (same as teststand) -->
  <!-- Left finger frame already has 180deg Z rotation, so IK frame matches it -->
  <link name="left_finger_ik_frame"/>
  <joint name="left_finger_ik_frame_joint" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0" />
    <parent link="left_finger_frame" />
    <child link="left_finger_ik_frame" />
  </joint>
  <!-- Right finger frame has no rotation, so add 180deg Z rotation for symmetry -->
  <link name="right_finger_ik_frame"/>
  <joint name="right_finger_ik_frame_joint" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 3.1416" />
    <parent link="right_finger_frame" />
    <child link="right_finger_ik_frame" />
  </joint>
</robot>
"""
    temp_file = tempfile.NamedTemporaryFile("w", suffix=".xacro", prefix="alpha_wheelbase_gripper_", delete=False)
    temp_file.write(content)
    temp_file.flush()
    temp_file.close()
    temp_path = Path(temp_file.name)
    atexit.register(_safe_unlink, temp_path)
    if verbose:
        LOGGER.info("Created temporary wheelbase gripper xacro at %s", temp_path)
    return temp_path


def harmonize_teststand_finger_frames(urdf_path: Path, *, verbose: bool = False) -> None:
    """Placeholder for backwards compatibility; no URDF rewrite required."""
    if verbose:
        LOGGER.info("Skipping finger-frame harmonisation; using gripper frames for IK.")


def get_arm_config_for_reachability(
    arm: str,
    is_teststand: bool,
    left_frame: str,
    right_frame: str,
    joint_names: Sequence[str],
) -> Tuple[str, Sequence[str]]:
    """Get end-effector frame and joint names for reachability analysis.

    Args:
        arm: "left" or "right"
        is_teststand: Whether using teststand model
        left_frame: Left arm end-effector frame name
        right_frame: Right arm end-effector frame name
        joint_names: All joint names in the robot

    Returns:
        Tuple of (end_effector_frame, arm_joint_names)
    """
    if arm == "left":
        end_effector = left_frame
        arm_joints = list(LEFT_ARM_TESTSTAND if is_teststand else LEFT_ARM)
    elif arm == "right":
        end_effector = right_frame
        arm_joints = list(RIGHT_ARM_TESTSTAND if is_teststand else RIGHT_ARM)
    else:
        raise ValueError(f"Invalid arm selection: {arm}. Must be 'left' or 'right'.")

    # Filter to only joints that exist in the robot
    arm_joint_names = [name for name in arm_joints if name in joint_names]
    return end_effector, arm_joint_names


def create_robot_config_for_ik(
    urdf_path: Path,
    arm: str,
    end_effector: str,
    joint_names: Sequence[str],
    arm_joint_names: Sequence[str],
    *,
    limit_overrides: Optional[Dict[str, Tuple[float, float]]] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> "RobotConfig":  # type: ignore
    """Create RobotConfig for IK reachability analysis.

    Args:
        urdf_path: Path to URDF file
        arm: "left" or "right"
        end_effector: End-effector frame name
        joint_names: All joint names
        arm_joint_names: Arm-specific joint names

    Returns:
        RobotConfig instance
    """
    # Import here to avoid circular dependency
    from example_helpers.reachability_robot_configs import RobotConfig

    robot_key = f"alpha_{arm}_arm"
    return RobotConfig(
        key=robot_key,
        display_name=f"Alpha {arm.capitalize()} Arm",
        urdf_path=urdf_path,
        description_name=f"alpha_{arm}_arm",
        target_link=end_effector,
        joint_labels=[name.replace("_joint", "").replace("_", " ").title() for name in arm_joint_names],
        joint_names=list(arm_joint_names),
        default_configuration=np.zeros(len(arm_joint_names)),
        default_offset=np.array([0.0, 0.0, 0.0]),
        collision_exclusions=[],
        limit_overrides=dict(limit_overrides or {}),
        metadata=dict(metadata or {}),
    )


def inject_gripper_visuals(urdf_path: Path, *, verbose: bool = False) -> None:
    """Attach simple visual meshes to the gripper frames for nicer rendering."""
    try:
        tree = ET.parse(urdf_path)
    except ET.ParseError as exc:  # pragma: no cover - malformed URDF should not happen
        LOGGER.warning("Failed to parse URDF for visual injection at %s: %s", urdf_path, exc)
        return

    root = tree.getroot()

    # Bail out early if the helper frames are missing (e.g. wheelbase without grippers).
    if root.find("./link[@name='left_gripper_frame']") is None or root.find("./link[@name='right_gripper_frame']") is None:
        return

    changed = False

    def _attach_mesh_link(
        parent_link: str,
        link_name: str,
        mesh_path: str,
        xyz: str,
        rpy: str,
        scale: Optional[str] = None,
    ) -> None:
        """Create a fixed joint that carries a purely visual mesh."""
        nonlocal changed
        if root.find(f"./link[@name='{link_name}']") is not None:
            return

        link = ET.SubElement(root, "link", {"name": link_name})
        visual = ET.SubElement(link, "visual")
        ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ET.SubElement(visual, "geometry")
        mesh_attrs = {"filename": mesh_path}
        if scale is not None:
            mesh_attrs["scale"] = scale
        ET.SubElement(geometry, "mesh", mesh_attrs)

        joint = ET.SubElement(root, "joint", {"name": f"{link_name}_joint", "type": "fixed"})
        ET.SubElement(joint, "parent", {"link": parent_link})
        ET.SubElement(joint, "child", {"link": link_name})
        ET.SubElement(joint, "origin", {"xyz": xyz, "rpy": rpy})
        changed = True

    # Static Robotiq geometry approximations (open gripper pose).
    base_mesh = "package://alpha_description/robotiq_gripper/meshes/visual/2f_85/robotiq_base.dae"
    adapter_mesh = "package://alpha_description/robotiq_gripper/meshes/visual/2f_85/ur_to_robotiq_adapter.dae"
    left_knuckle_mesh = "package://alpha_description/robotiq_gripper/meshes/visual/2f_85/left_knuckle.dae"
    right_knuckle_mesh = "package://alpha_description/robotiq_gripper/meshes/visual/2f_85/right_knuckle.dae"
    left_finger_mesh = "package://alpha_description/robotiq_gripper/meshes/visual/2f_85/left_finger.dae"
    right_finger_mesh = "package://alpha_description/robotiq_gripper/meshes/visual/2f_85/right_finger.dae"
    left_inner_knuckle_mesh = "package://alpha_description/robotiq_gripper/meshes/visual/2f_85/left_inner_knuckle.dae"
    right_inner_knuckle_mesh = "package://alpha_description/robotiq_gripper/meshes/visual/2f_85/right_inner_knuckle.dae"
    left_finger_tip_mesh = "package://alpha_description/robotiq_gripper/meshes/visual/2f_85/left_finger_tip.dae"
    right_finger_tip_mesh = "package://alpha_description/robotiq_gripper/meshes/visual/2f_85/right_finger_tip.dae"
    left_ft_mesh = "package://alpha_description/alpha_left_arm/meshes/left_hand_FT_angled_plastic_link.STL"
    right_ft_mesh = "package://alpha_description/alpha_right_arm/meshes/right_hand_FT_angled_plastic_link.STL"
    left_pad_mesh = "package://alpha_description/alpha_left_arm/meshes/left_hand_pad_link.STL"
    right_pad_mesh = "package://alpha_description/alpha_right_arm/meshes/right_hand_pad_link.STL"

    # Transform constants lifted from the original Robotiq macro (zero joint configuration).
    finger_offset_pos = "0.062127 0 0.051141"
    inner_knuckle_offset_pos = "0.0127 0 0.06142"
    inner_knuckle_offset_neg = "-0.0127 0 0.06142"
    finger_offset_neg = "-0.062127 0 0.051141"
    finger_tip_offset_pos = "0.00563134 0 0.04718515"
    finger_tip_offset_neg = "-0.00563134 0 0.04718515"
    knuckle_offset_pos = "0.03060114 0 0.05490452"
    knuckle_offset_neg = "-0.03060114 0 0.05490452"

    _attach_mesh_link(
        parent_link="left_hand_frame",
        link_name="left_gripper_visual_ft",
        mesh_path=left_ft_mesh,
        xyz="0 0 0",
        rpy="0 0 0",
        scale="0.001 0.001 0.001",
    )
    _attach_mesh_link(
        parent_link="left_gripper_visual_ft",
        link_name="left_gripper_visual_pad",
        mesh_path=left_pad_mesh,
        xyz="-0.01267 0 0.02716",
        rpy="-0.4363 0 1.5708",
    )
    _attach_mesh_link(
        parent_link="left_gripper_visual_pad",
        link_name="left_gripper_visual_base",
        mesh_path=base_mesh,
        xyz="0 0 0.03",
        rpy="0 0 3.1416",
    )
    _attach_mesh_link(
        parent_link="left_gripper_visual_base",
        link_name="left_gripper_visual_adapter",
        mesh_path=adapter_mesh,
        xyz="0 0 0",
        rpy="0 0 0",
    )
    _attach_mesh_link(
        parent_link="left_gripper_visual_base",
        link_name="left_gripper_visual_knuckle_left",
        mesh_path=left_knuckle_mesh,
        xyz=knuckle_offset_pos,
        rpy="0 0 0",
    )
    _attach_mesh_link(
        parent_link="left_gripper_visual_base",
        link_name="left_gripper_visual_knuckle_right",
        mesh_path=right_knuckle_mesh,
        xyz=knuckle_offset_neg,
        rpy="0 0 0",
    )
    _attach_mesh_link(
        parent_link="left_gripper_visual_base",
        link_name="left_gripper_visual_finger_left",
        mesh_path=left_finger_mesh,
        xyz=finger_offset_pos,
        rpy="0 0 0",
    )
    _attach_mesh_link(
        parent_link="left_gripper_visual_base",
        link_name="left_gripper_visual_inner_knuckle_left",
        mesh_path=left_inner_knuckle_mesh,
        xyz=inner_knuckle_offset_pos,
        rpy="0 0 0",
    )
    _attach_mesh_link(
        parent_link="left_gripper_visual_base",
        link_name="left_gripper_visual_finger_right",
        mesh_path=right_finger_mesh,
        xyz=finger_offset_neg,
        rpy="0 0 0",
    )
    _attach_mesh_link(
        parent_link="left_gripper_visual_base",
        link_name="left_gripper_visual_inner_knuckle_right",
        mesh_path=right_inner_knuckle_mesh,
        xyz=inner_knuckle_offset_neg,
        rpy="0 0 0",
    )
    _attach_mesh_link(
        parent_link="left_gripper_visual_finger_left",
        link_name="left_gripper_visual_finger_tip_left",
        mesh_path=left_finger_tip_mesh,
        xyz=finger_tip_offset_pos,
        rpy="0 0 0",
    )
    _attach_mesh_link(
        parent_link="left_gripper_visual_finger_right",
        link_name="left_gripper_visual_finger_tip_right",
        mesh_path=right_finger_tip_mesh,
        xyz=finger_tip_offset_neg,
        rpy="0 0 0",
    )

    _attach_mesh_link(
        parent_link="right_hand_frame",
        link_name="right_gripper_visual_ft",
        mesh_path=right_ft_mesh,
        xyz="0 0 0",
        rpy="0 0 0",
        scale="0.001 0.001 0.001",
    )
    _attach_mesh_link(
        parent_link="right_gripper_visual_ft",
        link_name="right_gripper_visual_pad",
        mesh_path=right_pad_mesh,
        xyz="-0.01267 0 0.02716",
        rpy="0.4363 0 -1.5708",
    )
    _attach_mesh_link(
        parent_link="right_gripper_visual_pad",
        link_name="right_gripper_visual_base",
        mesh_path=base_mesh,
        xyz="0 0 0.03",
        rpy="0 0 0",
    )
    _attach_mesh_link(
        parent_link="right_gripper_visual_base",
        link_name="right_gripper_visual_adapter",
        mesh_path=adapter_mesh,
        xyz="0 0 0",
        rpy="0 0 0",
    )
    _attach_mesh_link(
        parent_link="right_gripper_visual_base",
        link_name="right_gripper_visual_knuckle_left",
        mesh_path=left_knuckle_mesh,
        xyz=knuckle_offset_pos,
        rpy="0 0 0",
    )
    _attach_mesh_link(
        parent_link="right_gripper_visual_base",
        link_name="right_gripper_visual_knuckle_right",
        mesh_path=right_knuckle_mesh,
        xyz=knuckle_offset_neg,
        rpy="0 0 0",
    )
    _attach_mesh_link(
        parent_link="right_gripper_visual_base",
        link_name="right_gripper_visual_finger_left",
        mesh_path=left_finger_mesh,
        xyz=finger_offset_pos,
        rpy="0 0 0",
    )
    _attach_mesh_link(
        parent_link="right_gripper_visual_base",
        link_name="right_gripper_visual_inner_knuckle_left",
        mesh_path=left_inner_knuckle_mesh,
        xyz=inner_knuckle_offset_pos,
        rpy="0 0 0",
    )
    _attach_mesh_link(
        parent_link="right_gripper_visual_base",
        link_name="right_gripper_visual_finger_right",
        mesh_path=right_finger_mesh,
        xyz=finger_offset_neg,
        rpy="0 0 0",
    )
    _attach_mesh_link(
        parent_link="right_gripper_visual_base",
        link_name="right_gripper_visual_inner_knuckle_right",
        mesh_path=right_inner_knuckle_mesh,
        xyz=inner_knuckle_offset_neg,
        rpy="0 0 0",
    )
    _attach_mesh_link(
        parent_link="right_gripper_visual_finger_left",
        link_name="right_gripper_visual_finger_tip_left",
        mesh_path=left_finger_tip_mesh,
        xyz=finger_tip_offset_pos,
        rpy="0 0 0",
    )
    _attach_mesh_link(
        parent_link="right_gripper_visual_finger_right",
        link_name="right_gripper_visual_finger_tip_right",
        mesh_path=right_finger_tip_mesh,
        xyz=finger_tip_offset_neg,
        rpy="0 0 0",
    )

    if changed:
        tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
        if verbose:
            LOGGER.info("Injected gripper visual meshes into URDF at %s", urdf_path)


def generate_alpha_arm_reachability(
    *,
    xacro_path: Path,
    end_effector: str,
    output_dir: Path,
    job_name: str,
    # Configuration parameters
    total_samples: int,
    batch_size: int,
    cartesian_resolution: float,
    angular_resolution: float,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
    z_limits: Tuple[float, float],
    roll_limits: Tuple[float, float],
    pitch_limits: Tuple[float, float],
    yaw_limits: Tuple[float, float],
    save_every: int,
    post_process: bool,
    device: str,
    dtype,
    # Arm selection
    arm_joints: Sequence[str],
    # Callbacks
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    limit_overrides: Optional[Dict[str, Tuple[float, float]]] = None,
    metadata: Optional[Dict[str, object]] = None,
    resolver: Optional[PackagePathResolver] = None,
    resolved_urdf: Optional[Path] = None,
) -> Dict[str, Path]:
    """Generate FK-based reachability map for Alpha wheelbase arm.

    This is a wrapper around the shared generate_fk_reachability_map function
    that handles Alpha-specific Xacro resolution, URDF joint limit extraction,
    and base joint identification.

    Args:
        xacro_path: Path to Alpha wheelbase Xacro file
        end_effector: End effector frame name (e.g., "left_gripper_frame")
        output_dir: Directory for output files
        job_name: Job name for output files
        total_samples: Total number of samples to generate
        batch_size: Batch size for sampling
        cartesian_resolution: Spatial resolution in meters
        angular_resolution: Angular resolution in radians
        x_limits: X-axis limits (min, max)
        y_limits: Y-axis limits (min, max)
        z_limits: Z-axis limits (min, max)
        roll_limits: Roll angle limits (min, max)
        pitch_limits: Pitch angle limits (min, max)
        yaw_limits: Yaw angle limits (min, max)
        save_every: Save interval (in loops)
        post_process: Whether to generate filtered maps
        device: Device string ("auto", "cpu", "cuda")
        dtype: Torch dtype for computations
        arm_joints: Sequence of arm joint names (e.g., LEFT_ARM or RIGHT_ARM)
        progress_callback: Optional progress callback function
        cancel_event: Optional cancellation event
        limit_overrides: Optional dict mapping joint name to (min, max) values that
            should override URDF/robot limits during map generation.
        metadata: Optional dictionary of robot metadata to embed into outputs.
        resolver: Optional PackagePathResolver to reuse for resource lookup.
        resolved_urdf: Optional pre-resolved URDF path. If provided, xacro processing
            is skipped and this path is used directly.

    Returns:
        Dictionary with output file paths
    """
    import math
    import threading
    from typing import Callable

    import embodik
    import torch
    import yourdfpy

    from example_helpers.reachability_fk import generate_fk_reachability_map

    if resolved_urdf is None:
        if not xacro_path.exists():
            raise FileNotFoundError(f"Xacro file not found: {xacro_path}")
        if resolver is None:
            layout = detect_layout(Path(__file__).resolve())
            resolver = PackagePathResolver(layout)
        resolved_urdf = process_xacro_to_temp_urdf(
            xacro_path,
            resolver,
            mappings={"ros2_control_hardware_type": "mock_components", "gripper": "robotiq"},
        )
        LOGGER.info("Resolved Xacro -> %s", resolved_urdf)
    else:
        resolved_urdf = resolved_urdf.resolve()
        LOGGER.info("Using pre-resolved URDF for FK reachability: %s", resolved_urdf)

    # Load robot model and extract joint limits
    robot = embodik.RobotModel(str(resolved_urdf))
    joint_names = robot.get_joint_names()
    lower_limits, upper_limits = robot.get_joint_limits()

    # Extract joint limits from URDF (may override robot model limits)
    urdf_model = yourdfpy.URDF.load(str(resolved_urdf), load_meshes=False)
    urdf_limits: Dict[str, Tuple[float, float]] = {}
    for joint_name, joint in urdf_model.joint_map.items():
        limit = getattr(joint, "limit", None)
        if limit is None:
            continue
        lower = getattr(limit, "lower", None)
        upper = getattr(limit, "upper", None)
        if lower is None or upper is None:
            continue
        if not math.isfinite(lower) or not math.isfinite(upper):
            continue
        urdf_limits[joint_name] = (float(lower), float(upper))

    # Build joint limit map (URDF limits override robot model limits)
    joint_limit_map: Dict[str, Tuple[float, float]] = {}
    for idx, name in enumerate(joint_names):
        lower = float(lower_limits[idx])
        upper = float(upper_limits[idx])
        if name in urdf_limits:
            lower, upper = urdf_limits[name]
        joint_limit_map[name] = (lower, upper)
    if limit_overrides:
        for joint_name, bounds in limit_overrides.items():
            joint_limit_map[joint_name] = bounds

    # Call shared function with Alpha-specific parameters
    return generate_fk_reachability_map(
        urdf_path=resolved_urdf,
        chain_joint_names=arm_joints,
        end_effector=end_effector,
        output_dir=output_dir,
        job_name=job_name,
        total_samples=total_samples,
        batch_size=batch_size,
        cartesian_resolution=cartesian_resolution,
        angular_resolution=angular_resolution,
        x_limits=x_limits,
        y_limits=y_limits,
        z_limits=z_limits,
        save_every=save_every,
        post_process=post_process,
        device=device,
        dtype=dtype,
        # Alpha uses explicit base joint identification
        base_joint_names=TORSO,
        # Alpha uses configurable angular limits
        roll_limits=roll_limits,
        pitch_limits=pitch_limits,
        yaw_limits=yaw_limits,
        # Alpha uses URDF-based joint limits
        joint_limit_map=joint_limit_map,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        # samples_joint_names=None means use chain_joint_names (Alpha mode)
        samples_joint_names=None,
        robot_metadata=metadata,
    )
