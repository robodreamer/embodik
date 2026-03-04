"""Build a dual LBR iiwa 14 arm URDF from robot_descriptions iiwa14_description.

Composes two KUKA LBR iiwa 14 kg arms into a single URDF with a shared base
using the same layout as the MATLAB GUI_example_LBRiiwa_extensions.m reference:
  - Left arm at Y = +0.3 m, no rotation
  - Right arm at Y = -0.3 m, no rotation
"""

from pathlib import Path
import re


def _resolve_drake_package_paths(urdf_text: str, drake_root: Path) -> str:
    """Replace package://drake/... with absolute file paths."""

    def repl(m: re.Match) -> str:
        rel = m.group(1)
        return str(drake_root / rel)

    return re.sub(r"package://drake/([^\"]+)", repl, urdf_text)


def _replace_mesh_collision_with_primitives(urdf_text: str) -> str:
    """Replace mesh-based collision geometries for iiwa links 6 and 7 with spheres.

    The Drake iiwa14 URDF intentionally uses high-poly meshes for the wrist
    (link_6: ~9 800 vertices) and flange (link_7: ~34 000 vertices) even in
    its "primitive" collision mode.  These make GJK distance queries ~100×
    more expensive than the simple cylinders used for links 0–5.

    We match each mesh collision by its unique filename and substitute a sphere
    that conservatively bounds the original mesh:
      - link_6 (wrist) : sphere r = 0.09 m, centred at link origin
      - link_7 (flange): sphere r = 0.06 m, centred at link origin
    """
    substitutions = [
        # (unique substring to identify the <geometry> block, replacement sphere tag)
        ("meshes/collision/link_6.obj", '<sphere radius="0.09"/>'),
        ("meshes/collision/link_7.obj", '<sphere radius="0.06"/>'),
    ]
    for mesh_id, sphere_tag in substitutions:
        # Match the entire <geometry>…</geometry> block that contains this mesh path.
        # The block may contain an XML comment before the <mesh/> tag (as Drake does).
        urdf_text = re.sub(
            r"<geometry>(?:(?!<geometry>).)*?" + re.escape(mesh_id) + r".*?</geometry>",
            f"<geometry>{sphere_tag}</geometry>",
            urdf_text,
            flags=re.DOTALL,
        )
    return urdf_text


def _strip_drake_namespace_attributes(urdf_text: str) -> str:
    """Remove drake:* attributes so the URDF can be parsed without xmlns:drake.

    The Drake iiwa URDF uses e.g. <limit drake:acceleration="8.57" .../>. Our
    composed dual_iiwa URDF does not declare xmlns:drake, so parsers warn or fail.
    Stripping these attributes avoids the issue; standard limit attributes remain.
    """
    # Match optional spaces, then drake:name="value" (value can contain spaces in quotes)
    return re.sub(r'\s+drake:\w+="[^"]*"', "", urdf_text)


def _prefix_iiwa_content(urdf_text: str, prefix: str) -> str:
    """Add *prefix* to all iiwa link/joint/transmission names in URDF content."""
    patterns = [
        # Links (order: longer patterns first to avoid partial substitutions)
        (r"\biiwa_link_ee_kuka\b", f"{prefix}iiwa_link_ee_kuka"),
        (r"\biiwa_link_ee\b", f"{prefix}iiwa_link_ee"),
        (r"\biiwa_link_(\d+)\b", f"{prefix}iiwa_link_\\1"),
        # Joints
        (r"\biiwa_joint_ee\b", f"{prefix}iiwa_joint_ee"),
        (r"\biiwa_joint_(\d+)\b", f"{prefix}iiwa_joint_\\1"),
        (r"\btool0_joint\b", f"{prefix}tool0_joint"),
        # Transmissions / actuators
        (r"\biiwa_tran_(\d+)\b", f"{prefix}iiwa_tran_\\1"),
        (r"\biiwa_motor_(\d+)\b", f"{prefix}iiwa_motor_\\1"),
    ]
    result = urdf_text
    for pat, repl in patterns:
        result = re.sub(pat, repl, result)
    return result


def build_dual_iiwa_urdf() -> str:
    """Build dual LBR iiwa 14 URDF from robot_descriptions.

    Returns:
        URDF string for a dual iiwa robot.
        Left arm at Y = +0.3 m, right arm at Y = -0.3 m, no rotation.

    Raises:
        ImportError: If robot_descriptions is not installed.
        FileNotFoundError: If iiwa14_description URDF is not found.
    """
    try:
        from robot_descriptions import iiwa14_description
    except ImportError as e:
        raise ImportError(
            "robot_descriptions is required for dual iiwa. "
            "Install with: pip install robot_descriptions"
        ) from e

    urdf_path = Path(iiwa14_description.URDF_PATH)
    if not urdf_path.exists():
        raise FileNotFoundError(
            f"iiwa14 URDF not found: {urdf_path}. "
            "Try: pip install robot_descriptions"
        )

    # Drake root: .../drake/manipulation/models/... → .../drake
    # urdf_path is .../drake/manipulation/models/iiwa_description/urdf/iiwa14_...urdf
    drake_root = urdf_path.parent.parent.parent.parent.parent

    with open(urdf_path, "r") as f:
        iiwa_urdf = f.read()

    # Extract body content: from iiwa_link_0 to the closing </robot> tag.
    # The original URDF already has a "base" link and "iiwa_base_joint", which we
    # replace with our own shared base and per-arm attachment joints.
    start = iiwa_urdf.find('<link name="iiwa_link_0">')
    end = iiwa_urdf.rfind("</robot>")
    iiwa_content = iiwa_urdf[start:end]

    # Resolve mesh paths before prefixing (package://drake/... → absolute)
    iiwa_content = _resolve_drake_package_paths(iiwa_content, drake_root)
    # Replace high-poly mesh collision shapes (links 6 & 7) with spheres.
    # Must run BEFORE path resolution is consumed — paths are still absolute here.
    iiwa_content = _replace_mesh_collision_with_primitives(iiwa_content)
    # Remove drake:* attributes so the composed URDF does not require xmlns:drake
    iiwa_content = _strip_drake_namespace_attributes(iiwa_content)

    left_content = _prefix_iiwa_content(iiwa_content, "iiwa_left_")
    right_content = _prefix_iiwa_content(iiwa_content, "iiwa_right_")

    # Shared base + fixed attachment joints.
    # MATLAB layout: left at Y=+0.3, right at Y=-0.3, no rotation.
    # Materials are defined globally here (not per-arm) to avoid duplicate-name
    # warnings from urdfdom. The iiwa body content references them by name.
    base_and_joints = """<robot name="dual_iiwa">
  <!-- KUKA iiwa material palette -->
  <material name="Black"><color rgba="0.0 0.0 0.0 1.0"/></material>
  <material name="Blue"><color rgba="0.0 0.0 0.8 1.0"/></material>
  <material name="Green"><color rgba="0.0 0.8 0.0 1.0"/></material>
  <material name="Grey"><color rgba="0.4 0.4 0.4 1.0"/></material>
  <material name="Silver"><color rgba="0.6 0.6 0.6 1.0"/></material>
  <material name="Orange"><color rgba="1.0 0.4235 0.0392 1.0"/></material>
  <material name="Brown"><color rgba="0.8706 0.8118 0.7647 1.0"/></material>
  <material name="Red"><color rgba="0.8 0.0 0.0 1.0"/></material>
  <material name="White"><color rgba="1.0 1.0 1.0 1.0"/></material>

  <link name="base">
    <visual>
      <origin rpy="0 0 0" xyz="0 0 -0.02"/>
      <geometry>
        <box size="0.8 0.8 0.04"/>
      </geometry>
      <material name="table_grey">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
  <joint name="iiwa_left_base_joint" type="fixed">
    <parent link="base"/>
    <child link="iiwa_left_iiwa_link_0"/>
    <origin rpy="0 0 0" xyz="0 0.3 0"/>
  </joint>
  <joint name="iiwa_right_base_joint" type="fixed">
    <parent link="base"/>
    <child link="iiwa_right_iiwa_link_0"/>
    <origin rpy="0 0 0" xyz="0 -0.3 0"/>
  </joint>
"""

    return base_and_joints + left_content + "\n  " + right_content + "\n</robot>"


def get_dual_iiwa_frame_names() -> tuple[str, str]:
    """Return (left_ee_frame, right_ee_frame) for ECTS tasks.

    Uses iiwa_link_7 (flange) as the end-effector frame, matching MATLAB.
    """
    return ("iiwa_left_iiwa_link_7", "iiwa_right_iiwa_link_7")


def get_dual_iiwa_joint_names() -> tuple[list[str], list[str]]:
    """Return (left_joint_names, right_joint_names) for the 7 revolute joints each."""
    left = [f"iiwa_left_iiwa_joint_{i}" for i in range(1, 8)]
    right = [f"iiwa_right_iiwa_joint_{i}" for i in range(1, 8)]
    return left, right


def get_dual_iiwa_default_configuration() -> list[float]:
    """Default joint configuration for dual iiwa (7 DOF per arm, 14 total).

    Matches MATLAB q_init = deg2rad([0 45 0 -90 0 45 0]') for both arms.
    """
    import math

    q_arm = [
        math.radians(0),
        math.radians(45),
        math.radians(0),
        math.radians(-90),
        math.radians(0),
        math.radians(45),
        math.radians(0),
    ]
    return q_arm + q_arm
