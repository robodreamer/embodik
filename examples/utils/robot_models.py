"""Robot model presets loader and configuration utilities.

This module provides functions to load robot model configurations from
robot_presets.yaml, which can be shared across multiple examples.
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import logging
import os

# Get the robot_models directory
# Use resolve() to ensure we get absolute paths and handle symlinks correctly
_ROBOT_MODELS_DIR = Path(__file__).resolve().parent.parent / "robot_models"
_PRESETS_FILE = _ROBOT_MODELS_DIR / "robot_presets.yaml"

logger = logging.getLogger(__name__)

# Built-in presets so pip-installed examples work even if `robot_models/robot_presets.yaml`
# is not present (or you don't want to ship model assets). These use `robot_descriptions`,
# which downloads/caches models on demand.
#
# Users can still override/extend these by providing `robot_models/robot_presets.yaml`.
_BUILTIN_PRESETS: Dict[str, Dict[str, Any]] = {
    "panda": {
        "description_name": "panda_description",
        "urdf_import": "robot_descriptions.panda_description",
        "urdf_attr": "URDF_PATH",
        "target_link": "panda_hand",
        "display_name": "Franka Emika Panda",
        "default_configuration": np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
        "extra_gripper_default": np.array([0.05, 0.05]),
        "default_offset": np.array([0.05, 0.0, 0.0]),
        "collision_exclusions": "auto",
        "collision_exclusion_overrides": [],
    },
    "iiwa": {
        "description_name": "iiwa14_description",
        "urdf_import": "robot_descriptions.iiwa14_description",
        "urdf_attr": "URDF_PATH",
        "target_link": "iiwa_link_7",
        "display_name": "KUKA LBR iiwa14",
        "default_configuration": np.array([0.0, 0.7854, 0.0, -1.5708, 0.0, 0.7854, 0.0]),
        "default_offset": np.array([0.05, 0.0, 0.0]),
        "collision_exclusions": "auto",
        "collision_exclusion_overrides": [],
    },
}


def load_robot_presets() -> Dict[str, Dict[str, Any]]:
    """Load robot presets.

    Returns:
        Dictionary mapping robot keys to their configuration dictionaries.
        Each configuration includes:
        - description_name: Robot description name
        - urdf_path: Path to URDF file (relative to examples/ directory)
        - target_link: Target link name for IK
        - display_name: Display name for the robot
        - default_configuration: Default joint configuration (numpy array)
        - extra_gripper_default: Optional gripper joint defaults (for robots with grippers)

    Note: Joint labels are automatically extracted from the URDF model, so they don't
    need to be specified in the YAML file.

    Behavior:
        - If `examples/robot_models/robot_presets.yaml` exists, it is loaded and used
          to override/extend the built-in presets.
        - If it does not exist, built-in presets are returned (robot_descriptions-based).
    """
    presets: Dict[str, Dict[str, Any]] = {k: dict(v) for k, v in _BUILTIN_PRESETS.items()}

    if _PRESETS_FILE.exists():
        try:
            import yaml  # optional dependency; only needed when using YAML presets
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency: PyYAML.\n"
                "You have `robot_models/robot_presets.yaml`, so YAML parsing is required.\n"
                "Install it via one of:\n"
                "  - pip install pyyaml\n"
                "  - pip install 'embodik[examples]'\n"
            ) from e

        with open(_PRESETS_FILE, "r") as f:
            loaded = yaml.safe_load(f)

        if loaded:
            # YAML overrides/extends built-ins
            presets.update(loaded)

    # Convert numeric lists to numpy arrays
    for robot_key, config in presets.items():
        if 'default_configuration' in config:
            config['default_configuration'] = np.array(config['default_configuration'])
        if 'extra_gripper_default' in config:
            config['extra_gripper_default'] = np.array(config['extra_gripper_default'])
        if 'default_offset' in config:
            config['default_offset'] = np.array(config['default_offset'])

    return presets


def get_robot_preset(robot_key: str) -> Dict[str, Any]:
    """Get a specific robot preset by key.

    Args:
        robot_key: Robot key (e.g., "panda", "iiwa")

    Returns:
        Robot configuration dictionary

    Raises:
        KeyError: If robot_key is not found in presets
    """
    presets = load_robot_presets()
    robot_key = robot_key.lower()

    if robot_key not in presets:
        available = sorted(presets.keys())
        raise KeyError(
            f"Robot preset '{robot_key}' not found. "
            f"Available options: {available}"
        )

    return presets[robot_key]


def ensure_ros_package_path(urdf_path: Path) -> None:
    """Ensure ROS_PACKAGE_PATH includes ancestors that contain meshes.

    This function sets up the ROS_PACKAGE_PATH environment variable to help
    resolve package:// URIs in URDF files. It looks for common package root
    directories in the URDF path's ancestors.

    Args:
        urdf_path: Path to the URDF file
    """
    resolved = urdf_path.resolve()
    candidate_roots: list[Path] = []
    for depth in range(1, 5):
        if len(resolved.parents) > depth:
            candidate_roots.append(resolved.parents[depth])

    current = os.environ.get("ROS_PACKAGE_PATH", "")
    paths = [Path(p) for p in current.split(":") if p]
    updated = False

    for candidate in candidate_roots:
        if candidate.exists() and candidate not in paths:
            paths.append(candidate)
            updated = True

    if updated:
        os.environ["ROS_PACKAGE_PATH"] = ":".join(str(p) for p in paths)


def generate_joint_labels_from_names(joint_names: list[str], robot_key: str = "") -> list[str]:
    """Generate joint labels automatically from joint names.

    Simply uses the original joint names from the URDF as labels.
    This preserves the original naming convention from the robot description.

    Args:
        joint_names: List of joint names from the robot model
        robot_key: Optional robot key (unused, kept for API compatibility)

    Returns:
        List of joint labels (same as joint names)
    """
    # Use joint names directly as labels
    return list(joint_names)


def resolve_robot_configuration(robot_key: str) -> Dict[str, Any]:
    """Resolve robot configuration from presets and load the robot model.

    This function loads the robot model from either:
    1. Local URDF file (if urdf_path is specified)
    2. robot_descriptions package (if urdf_import and urdf_attr are specified)

    It handles default configurations and returns a complete configuration dictionary
    ready for use in examples.

    Args:
        robot_key: Robot key from presets (e.g., "panda", "iiwa")

    Returns:
        Dictionary with robot configuration including:
        - robot: embodik.RobotModel instance
        - target_link: Target link name for IK
        - display_name: Display name for the robot
        - default_configuration: Default joint configuration (numpy array)
        - key: Robot key

    Note: Joint names are automatically extracted from the URDF model using
    robot.get_joint_names(), so they don't need to be specified in the preset.

    Raises:
        ValueError: If robot_key is not found in presets or configuration is invalid
        FileNotFoundError: If local URDF file is not found
        ImportError: If robot_descriptions package is not available
    """
    try:
        import embodik
    except ImportError:
        raise ImportError(
            "embodik package is required. Install it with: pip install -e ."
        )

    # Check if C++ extension is available
    if not hasattr(embodik, 'RobotModel'):
        cpp_available = getattr(embodik, '_cpp_extension_available', False)
        if not cpp_available:
            import sys
            cpp_err = getattr(embodik, "_cpp_extension_error", None)
            raise ImportError(
                "embodik C++ extension is not available. RobotModel cannot be used.\n\n"
                f"Debug info:\n"
                f"  - sys.executable: {sys.executable}\n"
                f"  - embodik.__file__: {getattr(embodik, '__file__', '(unknown)')}\n"
                f"  - C++ import error: {cpp_err or '(unknown)'}\n\n"
                "How to fix:\n"
                "  - If you installed from PyPI, you may be on a platform without a prebuilt wheel.\n"
                "    Try upgrading pip and reinstalling:\n"
                "      pip install -U pip\n"
                "      pip install --force-reinstall --no-cache-dir embodik\n\n"
                "  - If you are building from source:\n"
                "      pip install -e .\n"
                "    (You will need system deps like CMake/Eigen and a working Pinocchio install.)\n\n"
                "See docs/installation.md for detailed instructions."
            )
        else:
            raise AttributeError(
                "embodik.RobotModel is not available even though C++ extension is marked as available.\n"
                "This may indicate a build or installation issue. Please rebuild the package."
            )

    robot_key = robot_key.lower()
    presets = load_robot_presets()

    if robot_key not in presets:
        available = sorted(presets.keys())
        raise ValueError(
            f"Unsupported robot '{robot_key}'. "
            f"Available options: {available}\n"
            f"See examples/robot_models/README.md for instructions on adding new robots."
        )

    preset = presets[robot_key]

    # Resolve URDF path - support both local files and robot_descriptions
    urdf_path = None

    # Priority 1: Check for local urdf_path (for backward compatibility and custom models)
    urdf_path_str = preset.get("urdf_path")
    if urdf_path_str:
        examples_dir = _ROBOT_MODELS_DIR.parent
        urdf_path = examples_dir / urdf_path_str
        if not urdf_path.exists():
            raise FileNotFoundError(
                f"URDF file not found: {urdf_path}\n"
                f"Expected at: {urdf_path_str} (relative to examples/ directory)\n"
                f"See examples/robot_models/README.md for instructions on adding robot models."
            )
        logger.info(f"Loading robot model from local file: {urdf_path}")

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
            urdf_path = Path(getattr(module, urdf_attr))
        except ImportError as exc:
            raise ImportError(
                f"Robot description package '{urdf_import}' is required for the '{robot_key}' model.\n"
                "Install the 'robot_descriptions' package to use this example:\n"
                "  pip install robot_descriptions\n"
                "Or install with examples dependencies:\n"
                "  pip install embodik[examples]"
            ) from exc
        except AttributeError as exc:
            raise ValueError(
                f"Robot description module '{urdf_import}' does not have attribute '{urdf_attr}'.\n"
                f"Available attributes: {dir(module)}"
            ) from exc

        if not urdf_path.exists():
            raise FileNotFoundError(
                f"URDF file from robot_descriptions not found: {urdf_path}\n"
                f"This may indicate a caching issue. Try clearing ~/.cache/robot_descriptions/"
            )

        logger.info(f"Loading robot model from robot_descriptions: {urdf_path}")

    # Ensure ROS_PACKAGE_PATH is set up for package:// URI resolution
    ensure_ros_package_path(urdf_path)

    # Load robot model (use floating_base=False to match example 02 behavior)
    robot = embodik.RobotModel(str(urdf_path), floating_base=False)

    # Handle default configuration
    q_default = preset.get("default_configuration", np.zeros(robot.nq))
    if isinstance(q_default, list):
        q_default = np.array(q_default)

    # Handle gripper joints for panda (if robot has 9 DOF)
    if robot_key == "panda" and robot.nq == 9:
        extra_gripper = preset.get("extra_gripper_default", np.array([0.05, 0.05]))
        if isinstance(extra_gripper, list):
            extra_gripper = np.array(extra_gripper)
        q_default = np.concatenate([q_default, extra_gripper])

    return {
        "robot": robot,
        "target_link": preset.get("target_link", "end_effector"),
        "display_name": preset.get("display_name", robot_key),
        "default_configuration": q_default,
        "key": robot_key,
    }


def resolve_robot_configuration_with_labels(robot_key: str) -> Dict[str, Any]:
    """Resolve robot configuration including auto-generated joint labels.

    This is a convenience wrapper around resolve_robot_configuration that
    automatically generates joint_labels from joint names if not specified.

    Args:
        robot_key: Robot key from presets (e.g., "panda", "iiwa")

    Returns:
        Dictionary with robot configuration including joint_labels
    """
    config = resolve_robot_configuration(robot_key)

    # Auto-generate joint labels if not in preset
    preset = get_robot_preset(robot_key)
    if "joint_labels" not in preset or not preset.get("joint_labels"):
        robot = config["robot"]
        joint_names = robot.get_joint_names()
        config["joint_labels"] = generate_joint_labels_from_names(joint_names, robot_key)

    return config

