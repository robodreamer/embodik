"""Robot model presets loader and configuration utilities.

This module provides functions to load robot model configurations from
robot_presets.yaml, which can be shared across multiple examples.
"""

from pathlib import Path
from typing import Dict, Any
import yaml
import numpy as np
import logging

# Get the robot_models directory
_ROBOT_MODELS_DIR = Path(__file__).parent.parent / "robot_models"
_PRESETS_FILE = _ROBOT_MODELS_DIR / "robot_presets.yaml"

logger = logging.getLogger(__name__)


def load_robot_presets() -> Dict[str, Dict[str, Any]]:
    """Load robot presets from robot_presets.yaml.

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

    Raises:
        FileNotFoundError: If robot_presets.yaml is not found
        yaml.YAMLError: If the YAML file is invalid
    """
    if not _PRESETS_FILE.exists():
        raise FileNotFoundError(
            f"Robot presets file not found: {_PRESETS_FILE}\n"
            f"Expected at: {_PRESETS_FILE.relative_to(_ROBOT_MODELS_DIR.parent.parent)}"
        )

    with open(_PRESETS_FILE, 'r') as f:
        presets = yaml.safe_load(f)

    if presets is None:
        return {}

    # Convert default_configuration lists to numpy arrays
    for robot_key, config in presets.items():
        if 'default_configuration' in config:
            config['default_configuration'] = np.array(config['default_configuration'])
        if 'extra_gripper_default' in config:
            config['extra_gripper_default'] = np.array(config['extra_gripper_default'])

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


def resolve_robot_configuration(robot_key: str) -> Dict[str, Any]:
    """Resolve robot configuration from presets and load the robot model.

    This function loads the robot model from the URDF file specified in the preset,
    handles default configurations, and returns a complete configuration dictionary
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
        ValueError: If robot_key is not found in presets
        FileNotFoundError: If URDF file is not found
    """
    try:
        import embodik
    except ImportError:
        raise ImportError(
            "embodik package is required. Install it with: pip install -e ."
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

    # Resolve URDF path (relative to examples/ directory)
    urdf_path_str = preset.get("urdf_path")
    if not urdf_path_str:
        raise ValueError(f"Robot preset '{robot_key}' missing 'urdf_path' in robot_presets.yaml")

    # Get examples directory (parent of robot_models/)
    examples_dir = _ROBOT_MODELS_DIR.parent
    urdf_path = examples_dir / urdf_path_str

    if not urdf_path.exists():
        raise FileNotFoundError(
            f"URDF file not found: {urdf_path}\n"
            f"Expected at: {urdf_path_str} (relative to examples/ directory)\n"
            f"See examples/robot_models/README.md for instructions on adding robot models."
        )

    logger.info(f"Loading robot model from: {urdf_path}")
    robot = embodik.RobotModel(str(urdf_path))

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

