"""Shared robot presets and helpers for reachability demos."""

from __future__ import annotations

import atexit
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yourdfpy

import embodik

LOGGER = logging.getLogger("embodik.reachability_robot_configs")

_LINK_INDEX_PATTERN = re.compile(r"link_?([0-9]+)")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROBOT_MODELS_DIR = PROJECT_ROOT / "robot_models_urdf"
_URDF_PREPROCESS_CACHE: Dict[Path, Path] = {}
_URDF_TEMP_FILES: List[Path] = []


def _extract_link_index(name: str) -> Optional[int]:
    match = _LINK_INDEX_PATTERN.search(name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
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


_XML_DECL_PATTERN = re.compile(r'^\s*<\?xml[^>]*\?>\s*', re.IGNORECASE)
_XML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
_MOBILE_TAG_PATTERN = re.compile(r"<\s*mobile\b[^>]*>", re.IGNORECASE)
_CAPSULE_PATTERN = re.compile(r"<\s*capsule\b([^>/]*)(\/?)>", re.IGNORECASE)


def _extract_attr(attrs: str, name: str) -> Optional[str]:
    match = re.search(rf'{name}\s*=\s*["\']([^"\']+)["\']', attrs, re.IGNORECASE)
    return match.group(1) if match else None


def _preprocess_urdf_for_swiftik(path: Path) -> Path:
    canonical = path.resolve()
    cached = _URDF_PREPROCESS_CACHE.get(canonical)
    if cached is not None:
        return cached

    try:
        text = canonical.read_text(encoding="utf-8")
    except Exception as exc:
        LOGGER.warning("Failed to read URDF %s: %s", canonical, exc)
        _URDF_PREPROCESS_CACHE[canonical] = canonical
        return canonical

    changed = False

    new_text, count = _XML_DECL_PATTERN.subn("", text, count=1)
    if count:
        changed = True
    else:
        new_text = text

    def _capsule_repl(match: re.Match) -> str:
        nonlocal changed
        attrs = match.group(1)
        radius = _extract_attr(attrs, "radius")
        length = _extract_attr(attrs, "length")
        if not radius:
            LOGGER.warning("Capsule missing radius attribute in %s; leaving as-is.", canonical.name)
            return match.group(0)
        if not length:
            changed = True
            return f'<sphere radius="{radius}"/>'
        changed = True
        if match.group(2) == "/":
            return f'<cylinder radius="{radius}" length="{length}"/>'
        return f'<cylinder radius="{radius}" length="{length}"></cylinder>'

    new_text = _XML_COMMENT_PATTERN.sub("", new_text)
    new_text = _MOBILE_TAG_PATTERN.sub("", new_text)
    new_text = _CAPSULE_PATTERN.sub(_capsule_repl, new_text)

    if not changed:
        _URDF_PREPROCESS_CACHE[canonical] = canonical
        return canonical

    fd, temp_path_str = tempfile.mkstemp(prefix=f"{canonical.stem}_", suffix=".urdf")
    temp_path = Path(temp_path_str)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(new_text)
    _URDF_TEMP_FILES.append(temp_path)
    _URDF_PREPROCESS_CACHE[canonical] = temp_path
    LOGGER.info("Preprocessed URDF %s -> %s", canonical.name, temp_path.name)
    return temp_path


def get_processed_urdf_path(path: Path) -> Path:
    """Return a URDF path safe for urdfdom/pytorch_kinematics (capsules removed, no XML header)."""
    return _preprocess_urdf_for_swiftik(path)


def _cleanup_temp_urdfs() -> None:
    for temp in _URDF_TEMP_FILES:
        try:
            temp.unlink()
        except OSError:
            pass
    _URDF_TEMP_FILES.clear()


atexit.register(_cleanup_temp_urdfs)


def _format_joint_label(name: str) -> str:
    label = name
    if label.endswith("_joint"):
        label = label[: -len("_joint")]
    label = label.replace("_", " ").strip()
    return label.title() if label else name


def _infer_target_link(urdf_model: yourdfpy.URDF) -> str:
    link_names = list(urdf_model.link_map.keys())
    if not link_names:
        return "tool0"
    priority_tokens = ("tool", "tcp", "ee", "gripper", "hand", "wrist", "end_effector")
    for token in priority_tokens:
        for link_name in link_names:
            if token in link_name.lower():
                return link_name
    return link_names[-1]


def _discover_local_robot_presets() -> Dict[str, Dict[str, object]]:
    presets: Dict[str, Dict[str, object]] = {}
    base_dir = LOCAL_ROBOT_MODELS_DIR
    if not base_dir.exists():
        return presets

    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        urdf_candidates = sorted(entry.glob("*.urdf"))
        if not urdf_candidates:
            continue
        urdf_path = urdf_candidates[0]
        processed_path = get_processed_urdf_path(urdf_path)
        try:
            robot_model = embodik.RobotModel(str(processed_path), floating_base=False)
        except Exception as exc:
            LOGGER.warning("Failed to load local robot model '%s': %s", entry.name, exc)
            continue
        joint_names = list(robot_model.get_joint_names())
        if not joint_names:
            LOGGER.warning("Local robot '%s' has no actuated joints; skipping", entry.name)
            continue
        try:
            urdf_model = yourdfpy.URDF.load(str(processed_path), load_meshes=False)
        except Exception as exc:
            LOGGER.warning("Failed to parse URDF for '%s': %s", entry.name, exc)
            continue

        target_link = _infer_target_link(urdf_model)
        joint_labels = [_format_joint_label(name) for name in joint_names]
        default_configuration = np.zeros(len(joint_names), dtype=float)

        presets[entry.name.lower()] = {
            "display_name": entry.name,
            "description_name": entry.name,
            "urdf_path": urdf_path,
            "target_link": target_link,
            "joint_labels": joint_labels,
            "joint_names": joint_names,
            "default_configuration": default_configuration,
            "default_offset": np.zeros(3, dtype=float),
            "collision_exclusions": [],
            "metadata": {
                "source": "robot_models_urdf",
                "folder": str(entry.relative_to(PROJECT_ROOT)),
            },
        }

    if presets:
        LOGGER.info("Discovered %d local robot model(s) in %s", len(presets), base_dir)
    return presets


ROBOT_PRESETS: Dict[str, Dict[str, object]] = {
    "panda": {
        "description_name": "panda_description",
        "urdf_import": "robot_descriptions.panda_description",
        "urdf_attr": "URDF_PATH",
        "target_link": "panda_hand",
        "joint_labels": ["J1", "J2", "J3", "J4", "J5", "J6", "J7"],
        "joint_names": [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ],
        "display_name": "Franka Emika Panda",
        "default_configuration": np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
        "default_offset": np.array([0.05, 0.0, 0.0]),
        "collision_exclusions": "auto",
        "collision_exclusion_overrides": [],
    },
    "iiwa": {
        "description_name": "iiwa14_description",
        "urdf_import": "robot_descriptions.iiwa14_description",
        "urdf_attr": "URDF_PATH",
        "target_link": "iiwa_link_7",
        "joint_labels": ["A1", "A2", "A3", "A4", "A5", "A6", "A7"],
        "joint_names": [
            "iiwa_joint_1",
            "iiwa_joint_2",
            "iiwa_joint_3",
            "iiwa_joint_4",
            "iiwa_joint_5",
            "iiwa_joint_6",
            "iiwa_joint_7",
        ],
        "display_name": "KUKA LBR iiwa14",
        "default_configuration": np.array([0.0, 0.7854, 0.0, -1.5708, 0.0, 0.7854, 0.0]),
        "default_offset": np.array([0.05, 0.0, 0.0]),
        "collision_exclusions": "auto",
        "collision_exclusion_overrides": [],
    },
}

# Merge auto-discovered local models (if any)
ROBOT_PRESETS.update(_discover_local_robot_presets())


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
    limit_overrides: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def loadable_urdf_path(self) -> Path:
        return get_processed_urdf_path(self.urdf_path)


def resolve_robot_configuration(robot_key: str) -> RobotConfig:
    robot_key = robot_key.lower()
    if robot_key not in ROBOT_PRESETS:
        raise ValueError(f"Unsupported robot '{robot_key}'. Available options: {sorted(ROBOT_PRESETS)}")

    preset = ROBOT_PRESETS[robot_key]

    if "urdf_path" in preset:
        urdf_path = Path(preset["urdf_path"]).expanduser()
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF path for robot '{robot_key}' not found: {urdf_path}")
    else:
        try:
            module = __import__(preset["urdf_import"], fromlist=[preset["urdf_attr"]])
            urdf_path = Path(getattr(module, preset["urdf_attr"]))  # type: ignore[arg-type]
        except ImportError as exc:
            raise RuntimeError(
                f"Robot description package '{preset['urdf_import']}' is required for the '{robot_key}' model. "
                "Install the 'robot_descriptions' package to use this example."
            ) from exc
    processed_urdf_path = get_processed_urdf_path(urdf_path)

    raw_exclusions = preset.get("collision_exclusions", [])
    auto_collision = False
    if raw_exclusions == "auto":
        auto_collision = True
        overrides = [tuple(pair) for pair in preset.get("collision_exclusion_overrides", [])]
        ensure_ros_package_path(urdf_path)
        temp_robot = embodik.RobotModel(str(processed_urdf_path), floating_base=False)
        auto_list = generate_auto_collision_exclusions(temp_robot, robot_key)
        collision_exclusions = auto_list + overrides
    else:
        collision_exclusions = [tuple(pair) for pair in raw_exclusions]  # type: ignore[arg-type]
        ensure_ros_package_path(urdf_path)

    return RobotConfig(
        key=robot_key,
        display_name=preset["display_name"],  # type: ignore[arg-type]
        urdf_path=urdf_path,
        description_name=preset["description_name"],  # type: ignore[arg-type]
        target_link=preset["target_link"],  # type: ignore[arg-type]
        joint_labels=list(preset["joint_labels"]),  # type: ignore[arg-type]
        joint_names=list(preset["joint_names"]),  # type: ignore[arg-type]
        default_configuration=np.array(preset["default_configuration"], dtype=float),
        default_offset=np.array(preset["default_offset"], dtype=float),
        collision_exclusions=collision_exclusions,
        metadata=dict(preset.get("metadata", {})),
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
        if root not in paths:
            paths.append(root)
            updated = True

    if updated:
        os.environ["ROS_PACKAGE_PATH"] = ":".join(str(p) for p in paths)
