"""Optional EmbodiK whole-body IK overlay for Spot examples.

EmbodiK consumes URDF models for IK, while the mjviser rollout uses MJCF for
MuJoCo. The examples therefore bundle a Spot-with-arm URDF alongside the MuJoCo
MJCF asset so both regular Viser and mjviser examples work without custom model
paths.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np

from .ik_common import DEFAULT_COLLISION_TUNING_MODE, apply_collision_tuning_mode
from .robust_ik_runtime import configure_primary_solve_mode, robust_solve_position_step
from .spot_locomanip_policy import (
    DEFAULT_ARM_COMMAND,
    DEFAULT_BODY_ROLL_PITCH_HEIGHT,
    INITIAL_ARM_COMMAND,
    MJCF_ARM_JOINT_NAMES,
    MJCF_GRIPPER_JOINT_NAME,
    DEFAULT_STAND_BASE_HEIGHT,
    DEFAULT_STAND_LEG_JOINTS,
    SPOT_LEG_JOINT_NAMES,
    normalize_mjcf_joint_name,
)

FOOT_FRAME_CANDIDATES: tuple[str, ...] = (
    "front_left_foot_center",
    "front_right_foot_center",
    "rear_left_foot_center",
    "rear_right_foot_center",
    "FL",
    "FR",
    "HL",
    "HR",
)
BODY_FRAME_CANDIDATES: tuple[str, ...] = ("body", "base_link", "torso")
TOOL_FRAME_CANDIDATES: tuple[str, ...] = (
    # Prefer the wrist frame over the finger frame so opening/closing the
    # gripper does not rotate the IK task frame and back-drive wrist pitch.
    "arm0_link_wr1",
    "arm_link_wr1",
    "arm0_link_fngr",
    "arm_link_fngr",
    "EE",
    "tool",
    "hand",
)
SPOT_URDF_ENV_VAR = "EMBODIK_SPOT_IK_URDF"
_LOCAL_SPOT_URDF_FALLBACKS: tuple[Path, ...] = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "spot_description"
    / "urdf"
    / "spot_with_arm.urdf",
    Path(
        "/path/to/local/backup/backup-x13/Projects/other-bdai-repos/"
        "bdai-data-lfs/bdai/spot_isaac_sim_assets/urdf/spot_whole_body.urdf"
    ),
)
_SPOT_URDF_NAME_PREFERENCE: tuple[str, ...] = (
    "spot_whole_body.urdf",
    "spot_with_arm.urdf",
    "spot_arm.urdf",
    "spot.urdf",
)
_LOCO_TORSO_ROLL_PITCH_HALF_RANGE = np.deg2rad(15.0)
_LOCO_TORSO_YAW_HALF_RANGE = np.deg2rad(60.0)
_LOCO_TORSO_POSE_HALF_RANGE = np.array(
    [
        1.5,
        1.5,
        0.15,
        _LOCO_TORSO_ROLL_PITCH_HALF_RANGE,
        _LOCO_TORSO_ROLL_PITCH_HALF_RANGE,
        _LOCO_TORSO_YAW_HALF_RANGE,
    ],
    dtype=float,
)
_ARM_TORSO_POSE_AXIS_MASK = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0], dtype=float)
_LOCO_NULLSPACE_BASE_GAINS = np.array([1e-4, 1e-4, 4.0, 5.0, 5.0, 1e-4], dtype=float)
_LOCO_ARM_BIASED_BASE_GAINS = np.array([10.0, 10.0, 4.0, 5.0, 5.0, 10.0], dtype=float)
_LOCO_NULLSPACE_ARM_GAINS = np.array([3.0, 0.2, 0.2, 1.0, 0.2, 3.0], dtype=float)
_LOCO_NULLSPACE_GAIN = 1.0
_LOCO_CONDITION_HANDOFF_ENTER = 150.0
_LOCO_CONDITION_HANDOFF_EXIT = 75.0
_LOCO_CONDITION_ARM_WEIGHT_SCALE_DEFAULT = 3.0
_LOCO_CONDITION_RETRY_LINEAR_EPS = 1e-5
_LOCO_CONDITION_RETRY_ANGULAR_EPS = 1e-4
_LOCO_COLLISION_TUNING_MODE = "balanced"
_LOCO_BASE_VEL_MAX = 0.5
_LOCO_BASE_ACC_MAX = 0.5
_LOCO_MAX_LINEAR_SPEED = 0.0
_LOCO_MAX_ANGULAR_SPEED = 0.0
_LOCO_MAX_TANGENT_STEP = 0.8
_FULL_BODY_NULLSPACE_BASE_GAINS = np.array([4.0, 4.0, 4.0, 5.0, 5.0, 5.0], dtype=float)
_FULL_BODY_NULLSPACE_ARM_GAINS = np.array([3.0, 1e-6, 1e-6, 3.0, 1e-6, 3.0], dtype=float)
SPOT_COLLISION_MIN_DISTANCE_M = 0.05
SPOT_COLLISION_PAIR_REFERENCES: tuple[tuple[str, str], ...] = (
    ("arm0_link_wr0_0", "body_0"),
    ("arm0_link_el0_0", "body_0"),
    ("arm0_link_el1_0", "body_0"),
    ("arm0_link_wr1_0", "body_0"),
    ("arm0_link_fngr_0", "body_0"),
    ("arm0_link_fngr_1", "body_0"),
    ("arm0_link_fngr_2", "body_0"),
    ("arm0_link_fngr_3", "body_0"),
    ("arm0_link_fngr_4", "body_0"),
    ("arm0_link_fngr_5", "body_0"),
    ("arm0_link_jaw_0", "body_0"),
    ("arm0_link_jaw_1", "body_0"),
    ("arm0_link_jaw_2", "body_0"),
    ("arm0_link_el0_0", "body_1"),
    ("arm0_link_el1_0", "body_1"),
    ("arm0_link_el1_1", "body_1"),
    ("arm0_link_wr0_0", "body_1"),
    ("arm0_link_wr1_0", "body_1"),
    ("arm0_link_fngr_0", "body_1"),
    ("arm0_link_fngr_1", "body_1"),
    ("arm0_link_fngr_2", "body_1"),
    ("arm0_link_fngr_3", "body_1"),
    ("arm0_link_fngr_4", "body_1"),
    ("arm0_link_fngr_5", "body_1"),
    ("arm0_link_jaw_0", "body_1"),
    ("arm0_link_jaw_1", "body_1"),
    ("arm0_link_jaw_2", "body_1"),
    ("arm0_link_el0_0", "body_2"),
    ("arm0_link_el1_0", "body_2"),
    ("arm0_link_el1_1", "body_2"),
    ("arm0_link_wr0_0", "body_2"),
    ("arm0_link_wr1_0", "body_2"),
    ("arm0_link_fngr_0", "body_2"),
    ("arm0_link_fngr_1", "body_2"),
    ("arm0_link_fngr_2", "body_2"),
    ("arm0_link_fngr_3", "body_2"),
    ("arm0_link_fngr_4", "body_2"),
    ("arm0_link_fngr_5", "body_2"),
    ("arm0_link_jaw_0", "body_2"),
    ("arm0_link_jaw_1", "body_2"),
    ("arm0_link_jaw_2", "body_2"),
    ("arm0_link_wr0_0", "fl_uleg_0"),
    ("arm0_link_wr0_0", "fl_lleg_0"),
    ("arm0_link_wr0_0", "fr_uleg_0"),
    ("arm0_link_wr0_0", "fr_lleg_0"),
    ("arm0_link_el0_0", "fl_uleg_0"),
    ("arm0_link_el0_0", "fr_uleg_0"),
    ("arm0_link_el1_0", "fl_uleg_0"),
    ("arm0_link_el1_0", "fr_uleg_0"),
    ("arm0_link_wr0_0", "arm0_link_sh0_0"),
    ("arm0_link_el0_0", "arm0_link_sh0_0"),
    ("arm0_link_el1_0", "arm0_link_sh0_0"),
    ("arm0_link_sh1_0", "arm0_link_el1_0"),
    ("arm_link_wr0_0", "body_0"),
    ("arm_link_el0_0", "body_0"),
    ("arm_link_el1_0", "body_0"),
    ("arm_link_wr0_0", "front_left_upper_leg_0"),
    ("arm_link_wr0_0", "front_left_lower_leg_0"),
    ("arm_link_wr0_0", "front_right_upper_leg_0"),
    ("arm_link_wr0_0", "front_right_lower_leg_0"),
    ("arm_link_el0_0", "front_left_upper_leg_0"),
    ("arm_link_el0_0", "front_right_upper_leg_0"),
    ("arm_link_el1_0", "front_left_upper_leg_0"),
    ("arm_link_el1_0", "front_right_upper_leg_0"),
    ("arm_link_wr0_0", "arm_link_sh0_0"),
    ("arm_link_el0_0", "arm_link_sh0_0"),
)

_STANDARD_TORSO_ROLL_PITCH_YAW_HALF_RANGE = np.deg2rad(15.0)
STANDARD_FULL_BODY_TORSO_POSE_HALF_RANGE = np.array(
    [
        0.1,
        0.1,
        0.1,
        _STANDARD_TORSO_ROLL_PITCH_YAW_HALF_RANGE,
        _STANDARD_TORSO_ROLL_PITCH_YAW_HALF_RANGE,
        _STANDARD_TORSO_ROLL_PITCH_YAW_HALF_RANGE,
    ],
    dtype=float,
)
STANDARD_FULL_BODY_BASE_VEL_MAX = 0.7
STANDARD_FULL_BODY_BASE_ACC_MAX = 0.3
STANDARD_FULL_BODY_NULLSPACE_GAIN = 1.0

FOOT_FRAME_CANDIDATE_SETS: tuple[tuple[str, ...], ...] = (
    ("fl_foot", "front_left_foot_center", "front_left_foot", "FL"),
    ("fr_foot", "front_right_foot_center", "front_right_foot", "FR"),
    ("hl_foot", "rear_left_foot_center", "hind_left_foot", "rear_left_foot", "HL"),
    ("hr_foot", "rear_right_foot_center", "hind_right_foot", "rear_right_foot", "HR"),
)
_PUBLIC_SPOT_DESCRIPTION_JOINT_ALIASES: dict[str, tuple[str, ...]] = {
    "fl_hx": ("front_left_hip_x",),
    "fl_hy": ("front_left_hip_y",),
    "fl_kn": ("front_left_knee",),
    "fr_hx": ("front_right_hip_x",),
    "fr_hy": ("front_right_hip_y",),
    "fr_kn": ("front_right_knee",),
    "hl_hx": ("rear_left_hip_x",),
    "hl_hy": ("rear_left_hip_y",),
    "hl_kn": ("rear_left_knee",),
    "hr_hx": ("rear_right_hip_x",),
    "hr_hy": ("rear_right_hip_y",),
    "hr_kn": ("rear_right_knee",),
}


@dataclass
class SpotIKResult:
    success: bool
    targets: dict[str, float]
    arm_command: np.ndarray | None = None
    body_command: np.ndarray | None = None
    desired_pose_command: np.ndarray | None = None
    arm_gravity_torque: np.ndarray | None = None
    solve_time_ms: float = 0.0
    collision_constraint_time_ms: float = 0.0
    condition_number: float = 1.0
    position_error: float = 0.0
    orientation_error: float = 0.0
    status_name: str = ""
    message: str = ""


class SpotFullBodyIKMode(Enum):
    """Interactive solve modes for the regular-Viser Spot IK example."""

    ARM_TORSO = "arm-torso"
    TORSO_ONLY = "torso-only"
    FULL_BODY = "full-body"
    TWO_STAGE = "two-stage"


@dataclass
class SpotFullBodyIKConfig:
    dt: float = 0.01
    position_gain: float = 60.0
    orientation_gain: float = 60.0
    foot_position_gain: float = 120.0
    max_steps: int = 1
    damping: float = 0.1
    tolerance: float = 0.1
    adaptive_dt: bool = True
    adaptive_dt_max_scale: float = 3.0
    adaptive_dt_reference_distance: float = 0.04
    max_linear_speed: float = 1.8
    max_angular_speed: float = 2.5
    max_tangent_step: float = 0.8
    torso_pose_half_range: np.ndarray | None = None
    base_velocity_limit: float = STANDARD_FULL_BODY_BASE_VEL_MAX
    base_acceleration_limit: float = STANDARD_FULL_BODY_BASE_ACC_MAX
    nullspace_gain: float = STANDARD_FULL_BODY_NULLSPACE_GAIN
    use_contact_projection: bool = True
    target_solve_mode: str = "SCALE_ELASTIC"
    enable_collision: bool = True
    collision_min_distance: float = SPOT_COLLISION_MIN_DISTANCE_M
    collision_max_constraints: int = 3
    collision_tuning_mode: str = DEFAULT_COLLISION_TUNING_MODE


def _quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=float).reshape(4)
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)


def _quat_xyzw_to_rotation(quat_xyzw: np.ndarray) -> np.ndarray:
    from embodik import q2r

    return q2r(np.asarray(quat_xyzw, dtype=float), order="xyzs")


def _roll_pitch_from_rotation(rotation: np.ndarray) -> tuple[float, float]:
    rot = np.asarray(rotation, dtype=float).reshape(3, 3)
    roll = float(np.arctan2(rot[2, 1], rot[2, 2]))
    pitch = float(np.arctan2(-rot[2, 0], np.hypot(rot[2, 1], rot[2, 2])))
    return roll, pitch


def _yaw_from_rotation(rotation: np.ndarray) -> float:
    rot = np.asarray(rotation, dtype=float).reshape(3, 3)
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def _yaw_from_wxyz(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in np.asarray(quat_wxyz, dtype=float).reshape(4)]
    norm = float(np.linalg.norm([w, x, y, z]))
    if norm <= 1e-12:
        return 0.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(float(roll)), np.sin(float(roll))
    cp, sp = np.cos(float(pitch)), np.sin(float(pitch))
    cy, sy = np.cos(float(yaw)), np.sin(float(yaw))
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _rotation_to_xyzw(rotation: np.ndarray) -> np.ndarray:
    from embodik import r2q

    return np.asarray(r2q(np.asarray(rotation, dtype=float), order="xyzs"), dtype=float)


def _rotation_delta_rpy(seed_rotation: np.ndarray, command_rotation: np.ndarray) -> tuple[float, float, float]:
    delta = np.asarray(command_rotation, dtype=float) @ np.asarray(seed_rotation, dtype=float).T
    roll, pitch = _roll_pitch_from_rotation(delta)
    return roll, pitch, _yaw_from_rotation(delta)


def _pose_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.asarray(rotation, dtype=float)
    matrix[:3, 3] = np.asarray(translation, dtype=float)
    return matrix


def _pose_delta_exceeds(
    seed_rotation: np.ndarray,
    seed_translation: np.ndarray,
    target_rotation: np.ndarray,
    target_translation: np.ndarray,
    *,
    linear_eps: float,
    angular_eps: float,
) -> bool:
    linear_delta = float(
        np.linalg.norm(
            np.asarray(target_translation, dtype=float) - np.asarray(seed_translation, dtype=float)
        )
    )
    if linear_delta > linear_eps:
        return True
    rotation_delta = np.asarray(target_rotation, dtype=float) @ np.asarray(seed_rotation, dtype=float).T
    cos_angle = float(np.clip((np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cos_angle)) > angular_eps


def _commanded_base_pose_wxyz(body_command: np.ndarray, desired_pose_command: np.ndarray) -> np.ndarray:
    body = np.asarray(body_command, dtype=float)
    desired_pose = np.asarray(desired_pose_command, dtype=float)
    quat_xyzw = _rotation_to_xyzw(_rotation_from_rpy(body[0], body[1], desired_pose[2]))
    return np.array(
        [
            desired_pose[0],
            desired_pose[1],
            body[2],
            quat_xyzw[3],
            quat_xyzw[0],
            quat_xyzw[1],
            quat_xyzw[2],
        ],
        dtype=float,
    )


def _select_existing_frame(robot, candidates: tuple[str, ...]) -> str:
    for frame in candidates:
        if robot.has_frame(frame):
            return frame
    raise RuntimeError(f"Spot URDF does not contain any expected frame: {candidates}")


def spot_collision_pairs_from_references(robot) -> list[tuple[str, str]]:
    """Return the curated Spot collision pairs that exist in the loaded model."""

    if not hasattr(robot, "get_collision_geometry_names") or not hasattr(
        robot, "get_collision_pair_names"
    ):
        return []
    geometry_names = {str(name) for name in robot.get_collision_geometry_names()}
    wanted = {
        frozenset((a, b))
        for a, b in SPOT_COLLISION_PAIR_REFERENCES
        if a in geometry_names and b in geometry_names
    }
    pairs: list[tuple[str, str]] = []
    for a, b in robot.get_collision_pair_names():
        key = frozenset((str(a), str(b)))
        if key in wanted:
            pairs.append((str(a), str(b)))
    return pairs


def _joint_name_candidates(mjcf_name: str) -> tuple[str, ...]:
    if mjcf_name.startswith("arm_"):
        suffix = mjcf_name.removeprefix("arm_")
        return (f"arm0_{suffix}", mjcf_name)
    return (mjcf_name, *_PUBLIC_SPOT_DESCRIPTION_JOINT_ALIASES.get(mjcf_name, ()))


def _existing_path(value) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_file() else None


def _first_preferred_urdf(root: Path) -> Path | None:
    if not root.exists():
        return None
    urdfs = [path for path in root.rglob("*.urdf") if path.is_file()]
    if not urdfs:
        return None
    by_name = {path.name: path for path in urdfs}
    for name in _SPOT_URDF_NAME_PREFERENCE:
        if name in by_name:
            return by_name[name]
    return sorted(urdfs, key=lambda path: (len(path.parts), str(path)))[0]


def _urdf_from_robot_descriptions_module(module_name: str) -> Path | None:
    try:
        module = importlib.import_module(f"robot_descriptions.{module_name}")
    except Exception:
        return None
    for attr in ("URDF_PATH", "XACRO_PATH"):
        if path := _existing_path(getattr(module, attr, None)):
            return path
    if package_path := getattr(module, "PACKAGE_PATH", None):
        return _first_preferred_urdf(Path(package_path).expanduser())
    return None


def _discover_spot_urdf_from_robot_descriptions() -> Path | None:
    """Return a Spot URDF exposed by robot_descriptions, if the installed set has one."""

    try:
        from robot_descriptions import DESCRIPTIONS
    except Exception:
        return None
    candidates: list[str] = []
    for name, description in DESCRIPTIONS.items():
        robot = getattr(description, "robot", "")
        has_urdf = bool(getattr(description, "has_urdf", False))
        if has_urdf and ("spot" in name.lower() or robot.lower() == "spot"):
            candidates.append(name)
    candidates.extend(
        name for name in ("spot_description", "spot_arm_description") if name not in candidates
    )
    for name in candidates:
        if path := _urdf_from_robot_descriptions_module(name):
            return path
    return None


def resolve_spot_ik_urdf(
    value: str | Path | None = None,
    *,
    include_local_fallbacks: bool = True,
) -> Path | None:
    """Resolve a compatible Spot whole-body URDF for the optional IK overlay."""

    for candidate in (value, os.environ.get(SPOT_URDF_ENV_VAR)):
        if path := _existing_path(candidate):
            return path
    if include_local_fallbacks:
        for candidate in _LOCAL_SPOT_URDF_FALLBACKS:
            if path := _existing_path(candidate):
                return path
    if path := _discover_spot_urdf_from_robot_descriptions():
        return path
    return None


class OptionalSpotWholeBodyIK:
    """Small adapter that keeps the example policy-first and URDF-optional."""

    def __init__(self, urdf_path: str | Path | None, *, dt: float = 0.01):
        self.enabled = urdf_path is not None
        self.message = "disabled: no Spot URDF supplied"
        self.robot = None
        self.solver = None
        self.arm_joint_names = MJCF_ARM_JOINT_NAMES
        self.tool_frame = ""
        self.body_frame = ""
        self._eik = None
        self._task_name = "spot_gripper_ik"
        self._posture_task_name = "spot_ik_posture"
        self._arm_joint_map: dict[str, str] = {}
        self._leg_joint_map: dict[str, str] = {}
        self._gripper_joint_name = ""
        self._arm_velocity_indices: list[int] = []
        self._posture_velocity_indices: list[int] = []
        self._reference_posture_velocity_indices: list[int] = []
        self._locomanip_posture_velocity_indices: list[int] = []
        self._q = np.array([], dtype=float)
        self._lower = np.array([], dtype=float)
        self._upper = np.array([], dtype=float)
        self._step_opts = None
        self._ik_opts = None
        self._task = None
        self._posture_task = None
        self._torso_bounds_reference_pose: np.ndarray | None = None
        self._torso_reference_translation: np.ndarray | None = None
        self._torso_reference_rotation: np.ndarray | None = None
        self._body_command_reference: np.ndarray | None = None
        self._desired_pose_command_reference: np.ndarray | None = None
        self._condition_locomotion_active = False
        self._posture_reference_q: np.ndarray | None = None
        self._locomotion_sensitivity = 1.0
        self._condition_arm_weight_scale = _LOCO_CONDITION_ARM_WEIGHT_SCALE_DEFAULT
        self.dt = float(dt)
        self.position_gain = 60.0
        self.orientation_gain = 60.0
        self.max_steps = 1
        self.adaptive_dt = True
        self.adaptive_dt_max_scale = 3.0
        self.adaptive_dt_reference_distance = 0.04
        self.target_solve_mode = "SCALE_ELASTIC"
        # Locomanipulation runs the policy/control loop continuously, so keep
        # collision constraints and debug evaluation opt-in from the UI.  The
        # regular full-body IK example keeps collision enabled by default via
        # SpotFullBodyIKConfig; this optional overlay must not add hidden
        # per-frame collision cost until requested.
        self.enable_collision = False
        # Keep locomanipulation collision avoidance useful by default while
        # the async IK loop caps request rate and drops stale requests.
        self.collision_min_distance = 0.02
        self.collision_max_constraints = 3
        self.collision_tuning_mode = _LOCO_COLLISION_TUNING_MODE
        self._collision_include_pairs: list[tuple[str, str]] = []
        self._collision_config_key: tuple[object, ...] | None = None
        if urdf_path is None:
            return

        try:
            import embodik as eik
        except ImportError as exc:
            self.enabled = False
            self.message = f"disabled: EmbodiK import failed ({exc})"
            return

        path = Path(urdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Spot URDF not found: {path}")

        self.robot = eik.RobotModel(str(path), floating_base=True)
        self.solver = eik.KinematicsSolver(self.robot)
        self.solver.dt = dt
        self.solver.set_damping(0.1)
        self.solver.set_tolerance(0.1)
        self._eik = eik
        self.tool_frame = _select_existing_frame(self.robot, TOOL_FRAME_CANDIDATES)
        self.body_frame = _select_existing_frame(self.robot, BODY_FRAME_CANDIDATES)
        self._arm_joint_map = self._resolve_joint_map(MJCF_ARM_JOINT_NAMES)
        self._leg_joint_map = self._resolve_joint_map(SPOT_LEG_JOINT_NAMES)
        self._gripper_joint_name = self._resolve_joint_name(MJCF_GRIPPER_JOINT_NAME, required=False)
        self._arm_velocity_indices = self._velocity_indices(tuple(self._arm_joint_map.values()))

        self._task = self.solver.add_frame_task(self._task_name, self.tool_frame)
        self._task.priority = 0
        self._task.weight = 1.0
        self._task.solve_mode = getattr(eik.TaskSolveMode, "SCALE_ELASTIC", eik.TaskSolveMode.SCALE)
        self._task.allow_min_error_fallback = False

        self._reference_posture_velocity_indices = [2, 3, 4, *self._arm_velocity_indices]
        self._locomanip_posture_velocity_indices = [0, 1, 2, 3, 4, 5, *self._arm_velocity_indices]
        self._posture_velocity_indices = self._reference_posture_velocity_indices.copy()
        self._posture_task = self.solver.add_posture_task(
            self._posture_task_name,
            self._posture_velocity_indices,
        )
        self._posture_task.priority = 1
        self._posture_task.weight = _LOCO_NULLSPACE_GAIN
        self._posture_task.set_controlled_joint_weights(
            np.concatenate((_LOCO_NULLSPACE_BASE_GAINS[[2, 3, 4]], _LOCO_NULLSPACE_ARM_GAINS))
        )
        self._posture_task.solve_mode = eik.TaskSolveMode.MIN_ERROR
        self._posture_task.allow_min_error_fallback = False

        self._step_opts = eik.PositionStepOptions()
        self._step_opts.position_gain = 60.0
        self._step_opts.orientation_gain = 60.0
        self._step_opts.max_steps = 1
        self._step_opts.adaptive_dt = True
        self._step_opts.adaptive_dt_max_scale = 3.0
        self._step_opts.adaptive_dt_reference_distance = 0.04
        self._step_opts.max_linear_speed = _LOCO_MAX_LINEAR_SPEED
        self._step_opts.max_angular_speed = _LOCO_MAX_ANGULAR_SPEED
        zeroed_velocity_indices = self._velocity_indices(
            (*self._leg_joint_map.values(), self._gripper_joint_name)
        )
        self._step_opts.excluded_joint_indices = zeroed_velocity_indices
        self._step_opts.integration_zero_velocity_indices = zeroed_velocity_indices
        self._step_opts.torso_constraint.enabled = True
        self._step_opts.torso_constraint.frame_name = self.body_frame
        self._step_opts.torso_constraint.pose_lower_bounds = -_LOCO_TORSO_POSE_HALF_RANGE
        self._step_opts.torso_constraint.pose_upper_bounds = _LOCO_TORSO_POSE_HALF_RANGE
        self._step_opts.torso_constraint.pose_axis_mask = _ARM_TORSO_POSE_AXIS_MASK.copy()
        self._step_opts.torso_constraint.velocity_limits = np.full(
            6, _LOCO_BASE_VEL_MAX, dtype=float
        )
        self._step_opts.torso_constraint.acceleration_limits = np.full(
            6, _LOCO_BASE_ACC_MAX, dtype=float
        )

        self._ik_opts = eik.PositionIKOptions()
        self._ik_opts.max_iterations = 1
        self._ik_opts.dt = dt
        self._ik_opts.position_gain = 60.0
        self._ik_opts.orientation_gain = 60.0
        self._ik_opts.primary_solve_mode = getattr(
            eik.TaskSolveMode,
            "SCALE_ELASTIC",
            eik.TaskSolveMode.SCALE,
        )
        self._ik_opts.primary_allow_min_error_fallback = False
        self._ik_opts.nullspace_gain = _LOCO_NULLSPACE_GAIN
        self._ik_opts.nullspace_active_joints = self._posture_velocity_indices
        self._ik_opts.nullspace_joint_weights = np.concatenate(
            (_LOCO_NULLSPACE_BASE_GAINS[[2, 3, 4]], _LOCO_NULLSPACE_ARM_GAINS)
        )
        self._ik_opts.torso_constraint.enabled = True
        self._ik_opts.torso_constraint.frame_name = self.body_frame
        self._ik_opts.torso_constraint.pose_lower_bounds = -_LOCO_TORSO_POSE_HALF_RANGE
        self._ik_opts.torso_constraint.pose_upper_bounds = _LOCO_TORSO_POSE_HALF_RANGE
        self._ik_opts.torso_constraint.pose_axis_mask = _ARM_TORSO_POSE_AXIS_MASK.copy()
        self._ik_opts.torso_constraint.velocity_limits = np.full(
            6, _LOCO_BASE_VEL_MAX, dtype=float
        )
        self._ik_opts.torso_constraint.acceleration_limits = np.full(
            6, _LOCO_BASE_ACC_MAX, dtype=float
        )

        self._q = np.asarray(self.robot.get_current_configuration(), dtype=float)
        self._lower, self._upper = [np.asarray(v, dtype=float) for v in self.robot.get_joint_limits()]
        self._collision_include_pairs = spot_collision_pairs_from_references(self.robot)
        self.message = f"ready: {self.tool_frame} IK target"

    def apply_runtime_options(self) -> None:
        """Apply UI-tunable IK options to the step and IK option objects."""

        if self.solver is not None:
            self.solver.dt = float(self.dt)
        if self._step_opts is not None:
            self._step_opts.position_gain = float(self.position_gain)
            self._step_opts.orientation_gain = float(self.orientation_gain)
            self._step_opts.max_steps = int(self.max_steps)
            self._step_opts.adaptive_dt = bool(self.adaptive_dt)
            self._step_opts.adaptive_dt_max_scale = float(self.adaptive_dt_max_scale)
            self._step_opts.adaptive_dt_reference_distance = float(
                self.adaptive_dt_reference_distance
            )
            if self._eik is not None:
                configure_primary_solve_mode(
                    self._step_opts,
                    getattr(
                        self._eik.TaskSolveMode,
                        str(self.target_solve_mode),
                        self._eik.TaskSolveMode.SCALE_ELASTIC,
                    ),
                    False,
                )
        if self._ik_opts is not None:
            self._ik_opts.dt = float(self.dt)
            self._ik_opts.position_gain = float(self.position_gain)
            self._ik_opts.orientation_gain = float(self.orientation_gain)
            self._ik_opts.max_iterations = int(self.max_steps)
            self._ik_opts.primary_solve_mode = getattr(
                self._eik.TaskSolveMode,
                str(self.target_solve_mode),
                self._eik.TaskSolveMode.SCALE_ELASTIC,
            )

    def reset_reference(self) -> None:
        self._torso_bounds_reference_pose = None
        self._torso_reference_translation = None
        self._torso_reference_rotation = None
        self._body_command_reference = None
        self._desired_pose_command_reference = None
        self._posture_reference_q = None
        self._condition_locomotion_active = False
        self._set_locomanip_posture_bias(condition_protect=False)
        self._collision_config_key = None

    @property
    def collision_include_pairs(self) -> list[tuple[str, str]]:
        return list(self._collision_include_pairs)

    @property
    def collision_available(self) -> bool:
        return bool(
            self.solver is not None
            and hasattr(self.solver, "configure_collision_constraint")
            and self._collision_include_pairs
        )

    @property
    def collision_debug_available(self) -> bool:
        return bool(
            self.solver is not None
            and (
                hasattr(self.solver, "evaluate_collision_debug")
                or hasattr(self.solver, "get_last_collision_debug_list")
                or hasattr(self.solver, "get_last_collision_debug")
            )
        )

    def _configure_collision_constraint(self) -> None:
        if self.solver is None or not hasattr(self.solver, "configure_collision_constraint"):
            return
        enabled = bool(self.enable_collision and self._collision_include_pairs)
        key: tuple[object, ...] = (
            enabled,
            round(float(self.collision_min_distance), 6),
            int(self.collision_max_constraints),
            str(self.collision_tuning_mode),
        )
        if key == self._collision_config_key:
            return
        if not enabled:
            if hasattr(self.solver, "clear_collision_constraint"):
                self.solver.clear_collision_constraint()
            self._collision_config_key = key
            return
        apply_collision_tuning_mode(self.solver, str(self.collision_tuning_mode))
        self.solver.configure_collision_constraint(
            min_distance=float(self.collision_min_distance),
            include_pairs=list(self._collision_include_pairs),
            exclude_pairs=[],
            nearest_points_all_pairs=False,
            max_constraints=int(self.collision_max_constraints),
        )
        self._collision_config_key = key

    def collision_debug_rows(self) -> list[object]:
        if self.solver is None or not self.enable_collision:
            return []
        debug_rows = []
        if hasattr(self.solver, "get_last_collision_debug_list"):
            try:
                debug_rows = list(self.solver.get_last_collision_debug_list())
            except Exception:
                debug_rows = []
        if not debug_rows and hasattr(self.solver, "get_last_collision_debug"):
            debug = self.solver.get_last_collision_debug()
            debug_rows = [] if debug is None else [debug]
        if not debug_rows and hasattr(self.solver, "evaluate_collision_debug") and self._q.size:
            try:
                debug = self.solver.evaluate_collision_debug(np.asarray(self._q, dtype=float))
            except Exception:
                debug = None
            debug_rows = [] if debug is None else [debug]
        return debug_rows

    def sync_measured_configuration_for_debug(
        self,
        observation: Mapping[str, np.ndarray],
        arm_command: np.ndarray,
    ) -> np.ndarray:
        """Sync ``self._q`` to the measured MuJoCo state for visual debug overlays.

        The locomanip IK solve can reason in a commanded torso frame so that the
        policy receives smooth torso/locomotion commands. Collision debug markers,
        however, should align with the current mjviser render, which is the
        measured MuJoCo state after physics integration.
        """

        if self.robot is None:
            return np.array([], dtype=float)
        q = self._sync_configuration(
            observation,
            np.asarray(arm_command, dtype=float),
            base_pose_override=None,
        )
        self._q = np.asarray(q, dtype=float).copy()
        return self._q

    def _resolve_joint_name(self, mjcf_name: str, *, required: bool = True) -> str:
        if self.robot is None:
            return ""
        for name in _joint_name_candidates(mjcf_name):
            if self.robot.has_joint(name):
                return name
        if required:
            raise RuntimeError(f"Spot URDF does not contain joint compatible with {mjcf_name!r}")
        return ""

    def _resolve_joint_map(self, mjcf_names: tuple[str, ...]) -> dict[str, str]:
        return {name: self._resolve_joint_name(name) for name in mjcf_names}

    def _velocity_indices(self, joint_names: tuple[str, ...]) -> list[int]:
        if self.robot is None:
            return []
        indices: list[int] = []
        for joint_name in joint_names:
            if not joint_name:
                continue
            start = int(self.robot.get_joint_velocity_index(joint_name))
            size = int(self.robot.get_joint_velocity_size(joint_name))
            indices.extend(range(start, start + size))
        return indices

    def _set_joint_position(self, q: np.ndarray, joint_name: str, value: float) -> None:
        if self.robot is None:
            return
        idx = int(self.robot.get_joint_config_index(joint_name))
        size = int(self.robot.get_joint_config_size(joint_name))
        if size == 1 and idx < q.size:
            q[idx] = float(value)

    def _read_joint_position(self, q: np.ndarray, joint_name: str) -> float:
        if self.robot is None:
            return 0.0
        idx = int(self.robot.get_joint_config_index(joint_name))
        return float(q[idx])

    def _sync_configuration(
        self,
        observation: Mapping[str, np.ndarray],
        arm_command: np.ndarray,
        *,
        base_pose_override: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.robot is None:
            return np.array([], dtype=float)
        q = np.asarray(self.robot.get_current_configuration(), dtype=float).copy()
        base_pose = np.asarray(
            observation["base_pose"] if base_pose_override is None else base_pose_override,
            dtype=float,
        )
        if q.size >= 7:
            q[:7] = np.array(
                [
                    base_pose[0],
                    base_pose[1],
                    base_pose[2],
                    base_pose[4],
                    base_pose[5],
                    base_pose[6],
                    base_pose[3],
                ],
                dtype=float,
            )
        leg_values = np.asarray(observation.get("joint_pos", DEFAULT_STAND_LEG_JOINTS), dtype=float)
        for mjcf_name, value in zip(SPOT_LEG_JOINT_NAMES, leg_values):
            self._set_joint_position(q, self._leg_joint_map[mjcf_name], float(value))
        for mjcf_name, value in zip(MJCF_ARM_JOINT_NAMES, arm_command[:6]):
            self._set_joint_position(q, self._arm_joint_map[mjcf_name], float(value))
        if self._gripper_joint_name:
            self._set_joint_position(q, self._gripper_joint_name, float(arm_command[6]))
        self.robot.update_configuration(q)
        return q

    def _neutral_torso_posture_target(
        self,
        q: np.ndarray,
        seed_torso_rotation: np.ndarray,
    ) -> np.ndarray:
        """Return a posture target that keeps locomotion x/y/yaw free.

        The locomanipulation controller should be able to translate in x/y and
        yaw the base, while the IK nullspace softly restores the torso height,
        roll, and pitch captured when the marker was armed.
        """

        q_target = np.asarray(q, dtype=float).copy()
        if q_target.size < 7 or self._body_command_reference is None:
            return q_target

        if self._posture_reference_q is not None:
            q_reference = np.asarray(self._posture_reference_q, dtype=float)
            for velocity_index in self._arm_velocity_indices:
                config_index = int(velocity_index) + 1
                if config_index < q_target.size and config_index < q_reference.size:
                    q_target[config_index] = q_reference[config_index]

        neutral_body = np.asarray(self._body_command_reference, dtype=float)
        yaw = _yaw_from_rotation(seed_torso_rotation)
        neutral_rotation_xyzw = _rotation_to_xyzw(
            _rotation_from_rpy(neutral_body[0], neutral_body[1], yaw)
        )
        q_target[2] = float(neutral_body[2])
        q_target[3:7] = neutral_rotation_xyzw
        return q_target

    def _normalize_and_clip(self, q: np.ndarray) -> np.ndarray:
        out = np.asarray(q, dtype=float).copy()
        if out.size >= 7:
            quat = out[3:7]
            norm = float(np.linalg.norm(quat))
            if norm > 1e-12:
                out[3:7] = quat / norm
        if out.size > 7:
            out[7:] = np.clip(out[7:], self._lower[7:], self._upper[7:])
        return out

    def _limit_tangent_step(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        if self.robot is None:
            return np.asarray(q_to, dtype=float)
        max_component = _LOCO_MAX_TANGENT_STEP
        if max_component <= 0.0:
            return np.asarray(q_to, dtype=float)
        q_step = np.asarray(self.robot.difference(q_from, q_to), dtype=float)
        step_component = float(np.max(np.abs(q_step))) if q_step.size else 0.0
        if step_component <= max_component or step_component <= 1e-12:
            return np.asarray(q_to, dtype=float)
        return np.asarray(
            self.robot.integrate(q_from, q_step * (max_component / step_component), 1.0),
            dtype=float,
        )

    def _solve_position_step_guarded(
        self,
        q: np.ndarray,
        target_matrix: np.ndarray,
    ) -> tuple[np.ndarray, object]:
        step = robust_solve_position_step(
            robot=self.robot,
            solver=self.solver,
            q_current=np.asarray(q, dtype=float),
            targets=[
                self._eik.TaskTarget(
                    self._task_name,
                    target_matrix,
                    float(self._step_opts.position_gain),
                    float(self._step_opts.orientation_gain),
                )
            ],
            options=self._step_opts,
            q_lo=self._lower,
            q_hi=self._upper,
            zero_velocity_indices=self._step_opts.integration_zero_velocity_indices,
            fallback_status_names=("INVALID_INPUT",),
            hold_status_names=(
                "NON_FINITE_INPUT",
                "INFEASIBLE",
                "NUMERICAL_ERROR",
                "NO_PROGRESS",
                "COLLISION_VIOLATED",
            ),
            allow_solver_intervention=True,
            apply_collision_violated_q_solution=False,
        )
        q_out = self._normalize_and_clip(self._limit_tangent_step(q, step.q_next))
        return q_out, step.solver_result

    def arm_gravity_torque_from_configuration(self, q: np.ndarray) -> np.ndarray | None:
        """Return EmbodiK gravity-compensation torques for the six Spot arm joints."""

        if self.robot is None or not self._arm_velocity_indices:
            return None
        gravity = np.asarray(self.robot.compute_generalized_gravity(np.asarray(q, dtype=float)))
        return gravity[np.asarray(self._arm_velocity_indices, dtype=int)]

    def compute_arm_gravity_torque(
        self,
        observation: Mapping[str, np.ndarray],
        arm_command: np.ndarray,
    ) -> np.ndarray | None:
        """Sync the current Spot state and compute arm gravity compensation."""

        if not self.enabled or self.robot is None:
            return None
        q = self._sync_configuration(observation, np.asarray(arm_command, dtype=float))
        return self.arm_gravity_torque_from_configuration(q)

    def command_tool_pose(
        self,
        observation: Mapping[str, np.ndarray],
        arm_command: np.ndarray,
        *,
        body_command: np.ndarray,
        desired_pose_command: np.ndarray,
    ):
        """Return the tool pose in the policy command-space configuration."""

        if not self.enabled or self.robot is None or self._eik is None:
            return None
        self._sync_configuration(
            observation,
            np.asarray(arm_command, dtype=float),
            base_pose_override=_commanded_base_pose_wxyz(body_command, desired_pose_command),
        )
        pose = self.robot.get_frame_pose(self.tool_frame)
        return self._eik.Rt(
            R=np.asarray(pose.rotation, dtype=float),
            t=np.asarray(pose.translation, dtype=float),
        )

    def _locomanip_base_gains(self) -> np.ndarray:
        sensitivity = float(np.clip(self._locomotion_sensitivity, 0.25, 4.0))
        blend = (np.log(sensitivity) - np.log(0.25)) / (np.log(4.0) - np.log(0.25))
        blend = float(np.clip(blend, 0.0, 1.0))
        locked = np.maximum(_LOCO_ARM_BIASED_BASE_GAINS, 1e-9)
        free = np.maximum(_LOCO_NULLSPACE_BASE_GAINS, 1e-9)
        return np.exp((1.0 - blend) * np.log(locked) + blend * np.log(free))

    def _set_locomanip_posture_bias(self, *, condition_protect: bool = False) -> None:
        if self._posture_task is None or self._ik_opts is None:
            return
        if condition_protect:
            indices = self._reference_posture_velocity_indices
            weights = np.concatenate(
                (
                    _LOCO_NULLSPACE_BASE_GAINS[[2, 3, 4]],
                    _LOCO_NULLSPACE_ARM_GAINS * self._condition_arm_weight_scale,
                )
            )
        else:
            indices = self._locomanip_posture_velocity_indices
            weights = np.concatenate((self._locomanip_base_gains(), _LOCO_NULLSPACE_ARM_GAINS))
        self._posture_velocity_indices = list(indices)
        self._posture_task.set_controlled_joint_indices(self._posture_velocity_indices)
        self._posture_task.set_controlled_joint_weights(weights)
        self._ik_opts.nullspace_active_joints = self._posture_velocity_indices
        self._ik_opts.nullspace_joint_weights = weights

    def _condition_number_from_result(self, result: object) -> float:
        value = float(getattr(result, "condition_number", 1.0))
        return value if np.isfinite(value) and value >= 1.0 else 1.0

    def _collision_time_from_result(self, result: object) -> float:
        value = float(getattr(result, "collision_constraint_time_ms", 0.0))
        return value if np.isfinite(value) and value >= 0.0 else 0.0

    def _update_condition_locomotion_state(self, condition_number: float) -> bool:
        if condition_number >= _LOCO_CONDITION_HANDOFF_ENTER:
            self._condition_locomotion_active = True
        elif condition_number <= _LOCO_CONDITION_HANDOFF_EXIT:
            self._condition_locomotion_active = False
        return self._condition_locomotion_active

    @property
    def locomotion_sensitivity(self) -> float:
        """Scale solver-native x/y/yaw base assist for locomanip IK.

        Values above 1.0 lower base x/y/yaw restoration gains so the solver can
        move the base earlier. Values below 1.0 raise those gains so the arm
        absorbs more target motion before the base moves.
        """

        return float(self._locomotion_sensitivity)

    @locomotion_sensitivity.setter
    def locomotion_sensitivity(self, value: float) -> None:
        self._locomotion_sensitivity = float(np.clip(value, 0.25, 4.0))

    @property
    def condition_arm_weight_scale(self) -> float:
        """Scale the arm posture bias used while condition protection is active."""

        return float(self._condition_arm_weight_scale)

    @condition_arm_weight_scale.setter
    def condition_arm_weight_scale(self, value: float) -> None:
        self._condition_arm_weight_scale = float(np.clip(value, 1.0, 6.0))

    def solve_command(
        self,
        observation: Mapping[str, np.ndarray],
        arm_command: np.ndarray,
        *,
        target_pose,
        body_command: np.ndarray | None = None,
        desired_pose_command: np.ndarray | None = None,
    ) -> SpotIKResult:
        """Solve one EmbodiK position step and return policy command targets."""

        if not self.enabled:
            return SpotIKResult(False, {}, message=self.message)
        if self.robot is None or self.solver is None or self._step_opts is None:
            return SpotIKResult(False, {}, message="disabled: IK model was not initialized")
        if target_pose is None:
            return SpotIKResult(False, {}, message="ready: enable IK and drag gripper target")

        self.apply_runtime_options()
        base_pose_override = None
        if body_command is not None and desired_pose_command is not None:
            base_pose_override = _commanded_base_pose_wxyz(body_command, desired_pose_command)
        q = self._sync_configuration(
            observation,
            np.asarray(arm_command, dtype=float),
            base_pose_override=base_pose_override,
        )
        solve_start = perf_counter()
        seed_torso_pose = self.robot.get_frame_pose(self.body_frame)
        seed_torso_translation = np.asarray(seed_torso_pose.translation, dtype=float)
        seed_torso_rotation = np.asarray(seed_torso_pose.rotation, dtype=float)
        if self._torso_bounds_reference_pose is None:
            self._posture_reference_q = q.copy()
            self._torso_reference_translation = seed_torso_translation.copy()
            self._torso_reference_rotation = seed_torso_rotation.copy()
            self._torso_bounds_reference_pose = _pose_matrix(
                seed_torso_rotation,
                seed_torso_translation,
            )
            seed_roll, seed_pitch = _roll_pitch_from_rotation(seed_torso_rotation)
            self._body_command_reference = np.asarray(
                (
                    [seed_roll, seed_pitch, float(seed_torso_translation[2])]
                    if body_command is None
                    else body_command
                ),
                dtype=float,
            ).copy()
            self._desired_pose_command_reference = np.asarray(
                (
                    [
                        float(seed_torso_translation[0]),
                        float(seed_torso_translation[1]),
                        _yaw_from_rotation(seed_torso_rotation),
                    ]
                    if desired_pose_command is None
                    else desired_pose_command
                ),
                dtype=float,
            ).copy()
        condition_protect = self._condition_locomotion_active
        self._set_locomanip_posture_bias(condition_protect=condition_protect)
        self._posture_task.set_target_configuration(
            self._neutral_torso_posture_target(q, seed_torso_rotation)
        )
        self._step_opts.torso_constraint.pose_bounds_reference_pose = self._torso_bounds_reference_pose
        self._configure_collision_constraint()
        target_matrix = np.eye(4, dtype=float)
        target_matrix[:3, :3] = np.asarray(target_pose.rotation, dtype=float)
        target_matrix[:3, 3] = np.asarray(target_pose.translation, dtype=float)
        seed_tool_pose = self.robot.get_frame_pose(self.tool_frame)
        target_requires_motion = _pose_delta_exceeds(
            np.asarray(seed_tool_pose.rotation, dtype=float),
            np.asarray(seed_tool_pose.translation, dtype=float),
            target_matrix[:3, :3],
            target_matrix[:3, 3],
            linear_eps=_LOCO_CONDITION_RETRY_LINEAR_EPS,
            angular_eps=_LOCO_CONDITION_RETRY_ANGULAR_EPS,
        )

        q_command, result = self._solve_position_step_guarded(q, target_matrix)
        condition_number = self._condition_number_from_result(result)
        collision_constraint_time_ms = self._collision_time_from_result(result)
        if (
            target_requires_motion
            and not condition_protect
            and condition_number >= _LOCO_CONDITION_HANDOFF_ENTER
        ):
            condition_protect = True
            self._condition_locomotion_active = True
            self._set_locomanip_posture_bias(
                condition_protect=True,
            )
            self._posture_task.set_target_configuration(
                self._neutral_torso_posture_target(q, seed_torso_rotation)
            )
            q_command, result = self._solve_position_step_guarded(q, target_matrix)
            condition_number = self._condition_number_from_result(result)
            collision_constraint_time_ms += self._collision_time_from_result(result)
        if target_requires_motion or condition_protect:
            self._update_condition_locomotion_state(condition_number)
        solve_time_ms = (perf_counter() - solve_start) * 1e3

        status_name = result.status.name
        accepted = status_name in {"SUCCESS", "INFEASIBLE", "NUMERICAL_ERROR", "NO_PROGRESS"}
        if not accepted:
            return SpotIKResult(
                False,
                {},
                solve_time_ms=solve_time_ms,
                collision_constraint_time_ms=collision_constraint_time_ms,
                condition_number=condition_number,
                status_name=status_name,
                message=f"{status_name}: holding IK command",
            )

        if self._gripper_joint_name:
            idx = int(self.robot.get_joint_config_index(self._gripper_joint_name))
            q_command[idx] = q[idx]

        self.robot.update_configuration(q_command)
        self._q = np.asarray(q_command, dtype=float).copy()
        arm_gravity_torque = self.arm_gravity_torque_from_configuration(q_command)
        arm_out = np.asarray(arm_command, dtype=float).copy()
        for i, mjcf_name in enumerate(MJCF_ARM_JOINT_NAMES):
            arm_out[i] = self._read_joint_position(q_command, self._arm_joint_map[mjcf_name])

        torso_pose = self.robot.get_frame_pose(self.body_frame)
        torso_rotation = np.asarray(torso_pose.rotation, dtype=float)
        torso_translation = np.asarray(torso_pose.translation, dtype=float)
        if (
            self._body_command_reference is not None
            and self._desired_pose_command_reference is not None
            and self._torso_reference_translation is not None
            and self._torso_reference_rotation is not None
        ):
            body_reference = np.asarray(self._body_command_reference, dtype=float)
            desired_reference = np.asarray(self._desired_pose_command_reference, dtype=float)
            translation_delta = torso_translation - np.asarray(
                self._torso_reference_translation,
                dtype=float,
            )
            _, _, yaw_delta = _rotation_delta_rpy(
                np.asarray(self._torso_reference_rotation, dtype=float),
                torso_rotation,
            )
            body_out = body_reference.copy()
            raw_pose_delta = np.array(
                [
                    float(translation_delta[0]),
                    float(translation_delta[1]),
                    yaw_delta,
                ],
                dtype=float,
            )
            desired_pose_out = desired_reference + raw_pose_delta
        else:
            roll, pitch = _roll_pitch_from_rotation(torso_rotation)
            yaw = _yaw_from_rotation(torso_rotation)
            body_out = np.array(
                [roll, pitch, float(torso_translation[2])],
                dtype=float,
            )
            desired_pose_out = np.array(
                [
                    float(torso_translation[0]),
                    float(torso_translation[1]),
                    yaw,
                ],
                dtype=float,
            )
        self.robot.update_configuration(q_command)

        return SpotIKResult(
            True,
            {},
            arm_command=arm_out,
            body_command=body_out,
            desired_pose_command=desired_pose_out,
            arm_gravity_torque=arm_gravity_torque,
            solve_time_ms=solve_time_ms,
            collision_constraint_time_ms=collision_constraint_time_ms,
            condition_number=condition_number,
            position_error=float(result.position_error),
            orientation_error=float(result.orientation_error),
            status_name=status_name,
            message=(
                f"{status_name}: pos={float(result.position_error) * 1e3:.1f} mm, "
                f"rot={float(result.orientation_error):.3f} rad, "
                f"cond={condition_number:.0f}"
            ),
        )

    def maybe_apply(
        self,
        current_targets: Mapping[str, float],
        *,
        target_pose: np.ndarray | None = None,
    ) -> SpotIKResult:
        """Return IK-refined targets when enabled, otherwise pass policy targets through.

        The v1 hook intentionally avoids inventing a URDF/MJCF conversion. Once a
        compatible Spot URDF is supplied, this is the integration point for adding
        EmbodiK frame tasks and CoM constraints while keeping the policy code stable.
        """

        targets = {
            normalize_mjcf_joint_name(name): float(value) for name, value in current_targets.items()
        }
        if not self.enabled:
            return SpotIKResult(False, targets, message=self.message)
        if self.robot is None or self.solver is None:
            return SpotIKResult(False, targets, message="disabled: IK model was not initialized")
        if target_pose is None:
            return SpotIKResult(True, targets, message="enabled: no IK target requested")
        return SpotIKResult(
            True,
            targets,
            message=(
                "enabled: URDF loaded; frame-task target application is reserved for "
                "the Spot URDF mapping pass"
            ),
        )


def target_pose_from_wxyz(position: np.ndarray, quat_wxyz: np.ndarray):
    """Build an EmbodiK SE3 target from viser/MuJoCo ``position`` and ``wxyz`` quaternion."""

    from embodik import Rt

    return Rt(R=_quat_xyzw_to_rotation(_quat_wxyz_to_xyzw(quat_wxyz)), t=np.asarray(position, dtype=float))


class SpotFullBodyIK:
    """Regular-Viser friendly Spot whole-body IK backend.

    The two-stage mode follows the classic Spot pattern:
    first solve gripper + floating torso with the legs excluded, then use a
    torso task with the arm excluded so the legs realize the torso displacement
    while foot targets remain pinned.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        config: SpotFullBodyIKConfig | None = None,
    ):
        try:
            import embodik as eik
        except ImportError as exc:
            raise RuntimeError("embodik is required for Spot full-body IK") from exc

        path = Path(urdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Spot URDF not found: {path}")

        self.eik = eik
        self.config = config or SpotFullBodyIKConfig()
        if self.config.torso_pose_half_range is None:
            self.config.torso_pose_half_range = STANDARD_FULL_BODY_TORSO_POSE_HALF_RANGE.copy()
        self.robot = eik.RobotModel(str(path), floating_base=True)
        self.solver = eik.KinematicsSolver(self.robot)
        self.solver.dt = float(self.config.dt)
        self.solver.set_damping(float(self.config.damping))
        self.solver.set_tolerance(float(self.config.tolerance))

        self.body_frame = _select_existing_frame(self.robot, BODY_FRAME_CANDIDATES)
        self.tool_frame = _select_existing_frame(self.robot, TOOL_FRAME_CANDIDATES)
        self.foot_frames = tuple(
            _select_existing_frame(self.robot, candidates)
            for candidates in FOOT_FRAME_CANDIDATE_SETS
        )
        self._arm_joint_map = self._resolve_joint_map(MJCF_ARM_JOINT_NAMES)
        self._leg_joint_map = self._resolve_joint_map(SPOT_LEG_JOINT_NAMES)
        self._gripper_joint_name = self._resolve_joint_name(MJCF_GRIPPER_JOINT_NAME, required=False)

        self._arm_velocity_indices = self._velocity_indices(tuple(self._arm_joint_map.values()))
        self._leg_velocity_indices = self._velocity_indices(tuple(self._leg_joint_map.values()))
        self._gripper_velocity_indices = self._velocity_indices((self._gripper_joint_name,))
        self._base_velocity_indices = list(range(6))
        self._posture_velocity_indices = [
            *self._base_velocity_indices,
            *self._leg_velocity_indices,
            *self._arm_velocity_indices,
        ]
        self._stage1_posture_velocity_indices = [
            *self._base_velocity_indices,
            *self._arm_velocity_indices,
        ]
        self._full_posture_weights = np.concatenate(
            (
                _FULL_BODY_NULLSPACE_BASE_GAINS,
                np.full(len(self._leg_velocity_indices), 0.25, dtype=float),
                _FULL_BODY_NULLSPACE_ARM_GAINS,
            )
        )
        self._stage1_posture_weights = np.concatenate(
            (
                _FULL_BODY_NULLSPACE_BASE_GAINS,
                _FULL_BODY_NULLSPACE_ARM_GAINS,
            )
        )

        self._tool_task_name = "spot_tool_pose"
        self._torso_task_name = "spot_torso_pose"
        self._torso_bias_task_name = "spot_torso_secondary_bias"
        self._foot_task_names = tuple(f"spot_foot_{i}_position" for i in range(len(self.foot_frames)))
        self._posture_task_name = "spot_full_body_posture"

        self._tool_task = self.solver.add_frame_task(
            self._tool_task_name,
            self.tool_frame,
            eik.TaskType.FRAME_POSE,
        )
        self._tool_task.priority = 0
        self._tool_task.weight = 1.0
        self._tool_task.solve_mode = getattr(eik.TaskSolveMode, "SCALE_ELASTIC", eik.TaskSolveMode.SCALE)
        self._tool_task.allow_min_error_fallback = False

        self._torso_task = self.solver.add_frame_task(
            self._torso_task_name,
            self.body_frame,
            eik.TaskType.FRAME_POSE,
        )
        self._torso_task.priority = 0
        self._torso_task.weight = 1.0
        self._torso_task.solve_mode = getattr(eik.TaskSolveMode, "SCALE_ELASTIC", eik.TaskSolveMode.SCALE)
        self._torso_task.allow_min_error_fallback = False

        self._torso_bias_task = self.solver.add_frame_task(
            self._torso_bias_task_name,
            self.body_frame,
            eik.TaskType.FRAME_POSE,
        )
        self._torso_bias_task.priority = 1
        self._torso_bias_task.weight = 1.0
        self._torso_bias_task.solve_mode = eik.TaskSolveMode.MIN_ERROR
        self._torso_bias_task.allow_min_error_fallback = False

        self._foot_tasks = []
        for task_name, frame_name in zip(self._foot_task_names, self.foot_frames):
            task = self.solver.add_frame_task(task_name, frame_name, eik.TaskType.FRAME_POSITION)
            task.priority = 0
            task.weight = 1.0
            task.solve_mode = getattr(eik.TaskSolveMode, "SCALE_ELASTIC", eik.TaskSolveMode.SCALE)
            task.allow_min_error_fallback = False
            self._foot_tasks.append(task)

        self._posture_task = self.solver.add_posture_task(
            self._posture_task_name,
            self._posture_velocity_indices,
        )
        self._posture_task.priority = 1
        self._posture_task.weight = float(self.config.nullspace_gain)
        self._posture_task.set_controlled_joint_weights(self._full_posture_weights)
        self._posture_task.solve_mode = eik.TaskSolveMode.MIN_ERROR
        self._posture_task.allow_min_error_fallback = False
        self._target_tasks = (self._tool_task, self._torso_task, *self._foot_tasks)
        self._apply_target_solve_mode()

        self._lower, self._upper = [
            np.asarray(v, dtype=float) for v in self.robot.get_joint_limits()
        ]
        self.q0 = self._make_initial_configuration()
        self.q = self.q0.copy()
        self._posture_bias_q = self.q0.copy()
        self.robot.update_configuration(self.q)
        self._torso_bias_pose = self._pose_homogeneous(self.body_frame)
        self._torso_bounds_reference_pose = self._pose_homogeneous(self.body_frame)
        self._foot_anchor_poses = self._capture_frame_poses(self.foot_frames)
        self._contact_projection_enabled = False
        self._collision_include_pairs = spot_collision_pairs_from_references(self.robot)
        self._collision_config_key: tuple[object, ...] | None = None

    def _resolve_joint_name(self, mjcf_name: str, *, required: bool = True) -> str:
        for name in _joint_name_candidates(mjcf_name):
            if self.robot.has_joint(name):
                return name
        if required:
            raise RuntimeError(f"Spot URDF does not contain joint compatible with {mjcf_name!r}")
        return ""

    def _resolve_joint_map(self, mjcf_names: tuple[str, ...]) -> dict[str, str]:
        return {name: self._resolve_joint_name(name) for name in mjcf_names}

    def _velocity_indices(self, joint_names: tuple[str, ...]) -> list[int]:
        indices: list[int] = []
        for joint_name in joint_names:
            if not joint_name:
                continue
            start = int(self.robot.get_joint_velocity_index(joint_name))
            size = int(self.robot.get_joint_velocity_size(joint_name))
            indices.extend(range(start, start + size))
        return indices

    def _set_joint_position(self, q: np.ndarray, joint_name: str, value: float) -> None:
        if not joint_name:
            return
        idx = int(self.robot.get_joint_config_index(joint_name))
        size = int(self.robot.get_joint_config_size(joint_name))
        if size == 1 and idx < q.size:
            q[idx] = float(value)

    def _make_initial_configuration(self) -> np.ndarray:
        q = np.asarray(self.robot.get_current_configuration(), dtype=float).copy()
        if q.size >= 7:
            q[:7] = np.array(
                [0.0, 0.0, DEFAULT_STAND_BASE_HEIGHT, 0.0, 0.0, 0.0, 1.0],
                dtype=float,
            )
        for mjcf_name, value in zip(SPOT_LEG_JOINT_NAMES, DEFAULT_STAND_LEG_JOINTS):
            self._set_joint_position(q, self._leg_joint_map[mjcf_name], float(value))
        for mjcf_name, value in zip(MJCF_ARM_JOINT_NAMES, INITIAL_ARM_COMMAND[:6]):
            self._set_joint_position(q, self._arm_joint_map[mjcf_name], float(value))
        if self._gripper_joint_name:
            self._set_joint_position(q, self._gripper_joint_name, float(INITIAL_ARM_COMMAND[6]))
        return q

    def set_arm_configuration(self, arm_command: np.ndarray) -> None:
        command = np.asarray(arm_command, dtype=float)
        if command.shape != (7,):
            raise ValueError(f"Expected 7 arm/gripper values, got {command.shape}")
        q = self.q.copy()
        for mjcf_name, value in zip(MJCF_ARM_JOINT_NAMES, command[:6]):
            self._set_joint_position(q, self._arm_joint_map[mjcf_name], float(value))
        if self._gripper_joint_name:
            self._set_joint_position(q, self._gripper_joint_name, float(command[6]))
        self.q = self._normalize_and_clip(q)
        self.robot.update_configuration(self.q)

    def set_gripper_configuration(self, gripper_command: float) -> None:
        if not self._gripper_joint_name:
            return
        q = self.q.copy()
        self._set_joint_position(q, self._gripper_joint_name, float(gripper_command))
        self.q = self._normalize_and_clip(q)
        self.robot.update_configuration(self.q)

    def _pose_homogeneous(self, frame_name: str) -> np.ndarray:
        pose = self.robot.get_frame_pose(frame_name)
        return _pose_matrix(
            np.asarray(pose.rotation, dtype=float),
            np.asarray(pose.translation, dtype=float),
        )

    def _as_homogeneous(self, pose) -> np.ndarray:
        return np.asarray(pose.homogeneous() if hasattr(pose, "homogeneous") else pose, dtype=float)

    def _capture_frame_poses(self, frame_names: tuple[str, ...]) -> tuple[np.ndarray, ...]:
        return tuple(self._pose_homogeneous(frame_name) for frame_name in frame_names)

    def _step_options(self, excluded: list[int] | tuple[int, ...] = ()) -> object:
        opts = self.eik.PositionStepOptions()
        opts.position_gain = float(self.config.position_gain)
        opts.orientation_gain = float(self.config.orientation_gain)
        opts.max_steps = int(self.config.max_steps)
        opts.dt = float(self.config.dt)
        opts.adaptive_dt = bool(self.config.adaptive_dt)
        opts.adaptive_dt_max_scale = float(self.config.adaptive_dt_max_scale)
        opts.adaptive_dt_reference_distance = float(self.config.adaptive_dt_reference_distance)
        opts.max_linear_speed = float(self.config.max_linear_speed)
        opts.max_angular_speed = float(self.config.max_angular_speed)
        if hasattr(opts, "stall_recovery"):
            opts.stall_recovery = bool(self.config.enable_collision)
        if hasattr(opts, "no_progress_max_steps"):
            opts.no_progress_max_steps = 5
        if hasattr(opts, "no_progress_error_tolerance"):
            opts.no_progress_error_tolerance = 1e-5
        if hasattr(opts, "no_progress_dq_norm_tolerance"):
            opts.no_progress_dq_norm_tolerance = 1e-6
        configure_primary_solve_mode(
            opts,
            getattr(self.eik.TaskSolveMode, str(self.config.target_solve_mode), self.eik.TaskSolveMode.SCALE_ELASTIC),
            False,
        )
        opts.excluded_joint_indices = sorted(set(int(i) for i in excluded))
        opts.integration_zero_velocity_indices = sorted(set(self._gripper_velocity_indices))
        opts.torso_constraint.enabled = True
        opts.torso_constraint.frame_name = self.body_frame
        opts.torso_constraint.pose_lower_bounds = -np.asarray(
            self.config.torso_pose_half_range,
            dtype=float,
        )
        opts.torso_constraint.pose_upper_bounds = np.asarray(
            self.config.torso_pose_half_range,
            dtype=float,
        )
        opts.torso_constraint.pose_axis_mask = np.ones(6, dtype=float)
        opts.torso_constraint.velocity_limits = np.full(
            6,
            float(self.config.base_velocity_limit),
            dtype=float,
        )
        opts.torso_constraint.acceleration_limits = np.full(
            6,
            float(self.config.base_acceleration_limit),
            dtype=float,
        )
        opts.torso_constraint.pose_bounds_reference_pose = self._torso_bounds_reference_pose
        return opts

    def _foot_targets(self) -> list[object]:
        return [
            self.eik.TaskTarget(task_name, pose, float(self.config.foot_position_gain), 0.0)
            for task_name, pose in zip(self._foot_task_names, self._foot_anchor_poses)
        ]

    def _apply_target_solve_mode(self) -> None:
        mode_name = str(self.config.target_solve_mode)
        mode = getattr(self.eik.TaskSolveMode, mode_name, self.eik.TaskSolveMode.SCALE_ELASTIC)
        for task in self._target_tasks:
            task.solve_mode = mode

    def _set_contact_projection(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._contact_projection_enabled:
            return
        self.solver.clear_contact_frames()
        if enabled:
            self.solver.configure_contact_frames(
                list(self.foot_frames),
                self.eik.ContactType.POINT_CONTACT,
            )
        self._contact_projection_enabled = enabled

    def _configure_collision_constraint(self) -> None:
        if not hasattr(self.solver, "configure_collision_constraint"):
            return
        enabled = bool(self.config.enable_collision and self._collision_include_pairs)
        key: tuple[object, ...] = (
            enabled,
            round(float(self.config.collision_min_distance), 6),
            int(self.config.collision_max_constraints),
            str(self.config.collision_tuning_mode),
        )
        if key == self._collision_config_key:
            return
        if not enabled:
            if hasattr(self.solver, "clear_collision_constraint"):
                self.solver.clear_collision_constraint()
            self._collision_config_key = key
            return
        apply_collision_tuning_mode(self.solver, str(self.config.collision_tuning_mode))
        if hasattr(self.solver, "enable_stall_handler"):
            self.solver.enable_stall_handler(float(self.config.collision_min_distance))
            if hasattr(self.solver, "configure_stall_handler"):
                self.solver.configure_stall_handler(
                    stall_threshold=3,
                    restore_rate=0.2,
                    floor_fraction=0.0,
                )
        self.solver.configure_collision_constraint(
            min_distance=float(self.config.collision_min_distance),
            include_pairs=list(self._collision_include_pairs),
            exclude_pairs=[],
            nearest_points_all_pairs=False,
            max_constraints=int(self.config.collision_max_constraints),
        )
        self._collision_config_key = key

    def _append_foot_targets_if_needed(self, targets: list[object]) -> list[object]:
        if self.config.use_contact_projection:
            return targets
        return [*targets, *self._foot_targets()]

    def _normalize_and_clip(self, q: np.ndarray) -> np.ndarray:
        out = np.asarray(q, dtype=float).copy()
        if out.size >= 7:
            quat = out[3:7]
            norm = float(np.linalg.norm(quat))
            if norm > 1e-12:
                out[3:7] = quat / norm
        if out.size > 7:
            out[7:] = np.clip(out[7:], self._lower[7:], self._upper[7:])
        return out

    def _accept_solution(self, result) -> bool:
        return result.status.name in {"SUCCESS", "INFEASIBLE", "NO_PROGRESS", "NUMERICAL_ERROR"}

    def _limit_tangent_step(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        max_component = float(self.config.max_tangent_step)
        if max_component <= 0.0:
            return np.asarray(q_to, dtype=float)
        q_step = np.asarray(self.robot.difference(q_from, q_to), dtype=float)
        step_component = float(np.max(np.abs(q_step))) if q_step.size else 0.0
        if step_component <= max_component or step_component <= 1e-12:
            return np.asarray(q_to, dtype=float)
        return np.asarray(self.robot.integrate(q_from, q_step * (max_component / step_component), 1.0), dtype=float)

    def _solve_position_step_guarded(
        self,
        q: np.ndarray,
        targets: list[object],
        opts: object,
    ) -> tuple[np.ndarray, object]:
        step = robust_solve_position_step(
            robot=self.robot,
            solver=self.solver,
            q_current=np.asarray(q, dtype=float),
            targets=targets,
            options=opts,
            q_lo=self._lower,
            q_hi=self._upper,
            zero_velocity_indices=self._gripper_velocity_indices,
            fallback_status_names=("INVALID_INPUT",),
            hold_status_names=(
                "NON_FINITE_INPUT",
                "INFEASIBLE",
                "NUMERICAL_ERROR",
                "NO_PROGRESS",
                "COLLISION_VIOLATED",
            ),
            allow_solver_intervention=True,
            apply_collision_violated_q_solution=False,
        )
        q_out = self._normalize_and_clip(self._limit_tangent_step(q, step.q_next))
        return q_out, step.solver_result

    def _update_posture_target(self) -> None:
        self._posture_task.weight = float(self.config.nullspace_gain)
        self._posture_task.set_target_configuration(np.asarray(self._posture_bias_q, dtype=float))

    def _torso_bias_target(self) -> object:
        self._torso_bias_task.weight = 1.0
        self._torso_bias_task.set_target_pose(
            self._torso_bias_pose[:3, 3],
            self._torso_bias_pose[:3, :3],
        )
        return self.eik.TaskTarget(
            self._torso_bias_task_name,
            self._torso_bias_pose,
            float(self.config.position_gain),
            float(self.config.orientation_gain),
        )

    def _set_posture_bias_scope(self, *, stage1: bool) -> None:
        if stage1:
            self._posture_task.set_controlled_joint_indices(self._stage1_posture_velocity_indices)
            self._posture_task.set_controlled_joint_weights(self._stage1_posture_weights)
        else:
            self._posture_task.set_controlled_joint_indices(self._posture_velocity_indices)
            self._posture_task.set_controlled_joint_weights(self._full_posture_weights)

    def _solve_tool_arm_torso_from(
        self,
        q: np.ndarray,
        target_pose,
        *,
        maintain_contacts: bool,
        lock_base: bool,
    ) -> tuple[np.ndarray, object]:
        self.robot.update_configuration(q)
        self._update_posture_target()
        self._set_posture_bias_scope(stage1=True)
        torso_bias_target = self._torso_bias_target()
        self._apply_target_solve_mode()
        self._set_contact_projection(self.config.use_contact_projection and maintain_contacts)
        self._configure_collision_constraint()
        excluded = [*self._leg_velocity_indices, *self._gripper_velocity_indices]
        if lock_base:
            excluded.extend(self._base_velocity_indices)
        opts = self._step_options(excluded)
        opts.torso_constraint.pose_axis_mask = _ARM_TORSO_POSE_AXIS_MASK.copy()
        targets = [
            self.eik.TaskTarget(
                self._tool_task_name,
                self._as_homogeneous(target_pose),
                float(self.config.position_gain),
                float(self.config.orientation_gain),
            ),
            torso_bias_target,
        ]
        return self._solve_position_step_guarded(q, targets, opts)

    def _solve_torso_with_legs_from(self, q: np.ndarray, torso_target_pose) -> tuple[np.ndarray, object]:
        self.robot.update_configuration(q)
        self._update_posture_target()
        self._set_posture_bias_scope(stage1=False)
        self._apply_target_solve_mode()
        self._set_contact_projection(self.config.use_contact_projection)
        self._configure_collision_constraint()
        opts = self._step_options(
            (
                *self._arm_velocity_indices,
                *self._gripper_velocity_indices,
            )
        )
        target = self.eik.TaskTarget(
            self._torso_task_name,
            self._as_homogeneous(torso_target_pose),
            float(self.config.position_gain),
            float(self.config.orientation_gain),
        )
        return self._solve_position_step_guarded(q, self._append_foot_targets_if_needed([target]), opts)

    def _solve_full_body_from(self, q: np.ndarray, target_pose) -> tuple[np.ndarray, object]:
        self.robot.update_configuration(q)
        self._update_posture_target()
        self._set_posture_bias_scope(stage1=False)
        self._apply_target_solve_mode()
        self._set_contact_projection(self.config.use_contact_projection)
        self._configure_collision_constraint()
        opts = self._step_options(self._gripper_velocity_indices)
        targets = [
            self.eik.TaskTarget(
                self._tool_task_name,
                self._as_homogeneous(target_pose),
                float(self.config.position_gain),
                float(self.config.orientation_gain),
            ),
            self._torso_bias_target(),
        ]
        return self._solve_position_step_guarded(q, self._append_foot_targets_if_needed(targets), opts)

    def reset(self) -> None:
        self.q = self.q0.copy()
        self.robot.update_configuration(self.q)
        self._torso_bounds_reference_pose = self._pose_homogeneous(self.body_frame)
        self._foot_anchor_poses = self._capture_frame_poses(self.foot_frames)

    def reanchor_feet_and_torso(self) -> None:
        self.robot.update_configuration(self.q)
        self._torso_bounds_reference_pose = self._pose_homogeneous(self.body_frame)
        self._foot_anchor_poses = self._capture_frame_poses(self.foot_frames)

    def current_tool_pose(self):
        self.robot.update_configuration(self.q)
        return self.robot.get_frame_pose(self.tool_frame)

    def current_torso_pose(self):
        self.robot.update_configuration(self.q)
        return self.robot.get_frame_pose(self.body_frame)

    def foot_anchor_error(self) -> float:
        self.robot.update_configuration(self.q)
        max_error = 0.0
        for frame_name, anchor in zip(self.foot_frames, self._foot_anchor_poses):
            current = self._pose_homogeneous(frame_name)
            max_error = max(max_error, float(np.linalg.norm(current[:3, 3] - anchor[:3, 3])))
        return max_error

    def posture_bias_error(self) -> dict[str, float]:
        q = np.asarray(self.q, dtype=float)
        q_bias = np.asarray(self._posture_bias_q, dtype=float)
        base_position_error = float(np.linalg.norm(q[:3] - q_bias[:3])) if q.size >= 3 else 0.0
        arm_error = 0.0
        for joint_name in self._arm_joint_map.values():
            idx = int(self.robot.get_joint_config_index(joint_name))
            arm_error = max(arm_error, abs(float(q[idx] - q_bias[idx])))
        leg_error = 0.0
        for joint_name in self._leg_joint_map.values():
            idx = int(self.robot.get_joint_config_index(joint_name))
            leg_error = max(leg_error, abs(float(q[idx] - q_bias[idx])))
        return {
            "base_position": base_position_error,
            "arm": arm_error,
            "leg": leg_error,
        }

    def solve_arm_torso(self, target_pose) -> object:
        self.q, result = self._solve_tool_arm_torso_from(
            self.q,
            target_pose,
            maintain_contacts=False,
            lock_base=False,
        )
        self.robot.update_configuration(self.q)
        return result

    def solve_torso_only(self, torso_target_pose) -> object:
        target = self._as_homogeneous(torso_target_pose)
        self.q, result = self._solve_torso_with_legs_from(self.q, target)
        self.robot.update_configuration(self.q)
        return result

    def solve_full_body(self, target_pose) -> object:
        self.q, result = self._solve_full_body_from(self.q, target_pose)
        self.robot.update_configuration(self.q)
        return result

    def solve_two_stage(self, target_pose) -> tuple[object, object]:
        q_seed = self.q.copy()
        q_arm, _arm_result = self._solve_tool_arm_torso_from(
            q_seed,
            target_pose,
            maintain_contacts=False,
            lock_base=True,
        )
        self.robot.update_configuration(q_arm)
        torso_target = self._pose_homogeneous(self.body_frame)
        tool_pose = self._pose_homogeneous(self.tool_frame)
        target_matrix = self._as_homogeneous(target_pose)
        tool_residual = target_matrix[:3, 3] - tool_pose[:3, 3]
        torso_target[:3, 3] = self._torso_bias_pose[:3, 3] + tool_residual
        torso_target[:3, :3] = self._torso_bias_pose[:3, :3]

        q_intermediate = q_seed.copy()
        for joint_name in self._arm_joint_map.values():
            idx = int(self.robot.get_joint_config_index(joint_name))
            q_intermediate[idx] = q_arm[idx]
        if self._gripper_joint_name:
            idx = int(self.robot.get_joint_config_index(self._gripper_joint_name))
            q_intermediate[idx] = q_seed[idx]

        self.q, second = self._solve_torso_with_legs_from(q_intermediate, torso_target)
        self.robot.update_configuration(self.q)
        return _arm_result, second

    def solve(self, mode: SpotFullBodyIKMode, target_pose, torso_target_pose=None):
        if mode == SpotFullBodyIKMode.ARM_TORSO:
            return self.solve_arm_torso(target_pose)
        if mode == SpotFullBodyIKMode.TORSO_ONLY:
            torso_target = torso_target_pose if torso_target_pose is not None else target_pose
            return self.solve_torso_only(torso_target)
        if mode == SpotFullBodyIKMode.FULL_BODY:
            return self.solve_full_body(target_pose)
        if mode == SpotFullBodyIKMode.TWO_STAGE:
            return self.solve_two_stage(target_pose)
        raise ValueError(f"Unsupported Spot IK mode: {mode}")


def support_polygon_from_mujoco_foot_sites(model, data) -> np.ndarray:
    """Return available foot-site XY positions for CoM/support-polygon visualization."""

    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("mujoco is required to read foot sites") from exc

    points: list[np.ndarray] = []
    for site_name in ("FL", "FR", "HL", "HR"):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id >= 0:
            points.append(np.asarray(data.site_xpos[site_id, :2], dtype=float).copy())
    return np.asarray(points, dtype=float)


def controlled_joint_names() -> tuple[str, ...]:
    return (*SPOT_LEG_JOINT_NAMES, *MJCF_ARM_JOINT_NAMES)
