#!/usr/bin/env python3
"""Interactive collision-aware IK using embodiK and Viser.

Use --gpu-wbc-interactive to add an explicit CPU/GPU backend selector. The GPU
path derives its shape from the loaded model and never silently falls back to
CPU. ``--gpu-wbc-manifest`` remains an optional compatibility validation input.
"""

from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pinocchio as pin
import viser
from embodik.gpu.wbc import GpuWbcMultiFrameSolver
from example_helpers.ik_common import (
    COLLISION_DEBUG_LOG_PERIOD_S,
    COLLISION_TUNING_OPTIONS,
    DEFAULT_ADAPTIVE_DT,
    DEFAULT_ADAPTIVE_DT_MAX_SCALE,
    DEFAULT_ADAPTIVE_DT_REFERENCE_DISTANCE,
    DEFAULT_COLLISION_TUNING_MODE,
    DEFAULT_NULLSPACE_ENABLED,
    DEFAULT_NULLSPACE_GAIN,
    DEFAULT_POS_GAIN,
    DEFAULT_ROT_GAIN,
    DEFAULT_SOLVER_DT,
    DEFAULT_VISER_PORT,
    SOLVER_LEVEL_ACCELERATION,
    SOLVER_LEVEL_OPTIONS,
    SOLVER_LEVEL_VELOCITY,
    apply_collision_tuning_mode,
    configure_solver_runtime_policy,
)
from robot_descriptions.loaders.yourdfpy import load_robot_description
from utils.robot_models import ensure_ros_package_path, load_robot_presets
from viser.extras import ViserUrdf

import embodik
from embodik import Rt, q2r, r2q

# -----------------------------------------------------------------------------
# Default numeric constants
# -----------------------------------------------------------------------------

MAX_LINEAR_STEP = 2.0
MAX_ANGULAR_STEP = 2.0
DEFAULT_COLLISION_GAIN = 1.0
COLLISION_EXAMPLE_SOLVER_LEVEL_OPTIONS = SOLVER_LEVEL_OPTIONS
DEFAULT_ACCELERATION_SOLVER_LIMIT = 15.0
DEFAULT_ACCELERATION_COLLISION_VALIDATION_SUBSTEPS = 2
ACCELERATION_EXAMPLE_TASK_MODE = "SCALE"
CPU_SOLVER_LABEL = "CPU embodiK"
GPU_SOLVER_LABEL = "GPU Newton/Warp"

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


def generate_auto_collision_exclusions(
    robot: embodik.RobotModel, robot_key: str
) -> List[Tuple[str, str]]:
    exclusions: List[Tuple[str, str]] = []
    for name_a, name_b in robot.get_collision_pair_names():
        if _should_auto_exclude_pair(name_a, name_b, robot_key):
            exclusions.append((name_a, name_b))
    return exclusions


def gpu_collision_include_pairs(
    robot: embodik.RobotModel, exclusions: List[Tuple[str, str]]
) -> tuple[tuple[str, str], ...]:
    """Return the model's collision pairs after order-independent filtering."""

    excluded = {frozenset((str(a), str(b))) for a, b in exclusions}
    return tuple(
        (str(a), str(b))
        for a, b in robot.get_collision_pair_names()
        if frozenset((str(a), str(b))) not in excluded
    )


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
        raise ValueError(
            f"Unsupported robot '{robot_key}'. Available options: {sorted(ROBOT_PRESETS)}"
        )

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
                '  python -m pip install "embodik[examples]"'
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
                extra_gripper = preset.get("extra_gripper_default", np.array([0.02, 0.02]))
                if isinstance(extra_gripper, list):
                    extra_gripper = np.array(extra_gripper)
                default_config = np.concatenate([default_config, extra_gripper])
            else:
                # Truncate to match
                default_config = default_config[: len(joint_names)]

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
    primary_mode: str = "SCALE"
    primary_fallback: bool = False
    primary_scale: float = 1.0
    collision_time_ms: float = 0.0
    collision_sphere_culled: int = 0
    collision_exact_queries: int = 0
    condition_number: float = 1.0
    solver_level: str = SOLVER_LEVEL_VELOCITY
    velocity_collision_lift_applied: bool = False
    collision_step_certified: bool = False
    collision_validation_samples: int = 0
    velocity_collision_validation_allowed_pairs: int = 0
    velocity_collision_validation_pairs_checked: int = 0
    velocity_collision_validation_sample_exact_distance_queries: int = 0
    velocity_collision_validation_conservative_bound_certified_pairs: int = 0


def acceleration_api_unavailable_reason() -> str:
    required = (
        "AccelerationSolver",
        "AccelerationTaskReference",
        "AccelerationSolveOptions",
        "VelocityCollisionLiftOptions",
    )
    missing = [name for name in required if not hasattr(embodik, name)]
    if missing:
        return "missing Python acceleration API: " + ", ".join(missing)
    if not hasattr(embodik, "TaskType"):
        return "missing Python TaskType API"
    return ""


def _acceleration_reference(dimension: int, gain: float) -> object:
    reference = embodik.AccelerationTaskReference()
    reference.desired_velocity = np.zeros(dimension)
    reference.desired_acceleration = np.zeros(dimension)
    reference.proportional_gain = max(float(gain), 0.0)
    reference.derivative_gain = 2.0 * math.sqrt(reference.proportional_gain)
    return reference


def _rotation_error_rad(current: np.ndarray, target: np.ndarray) -> float:
    relative = np.asarray(target, dtype=float).T @ np.asarray(current, dtype=float)
    cos_theta = (float(np.trace(relative)) - 1.0) * 0.5
    return float(math.acos(min(1.0, max(-1.0, cos_theta))))


class embodiKBackend:
    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        ensure_ros_package_path(cfg.urdf_path)
        self.robot = embodik.RobotModel(str(cfg.urdf_path), floating_base=False)
        self.solver = embodik.KinematicsSolver(self.robot)
        self.solver.dt = DEFAULT_SOLVER_DT
        configure_solver_runtime_policy(self.solver)
        self.acceleration_solver = None
        self.acceleration_position_task = None
        self.acceleration_orientation_task = None
        self.acceleration_nullspace_task = None
        self._velocity_collision_lift_options = None
        self._acceleration_dq = np.zeros(self.robot.nv, dtype=float)
        self._solver_level = SOLVER_LEVEL_VELOCITY
        self._acceleration_unavailable_reason = acceleration_api_unavailable_reason()
        self._acceleration_collision_enabled = False
        self._acceleration_collision_reason = "self-collision is inactive"
        self._collision_tuning_mode = "balanced"
        apply_collision_tuning_mode(self.solver, self._collision_tuning_mode)

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
                self.default_arm = self.default_arm[: self.arm_dofs]

        self.default_full = np.zeros(self.full_dofs, dtype=float)
        self.default_full[: self.arm_dofs] = self.default_arm
        self.q = self.default_full.copy()
        self.robot.update_configuration(self.q)

        lower, upper = self.robot.get_joint_limits()
        self.lower = lower.astype(float)
        self.upper = upper.astype(float)
        self.initial_pose = self.get_pose()

        self.frame_task = self.solver.add_frame_task("ee_task", self.cfg.target_link)
        self.frame_task.priority = 0
        self.frame_task.weight = 1.0
        self.frame_task.solve_mode = embodik.TaskSolveMode.SCALE
        self.frame_task.allow_min_error_fallback = False

        self.nullspace_task = self.solver.add_posture_task("posture_task")
        self.nullspace_task.priority = 1
        self.nullspace_task.weight = 0.0
        self.nullspace_task.solve_mode = embodik.TaskSolveMode.MIN_ERROR
        self.nullspace_task.allow_min_error_fallback = False
        self.nullspace_task.set_target_configuration(self.q.copy())
        self.nullspace_task.set_controlled_joint_indices([])

        if not self._acceleration_unavailable_reason:
            try:
                self.acceleration_solver = embodik.AccelerationSolver(self.robot)
                self._velocity_collision_lift_options = embodik.VelocityCollisionLiftOptions()
                self._velocity_collision_lift_options.validation_substeps = (
                    DEFAULT_ACCELERATION_COLLISION_VALIDATION_SUBSTEPS
                )
                self.acceleration_position_task = self.acceleration_solver.add_frame_task(
                    "ee_position", self.cfg.target_link, embodik.TaskType.FRAME_POSITION
                )
                self.acceleration_orientation_task = self.acceleration_solver.add_frame_task(
                    "ee_orientation", self.cfg.target_link, embodik.TaskType.FRAME_ORIENTATION
                )
                self.acceleration_nullspace_task = self.acceleration_solver.add_posture_task(
                    "posture_task"
                )
                for task in (self.acceleration_position_task, self.acceleration_orientation_task):
                    task.priority = 0
                    task.weight = 1.0
                    task.solve_mode = embodik.TaskSolveMode.SCALE
                    task.allow_min_error_fallback = False
                self.acceleration_nullspace_task.priority = 1
                self.acceleration_nullspace_task.weight = 0.0
                self.acceleration_nullspace_task.solve_mode = embodik.TaskSolveMode.MIN_ERROR
                self.acceleration_nullspace_task.allow_min_error_fallback = False
                self.acceleration_nullspace_task.set_target_configuration(self.q.copy())
                self.acceleration_nullspace_task.set_controlled_joint_indices([])
            except Exception as exc:
                self.acceleration_solver = None
                self.acceleration_position_task = None
                self.acceleration_orientation_task = None
                self.acceleration_nullspace_task = None
                self._velocity_collision_lift_options = None
                self._acceleration_unavailable_reason = (
                    f"failed to construct AccelerationSolver runtime: {exc}"
                )

        self._step_opts = embodik.PositionStepOptions()

    def supports_solver_level(self, solver_level: str) -> bool:
        if solver_level == SOLVER_LEVEL_VELOCITY:
            return True
        if solver_level == SOLVER_LEVEL_ACCELERATION:
            supported, _ = self.acceleration_collision_status()
            return supported
        return False

    def acceleration_collision_status(self) -> Tuple[bool, str]:
        if self._acceleration_unavailable_reason:
            return False, self._acceleration_unavailable_reason
        if self.acceleration_solver is None:
            return False, "AccelerationSolver runtime is unavailable"
        if not getattr(self, "_collision_enabled", False):
            return True, "self-collision is inactive"
        if self._acceleration_collision_enabled:
            return True, ""
        return False, self._acceleration_collision_reason

    def solver_level_unavailable_reason(self, solver_level: str) -> str:
        if self.supports_solver_level(solver_level):
            return ""
        if solver_level == SOLVER_LEVEL_ACCELERATION:
            _, reason = self.acceleration_collision_status()
            return reason
        return f"unknown solver level '{solver_level}'"

    def set_solver_level(self, solver_level: str) -> None:
        if solver_level != self._solver_level:
            self._acceleration_dq = np.zeros(self.robot.nv, dtype=float)
        if solver_level == SOLVER_LEVEL_ACCELERATION and not self.supports_solver_level(
            solver_level
        ):
            self._solver_level = SOLVER_LEVEL_VELOCITY
            return
        self._solver_level = solver_level

    def get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self.lower[: self.arm_dofs], self.upper[: self.arm_dofs]

    def get_q(self) -> np.ndarray:
        return self.q[: self.arm_dofs].copy()

    def set_q(self, q_arm: np.ndarray) -> None:
        self.q[: self.arm_dofs] = np.clip(
            q_arm, self.lower[: self.arm_dofs], self.upper[: self.arm_dofs]
        )
        self._acceleration_dq = np.zeros(self.robot.nv, dtype=float)
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
        ee_mode: str = "SCALE",
        ee_fallback: bool = False,
        max_steps: int = 1,
        adaptive_dt: bool = False,
        adaptive_dt_max_scale: float = 5.0,
        adaptive_dt_reference_distance: float = 0.05,
        solver_level: str = SOLVER_LEVEL_VELOCITY,
        acceleration_limit: float = DEFAULT_ACCELERATION_SOLVER_LIMIT,
        acceleration_limits_enabled: bool = True,
    ) -> embodiKResult:
        if solver_level == SOLVER_LEVEL_ACCELERATION:
            if self.acceleration_control_unavailable_reason(
                ee_mode=ee_mode,
                ee_fallback=ee_fallback,
                max_steps=max_steps,
                adaptive_dt=adaptive_dt,
                acceleration_limits_enabled=acceleration_limits_enabled,
            ):
                self._acceleration_dq = np.zeros(self.robot.nv, dtype=float)
                return embodiKResult(
                    joints=self.get_q(),
                    status="UNSUPPORTED_CONSTRAINT",
                    position_error=float("nan"),
                    rotation_error=float("nan"),
                    elapsed_ms=0.0,
                    solver_level=SOLVER_LEVEL_ACCELERATION,
                )
            self.set_solver_level(solver_level)
            return self._solve_acceleration_step(
                target,
                pos_gain,
                rot_gain,
                active_indices,
                nullspace_bias,
                nullspace_gain,
                nullspace_enabled,
                acceleration_limit,
            )

        self.set_solver_level(SOLVER_LEVEL_VELOCITY)
        self.frame_task.solve_mode = getattr(
            embodik.TaskSolveMode, ee_mode, embodik.TaskSolveMode.SCALE
        )
        self.frame_task.allow_min_error_fallback = bool(ee_fallback)

        # Nullspace task
        if nullspace_enabled and active_indices and nullspace_gain > 0.0:
            bias_full = self.q.copy()
            for idx in active_indices:
                bias_full[idx] = nullspace_bias[idx]
            self.nullspace_task.set_target_configuration(bias_full)
            self.nullspace_task.set_controlled_joint_indices(active_indices)
            self.nullspace_task.weight = nullspace_gain
        else:
            self.nullspace_task.weight = 0.0
            self.nullspace_task.set_controlled_joint_indices([])

        self._step_opts.position_gain = pos_gain
        self._step_opts.orientation_gain = rot_gain
        self._step_opts.max_steps = max_steps
        self._step_opts.adaptive_dt = adaptive_dt
        self._step_opts.adaptive_dt_max_scale = adaptive_dt_max_scale
        self._step_opts.adaptive_dt_reference_distance = adaptive_dt_reference_distance

        ik_start = time.perf_counter()
        result = self.solver.solve_position_step(self.q, target, "ee_task", self._step_opts)
        elapsed_ms = (time.perf_counter() - ik_start) * 1000.0

        if result.status in (
            embodik.SolverStatus.SUCCESS,
            embodik.SolverStatus.INFEASIBLE,
            embodik.SolverStatus.NUMERICAL_ERROR,
        ):
            self.q = np.asarray(result.q_solution, dtype=float)
            self.robot.update_configuration(self.q)
        # COLLISION_VIOLATED: q_solution is the last safe config — apply it
        # so the robot holds at the safe position rather than freezing entirely.
        elif result.status == embodik.SolverStatus.COLLISION_VIOLATED:
            self.q = np.asarray(result.q_solution, dtype=float)
            self.robot.update_configuration(self.q)

        return embodiKResult(
            joints=self.get_q(),
            status=result.status.name,
            position_error=float(result.position_error),
            rotation_error=float(result.orientation_error),
            elapsed_ms=elapsed_ms,
            primary_mode=(
                result.task_modes_effective[0].name
                if len(result.task_modes_effective) > 0
                else "SCALE"
            ),
            primary_fallback=(
                bool(result.task_used_fallback[0]) if len(result.task_used_fallback) > 0 else False
            ),
            primary_scale=(float(result.task_scales[0]) if len(result.task_scales) > 0 else 1.0),
            collision_time_ms=float(getattr(result, "collision_constraint_time_ms", 0.0)),
            collision_sphere_culled=int(getattr(result, "collision_sphere_culled_pairs", 0)),
            collision_exact_queries=int(getattr(result, "collision_exact_distance_queries", 0)),
            condition_number=(
                float(result.condition_number) if hasattr(result, "condition_number") else 1.0
            ),
            solver_level=SOLVER_LEVEL_VELOCITY,
        )

    @staticmethod
    def acceleration_control_unavailable_reason(
        *,
        ee_mode: str,
        ee_fallback: bool,
        max_steps: int,
        adaptive_dt: bool,
        acceleration_limits_enabled: bool,
    ) -> str:
        if ee_mode != ACCELERATION_EXAMPLE_TASK_MODE:
            return "acceleration mode requires the SCALE task mode; " f"received {ee_mode}"
        if ee_fallback:
            return "acceleration mode does not support MIN_ERROR fallback"
        if max_steps != 1:
            return "acceleration mode advances exactly one explicit state step"
        if adaptive_dt:
            return "acceleration mode uses the fixed solver dt"
        if not acceleration_limits_enabled:
            return "acceleration mode requires native acceleration limits"
        return ""

    def _solve_acceleration_step(
        self,
        target: pin.SE3,
        pos_gain: float,
        rot_gain: float,
        active_indices: List[int],
        nullspace_bias: np.ndarray,
        nullspace_gain: float,
        nullspace_enabled: bool,
        acceleration_limit: float,
    ) -> embodiKResult:
        if not self.supports_solver_level(SOLVER_LEVEL_ACCELERATION):
            self._acceleration_dq = np.zeros(self.robot.nv, dtype=float)
            return embodiKResult(
                joints=self.get_q(),
                status="UNSUPPORTED_CONSTRAINT",
                position_error=float("nan"),
                rotation_error=float("nan"),
                elapsed_ms=0.0,
                solver_level=SOLVER_LEVEL_ACCELERATION,
            )
        assert self.acceleration_solver is not None
        assert self.acceleration_position_task is not None
        assert self.acceleration_orientation_task is not None
        assert self.acceleration_nullspace_task is not None

        self.acceleration_position_task.set_target_position(
            np.asarray(target.translation, dtype=float)
        )
        self.acceleration_orientation_task.set_target_orientation(
            np.asarray(target.rotation, dtype=float)
        )

        if nullspace_enabled and active_indices and nullspace_gain > 0.0:
            bias_full = self.q.copy()
            for idx in active_indices:
                bias_full[idx] = nullspace_bias[idx]
            self.acceleration_nullspace_task.set_target_configuration(bias_full)
            self.acceleration_nullspace_task.set_controlled_joint_indices(active_indices)
            self.acceleration_nullspace_task.weight = nullspace_gain
        else:
            self.acceleration_nullspace_task.weight = 0.0
            self.acceleration_nullspace_task.set_controlled_joint_indices([])

        self.acceleration_solver.set_task_reference(
            "ee_position", _acceleration_reference(3, pos_gain)
        )
        self.acceleration_solver.set_task_reference(
            "ee_orientation", _acceleration_reference(3, rot_gain)
        )
        reference_dimension = (
            len(active_indices)
            if nullspace_enabled and active_indices and nullspace_gain > 0.0
            else self.robot.nv
        )
        self.acceleration_solver.set_task_reference(
            "posture_task",
            _acceleration_reference(reference_dimension, max(float(nullspace_gain), 0.0)),
        )

        options = embodik.AccelerationSolveOptions()
        options.acceleration_limits_override = np.full(
            self.robot.nv, float(acceleration_limit), dtype=float
        )
        options.apply_velocity_limits = True
        options.apply_position_limits = True
        if hasattr(options, "collect_timing_breakdown"):
            options.collect_timing_breakdown = True

        ik_start = time.perf_counter()
        if getattr(self, "_collision_enabled", False):
            assert self._velocity_collision_lift_options is not None
            result = self.acceleration_solver.solve_with_velocity_collision(
                self.solver,
                self.q,
                self._acceleration_dq,
                DEFAULT_SOLVER_DT,
                options,
                self._velocity_collision_lift_options,
            )
        else:
            result = self.acceleration_solver.solve(
                self.q,
                self._acceleration_dq,
                DEFAULT_SOLVER_DT,
                options,
            )
        elapsed_ms = (time.perf_counter() - ik_start) * 1000.0

        if result.status is embodik.SolverStatus.SUCCESS:
            self.q = np.asarray(result.q_solution, dtype=float)
            self._acceleration_dq = np.asarray(result.joint_velocities_next, dtype=float)
            self.robot.update_configuration(self.q)
        else:
            self._acceleration_dq = np.zeros(self.robot.nv, dtype=float)
            self.robot.update_configuration(self.q)

        current_pose = self.get_pose()
        position_error = float(
            np.linalg.norm(
                np.asarray(current_pose.translation, dtype=float)
                - np.asarray(target.translation, dtype=float)
            )
        )
        rotation_error = _rotation_error_rad(
            np.asarray(current_pose.rotation, dtype=float),
            np.asarray(target.rotation, dtype=float),
        )
        diagnostics = getattr(result, "task_diagnostics", [])
        first_diag = diagnostics[0] if len(diagnostics) > 0 else None

        return embodiKResult(
            joints=self.get_q(),
            status=result.status.name,
            position_error=position_error,
            rotation_error=rotation_error,
            elapsed_ms=elapsed_ms,
            primary_mode=(first_diag.effective_mode.name if first_diag is not None else "SCALE"),
            primary_fallback=(
                bool(first_diag.used_min_error_fallback) if first_diag is not None else False
            ),
            primary_scale=(float(first_diag.scale) if first_diag is not None else 1.0),
            collision_time_ms=float(
                getattr(
                    result,
                    "collision_transaction_time_ms",
                    getattr(result, "velocity_collision_orchestration_time_ms", 0.0),
                )
            ),
            collision_exact_queries=int(
                getattr(
                    result,
                    "collision_primitive_distance_queries",
                    getattr(result, "collision_validation_exact_queries", 0),
                )
            ),
            condition_number=float(getattr(result, "condition_number", 1.0)),
            solver_level=SOLVER_LEVEL_ACCELERATION,
            velocity_collision_lift_applied=bool(
                getattr(result, "velocity_collision_lift_applied", False)
            ),
            collision_step_certified=bool(getattr(result, "collision_step_certified", False)),
            collision_validation_samples=int(getattr(result, "collision_validation_samples", 0)),
            velocity_collision_validation_allowed_pairs=int(
                getattr(result, "collision_validation_allowed_pairs", 0)
            ),
            velocity_collision_validation_pairs_checked=int(
                getattr(result, "collision_validation_pairs_checked", 0)
            ),
            velocity_collision_validation_sample_exact_distance_queries=int(
                getattr(result, "collision_validation_exact_queries", 0)
            ),
            velocity_collision_validation_conservative_bound_certified_pairs=int(
                getattr(
                    result,
                    "velocity_collision_validation_conservative_bound_certified_pairs",
                    0,
                )
            ),
        )

    def reset(self) -> pin.SE3:
        self.q = self.default_full.copy()
        self._acceleration_dq = np.zeros(self.robot.nv, dtype=float)
        self.robot.update_configuration(self.q)
        return self.get_pose()

    def _configure_acceleration_collision_constraint_if_possible(self) -> None:
        self._acceleration_collision_enabled = False
        self._acceleration_collision_reason = "velocity-lifted collision smoke check has not run"
        if self._acceleration_unavailable_reason:
            self._acceleration_collision_reason = self._acceleration_unavailable_reason
            return
        smoke_ok, smoke_reason = self._validate_acceleration_collision_example_smoke()
        if smoke_ok:
            self._acceleration_collision_enabled = True
            self._acceleration_collision_reason = ""
        else:
            self._acceleration_collision_reason = smoke_reason

    def _validate_acceleration_collision_example_smoke(self) -> Tuple[bool, str]:
        if (
            self.acceleration_solver is None
            or self.acceleration_position_task is None
            or self.acceleration_orientation_task is None
            or self.acceleration_nullspace_task is None
            or self._velocity_collision_lift_options is None
        ):
            return False, self._acceleration_unavailable_reason or "AccelerationSolver unavailable"

        pose = self.get_pose()
        self.acceleration_position_task.set_target_position(
            np.asarray(pose.translation, dtype=float)
        )
        self.acceleration_orientation_task.set_target_orientation(
            np.asarray(pose.rotation, dtype=float)
        )
        self.acceleration_nullspace_task.set_target_configuration(self.q.copy())
        self.acceleration_nullspace_task.set_controlled_joint_indices(list(range(self.arm_dofs)))
        self.acceleration_nullspace_task.weight = DEFAULT_NULLSPACE_GAIN
        self.acceleration_solver.set_task_reference(
            "ee_position", _acceleration_reference(3, DEFAULT_POS_GAIN)
        )
        self.acceleration_solver.set_task_reference(
            "ee_orientation", _acceleration_reference(3, DEFAULT_ROT_GAIN)
        )
        self.acceleration_solver.set_task_reference(
            "posture_task", _acceleration_reference(self.arm_dofs, DEFAULT_NULLSPACE_GAIN)
        )

        options = embodik.AccelerationSolveOptions()
        options.acceleration_limits_override = np.full(
            self.robot.nv, DEFAULT_ACCELERATION_SOLVER_LIMIT, dtype=float
        )
        options.apply_velocity_limits = True
        options.apply_position_limits = True
        if hasattr(options, "collect_timing_breakdown"):
            options.collect_timing_breakdown = True

        result = self.acceleration_solver.solve_with_velocity_collision(
            self.solver,
            self.q.copy(),
            np.zeros(self.robot.nv, dtype=float),
            DEFAULT_SOLVER_DT,
            options,
            self._velocity_collision_lift_options,
        )
        self._acceleration_dq = np.zeros(self.robot.nv, dtype=float)
        self.robot.update_configuration(self.q)
        if result.status is embodik.SolverStatus.SUCCESS:
            return True, ""
        message = getattr(result, "status_message", "")
        detail = f": {message}" if message else ""
        return (
            False,
            f"acceleration velocity-collision smoke returned {result.status.name}{detail}",
        )

    def enable_self_collision(self, enable: bool) -> None:
        if not hasattr(self, "_collision_enabled"):
            self._collision_enabled = False

        if enable and not self._collision_enabled:
            try:
                apply_collision_tuning_mode(
                    self.solver,
                    getattr(self, "_collision_tuning_mode", DEFAULT_COLLISION_TUNING_MODE),
                )
                self.solver.configure_collision_constraint(
                    min_distance=0.05,
                    include_pairs=[],
                    exclude_pairs=list(self._collision_exclusions),
                )
                self._collision_enabled = True
                self._configure_acceleration_collision_constraint_if_possible()
            except RuntimeError as exc:
                print(f"[embodiK] Collision configuration failed: {exc}")
                self._collision_enabled = False
                self._acceleration_collision_enabled = False
                self._acceleration_collision_reason = str(exc)
        elif not enable and self._collision_enabled:
            self.solver.clear_collision_constraint()
            self._collision_enabled = False
            self._acceleration_collision_enabled = False
            self._acceleration_collision_reason = "self-collision is inactive"

    def set_collision_tuning_mode(self, mode_label: str) -> None:
        self._collision_tuning_mode = mode_label.lower()
        apply_collision_tuning_mode(self.solver, self._collision_tuning_mode)
        if getattr(self, "_collision_enabled", False):
            self._configure_acceleration_collision_constraint_if_possible()


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Interactive GUI
# -----------------------------------------------------------------------------


def run_gui(cfg: RobotConfig, args: argparse.Namespace) -> None:
    backend = embodiKBackend(cfg)
    # Always enable timing breakdown so collision stats are populated.
    try:
        backend.solver.enable_timing_breakdown(True)
    except Exception as exc:  # pragma: no cover
        print(f"[embodiK] Warning: enable_timing_breakdown failed: {exc}")

    bp_status = (
        "ON" if getattr(backend.solver, "sphere_broadphase_enabled", lambda: False)() else "OFF"
    )
    print(f"[embodiK] Sphere broadphase: {bp_status}")

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

    gpu_wbc_solver: GpuWbcMultiFrameSolver | None = None
    gpu_target_offset: pin.SE3 | None = None
    if args.gpu_wbc:
        print("[GPU WBC] Initializing the fail-closed Newton/Warp backend...")
        gpu_robot = embodik.RobotModel(
            str(cfg.urdf_path),
            actuated_joint_names=tuple(cfg.joint_names),
            floating_base=False,
        )
        gpu_posture_indices = tuple(
            int(gpu_robot.get_joint_velocity_index(name))
            for name in cfg.joint_names
        )
        gpu_pairs = gpu_collision_include_pairs(
            gpu_robot, backend._collision_exclusions
        )
        gpu_wbc_solver = GpuWbcMultiFrameSolver(
            args.gpu_wbc_manifest,
            cfg.urdf_path,
            args.gpu_wbc_cache_dir,
            robot=gpu_robot,
            robot_name=cfg.key,
            frames=(cfg.target_link,),
            frame_task_dimensions=(6,),
            active_joint_names=tuple(cfg.joint_names),
            default_configuration=q_current,
            solver_backend="torch_srinv",
            iterations=2,
            dt=DEFAULT_SOLVER_DT,
            position_gain=DEFAULT_POS_GAIN,
            orientation_gain=DEFAULT_ROT_GAIN,
            max_joint_acceleration_rad_s2=4.0,
            adaptive_dt=DEFAULT_ADAPTIVE_DT,
            adaptive_dt_max_scale=DEFAULT_ADAPTIVE_DT_MAX_SCALE,
            adaptive_dt_reference_distance=DEFAULT_ADAPTIVE_DT_REFERENCE_DISTANCE,
            collision_pairs=gpu_pairs,
            collision_min_distance_m=0.03,
            collision_query_distance_m=0.11,
            posture_target_configuration=tuple(q_current),
            posture_velocity_indices=gpu_posture_indices,
            posture_weights=tuple(1.0 for _ in gpu_posture_indices),
            posture_gain=0.0,
        )
        visible_target_pose = backend.get_pose()
        gpu_tcp_pose = backend.robot.get_frame_pose(gpu_wbc_solver.frames[0])
        gpu_target_offset = visible_target_pose.inverse() * gpu_tcp_pose
        warmup_ms = gpu_wbc_solver.warm_up(
            q_current, (visible_target_pose * gpu_target_offset,)
        )
        print(
            f"[GPU WBC] Ready on {gpu_wbc_solver.device_label}; "
            f"one-time warm-up took {warmup_ms:.1f} ms."
        )

    urdf = load_robot_description(cfg.description_name)
    server = viser.ViserServer(port=args.port)
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
    initial_wxyz = (
        initial_quat_xyzw[3],
        initial_quat_xyzw[0],
        initial_quat_xyzw[1],
        initial_quat_xyzw[2],
    )

    ik_target = server.scene.add_transform_controls(
        "/ik_target",
        scale=0.2,
        position=tuple(pose.translation),
        wxyz=initial_wxyz,
    )

    acceleration_available, acceleration_reason = backend.acceleration_collision_status()
    solver_level_options = (
        COLLISION_EXAMPLE_SOLVER_LEVEL_OPTIONS
        if acceleration_available
        else (SOLVER_LEVEL_VELOCITY,)
    )

    with server.gui.add_folder("IK Controls"):
        solver_level_dropdown = server.gui.add_dropdown(
            "Solver Level",
            options=solver_level_options,
            initial_value=SOLVER_LEVEL_VELOCITY,
        )
        solver_options = [CPU_SOLVER_LABEL]
        if gpu_wbc_solver is not None:
            solver_options.append(GPU_SOLVER_LABEL)
        solver_backend_dropdown = server.gui.add_dropdown(
            "Solver Backend",
            options=tuple(solver_options),
            initial_value=CPU_SOLVER_LABEL,
        )
        solver_contract_text = server.gui.add_text(
            "Backend Contract",
            initial_value="CPU: full Example 02 controls",
        )
        timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
        pos_gain = server.gui.add_slider(
            "Position Gain", min=0.1, max=200.0, initial_value=DEFAULT_POS_GAIN, step=0.1
        )
        rot_gain = server.gui.add_slider(
            "Orientation Gain", min=0.1, max=200.0, initial_value=DEFAULT_ROT_GAIN, step=0.1
        )
        iterations_slider = server.gui.add_slider(
            "IK Iterations", min=1, max=20, initial_value=1, step=1
        )
        adaptive_dt_checkbox = server.gui.add_checkbox(
            "Adaptive dt", initial_value=DEFAULT_ADAPTIVE_DT
        )
        adaptive_dt_max_scale_slider = server.gui.add_slider(
            "Adaptive dt Max Scale",
            min=1.0,
            max=10.0,
            step=0.5,
            initial_value=DEFAULT_ADAPTIVE_DT_MAX_SCALE,
        )
        adaptive_dt_ref_dist_slider = server.gui.add_slider(
            "Adaptive dt Ref Dist (m)",
            min=0.01,
            max=0.20,
            step=0.01,
            initial_value=DEFAULT_ADAPTIVE_DT_REFERENCE_DISTANCE,
        )
        nullspace_enabled_checkbox = server.gui.add_checkbox(
            "Enable Nullspace Bias",
            initial_value=DEFAULT_NULLSPACE_ENABLED,
        )
        nullspace_gain = server.gui.add_slider(
            "Nullspace Gain", min=0.0, max=2.0, initial_value=DEFAULT_NULLSPACE_GAIN, step=0.05
        )
        self_collision_checkbox = server.gui.add_checkbox(
            "Enable Self-Collision",
            initial_value=True,
            disabled=not hasattr(backend, "enable_self_collision"),
        )
        collision_tuning_dropdown = server.gui.add_dropdown(
            "Collision Tuning",
            options=COLLISION_TUNING_OPTIONS,
            initial_value="balanced",
            disabled=not hasattr(backend, "set_collision_tuning_mode"),
        )
        ee_mode_dropdown = server.gui.add_dropdown(
            "EE Solve Mode",
            options=("SCALE", "SCALE_ELASTIC", "MIN_ERROR"),
            initial_value="SCALE_ELASTIC",
        )
        ee_fallback_checkbox = server.gui.add_checkbox(
            "Allow SCALE fallback to MIN_ERROR",
            initial_value=False,
        )
        collision_debug_checkbox = server.gui.add_checkbox(
            "Show Collision Debug",
            initial_value=True,
            disabled=not (
                hasattr(backend, "solver") and hasattr(backend.solver, "get_last_collision_debug")
            ),
        )
        collision_debug_text = server.gui.add_text("Collision Debug", initial_value="Collision: --")
        manual_control = server.gui.add_checkbox("Manual Joint Control", initial_value=False)
        status_initial = (
            "Status: Ready"
            if acceleration_available
            else f"Status: Acceleration unavailable ({acceleration_reason})"
        )
        status_handle = server.gui.add_text("Status", initial_value=status_initial)
        snap_target_button = server.gui.add_button("Snap Target to Current EE")
        reset_robot_button = server.gui.add_button("Reset Robot & Target")

    with server.gui.add_folder("Solver Diagnostics", expand_by_default=False):
        accel_limit_checkbox = server.gui.add_checkbox(
            "Acceleration Limits",
            initial_value=False,
        )
        accel_limit_slider = server.gui.add_slider(
            "Max Accel (rad/s^2)",
            min=1.0,
            max=50.0,
            initial_value=15.0,
            step=1.0,
        )
        conditioning_text = server.gui.add_text(
            "IK Conditioning",
            initial_value="condition: --",
        )

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
    gpu_collision_debug = None

    urdf_vis.update_cfg(make_visual_config(q_current))

    def gpu_backend_selected() -> bool:
        return solver_backend_dropdown.value == GPU_SOLVER_LABEL

    def update_collision_visuals() -> None:
        nonlocal collision_line_handle, last_collision_debug, collision_log_timestamp, collision_state
        nonlocal collision_debug_enabled, gpu_collision_debug
        solver_obj = getattr(backend, "solver", None)
        now = time.time()
        gpu_selected = gpu_backend_selected()
        cpu_debug_supported = solver_obj is not None and hasattr(
            solver_obj, "get_last_collision_debug"
        )

        debug_requested = (
            not collision_debug_checkbox.disabled
            and collision_debug_checkbox.value
            and self_collision_checkbox.value
            and (
                gpu_wbc_solver is not None and gpu_wbc_solver.collision_supported
                if gpu_selected
                else cpu_debug_supported
            )
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

        if not gpu_selected and not cpu_debug_supported:
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

        debug_info = gpu_collision_debug if gpu_selected else solver_obj.get_last_collision_debug()
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
            or now - collision_log_timestamp > COLLISION_DEBUG_LOG_PERIOD_S
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

    velocity_control_state = {
        "ee_mode": str(ee_mode_dropdown.value),
        "ee_fallback": bool(ee_fallback_checkbox.value),
        "iterations": int(iterations_slider.value),
        "adaptive_dt": bool(adaptive_dt_checkbox.value),
        "acceleration_limits": bool(accel_limit_checkbox.value),
    }

    def sync_solver_specific_controls() -> None:
        acceleration_selected = str(solver_level_dropdown.value) == SOLVER_LEVEL_ACCELERATION
        if acceleration_selected:
            if not ee_mode_dropdown.disabled:
                velocity_control_state["ee_mode"] = str(ee_mode_dropdown.value)
                velocity_control_state["ee_fallback"] = bool(ee_fallback_checkbox.value)
                velocity_control_state["iterations"] = int(iterations_slider.value)
                velocity_control_state["adaptive_dt"] = bool(adaptive_dt_checkbox.value)
                velocity_control_state["acceleration_limits"] = bool(accel_limit_checkbox.value)
            ee_mode_dropdown.value = ACCELERATION_EXAMPLE_TASK_MODE
            ee_fallback_checkbox.value = False
            iterations_slider.value = 1
            adaptive_dt_checkbox.value = False
            accel_limit_checkbox.value = True
        else:
            restored_ee_mode = str(velocity_control_state["ee_mode"])
            if restored_ee_mode == "SCALE_ELASTIC":
                ee_mode_dropdown.value = "SCALE_ELASTIC"
            elif restored_ee_mode == "MIN_ERROR":
                ee_mode_dropdown.value = "MIN_ERROR"
            else:
                ee_mode_dropdown.value = "SCALE"
            ee_fallback_checkbox.value = bool(velocity_control_state["ee_fallback"])
            iterations_slider.value = int(velocity_control_state["iterations"])
            adaptive_dt_checkbox.value = bool(velocity_control_state["adaptive_dt"])
            accel_limit_checkbox.value = bool(velocity_control_state["acceleration_limits"])

        for control in (
            ee_mode_dropdown,
            ee_fallback_checkbox,
            iterations_slider,
            adaptive_dt_checkbox,
            adaptive_dt_max_scale_slider,
            adaptive_dt_ref_dist_slider,
        ):
            control.disabled = acceleration_selected
        accel_limit_checkbox.disabled = acceleration_selected

    def sync_solver_level_options() -> None:
        acceleration_ok, reason = backend.acceleration_collision_status()
        next_options = (
            COLLISION_EXAMPLE_SOLVER_LEVEL_OPTIONS if acceleration_ok else (SOLVER_LEVEL_VELOCITY,)
        )
        if tuple(solver_level_dropdown.options) != tuple(next_options):
            solver_level_dropdown.options = next_options
        if not acceleration_ok:
            if solver_level_dropdown.value != SOLVER_LEVEL_VELOCITY:
                solver_level_dropdown.value = SOLVER_LEVEL_VELOCITY
            backend.set_solver_level(SOLVER_LEVEL_VELOCITY)
            status_handle.value = f"Status: Acceleration unavailable ({reason})"
        sync_solver_specific_controls()

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
        if hasattr(backend, "enable_self_collision") and not gpu_backend_selected():
            backend.enable_self_collision(
                self_collision_checkbox.value and not self_collision_checkbox.disabled
            )
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
        sync_solver_level_options()

    cpu_only_controls = (ee_mode_dropdown, ee_fallback_checkbox)
    gpu_runtime_fault: str | None = None

    def update_backend_controls() -> None:
        gpu_selected = gpu_backend_selected()
        for handle in cpu_only_controls:
            handle.disabled = gpu_selected
        self_collision_checkbox.disabled = (
            not gpu_wbc_solver.collision_supported
            if gpu_selected and gpu_wbc_solver is not None
            else not hasattr(backend, "enable_self_collision")
        )
        collision_tuning_dropdown.disabled = gpu_selected or not hasattr(
            backend, "set_collision_tuning_mode"
        )
        collision_debug_checkbox.disabled = (
            gpu_wbc_solver is None or not gpu_wbc_solver.collision_supported
            if gpu_selected
            else not (
                hasattr(backend, "solver") and hasattr(backend.solver, "get_last_collision_debug")
            )
        )
        if gpu_selected:
            if gpu_wbc_solver is not None and gpu_wbc_solver.collision_supported:
                solver_contract_text.value = (
                    "GPU: directional SRINV + limits, posture, adaptive dt, "
                    "runtime collision, no fallback"
                )
                collision_debug_text.value = "Collision: see GPU status diagnostics"
            else:
                solver_contract_text.value = (
                    "GPU: model-derived pose + limits, collision unavailable, no fallback"
                )
                collision_debug_text.value = "Collision: unsupported for this GPU model"
        else:
            solver_contract_text.value = "CPU: full Example 02 controls"

    @solver_backend_dropdown.on_update
    def _(_evt) -> None:
        nonlocal gpu_runtime_fault, gpu_collision_debug
        gpu_runtime_fault = None
        gpu_collision_debug = None
        if gpu_wbc_solver is not None:
            gpu_wbc_solver.reset_state()
        update_backend_controls()
        if not gpu_backend_selected():
            backend.set_collision_tuning_mode(collision_tuning_dropdown.value)
            backend.enable_self_collision(self_collision_checkbox.value)
        update_collision_visuals()
        status_handle.value = f"Status: Switched to {solver_backend_dropdown.value}"

    sync_from_backend(update_target=True)
    update_backend_controls()
    if hasattr(backend, "set_collision_tuning_mode"):
        backend.set_collision_tuning_mode(collision_tuning_dropdown.value)

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
        if gpu_wbc_solver is not None:
            gpu_wbc_solver.reset_state()
        sync_from_backend(update_target=True)
        status_handle.value = "Status: Robot reset to default configuration"

    @self_collision_checkbox.on_update
    def _(_evt) -> None:
        if gpu_backend_selected() and gpu_wbc_solver is not None:
            gpu_wbc_solver.configure_runtime(
                collision_enabled=bool(self_collision_checkbox.value)
            )
            gpu_wbc_solver.reset_state()
        elif hasattr(backend, "enable_self_collision"):
            if hasattr(backend, "set_collision_tuning_mode"):
                backend.set_collision_tuning_mode(collision_tuning_dropdown.value)
            backend.enable_self_collision(
                self_collision_checkbox.value and not self_collision_checkbox.disabled
            )
            solver_obj = getattr(backend, "solver", None)
            if (
                collision_debug_checkbox.value
                and self_collision_checkbox.value
                and solver_obj is not None
                and hasattr(solver_obj, "get_active_collision_pairs")
            ):
                active_pairs = solver_obj.get_active_collision_pairs()
                print(f"[embodiK] Active collision pairs: {len(active_pairs)}")
        sync_solver_level_options()
        if not collision_debug_checkbox.value:
            collision_point_a.visible = False
            collision_point_b.visible = False
            if collision_line_handle is not None:
                collision_line_handle.visible = False
        update_collision_visuals()

    @collision_tuning_dropdown.on_update
    def _(_evt) -> None:
        if hasattr(backend, "set_collision_tuning_mode"):
            backend.set_collision_tuning_mode(collision_tuning_dropdown.value)
        sync_solver_level_options()
        if backend.supports_solver_level(SOLVER_LEVEL_ACCELERATION):
            status_handle.value = (
                f"Status: Collision tuning set to {collision_tuning_dropdown.value}"
            )

    @solver_level_dropdown.on_update
    def _(_evt) -> None:
        requested = str(solver_level_dropdown.value)
        if not backend.supports_solver_level(requested):
            reason = backend.solver_level_unavailable_reason(requested)
            solver_level_dropdown.value = SOLVER_LEVEL_VELOCITY
            backend.set_solver_level(SOLVER_LEVEL_VELOCITY)
            status_handle.value = f"Status: Acceleration unavailable ({reason})"
            return
        backend.set_solver_level(requested)
        sync_solver_specific_controls()
        if requested == SOLVER_LEVEL_ACCELERATION:
            status_handle.value = (
                "Status: Solver level set to Acceleration "
                "(SCALE, one fixed-dt step, native acceleration limits)"
            )
        else:
            status_handle.value = f"Status: Solver level set to {requested}"

    @accel_limit_checkbox.on_update
    def _(_evt) -> None:
        if str(solver_level_dropdown.value) == SOLVER_LEVEL_ACCELERATION:
            if not accel_limit_checkbox.value:
                accel_limit_checkbox.value = True
            status_handle.value = (
                "Status: Native acceleration limits are required in Acceleration mode"
            )
            return
        velocity_control_state["acceleration_limits"] = bool(accel_limit_checkbox.value)
        backend.solver.enable_acceleration_limits(accel_limit_checkbox.value)
        if accel_limit_checkbox.value:
            backend.solver.set_acceleration_limits(
                np.full(backend.robot.nv, accel_limit_slider.value)
            )
        status_handle.value = (
            f"Status: Accel limits {'ON' if accel_limit_checkbox.value else 'OFF'}"
        )

    @accel_limit_slider.on_update
    def _(_evt) -> None:
        if str(solver_level_dropdown.value) == SOLVER_LEVEL_VELOCITY and accel_limit_checkbox.value:
            backend.solver.set_acceleration_limits(
                np.full(backend.robot.nv, accel_limit_slider.value)
            )

    @collision_debug_checkbox.on_update
    def _(_evt) -> None:
        update_collision_visuals()

    iteration_count = 0
    while True:
        solver_elapsed_ms = 0.0
        result: embodiKResult | None = None

        if manual_control.value:
            if not prev_manual_state:
                # entering manual mode, ensure sliders reflect current joint state
                q_current = backend.get_q()
                for slider, value in zip(joint_sliders, q_current):
                    slider.value = float(value)
            q_current = np.array([slider.value for slider in joint_sliders], dtype=float)
            backend.set_q(q_current)
            if gpu_wbc_solver is not None:
                gpu_wbc_solver.reset_state()
            status_handle.value = (
                f"Status: Manual joint control active ({solver_backend_dropdown.value})"
            )
        else:
            target_position = np.array(ik_target.position, dtype=float)
            target_wxyz = np.array(ik_target.wxyz, dtype=float)
            target_xyzw = np.array([target_wxyz[1], target_wxyz[2], target_wxyz[3], target_wxyz[0]])
            target_rotation = q2r(target_xyzw, order="xyzs")
            target_pose = Rt(R=target_rotation, t=target_position)

            if gpu_backend_selected():
                assert gpu_wbc_solver is not None
                assert gpu_target_offset is not None
                if prev_manual_state:
                    gpu_wbc_solver.reset_state()
                if gpu_runtime_fault is None:
                    try:
                        gpu_posture_weights = tuple(
                            1.0 if checkbox.value else 0.0
                            for checkbox in nullspace_checkboxes
                        )
                        gpu_wbc_solver.configure_runtime(
                            iterations=int(iterations_slider.value),
                            frame_position_gains=(float(pos_gain.value),),
                            frame_orientation_gains=(float(rot_gain.value),),
                            adaptive_dt=bool(adaptive_dt_checkbox.value),
                            adaptive_dt_max_scale=float(
                                adaptive_dt_max_scale_slider.value
                            ),
                            adaptive_dt_reference_distance=float(
                                adaptive_dt_ref_dist_slider.value
                            ),
                            acceleration_limits_enabled=bool(
                                accel_limit_checkbox.value
                            ),
                            max_joint_acceleration_rad_s2=float(
                                accel_limit_slider.value
                            ),
                            collision_enabled=bool(
                                self_collision_checkbox.value
                            ),
                            posture_target_configuration=tuple(nullspace_bias),
                            posture_weights=gpu_posture_weights,
                            posture_gain=(
                                float(nullspace_gain.value)
                                if nullspace_enabled_checkbox.value
                                else 0.0
                            ),
                        )
                        gpu_result = gpu_wbc_solver.solve_step(
                            q_current,
                            (target_pose * gpu_target_offset,),
                            include_collision_debug=bool(
                                collision_debug_checkbox.value
                                and self_collision_checkbox.value
                            ),
                        )
                    except Exception as exc:  # Fail closed: hold; never invoke CPU implicitly.
                        gpu_runtime_fault = f"{type(exc).__name__}: {exc}"
                        gpu_collision_debug = None
                        status_handle.value = f"Status: GPU FAULT — SAFE HOLD | {gpu_runtime_fault}"
                    else:
                        gpu_collision_debug = gpu_result.collision_debug
                        q_current = np.asarray(gpu_result.joints, dtype=float)
                        backend.set_q(q_current)
                        solver_elapsed_ms = gpu_result.elapsed_ms
                        result = embodiKResult(
                            joints=q_current,
                            status=gpu_result.status,
                            position_error=max(gpu_result.position_errors),
                            rotation_error=max(gpu_result.rotation_errors),
                            elapsed_ms=gpu_result.elapsed_ms,
                            primary_mode="GPU_DIRECTIONAL_SRINV",
                        )
                        collision_text = "collision=unsupported"
                        if gpu_result.minimum_collision_distance_m is not None:
                            active_text = (
                                "active" if gpu_result.collision_active else "clear"
                            )
                            collision_text = (
                                f"dmin={gpu_result.minimum_collision_distance_m*1e3:.1f} mm "
                                f"({active_text})"
                            )
                        status_handle.value = (
                            f"Status: GPU {gpu_result.status} | "
                            f"pos={max(gpu_result.position_errors)*1e3:.2f} mm, "
                            f"rot={max(gpu_result.rotation_errors):.4f} rad | "
                            f"{collision_text} | wall={gpu_result.elapsed_ms:.2f}ms, "
                            f"FI={gpu_result.kernel_time_ms:.2f}ms"
                        )
                        conditioning_text.value = (
                            "GPU hierarchy: directional SRINV + posture/nullspace"
                        )
            else:
                gpu_collision_debug = None
                active_indices = [
                    i for i, checkbox in enumerate(nullspace_checkboxes) if checkbox.value
                ]
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
                    ee_mode=ee_mode_dropdown.value,
                    ee_fallback=ee_fallback_checkbox.value,
                    max_steps=int(iterations_slider.value),
                    adaptive_dt=bool(adaptive_dt_checkbox.value),
                    adaptive_dt_max_scale=float(adaptive_dt_max_scale_slider.value),
                    adaptive_dt_reference_distance=float(adaptive_dt_ref_dist_slider.value),
                    solver_level=str(solver_level_dropdown.value),
                    acceleration_limit=float(accel_limit_slider.value),
                    acceleration_limits_enabled=bool(accel_limit_checkbox.value),
                )
                q_current = result.joints
                solver_elapsed_ms = result.elapsed_ms
                col_ms = result.collision_time_ms
                status_prefix = (
                    "⚠ COLLISION_VIOLATED"
                    if result.status == "COLLISION_VIOLATED"
                    else f"embodiK {result.status}"
                )
                cert_suffix = ""
                if (
                    result.solver_level == SOLVER_LEVEL_ACCELERATION
                    and result.velocity_collision_lift_applied
                ):
                    cert_suffix = (
                        f" | sampled-lift={result.collision_validation_samples}"
                        f" cert={result.collision_step_certified}"
                    )
                adt_suffix = (
                    f" | adt×{adaptive_dt_max_scale_slider.value:.1f}"
                    if adaptive_dt_checkbox.value
                    else ""
                )
                status_handle.value = (
                    f"Status: {status_prefix} | "
                    f"pos={result.position_error*1e3:.2f} mm, "
                    f"rot={result.rotation_error:.4f} rad | "
                    f"mode={result.primary_mode}, scale={result.primary_scale:.3f} | "
                    f"col={col_ms:.2f}ms{cert_suffix}{adt_suffix}"
                )
                conditioning_text.value = f"condition: {result.condition_number:.1f}"

            for slider, value in zip(joint_sliders, q_current):
                slider.value = float(value)

        prev_manual_state = manual_control.value

        urdf_vis.update_cfg(make_visual_config(q_current))
        update_collision_visuals()

        timing_handle.value = 0.9 * timing_handle.value + 0.1 * solver_elapsed_ms

        # Optional performance reporting (CLI-controlled)
        iteration_count += 1
        if (
            args.perf_log != "off"
            and args.perf_every > 0
            and iteration_count % args.perf_every == 0
        ):
            # Note: solve_step currently measures wall time around single-step position IK.
            # If C++ timing breakdown is enabled, fetch it from an extra solve call
            # would be intrusive; instead we show the wall-time here and rely on
            # collision_debug/self_collision toggles for deeper analysis.
            col_info = ""
            if not manual_control.value and result is not None:
                col_info = (
                    f" | collision={result.collision_time_ms:.3f}ms"
                    f" sphere_culled={result.collision_sphere_culled}"
                    f" exact={result.collision_exact_queries}"
                )
            print(
                f"[Performance] Iter {iteration_count}: solve_step={solver_elapsed_ms:.3f}ms{col_info}"
            )

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
    parser.add_argument(
        "--perf-log",
        choices=["off", "basic", "verbose"],
        default="off",
        help="Performance logging mode. 'basic' prints periodic summaries, "
        "'verbose' also prints extra warnings.",
    )
    parser.add_argument(
        "--perf-every",
        type=int,
        default=200,
        help="Print performance summary every N iterations when --perf-log is enabled.",
    )
    parser.add_argument(
        "--timing-breakdown",
        action="store_true",
        help="Enable C++ timing breakdown fields in VelocitySolverResult (debug).",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_VISER_PORT, help="Viser server port.")
    parser.add_argument(
        "--gpu-wbc",
        "--gpu-wbc-interactive",
        dest="gpu_wbc",
        action="store_true",
        help=(
            "Initialize the experimental Newton/Warp GPU-WBC backend and add "
            "a CPU/GPU dropdown to Viser (no CPU fallback)."
        ),
    )
    parser.add_argument(
        "--gpu-wbc-manifest",
        type=Path,
        default=None,
        help="Optional legacy manifest used to validate the loaded model contract.",
    )
    parser.add_argument(
        "--gpu-wbc-cache-dir",
        type=Path,
        default=Path("build/gpu-wbc-newton-cache"),
        help="Cache directory for the stripped Newton Panda model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = resolve_robot_configuration(args.robot)
    print(f"Interactive collision-aware IK demo ({cfg.display_name})")
    print(f"  - URDF path: {cfg.urdf_path}")
    print(f"  - Target link: {cfg.target_link}")

    print(f"  - GPU-WBC selector: {'ENABLED' if args.gpu_wbc else 'disabled'}")

    run_gui(cfg, args)


if __name__ == "__main__":  # pragma: no cover
    main()
