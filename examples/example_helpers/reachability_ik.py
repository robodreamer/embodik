"""IK-based reachability analysis module.

This module provides functionality for generating reachability maps using
inverse kinematics (IK) to test reachability at sampled Cartesian positions.
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing
import os
import sys
import threading
import time
import warnings
from collections import deque
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Set
from types import SimpleNamespace

import re

import h5py
import numpy as np
import viser
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

try:
    import torch
except ImportError:
    torch = None

import embodik
from .reachability_robot_configs import RobotConfig, ensure_ros_package_path
from .reachability_metrics import ReachabilityMetricHelper
from .reachability_scoring import compute_ik_global_scores
from .reachability_constants import (
    RANGE_OF_MOTION_NORMALIZED_LIMIT,
    IK_REACH_MAP_XYZ_START,
    IK_REACH_MAP_XYZ_END,
    IK_REACH_MAP_RPY_START,
    IK_REACH_MAP_RPY_END,
    IK_REACH_MAP_VISITATION_COL,
    IK_REACH_MAP_MANIP_COL,
    IK_REACH_MAP_ROM_COL,
    IK_REACH_MAP_SINGULARITY_COL,
    IK_REACH_MAP_QUAT_START,
    IK_REACH_MAP_QUAT_END,
    SPHERE_DATASET_MANIP_COL,
    SPHERE_DATASET_VISITATION_COL,
    SPHERE_DATASET_ROM_COL,
    SPHERE_DATASET_SINGULARITY_COL,
    IK_PROCESS_VISUALIZATION_DELAY_SEC,
)
from embodik.utils import (
    PoseData,
    compute_pose_error,
    limit_task_velocity,
)
from embodik import r2q

LOGGER = logging.getLogger("embodik.reachability_ik")
_worker_limit_overrides: Dict[str, Tuple[float, float]] = {}
_worker_joint_index_map: Dict[str, int] = {}


def _strip_xml_prolog_and_comments(xml: str) -> str:
    """Remove XML declaration and leading comments to satisfy third-party parsers."""
    if not xml:
        return xml
    stripped = xml.lstrip()
    if stripped.startswith("<?xml"):
        stripped = re.sub(r'^<\?xml[^>]*\?>', "", stripped, count=1).lstrip()
    # Remove leading comments (often inserted by xacro)
    stripped = re.sub(r'^\s*<!--.*?-->\s*', "", stripped, flags=re.DOTALL)
    return stripped


def _apply_limit_overrides_array(
    joint_names: Sequence[str],
    lower: Sequence[float],
    upper: Sequence[float],
    overrides: Optional[Dict[str, Tuple[float, float]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply joint limit overrides to lower/upper arrays."""

    lower_arr = np.asarray(lower, dtype=float).copy()
    upper_arr = np.asarray(upper, dtype=float).copy()
    if not overrides:
        return lower_arr, upper_arr

    index_map = {name: idx for idx, name in enumerate(joint_names)}
    for joint_name, (new_min, new_max) in overrides.items():
        idx = index_map.get(joint_name)
        if idx is None:
            continue
        lower_arr[idx] = float(new_min)
        upper_arr[idx] = float(new_max)
    return lower_arr, upper_arr

# Context manager to suppress URDF parsing warnings
@contextmanager
def suppress_urdf_warnings():
    """Temporarily suppress URDF parsing warnings from C++ code."""
    # Suppress Python warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Redirect stderr at file descriptor level to catch C++ output
        original_stderr_fd = sys.stderr.fileno()
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            # Save original stderr
            saved_stderr_fd = os.dup(original_stderr_fd)
            # Redirect stderr to devnull
            os.dup2(devnull_fd, original_stderr_fd)
            yield
        finally:
            # Restore original stderr
            os.dup2(saved_stderr_fd, original_stderr_fd)
            os.close(saved_stderr_fd)
            os.close(devnull_fd)

__all__ = [
    "OrientationMode",
    "GridTraversalMode",
    "IKReachabilityConfig",
    "generate_ik_reachability_map",
    "ik_job_name",
    "generate_cartesian_grid",
    "get_target_orientations",
    "create_grid_preview_file",
]


class OrientationMode(str, Enum):
    """Allowed orientation sampling modes."""

    FIXED = "fixed"
    REFERENCE = "reference"
    MULTIPLE = "multiple"


class GridTraversalMode(str, Enum):
    """Ordering strategies for visiting grid points."""

    RAW = "raw"
    DISTANCE = "distance"
    AXIS_SWEEP = "axis_sweep"


@dataclass
class IKReachabilityConfig:
    """Configuration for IK-based reachability analysis."""

    # Grid parameters
    grid_step_size: float = 0.05  # meters (defaults to cartesian_resolution, can be finer)
    x_limits: Tuple[float, float] = (-0.8, 0.8)
    y_limits: Tuple[float, float] = (-0.8, 0.8)
    z_limits: Tuple[float, float] = (0.0, 1.3)
    reach_radius: Optional[float] = None  # None = no filtering

    # Orientation handling
    orientation_mode: OrientationMode = OrientationMode.REFERENCE
    reference_orientation: Optional[np.ndarray] = None  # 3x3 rotation matrix
    orientation_samples: int = 1  # For "multiple" mode

    # IK solver options
    position_tolerance: float = 5e-3
    orientation_tolerance: float = 1e-1
    max_iterations: int = 100
    dt: float = 0.05

    # Nullspace control
    use_nullspace_bias: bool = False
    nullspace_bias: Optional[np.ndarray] = None  # Joint configuration
    nullspace_gain: float = 1e-2
    nullspace_active_joints: Optional[List[int]] = None

    # Step size limits
    max_linear_step: float = 0.2
    max_angular_step: float = 0.2

    # Output format
    cartesian_resolution: float = 0.20  # For binning (matches FK approach). Minimum: 0.01m (1cm)
    angular_resolution: float = np.pi / 6.0  # For binning
    manip_scaling: float = 500.0
    job_name: Optional[str] = None
    traversal_mode: GridTraversalMode = GridTraversalMode.DISTANCE

    # Parallelization
    num_workers: int = 1  # Number of parallel workers (1 = sequential)

    # Retry strategy (sequential mode only for now)
    retry_attempts: int = 3  # Total attempts per grid point
    retry_seed_pool: int = 4  # Additional seeds to try from cache
    retry_seed_jitter: float = 0.05  # Random jitter (rad) applied when generating fallback seeds
    retry_tolerance_scaling: float = 2.0  # Multiplier per failed attempt
    retry_iteration_scaling: float = 1.0  # Multiplier per failed attempt (kept for compatibility)
    max_retry_position_tolerance: Optional[float] = None  # Absolute clamp (meters)
    max_retry_orientation_tolerance: Optional[float] = None  # Absolute clamp (radians)
    max_retry_iterations: Optional[int] = None  # Absolute clamp on iterations
    seed_score_metric: str = "singularity"  # Which metric to prioritize when ranking cached seeds
    fk_envelope_pruning: bool = True
    fk_envelope_samples: int = 1500
    fk_envelope_margin: float = 0.05
    use_interactive_marker_solver: bool = False
    interactive_pos_gain: float = 30.0
    interactive_rot_gain: float = 30.0
    interactive_max_iterations: int = 200
    interactive_stagnation_threshold: float = 1e-5
    interactive_stagnation_window: int = 5
    interactive_path_resolution: float = 0.01

    def __post_init__(self) -> None:
        if isinstance(self.orientation_mode, str):
            self.orientation_mode = OrientationMode(self.orientation_mode.lower())
        if isinstance(self.seed_score_metric, Enum):
            self.seed_score_metric = self.seed_score_metric.value  # type: ignore[assignment]
        else:
            self.seed_score_metric = str(self.seed_score_metric or "singularity").lower()
        if isinstance(self.traversal_mode, str):
            self.traversal_mode = GridTraversalMode(self.traversal_mode.lower())
        self.fk_envelope_samples = max(0, int(self.fk_envelope_samples))
        self.fk_envelope_margin = max(0.0, float(self.fk_envelope_margin))
        self.interactive_pos_gain = max(0.0, float(self.interactive_pos_gain))
        self.interactive_rot_gain = max(0.0, float(self.interactive_rot_gain))
        self.interactive_max_iterations = max(1, int(self.interactive_max_iterations))
        self.interactive_stagnation_threshold = max(0.0, float(self.interactive_stagnation_threshold))
        self.interactive_stagnation_window = max(1, int(self.interactive_stagnation_window))
        self.interactive_path_resolution = max(1e-4, float(self.interactive_path_resolution))


def ik_job_name(robot_key: str, config: IKReachabilityConfig) -> str:
    """Generate a job name for IK reachability analysis."""
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    return f"{robot_key}_ik_{config.grid_step_size:.3f}_{timestamp}"


def generate_cartesian_grid(
    config: IKReachabilityConfig,
    reference_position: np.ndarray,
    reach_radius: Optional[float] = None,
) -> np.ndarray:
    """Generate 3D Cartesian grid points.

    Matches MATLAB approach:
    - Creates grid around base origin
    - Filters by reach radius if provided
    - Sorts by distance from reference position
    - Grid points are aligned with bucket centers for consistent quantization

    Args:
        config: IK reachability configuration
        reference_position: Reference position (typically base origin or initial pose)
        reach_radius: Optional radius to filter points (None = no filtering)

    Returns:
        Array of grid points (N, 3)
    """
    # Generate grid points aligned with bucket centers
    # Bucket centers are at: lower_bounds + (bucket_index + 0.5) * resolution
    # So grid points should be at: lower_bounds + (n + 0.5) * grid_step_size
    # where n = 0, 1, 2, ...
    # But we want grid_step_size to equal cartesian_resolution for alignment
    lower_bounds = np.array([config.x_limits[0], config.y_limits[0], config.z_limits[0]], dtype=np.float32)

    # Ensure grid_step_size matches cartesian_resolution for bucket alignment
    effective_step = config.grid_step_size
    if abs(effective_step - config.cartesian_resolution) > 1e-6:
        LOGGER.warning(
            "grid_step_size (%.6f) != cartesian_resolution (%.6f), using cartesian_resolution for alignment",
            effective_step, config.cartesian_resolution
        )
        effective_step = config.cartesian_resolution

    # Generate grid points at bucket centers: lower_bounds + (n + 0.5) * resolution
    x_range = config.x_limits[1] - config.x_limits[0]
    y_range = config.y_limits[1] - config.y_limits[0]
    z_range = config.z_limits[1] - config.z_limits[0]

    x_count = int(np.ceil(x_range / effective_step))
    y_count = int(np.ceil(y_range / effective_step))
    z_count = int(np.ceil(z_range / effective_step))

    x_points = np.array([config.x_limits[0] + (i + 0.5) * effective_step for i in range(x_count)])
    y_points = np.array([config.y_limits[0] + (i + 0.5) * effective_step for i in range(y_count)])
    z_points = np.array([config.z_limits[0] + (i + 0.5) * effective_step for i in range(z_count)])

    # Clip to limits
    x_points = np.clip(x_points, config.x_limits[0], config.x_limits[1])
    y_points = np.clip(y_points, config.y_limits[0], config.y_limits[1])
    z_points = np.clip(z_points, config.z_limits[0], config.z_limits[1])

    # Cartesian product
    grid_points = np.array(np.meshgrid(x_points, y_points, z_points)).T.reshape(-1, 3)

    # Always include the actual reference position as the highest-priority sample.
    ref_point = np.asarray(reference_position, dtype=np.float64).reshape(1, 3)
    ref_point = np.clip(
        ref_point,
        [config.x_limits[0], config.y_limits[0], config.z_limits[0]],
        [config.x_limits[1], config.y_limits[1], config.z_limits[1]],
    )
    if grid_points.size == 0:
        grid_points = ref_point.copy()
    else:
        ref_distance = np.linalg.norm(grid_points - ref_point, axis=1)
        if not np.any(ref_distance < 1e-6):
            grid_points = np.vstack([grid_points, ref_point])

    # Filter by reach radius if provided
    if reach_radius is not None:
        distances = np.linalg.norm(grid_points - reference_position, axis=1)
        valid_mask = distances <= reach_radius
        grid_points = grid_points[valid_mask]

    # Sort by distance from reference
    distances = np.linalg.norm(grid_points - reference_position, axis=1)
    sort_indices = np.argsort(distances)
    grid_points = grid_points[sort_indices]

    return grid_points


def get_target_orientations(
    config: IKReachabilityConfig,
    reference_pose: SE3,
) -> List[np.ndarray]:
    """Get target orientations based on mode.

    Args:
        config: IK reachability configuration
        reference_pose: Reference pose (typically initial end-effector pose)

    Returns:
        List of 3x3 rotation matrices
    """
    mode = config.orientation_mode
    if mode is OrientationMode.FIXED:
        if config.reference_orientation is not None:
            return [config.reference_orientation]
        return [reference_pose.rotation]
    elif mode is OrientationMode.REFERENCE:
        return [reference_pose.rotation]
    elif mode is OrientationMode.MULTIPLE:
        # Sample orientations (e.g., identity + rotations around axes)
        orientations = [np.eye(3)]  # Identity
        # Add more samples as needed (simplified for now)
        return orientations[: config.orientation_samples]
    else:
        raise ValueError(f"Unknown orientation_mode: {config.orientation_mode}")


# Module-level worker functions for multiprocessing (must be picklable)
_worker_robot_model: Optional[embodik.RobotModel] = None
_worker_solver: Optional[embodik.KinematicsSolver] = None
_worker_velocity_solver: Optional[embodik.KinematicsSolver] = None
_worker_velocity_task: Optional[Any] = None
_worker_metric_helper: Optional[ReachabilityMetricHelper] = None
_worker_end_effector: Optional[str] = None
_worker_config: Optional[IKReachabilityConfig] = None
_worker_metric_ranges: Optional[Dict[str, Tuple[float, float]]] = None
_worker_excluded_joint_indices: Optional[List[int]] = None


def _init_worker_process(
    urdf_path: str,
    end_effector: str,
    config_dict: Dict,
    chain_joint_names: List[str],
    metric_ranges_dict: Dict[str, Tuple[float, float]],
    manipulability_raw_max: float,
    range_of_motion_raw_max: float,
    singularity_weighted_max: float,
    limit_overrides_dict: Optional[Dict[str, Tuple[float, float]]] = None,
    excluded_joint_indices: Optional[List[int]] = None,
) -> None:
    """Initialize worker process with robot model and solver.

    This is called once per worker process to set up the robot model,
    solver, and metric helper. These are stored in module-level variables
    to avoid pickling overhead.

    Args:
        urdf_path: Path to URDF file
        end_effector: End-effector frame name
        config_dict: IKReachabilityConfig as dictionary (for pickling)
        chain_joint_names: Joint names for kinematic chain
        metric_ranges_dict: Metric value ranges
        manipulability_raw_max: Maximum manipulability for normalization
        range_of_motion_raw_max: Maximum ROM for normalization
        singularity_weighted_max: Maximum singularity for normalization
        limit_overrides_dict: Optional joint limit overrides applied in this worker
    """
    global _worker_robot_model, _worker_solver, _worker_velocity_solver, _worker_velocity_task
    global _worker_metric_helper, _worker_end_effector, _worker_config, _worker_metric_ranges
    global _worker_limit_overrides, _worker_joint_index_map, _worker_excluded_joint_indices

    # Reconstruct config from dict
    _worker_config = IKReachabilityConfig(**config_dict)
    _worker_end_effector = end_effector
    _worker_metric_ranges = metric_ranges_dict
    _worker_limit_overrides = dict(limit_overrides_dict or {})
    _worker_excluded_joint_indices = list(excluded_joint_indices) if excluded_joint_indices else None

    # Create robot model (suppress URDF parsing warnings)
    with suppress_urdf_warnings():
        _worker_robot_model = embodik.RobotModel(urdf_path, floating_base=False)

    # Build joint index map now that the robot model exists
    joint_names = _worker_robot_model.get_joint_names()
    _worker_joint_index_map = {name: idx for idx, name in enumerate(joint_names)}

    # Create solver
    _worker_solver = embodik.KinematicsSolver(_worker_robot_model)
    _worker_solver.dt = _worker_config.dt
    _worker_solver.set_damping(0.1)
    _worker_solver.set_tolerance(0.1)

    # Create velocity solver and task if excluded joints are needed
    _worker_velocity_solver = None
    _worker_velocity_task = None
    if _worker_excluded_joint_indices:
        _worker_velocity_solver = embodik.KinematicsSolver(_worker_robot_model)
        _worker_velocity_solver.dt = _worker_config.dt
        _worker_velocity_solver.set_damping(0.1)
        _worker_velocity_solver.set_tolerance(0.1)
        _worker_velocity_task = _worker_velocity_solver.add_frame_task("worker_ik_task", end_effector)
        _worker_velocity_task.priority = 0
        _worker_velocity_task.weight = 0.0
        _worker_velocity_task.set_excluded_joint_indices(_worker_excluded_joint_indices)
        zero_velocity = np.zeros(6, dtype=float)
        _worker_velocity_task.set_target_velocity(zero_velocity)

    # Build PyTorch kinematic chain for manipulability (if available)
    metric_chain = None
    manip_chain_indices = None
    chain_joint_order = []
    metric_sample_joint_indices = []
    device = None
    dtype = None

    try:
        import pytorch_kinematics as pk
        if torch is None:
            raise ImportError("torch not available")
        with open(urdf_path, "r", encoding="utf-8") as f:
            urdf_xml = f.read()
        urdf_xml = _strip_xml_prolog_and_comments(urdf_xml)
        # Suppress URDF parsing warnings when building metric chain
        with suppress_urdf_warnings():
            metric_chain = pk.build_serial_chain_from_urdf(urdf_xml, end_effector)
        # Force CPU mode for multiprocessing workers (CUDA contexts can't be shared across processes)
        device = torch.device("cpu")
        dtype = torch.float32
        metric_chain = metric_chain.to(dtype=dtype, device=device)
        chain_joint_order = metric_chain.get_joint_parameter_names()

        robot_joint_names = _worker_robot_model.get_joint_names()
        name_to_index = {name: idx for idx, name in enumerate(robot_joint_names)}
        manip_chain_indices = [
            name_to_index.get(name) for name in chain_joint_order
        ]

        chain_joint_set = set(chain_joint_order)
        sample_joint_set = set(chain_joint_names)
        metric_sample_joint_indices = [
            idx for idx, name in enumerate(chain_joint_order) if name in sample_joint_set
        ]

        import logging
        worker_logger = logging.getLogger("embodik.reachability_ik.worker")
        worker_logger.debug(
            "Metric chain built in worker process: %d joints, device=cpu (forced for multiprocessing)",
            len(chain_joint_order)
        )
    except Exception as exc:
        # Metric chain not available, will compute zero manipulability
        import logging
        worker_logger = logging.getLogger("embodik.reachability_ik.worker")
        worker_logger.warning(
            "Failed to build metric chain in worker process: %s. "
            "Manipulability will be zero. Install pytorch_kinematics for manipulability computation.",
            exc
        )
        import traceback
        worker_logger.debug("Traceback: %s", traceback.format_exc())
        metric_chain = None
        manip_chain_indices = None
        chain_joint_order = []
        metric_sample_joint_indices = []
        device = None
        dtype = None

    # Create metric helper
    joint_name_to_idx = {
        name: idx for idx, name in enumerate(_worker_robot_model.get_joint_names())
    }
    arm_joint_indices = [
        joint_name_to_idx.get(name) for name in chain_joint_names
    ]
    arm_joint_indices = [idx for idx in arm_joint_indices if idx is not None]

    joint_names = _worker_robot_model.get_joint_names()
    raw_lower_bounds, raw_upper_bounds = _worker_robot_model.get_joint_limits()
    lower_bounds, upper_bounds = _apply_limit_overrides_array(
        joint_names,
        raw_lower_bounds,
        raw_upper_bounds,
        _worker_limit_overrides,
    )
    arm_lower = np.array([lower_bounds[idx] for idx in arm_joint_indices])
    arm_upper = np.array([upper_bounds[idx] for idx in arm_joint_indices])

    _worker_metric_helper = ReachabilityMetricHelper(
        robot=_worker_robot_model,
        arm_joint_indices=arm_joint_indices,
        arm_lower_bounds=arm_lower,
        arm_upper_bounds=arm_upper,
        manip_scaling=_worker_config.manip_scaling,
        metric_chain=metric_chain,
        manip_chain_indices=manip_chain_indices,
        chain_joint_order=chain_joint_order,
        metric_sample_joint_indices=metric_sample_joint_indices,
        metric_chain_device=device,
        metric_chain_dtype=dtype,
    )

    # Set metadata
    _worker_metric_helper.metric_metadata["ManipulabilityRawMax"] = manipulability_raw_max
    _worker_metric_helper.metric_metadata["RangeOfMotionRawMax"] = range_of_motion_raw_max
    _worker_metric_helper.metric_metadata["SingularityWeightedMax"] = singularity_weighted_max


def _process_single_ik_task(
    task_data: Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]
) -> Optional[Dict]:
    """Process a single IK task (module-level function for multiprocessing).

    Args:
        task_data: Tuple of (grid_idx, orient_idx, grid_point, target_orientation, q_seed)

    Returns:
        Dictionary with results, or None if failed
    """
    global _worker_robot_model, _worker_solver, _worker_velocity_solver, _worker_velocity_task
    global _worker_metric_helper, _worker_end_effector, _worker_config, _worker_metric_ranges
    global _worker_excluded_joint_indices

    if _worker_solver is None or _worker_config is None:
        LOGGER.error("Worker process not initialized!")
        return None

    grid_idx, orient_idx, grid_point, target_orientation, q_seed = task_data

    # Build target pose matrix
    target_pose = np.eye(4)
    target_pose[:3, :3] = target_orientation
    target_pose[:3, 3] = grid_point

    # Use velocity-based solving if excluded joints are needed (since solve_position doesn't support it)
    if _worker_excluded_joint_indices and _worker_velocity_solver is not None and _worker_velocity_task is not None:
        # Use velocity-based solving with excluded joints
        from types import SimpleNamespace
        from example_helpers.dual_arm_ik_helper import PoseData, compute_pose_error

        target_pose_data = PoseData(target_orientation, grid_point)
        q_current = q_seed.copy()
        _worker_robot_model.update_configuration(q_current)
        _worker_velocity_task.weight = 1.0
        zero_velocity = np.zeros(6, dtype=float)
        _worker_velocity_task.set_target_velocity(zero_velocity)

        ik_solve_start = time.perf_counter()
        pos_gain = 75.0  # Default position gain
        rot_gain = 60.0  # Default rotation gain

        for iteration in range(_worker_config.max_iterations):
            current_pose = PoseData.wrap(_worker_robot_model.get_frame_pose(_worker_end_effector))
            error_vec = compute_pose_error(current_pose, target_pose_data)
            pos_err = float(np.linalg.norm(error_vec[:3]))
            rot_err = float(np.linalg.norm(error_vec[3:]))

            if pos_err <= _worker_config.position_tolerance and rot_err <= _worker_config.orientation_tolerance:
                _worker_velocity_task.set_target_velocity(zero_velocity)
                _worker_velocity_task.weight = 0.0
                ik_solve_elapsed = time.perf_counter() - ik_solve_start
                result = SimpleNamespace(
                    status=embodik.SolverStatus.SUCCESS,
                    q_solution=q_current.copy(),
                    position_error=pos_err,
                    orientation_error=rot_err,
                )
                break

            # Compute target velocity
            target_vel = np.zeros(6, dtype=float)
            target_vel[:3] = error_vec[:3] * pos_gain
            target_vel[3:] = error_vec[3:] * rot_gain

            # Clamp velocities
            max_linear = _worker_config.max_linear_step
            max_angular = _worker_config.max_angular_step
            linear_norm = np.linalg.norm(target_vel[:3])
            if linear_norm > max_linear:
                target_vel[:3] = target_vel[:3] / linear_norm * max_linear
            angular_norm = np.linalg.norm(target_vel[3:])
            if angular_norm > max_angular:
                target_vel[3:] = target_vel[3:] / angular_norm * max_angular

            _worker_velocity_task.set_target_velocity(target_vel)
            solve_result = _worker_velocity_solver.solve_velocity(q_current, apply_limits=True)
            dq = solve_result.dq
            q_current = np.clip(q_current + dq * _worker_config.dt,
                               _worker_robot_model.get_joint_limits()[0],
                               _worker_robot_model.get_joint_limits()[1])
            _worker_robot_model.update_configuration(q_current)
        else:
            # Max iterations reached
            ik_solve_elapsed = time.perf_counter() - ik_solve_start
            result = SimpleNamespace(
                status=embodik.SolverStatus.NUMERICAL_ERROR,
                q_solution=q_current.copy(),
                position_error=pos_err,
                orientation_error=rot_err,
            )
    else:
        # Use standard solve_position (no excluded joints)
        # Configure IK options
        options = embodik.PositionIKOptions()
        options.position_tolerance = _worker_config.position_tolerance
        options.orientation_tolerance = _worker_config.orientation_tolerance
        options.max_iterations = _worker_config.max_iterations
        options.dt = _worker_config.dt
        options.max_linear_step = _worker_config.max_linear_step
        options.max_angular_step = _worker_config.max_angular_step

        if _worker_config.use_nullspace_bias and _worker_config.nullspace_bias is not None:
            options.nullspace_bias = _worker_config.nullspace_bias
            options.nullspace_gain = _worker_config.nullspace_gain
            if _worker_config.nullspace_active_joints is not None:
                options.nullspace_active_joints = _worker_config.nullspace_active_joints

        # Solve IK (with timing)
        ik_solve_start = time.perf_counter()
        result = _worker_solver.solve_position(q_seed, target_pose, _worker_end_effector, options)
        ik_solve_elapsed = time.perf_counter() - ik_solve_start

    if result.status == embodik.SolverStatus.SUCCESS:
        q_solution = result.q_solution
        if _worker_limit_overrides:
            for joint_name, (joint_min, joint_max) in _worker_limit_overrides.items():
                idx = _worker_joint_index_map.get(joint_name)
                if idx is None:
                    continue
                value = q_solution[idx]
                if value < joint_min - 1e-6 or value > joint_max + 1e-6:
                    return {
                        "success": False,
                        "grid_point": grid_point,
                        "ik_time": ik_solve_elapsed,
                        "reason": "limit_violation",
                    }
        metrics = _worker_metric_helper.compute_metrics(q_solution, _worker_metric_ranges)
        return {
            "success": True,
            "grid_point": grid_point,
            "target_orientation": target_orientation,
            "q_solution": q_solution,
            "metrics": metrics,
            "ik_time": ik_solve_elapsed,
        }
    else:
        return {
            "success": False,
            "grid_point": grid_point,
            "ik_time": ik_solve_elapsed,
        }


def create_grid_preview_file(
    config: IKReachabilityConfig,
    grid_points: np.ndarray,
    output_dir: Path,
    reference_position: np.ndarray,
) -> Optional[Path]:
    """Create a preview HDF5 file for grid points.

    Args:
        config: IK reachability configuration
        grid_points: Array of grid points (N, 3)
        output_dir: Output directory for the preview file
        reference_position: Reference position for bounds

    Returns:
        Path to the preview file, or None if creation failed
    """
    try:
        if len(grid_points) == 0:
            LOGGER.warning("No grid points to create preview for")
            return None

        # Compute bounds for the grid
        lower_bounds_np = np.array(
            [
                config.x_limits[0],
                config.y_limits[0],
                config.z_limits[0],
            ],
            dtype=np.float32,
        )

        # Create sphere dataset with grid points (dummy metrics for preview)
        # Format: x, y, z, manipulability, visitation, rom, sing
        # Create array directly instead of appending to list for efficiency
        num_points = len(grid_points)
        sphere_data = np.zeros((num_points, 7), dtype=np.float32)
        sphere_data[:, :3] = grid_points.astype(np.float32)  # x, y, z
        sphere_data[:, 3] = 0.0  # manipulability (dummy, will show as gray/low score)
        sphere_data[:, 4] = 0.0  # visitation (dummy)
        sphere_data[:, 5] = 0.0  # rom (dummy)
        sphere_data[:, 6] = 0.0  # sing (dummy)

        # Save temporary preview HDF5 file
        grid_preview_path = output_dir / "ik_grid_preview.h5"
        with h5py.File(grid_preview_path, "w") as handle:
            sphere_group = handle.create_group("/Spheres")
            dataset = sphere_group.create_dataset("sphere_dataset", data=sphere_data)
            dataset.attrs.create("Resolution", data=config.cartesian_resolution)
            dataset.attrs.create("LowerBounds", data=lower_bounds_np)
            dataset.attrs.create(
                "MetricNames",
                np.asarray(
                    ["Manipulability", "RangeOfMotion", "SingularityAvoidance"],
                    dtype="S32",
                ),
            )
            dataset.attrs.create("MetricColumns", np.asarray([3, 5, 6], dtype=np.int32))
            dataset.attrs.create("VisitationColumn", np.int32(4))
            dataset.attrs.create("ManipulabilityRawMax", 1.0)  # Placeholder for preview
            dataset.attrs.create("RangeOfMotionRawMin", 0.0)
            dataset.attrs.create("RangeOfMotionRawMax", 1.0)
            dataset.attrs.create("SingularityWeightedMax", 1.0)

            # Empty poses dataset
            pose_group = handle.create_group("/Poses")
            pose_group.create_dataset(
                "poses_dataset", data=np.zeros((0, 10), dtype=np.float32)
            )

        LOGGER.info(
            "Grid preview file created: %s (%d points)",
            grid_preview_path.name,
            len(grid_points),
        )
        return grid_preview_path
    except Exception as exc:
        LOGGER.warning("Failed to create grid preview file: %s", exc)
        import traceback

        LOGGER.debug("Traceback: %s", traceback.format_exc())
        return None


def generate_ik_reachability_map(
    robot_cfg: RobotConfig,
    config: IKReachabilityConfig,
    *,
    chain_joint_names: Sequence[str],
    output_dir: Path,
    end_effector: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    server: Optional[viser.ViserServer] = None,
    viewer: Optional["ReachabilityMapViewer"] = None,
    show_ik_process: bool = False,
    visualization_callback: Optional[Callable[[np.ndarray], None]] = None,
    show_grid_preview: bool = True,
    seed_configuration: Optional[Dict[str, float]] = None,
    debug_wait_handler: Optional[Callable[[Dict[str, Any]], None]] = None,
    excluded_joint_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Path]:
    """Generate reachability map using IK-based analysis.

    This function:
    1. Generates a 3D Cartesian grid of target positions
    2. For each grid point, solves IK to test reachability

    Args:
        robot_cfg: Robot configuration
        config: IK reachability configuration
        chain_joint_names: Names of joints in the kinematic chain
        output_dir: Directory to save outputs
        end_effector: End-effector frame name
        progress_callback: Optional callback for progress updates
        cancel_event: Optional event to cancel generation
        server: Optional Viser server for visualization
        viewer: Optional reachability map viewer
        show_ik_process: Whether to show IK solving process
        visualization_callback: Optional callback for visualization
        show_grid_preview: Whether to show grid preview
        seed_configuration: Optional mapping of joint name -> position (radians/meters)
            used to initialize the IK solver instead of the default configuration.
        debug_wait_handler: Optional callback invoked after each grid target is processed
            (used for interactive debugging/stepping).
    """
    # Validate minimum cartesian resolution (1cm)
    if config.cartesian_resolution < 0.01:
        LOGGER.warning(
            "cartesian_resolution (%.4f m) is below minimum (0.01 m), clamping to 0.01 m",
            config.cartesian_resolution
        )
        config.cartesian_resolution = 0.01

    ensure_ros_package_path(robot_cfg.urdf_path)
    loadable_urdf_path = robot_cfg.loadable_urdf_path
    LOGGER.info("Starting IK reachability map generation")
    LOGGER.info("Robot config: %s", robot_cfg.key)
    LOGGER.info("End effector: %s", end_effector)
    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info("Configuration: num_workers=%d, orientation_mode=%s, grid_step=%.4f m",
                config.num_workers, config.orientation_mode.value, config.grid_step_size)

    # Start overall timing
    total_start_time = time.perf_counter()
    phase_timings: Dict[str, float] = {}

    # Cache for previously successful IK solutions (used to pick future seeds).
    MAX_SEED_CACHE = 2048
    seed_cache: deque = deque(maxlen=MAX_SEED_CACHE)

    def _seed_metric_value(metrics: Optional[Dict[str, float]]) -> float:
        """Derive a ranking score for cached seeds based on config."""
        if not metrics:
            return 0.0
        metric_key = config.seed_score_metric.lower()
        if metric_key == "singularity":
            return float(metrics.get("sing_norm", 0.0))
        if metric_key in ("rom", "range_of_motion"):
            return float(metrics.get("rom_norm", 0.0))
        if metric_key in ("limit", "limit_distance"):
            return float(metrics.get("limit_norm", 0.0))
        # Fallback composite
        return (
            float(metrics.get("sing_norm", 0.0)) * 2.0
            + float(metrics.get("rom_norm", 0.0))
            + float(metrics.get("limit_norm", 0.0))
        )

    def _register_seed_candidate(
        point: Optional[np.ndarray],
        config_vec: Optional[np.ndarray],
        metrics: Optional[Dict[str, float]],
    ) -> None:
        """Store a solved configuration for later seeding."""
        if point is None or config_vec is None:
            return
        score = _seed_metric_value(metrics)
        seed_cache.append(
            {
                "point": point.copy(),
                "config": config_vec.copy(),
                "score": score,
                "metrics": dict(metrics) if metrics else None,
            }
        )

    rng = np.random.default_rng()

    def _within_override_bounds(q_vec: np.ndarray) -> bool:
        """Check joint-limit overrides for a configuration."""
        if not robot_cfg.limit_overrides:
            return True
        for joint_name, (joint_min, joint_max) in robot_cfg.limit_overrides.items():
            idx = joint_name_to_idx.get(joint_name)
            if idx is None:
                continue
            value = q_vec[idx]
            if value < joint_min - 1e-6 or value > joint_max + 1e-6:
                return False
        return True

    def _select_seed_for_point(
        target_point: np.ndarray,
        default_seed: np.ndarray,
        neighborhood_radius: float,
    ) -> np.ndarray:
        """Pick the best cached seed (if any) for a target point."""
        if not seed_cache:
            return default_seed.copy()

        best_local_entry = None
        best_local_score = -float("inf")
        closest_entry = None
        closest_dist = float("inf")

        for entry in seed_cache:
            point = entry["point"]
            dist = float(np.linalg.norm(target_point - point))
            if dist < closest_dist:
                closest_dist = dist
                closest_entry = entry
            if dist <= neighborhood_radius:
                score = float(entry.get("score", entry.get("quality", 0.0)))
                if best_local_entry is None or score > best_local_score or (
                    math.isclose(score, best_local_score)
                    and np.linalg.norm(target_point - entry["point"])
                    < np.linalg.norm(target_point - best_local_entry["point"])
                ):
                    best_local_entry = entry
                    best_local_score = score

        chosen_entry = best_local_entry or closest_entry
        if chosen_entry is None:
            return default_seed.copy()
        return chosen_entry["config"].copy()

    def _ranked_seed_candidates(
        target_point: np.ndarray,
        max_candidates: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Return cached seeds sorted by singularity score (desc) then proximity."""
        if not seed_cache:
            return []
        scored: List[Tuple[float, float, Dict[str, Any]]] = []
        for entry in seed_cache:
            score = float(entry.get("score", entry.get("quality", 0.0)))
            dist = float(np.linalg.norm(target_point - entry["point"]))
            # Negative score to sort descending by metric, then ascending distance
            scored.append((-score, dist, entry))
        scored.sort(key=lambda item: (item[0], item[1]))
        candidates: List[np.ndarray] = []
        for _, _, entry in scored:
            candidates.append(entry["config"].copy())
            if max_candidates is not None and len(candidates) >= max_candidates:
                break
        return candidates

    def _build_seed_queue(
        primary_seed: np.ndarray,
        target_point: np.ndarray,
        current_seed: np.ndarray,
        initial_seed: np.ndarray,
        *,
        max_attempts: int,
    ) -> List[np.ndarray]:
        """Assemble an ordered list of seeds to try for a grid point."""
        queue: List[np.ndarray] = []
        seen: Set[bytes] = set()

        def _add_seed(seed: Optional[np.ndarray]) -> None:
            if seed is None:
                return
            key = seed.tobytes()
            if key in seen:
                return
            queue.append(seed.copy())
            seen.add(key)

        _add_seed(primary_seed)
        _add_seed(current_seed)
        _add_seed(initial_seed)

        if config.use_interactive_marker_solver:
            queue = []
            seen.clear()
            nearest_seed = _select_seed_for_point(target_point, current_seed, 0.0)
            _add_seed(nearest_seed)
            if not queue:
                _add_seed(current_seed)
            return queue

        # Append best cached seeds ranked by singularity avoidance/quality
        for cached_seed in _ranked_seed_candidates(target_point, max_candidates=config.retry_seed_pool):
            _add_seed(cached_seed)
            if len(queue) >= max_attempts + config.retry_seed_pool:
                break

        # Add jittered variants of top seeds to escape local minima
        if config.retry_seed_jitter > 0.0 and queue:
            jitter_sources = queue[: min(len(queue), max(1, config.retry_seed_pool))]
            for base_seed in jitter_sources:
                noise = rng.normal(scale=config.retry_seed_jitter, size=base_seed.shape)
                candidate = base_seed + noise
                candidate = np.clip(candidate, lower_bounds, upper_bounds)
                _add_seed(candidate)
                if len(queue) >= max_attempts + config.retry_seed_pool:
                    break

        return queue

    def _solve_position_ik_target(
        q_seed: np.ndarray,
        grid_point: np.ndarray,
        target_orientation: np.ndarray,
        solver: embodik.KinematicsSolver,
        end_effector: str,
        config: IKReachabilityConfig,
        *,
        position_tolerance: Optional[float] = None,
        orientation_tolerance: Optional[float] = None,
        max_iterations: Optional[int] = None,
        excluded_joint_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[embodik.IKResult, float, np.ndarray]:
        """Solve IK for a single target pose."""
        target_pose = np.eye(4)
        target_pose[:3, :3] = target_orientation
        target_pose[:3, 3] = grid_point

        options = embodik.PositionIKOptions()
        options.position_tolerance = float(position_tolerance if position_tolerance is not None else config.position_tolerance)
        options.orientation_tolerance = float(
            orientation_tolerance if orientation_tolerance is not None else config.orientation_tolerance
        )
        options.max_iterations = int(max_iterations if max_iterations is not None else config.max_iterations)
        options.max_iterations = max(1, options.max_iterations)
        options.dt = config.dt
        options.max_linear_step = config.max_linear_step
        options.max_angular_step = config.max_angular_step

        if config.use_nullspace_bias and config.nullspace_bias is not None:
            options.nullspace_bias = config.nullspace_bias
            options.nullspace_gain = config.nullspace_gain
            if config.nullspace_active_joints is not None:
                options.nullspace_active_joints = config.nullspace_active_joints

        ik_solve_start = time.perf_counter()
        result = solver.solve_position(q_seed, target_pose, end_effector, options)
        ik_solve_elapsed = time.perf_counter() - ik_solve_start
        return result, ik_solve_elapsed, target_pose

    def _solve_interactive_velocity_target(
        q_seed: np.ndarray,
        grid_point: np.ndarray,
        target_orientation: np.ndarray,
        solver: embodik.KinematicsSolver,
        task: Any,
        *,
        pos_gain: float,
        rot_gain: float,
        max_iterations: int,
        debug_enabled: bool,
        stagnation_threshold: float,
        stagnation_window: int,
    ) -> Tuple[SimpleNamespace, float, np.ndarray]:
        """Solve IK using the velocity-controller loop (interactive style)."""

        if solver is None or task is None:
            raise RuntimeError("Interactive IK solver requested but not initialized.")

        target_pose = np.eye(4)
        target_pose[:3, :3] = target_orientation
        target_pose[:3, 3] = grid_point
        target_pose_data = PoseData(target_orientation, grid_point)

        q_current = q_seed.copy()
        robot.update_configuration(q_current)
        task.weight = 1.0
        task.set_target_velocity(zero_velocity_vec.copy())

        ik_solve_start = time.perf_counter()
        last_status = embodik.SolverStatus.NUMERICAL_ERROR
        last_pos_err = None
        last_rot_err = None

        stagnation_buffer: List[float] = []
        no_progress_count = 0

        for iteration in range(max_iterations):
            current_pose = PoseData.wrap(robot.get_frame_pose(end_effector))
            error_vec = compute_pose_error(current_pose, target_pose_data)
            pos_err = float(np.linalg.norm(error_vec[:3]))
            rot_err = float(np.linalg.norm(error_vec[3:]))
            last_pos_err = pos_err
            last_rot_err = rot_err
            if debug_enabled:
                LOGGER.info(
                    "[InteractiveIK][IKDebug] iter %d pos_err=%.6f ori_err=%.6f",
                    iteration,
                    pos_err,
                    rot_err,
                )
            if pos_err <= config.position_tolerance and rot_err <= config.orientation_tolerance:
                task.set_target_velocity(zero_velocity_vec.copy())
                task.weight = 0.0
                ik_elapsed = time.perf_counter() - ik_solve_start
                return (
                    SimpleNamespace(
                        status=embodik.SolverStatus.SUCCESS,
                        q_solution=q_current.copy(),
                        num_iterations=iteration + 1,
                        position_error=pos_err,
                        orientation_error=rot_err,
                    ),
                    ik_elapsed,
                    target_pose,
                )

            stagnation_buffer.append(pos_err + rot_err)
            if len(stagnation_buffer) > stagnation_window:
                stagnation_buffer.pop(0)
                if max(stagnation_buffer) - min(stagnation_buffer) <= stagnation_threshold:
                    no_progress_count += 1
                else:
                    no_progress_count = 0
                if no_progress_count >= stagnation_window:
                    if debug_enabled:
                        LOGGER.info(
                            "[InteractiveIK][IKDebug] Early terminate due to stagnation after %d iterations (pos_err=%.6f ori_err=%.6f)",
                            iteration + 1,
                            pos_err,
                            rot_err,
                        )
                    break

            desired_velocity = np.concatenate(
                [
                    pos_gain * error_vec[:3],
                    rot_gain * error_vec[3:],
                ]
            )
            limited_velocity = limit_task_velocity(
                desired_velocity,
                max_linear_step=config.max_linear_step,
                max_angular_step=config.max_angular_step,
            )
            task.set_target_velocity(limited_velocity)
            result = solver.solve_velocity(q_current, apply_limits=True)
            last_status = result.status
            if result.status != embodik.SolverStatus.SUCCESS:
                break
            joint_velocities = np.asarray(result.joint_velocities, dtype=float)
            dq_full = np.zeros_like(q_current)
            dq_full[: joint_velocities.size] = joint_velocities
            q_current = q_current + dq_full * solver.dt
            q_current = np.clip(q_current, lower_bounds, upper_bounds)
            robot.update_configuration(q_current)

        task.set_target_velocity(zero_velocity_vec.copy())
        task.weight = 0.0
        ik_elapsed = time.perf_counter() - ik_solve_start
        final_status = last_status
        if (
            final_status != embodik.SolverStatus.SUCCESS
            and last_pos_err is not None
            and last_rot_err is not None
            and last_pos_err <= config.position_tolerance
            and last_rot_err <= config.orientation_tolerance
        ):
            final_status = embodik.SolverStatus.SUCCESS
        return (
            SimpleNamespace(
                status=final_status,
                q_solution=q_current.copy(),
                num_iterations=max_iterations,
                position_error=last_pos_err,
                orientation_error=last_rot_err,
            ),
            ik_elapsed,
            target_pose,
        )

    def _solve_interactive_path_target(
        q_seed: np.ndarray,
        grid_point: np.ndarray,
        target_orientation: np.ndarray,
        solver: embodik.KinematicsSolver,
        task: Any,
        *,
        pos_gain: float,
        rot_gain: float,
        max_iterations: int,
        debug_enabled: bool,
        stagnation_threshold: float,
        stagnation_window: int,
    ) -> Tuple[SimpleNamespace, float, np.ndarray]:
        """Follow a linear path from current pose to target using interactive IK."""

        robot.update_configuration(q_seed)
        current_pose = robot.get_frame_pose(end_effector)
        start_position = current_pose.translation.copy()
        total_displacement = grid_point - start_position
        distance = float(np.linalg.norm(total_displacement))
        if distance < 1e-6:
            return _solve_interactive_velocity_target(
                q_seed,
                grid_point,
                target_orientation,
                solver,
                task,
                pos_gain=pos_gain,
                rot_gain=rot_gain,
                max_iterations=max_iterations,
                debug_enabled=debug_enabled,
                stagnation_threshold=stagnation_threshold,
                stagnation_window=stagnation_window,
            )

        step_size = max(config.interactive_path_resolution, 1e-3)
        path_steps = max(1, int(np.ceil(distance / step_size)))
        iterations_per_step = max(5, max_iterations // max(1, path_steps))

        q_current = q_seed.copy()
        total_time = 0.0
        total_iterations = 0
        last_status = embodik.SolverStatus.SUCCESS
        last_pos_err = 0.0
        last_rot_err = 0.0
        last_pose = np.eye(4)

        for step_idx in range(path_steps):
            alpha = float(step_idx + 1) / path_steps
            sub_point = start_position + total_displacement * alpha
            sub_result, sub_time, sub_pose = _solve_interactive_velocity_target(
                q_current,
                sub_point,
                target_orientation,
                solver,
                task,
                pos_gain=pos_gain,
                rot_gain=rot_gain,
                max_iterations=iterations_per_step,
                debug_enabled=debug_enabled,
                stagnation_threshold=stagnation_threshold,
                stagnation_window=stagnation_window,
            )
            total_time += sub_time
            total_iterations += getattr(sub_result, "num_iterations", iterations_per_step)
            last_status = sub_result.status
            last_pos_err = getattr(sub_result, "position_error", last_pos_err)
            last_rot_err = getattr(sub_result, "orientation_error", last_rot_err)
            last_pose = sub_pose
            q_current = sub_result.q_solution.copy()
            if sub_result.status != embodik.SolverStatus.SUCCESS:
                if (
                    sub_result.position_error is not None
                    and sub_result.orientation_error is not None
                    and sub_result.position_error <= config.position_tolerance
                    and sub_result.orientation_error <= config.orientation_tolerance
                ):
                    return (
                        SimpleNamespace(
                            status=embodik.SolverStatus.SUCCESS,
                            q_solution=q_current.copy(),
                            num_iterations=total_iterations,
                            position_error=sub_result.position_error,
                            orientation_error=sub_result.orientation_error,
                        ),
                        total_time,
                        last_pose,
                    )
                else:
                    return (
                        SimpleNamespace(
                            status=sub_result.status,
                            q_solution=q_current.copy(),
                            num_iterations=total_iterations,
                            position_error=last_pos_err,
                            orientation_error=last_rot_err,
                        ),
                        total_time,
                        last_pose,
                    )

        return (
            SimpleNamespace(
                status=last_status,
                q_solution=q_current.copy(),
                num_iterations=total_iterations,
                position_error=last_pos_err,
                orientation_error=last_rot_err,
            ),
            total_time,
            last_pose,
        )

    def _solve_target(
        q_seed: np.ndarray,
        grid_point: np.ndarray,
        target_orientation: np.ndarray,
        solver: embodik.KinematicsSolver,
        end_effector: str,
        config: IKReachabilityConfig,
        *,
        position_tolerance: Optional[float] = None,
        orientation_tolerance: Optional[float] = None,
        max_iterations: Optional[int] = None,
        excluded_joint_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[Any, float, np.ndarray]:
        # If excluded joint indices are provided, use velocity-based solving (interactive mode)
        # since solve_position doesn't support excluded joints directly
        use_velocity_solver = config.use_interactive_marker_solver or (excluded_joint_indices is not None)

        if use_velocity_solver:
            # Ensure excluded joint indices are set on velocity_task if provided
            if excluded_joint_indices is not None and velocity_task is not None:
                velocity_task.set_excluded_joint_indices(list(excluded_joint_indices))
            return _solve_interactive_path_target(
                q_seed,
                grid_point,
                target_orientation,
                velocity_solver,
                velocity_task,
                pos_gain=config.interactive_pos_gain,
                rot_gain=config.interactive_rot_gain,
                max_iterations=config.interactive_max_iterations,
                debug_enabled=debug_logging_enabled,
                stagnation_threshold=config.interactive_stagnation_threshold,
                stagnation_window=config.interactive_stagnation_window,
            )
        return _solve_position_ik_target(
            q_seed,
            grid_point,
            target_orientation,
            solver,
            end_effector,
            config,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            max_iterations=max_iterations,
            excluded_joint_indices=excluded_joint_indices,
        )

    # Load robot model (suppress URDF parsing warnings)
    with suppress_urdf_warnings():
        robot = embodik.RobotModel(str(loadable_urdf_path), floating_base=False)
    LOGGER.info("Robot model loaded: nq=%d, nv=%d", robot.nq, robot.nv)
    robot_joint_names = robot.get_joint_names()
    joint_name_to_idx = {name: idx for idx, name in enumerate(robot_joint_names)}
    raw_lower_bounds, raw_upper_bounds = robot.get_joint_limits()
    lower_bounds, upper_bounds = _apply_limit_overrides_array(
        robot_joint_names,
        raw_lower_bounds,
        raw_upper_bounds,
        robot_cfg.limit_overrides,
    )
    if robot_cfg.limit_overrides:
        LOGGER.info(
            "Joint limit overrides applied for IK: %s",
            ", ".join(
                f"{joint}: [{bounds[0]:.4f}, {bounds[1]:.4f}]"
                for joint, bounds in robot_cfg.limit_overrides.items()
            ),
        )

    def _compute_pose_error_for_solution(
        q_solution: np.ndarray,
        target_pose: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute FK pose error for the solved configuration."""

        robot.update_configuration(q_solution)
        actual_pose = robot.get_frame_pose(end_effector)
        actual = PoseData.wrap(actual_pose)
        desired = PoseData(target_pose[:3, :3], target_pose[:3, 3])
        error_vec = compute_pose_error(actual, desired)
        pos_error = float(np.linalg.norm(error_vec[:3]))
        rot_error = float(np.linalg.norm(error_vec[3:]))
        return pos_error, rot_error

    def _estimate_fk_envelope(sample_count: int) -> Optional[Dict[str, np.ndarray]]:
        """Sample random joint configurations to approximate a workspace envelope."""

        if sample_count <= 0:
            return None

        sample_joint_indices = [
            joint_name_to_idx.get(name) for name in chain_joint_names
        ]
        sample_joint_indices = [idx for idx in sample_joint_indices if idx is not None]
        if not sample_joint_indices:
            LOGGER.warning(
                "FK envelope pruning enabled but no valid joint indices were found; skipping."
            )
            return None

        lower_subset = lower_bounds[sample_joint_indices]
        upper_subset = upper_bounds[sample_joint_indices]
        samples = np.zeros((sample_count, 3), dtype=float)
        base_configuration = q_seed.copy()

        for idx in range(sample_count):
            q_candidate = base_configuration.copy()
            q_candidate[sample_joint_indices] = rng.uniform(
                lower_subset,
                upper_subset,
            )
            robot.update_configuration(q_candidate)
            pose = robot.get_frame_pose(end_effector)
            samples[idx] = pose.translation

        robot.update_configuration(base_configuration)

        center = np.mean(samples, axis=0)
        distances = np.linalg.norm(samples - center, axis=1)
        radius = float(np.max(distances))
        if not np.isfinite(radius):
            LOGGER.warning("FK envelope sampling produced invalid radius; skipping pruning.")
            return None

        return {
            "center": center,
            "radius": radius,
            "aabb_min": np.min(samples, axis=0),
            "aabb_max": np.max(samples, axis=0),
            "sample_count": sample_count,
        }

    def _filter_grid_points_by_envelope(
        points: np.ndarray,
        stats: Dict[str, np.ndarray],
        margin: float,
    ) -> Tuple[np.ndarray, int]:
        """Filter grid points using the approximated FK envelope."""
        if points.size == 0:
            return points, 0
        radius = stats["radius"] + max(0.0, margin)
        center = stats["center"]
        distances = np.linalg.norm(points - center, axis=1)
        mask = distances <= radius + 1e-9
        filtered = points[mask]
        removed = int(points.shape[0] - filtered.shape[0])
        return filtered, removed

    # Build PyTorch kinematic chain for manipulability computation
    try:
        import pytorch_kinematics as pk

        with open(loadable_urdf_path, "r", encoding="utf-8") as f:
            urdf_xml = f.read()

        urdf_xml = _strip_xml_prolog_and_comments(urdf_xml)

        # Suppress URDF parsing warnings when building metric chain
        with suppress_urdf_warnings():
            metric_chain = pk.build_serial_chain_from_urdf(urdf_xml, end_effector)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        metric_chain = metric_chain.to(dtype=dtype, device=device)
        chain_joint_order = metric_chain.get_joint_parameter_names()

        # Map chain joints to robot joint indices
        name_to_index = {name: idx for idx, name in enumerate(robot_joint_names)}
        manip_chain_indices: List[Optional[int]] = [
            name_to_index.get(name) for name in chain_joint_order
        ]

        # Find which chain joints are in our arm joint set
        chain_joint_set = set(chain_joint_order)
        sample_joint_set = set(chain_joint_names)
        metric_sample_joint_indices = [
            idx for idx, name in enumerate(chain_joint_order) if name in sample_joint_set
        ]

        LOGGER.info("Metric chain built: %d joints, device=%s", len(chain_joint_order), device.type)
        LOGGER.info("Chain joint order: %s", chain_joint_order)
        LOGGER.info("Manip chain indices mapping: %s", manip_chain_indices)
    except Exception as exc:
        LOGGER.warning("Failed to build metric chain for manipulability: %s", exc)
        LOGGER.warning("Manipulability will be zero. Install pytorch_kinematics for manipulability computation.")
        import traceback
        LOGGER.debug("Traceback: %s", traceback.format_exc())
        metric_chain = None
        manip_chain_indices = None
        chain_joint_order = []
        metric_sample_joint_indices = []
        device = None
        dtype = None

    debug_logging_enabled = debug_wait_handler is not None
    solver = embodik.KinematicsSolver(robot)
    solver.enable_position_ik_debug(debug_logging_enabled)
    solver.dt = config.dt
    solver.set_damping(0.1)
    solver.set_tolerance(0.1)
    LOGGER.info("IK solver configured: dt=%.4f", config.dt)

    velocity_solver: Optional[embodik.KinematicsSolver] = None
    velocity_task = None
    zero_velocity_vec = np.zeros(6, dtype=float)
    # Create velocity solver if using interactive mode OR if excluded joints are needed
    if config.use_interactive_marker_solver or (excluded_joint_indices is not None):
        velocity_solver = embodik.KinematicsSolver(robot)
        velocity_solver.dt = config.dt
        velocity_solver.set_damping(0.1)
        velocity_solver.set_tolerance(0.1)
        velocity_solver.enable_position_ik_debug(debug_logging_enabled)
        velocity_task = velocity_solver.add_frame_task("ik_reachability_interactive_task", end_effector)
        velocity_task.priority = 0
        velocity_task.weight = 0.0
        velocity_task.set_target_velocity(zero_velocity_vec.copy())
        # Apply excluded joint indices if provided (e.g., for torso locking)
        if excluded_joint_indices is not None:
            velocity_task.set_excluded_joint_indices(list(excluded_joint_indices))
            LOGGER.info("Excluded joint indices set for IK reachability: %s", excluded_joint_indices)
        LOGGER.info(
            "Interactive IK mode enabled: pos_gain=%.1f rot_gain=%.1f max_iter=%d",
            config.interactive_pos_gain,
            config.interactive_rot_gain,
            config.interactive_max_iterations,
        )

    # Set initial configuration (use default from robot config)
    q_seed = robot_cfg.default_configuration.copy()
    if q_seed.shape[0] != robot.nq:
        q_seed = np.zeros(robot.nq, dtype=float)
        for name, value in zip(robot_cfg.joint_names, robot_cfg.default_configuration):
            idx = joint_name_to_idx.get(name)
            if idx is not None:
                q_seed[idx] = float(value)
    else:
        # Ensure we have a mutable float array
        q_seed = np.asarray(q_seed, dtype=float).copy()

    if seed_configuration:
        applied = 0
        for joint_name, joint_value in seed_configuration.items():
            idx = joint_name_to_idx.get(joint_name)
            if idx is None:
                continue
            q_seed[idx] = float(joint_value)
            applied += 1
        LOGGER.info(
            "Using external IK seed from current configuration (%d joints applied)",
            applied,
        )

    robot.update_configuration(q_seed)
    LOGGER.info("Initial seed configuration set: %s", q_seed)

    # Get initial end-effector pose for reference
    initial_pose = robot.get_frame_pose(end_effector)
    reference_position = initial_pose.translation.copy()
    LOGGER.info(
        "Initial end-effector pose: position=%s, rotation=\n%s",
        reference_position,
        initial_pose.rotation,
    )

    # Ensure the current pose/seed is the first cached candidate.
    _register_seed_candidate(reference_position, q_seed, metrics=None)

    # Get target orientations
    target_orientations = get_target_orientations(config, initial_pose)
    LOGGER.info(
        "Target orientations: mode=%s, count=%d",
        config.orientation_mode.value,
        len(target_orientations),
    )

    use_parallel = config.num_workers > 1
    if config.use_interactive_marker_solver and use_parallel:
        LOGGER.warning(
            "Interactive IK solver mode is not compatible with multiprocessing; forcing sequential execution."
        )
        use_parallel = False

    # Generate Cartesian grid
    LOGGER.info("Generating Cartesian grid...")
    grid_gen_start = time.perf_counter()
    LOGGER.info(
        "Grid limits: x=[%.3f, %.3f], y=[%.3f, %.3f], z=[%.3f, %.3f]",
        config.x_limits[0],
        config.x_limits[1],
        config.y_limits[0],
        config.y_limits[1],
        config.z_limits[0],
        config.z_limits[1],
    )
    LOGGER.info("Grid step size: %.4f m", config.grid_step_size)
    LOGGER.info(
        "Reach radius filter: %s", config.reach_radius if config.reach_radius else "None"
    )

    grid_points = generate_cartesian_grid(
        config, reference_position, config.reach_radius
    )
    initial_grid_point_count = len(grid_points)
    fk_envelope_samples_used = 0
    fk_envelope_removed_points = 0
    total_points = len(grid_points) * len(target_orientations)
    grid_gen_elapsed = time.perf_counter() - grid_gen_start
    phase_timings["grid_generation"] = grid_gen_elapsed

    LOGGER.info(
        "Grid generation complete: grid_points=%d | orientations=%d | total_targets=%d (%.3f s)",
        len(grid_points),
        len(target_orientations),
        total_points,
        grid_gen_elapsed,
    )

    def _order_grid_points(points: np.ndarray) -> np.ndarray:
        if points.size == 0 or config.traversal_mode == GridTraversalMode.RAW:
            return points
        offsets = points - reference_position
        if config.traversal_mode == GridTraversalMode.DISTANCE:
            order = np.argsort(np.linalg.norm(offsets, axis=1))
            return points[order]
        if config.traversal_mode == GridTraversalMode.AXIS_SWEEP:
            step = max(config.cartesian_resolution, 1e-9)
            index_offsets = np.rint(offsets / step).astype(int)
            key_to_index: Dict[Tuple[int, int, int], int] = {
                tuple(idx_tuple): i for i, idx_tuple in enumerate(index_offsets)
            }
            if not key_to_index:
                return points
            start_idx = int(np.argmin(np.linalg.norm(offsets, axis=1)))
            start_key = tuple(index_offsets[start_idx])
            queue: deque[Tuple[int, int, int]] = deque([start_key])
            seen_keys: Set[Tuple[int, int, int]] = {start_key}
            axis_neighbors = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]
            visited = np.zeros(len(points), dtype=bool)
            sweep_order: List[int] = []

            while queue:
                key = queue.popleft()
                idx = key_to_index.get(key)
                if idx is None or visited[idx]:
                    continue
                sweep_order.append(idx)
                visited[idx] = True
                for axis, direction in axis_neighbors:
                    neighbor = list(key)
                    neighbor[axis] += direction
                    neighbor_key = tuple(neighbor)
                    if neighbor_key in key_to_index and neighbor_key not in seen_keys:
                        queue.append(neighbor_key)
                        seen_keys.add(neighbor_key)

            if not np.all(visited):
                remaining = np.where(~visited)[0]
                remaining_order = remaining[np.argsort(np.linalg.norm(offsets[remaining], axis=1))]
                sweep_order.extend(remaining_order.tolist())

            if sweep_order:
                return points[np.array(sweep_order, dtype=int)]
        return points

    if config.fk_envelope_pruning:
        fk_envelope_samples_used = config.fk_envelope_samples
        envelope_sample_start = time.perf_counter()
        envelope_stats = _estimate_fk_envelope(config.fk_envelope_samples)
        envelope_sample_elapsed = time.perf_counter() - envelope_sample_start
        phase_timings["fk_envelope_sampling"] = envelope_sample_elapsed
        if envelope_stats is None:
            LOGGER.info("FK envelope pruning skipped (unable to estimate envelope).")
        elif len(grid_points) == 0:
            LOGGER.info("FK envelope pruning skipped (no grid points to filter).")
        else:
            LOGGER.info(
                "FK envelope estimate: samples=%d center=%s radius=%.3fm aabb_min=%s aabb_max=%s (%.3fs)",
                envelope_stats["sample_count"],
                np.array2string(envelope_stats["center"], precision=3),
                envelope_stats["radius"],
                np.array2string(envelope_stats["aabb_min"], precision=3),
                np.array2string(envelope_stats["aabb_max"], precision=3),
                envelope_sample_elapsed,
            )
            filter_start = time.perf_counter()
            original_count = len(grid_points)
            filtered_points, removed = _filter_grid_points_by_envelope(
                grid_points,
                envelope_stats,
                config.fk_envelope_margin,
            )
            filter_elapsed = time.perf_counter() - filter_start
            phase_timings["fk_envelope_filter"] = filter_elapsed
            if removed > 0:
                fk_envelope_removed_points = removed
                LOGGER.info(
                    "FK envelope pruning removed %d / %d grid points (%.1f%%) "
                    "using radius=%.3fm (margin=%.3fm, %.3fs).",
                    removed,
                    original_count,
                    100.0 * removed / max(1, original_count),
                    envelope_stats["radius"],
                    config.fk_envelope_margin,
                    filter_elapsed,
                )
                grid_points = filtered_points
                total_points = len(grid_points) * len(target_orientations)
            else:
                LOGGER.info(
                    "FK envelope pruning kept all %d grid points (radius=%.3fm, margin=%.3fm).",
                    len(grid_points),
                    envelope_stats["radius"],
                    config.fk_envelope_margin,
                )

    if config.traversal_mode != GridTraversalMode.RAW and len(grid_points) > 0:
        ordering_start = time.perf_counter()
        grid_points = _order_grid_points(grid_points)
        ordering_elapsed = time.perf_counter() - ordering_start
        LOGGER.info(
            "Grid traversal mode '%s' applied (%.3f s).",
            config.traversal_mode.value,
            ordering_elapsed,
        )

    # Initialize outputs early so it can be used in grid preview code
    outputs: Dict[str, Path] = {}

    # Create preview HDF5 file for grid points (reuses FK reachability map visualization)
    if show_grid_preview and viewer is not None and output_dir is not None:
        LOGGER.info("Creating grid preview file for visualization...")
        reference_position_preview = np.array([0.0, 0.0, 0.0])  # Base origin
        grid_preview_path = create_grid_preview_file(
            config,
            grid_points,
            output_dir,
            reference_position_preview,
        )
        LOGGER.info(
            "Grid preview (post-ordering) contains %d grid points.",
            len(grid_points),
        )
        if grid_preview_path is not None and grid_preview_path.exists():
            try:
                if viewer._show_checkbox is not None:
                    viewer._show_checkbox.value = True
                preview_name = grid_preview_path.name
                if hasattr(viewer, "_map_cache"):
                    viewer._map_cache.pop(preview_name, None)
                if getattr(viewer, "_current_map_id", None) == preview_name:
                    viewer._current_map_id = None
                if grid_preview_path.parent.resolve() != viewer._base_directory.resolve():
                    LOGGER.warning(
                        "Grid preview file is not in viewer base directory. Expected: %s, Got: %s",
                        viewer._base_directory,
                        grid_preview_path.parent,
                    )
                    viewer.refresh_options()
                    if (
                        viewer._map_dropdown
                        and "ik_grid_preview.h5" in viewer._map_dropdown.options
                    ):
                        viewer._map_dropdown.value = "ik_grid_preview.h5"
                        viewer.load_selected_map()
                else:
                    LOGGER.info(
                        "Grid preview file is in viewer base directory, using on_new_map()..."
                    )
                    viewer.on_new_map(grid_preview_path)

                if viewer._current_map_id != "ik_grid_preview.h5":
                    LOGGER.warning(
                        "Map not loaded via normal path, attempting manual load..."
                    )
                    try:
                        loaded_data = viewer._load_map(grid_preview_path)
                        viewer._map_cache["ik_grid_preview.h5"] = loaded_data
                        viewer._current_map_id = "ik_grid_preview.h5"
                        LOGGER.info(
                            "Manually loaded grid preview (%d points)",
                            loaded_data.points.shape[0],
                        )
                        if viewer._map_dropdown:
                            viewer.refresh_options()
                            if "ik_grid_preview.h5" in viewer._map_dropdown.options:
                                viewer._map_dropdown.value = "ik_grid_preview.h5"
                        viewer._update_scene()
                    except Exception as load_exc:
                        LOGGER.exception(
                            "Failed to manually load grid preview: %s", load_exc
                        )

                if viewer._current_map_id == "ik_grid_preview.h5":
                    loaded_data = viewer._map_cache.get("ik_grid_preview.h5")
                    if loaded_data is not None:
                        LOGGER.info(
                            "Grid preview generated and displayed (%d points)",
                            loaded_data.points.shape[0],
                        )
                    else:
                        LOGGER.warning(
                            "Grid preview file created but map data not cached"
                        )
                else:
                    LOGGER.warning(
                        "Grid preview file created but map ID not set (current: %s)",
                        viewer._current_map_id,
                    )
            except Exception as exc:
                LOGGER.warning("Failed to load grid preview in viewer: %s", exc)
                import traceback

                LOGGER.debug("Traceback: %s", traceback.format_exc())
            outputs["grid_preview"] = grid_preview_path

    # Initialize metric helper
    LOGGER.info("Initializing metric helper...")
    metric_init_start = time.perf_counter()
    arm_joint_indices = [
        joint_name_to_idx.get(name) for name in chain_joint_names
    ]
    arm_joint_indices = [idx for idx in arm_joint_indices if idx is not None]
    LOGGER.info("Arm joint indices: %s", arm_joint_indices)
    LOGGER.info(
        "Arm joint names: %s",
        [robot_joint_names[idx] for idx in arm_joint_indices],
    )

    arm_lower = np.array([lower_bounds[idx] for idx in arm_joint_indices])
    arm_upper = np.array([upper_bounds[idx] for idx in arm_joint_indices])
    LOGGER.info("Arm joint limits: lower=%s, upper=%s", arm_lower, arm_upper)

    metric_helper = ReachabilityMetricHelper(
        robot=robot,
        arm_joint_indices=arm_joint_indices,
        arm_lower_bounds=arm_lower,
        arm_upper_bounds=arm_upper,
        manip_scaling=config.manip_scaling,
        metric_chain=metric_chain,
        manip_chain_indices=manip_chain_indices,
        chain_joint_order=chain_joint_order,
        metric_sample_joint_indices=metric_sample_joint_indices,
        metric_chain_device=device,
        metric_chain_dtype=dtype,
    )
    metric_init_elapsed = time.perf_counter() - metric_init_start
    phase_timings["metric_helper_init"] = metric_init_elapsed
    LOGGER.info("Metric helper initialized (%.3f s)", metric_init_elapsed)

    # Estimate metric ranges (use a small sample)
    LOGGER.info("Estimating metric ranges (sampling 1000 configurations)...")
    metric_range_start = time.perf_counter()
    (
        manipulability_raw_max,
        range_of_motion_raw_max,
        singularity_weighted_max,
    ) = metric_helper.estimate_metric_ranges(1000)
    metric_range_elapsed = time.perf_counter() - metric_range_start
    phase_timings["metric_range_estimation"] = metric_range_elapsed
    LOGGER.info(
        "Metric ranges estimated: Manip_max=%.4f, ROM_max=%.4f, Singularity_max=%.4f (%.3f s)",
        manipulability_raw_max,
        range_of_motion_raw_max,
        singularity_weighted_max,
        metric_range_elapsed,
    )

    # Update metric helper metadata with estimated ranges
    metric_helper.metric_metadata["ManipulabilityRawMax"] = manipulability_raw_max
    metric_helper.metric_metadata["RangeOfMotionRawMax"] = range_of_motion_raw_max
    metric_helper.metric_metadata["SingularityWeightedMax"] = singularity_weighted_max

    metric_ranges = {
        "Manipulability": (0.0, manipulability_raw_max),  # Use estimated max, not scaling
        "RangeOfMotion": (0.0, RANGE_OF_MOTION_NORMALIZED_LIMIT),
        "SingularityAvoidance": (0.0, 1.0),
    }

    # Storage for results
    valid_points: List[np.ndarray] = []
    valid_orientations: List[np.ndarray] = []
    valid_configs: List[np.ndarray] = []
    valid_metrics: Dict[str, List[float]] = {
        "Manipulability": [],
        "RangeOfMotion": [],
        "SingularityAvoidance": [],
    }
    invalid_points: List[np.ndarray] = []

    job_name = config.job_name or ik_job_name(robot_cfg.key, config)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    points_processed = 0

    # Track IK computation timing
    ik_solve_times: List[float] = []
    total_ik_solve_time = 0.0

    if progress_callback:
        progress_callback(0.0, "starting")

    LOGGER.info("Starting IK solving loop...")
    LOGGER.info(
        "IK solver options: position_tol=%.4f, orientation_tol=%.4f, max_iter=%d",
        config.position_tolerance,
        config.orientation_tolerance,
        config.max_iterations,
    )
    LOGGER.info("Nullspace bias: enabled=%s", config.use_nullspace_bias)
    LOGGER.info("Parallelization: %d worker(s)", config.num_workers)

    # Note: No data_lock needed for multiprocessing (processes don't share memory)



    def _process_batch_sequential(
        batch_tasks: List[Tuple[int, int, np.ndarray, np.ndarray]],
        q_seed: np.ndarray,
        initial_seed: np.ndarray,
        solver: embodik.KinematicsSolver,
        end_effector: str,
        config: IKReachabilityConfig,
        metric_helper: ReachabilityMetricHelper,
        metric_ranges: Dict[str, Tuple[float, float]],
        valid_points: List[np.ndarray],
        valid_orientations: List[np.ndarray],
        valid_configs: List[np.ndarray],
        valid_metrics: Dict[str, List[float]],
        invalid_points: List[np.ndarray],
        ik_solve_times: List[float],
        show_ik_process: bool,
        visualization_callback: Optional[Callable[[np.ndarray], None]],
        total_tasks: int,
        tasks_start_idx: int,
        progress_callback: Optional[Callable[[float, str], None]],
        joint_name_to_idx: Dict[str, int],
        limit_overrides: Optional[Dict[str, Tuple[float, float]]] = None,
        cancel_event: Optional[threading.Event] = None,
        debug_wait_handler: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[int, np.ndarray, bool]:
        """Process a batch of IK tasks sequentially.

        Args:
            batch_tasks: List of (grid_idx, orient_idx, grid_point, target_orientation)
            q_seed: Seed configuration (updated during processing)
            initial_seed: Original configuration captured at job start (used for retries)
            solver: IK solver instance
            end_effector: End-effector frame name
            config: IK configuration
            metric_helper: Metric computation helper
            metric_ranges: Metric value ranges
            valid_points: List to append valid points
            valid_orientations: List to append valid orientations
            valid_configs: List to append valid configurations
            valid_metrics: Dict to append valid metrics
            invalid_points: List to append invalid points
            ik_solve_times: List to append solve times
            show_ik_process: Whether to visualize IK process
            visualization_callback: Optional visualization callback
            total_tasks: Total number of tasks (for progress reporting)
            tasks_start_idx: Starting index of this batch (for progress reporting)
            progress_callback: Optional progress callback
            joint_name_to_idx: Mapping from joint name to index in configuration vector
            limit_overrides: Optional joint limit overrides to enforce post-solve
            cancel_event: Optional cancellation event
            debug_wait_handler: Optional callback for interactive debugging after each target

        Returns:
            Tuple of (points_processed, updated_q_seed, cancelled_flag)
        """
        batch_points_processed = 0
        current_seed = q_seed.copy()
        last_logged_progress = -1  # Track last logged progress percentage
        neighborhood_radius = max(config.grid_step_size * 1.5, 0.05)
        cancelled = False
        attempts_allowed = max(1, config.retry_attempts)

        for task_idx, (grid_idx, orient_idx, grid_point, target_orientation) in enumerate(batch_tasks):
            if cancel_event is not None and cancel_event.is_set():
                LOGGER.warning("Cancellation requested during sequential batch; stopping early.")
                cancelled = True
                break

            last_status = "PENDING"
            last_reason: Optional[str] = None
            attempts_used = 0

            seed_for_point = _select_seed_for_point(grid_point, current_seed, neighborhood_radius)
            seed_queue = _build_seed_queue(
                seed_for_point,
                grid_point,
                current_seed,
                initial_seed,
                max_attempts=attempts_allowed + max(0, config.retry_seed_pool),
            )

            if debug_wait_handler and seed_queue:
                debug_wait_handler(
                    {
                        "phase": "seed_preview",
                        "grid_idx": grid_idx,
                        "orient_idx": orient_idx,
                        "grid_point": grid_point.copy(),
                        "seed_configuration": seed_queue[0].copy(),
                    }
                )
                debug_wait_handler(
                    {
                        "phase": "solve_request",
                        "grid_idx": grid_idx,
                        "orient_idx": orient_idx,
                        "grid_point": grid_point.copy(),
                    }
                )

            attempt_success = False
            attempt_metrics: Optional[Dict[str, float]] = None
            attempt_solution: Optional[np.ndarray] = None

            for attempt_idx, candidate_seed in enumerate(seed_queue):
                if attempt_idx >= attempts_allowed:
                    break
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break

                attempts_used = attempt_idx + 1
                pos_tol = float(config.position_tolerance * (config.retry_tolerance_scaling ** attempt_idx))
                if config.max_retry_position_tolerance is not None:
                    pos_tol = min(pos_tol, config.max_retry_position_tolerance)
                ori_tol = float(config.orientation_tolerance * (config.retry_tolerance_scaling ** attempt_idx))
                if config.max_retry_orientation_tolerance is not None:
                    ori_tol = min(ori_tol, config.max_retry_orientation_tolerance)
                max_iters = config.max_iterations
                if config.max_retry_iterations is not None:
                    max_iters = min(max_iters, int(config.max_retry_iterations))

                result, ik_solve_elapsed, target_pose = _solve_target(
                    candidate_seed,
                    grid_point,
                    target_orientation,
                    solver,
                    end_effector,
                    config,
                    position_tolerance=pos_tol,
                    orientation_tolerance=ori_tol,
                    max_iterations=max_iters,
                    excluded_joint_indices=excluded_joint_indices,
                )
                ik_solve_times.append(ik_solve_elapsed)

                if show_ik_process and visualization_callback is not None:
                    if result.status == embodik.SolverStatus.SUCCESS:
                        visualization_callback(result.q_solution)
                    else:
                        visualization_callback(candidate_seed)
                    time.sleep(IK_PROCESS_VISUALIZATION_DELAY_SEC)

                if result.status != embodik.SolverStatus.SUCCESS:
                    last_status = result.status.name
                    iterations = getattr(result, "num_iterations", getattr(result, "iterations", None))
                    pos_err = getattr(result, "position_error", getattr(result, "final_position_error", None))
                    rot_err = getattr(result, "orientation_error", getattr(result, "final_orientation_error", None))
                    if debug_logging_enabled:
                        LOGGER.warning(
                            "IK attempt failed (grid %d, orient %d, attempt %d): status=%s iterations=%s pos_err=%s rot_err=%s",
                            grid_idx,
                            orient_idx,
                            attempt_idx + 1,
                            result.status.name,
                            iterations if iterations is not None else "--",
                            pos_err if pos_err is not None else "--",
                            rot_err if rot_err is not None else "--",
                        )
                    last_reason = f"solver status {result.status.name}"
                    continue

                q_solution = result.q_solution
                if not _within_override_bounds(q_solution):
                    last_status = result.status.name
                    last_reason = "joint limit violation"
                    if debug_logging_enabled:
                        LOGGER.warning(
                            "IK joint limit violation (grid %d, orient %d, attempt %d): status=%s",
                            grid_idx,
                            orient_idx,
                            attempt_idx + 1,
                            result.status.name,
                        )
                    continue

                pos_err, rot_err = _compute_pose_error_for_solution(q_solution, target_pose)
                if pos_err > config.position_tolerance or rot_err > config.orientation_tolerance:
                    last_status = result.status.name
                    last_reason = (
                        f"pose error pos={pos_err:.5f}m rot={rot_err:.5f}rad exceeds "
                        f"tol pos={config.position_tolerance:.5f} rot={config.orientation_tolerance:.5f}"
                    )
                    if debug_wait_handler is not None:
                        LOGGER.warning(
                            "IK pose validation failed (grid %d, orient %d, attempt %d): pos_err=%.6f rot_err=%.6f",
                            grid_idx,
                            orient_idx,
                            attempt_idx + 1,
                            pos_err,
                            rot_err,
                        )
                    continue

                metrics = metric_helper.compute_metrics(q_solution, metric_ranges)
                attempt_success = True
                attempt_solution = q_solution.copy()
                attempt_metrics = metrics
                current_seed = q_solution.copy()
                _register_seed_candidate(grid_point, q_solution, metrics)
                last_status = "SUCCESS"
                last_reason = None
                break

            if cancelled:
                break

            if attempt_success and attempt_solution is not None and attempt_metrics is not None:
                valid_points.append(grid_point.copy())
                valid_orientations.append(target_orientation.copy())
                valid_configs.append(attempt_solution.copy())
                valid_metrics["Manipulability"].append(attempt_metrics["manip_raw"])
                valid_metrics["RangeOfMotion"].append(attempt_metrics["rom_norm"])
                valid_metrics["SingularityAvoidance"].append(attempt_metrics["sing_norm"])
            else:
                invalid_points.append(grid_point.copy())
                if last_reason is None and not cancelled:
                    last_reason = "exhausted attempts"

            batch_points_processed += 1

            if debug_wait_handler and not cancelled:
                debug_wait_handler(
                    {
                        "phase": "result",
                        "grid_idx": grid_idx,
                        "orient_idx": orient_idx,
                        "grid_point": grid_point.copy(),
                        "success": attempt_success,
                        "status": last_status,
                        "reason": last_reason,
                        "attempts": attempts_used,
                        "solution_configuration": attempt_solution.copy() if attempt_solution is not None else None,
                    }
                )

            # Log progress every 5% for sequential mode
            if progress_callback and total_tasks > 0:
                current_task_idx = tasks_start_idx + task_idx + 1
                progress = current_task_idx / total_tasks
                progress_percent = int(progress * 100)

                # Log every 5% milestone
                if progress_percent >= last_logged_progress + 5:
                    success_rate = (
                        len(valid_points) / current_task_idx if current_task_idx > 0 else 0.0
                    )
                    avg_ik_time_ms = (
                        (sum(ik_solve_times) / current_task_idx * 1000.0)
                        if current_task_idx > 0
                        else 0.0
                    )
                    LOGGER.info(
                        "Progress: %d%% (%d/%d tasks, %.1f%% success, avg IK: %.2f ms)",
                        progress_percent,
                        current_task_idx,
                        total_tasks,
                        success_rate * 100.0,
                        avg_ik_time_ms,
                    )
                    last_logged_progress = progress_percent

        return batch_points_processed, current_seed.copy(), cancelled

    def _process_batch_parallel(
        batch_tasks: List[Tuple[int, int, np.ndarray, np.ndarray]],
        q_seed: np.ndarray,
        urdf_path: str,
        end_effector: str,
        config: IKReachabilityConfig,
        chain_joint_names: List[str],
        metric_ranges: Dict[str, Tuple[float, float]],
        manipulability_raw_max: float,
        range_of_motion_raw_max: float,
        singularity_weighted_max: float,
        valid_points: List[np.ndarray],
        valid_orientations: List[np.ndarray],
        valid_configs: List[np.ndarray],
        valid_metrics: Dict[str, List[float]],
        invalid_points: List[np.ndarray],
        ik_solve_times: List[float],
        num_workers: int,
        executor: ProcessPoolExecutor,
    ) -> int:
        """Process a batch of IK tasks in parallel using multiprocessing.

        Args:
            batch_tasks: List of (grid_idx, orient_idx, grid_point, target_orientation)
            q_seed: Seed configuration (shared across batch)
            urdf_path: Path to URDF file (for logging, executor already initialized)
            end_effector: End-effector frame name (for logging, executor already initialized)
            config: IK configuration (for logging, executor already initialized)
            chain_joint_names: Joint names (for logging, executor already initialized)
            metric_ranges: Metric value ranges (for logging, executor already initialized)
            manipulability_raw_max: Maximum manipulability (for logging, executor already initialized)
            range_of_motion_raw_max: Maximum ROM (for logging, executor already initialized)
            singularity_weighted_max: Maximum singularity (for logging, executor already initialized)
            valid_points: List to append valid points
            valid_orientations: List to append valid orientations
            valid_configs: List to append valid configurations
            valid_metrics: Dict to append valid metrics
            invalid_points: List to append invalid points
            ik_solve_times: List to append solve times
            num_workers: Number of parallel workers (for logging)
            executor: ProcessPoolExecutor instance (reused across batches)

        Returns:
            Number of points processed
        """
        # Prepare task data with q_seed included
        task_data_list = [
            (grid_idx, orient_idx, grid_point, target_orientation, q_seed)
            for grid_idx, orient_idx, grid_point, target_orientation in batch_tasks
        ]

        # Process batch in parallel using the provided executor (reused across batches)
        batch_points_processed = 0
        futures = {executor.submit(_process_single_ik_task, task_data): task_data for task_data in task_data_list}

        for future in as_completed(futures):
            try:
                result_data = future.result()
                if result_data is None:
                    continue

                ik_solve_times.append(result_data["ik_time"])

                if result_data["success"]:
                    valid_points.append(result_data["grid_point"].copy())
                    valid_orientations.append(result_data["target_orientation"].copy())
                    valid_configs.append(result_data["q_solution"].copy())
                    valid_metrics["Manipulability"].append(result_data["metrics"]["manip_raw"])
                    valid_metrics["RangeOfMotion"].append(result_data["metrics"]["rom_norm"])
                    valid_metrics["SingularityAvoidance"].append(result_data["metrics"]["sing_norm"])
                else:
                    invalid_points.append(result_data["grid_point"].copy())

                batch_points_processed += 1
            except Exception as exc:
                LOGGER.warning("Error in parallel worker: %s", exc)

        return batch_points_processed

    # IK solving phase timing
    ik_solving_start = time.perf_counter()
    # Prepare all tasks: (grid_idx, orient_idx, grid_point, target_orientation)
    all_tasks: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
    for grid_idx, grid_point in enumerate(grid_points):
        for orient_idx, target_orientation in enumerate(target_orientations):
            all_tasks.append((grid_idx, orient_idx, grid_point.copy(), target_orientation.copy()))

    total_points = len(all_tasks)
    LOGGER.info("Total IK tasks: %d (grid_points=%d × orientations=%d)", total_points, len(grid_points), len(target_orientations))

    # Determine batch size and processing mode
    # For multiprocessing, use larger batches to amortize process initialization overhead
    # For sequential, process all tasks in one batch
    if use_parallel:
        # Larger batch size for multiprocessing to amortize initialization cost
        # Each worker initializes its own robot model, so we need enough work per batch
        batch_size = max(200, total_points // config.num_workers)
        LOGGER.info(
            "Using multiprocessing with %d workers. "
            "Note: Process initialization overhead is significant. "
            "Multiprocessing is only beneficial for large workloads (>1000 tasks). "
            "For smaller workloads, use num_workers=1.",
            config.num_workers
        )
    else:
        batch_size = total_points
    LOGGER.info("Batch processing: batch_size=%d, parallel=%s", batch_size, use_parallel)

    # Warning about GIL limitation with threading (now using multiprocessing)
    if use_parallel:
        LOGGER.info(
            "Using ProcessPoolExecutor with %d workers for true parallelism. "
            "Each worker will initialize its own robot model and solver.",
            config.num_workers
        )

    # Disable visualization in parallel mode (not thread-safe)
    if use_parallel and show_ik_process:
        LOGGER.info("Visualization disabled in parallel mode (not thread-safe)")
        show_ik_process = False
        visualization_callback = None

    # Process in batches
    initial_q_seed = q_seed.copy()

    # For multiprocessing, create ProcessPoolExecutor once and reuse across batches
    executor = None
    if use_parallel:
        # Convert config to dictionary for pickling (only need to do this once)
        config_dict = {
            "grid_step_size": config.grid_step_size,
            "x_limits": config.x_limits,
            "y_limits": config.y_limits,
            "z_limits": config.z_limits,
            "reach_radius": config.reach_radius,
            "orientation_mode": config.orientation_mode.value,
            "reference_orientation": config.reference_orientation,
            "orientation_samples": config.orientation_samples,
            "position_tolerance": config.position_tolerance,
            "orientation_tolerance": config.orientation_tolerance,
            "max_iterations": config.max_iterations,
            "dt": config.dt,
            "use_nullspace_bias": config.use_nullspace_bias,
            "nullspace_bias": config.nullspace_bias,
            "nullspace_gain": config.nullspace_gain,
            "nullspace_active_joints": config.nullspace_active_joints,
            "max_linear_step": config.max_linear_step,
            "max_angular_step": config.max_angular_step,
            "cartesian_resolution": config.cartesian_resolution,
            "angular_resolution": config.angular_resolution,
            "manip_scaling": config.manip_scaling,
            "job_name": config.job_name,
            "num_workers": config.num_workers,
            "retry_attempts": config.retry_attempts,
            "retry_seed_pool": config.retry_seed_pool,
            "retry_seed_jitter": config.retry_seed_jitter,
            "retry_tolerance_scaling": config.retry_tolerance_scaling,
            "retry_iteration_scaling": config.retry_iteration_scaling,
            "max_retry_position_tolerance": config.max_retry_position_tolerance,
            "max_retry_orientation_tolerance": config.max_retry_orientation_tolerance,
            "max_retry_iterations": config.max_retry_iterations,
            "fk_envelope_pruning": config.fk_envelope_pruning,
            "fk_envelope_samples": config.fk_envelope_samples,
            "fk_envelope_margin": config.fk_envelope_margin,
            "use_interactive_marker_solver": config.use_interactive_marker_solver,
            "interactive_pos_gain": config.interactive_pos_gain,
            "interactive_rot_gain": config.interactive_rot_gain,
            "interactive_max_iterations": config.interactive_max_iterations,
        }
        LOGGER.info("Initializing ProcessPoolExecutor with %d workers...", config.num_workers)
        executor = ProcessPoolExecutor(
            max_workers=config.num_workers,
            initializer=_init_worker_process,
            initargs=(
                str(loadable_urdf_path),
                end_effector,
                config_dict,
                list(chain_joint_names),
                metric_ranges,
                manipulability_raw_max,
                range_of_motion_raw_max,
                singularity_weighted_max,
                dict(robot_cfg.limit_overrides),
                list(excluded_joint_indices) if excluded_joint_indices else None,
            ),
        )
        LOGGER.info("ProcessPoolExecutor initialized. Workers ready.")

    try:
        for batch_start in range(0, total_points, batch_size):
            if cancel_event is not None and cancel_event.is_set():
                LOGGER.warning("Cancellation requested; aborting IK reachability generation.")
                break

            batch_end = min(batch_start + batch_size, total_points)
            batch_tasks = all_tasks[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1
            total_batches = (total_points + batch_size - 1) // batch_size

            LOGGER.info(
                "Processing batch %d/%d (tasks %d-%d, %.1f%%)",
                batch_num,
                total_batches,
                batch_start + 1,
                batch_end,
                (batch_end / total_points * 100.0) if total_points > 0 else 0.0,
            )

            if use_parallel:
                # Parallel batch processing using multiprocessing (reuse executor)
                batch_points_processed = _process_batch_parallel(
                    batch_tasks=batch_tasks,
                    q_seed=initial_q_seed,  # Use fixed initial seed for all tasks
                    urdf_path=str(loadable_urdf_path),
                    end_effector=end_effector,
                    config=config,
                    chain_joint_names=list(chain_joint_names),
                    metric_ranges=metric_ranges,
                    manipulability_raw_max=manipulability_raw_max,
                    range_of_motion_raw_max=range_of_motion_raw_max,
                    singularity_weighted_max=singularity_weighted_max,
                    valid_points=valid_points,
                    valid_orientations=valid_orientations,
                    valid_configs=valid_configs,
                    valid_metrics=valid_metrics,
                    invalid_points=invalid_points,
                    ik_solve_times=ik_solve_times,
                    num_workers=config.num_workers,
                    executor=executor,  # Reuse executor across batches
                )
                # Update seed: use average of successful solutions from batch (or keep current)
                # For simplicity, keep current seed for next batch
                # TODO: Could compute average successful solution from batch
            else:
                # Sequential batch processing
                batch_points_processed, _, batch_cancelled = _process_batch_sequential(
                    batch_tasks=batch_tasks,
                    q_seed=initial_q_seed,
                    initial_seed=initial_q_seed,
                    solver=solver,
                    end_effector=end_effector,
                    config=config,
                    metric_helper=metric_helper,
                    metric_ranges=metric_ranges,
                    valid_points=valid_points,
                    valid_orientations=valid_orientations,
                    valid_configs=valid_configs,
                    valid_metrics=valid_metrics,
                    invalid_points=invalid_points,
                    ik_solve_times=ik_solve_times,
                    show_ik_process=show_ik_process,
                    visualization_callback=visualization_callback,
                    total_tasks=total_points,
                    tasks_start_idx=batch_start,
                    progress_callback=progress_callback,
                    joint_name_to_idx=joint_name_to_idx,
                    limit_overrides=robot_cfg.limit_overrides,
                    cancel_event=cancel_event,
                    debug_wait_handler=debug_wait_handler,
                )
                if batch_cancelled:
                    break

            points_processed += batch_points_processed
            total_ik_solve_time = sum(ik_solve_times)

            # Progress reporting
            if progress_callback:
                progress = points_processed / total_points if total_points > 0 else 0.0
                success_rate = (
                    len(valid_points) / points_processed if points_processed > 0 else 0.0
                )
                avg_ik_time_ms = (
                    (total_ik_solve_time / points_processed * 1000.0)
                    if points_processed > 0
                    else 0.0
                )
                progress_callback(
                    progress,
                    f"solving ({points_processed}/{total_points}, {success_rate*100:.1f}% success, avg IK: {avg_ik_time_ms:.2f}ms)",
                )
    finally:
        # Clean up executor if we created it
        if executor is not None:
            LOGGER.info("Shutting down ProcessPoolExecutor...")
            executor.shutdown(wait=True)
            LOGGER.info("ProcessPoolExecutor shut down.")

    # IK solving phase complete
    ik_solving_elapsed = time.perf_counter() - ik_solving_start
    phase_timings["ik_solving"] = ik_solving_elapsed
    LOGGER.info("IK solving phase completed (%.3f s)", ik_solving_elapsed)

    # Convert to numpy arrays
    LOGGER.info("Converting results to numpy arrays...")
    data_conversion_start = time.perf_counter()
    LOGGER.info(
        "Valid solutions: %d, Invalid solutions: %d",
        len(valid_points),
        len(invalid_points),
    )

    if valid_points:
        valid_points_np = np.vstack(valid_points).astype(np.float32)
        valid_orientations_np = np.array(valid_orientations)
        valid_configs_np = np.vstack(valid_configs).astype(np.float32)
        valid_metrics_np = {
            k: np.array(v, dtype=np.float32) for k, v in valid_metrics.items()
        }

        # Log metric statistics for debugging
        LOGGER.info("Metric statistics:")
        for metric_name, metric_values in valid_metrics_np.items():
            if metric_values.size > 0:
                LOGGER.info(
                    "  %s: min=%.6f, max=%.6f, mean=%.6f, std=%.6f",
                    metric_name,
                    float(np.min(metric_values)),
                    float(np.max(metric_values)),
                    float(np.mean(metric_values)),
                    float(np.std(metric_values)),
                )
            else:
                LOGGER.warning("  %s: No values!", metric_name)
    else:
        valid_points_np = np.zeros((0, 3), dtype=np.float32)
        valid_orientations_np = np.zeros((0, 3, 3), dtype=np.float32)
        valid_configs_np = np.zeros((0, robot.nq), dtype=np.float32)
        valid_metrics_np = {
            k: np.zeros(0, dtype=np.float32) for k in valid_metrics.keys()
        }

    invalid_points_np = (
        np.vstack(invalid_points).astype(np.float32)
        if invalid_points
        else np.zeros((0, 3), dtype=np.float32)
    )

    # Compute global scores
    LOGGER.info("Computing global scores...")
    global_scores = compute_ik_global_scores(
        valid_points_np, invalid_points_np, valid_metrics_np
    )
    LOGGER.info(
        "Global scores computed: coverage=%.1f%%, valid=%d/%d",
        global_scores.get("coverage", {}).get("coverage_ratio", 0.0) * 100.0,
        global_scores.get("coverage", {}).get("valid_count", 0),
        global_scores.get("coverage", {}).get("total_count", 0),
    )

    # Convert orientations to RPY
    LOGGER.info("Converting orientations to RPY and quaternion...")
    valid_orientations_rpy = np.zeros((len(valid_points_np), 3), dtype=np.float32)
    valid_orientations_quat = np.zeros((len(valid_points_np), 4), dtype=np.float32)
    for i, rot_mat in enumerate(valid_orientations_np):
        se3_pose = pin.SE3(rot_mat, valid_points_np[i])
        # Convert rotation matrix to RPY using scipy
        rpy = R.from_matrix(rot_mat).as_euler('xyz', degrees=False)
        valid_orientations_rpy[i] = rpy
        quat = tuple(r2q(rot_mat))
        valid_orientations_quat[i] = quat

    # Bin into same grid as FK approach for compatibility
    LOGGER.info("Binning results into FK-compatible format...")
    data_conversion_elapsed = time.perf_counter() - data_conversion_start
    phase_timings["data_conversion"] = data_conversion_elapsed
    LOGGER.info("Data conversion completed (%.3f s)", data_conversion_elapsed)
    x_bins = int(
        np.ceil((config.x_limits[1] - config.x_limits[0]) / config.cartesian_resolution)
    )
    y_bins = int(
        np.ceil((config.y_limits[1] - config.y_limits[0]) / config.cartesian_resolution)
    )
    z_bins = int(
        np.ceil((config.z_limits[1] - config.z_limits[0]) / config.cartesian_resolution)
    )
    roll_bins = int(np.ceil((np.pi * 2.0) / config.angular_resolution))
    pitch_bins = int(np.ceil((np.pi) / config.angular_resolution))
    yaw_bins = int(np.ceil((np.pi * 2.0) / config.angular_resolution))

    lower_bounds_np = np.array(
        [
            config.x_limits[0],
            config.y_limits[0],
            config.z_limits[0],
            -np.pi,
            -np.pi / 2.0,
            -np.pi,
        ],
        dtype=np.float32,
    )
    resolutions_np = np.array(
        [
            config.cartesian_resolution,
            config.cartesian_resolution,
            config.cartesian_resolution,
            config.angular_resolution,
            config.angular_resolution,
            config.angular_resolution,
        ],
        dtype=np.float32,
    )
    centers_np = lower_bounds_np + 0.5 * resolutions_np

    # Bin valid points
    reach_map_rows: List[np.ndarray] = []
    for i in range(len(valid_points_np)):
        xyz = valid_points_np[i]
        rpy = valid_orientations_rpy[i]
        quat = valid_orientations_quat[i]  # Use quaternion directly
        pose_6d = np.concatenate([xyz, rpy])

        # Compute bin indices
        bins = ((pose_6d - lower_bounds_np) / resolutions_np).astype(np.int32)
        bins = np.clip(
            bins,
            0,
            [
                x_bins - 1,
                y_bins - 1,
                z_bins - 1,
                roll_bins - 1,
                pitch_bins - 1,
                yaw_bins - 1,
            ],
        )

        # Store: xyz(3) + rpy(3) + visitation(1) + manip(1) + rom_sum(1) + sing_sum(1) + quat(4) = 14 columns
        # We add quaternion to avoid recomputing it later
        manip = valid_metrics_np["Manipulability"][i]
        rom = valid_metrics_np["RangeOfMotion"][i]
        sing = valid_metrics_np["SingularityAvoidance"][i]

        reach_map_rows.append(
            np.array(
                [
                    xyz[0],
                    xyz[1],
                    xyz[2],
                    rpy[0],
                    rpy[1],
                    rpy[2],
                    1.0,  # visitation
                    manip,  # manipulability
                    rom,  # ROM (normalized)
                    sing,  # singularity (normalized)
                    quat[0],  # quaternion w
                    quat[1],  # quaternion x
                    quat[2],  # quaternion y
                    quat[3],  # quaternion z
                ],
                dtype=np.float32,
            )
        )

    # Save reachability map
    LOGGER.info("Saving reachability map...")
    map_path: Optional[Path] = None
    if reach_map_rows:
        reach_map_np = np.vstack(reach_map_rows)
        map_path = output_dir / f"reach_map_{job_name}.pkl"
        import pickle

        with open(map_path, "wb") as handle:
            pickle.dump(reach_map_np, handle)
        outputs["map"] = map_path

        # Store sphere points directly at grid point positions (no aggregation)
        # For IK-based reachability, each grid point is a distinct test case
        # Format: xyz(3) + manipulability(1) + visitation(1) + rom(1) + sing(1) = 7 columns
        sphere_rows: List[np.ndarray] = []
        pose_rows: List[np.ndarray] = []

        # Log statistics before creating sphere dataset
        if reach_map_rows:
            reach_map_np_for_log = np.vstack(reach_map_rows)
            LOGGER.info("Reach map statistics:")
            LOGGER.info("  Manipulability (col %d): min=%.6f, max=%.6f, mean=%.6f",
                       IK_REACH_MAP_MANIP_COL,
                       float(np.min(reach_map_np_for_log[:, IK_REACH_MAP_MANIP_COL])),
                       float(np.max(reach_map_np_for_log[:, IK_REACH_MAP_MANIP_COL])),
                       float(np.mean(reach_map_np_for_log[:, IK_REACH_MAP_MANIP_COL])))
            LOGGER.info("  ROM (col %d): min=%.6f, max=%.6f, mean=%.6f",
                       IK_REACH_MAP_ROM_COL,
                       float(np.min(reach_map_np_for_log[:, IK_REACH_MAP_ROM_COL])),
                       float(np.max(reach_map_np_for_log[:, IK_REACH_MAP_ROM_COL])),
                       float(np.mean(reach_map_np_for_log[:, IK_REACH_MAP_ROM_COL])))
            LOGGER.info("  Singularity (col %d): min=%.6f, max=%.6f, mean=%.6f",
                       IK_REACH_MAP_SINGULARITY_COL,
                       float(np.min(reach_map_np_for_log[:, IK_REACH_MAP_SINGULARITY_COL])),
                       float(np.max(reach_map_np_for_log[:, IK_REACH_MAP_SINGULARITY_COL])),
                       float(np.mean(reach_map_np_for_log[:, IK_REACH_MAP_SINGULARITY_COL])))

        # Create one sphere point per valid IK solution at the actual grid point position
        for row in reach_map_rows:
            xyz = row[IK_REACH_MAP_XYZ_START:IK_REACH_MAP_XYZ_END]  # Actual grid point position
            manip = row[IK_REACH_MAP_MANIP_COL]  # Manipulability (raw)
            visitation = row[IK_REACH_MAP_VISITATION_COL]  # Visitation count (always 1 for IK)
            rom = row[IK_REACH_MAP_ROM_COL]  # ROM (normalized)
            sing = row[IK_REACH_MAP_SINGULARITY_COL]  # Singularity (normalized)
            rpy = row[IK_REACH_MAP_RPY_START:IK_REACH_MAP_RPY_END]  # RPY orientation
            quat = row[IK_REACH_MAP_QUAT_START:IK_REACH_MAP_QUAT_END]  # Quaternion

            sphere_rows.append(
                np.array(
                    [
                        xyz[0],
                        xyz[1],
                        xyz[2],
                        manip,
                        visitation,
                        rom,
                        sing,
                    ],
                    dtype=np.float32,
                )
            )

            # Store pose at grid point position
            pose_rows.append(
                np.concatenate([xyz, rpy, quat])
            )
    else:
        sphere_rows = []
        pose_rows = []

    # Save HDF5 file
    LOGGER.info("Saving HDF5 file...")
    file_saving_start = time.perf_counter()
    h5_path = output_dir / f"3D_{job_name}.h5"
    with h5py.File(h5_path, "w") as handle:
        sphere_group = handle.create_group("/Spheres")
        if sphere_rows:
            sphere_data_np = np.vstack(sphere_rows)
            # Log statistics of final sphere data
            LOGGER.debug("Final sphere dataset statistics:")
            LOGGER.debug("  Manipulability (col %d): min=%.6f, max=%.6f, mean=%.6f",
                       SPHERE_DATASET_MANIP_COL,
                       float(np.min(sphere_data_np[:, SPHERE_DATASET_MANIP_COL])),
                       float(np.max(sphere_data_np[:, SPHERE_DATASET_MANIP_COL])),
                       float(np.mean(sphere_data_np[:, SPHERE_DATASET_MANIP_COL])))
            LOGGER.debug("  ROM (col %d): min=%.6f, max=%.6f, mean=%.6f",
                       SPHERE_DATASET_ROM_COL,
                       float(np.min(sphere_data_np[:, SPHERE_DATASET_ROM_COL])),
                       float(np.max(sphere_data_np[:, SPHERE_DATASET_ROM_COL])),
                       float(np.mean(sphere_data_np[:, SPHERE_DATASET_ROM_COL])))
            LOGGER.debug("  Singularity (col %d): min=%.6f, max=%.6f, mean=%.6f",
                       SPHERE_DATASET_SINGULARITY_COL,
                       float(np.min(sphere_data_np[:, SPHERE_DATASET_SINGULARITY_COL])),
                       float(np.max(sphere_data_np[:, SPHERE_DATASET_SINGULARITY_COL])),
                       float(np.mean(sphere_data_np[:, SPHERE_DATASET_SINGULARITY_COL])))

            dataset = sphere_group.create_dataset(
                "sphere_dataset", data=sphere_data_np
            )
            dataset.attrs.create("Resolution", data=config.cartesian_resolution)
            dataset.attrs.create("CenterOffset", data=centers_np[:3])
            dataset.attrs.create("LowerBounds", data=lower_bounds_np[:3])
            dataset.attrs.create(
                "MetricNames",
                np.asarray(
                    ["Manipulability", "RangeOfMotion", "SingularityAvoidance"],
                    dtype="S32",
                ),
            )
            dataset.attrs.create("MetricColumns", np.asarray([SPHERE_DATASET_MANIP_COL, SPHERE_DATASET_ROM_COL, SPHERE_DATASET_SINGULARITY_COL], dtype=np.int32))
            dataset.attrs.create("VisitationColumn", np.int32(SPHERE_DATASET_VISITATION_COL))
            dataset.attrs.create("ManipulabilityRawMax", manipulability_raw_max)
            dataset.attrs.create("RangeOfMotionRawMin", 0.0)
            dataset.attrs.create("RangeOfMotionRawMax", range_of_motion_raw_max)
            dataset.attrs.create("SingularityWeightedMax", singularity_weighted_max)

            # Store global scores
            scores_json = {}
            for metric_name, scores in global_scores.get("metrics", {}).items():
                scores_json[metric_name] = {
                    k: float(v)
                    if isinstance(v, (np.integer, np.floating))
                    else int(v)
                    if isinstance(v, np.integer)
                    else v
                    for k, v in scores.items()
                }
            if "coverage" in global_scores:
                scores_json["coverage"] = {
                    k: float(v)
                    if isinstance(v, (np.integer, np.floating))
                    else int(v)
                    if isinstance(v, np.integer)
                    else v
                    for k, v in global_scores["coverage"].items()
                }
            # Store JSON string using h5py's string_dtype for variable-length UTF-8 strings
            json_str = json.dumps(scores_json)
            str_dtype = h5py.string_dtype(encoding='utf-8')
            if robot_cfg.metadata:
                def _metadata_default(obj: Any) -> Any:
                    if isinstance(obj, (np.floating, np.integer)):
                        return float(obj)
                    if isinstance(obj, (np.ndarray,)):
                        return obj.tolist()
                    if isinstance(obj, (set, tuple)):
                        return list(obj)
                    return str(obj)

                try:
                    robot_metadata_json = json.dumps(
                        robot_cfg.metadata, default=_metadata_default
                    )
                except (TypeError, ValueError) as exc:
                    LOGGER.warning(
                        "Failed to serialize robot metadata for IK reachability '%s': %s",
                        job_name,
                        exc,
                    )
                else:
                    dataset.attrs.create(
                        "RobotMetadata",
                        robot_metadata_json,
                        dtype=str_dtype,
                    )
            dataset.attrs.create("GlobalScores", json_str, dtype=str_dtype)
            LOGGER.info(
                "Computed global scores: %s",
                ", ".join(
                    f"{k}: {v.get('average', 0):.3f}"
                    for k, v in global_scores.get("metrics", {}).items()
                ),
            )

        pose_group = handle.create_group("/Poses")
        if pose_rows:
            pose_group.create_dataset("poses_dataset", data=np.vstack(pose_rows))
        else:
            pose_group.create_dataset(
                "poses_dataset", data=np.zeros((0, 10), dtype=np.float32)
            )

    outputs["map_3d"] = h5_path
    if map_path is not None:
        LOGGER.info(
            "Saved IK reachability snapshot: %s, %s", map_path.name, h5_path.name
        )
    else:
        LOGGER.info(
            "Saved IK reachability snapshot: (no map), %s", h5_path.name
        )

    # Save samples
    if valid_configs_np.shape[0] > 0:
        samples_path = output_dir / f"samples_{job_name}.npz"
        robot_joint_names = robot.get_joint_names()

        # Verify dimensions match
        # Note: q_solution has robot.nq dimensions (all DOFs), which may include fixed/floating base DOFs
        # that aren't in get_joint_names(). Compare against robot.nq instead.
        if valid_configs_np.shape[1] != robot.nq:
            LOGGER.error(
                "Mismatch: valid_configs_np.shape[1]=%d but robot.nq=%d (robot has %d joint names)",
                valid_configs_np.shape[1],
                robot.nq,
                len(robot_joint_names),
            )
            raise ValueError(
                f"Configuration dimension mismatch: configs have {valid_configs_np.shape[1]} DOFs "
                f"but robot.nq={robot.nq} (robot has {len(robot_joint_names)} joint names)"
            )

        # Extract only joint DOFs if robot.nq > len(joint_names) (e.g., fixed joints or floating base)
        # For robots with floating_base=False, joint indices should match configuration indices
        # But if there's a mismatch, we need to extract only the joint DOFs
        if valid_configs_np.shape[1] == len(robot_joint_names):
            # Direct match: use configurations as-is
            samples_q = valid_configs_np
            joint_names_array = np.asarray(robot_joint_names, dtype=np.dtype("U64"))
        else:
            # Mismatch: extract only joint DOFs
            # For fixed-base robots, joint indices should be 0..len(joint_names)-1
            # For floating-base robots, joint indices are offset by 7
            # Since we're using floating_base=False, assume joints are the first len(joint_names) DOFs
            LOGGER.warning(
                "Robot has %d DOFs but only %d joint names. Extracting first %d DOFs for samples.",
                valid_configs_np.shape[1],
                len(robot_joint_names),
                len(robot_joint_names),
            )
            samples_q = valid_configs_np[:, :len(robot_joint_names)]
            joint_names_array = np.asarray(robot_joint_names, dtype=np.dtype("U64"))

        # Compute bucket indices for valid points
        # Use the same computation as the viewer's _compute_indices to ensure matching keys
        samples_ijk = np.zeros((len(valid_points_np), 3), dtype=np.int32)
        for i, xyz in enumerate(valid_points_np):
            # Match viewer's _compute_indices: np.floor((points - lb) / resolution + 1e-9)
            bins = np.floor((xyz - lower_bounds_np[:3]) / resolutions_np[:3] + 1e-9).astype(np.int32)
            bins = np.clip(bins, 0, [x_bins - 1, y_bins - 1, z_bins - 1])
            samples_ijk[i] = bins

        # Log first few bucket indices for debugging
        if len(valid_points_np) > 0:
            LOGGER.info("Sample bucket indices (first 5): %s", [tuple(row.tolist()) for row in samples_ijk[:5]])
            LOGGER.info("Sample positions (first 5): %s", valid_points_np[:5].tolist())
            LOGGER.info("Lower bounds: %s, Resolution: %s", lower_bounds_np[:3], resolutions_np[:3])

        np.savez_compressed(
            samples_path,
            xyz=valid_points_np,
            rpy=valid_orientations_rpy,
            quat=valid_orientations_quat,
            ijk=samples_ijk,
            q=samples_q,
            joint_names=joint_names_array,
        )
        outputs["samples"] = samples_path
        LOGGER.info(
            "Stored IK sampled configurations: %s (%d samples)",
            samples_path.name,
            len(valid_points_np),
        )
        # Verify the samples file exists and can be loaded
        if samples_path.exists():
            try:
                test_data = np.load(samples_path)
                LOGGER.info(
                    "Verified samples file: xyz=%s, q=%s, ijk=%s",
                    test_data["xyz"].shape if "xyz" in test_data else "missing",
                    test_data["q"].shape if "q" in test_data else "missing",
                    test_data["ijk"].shape if "ijk" in test_data else "missing",
                )
            except Exception as exc:
                LOGGER.warning("Failed to verify samples file: %s", exc)
        else:
            LOGGER.error("Samples file was not created: %s", samples_path)

    file_saving_elapsed = time.perf_counter() - file_saving_start
    phase_timings["file_saving"] = file_saving_elapsed
    LOGGER.info("File saving completed (%.3f s)", file_saving_elapsed)

    # Calculate total elapsed time
    total_elapsed = time.perf_counter() - total_start_time
    phase_timings["total"] = total_elapsed

    # Log comprehensive timing summary
    LOGGER.info("=" * 80)
    LOGGER.info("REACHABILITY MAP GENERATION TIMING SUMMARY")
    LOGGER.info("=" * 80)
    LOGGER.info("Configuration: num_workers=%d, grid_points=%d, orientations=%d, total_targets=%d",
                config.num_workers, len(grid_points), len(target_orientations), total_points)
    LOGGER.info("Results: valid=%d, invalid=%d, success_rate=%.1f%%",
                len(valid_points), len(invalid_points),
                (len(valid_points) / total_points * 100.0) if total_points > 0 else 0.0)
    if config.fk_envelope_pruning:
        LOGGER.info(
            "FK envelope pruning summary: samples=%d, removed=%d, remaining=%d (initial=%d, margin=%.3fm)",
            fk_envelope_samples_used,
            fk_envelope_removed_points,
            len(grid_points),
            initial_grid_point_count,
            config.fk_envelope_margin,
        )
    else:
        LOGGER.info(
            "FK envelope pruning disabled (initial grid points=%d).",
            initial_grid_point_count,
        )
    LOGGER.info("-" * 80)
    LOGGER.info("Phase Timings:")
    for phase_name, phase_time in sorted(phase_timings.items()):
        if phase_name != "total":
            percentage = (phase_time / total_elapsed * 100.0) if total_elapsed > 0 else 0.0
            LOGGER.info("  %-30s: %8.3f s (%5.1f%%)", phase_name, phase_time, percentage)
    LOGGER.info("-" * 80)
    LOGGER.info("  %-30s: %8.3f s (100.0%%)", "TOTAL", total_elapsed)

    # Performance analysis for parallel mode
    if config.num_workers > 1 and ik_solve_times:
        total_ik_wall_time = phase_timings.get("ik_solving", 0.0)
        total_ik_cpu_time = sum(ik_solve_times)
        overhead_ratio = (total_ik_wall_time / total_ik_cpu_time) if total_ik_cpu_time > 0 else 0.0
        LOGGER.info("-" * 80)
        LOGGER.info("Parallel Performance Analysis:")
        LOGGER.info("  Total IK solve CPU time (sum): %.3f s", total_ik_cpu_time)
        LOGGER.info("  IK solving wall-clock time: %.3f s", total_ik_wall_time)
        LOGGER.info("  Overhead ratio: %.2fx (lower is better, 1.0 = perfect parallelization)", overhead_ratio)
        if overhead_ratio > 2.0:
            LOGGER.warning(
                "  High overhead detected! Multiprocessing overhead is %.1fx the actual compute time. "
                "Consider using num_workers=1 (sequential) for better performance.",
                overhead_ratio
            )

    LOGGER.info("=" * 80)

    metadata_payload = {
        "robot_key": robot_cfg.key,
        "end_effector": end_effector,
        "limit_overrides": robot_cfg.limit_overrides,
        "metadata": robot_cfg.metadata,
        "job_name": job_name,
        "fk_envelope": {
            "enabled": bool(config.fk_envelope_pruning),
            "samples": fk_envelope_samples_used if config.fk_envelope_pruning else 0,
            "removed_points": fk_envelope_removed_points if config.fk_envelope_pruning else 0,
            "initial_grid_points": initial_grid_point_count,
            "remaining_grid_points": len(grid_points),
            "margin_m": config.fk_envelope_margin if config.fk_envelope_pruning else 0.0,
        },
        "interactive_solver_enabled": bool(config.use_interactive_marker_solver),
    }
    metadata_path = output_dir / f"{job_name}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as meta_handle:
        json.dump(metadata_payload, meta_handle, indent=2, sort_keys=True)
    outputs["metadata"] = metadata_path

    elapsed = time.perf_counter() - start_time

    # Calculate and log IK timing statistics
    if ik_solve_times:
        avg_ik_time = np.mean(ik_solve_times)
        min_ik_time = np.min(ik_solve_times)
        max_ik_time = np.max(ik_solve_times)
        median_ik_time = np.median(ik_solve_times)
        std_ik_time = np.std(ik_solve_times)

        LOGGER.info("IK computation timing statistics:")
        LOGGER.info("  Total IK solves: %d", len(ik_solve_times))
        LOGGER.info("  Total IK solve time: %.3f s (%.1f%% of total)", total_ik_solve_time, (total_ik_solve_time / elapsed * 100.0) if elapsed > 0 else 0.0)
        LOGGER.info("  Average IK solve time: %.3f ms", avg_ik_time * 1000.0)
        LOGGER.info("  Median IK solve time: %.3f ms", median_ik_time * 1000.0)
        LOGGER.info("  Min IK solve time: %.3f ms", min_ik_time * 1000.0)
        LOGGER.info("  Max IK solve time: %.3f ms", max_ik_time * 1000.0)
        LOGGER.info("  Std dev IK solve time: %.3f ms", std_ik_time * 1000.0)
    else:
        LOGGER.warning("No IK solve times recorded")

    LOGGER.info("IK reachability generation completed in %.2f s", elapsed)
    if progress_callback:
        progress_callback(1.0, "done")

    # Clear visualization callback to ensure it doesn't interfere with sample picking
    # The callback should only be active during IK solving, not after completion
    LOGGER.info("IK generation complete - visualization callback is no longer active")

    return outputs
