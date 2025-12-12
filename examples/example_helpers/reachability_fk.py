#!/usr/bin/env python3
"""Shared FK-based reachability map generation functions.

This module provides the core forward kinematics sampling logic for generating
reachability maps. It supports both generic robots (example 03) and Alpha-specific
configurations (example 04) without code duplication.
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

try:
    import pytorch_kinematics as pk
except ImportError as exc:
    raise ImportError(
        "pytorch_kinematics is required for FK reachability generation. "
        "Install sampled_reachability_maps dependencies."
    ) from exc

import embodik
import yourdfpy

from .reachability_constants import (
    DEFAULT_JOINT_LIMIT_DEADBAND_RATIO,
    JOINT_LIMIT_EPSILON,
    RANGE_OF_MOTION_NORMALIZED_LIMIT,
    FK_REACH_MAP_VISITATION_COL,
    FK_REACH_MAP_MANIP_COL,
    FK_REACH_MAP_ROM_COL,
    FK_REACH_MAP_SINGULARITY_COL,
    SPHERE_DATASET_MANIP_COL,
    SPHERE_DATASET_VISITATION_COL,
    SPHERE_DATASET_ROM_COL,
    SPHERE_DATASET_SINGULARITY_COL,
)
from .reachability_metrics import (
    normalize_range_of_motion_scalar,
    normalize_singularity_scalar,
)
from .reachability_scoring import (
    compute_average_metric_score,
    compute_weighted_metric_score,
)

LOGGER = logging.getLogger("embodik.reachability_fk")

# Column indices for reach_map tensor (aliases for constants)
VISITATION_COL = FK_REACH_MAP_VISITATION_COL
MANIP_COL = FK_REACH_MAP_MANIP_COL
ROM_SUM_COL = FK_REACH_MAP_ROM_COL
SINGULARITY_SUM_COL = FK_REACH_MAP_SINGULARITY_COL


def _rpy_to_quaternion_numpy(rpy: np.ndarray) -> np.ndarray:
    """Convert roll-pitch-yaw (XYZ) angles to normalized quaternions."""
    if rpy.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    rpy = np.asarray(rpy, dtype=np.float64)
    if rpy.ndim == 1:
        rpy = rpy.reshape(1, 3)

    half_roll = 0.5 * rpy[:, 0]
    half_pitch = 0.5 * rpy[:, 1]
    half_yaw = 0.5 * rpy[:, 2]

    cr = np.cos(half_roll)
    sr = np.sin(half_roll)
    cp = np.cos(half_pitch)
    sp = np.sin(half_pitch)
    cy = np.cos(half_yaw)
    sy = np.sin(half_yaw)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    quat = np.stack((w, x, y, z), axis=1).astype(np.float32, copy=False)
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-8
    if np.any(valid):
        quat[valid] /= norms[valid]
    if np.any(~valid):
        quat[~valid] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat


def _torch_range_of_motion(
    samples: torch.Tensor,
    arm_mid: torch.Tensor,
    arm_span: torch.Tensor,
) -> torch.Tensor:
    """Compute range of motion metric for joint configurations."""
    normalized = (samples - arm_mid) / arm_span
    denom = (1.0 + JOINT_LIMIT_EPSILON - normalized) * (normalized + 1.0 + JOINT_LIMIT_EPSILON)
    denom = torch.clamp(denom, min=1e-6)
    raw = torch.sum((normalized * normalized) / denom, dim=1)
    return torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)


def _torch_joint_limit_distance(
    samples: torch.Tensor,
    lower_deadband: torch.Tensor,
    upper_deadband: torch.Tensor,
    deadband_delta: torch.Tensor,
) -> torch.Tensor:
    """Compute joint limit distance metric."""
    lower_diff = (samples - lower_deadband) / deadband_delta
    upper_diff = (samples - upper_deadband) / deadband_delta
    pos_normalized = torch.zeros_like(samples)
    pos_normalized = torch.where(samples < lower_deadband, lower_diff, pos_normalized)
    pos_normalized = torch.where(samples > upper_deadband, upper_diff, pos_normalized)
    pos_normalized = torch.nan_to_num(pos_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    denom = (1.0 + JOINT_LIMIT_EPSILON - pos_normalized) * (
        pos_normalized + 1.0 + JOINT_LIMIT_EPSILON
    )
    denom = torch.clamp(denom, min=1e-6)
    contrib = (pos_normalized * pos_normalized) / denom
    contrib = torch.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0)
    distance = torch.sum(contrib, dim=1)
    return torch.nan_to_num(distance, nan=0.0, posinf=0.0, neginf=0.0)


def _torch_singularity_weighted(
    singularity_raw: torch.Tensor, joint_limit_distance: torch.Tensor
) -> torch.Tensor:
    """Compute weighted singularity metric."""
    weighted = singularity_raw / (joint_limit_distance * joint_limit_distance + 1.0)
    return torch.nan_to_num(weighted, nan=0.0, posinf=0.0, neginf=0.0)


def _resolve_device(device: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(device)
    except RuntimeError as exc:
        raise ValueError(f"Invalid device specification: {device}") from exc


def _estimate_memory_bytes(
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
    z_limits: Tuple[float, float],
    roll_limits: Tuple[float, float],
    pitch_limits: Tuple[float, float],
    yaw_limits: Tuple[float, float],
    cartesian_resolution: float,
    angular_resolution: float,
    dtype: torch.dtype,
) -> int:
    """Estimate memory usage for reachability map."""
    x_bins = math.ceil((x_limits[1] - x_limits[0]) / cartesian_resolution)
    y_bins = math.ceil((y_limits[1] - y_limits[0]) / cartesian_resolution)
    z_bins = math.ceil((z_limits[1] - z_limits[0]) / cartesian_resolution)
    roll_bins = math.ceil((roll_limits[1] - roll_limits[0]) / angular_resolution)
    pitch_bins = math.ceil((pitch_limits[1] - pitch_limits[0]) / angular_resolution)
    yaw_bins = math.ceil((yaw_limits[1] - yaw_limits[0]) / angular_resolution)
    num_voxels = x_bins * y_bins * z_bins * roll_bins * pitch_bins * yaw_bins
    num_values = 10  # xyz rpy + visitation count + manipulability + ROM sum + Singularity sum
    bytes_per_val = torch.finfo(dtype).bits // 8
    return num_voxels * num_values * bytes_per_val


def generate_fk_reachability_map(
    *,
    urdf_path: Path,
    chain_joint_names: Sequence[str],
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
    save_every: int,
    post_process: bool,
    device: str,
    dtype: torch.dtype,
    # Base joint handling
    base_joint_names: Optional[Sequence[str]] = None,  # If None, auto-detect by exclusion
    # Angular limits (optional, defaults match example 03)
    roll_limits: Optional[Tuple[float, float]] = None,
    pitch_limits: Optional[Tuple[float, float]] = None,
    yaw_limits: Optional[Tuple[float, float]] = None,
    # Joint limit handling (optional, for Alpha's URDF-based limits)
    joint_limit_map: Optional[Dict[str, Tuple[float, float]]] = None,
    # Callbacks
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    # Samples file joint names (for example 03 compatibility)
    samples_joint_names: Optional[Sequence[str]] = None,
    # Optional robot metadata to embed in outputs
    robot_metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, Path]:
    """Generate FK-based reachability map for any robot configuration.

    This function consolidates the FK reachability generation logic from both
    example 03 (generic robots) and alpha_reachability_map.py (Alpha-specific).

    Args:
        urdf_path: Path to resolved URDF file (already processed)
        chain_joint_names: Names of joints to sample (arm joints)
        end_effector: End effector frame name
        output_dir: Directory for output files
        job_name: Job name for output files
        total_samples: Total number of samples to generate
        batch_size: Batch size for sampling
        cartesian_resolution: Spatial resolution in meters
        angular_resolution: Angular resolution in radians
        x_limits: X-axis limits (min, max)
        y_limits: Y-axis limits (min, max)
        z_limits: Z-axis limits (min, max)
        save_every: Save interval (in loops)
        post_process: Whether to generate filtered maps
        device: Device string ("auto", "cpu", "cuda")
        dtype: Torch dtype for computations
        base_joint_names: Optional explicit list of base joint names.
            If None, base joints are auto-detected by exclusion.
        roll_limits: Optional roll angle limits (default: (-π, π))
        pitch_limits: Optional pitch angle limits (default: (-π/2, π/2))
        yaw_limits: Optional yaw angle limits (default: (-π, π))
        joint_limit_map: Optional dict mapping joint names to (lower, upper) limits.
            If None, limits are read from robot model.
        progress_callback: Optional progress callback function
        cancel_event: Optional cancellation event
        samples_joint_names: Optional joint names for samples file.
            If None, uses chain_joint_names.
        robot_metadata: Optional dictionary describing the robot configuration.
            When provided, serialized metadata is embedded into generated HDF5
            files (both raw and filtered summaries).

    Returns:
        Dictionary with output file paths (keys: 'map', 'map_3d', 'samples',
        optionally 'map_filtered', 'map_filtered_3d')
    """
    # Validate minimum cartesian resolution (1cm)
    if cartesian_resolution < 0.01:
        LOGGER.warning(
            "cartesian_resolution (%.4f m) is below minimum (0.01 m), clamping to 0.01 m",
            cartesian_resolution
        )
        cartesian_resolution = 0.01

    # Set default angular limits if not provided (example 03 defaults)
    if roll_limits is None:
        roll_limits = (-math.pi, math.pi)
    if pitch_limits is None:
        pitch_limits = (-math.pi / 2.0, math.pi / 2.0)
    if yaw_limits is None:
        yaw_limits = (-math.pi, math.pi)

    # Load robot model and URDF XML
    robot = embodik.RobotModel(str(urdf_path), floating_base=False)

    with open(urdf_path, "r", encoding="utf-8") as f:
        urdf_xml = f.read()

    # Build kinematic chain
    chain = pk.build_serial_chain_from_urdf(urdf_xml, end_effector)
    device_torch = _resolve_device(device)
    chain = chain.to(dtype=dtype, device=device_torch)
    chain_joint_order = chain.get_joint_parameter_names()

    # Identify base and sample joints
    sample_joint_set = set(chain_joint_names)
    sample_joint_indices: List[int] = []
    base_joint_indices: List[int] = []
    ordered_sample_joints: List[str] = []

    if base_joint_names is not None:
        # Explicit base joint identification (Alpha mode)
        base_joint_set = set(base_joint_names)
        for idx, name in enumerate(chain_joint_order):
            if name in base_joint_set:
                base_joint_indices.append(idx)
            elif name in sample_joint_set:
                sample_joint_indices.append(idx)
                ordered_sample_joints.append(name)
            else:
                # Unrecognized joints treated as base (fixed zero)
                LOGGER.warning("Chain joint %s not recognized; treating as fixed zero.", name)
                base_joint_indices.append(idx)
    else:
        # Auto-detect by exclusion (example 03 mode)
        for idx, name in enumerate(chain_joint_order):
            if name in sample_joint_set:
                sample_joint_indices.append(idx)
                ordered_sample_joints.append(name)
            else:
                base_joint_indices.append(idx)

    if not sample_joint_indices:
        raise RuntimeError("Failed to locate active joints in the kinematic chain.")

    # Get joint limits
    if joint_limit_map is not None:
        # Use provided joint limit map (Alpha mode with URDF overrides)
        joint_name_to_limits = joint_limit_map.copy()
    else:
        # Use robot model limits (example 03 mode)
        joint_name_to_limits = {
            name: (float(robot.get_joint_limits()[0][idx]), float(robot.get_joint_limits()[1][idx]))
            for idx, name in enumerate(robot.get_joint_names())
        }

    # Build sampling tensors
    arm_low = torch.tensor(
        [joint_name_to_limits.get(name, (-math.pi, math.pi))[0] for name in ordered_sample_joints],
        dtype=dtype,
        device=device_torch
    )
    arm_high = torch.tensor(
        [joint_name_to_limits.get(name, (-math.pi, math.pi))[1] for name in ordered_sample_joints],
        dtype=dtype,
        device=device_torch
    )
    sampler = torch.distributions.uniform.Uniform(arm_low, arm_high)
    arm_mid = 0.5 * (arm_low + arm_high)
    arm_span = torch.clamp(0.5 * (arm_high - arm_low), min=torch.finfo(dtype).eps)
    deadband_delta = torch.clamp(
        (arm_high - arm_low) * (1.0 - DEFAULT_JOINT_LIMIT_DEADBAND_RATIO) * 0.5,
        min=torch.finfo(dtype).eps,
    )
    lower_deadband = arm_low + deadband_delta
    upper_deadband = arm_high - deadband_delta

    # Base configuration (zeros for base joints)
    base_configuration = torch.zeros(len(base_joint_indices), dtype=dtype, device=device_torch)

    # Compute grid dimensions
    total_samples_val = total_samples
    batch_size_val = min(batch_size, total_samples_val)
    num_loops = max(1, int(np.ceil(total_samples_val / batch_size_val)))
    save_every_val = max(1, save_every)

    x_bins = int(np.ceil((x_limits[1] - x_limits[0]) / cartesian_resolution))
    y_bins = int(np.ceil((y_limits[1] - y_limits[0]) / cartesian_resolution))
    z_bins = int(np.ceil((z_limits[1] - z_limits[0]) / cartesian_resolution))
    roll_bins = int(np.ceil((roll_limits[1] - roll_limits[0]) / angular_resolution))
    pitch_bins = int(np.ceil((pitch_limits[1] - pitch_limits[0]) / angular_resolution))
    yaw_bins = int(np.ceil((yaw_limits[1] - yaw_limits[0]) / angular_resolution))

    num_voxels = x_bins * y_bins * z_bins * roll_bins * pitch_bins * yaw_bins
    reach_map = torch.zeros((num_voxels, 10), dtype=dtype, device="cpu")

    # Compute index offsets for 6D grid flattening
    yaw_offset = 1
    pitch_offset = yaw_bins * yaw_offset
    roll_offset = pitch_bins * pitch_offset
    z_offset = roll_bins * roll_offset
    y_offset = z_bins * z_offset
    x_offset = y_bins * y_offset
    index_offsets = torch.tensor(
        [x_offset, y_offset, z_offset, roll_offset, pitch_offset, yaw_offset],
        dtype=torch.long,
        device=device_torch
    )

    # Lower bounds and resolutions
    lower_bounds = torch.tensor(
        [
            x_limits[0],
            y_limits[0],
            z_limits[0],
            roll_limits[0],
            pitch_limits[0],
            yaw_limits[0],
        ],
        dtype=dtype,
        device=device_torch,
    )
    resolutions = torch.tensor(
        [
            cartesian_resolution,
            cartesian_resolution,
            cartesian_resolution,
            angular_resolution,
            angular_resolution,
            angular_resolution,
        ],
        dtype=dtype,
        device=device_torch,
    )
    centers = lower_bounds + 0.5 * resolutions
    max_bins = torch.tensor(
        [x_bins, y_bins, z_bins, roll_bins, pitch_bins, yaw_bins],
        dtype=torch.long,
        device=device_torch
    )

    # Logging and device info
    if device_torch.type == "cuda":
        torch.cuda.empty_cache()
        LOGGER.info(
            "Using CUDA device %s (memory %.1f GiB free)",
            torch.cuda.get_device_name(device_torch.index or 0),
            (torch.cuda.get_device_properties(device_torch.index or 0).total_memory - torch.cuda.memory_reserved(device_torch.index or 0))
            / 1024**3,
        )
    else:
        LOGGER.info("Using CPU for sampling; consider --device cuda for faster runs.")

    est_bytes = _estimate_memory_bytes(
        x_limits, y_limits, z_limits,
        roll_limits, pitch_limits, yaw_limits,
        cartesian_resolution, angular_resolution, dtype
    )
    LOGGER.info(
        "6D grid: (%d x %d x %d) positions, (%d x %d x %d) orientations -> %d voxels (~%.2f MiB)",
        x_bins, y_bins, z_bins, roll_bins, pitch_bins, yaw_bins, num_voxels, est_bytes / (1024**2),
    )
    LOGGER.info(
        "Reachability job %s | samples=%d | batch=%d | cartesian=%.3fm | angular=%.1fdeg | device=%s",
        job_name, total_samples_val, batch_size_val, cartesian_resolution,
        math.degrees(angular_resolution), device_torch.type,
    )
    LOGGER.info("Keeping %d base joints fixed, sampling %d arm joints.", len(base_joint_indices), len(sample_joint_indices))

    # Initialize tracking variables
    start_time = time.perf_counter()
    samples_processed = 0
    sample_positions: List[np.ndarray] = []
    sample_orientations_rpy: List[np.ndarray] = []
    sample_orientations_quat: List[np.ndarray] = []
    sample_bucket_indices: List[np.ndarray] = []
    sample_configs: List[np.ndarray] = []

    range_of_motion_raw_max = 0.0
    singularity_weighted_max = 0.0
    manipulability_raw_max = 0.0

    if progress_callback:
        progress_callback(0.0, "starting")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}

    cancelled = False
    report_interval = max(1, num_loops // 10)

    # Main sampling loop
    for loop_idx in range(num_loops):
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            LOGGER.warning("Cancellation requested; aborting reachability generation.")
            break

        if progress_callback:
            progress_callback(samples_processed / max(1, total_samples_val), "sampling")

        current_batch = min(batch_size_val, total_samples_val - samples_processed)
        arm_samples = sampler.sample((current_batch,))
        full_batch = torch.zeros((current_batch, len(chain_joint_order)), dtype=dtype, device=device_torch)
        if base_joint_indices:
            full_batch[:, base_joint_indices] = base_configuration
        full_batch[:, sample_joint_indices] = arm_samples

        # Forward kinematics
        fk_mats = chain.forward_kinematics(full_batch).get_matrix()
        poses_xyz = fk_mats[:, :3, 3]
        poses_rpy = pk.transforms.matrix_to_euler_angles(fk_mats[:, :3, :3], "XYZ")
        poses_quat = pk.transforms.matrix_to_quaternion(fk_mats[:, :3, :3])
        poses_6d = torch.hstack((poses_xyz, poses_rpy))

        # Discretize poses
        indices_6d = torch.floor((poses_6d - lower_bounds) / resolutions)
        indices_6d = torch.clamp(indices_6d, min=0)
        overflow_mask = indices_6d >= max_bins.to(dtype=dtype)
        if torch.any(overflow_mask):
            LOGGER.debug("Detected discretisation overflow in loop %d; clamping to max bins.", loop_idx)
            indices_6d = torch.minimum(indices_6d, (max_bins - 1).to(dtype=dtype))

        discretised_pose = indices_6d * resolutions + centers
        indices_6d_long = indices_6d.to(dtype=torch.long)
        flat_indices = torch.sum(indices_6d_long * index_offsets, dim=1).to(device="cpu", dtype=torch.long)

        # Store samples
        poses_xyz_cpu = poses_xyz.to(device="cpu").numpy()
        poses_rpy_cpu = poses_rpy.to(device="cpu").numpy()
        poses_quat_cpu = poses_quat.to(device="cpu").numpy()
        sample_positions.append(poses_xyz_cpu)
        sample_orientations_rpy.append(poses_rpy_cpu)
        sample_orientations_quat.append(poses_quat_cpu)
        sample_bucket_indices.append(indices_6d_long[:, :3].to(device="cpu").numpy())
        sample_configs.append(arm_samples.to(device="cpu").numpy())

        # Compute metrics
        jacobian = chain.jacobian(full_batch)
        jacobian_arm = jacobian[:, :, sample_joint_indices]
        jj_t = torch.matmul(jacobian_arm, torch.transpose(jacobian_arm, 1, 2))
        manip_det = torch.det(jj_t)
        manip_det = torch.nan_to_num(manip_det, nan=0.0, posinf=0.0, neginf=0.0)
        if manip_det.numel() > 0:
            manipulability_raw_max = max(manipulability_raw_max, float(manip_det.max().item()))
        singularity_raw = torch.sqrt(torch.clamp(manip_det, min=0.0))
        singularity_raw = torch.nan_to_num(singularity_raw, nan=0.0, posinf=0.0, neginf=0.0)

        rom_metric = _torch_range_of_motion(arm_samples, arm_mid, arm_span)
        if rom_metric.numel() > 0:
            range_of_motion_raw_max = max(range_of_motion_raw_max, float(rom_metric.max().item()))

        limit_distance = _torch_joint_limit_distance(
            arm_samples, lower_deadband, upper_deadband, deadband_delta
        )
        sing_weighted = _torch_singularity_weighted(singularity_raw, limit_distance)
        if sing_weighted.numel() > 0:
            singularity_weighted_max = max(
                singularity_weighted_max, float(sing_weighted.max().cpu().item())
            )

        # Accumulate in reach map
        discretised_pose_cpu = discretised_pose.to(device="cpu")
        reach_map[flat_indices, :6] = discretised_pose_cpu
        reach_map[flat_indices, VISITATION_COL] += 1.0
        reach_map[flat_indices, MANIP_COL] = torch.maximum(
            reach_map[flat_indices, MANIP_COL], manip_det.to(device="cpu")
        )
        reach_map[flat_indices, ROM_SUM_COL] += rom_metric.to(device="cpu")
        reach_map[flat_indices, SINGULARITY_SUM_COL] += sing_weighted.to(device="cpu")

        samples_processed += current_batch

        # Progress reporting
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "Loop %d/%d | batch %d | cumulative %d/%d samples",
                loop_idx + 1, num_loops, current_batch, samples_processed, total_samples_val,
            )
        if (loop_idx + 1) % report_interval == 0 or loop_idx + 1 == num_loops:
            percent = samples_processed / max(1, total_samples_val)
            LOGGER.info(
                "Sampling progress: loop %d/%d (%.1f%%) with %d/%d samples",
                loop_idx + 1, num_loops, percent * 100.0, samples_processed, total_samples_val,
            )

        # Periodic saves
        if (loop_idx + 1) % save_every_val == 0 or loop_idx + 1 == num_loops:
            if range_of_motion_raw_max <= 0.0:
                range_of_motion_raw_max = 1.0
            if singularity_weighted_max <= 0.0:
                singularity_weighted_max = 1.0
            if manipulability_raw_max <= 0.0:
                manipulability_raw_max = 1.0

            nonzero_rows = torch.sum(torch.abs(reach_map), dim=1) > 0
            reach_map_np = reach_map[nonzero_rows].numpy()
            map_path = output_dir / f"reach_map_{job_name}.pkl"
            with open(map_path, "wb") as handle:
                pickle.dump(reach_map_np, handle)

            # Aggregate 3D scores
            z_block = z_bins * roll_bins * pitch_bins * yaw_bins
            sphere_rows: List[np.ndarray] = []
            pose_rows: List[np.ndarray] = []
            idx = 0
            while idx < reach_map_np.shape[0]:
                origin = reach_map_np[idx, :3]
                slice_end = min(idx + z_block, reach_map_np.shape[0])
                slice_rows = reach_map_np[idx:slice_end]
                match = np.all(slice_rows[:, :3] == origin, axis=1)
                reps = int(np.count_nonzero(match))
                if reps == 0:
                    idx += 1
                    continue
                leaf = slice_rows[:reps]

                # Average metrics over repetitions
                manipulability_score = float(leaf[:, MANIP_COL].mean())
                visitation_total = float(leaf[:, VISITATION_COL].sum())
                rom_total = float(leaf[:, ROM_SUM_COL].sum())
                sing_total = float(leaf[:, SINGULARITY_SUM_COL].sum())
                if visitation_total > 0.0:
                    rom_avg_raw = rom_total / visitation_total
                    sing_avg_norm = sing_total / visitation_total
                else:
                    rom_avg_raw = 0.0
                    sing_avg_norm = 0.0

                rom_norm = normalize_range_of_motion_scalar(rom_avg_raw, range_of_motion_raw_max)
                sing_norm = normalize_singularity_scalar(sing_avg_norm, singularity_weighted_max)

                sphere_rows.append(
                    np.array(
                        [origin[0], origin[1], origin[2], manipulability_score, visitation_total, rom_norm, sing_norm],
                        dtype=np.float32,
                    )
                )
                quat_row = _rpy_to_quaternion_numpy(leaf[0, 3:6])
                pose_rows.append(np.concatenate([leaf[0, :6], quat_row.reshape(-1)]))
                idx += reps

    def _attach_robot_metadata(dataset) -> None:
        if not robot_metadata:
            return

        def _metadata_default(obj: Any) -> Any:
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            if isinstance(obj, (set, tuple)):
                return list(obj)
            return str(obj)

        try:
            metadata_json = json.dumps(robot_metadata, default=_metadata_default)
        except (TypeError, ValueError) as exc:
            LOGGER.warning(
                "Failed to serialize robot metadata for FK reachability '%s': %s",
                job_name,
                exc,
            )
            return

        str_dtype = h5py.string_dtype(encoding="utf-8")
        if "RobotMetadata" in dataset.attrs:
            del dataset.attrs["RobotMetadata"]
        dataset.attrs.create("RobotMetadata", metadata_json, dtype=str_dtype)

    # Save HDF5 file
    centers_cpu = centers[:3].to(device="cpu").numpy().astype(np.float32)
    lower_cpu = lower_bounds[:3].to(device="cpu").numpy().astype(np.float32)
    h5_path = output_dir / f"3D_{job_name}.h5"
    with h5py.File(h5_path, "w") as handle:
        sphere_group = handle.create_group("/Spheres")
        if sphere_rows:
            dataset = sphere_group.create_dataset("sphere_dataset", data=np.vstack(sphere_rows))
        else:
            dataset = sphere_group.create_dataset("sphere_dataset", data=np.zeros((0, 7), dtype=np.float32))
        dataset.attrs.create("Resolution", data=cartesian_resolution)
        dataset.attrs.create("CenterOffset", data=centers_cpu)
        dataset.attrs.create("LowerBounds", data=lower_cpu)
        dataset.attrs.create(
            "MetricNames",
            np.asarray(["Manipulability", "RangeOfMotion", "SingularityAvoidance"], dtype="S32"),
        )
        dataset.attrs.create(
            "MetricColumns",
            np.asarray(
                [SPHERE_DATASET_MANIP_COL, SPHERE_DATASET_ROM_COL, SPHERE_DATASET_SINGULARITY_COL],
                dtype=np.int32,
            ),
        )
        dataset.attrs.create("VisitationColumn", np.int32(SPHERE_DATASET_VISITATION_COL))
        dataset.attrs.create("ManipulabilityScaling", data=1.0)
        dataset.attrs.create("ManipulabilityRawMax", manipulability_raw_max)
        dataset.attrs.create("RangeOfMotionRawMin", 0.0)
        dataset.attrs.create("RangeOfMotionRawMax", range_of_motion_raw_max)
        dataset.attrs.create("SingularityWeightedMax", singularity_weighted_max)
        _attach_robot_metadata(dataset)

        pose_group = handle.create_group("/Poses")
        if pose_rows:
            pose_group.create_dataset("poses_dataset", data=np.vstack(pose_rows))
        else:
            pose_group.create_dataset("poses_dataset", data=np.zeros((0, 10), dtype=np.float32))

            # Compute global scores
            visited_mask = reach_map_np[:, VISITATION_COL] > 0
            if np.any(visited_mask):
                visited_data = reach_map_np[visited_mask]
                visitation = visited_data[:, VISITATION_COL]
                manip_values = visited_data[:, MANIP_COL]

                rom_sum = visited_data[:, ROM_SUM_COL]
                rom_avg_raw = np.where(visitation > 0, rom_sum / visitation, 0.0)
                rom_norm_array = np.clip(
                    (rom_avg_raw - 0.0) / max(range_of_motion_raw_max - 0.0, 1e-6),
                    0.0,
                    RANGE_OF_MOTION_NORMALIZED_LIMIT,
                )

                sing_sum = visited_data[:, SINGULARITY_SUM_COL]
                sing_avg_norm = np.where(visitation > 0, sing_sum / visitation, 0.0)
                sing_norm_array = np.clip(
                    sing_avg_norm / max(singularity_weighted_max, 1e-6),
                    0.0,
                    1.0,
                )

                global_scores = {
                    "Manipulability": {
                        **compute_average_metric_score(manip_values, "Manipulability"),
                        **compute_weighted_metric_score(manip_values, visitation, "Manipulability"),
                    },
                    "RangeOfMotion": {
                        **compute_average_metric_score(rom_norm_array, "RangeOfMotion"),
                        **compute_weighted_metric_score(rom_norm_array, visitation, "RangeOfMotion"),
                    },
                    "SingularityAvoidance": {
                        **compute_average_metric_score(sing_norm_array, "SingularityAvoidance"),
                        **compute_weighted_metric_score(sing_norm_array, visitation, "SingularityAvoidance"),
                    },
                }

                # Store global scores in HDF5
                scores_json = {}
                for metric_name, scores in global_scores.items():
                    scores_json[metric_name] = {
                        k: float(v) if isinstance(v, (np.integer, np.floating)) else int(v) if isinstance(v, np.integer) else v
                        for k, v in scores.items()
                    }
                json_str = json.dumps(scores_json)
                with h5py.File(h5_path, "r+") as handle:
                    dataset = handle["/Spheres/sphere_dataset"]
                    str_dtype = h5py.string_dtype(encoding='utf-8')
                    dataset.attrs.create("GlobalScores", json_str, dtype=str_dtype)
                LOGGER.info("Computed global scores: %s", ", ".join(f"{k}: {v.get('average', 0):.3f}" for k, v in global_scores.items()))

            outputs["map"] = map_path
            outputs["map_3d"] = h5_path
            LOGGER.info("Saved reachability snapshot: %s, %s", map_path.name, h5_path.name)

            if progress_callback:
                progress_callback(samples_processed / max(1, total_samples_val), "writing")

    # Save samples file
    if sample_positions and sample_configs:
        samples_xyz = np.concatenate(sample_positions, axis=0).astype(np.float32)
        samples_rpy = (
            np.concatenate(sample_orientations_rpy, axis=0).astype(np.float32)
            if sample_orientations_rpy
            else np.zeros((0, 3), dtype=np.float32)
        )
        samples_quat = (
            np.concatenate(sample_orientations_quat, axis=0).astype(np.float32)
            if sample_orientations_quat
            else np.zeros((0, 4), dtype=np.float32)
        )
        samples_ijk = np.concatenate(sample_bucket_indices, axis=0).astype(np.int32)
        samples_q = np.concatenate(sample_configs, axis=0).astype(np.float32)
        samples_path = output_dir / f"samples_{job_name}.npz"

        # Use samples_joint_names if provided, otherwise use ordered_sample_joints
        joint_names_for_samples = samples_joint_names if samples_joint_names is not None else ordered_sample_joints
        joint_names_array = np.asarray(joint_names_for_samples, dtype=np.dtype("U64"))
        np.savez_compressed(
            samples_path,
            xyz=samples_xyz,
            rpy=samples_rpy,
            quat=samples_quat,
            ijk=samples_ijk,
            q=samples_q,
            joint_names=joint_names_array,
        )
        outputs["samples"] = samples_path
        LOGGER.info("Stored sampled configurations: %s", samples_path.name)

    # Post-processing (filtered maps)
    if post_process and "map" in outputs:
        with open(outputs["map"], "rb") as handle:
            reach_map_np = pickle.load(handle)
        reach_map_np = np.asarray(reach_map_np)

        filtered_mask = reach_map_np[:, 2] > 0.0
        filtered = reach_map_np[filtered_mask]
        filt_path = output_dir / f"filt_{job_name}.pkl"
        with open(filt_path, "wb") as handle:
            pickle.dump(filtered, handle)
        outputs["map_filtered"] = filt_path
        LOGGER.info("Filtered map saved: %s", filt_path.name)

        if filtered.size > 0:
            z_block = z_bins * roll_bins * pitch_bins * yaw_bins
            sphere_rows: List[np.ndarray] = []
            pose_rows: List[np.ndarray] = []
            idx = 0
            while idx < filtered.shape[0]:
                origin = filtered[idx, :3]
                slice_end = min(idx + z_block, filtered.shape[0])
                slice_rows = filtered[idx:slice_end]
                match = np.all(slice_rows[:, :3] == origin, axis=1)
                reps = int(np.count_nonzero(match))
                if reps == 0:
                    idx += 1
                    continue
                leaf = slice_rows[:reps]

                manipulability_score = float(leaf[:, MANIP_COL].mean())
                visitation_total = float(leaf[:, VISITATION_COL].sum())
                rom_total = float(leaf[:, ROM_SUM_COL].sum())
                sing_total = float(leaf[:, SINGULARITY_SUM_COL].sum())
                if visitation_total > 0.0:
                    rom_avg_raw = rom_total / visitation_total
                    sing_avg_norm = sing_total / visitation_total
                else:
                    rom_avg_raw = 0.0
                    sing_avg_norm = 0.0

                rom_norm = normalize_range_of_motion_scalar(rom_avg_raw, range_of_motion_raw_max)
                sing_norm = normalize_singularity_scalar(sing_avg_norm, singularity_weighted_max)

                sphere_rows.append(
                    np.array(
                        [origin[0], origin[1], origin[2], manipulability_score, visitation_total, rom_norm, sing_norm],
                        dtype=np.float32,
                    )
                )
                quat_row = _rpy_to_quaternion_numpy(leaf[0, 3:6])
                pose_rows.append(np.concatenate([leaf[0, :6], quat_row.reshape(-1)]))
                idx += reps

            centers_cpu = centers[:3].to(device="cpu").numpy().astype(np.float32)
            lower_cpu = lower_bounds[:3].to(device="cpu").numpy().astype(np.float32)
            filt_h5_path = output_dir / f"filt_3D_{job_name}.h5"
            with h5py.File(filt_h5_path, "w") as handle:
                sphere_group = handle.create_group("/Spheres")
                if sphere_rows:
                    dataset = sphere_group.create_dataset("sphere_dataset", data=np.vstack(sphere_rows))
                else:
                    dataset = sphere_group.create_dataset("sphere_dataset", data=np.zeros((0, 7), dtype=np.float32))
                dataset.attrs.create("Resolution", data=cartesian_resolution)
                dataset.attrs.create("CenterOffset", data=centers_cpu)
                dataset.attrs.create("LowerBounds", data=lower_cpu)
                dataset.attrs.create(
                    "MetricNames",
                    np.asarray(["Manipulability", "RangeOfMotion", "SingularityAvoidance"], dtype="S32"),
                )
                dataset.attrs.create(
                    "MetricColumns",
                    np.asarray(
                        [SPHERE_DATASET_MANIP_COL, SPHERE_DATASET_ROM_COL, SPHERE_DATASET_SINGULARITY_COL],
                        dtype=np.int32,
                    ),
                )
                dataset.attrs.create("VisitationColumn", np.int32(SPHERE_DATASET_VISITATION_COL))
                dataset.attrs.create("ManipulabilityScaling", data=1.0)
                dataset.attrs.create("ManipulabilityRawMax", manipulability_raw_max)
                dataset.attrs.create("RangeOfMotionRawMin", 0.0)
                dataset.attrs.create("RangeOfMotionRawMax", range_of_motion_raw_max)
                dataset.attrs.create("SingularityWeightedMax", singularity_weighted_max)
                _attach_robot_metadata(dataset)

                pose_group = handle.create_group("/Poses")
                if pose_rows:
                    pose_group.create_dataset("poses_dataset", data=np.vstack(pose_rows))
                else:
                    pose_group.create_dataset("poses_dataset", data=np.zeros((0, 10), dtype=np.float32))
            outputs["map_filtered_3d"] = filt_h5_path
            LOGGER.info("Filtered 3D summary saved: %s", filt_h5_path.name)

        if progress_callback:
            progress_callback(samples_processed / max(1, total_samples_val), "post_process")

    elapsed = time.perf_counter() - start_time
    if cancelled:
        LOGGER.info("Reachability generation cancelled after %.2f s", elapsed)
        if progress_callback:
            progress_callback(samples_processed / max(1, total_samples_val), "cancelled")
    else:
        LOGGER.info("Reachability generation completed in %.2f s", elapsed)
        if progress_callback:
            progress_callback(1.0, "done")

    return outputs
