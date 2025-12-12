"""Shared helper classes and functions for reachability metric computation and display.

This module provides reusable components for computing and displaying reachability
metrics (manipulability, range of motion, singularity avoidance, etc.) that can be
used across different robot examples.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .reachability_constants import (
    DEFAULT_JOINT_LIMIT_DEADBAND_RATIO,
    JOINT_LIMIT_DISTANCE_NORMALIZED_LIMIT,
    JOINT_LIMIT_EPSILON,
    NORMALIZATION_CACHE_DIRNAME,
    NORMALIZATION_KEYS,
    RANGE_OF_MOTION_NORMALIZED_LIMIT,
)

LOGGER = logging.getLogger("embodik.reachability_metrics")

__all__ = [
    "ReachabilityMetricHelper",
    "normalization_cache_path",
    "load_normalization_cache",
    "save_normalization_cache",
    "range_of_motion_numpy",
    "joint_limit_distance_numpy",
    "singularity_weighted_numpy",
    "normalize_range_of_motion_scalar",
    "normalize_singularity_scalar",
    "normalize_joint_limit_distance_scalar",
    "normalize_manipulability_scalar",
]


def range_of_motion_numpy(
    samples: np.ndarray, arm_mid: np.ndarray, arm_span: np.ndarray
) -> np.ndarray:
    """Compute range of motion metric for joint configurations.

    Args:
        samples: Joint configurations (N x DOF or 1D array)
        arm_mid: Midpoint of joint ranges
        arm_span: Half-span of joint ranges

    Returns:
        Range of motion scores (higher is better)
    """
    samples = np.asarray(samples, dtype=np.float64)
    squeeze = False
    if samples.ndim == 1:
        samples = samples[np.newaxis, :]
        squeeze = True
    normalized = (samples - arm_mid[np.newaxis, :]) / arm_span[np.newaxis, :]
    denom = (1.0 + JOINT_LIMIT_EPSILON - normalized) * (normalized + 1.0 + JOINT_LIMIT_EPSILON)
    denom = np.clip(denom, 1e-6, None)
    raw = np.sum((normalized * normalized) / denom, axis=1)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return raw[0] if squeeze else raw


def joint_limit_distance_numpy(
    samples: np.ndarray,
    lower_deadband: np.ndarray,
    upper_deadband: np.ndarray,
    deadband_delta: np.ndarray,
) -> np.ndarray:
    """Compute joint limit distance metric.

    Args:
        samples: Joint configurations (N x DOF or 1D array)
        lower_deadband: Lower deadband boundaries
        upper_deadband: Upper deadband boundaries
        deadband_delta: Deadband delta values

    Returns:
        Joint limit distance scores (lower is better, 0 when within deadband)
    """
    samples = np.asarray(samples, dtype=np.float64)
    squeeze = False
    if samples.ndim == 1:
        samples = samples[np.newaxis, :]
        squeeze = True
    lower_db = lower_deadband[np.newaxis, :]
    upper_db = upper_deadband[np.newaxis, :]
    deadband = np.clip(deadband_delta[np.newaxis, :], 1e-6, None)
    pos_normalized = np.where(samples < lower_db, (samples - lower_db) / deadband, 0.0)
    pos_normalized = np.where(samples > upper_db, (samples - upper_db) / deadband, pos_normalized)
    denom = (1.0 + JOINT_LIMIT_EPSILON - pos_normalized) * (
        pos_normalized + 1.0 + JOINT_LIMIT_EPSILON
    )
    denom = np.clip(denom, 1e-6, None)
    distance = np.sum((pos_normalized * pos_normalized) / denom, axis=1)
    distance = np.nan_to_num(distance, nan=0.0, posinf=0.0, neginf=0.0)
    return distance[0] if squeeze else distance


def singularity_weighted_numpy(
    manip_raw: np.ndarray, joint_limit_distance: np.ndarray
) -> np.ndarray:
    """Compute weighted singularity metric.

    Args:
        manip_raw: Raw manipulability values
        joint_limit_distance: Joint limit distance values

    Returns:
        Weighted singularity scores (higher is better)
    """
    weighted = manip_raw / (joint_limit_distance * joint_limit_distance + 1.0)
    return np.nan_to_num(weighted, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_manipulability_scalar(raw_value: float, max_value: float) -> float:
    """Normalize manipulability raw value to [0, 1] range.

    Args:
        raw_value: Raw manipulability value
        max_value: Maximum value for normalization

    Returns:
        Normalized value in [0, 1] (higher is better)
    """
    max_value = max(max_value, 1e-6)
    raw_value = max(raw_value, 0.0)  # Manipulability is always >= 0
    return min(raw_value / max_value, 1.0)


def normalize_range_of_motion_scalar(raw_value: float, max_value: float) -> float:
    """Normalize range of motion raw value to [0, 1] range.

    Args:
        raw_value: Raw range of motion value
        max_value: Maximum value for normalization

    Returns:
        Normalized value in [0, 1] (higher is better)
    """
    max_value = max(max_value, 1e-6)
    raw_value = min(max(raw_value, 0.0), max_value)
    return max(0.0, 1.0 - raw_value / max_value)


def normalize_singularity_scalar(weighted_value: float, max_value: float) -> float:
    """Normalize singularity weighted value to [0, 1] range.

    Args:
        weighted_value: Weighted singularity value
        max_value: Maximum value for normalization

    Returns:
        Normalized value in [0, 1] (higher is better)
    """
    max_value = max(max_value, 1e-6)
    weighted_value = min(max(weighted_value, 0.0), max_value)
    return weighted_value / max_value


def normalize_joint_limit_distance_scalar(
    raw_value: float, max_value: float = JOINT_LIMIT_DISTANCE_NORMALIZED_LIMIT
) -> float:
    """Normalize joint limit distance value to [0, 1] range.

    Args:
        raw_value: Raw joint limit distance value
        max_value: Maximum value for normalization

    Returns:
        Normalized value in [0, 1] (higher is better)
    """
    max_value = max(max_value, 1e-6)
    raw_value = min(max(raw_value, 0.0), max_value)
    return max(0.0, 1.0 - raw_value / max_value)


def normalization_cache_path(base: Path, robot_key: str) -> Path:
    """Get path to normalization cache file.

    Args:
        base: Base directory for cache
        robot_key: Robot identifier

    Returns:
        Path to cache file
    """
    return base / NORMALIZATION_CACHE_DIRNAME / f"{robot_key}.json"


def load_normalization_cache(path: Path) -> Dict[str, float]:
    """Load normalization cache from file.

    Args:
        path: Path to cache file

    Returns:
        Dictionary of normalization values
    """
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, dict):
        return {}
    result: Dict[str, float] = {}
    for key in NORMALIZATION_KEYS:
        value = payload.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            result[key] = float(value)
    return result


def save_normalization_cache(path: Path, values: Dict[str, float]) -> None:
    """Save normalization cache to file.

    Args:
        path: Path to cache file
        values: Dictionary of normalization values to save
    """
    payload = {key: float(values[key]) for key in NORMALIZATION_KEYS if key in values}
    if not payload:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    except OSError:
        LOGGER.warning("Failed to write normalization cache: %s", path)


class ReachabilityMetricHelper:
    """Helper class for computing and displaying reachability metrics.

    This class encapsulates the computation of manipulability, range of motion,
    singularity avoidance, and other metrics used in reachability analysis.
    """

    def __init__(
        self,
        robot: object,
        arm_joint_indices: List[int],
        arm_lower_bounds: np.ndarray,
        arm_upper_bounds: np.ndarray,
        metric_chain: Optional[object] = None,
        manip_chain_indices: Optional[List[Optional[int]]] = None,
        metric_metadata: Optional[Dict[str, float]] = None,
        manip_scaling: float = 500.0,
        chain_joint_order: Optional[List[str]] = None,
        metric_sample_joint_indices: Optional[List[int]] = None,
        metric_chain_device: Optional[object] = None,
        metric_chain_dtype: Optional[object] = None,
    ):
        """Initialize metric helper.

        Args:
            robot: Robot model instance
            arm_joint_indices: Indices of arm joints in robot configuration
            arm_lower_bounds: Lower joint limits for arm joints
            arm_upper_bounds: Upper joint limits for arm joints
            metric_chain: Optional PyTorch kinematic chain for manipulability computation
            manip_chain_indices: Optional mapping from chain joints to robot joints
            metric_metadata: Optional dictionary of metric metadata (normalization values)
            manip_scaling: Scaling factor for manipulability metric
            chain_joint_order: Optional list of joint names in kinematic chain order
            metric_sample_joint_indices: Optional indices of sample joints in chain
            metric_chain_device: Optional PyTorch device for metric chain
            metric_chain_dtype: Optional PyTorch dtype for metric chain
        """
        self.robot = robot
        self.arm_joint_indices = arm_joint_indices
        self.arm_lower_bounds = arm_lower_bounds
        self.arm_upper_bounds = arm_upper_bounds
        self.metric_chain = metric_chain
        self.manip_chain_indices = manip_chain_indices
        self.metric_metadata = metric_metadata or {}
        self.manip_scaling = manip_scaling
        self.chain_joint_order = chain_joint_order or []
        self.metric_sample_joint_indices = metric_sample_joint_indices or []
        self.metric_chain_device = metric_chain_device or (torch.device("cpu") if torch else None)
        self.metric_chain_dtype = metric_chain_dtype or (torch.float32 if torch else None)

        # Precompute deadband values
        arm_span = np.clip(0.5 * (arm_upper_bounds - arm_lower_bounds), 1e-6, None)
        self.arm_mid = 0.5 * (arm_lower_bounds + arm_upper_bounds)
        self.arm_span = arm_span
        self.deadband_delta = np.clip(
            (arm_upper_bounds - arm_lower_bounds) * (1.0 - DEFAULT_JOINT_LIMIT_DEADBAND_RATIO) * 0.5,
            1e-6,
            None,
        )
        self.lower_deadband = arm_lower_bounds + self.deadband_delta
        self.upper_deadband = arm_upper_bounds - self.deadband_delta

    def estimate_metric_ranges(self, sample_count: int) -> Tuple[float, float, float]:
        """Estimate normalization ranges by sampling random configurations.

        Args:
            sample_count: Number of random samples to use

        Returns:
            Tuple of (manip_max, rom_max, sing_max) normalization values
        """
        if sample_count <= 0:
            return (
                float(self.metric_metadata.get("ManipulabilityRawMax", 1.0)),
                float(
                    self.metric_metadata.get(
                        "RangeOfMotionRawMax", RANGE_OF_MOTION_NORMALIZED_LIMIT
                    )
                ),
                float(self.metric_metadata.get("SingularityWeightedMax", 1.0)),
            )

        if torch is None or self.metric_chain is None:
            # Fallback: use defaults if PyTorch not available
            return (
                1.0,
                RANGE_OF_MOTION_NORMALIZED_LIMIT,
                1.0,
            )

        rng = np.random.default_rng()
        total = 0
        manip_max = 0.0
        rom_max = 0.0
        sing_max = 0.0
        chunk = min(sample_count, 256)

        # Determine which joints to sample: use metric_sample_joint_indices if available,
        # otherwise use all arm joints
        if self.metric_sample_joint_indices and len(self.metric_sample_joint_indices) > 0:
            # Sample only the metric sample joints
            sample_joint_count = len(self.metric_sample_joint_indices)
            # Extract bounds for metric sample joints
            # metric_sample_joint_indices are indices into chain_joint_order
            # We need to map them to arm_joint_indices to get the correct bounds
            # For simplicity, assume metric_sample_joint_indices correspond to first N arm joints
            # If the mapping is different, we'll need to handle it properly
            if sample_joint_count <= len(self.arm_lower_bounds):
                sample_lower_bounds = self.arm_lower_bounds[:sample_joint_count]
                sample_upper_bounds = self.arm_upper_bounds[:sample_joint_count]
                sample_mid = self.arm_mid[:sample_joint_count]
                sample_span = self.arm_span[:sample_joint_count]
                sample_lower_deadband = self.lower_deadband[:sample_joint_count]
                sample_upper_deadband = self.upper_deadband[:sample_joint_count]
                sample_deadband_delta = self.deadband_delta[:sample_joint_count]
            else:
                # Fallback: use all arm bounds
                sample_joint_count = self.arm_lower_bounds.shape[0]
                sample_lower_bounds = self.arm_lower_bounds
                sample_upper_bounds = self.arm_upper_bounds
                sample_mid = self.arm_mid
                sample_span = self.arm_span
                sample_lower_deadband = self.lower_deadband
                sample_upper_deadband = self.upper_deadband
                sample_deadband_delta = self.deadband_delta
        else:
            # Use all arm joints
            sample_joint_count = self.arm_lower_bounds.shape[0]
            sample_lower_bounds = self.arm_lower_bounds
            sample_upper_bounds = self.arm_upper_bounds
            sample_mid = self.arm_mid
            sample_span = self.arm_span
            sample_lower_deadband = self.lower_deadband
            sample_upper_deadband = self.upper_deadband
            sample_deadband_delta = self.deadband_delta

        while total < sample_count:
            current = min(chunk, sample_count - total)
            samples = rng.uniform(
                sample_lower_bounds,
                sample_upper_bounds,
                size=(current, sample_joint_count),
            )
            full_batch = np.zeros((current, len(self.chain_joint_order)), dtype=np.float32)
            if self.metric_sample_joint_indices and len(self.metric_sample_joint_indices) > 0:
                full_batch[:, self.metric_sample_joint_indices] = samples.astype(np.float32)
            else:
                # If no metric_sample_joint_indices, assume samples correspond to first N joints in chain
                num_sample_joints = min(sample_joint_count, len(self.chain_joint_order))
                full_batch[:, :num_sample_joints] = samples[:, :num_sample_joints].astype(np.float32)
            torch_batch = torch.from_numpy(full_batch).to(
                device=self.metric_chain_device, dtype=self.metric_chain_dtype
            )
            with torch.no_grad():
                jacobian = self.metric_chain.jacobian(torch_batch)
                jj_t = jacobian @ torch.transpose(jacobian, 1, 2)
                det = torch.det(jj_t).clamp(min=0.0)
                manip_raw = torch.sqrt(det).cpu().numpy()
            rom_raw = range_of_motion_numpy(samples, sample_mid, sample_span)
            limit_distance = joint_limit_distance_numpy(
                samples, sample_lower_deadband, sample_upper_deadband, sample_deadband_delta
            )
            sing_weighted = singularity_weighted_numpy(manip_raw, limit_distance)
            if np.size(manip_raw):
                manip_max = max(manip_max, float(np.max(manip_raw)))
            if np.size(rom_raw):
                rom_max = max(rom_max, float(np.max(rom_raw)))
            if np.size(sing_weighted):
                sing_max = max(sing_max, float(np.max(sing_weighted)))
            total += current
        manip_max = max(1e-6, manip_max)
        rom_max = max(RANGE_OF_MOTION_NORMALIZED_LIMIT, rom_max)
        sing_max = max(1e-6, sing_max)
        return manip_max, rom_max, sing_max

    def compute_metrics(
        self,
        q_current: np.ndarray,
        current_metric_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        viewer: Optional[object] = None,
    ) -> Dict[str, float]:
        """Compute all metrics for current configuration.

        Args:
            q_current: Current joint configuration
            current_metric_ranges: Optional metric ranges for normalization
            viewer: Optional reachability viewer for visitation lookup

        Returns:
            Dictionary of computed metric values
        """
        current_metric_ranges = current_metric_ranges or {
            "Manipulability": (0.0, 1.0),
            "RangeOfMotion": (0.0, 1.0),
            "SingularityAvoidance": (0.0, 1.0),
            "Visitation": (0.0, 1.0),
        }

        # Compute manipulability
        manip_raw = 0.0
        if self.metric_chain is not None and self.manip_chain_indices is not None:
            chain_values: List[float] = []
            for idx in self.manip_chain_indices:
                if idx is None or idx >= q_current.shape[0]:
                    chain_values.append(0.0)
                else:
                    chain_values.append(float(q_current[idx]))
            try:
                if torch is not None:
                    with torch.no_grad():
                        q_tensor = torch.tensor(
                            [chain_values],
                            dtype=self.metric_chain_dtype,
                            device=self.metric_chain_device,
                        )
                        jacobian = self.metric_chain.jacobian(q_tensor)
                        jj_t = torch.matmul(jacobian, torch.transpose(jacobian, 1, 2))
                        det_val = torch.det(jj_t)
                        det_val = torch.clamp(det_val, min=0.0)
                        manip_raw = float(torch.sqrt(det_val).cpu().item())
            except Exception:
                manip_raw = 0.0

        # Normalize manipulability using min/max from samples (same as ROM and Singularity)
        manip_max = self.metric_metadata.get("ManipulabilityRawMax")
        if manip_max is None or manip_max <= 0.0:
            # Fallback to old scaling approach if max not available
            manip_scale = float(self.metric_metadata.get("ManipulabilityScaling", self.manip_scaling))
            manip_scaled = manip_raw * manip_scale
            manip_range = current_metric_ranges.get("Manipulability", (0.0, 1.0))
            manip_norm = float(
                np.clip(
                    (manip_scaled - manip_range[0]) / max(manip_range[1] - manip_range[0], 1e-6),
                    0.0,
                    1.0,
                )
            )
        else:
            manip_scaled = manip_raw  # No scaling needed, use raw value
            manip_norm = normalize_manipulability_scalar(manip_raw, manip_max)

        # Compute range of motion and joint limit distance
        if self.arm_joint_indices:
            arm_vals = q_current[self.arm_joint_indices]
            rom_raw = float(range_of_motion_numpy(arm_vals, self.arm_mid, self.arm_span))
            limit_distance = float(
                joint_limit_distance_numpy(
                    arm_vals, self.lower_deadband, self.upper_deadband, self.deadband_delta
                )
            )
        else:
            rom_raw = 0.0
            limit_distance = 0.0

        # Normalize range of motion
        rom_max = self.metric_metadata.get("RangeOfMotionRawMax", 1.0)
        if rom_max <= 0.0:
            rom_max = 1.0
        rom_norm = normalize_range_of_motion_scalar(rom_raw, rom_max)

        # Compute and normalize singularity
        sing_raw = float(singularity_weighted_numpy(manip_raw, limit_distance))
        sing_max = self.metric_metadata.get("SingularityWeightedMax")
        if sing_max is None or sing_max <= 0.0:
            sing_max = self.metric_metadata.get("SingularityRawMax", 1.0)
        if sing_max <= 0.0:
            sing_max = 1.0
        sing_norm = normalize_singularity_scalar(sing_raw, sing_max)
        limit_norm = normalize_joint_limit_distance_scalar(limit_distance)

        # Visitation lookup (if viewer available)
        visitation_raw = None
        vis_norm = None
        if viewer is not None:
            try:
                current_map = viewer._current_map()  # type: ignore[attr-defined]
                last_bucket_key = getattr(viewer, "_last_bucket_key", None)
                if current_map is not None and last_bucket_key is not None:
                    visitation_value = current_map.visitation_lookup.get(last_bucket_key)
                    if visitation_value is not None and np.isfinite(visitation_value):
                        visitation_raw = float(visitation_value)
                        vis_range = current_metric_ranges.get("Visitation", (0.0, 1.0))
                        vis_max = max(vis_range[1], 1e-6)
                        vis_norm = float(np.clip(visitation_raw / vis_max, 0.0, 1.0))
            except Exception:
                pass

        return {
            "manip_raw": manip_raw,
            "manip_scaled": manip_scaled,
            "manip_norm": manip_norm,
            "rom_raw": rom_raw,
            "rom_norm": rom_norm,
            "sing_raw": sing_raw,
            "sing_norm": sing_norm,
            "limit_distance": limit_distance,
            "limit_norm": limit_norm,
            "vis_raw": visitation_raw,
            "vis_norm": vis_norm,
        }

    def update_metric_display(
        self,
        metric_display_handles: Dict[str, object],
        q_current: np.ndarray,
        current_metric_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        viewer: Optional[object] = None,
    ) -> None:
        """Update metric display handles with current values.

        Args:
            metric_display_handles: Dictionary of GUI handles for metric display
            q_current: Current joint configuration
            current_metric_ranges: Optional metric ranges for normalization
            viewer: Optional reachability viewer for visitation lookup
        """
        if not metric_display_handles:
            return

        values = self.compute_metrics(q_current, current_metric_ranges, viewer)

        handle = metric_display_handles.get("Manipulability")
        if handle:
            handle.value = (
                f"norm: {values['manip_norm']:.3f} | scaled: {values['manip_scaled']:.4f} | raw: {values['manip_raw']:.4f}"
            )

        handle = metric_display_handles.get("RangeOfMotion")
        if handle:
            handle.value = f"norm: {values['rom_norm']:.3f} | raw: {values['rom_raw']:.4f}"

        handle = metric_display_handles.get("SingularityAvoidance")
        if handle:
            handle.value = f"norm: {values['sing_norm']:.3f} | raw: {values['sing_raw']:.4f}"

        handle = metric_display_handles.get("JointLimitDistance")
        if handle:
            handle.value = (
                f"score: {values['limit_norm']:.3f} | raw: {values['limit_distance']:.4f}"
            )

        handle = metric_display_handles.get("Visitation")
        if handle:
            if values["vis_raw"] is None or values["vis_norm"] is None:
                handle.value = "norm: -- | raw: --"
            else:
                handle.value = f"norm: {values['vis_norm']:.3f} | raw: {values['vis_raw']:.1f}"

