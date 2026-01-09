"""Utility functions for embodiK."""

from __future__ import annotations
from typing import Any, Optional, Tuple

import numpy as np

from ._runtime_deps import import_pinocchio as _import_pinocchio

__all__ = [
    "PoseData",
    "get_pose_error_vector",
    "compute_pose_error",
    "limit_task_velocity",
    "normalize_quaternion",
    "clamp_configuration",
    "apply_joint_limit_barrier_to_velocities",
    # Spatialmath-python compatible functions
    "r2q",
    "q2r",
    "Rt",
]


class PoseData:
    """Lightweight holder for rotation/translation pairs."""

    __slots__ = ("R", "t")

    def __init__(self, rotation: np.ndarray, translation: np.ndarray):
        self.R = np.asarray(rotation, dtype=float).reshape(3, 3)
        self.t = np.asarray(translation, dtype=float).reshape(3)

    @classmethod
    def wrap(cls, pose: object) -> "PoseData":
        if isinstance(pose, cls):
            return pose

        rotation = getattr(pose, "R", getattr(pose, "rotation", None))
        translation = getattr(pose, "t", getattr(pose, "translation", None))
        if rotation is None or translation is None:
            raise TypeError(
                "Expected pose-like object with rotation/translation attributes; got "
                f"{type(pose)!r}"
            )
        return cls(rotation, translation)


def get_pose_error_vector(pose_current, pose_goal):
    """Pose-error helper function.

    Uses Pinocchio's log3 for rotation error computation.

    Supports both PoseData objects and Pinocchio SE3 objects.
    """

    pose_error = np.zeros(6, dtype=np.float64)

    # Handle both PoseData (has .t) and Pinocchio SE3 (has .translation)
    if hasattr(pose_goal, 'translation'):
        # Pinocchio SE3 object
        t_goal = pose_goal.translation
        t_current = pose_current.translation
        R_goal = pose_goal.rotation
        R_current = pose_current.rotation
    elif hasattr(pose_goal, 't'):
        # PoseData object
        t_goal = pose_goal.t
        t_current = pose_current.t
        R_goal = pose_goal.R
        R_current = pose_current.R
    else:
        raise TypeError(f"Unsupported pose type: {type(pose_goal)}")

    pose_error[:3] = t_goal - t_current
    # Use Pinocchio's log3 for rotation error (equivalent to SO3.log(twist=True))
    # Compute relative rotation: R_error = R_goal * R_current^T
    R_error = R_goal @ R_current.T
    # Pinocchio's log3 handles rotation matrices directly (no need for explicit normalization)
    pin = _import_pinocchio()
    pose_error[3:] = pin.log3(R_error)
    return pose_error


def compute_pose_error(pose_current: PoseData | object, pose_goal: PoseData | object) -> np.ndarray:
    """Compute 6D pose error (goal - current) using Pinocchio's :func:`log3`.

    Optimized to work directly with Pinocchio SE3 objects without wrapping when possible.
    Extracts rotation/translation once to minimize Python binding overhead.
    """
    pin = _import_pinocchio()
    # Fast path for Pinocchio SE3 objects (most common case)
    if isinstance(pose_current, pin.SE3) and isinstance(pose_goal, pin.SE3):
        # Extract rotation and translation once to avoid repeated attribute access overhead
        t_current = pose_current.translation
        t_goal = pose_goal.translation
        R_current = pose_current.rotation
        R_goal = pose_goal.rotation

        error = np.empty(6, dtype=float)
        error[:3] = t_goal - t_current
        error[3:] = pin.log3(R_goal @ R_current.T)
        return error

    # Fallback to PoseData.wrap for other types
    current = PoseData.wrap(pose_current)
    goal = PoseData.wrap(pose_goal)
    error = np.empty(6, dtype=float)
    error[:3] = goal.t - current.t
    error[3:] = pin.log3(goal.R @ current.R.T)
    return error


def limit_task_velocity(
    velocity_error: np.ndarray,
    max_linear_step: float = 0.1,
    max_angular_step: float = 0.1,
    *,
    enable_debug: bool = False,
    debug_logger=None,
) -> np.ndarray:
    """Clamp linear and angular components of a 6D velocity vector."""

    limited_error = velocity_error.copy()

    linear_norm = float(np.linalg.norm(limited_error[:3]))
    if linear_norm > max_linear_step and linear_norm > 1e-9:
        scale = max_linear_step / linear_norm
        limited_error[:3] *= scale
        if enable_debug and debug_logger is not None:
            debug_logger.info(
                "Linear velocity limited: %.3f -> %.3f m/s", linear_norm, max_linear_step
            )

    angular_norm = float(np.linalg.norm(limited_error[3:]))
    if angular_norm > max_angular_step and angular_norm > 1e-9:
        scale = max_angular_step / angular_norm
        limited_error[3:] *= scale
        if enable_debug and debug_logger is not None:
            debug_logger.info(
                "Angular velocity limited: %.3f -> %.3f rad/s",
                angular_norm,
                max_angular_step,
            )

    return limited_error


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Return a unit quaternion, defaulting to ``[0,0,0,1]`` if norm is tiny."""

    quat = np.asarray(quaternion, dtype=float).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return quat / norm


def clamp_configuration(configuration: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Clip joint configuration to provided limits."""

    return np.clip(configuration, lower, upper)


def apply_joint_limit_barrier_to_velocities(
    q_current: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    joint_velocities: np.ndarray,
    dt: float,
    *,
    barrier_margin: float = 0.1,
    barrier_gain: float = 0.5,
    num_arm_joints: int = 7,
    enable_debug: bool = False,
    debug_logger: Optional[object] = None,
) -> np.ndarray:
    """
    Apply barrier function to modify joint velocities, pushing away from limits.

    This is more effective than modifying positions as it works with the solver's
    velocities rather than against them.

    Args:
        q_current: Current joint configuration
        q_lower: Lower joint limits
        q_upper: Upper joint limits
        joint_velocities: Joint velocities from solver
        dt: Time step for integration (unused but kept for API compatibility)
        barrier_margin: Distance from limit to activate barrier (radians)
        barrier_gain: Strength of barrier push
        num_arm_joints: Number of arm joints (excludes gripper)
        enable_debug: Whether to log debug info
        debug_logger: Logger instance for debug output

    Returns:
        Modified joint velocities with barrier forces added
    """
    modified_velocities = joint_velocities.copy()
    arm_limit = min(num_arm_joints, len(q_current), len(joint_velocities))

    for i in range(arm_limit):
        # Distance to limits
        dist_to_lower = q_current[i] - q_lower[i]
        dist_to_upper = q_upper[i] - q_current[i]

        # Apply barrier force if too close to limits
        if dist_to_lower < barrier_margin:
            # Barrier velocity away from lower limit
            barrier_vel = barrier_gain * (1.0 / max(dist_to_lower, 1e-6) - 1.0 / barrier_margin)
            barrier_vel = min(barrier_vel, 1.0)  # Cap maximum push

            # If solver wants to go towards limit, override it
            if joint_velocities[i] < 0:
                modified_velocities[i] = barrier_vel
            else:
                # Add to existing velocity away from limit
                modified_velocities[i] += barrier_vel

            if enable_debug and debug_logger is not None:
                debug_logger.info(
                    f"Barrier active for joint {i} (lower): dist={dist_to_lower:.4f}, "
                    f"original_vel={joint_velocities[i]:.4f}, modified_vel={modified_velocities[i]:.4f}"
                )

        elif dist_to_upper < barrier_margin:
            # Barrier velocity away from upper limit
            barrier_vel = barrier_gain * (1.0 / max(dist_to_upper, 1e-6) - 1.0 / barrier_margin)
            barrier_vel = min(barrier_vel, 1.0)  # Cap maximum push

            # If solver wants to go towards limit, override it
            if joint_velocities[i] > 0:
                modified_velocities[i] = -barrier_vel
            else:
                # Add to existing velocity away from limit
                modified_velocities[i] -= barrier_vel

            if enable_debug and debug_logger is not None:
                debug_logger.info(
                    f"Barrier active for joint {i} (upper): dist={dist_to_upper:.4f}, "
                    f"original_vel={joint_velocities[i]:.4f}, modified_vel={modified_velocities[i]:.4f}"
                )

    return modified_velocities


def r2q(rotation: np.ndarray, order: str = "sxyz") -> np.ndarray:
    """
    Convert rotation matrix to quaternion (spatialmath-python compatible).

    Optimized implementation using scipy for better performance than Pinocchio Quaternion.

    Args:
        rotation: 3x3 rotation matrix
        order: Quaternion order, 'sxyz' (default, [w,x,y,z]) or 'xyzs' ([x,y,z,w])

    Returns:
        Quaternion as numpy array:
        - If order='sxyz': [w, x, y, z] (scalar first, default)
        - If order='xyzs': [x, y, z, w] (scalar last)

    Examples:
        >>> R = np.eye(3)
        >>> q = r2q(R)  # Returns [1, 0, 0, 0] (wxyz format)
        >>> q = r2q(R, order='xyzs')  # Returns [0, 0, 0, 1] (xyzw format)
    """
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation matrix, got shape {rotation.shape}")

    # Use scipy for conversion (faster than Pinocchio Quaternion object creation)
    from scipy.spatial.transform import Rotation as R
    r = R.from_matrix(rotation)
    quat_xyzw = r.as_quat()  # Returns [x, y, z, w]

    if order == "sxyz" or order == "wxyz":
        # Scalar first: [w, x, y, z]
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)
    elif order == "xyzs" or order == "xyzw":
        # Scalar last: [x, y, z, w]
        return quat_xyzw.copy()
    else:
        raise ValueError(f"Unknown quaternion order: {order}. Use 'sxyz' or 'xyzs'")


def q2r(quaternion: np.ndarray, order: str = "sxyz") -> np.ndarray:
    """
    Convert quaternion to rotation matrix (spatialmath-python compatible).

    Optimized implementation using direct matrix computation to avoid Pinocchio Quaternion overhead.

    Args:
        quaternion: Quaternion as array
        order: Quaternion order, 'sxyz' (default, [w,x,y,z]) or 'xyzs' ([x,y,z,w])

    Returns:
        3x3 rotation matrix

    Examples:
        >>> q = np.array([1, 0, 0, 0])  # Identity quaternion (wxyz)
        >>> R = q2r(q)  # Returns 3x3 identity matrix
        >>> q = np.array([0, 0, 0, 1])  # Identity quaternion (xyzw)
        >>> R = q2r(q, order='xyzs')  # Returns 3x3 identity matrix
    """
    quaternion = np.asarray(quaternion, dtype=float)
    if quaternion.shape != (4,):
        raise ValueError(f"Expected 4-element quaternion, got shape {quaternion.shape}")

    if order == "sxyz" or order == "wxyz":
        # Scalar first: [w, x, y, z]
        w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    elif order == "xyzs" or order == "xyzw":
        # Scalar last: [x, y, z, w]
        x, y, z, w = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    else:
        raise ValueError(f"Unknown quaternion order: {order}. Use 'sxyz' or 'xyzs'")

    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-12:
        return np.eye(3, dtype=float)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    # Direct rotation matrix computation (faster than Pinocchio Quaternion object creation)
    # Using standard quaternion to rotation matrix formula
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=float)
    return R


def Rt(R: Optional[np.ndarray] = None, t: Optional[np.ndarray] = None) -> Any:
    """
    Create SE3 transform from rotation matrix and translation (spatialmath-python compatible).

    Equivalent to spatialmath-python's SE3.Rt(R, t).

    Args:
        R: 3x3 rotation matrix (default: identity)
        t: 3D translation vector (default: zero)

    Returns:
        Pinocchio SE3 transform

    Examples:
        >>> R = np.eye(3)
        >>> t = np.array([1, 2, 3])
        >>> T = Rt(R=R, t=t)  # Create SE3 from R and t
        >>> T = Rt(t=t)  # Create SE3 with identity rotation
        >>> T = Rt(R=R)  # Create SE3 with zero translation
        >>> T = Rt()  # Create identity transform
    """
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float)

    if R.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation matrix, got shape {R.shape}")
    if t.shape != (3,):
        raise ValueError(f"Expected 3D translation vector, got shape {t.shape}")

    pin = _import_pinocchio()
    return pin.SE3(R, t)
