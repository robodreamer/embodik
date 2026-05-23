"""Utility functions for embodiK."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

# Import native bindings for log3/exp3 and SE3
# These replace the Python pinocchio dependency
from . import _embodik_impl as _native

__all__ = [
    "PoseData",
    "get_pose_error_vector",
    "compute_pose_error",
    "limit_task_velocity",
    "normalize_quaternion",
    "clamp_configuration",
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
    if hasattr(pose_goal, "translation"):
        # Pinocchio SE3 object
        t_goal = pose_goal.translation
        t_current = pose_current.translation
        R_goal = pose_goal.rotation
        R_current = pose_current.rotation
    elif hasattr(pose_goal, "t"):
        # PoseData object
        t_goal = pose_goal.t
        t_current = pose_current.t
        R_goal = pose_goal.R
        R_current = pose_current.R
    else:
        raise TypeError(f"Unsupported pose type: {type(pose_goal)}")

    pose_error[:3] = t_goal - t_current
    # Use native log3 for rotation error (equivalent to SO3.log(twist=True))
    # Compute relative rotation: R_error = R_goal * R_current^T
    R_error = R_goal @ R_current.T
    pose_error[3:] = _native.log3(R_error)
    return pose_error


def compute_pose_error(pose_current: PoseData | object, pose_goal: PoseData | object) -> np.ndarray:
    """Compute 6D pose error (goal - current) using native :func:`log3`.

    Optimized to work directly with SE3 objects without wrapping when possible.
    Extracts rotation/translation once to minimize Python binding overhead.
    """
    # Fast path for native SE3 objects (most common case)
    if isinstance(pose_current, _native.SE3) and isinstance(pose_goal, _native.SE3):
        # Extract rotation and translation once to avoid repeated attribute access overhead
        t_current = pose_current.translation
        t_goal = pose_goal.translation
        R_current = pose_current.rotation
        R_goal = pose_goal.rotation

        error = np.empty(6, dtype=float)
        error[:3] = t_goal - t_current
        error[3:] = _native.log3(R_goal @ R_current.T)
        return error

    # Fallback to PoseData.wrap for other types
    current = PoseData.wrap(pose_current)
    goal = PoseData.wrap(pose_goal)
    error = np.empty(6, dtype=float)
    error[:3] = goal.t - current.t
    error[3:] = _native.log3(goal.R @ current.R.T)
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


def clamp_configuration(
    configuration: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Clip joint configuration to provided limits."""

    return np.clip(configuration, lower, upper)


def r2q(rotation: np.ndarray, order: str = "sxyz") -> np.ndarray:
    """
    Convert rotation matrix to quaternion (spatialmath-python compatible).

    Uses native embodiK/Pinocchio conversion (no SciPy dependency).

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

    if order == "sxyz" or order == "wxyz":
        # Scalar first: [w, x, y, z] via native wxyz converter
        quat = _native.matrix_to_quaternion_wxyz(rotation)
        return np.array(quat, dtype=float)
    elif order == "xyzs" or order == "xyzw":
        # Scalar last: [x, y, z, w] via native xyzw converter
        quat = _native.matrix_to_quaternion_xyzw(rotation)
        return np.array(quat, dtype=float)
    else:
        raise ValueError(f"Unknown quaternion order: {order}. Use 'sxyz' or 'xyzs'")


def q2r(quaternion: np.ndarray, order: str = "sxyz") -> np.ndarray:
    """
    Convert quaternion to rotation matrix (spatialmath-python compatible).

    Uses native embodiK/Pinocchio conversion (no SciPy dependency).

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

    # Normalize quaternion
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        return np.eye(3, dtype=float)
    qn = quaternion / norm

    if order == "sxyz" or order == "wxyz":
        # Scalar first: [w, x, y, z]
        R = _native.quaternion_wxyz_to_matrix(
            float(qn[0]), float(qn[1]), float(qn[2]), float(qn[3])
        )
    elif order == "xyzs" or order == "xyzw":
        # Scalar last: [x, y, z, w]
        R = _native.quaternion_xyzw_to_matrix(
            float(qn[0]), float(qn[1]), float(qn[2]), float(qn[3])
        )
    else:
        raise ValueError(f"Unknown quaternion order: {order}. Use 'sxyz' or 'xyzs'")

    return np.array(R)


def Rt(R: Optional[np.ndarray] = None, t: Optional[np.ndarray] = None) -> Any:
    """
    Create SE3 transform from rotation matrix and translation (spatialmath-python compatible).

    Equivalent to spatialmath-python's SE3.Rt(R, t).

    Args:
        R: 3x3 rotation matrix (default: identity)
        t: 3D translation vector (default: zero)

    Returns:
        SE3 transform

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

    return _native.SE3(R, t)
