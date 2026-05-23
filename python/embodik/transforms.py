"""Native spatial transform helpers (Rotation/SO3) replacing scipy.spatial.transform."""

from __future__ import annotations

from typing import Union

import numpy as np

from . import _embodik_impl as _native

__all__ = ["Rotation", "SO3"]


class Rotation:
    """
    Lightweight SO(3) rotation helper using native embodiK/Pinocchio ops.

    Replaces scipy.spatial.transform.Rotation for common use cases.
    Canonical methods (from_matrix, as_matrix, etc.) are primary;
    spatialmath-style shorthands (Rx, Rz, RPY, etc.) are convenience aliases.
    """

    __slots__ = ("_R",)

    def __init__(self, R: np.ndarray):
        self._R = np.asarray(R, dtype=float).reshape(3, 3).copy()

    # -------------------------------------------------------------------------
    # Canonical constructors
    # -------------------------------------------------------------------------

    @classmethod
    def from_matrix(cls, R: np.ndarray) -> "Rotation":
        """Create from 3x3 rotation matrix."""
        return cls(np.asarray(R, dtype=float))

    @classmethod
    def from_quat(cls, q: np.ndarray) -> "Rotation":
        """Create from quaternion [x, y, z, w] (scipy/ROS convention)."""
        q = np.asarray(q, dtype=float).ravel()
        if q.shape[0] != 4:
            raise ValueError(f"Expected 4-element quaternion, got shape {q.shape}")
        R = _native.quaternion_xyzw_to_matrix(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        return cls(np.array(R))

    @classmethod
    def from_rotvec(cls, omega: np.ndarray) -> "Rotation":
        """Create from axis-angle vector (rotation axis * angle in radians)."""
        omega = np.asarray(omega, dtype=float).ravel()
        if omega.shape[0] != 3:
            raise ValueError(f"Expected 3D rotvec, got shape {omega.shape}")
        R = _native.exp3(omega)
        return cls(np.array(R))

    @classmethod
    def from_euler(
        cls, seq: str, angles: Union[float, np.ndarray], degrees: bool = False
    ) -> "Rotation":
        """
        Create from Euler angles.

        Args:
            seq: 'xyz', 'x', 'y', or 'z'
            angles: Single angle (for single-axis) or [r, p, y] for 'xyz'
            degrees: If True, angles are in degrees
        """
        angles = np.atleast_1d(np.asarray(angles, dtype=float))
        if degrees:
            angles = np.deg2rad(angles)

        if seq == "xyz":
            if angles.size != 3:
                raise ValueError("For 'xyz', angles must be [r, p, y]")
            R = _native.rotation_from_rpy(float(angles[0]), float(angles[1]), float(angles[2]))
            return cls(np.array(R))
        elif seq == "x":
            return cls.Rx(float(angles.flat[0]))
        elif seq == "y":
            return cls.Ry(float(angles.flat[0]))
        elif seq == "z":
            return cls.Rz(float(angles.flat[0]))
        else:
            raise ValueError(f"Unsupported Euler sequence: {seq!r}. Use 'xyz', 'x', 'y', or 'z'.")

    @classmethod
    def identity(cls) -> "Rotation":
        """Create identity rotation."""
        return cls(np.eye(3))

    # -------------------------------------------------------------------------
    # Spatialmath-style shorthand constructors
    # -------------------------------------------------------------------------

    @classmethod
    def Rx(cls, theta: float) -> "Rotation":
        """Rotation about X-axis by theta radians."""
        return cls.from_rotvec(np.array([theta, 0.0, 0.0]))

    @classmethod
    def Ry(cls, theta: float) -> "Rotation":
        """Rotation about Y-axis by theta radians."""
        return cls.from_rotvec(np.array([0.0, theta, 0.0]))

    @classmethod
    def Rz(cls, theta: float) -> "Rotation":
        """Rotation about Z-axis by theta radians."""
        return cls.from_rotvec(np.array([0.0, 0.0, theta]))

    @classmethod
    def RPY(cls, rpy: np.ndarray, order: str = "xyz") -> "Rotation":
        """Create from roll-pitch-yaw [r, p, y] in radians."""
        rpy = np.asarray(rpy, dtype=float).ravel()
        if rpy.size != 3:
            raise ValueError("RPY must be [r, p, y]")
        if order != "xyz":
            raise ValueError("Only order='xyz' supported")
        R = _native.rotation_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        return cls(np.array(R))

    @classmethod
    def AngVec(cls, theta: float, axis: np.ndarray) -> "Rotation":
        """Create from angle (radians) and unit axis vector."""
        axis = np.asarray(axis, dtype=float).ravel()
        if axis.size != 3:
            raise ValueError("Axis must be 3D")
        n = np.linalg.norm(axis)
        if n < 1e-12:
            return cls.identity()
        omega = (theta / n) * axis
        return cls.from_rotvec(omega)

    @classmethod
    def EulerVec(cls, omega: np.ndarray) -> "Rotation":
        """Create from axis-angle vector (alias for from_rotvec)."""
        return cls.from_rotvec(omega)

    # -------------------------------------------------------------------------
    # Outputs (canonical)
    # -------------------------------------------------------------------------

    def as_matrix(self) -> np.ndarray:
        """Return 3x3 rotation matrix."""
        return self._R.copy()

    def as_quat(self) -> np.ndarray:
        """Return quaternion [x, y, z, w] (scipy/ROS convention)."""
        q = _native.matrix_to_quaternion_xyzw(self._R)
        return np.array(q)

    def as_rotvec(self) -> np.ndarray:
        """Return axis-angle vector (rotation axis * angle in radians)."""
        return np.array(_native.log3(self._R))

    # -------------------------------------------------------------------------
    # Property aliases (spatialmath-style)
    # -------------------------------------------------------------------------

    @property
    def R(self) -> np.ndarray:
        """Rotation matrix (alias for as_matrix())."""
        return self._R.copy()

    # -------------------------------------------------------------------------
    # Operations
    # -------------------------------------------------------------------------

    def inv(self) -> "Rotation":
        """Return inverse rotation."""
        return Rotation(self._R.T)

    def __mul__(self, other: "Rotation") -> "Rotation":
        """Compose rotations: self * other."""
        if not isinstance(other, Rotation):
            return NotImplemented
        return Rotation(self._R @ other._R)

    def apply(self, v: np.ndarray) -> np.ndarray:
        """Rotate 3D vector: R @ v."""
        v = np.asarray(v, dtype=float).reshape(3)
        return (self._R @ v).reshape(3)

    def __repr__(self) -> str:
        return f"Rotation(shape=(3,3))"


# Spatialmath-style alias
SO3 = Rotation
