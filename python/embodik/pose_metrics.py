"""
Pose metrics -- delegates to C++ for efficiency, with Python fallbacks.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import embodik as _eik


def joint_limit_distance(
    q: NDArray[np.float64],
    q_lower: NDArray[np.float64],
    q_upper: NDArray[np.float64],
    epsilon: float = 0.04,
) -> tuple[NDArray[np.float64], float]:
    """Per-joint and aggregate joint-limit distance metric."""
    return _eik.joint_limit_distance(q, q_lower, q_upper, epsilon)


def joint_limit_distance_gradient(
    q: NDArray[np.float64],
    q_lower: NDArray[np.float64],
    q_upper: NDArray[np.float64],
    epsilon: float = 0.04,
) -> NDArray[np.float64]:
    """Analytical gradient of joint_limit_distance (descent direction)."""
    return _eik.joint_limit_distance_gradient(q, q_lower, q_upper, epsilon)


def velocity_manipulability(jacobian: NDArray[np.float64]) -> float:
    """Velocity manipulability: sqrt(det(J * J^T))."""
    return _eik.velocity_manipulability(jacobian)


def singularity_joint_limit_metric(
    q: NDArray[np.float64],
    jacobian: NDArray[np.float64],
    q_lower: NDArray[np.float64],
    q_upper: NDArray[np.float64],
    epsilon: float = 0.04,
) -> float:
    """Combined singularity + joint-limit metric."""
    return _eik.singularity_joint_limit_metric(q, jacobian, q_lower, q_upper, epsilon)


def metric_gradient_numerical(
    q: NDArray[np.float64],
    metric_fn,
    epsilon: float = 1e-4,
) -> NDArray[np.float64]:
    """Central-difference gradient of any scalar metric function (Python only)."""
    n = len(q)
    grad = np.zeros(n)
    for i in range(n):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[i] += epsilon
        q_minus[i] -= epsilon
        grad[i] = (metric_fn(q_plus) - metric_fn(q_minus)) / (2.0 * epsilon)
    return grad
