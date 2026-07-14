"""Regression tests for pose and configuration health metrics."""

from __future__ import annotations

import numpy as np

import embodik as eik


def test_joint_limit_distance_gradient_matches_descent_direction() -> None:
    q = np.array([-0.8, 0.7], dtype=float)
    lower = np.array([-1.0, -2.0], dtype=float)
    upper = np.array([1.0, 2.0], dtype=float)
    epsilon = 0.04

    analytic = np.asarray(
        eik.joint_limit_distance_gradient(q, lower, upper, epsilon),
        dtype=float,
    )
    finite_difference = np.zeros_like(q)
    step = 1e-6
    for index in range(q.size):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[index] += step
        q_minus[index] -= step
        cost_plus = float(eik.joint_limit_distance(q_plus, lower, upper, epsilon)[1])
        cost_minus = float(eik.joint_limit_distance(q_minus, lower, upper, epsilon)[1])
        finite_difference[index] = -(cost_plus - cost_minus) / (2.0 * step)

    np.testing.assert_allclose(analytic, finite_difference, atol=1e-7, rtol=1e-6)
    assert analytic[0] > 0.0
    assert analytic[1] < 0.0
