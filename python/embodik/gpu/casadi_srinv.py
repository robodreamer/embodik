"""Shared CasADi regularized inverse helper for experimental GPU solvers."""

from __future__ import annotations

try:
    import casadi as ca
except ImportError:
    ca = None


def srinv(
    A: "ca.SX",
    tol: float,
    damping: float,
) -> "ca.SX":
    """Return the CasADi GPU-prototype regularized inverse.

    This is intentionally a symbolic-friendly approximation, not the full C++
    ``ComputeRegularizedInverse`` implementation.  The CPU solver adds
    per-singular-value damping from an SVD; CasADi's symbolic codegen path keeps
    the determinant guard and adds a uniform diagonal damping term instead.
    """
    if ca is None:
        raise RuntimeError("CasADi is required")

    aat = A @ A.T
    m = aat.size1()
    threshold_sq = tol * tol

    det_val = ca.det(aat)
    global_regularization = ca.if_else(
        det_val < threshold_sq,
        (1.0 - (det_val / threshold_sq) ** 2) * threshold_sq,
        0.0,
    )
    regularized = aat + global_regularization * ca.SX.eye(m)
    regularized = regularized + damping * tol * ca.SX.eye(m)

    return A.T @ ca.inv(regularized)
