"""
CasADi components matching the C++ eSNS solver (ik_baseline.hpp).

Each component can be tested in isolation before combining into the full solver.
"""

from __future__ import annotations

from typing import Tuple

try:
    import casadi as ca
except ImportError:
    ca = None


# Defaults matching cpp_core/include/embodik/types.hpp
DEFAULT_EPSILON = 1e-6
DEFAULT_REGULARIZATION_FACTOR = 1e-1
DEFAULT_MAGNITUDE_LIMIT = 1e10  # VelocitySolverConfig.magnitude_limit
K_MIN_SCALING_MAGNITUDE = 1e-10
K_MAX_SCALING_MAGNITUDE = 1e10


def compute_regularized_inverse(
    J: "ca.SX",
    epsilon: float = DEFAULT_EPSILON,
    regularization_factor: float = DEFAULT_REGULARIZATION_FACTOR,
) -> "ca.SX":
    """
    Compute regularized pseudo-inverse matching C++ ComputeRegularizedInverse.

    C++ Reference: ik_baseline.hpp lines 71-114.

    Steps:
    1. Gram matrix G = J @ J.T
    2. Determinant-based regularization: if det(G) < epsilon^2, add (1-(det/eps^2)^2)*eps^2 to diagonal
    3. Additional diagonal damping for numerical stability (C++ uses SVD-based damping; we use fixed factor for codegen)
    4. J_pinv = J.T @ inv(G_reg)

    Args:
        J: Input matrix (m x n), typically m < n
        epsilon: Numerical tolerance threshold
        regularization_factor: Extra diagonal damping (C++ uses this for SVD-based damping)

    Returns:
        Regularized pseudo-inverse (n x m)
    """
    if ca is None:
        raise RuntimeError("CasADi is required")

    gram = J @ J.T
    m = gram.size1()
    threshold_squared = epsilon * epsilon

    # Determinant-based regularization (matches C++)
    det_val = ca.det(gram)
    reg = ca.if_else(
        det_val < threshold_squared,
        (1.0 - (det_val / threshold_squared) ** 2) * threshold_squared,
        0.0,
    )
    gram_reg = gram + reg * ca.SX.eye(m)
    # Additional damping for stability (SVD-based in C++ not easily codegen-friendly)
    gram_reg = gram_reg + regularization_factor * epsilon * ca.SX.eye(m)
    return J.T @ ca.inv(gram_reg)


def compute_single_objective_step(
    dq_prev: "ca.SX",
    J: "ca.SX",
    P: "ca.SX",
    target: "ca.SX",
    scale: "ca.SX",
    epsilon: float = DEFAULT_EPSILON,
    regularization_factor: float = DEFAULT_REGULARIZATION_FACTOR,
) -> "ca.SX":
    """
    Single objective velocity update: dq = dq_prev + J_pinv @ (scale * target - J @ dq_prev).

    C++ Reference: ik_baseline.hpp lines 426-431 (no saturation term).
    J_pinv is the regularized inverse of (J @ P), so the update lies in the null-space P.

    Args:
        dq_prev: Previous velocity (n_dof,)
        J: Task Jacobian (task_dim, n_dof)
        P: Null-space projector (n_dof, n_dof), identity for first task
        target: Task target velocity (task_dim,)
        scale: Task scale in [0, 1]
        epsilon: Regularization epsilon
        regularization_factor: Regularization factor

    Returns:
        Updated velocity (n_dof,)
    """
    if ca is None:
        raise RuntimeError("CasADi is required")
    JP = J @ P
    J_pinv = compute_regularized_inverse(JP, epsilon, regularization_factor)
    residual = scale * target - J @ dq_prev
    dq_new = dq_prev + J_pinv @ residual
    return dq_new


def compute_feasible_velocity_scale(
    scaled_contribution: "ca.SX",
    unscaled_contribution: "ca.SX",
    lower: "ca.SX",
    upper: "ca.SX",
    sat_selector_diag: "ca.SX",
    n_constraints: int,
    epsilon: float = 1e-12,
) -> "ca.SX":
    """
    Compute velocity scale as minimum feasible scale over non-saturated constraints.

    C++ Reference: ik_baseline.hpp lines 458-477.
    min_margin = lower - unscaled_contribution, max_margin = upper - unscaled_contribution.
    For each constraint i not saturated: feasible_scale[i] = ComputeFeasibleScalingRange(min_margin[i], max_margin[i], scaled_contribution[i]).first
    velocity_scale = min(feasible_scale).

    Args:
        scaled_contribution: C @ J_pinv @ target (n_constraints,)
        unscaled_contribution: constraint_eval - scaled_contribution (n_constraints,)
        lower: Lower bounds (n_constraints,)
        upper: Upper bounds (n_constraints,)
        sat_selector_diag: Diagonal of saturation selector, 1 if saturated (n_constraints,)
        n_constraints: Number of constraints
        epsilon: Small value for feasible_scale

    Returns:
        Scalar scale in [0, 1]
    """
    if ca is None:
        raise RuntimeError("CasADi is required")
    min_margin = lower - unscaled_contribution
    max_margin = upper - unscaled_contribution
    scale = ca.SX.ones(1)
    for i in range(n_constraints):
        # If already saturated, skip (scale stays)
        sat = ca.if_else(sat_selector_diag[i] > 0.5, 1.0, 0.0)
        scale_i = compute_feasible_scale(
            min_margin[i], max_margin[i], scaled_contribution[i], epsilon
        )
        # When not saturated, take min(scale, scale_i); when saturated, keep scale
        scale = ca.if_else(sat > 0.5, scale, ca.fmin(scale, scale_i))
    # Clamp inf to 0 (handled by feasible_scale returning [0,1])
    return ca.fmax(0.0, ca.fmin(1.0, scale))


def compute_regularized_inverse_simple(
    J: "ca.SX",
    epsilon: float = DEFAULT_EPSILON,
) -> "ca.SX":
    """
    Simplified regularized inverse without SVD (determinant + fixed damping only).

    Use this when SVD causes codegen issues. Matches C++ behavior for well-conditioned J.
    """
    if ca is None:
        raise RuntimeError("CasADi is required")

    gram = J @ J.T
    m = gram.size1()
    threshold_squared = epsilon * epsilon

    det_val = ca.det(gram)
    reg = ca.if_else(
        det_val < threshold_squared,
        (1.0 - (det_val / threshold_squared) ** 2) * threshold_squared,
        0.0,
    )
    gram_reg = gram + reg * ca.SX.eye(m)
    return J.T @ ca.inv(gram_reg)


def compute_generalized_inverse(
    M: "ca.SX",
    epsilon: float = DEFAULT_EPSILON,
) -> "ca.SX":
    """
    Compute Moore-Penrose pseudo-inverse of M (handles zero matrix).

    C++ Reference: detail::ComputeGeneralizedInverse (COD-based).
    For M (k x n) with k <= n: M^+ = M.T @ (M @ M.T + reg)^-1. When M is zero, result is zero.

    Args:
        M: Matrix (n_constraints x n_dof)
        epsilon: Regularization for numerical stability

    Returns:
        Pseudo-inverse (n_dof x n_constraints)
    """
    if ca is None:
        raise RuntimeError("CasADi is required")
    k = M.size1()
    # Right pseudo-inverse for M (k x n), k <= n: M^+ = M.T @ (M @ M.T + reg)^-1
    MMT = M @ M.T
    reg = epsilon * epsilon * ca.SX.eye(k)
    return M.T @ ca.solve(MMT + reg, ca.SX.eye(k))


def compute_augmented_projector(
    J_pinv: "ca.SX",
    J: "ca.SX",
    C_sat: "ca.SX",
    P: "ca.SX",
    epsilon: float = DEFAULT_EPSILON,
) -> "ca.SX":
    """
    Augmented projector: (I - J_pinv @ J) @ pinv(C_sat @ P).

    C++ Reference: ik_baseline.hpp lines 418-423.
    Projects the saturation correction into the null-space of J.

    Args:
        J_pinv: (n_dof x task_dim) regularized inverse of J@P
        J: (task_dim x n_dof)
        C_sat: (n_constraints x n_dof) saturated constraint matrix (diag(sat_diag) @ C)
        P: (n_dof x n_dof) null-space projector
        epsilon: For generalized inverse

    Returns:
        (n_dof x n_constraints) matrix
    """
    if ca is None:
        raise RuntimeError("CasADi is required")
    n_dof = J_pinv.size1()
    I_minus_Jpinv_J = ca.SX.eye(n_dof) - J_pinv @ J
    M = C_sat @ P  # (n_constraints x n_dof)
    inv_M = compute_generalized_inverse(M, epsilon)  # (n_dof x n_constraints)
    return I_minus_Jpinv_J @ inv_M


def compute_single_objective_step_with_saturation(
    dq_prev: "ca.SX",
    J: "ca.SX",
    P: "ca.SX",
    target: "ca.SX",
    scale: "ca.SX",
    J_pinv: "ca.SX",
    P_aug: "ca.SX",
    sat_vals: "ca.SX",
    C_sat: "ca.SX",
) -> "ca.SX":
    """
    Velocity update with saturation correction: dq_prev + J_pinv @ (scale*target - J@dq_prev) + P_aug @ (sat_vals - C_sat @ dq_prev).

    C++ Reference: ik_baseline.hpp lines 426-431.
    """
    if ca is None:
        raise RuntimeError("CasADi is required")
    task_residual = scale * target - J @ dq_prev
    sat_residual = sat_vals - C_sat @ dq_prev
    return dq_prev + J_pinv @ task_residual + P_aug @ sat_residual


def compute_feasible_scale(
    bound_lower: "ca.SX",
    bound_upper: "ca.SX",
    coefficient: "ca.SX",
    epsilon: float = 1e-12,
) -> "ca.SX":
    """
    Compute maximum feasible scale in [0, 1] so that bound_lower <= scale * coefficient <= bound_upper.

    C++ Reference: ComputeFeasibleScalingRange, ik_baseline.hpp lines 116-134.
    Returns the first element of the pair: min(1, max_allowed_scale).

    For scale in [0,1]: we need bound_lower <= scale * c <= bound_upper.
    - If c < 0 and bound_lower < 0 and c <= bound_upper: max_scale = bound_lower/c
    - If c > 0 and bound_upper > 0 and c >= bound_lower: max_scale = bound_upper/c
    Return min(1, max_scale) clamped to [0, 1].
    """
    if ca is None:
        raise RuntimeError("CasADi is required")

    abs_c = ca.fabs(coefficient)
    negligible = ca.if_else(
        (abs_c < K_MIN_SCALING_MAGNITUDE) + (abs_c > K_MAX_SCALING_MAGNITUDE),
        1.0,
        0.0,
    )

    # Case c < 0, bound_lower < 0, c <= bound_upper: max_scale = bound_lower/c
    max_scale_neg = ca.if_else(
        ca.fabs(coefficient) > epsilon,
        bound_lower / coefficient,
        1.0,
    )
    # Case c > 0, bound_upper > 0, c >= bound_lower: max_scale = bound_upper/c
    max_scale_pos = ca.if_else(
        ca.fabs(coefficient) > epsilon,
        bound_upper / coefficient,
        1.0,
    )

    # Select: use neg case when c < 0 and bound_lower < 0 and c <= bound_upper
    use_neg = ca.if_else(
        (coefficient < 0) * (bound_lower < 0) * (coefficient <= bound_upper + epsilon),
        1.0,
        0.0,
    )
    use_pos = ca.if_else(
        (coefficient > 0) * (bound_upper > 0) * (coefficient >= bound_lower - epsilon),
        1.0,
        0.0,
    )
    max_allowed = ca.if_else(
        use_neg > 0.5,
        max_scale_neg,
        ca.if_else(use_pos > 0.5, max_scale_pos, 1.0),
    )
    max_allowed = ca.if_else(negligible > 0.5, 1.0, max_allowed)
    scale = ca.fmax(0.0, ca.fmin(1.0, max_allowed))
    return scale
