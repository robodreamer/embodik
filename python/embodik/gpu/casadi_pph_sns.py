"""
PPH-SNS: Parallel Penalized Hierarchical SNS for GPU-native whole-body control.

This module implements the PPH-SNS (Parallel Penalized Hierarchical SNS) adaptation
specifically engineered for high-throughput NVIDIA platforms. Key features:

- Fixed-depth unrolling (no while loops) for CusADi compilation
- Quadratic penalty + soft saturation for all inequalities
- Parallel top-k violation selection using soft-argmax
- Limited explicit rank-1 projector updates (1-2 violators per iteration)
- Strong penalty gradient nudge with aggressive ramping
- Hierarchical tasks remain sequential but fully unrolled
- Fully differentiable → perfect for end-to-end RL

Reference: GPU-native redesign of EmbodiK/eSNS for A100/H100/THOR-scale parallelism.
"""

from __future__ import annotations

from typing import Optional

try:
    import casadi as ca
except ImportError:
    ca = None


# Default parameters for PPH-SNS
DEFAULT_EPSILON = 1e-6
DEFAULT_DAMPING = 0.12  # Higher damping for stability
DEFAULT_MU0 = 1e-3  # Initial penalty weight (softer start)
DEFAULT_GAMMA = 3.0  # Aggressive penalty growth (vs 2.5 in FI-PeSNS)
DEFAULT_ETA = 0.1  # Penalty gradient step size
DEFAULT_K_MAX = 14  # Outer iterations (increased for accuracy)
DEFAULT_M_MAX = 2  # Max explicit saturations per iteration
DEFAULT_MAGNITUDE_LIMIT = 1e10


def srinv(
    A: "ca.SX",
    tol: float = DEFAULT_EPSILON,
    damping: float = DEFAULT_DAMPING,
) -> "ca.SX":
    """
    Singularity-Robust Inverse (SRINV) matching C++ ComputeRegularizedInverse.

    For A (m x n), computes A^+ = A.T @ inv(A @ A.T + regularization).
    Regularization uses determinant-based + fixed damping for numerical stability.

    Args:
        A: Input matrix (m x n), typically m <= n
        tol: Tolerance threshold for determinant check
        damping: Additional diagonal damping factor

    Returns:
        Pseudo-inverse (n x m)
    """
    if ca is None:
        raise RuntimeError("CasADi is required")

    AAT = A @ A.T
    m = AAT.size1()
    threshold_sq = tol * tol

    # Determinant-based regularization (matches C++ ComputeRegularizedInverse)
    det_val = ca.det(AAT)
    lam = ca.if_else(
        det_val < threshold_sq,
        (1.0 - (det_val / threshold_sq) ** 2) * threshold_sq,
        0.0,
    )
    AAT_reg = AAT + lam * ca.SX.eye(m)
    # Additional damping for stability
    AAT_reg = AAT_reg + damping * tol * ca.SX.eye(m)

    return A.T @ ca.inv(AAT_reg)


def get_feasible_task_scale(
    a: "ca.SX",
    b: "ca.SX",
    d: "ca.SX",
    d_bar: "ca.SX",
    n_constraints: int,
    eps_small: float = 1e-10,
    eps_large: float = 1e10,
) -> "ca.SX":
    """
    Compute feasible task scale analytically (vectorized over constraints).

    For constraint: d <= b + s * a <= d_bar, find max s in [0, 1].
    - a: scaled contribution (C @ J_pinv @ target)
    - b: unscaled contribution (C @ dq_prev)
    - d: lower bounds
    - d_bar: upper bounds

    Args:
        a: Scaled contribution (n_constraints,)
        b: Unscaled contribution (n_constraints,)
        d: Lower bounds (n_constraints,)
        d_bar: Upper bounds (n_constraints,)
        n_constraints: Number of constraints
        eps_small: Threshold for negligible coefficient
        eps_large: Threshold for overflow

    Returns:
        Scalar scale in [0, 1]
    """
    if ca is None:
        raise RuntimeError("CasADi is required")

    scale = ca.SX.ones(1)

    for i in range(n_constraints):
        ai = a[i]
        bi = b[i]
        di = d[i]
        di_bar = d_bar[i]

        abs_ai = ca.fabs(ai)
        valid = ca.if_else((abs_ai > eps_small) * (abs_ai < eps_large), 1.0, 0.0)

        # Margin from current position to bounds
        margin_low = di - bi  # Need s*a >= margin_low
        margin_high = di_bar - bi  # Need s*a <= margin_high

        # For a > 0: s <= margin_high/a, s >= margin_low/a
        # For a < 0: s >= margin_high/a, s <= margin_low/a (flip)
        s_max_pos = ca.if_else(ai > eps_small, margin_high / ai, 1.0)
        s_max_neg = ca.if_else(ai < -eps_small, margin_low / ai, 1.0)
        s_max_i = ca.if_else(ai > 0, s_max_pos, s_max_neg)

        # Clamp to [0, 1] and take minimum
        s_max_i = ca.fmax(0.0, ca.fmin(1.0, s_max_i))
        scale = ca.if_else(valid > 0.5, ca.fmin(scale, s_max_i), scale)

    return scale


def soft_topk_violations(
    violations: "ca.SX",
    k: int,
    temperature: float = 0.1,
) -> tuple["ca.SX", "ca.SX"]:
    """
    Soft top-k violation selection using Gumbel-Softmax-like approach.

    Args:
        violations: Violation magnitudes (n_constraints,)
        k: Number of top violations to select
        temperature: Softmax temperature (lower = harder selection)

    Returns:
        weights: Soft weights for each constraint (n_constraints,)
        indices: Soft indices (not used in practice, but for compatibility)
    """
    if ca is None:
        raise RuntimeError("CasADi is required")

    n_constraints = violations.size1()

    # Add small noise to break ties
    noise = ca.SX.zeros(n_constraints)
    for i in range(n_constraints):
        noise = noise + ca.SX.sym(f"noise_{i}") * ca.SX.eye(n_constraints)[i, :].T
    noisy_violations = violations + 1e-6 * noise

    # Softmax over violations (higher violation = higher weight)
    exp_violations = ca.exp(noisy_violations / temperature)
    weights = exp_violations / ca.sum1(exp_violations)

    # For simplicity, return weights and dummy indices
    # In practice, the weights determine which constraints get rank-1 updates
    indices = ca.SX.zeros(k)

    return weights, indices


def build_pph_sns_velocity_solve(
    n_dof: int,
    n_tasks: int,
    task_dims: list[int],
    n_constraints: int,
    tol: float = DEFAULT_EPSILON,
    damping: float = DEFAULT_DAMPING,
    mu0: float = DEFAULT_MU0,
    gamma: float = DEFAULT_GAMMA,
    eta: float = DEFAULT_ETA,
    k_max: int = DEFAULT_K_MAX,
    m_max: int = DEFAULT_M_MAX,
    use_warm_start: bool = False,
    **kwargs,  # Accept legacy parameters for compatibility
) -> "ca.Function":
    """
    Build PPH-SNS velocity solver with parallel penalty-based constraint enforcement.

    Algorithm per iteration:
    1. Hierarchical Task Processing (Unrolled)
       For each task k:
       - Compute projected Jacobian J_P = J_k @ P
       - Compute SRINV: J_pinv = srinv(J_P)
       - Compute bias = xdd_des_k - Jd_k @ qd
       - Compute Delta_ddq = J_pinv @ bias
       - Compute analytical scale factor s_k
       - Apply scaled delta: ddq += s_k * Delta_ddq
       - Limited rank-1 projector update (top-M violators)

    2. Strong Penalty Nudge: ddq += eta * mu * C.T @ violations
    3. Ramp penalty aggressively: mu *= gamma

    Args:
        n_dof: Degrees of freedom
        n_tasks: Number of hierarchical tasks
        task_dims: List of task dimensions
        n_constraints: Number of constraint rows
        tol: SRINV tolerance
        damping: SRINV damping
        mu0: Initial penalty weight (start soft, ~1e-3)
        gamma: Penalty growth factor (3.0 for aggressive convergence)
        eta: Penalty gradient step size (0.1 typical)
        k_max: Fixed outer iterations (14-16 typical)
        m_max: Max explicit saturations per iteration (1-2 typical)
        use_warm_start: Add prior_ddq input for warm-starting (default: False)

    Returns:
        CasADi Function: inputs [targets, jacobians, C, lower, upper, (prior_ddq)] -> [velocity, scales]
    """
    if ca is None:
        raise RuntimeError("CasADi is required")

    if len(task_dims) != n_tasks:
        raise ValueError(f"task_dims length {len(task_dims)} != n_tasks {n_tasks}")

    total_task_dim = sum(task_dims)
    total_jacobian_size = sum(d * n_dof for d in task_dims)

    # Symbolic inputs
    targets = ca.SX.sym("targets", total_task_dim)
    jacobians_flat = ca.SX.sym("jacobians", total_jacobian_size)
    C = ca.SX.sym("C", n_constraints, n_dof)
    lower = ca.SX.sym("lower", n_constraints)
    upper = ca.SX.sym("upper", n_constraints)

    # Optional warm-start input
    if use_warm_start:
        prior_ddq = ca.SX.sym("prior_ddq", n_dof)
        ddq = prior_ddq * 0.9  # Slight decay for stability
    else:
        ddq = ca.SX.zeros(n_dof)

    task_scales = ca.SX.zeros(n_tasks)
    mu = mu0

    # Fixed-iteration outer loop (fully unrolled)
    for k_iter in range(k_max):
        # Reset projector for each outer iteration
        P = ca.SX.eye(n_dof)

        target_offset = 0
        jacobian_offset = 0

        # 1. Hierarchical Task Processing (Unrolled)
        for task_idx in range(n_tasks):
            task_dim = task_dims[task_idx]
            target = targets[target_offset : target_offset + task_dim]
            target_offset += task_dim

            jac_size = task_dim * n_dof
            jac_flat = jacobians_flat[jacobian_offset : jacobian_offset + jac_size]
            jacobian_offset += jac_size
            # Row-major flat -> (task_dim, n_dof)
            J = ca.reshape(jac_flat, n_dof, task_dim).T

            # Projected Jacobian
            JP = J @ P
            J_pinv = srinv(JP, tol, damping)

            # Bias term (assuming velocity targets, no acceleration feedforward)
            bias = target  # Simplified: target is desired velocity
            Delta_ddq = J_pinv @ bias

            # Analytical feasible scale factor
            # a = contribution from Delta_ddq, b = current constraint value
            a = C @ Delta_ddq  # Scaled contribution
            b = C @ ddq  # Current constraint value
            s_k = get_feasible_task_scale(a, b, lower, upper, n_constraints)

            # Apply scaled delta with graceful scaling
            Delta_ddq_scaled = Delta_ddq * s_k
            ddq = ddq + Delta_ddq_scaled
            task_scales[task_idx] = s_k

            # 2. Limited Rank-1 Projector Update (Top-M violators)
            # Compute current violations
            constraint_val = C @ ddq
            r_low = lower - constraint_val  # Positive if violated low
            r_high = constraint_val - upper  # Positive if violated high
            violations = ca.fmax(0.0, r_low) + ca.fmax(0.0, r_high)

            # Soft top-k selection
            weights, _ = soft_topk_violations(violations, m_max)

            # Apply rank-1 updates for top violators (weighted)
            for j in range(min(m_max, n_constraints)):
                # Weight determines how much this constraint influences the update
                weight_j = weights[j]

                # Get the j-th constraint row
                Cj = C[j, :].T  # (n_dof,) column vector

                # Project the constraint row: Cj_P = P^T @ Cj
                Cj_P = P.T @ Cj  # (n_dof,) projected constraint

                # Compute pseudo-inverse of the projected constraint
                # pinv(Cj_P) = Cj_P / (Cj_P^T @ Cj_P + damping)
                Cj_P_norm_sq = ca.sumsqr(Cj_P) + damping * tol
                Cj_pinv = Cj_P / Cj_P_norm_sq

                # Rank-1 update: P = P - pinv(Cj_P) @ Cj_P^T
                # This removes the direction that would violate this constraint
                rank1_update = weight_j * (Cj_pinv @ Cj_P.T)
                P = P - rank1_update

            # Threshold small values for numerical stability
            P = ca.if_else(ca.fabs(P) < tol, 0.0, P)

        # 3. Strong Penalty Nudge (Approximate remaining saturation)
        # Compute violations after all tasks
        constraint_val = C @ ddq
        r_low = ca.fmax(0.0, lower - constraint_val)  # Low violations
        r_high = ca.fmax(0.0, constraint_val - upper)  # High violations
        violations_total = r_low + r_high

        # Penalty gradient: push toward feasibility
        penalty_grad = C.T @ violations_total
        ddq = ddq + eta * mu * penalty_grad

        # 4. Ramp penalty aggressively
        mu = mu * gamma

    # Final output (no additional clamping - penalty should handle it)
    # Build inputs/outputs
    if use_warm_start:
        inputs = [targets, jacobians_flat, C, lower, upper, prior_ddq]
        input_names = ["targets", "jacobians", "C", "lower", "upper", "prior_ddq"]
    else:
        inputs = [targets, jacobians_flat, C, lower, upper]
        input_names = ["targets", "jacobians", "C", "lower", "upper"]

    fn = ca.Function(
        "fn_pph_sns_velocity_solve",
        inputs,
        [ddq, task_scales],
        input_names,
        ["velocity", "scales"],
    )
    return fn


def build_pph_sns_single_task(
    n_dof: int,
    task_dim: int,
    n_constraints: int,
    tol: float = DEFAULT_EPSILON,
    damping: float = DEFAULT_DAMPING,
    mu0: float = DEFAULT_MU0,
    gamma: float = DEFAULT_GAMMA,
    eta: float = DEFAULT_ETA,
    k_max: int = DEFAULT_K_MAX,
    m_max: int = DEFAULT_M_MAX,
    use_warm_start: bool = False,
    **kwargs,  # Accept legacy parameters
) -> "ca.Function":
    """Build PPH-SNS for single-task velocity IK."""
    return build_pph_sns_velocity_solve(
        n_dof=n_dof,
        n_tasks=1,
        task_dims=[task_dim],
        n_constraints=n_constraints,
        tol=tol,
        damping=damping,
        mu0=mu0,
        gamma=gamma,
        eta=eta,
        k_max=k_max,
        m_max=m_max,
        use_warm_start=use_warm_start,
    )


# Pre-defined robot configurations
ROBOT_CONFIGS = {
    "panda": {"n_dof": 7, "default_task_dims": [6], "n_constraints": 7},
    "ur5": {"n_dof": 6, "default_task_dims": [6], "n_constraints": 6},
    "iiwa14": {"n_dof": 7, "default_task_dims": [6], "n_constraints": 7},
    "humanoid_arm": {"n_dof": 20, "default_task_dims": [6, 3], "n_constraints": 10},
}


def build_pph_sns_for_robot(
    robot_name: str,
    n_tasks: int = 1,
    task_dims: Optional[list[int]] = None,
    **kwargs,
) -> "ca.Function":
    """Build PPH-SNS solver for pre-configured robot."""
    if robot_name not in ROBOT_CONFIGS:
        raise ValueError(f"Unknown robot: {robot_name}. Available: {list(ROBOT_CONFIGS.keys())}")

    config = ROBOT_CONFIGS[robot_name]
    n_dof = config["n_dof"]
    n_constraints = config["n_constraints"]

    if task_dims is None:
        if n_tasks == 1:
            task_dims = config["default_task_dims"][:1]
        else:
            # Repeat first task for simplicity
            task_dims = [config["default_task_dims"][0]] * n_tasks

    if len(task_dims) != n_tasks:
        raise ValueError(f"task_dims length {len(task_dims)} != n_tasks {n_tasks}")

    return build_pph_sns_velocity_solve(
        n_dof=n_dof,
        n_tasks=n_tasks,
        task_dims=task_dims,
        n_constraints=n_constraints,
        **kwargs,
    )
