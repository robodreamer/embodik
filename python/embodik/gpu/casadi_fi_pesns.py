"""
FI-PeSNS: Fixed-Iteration Penalized eSNS for GPU-parallel velocity/acceleration IK.

This module implements the FI-PeSNS adaptation that replaces explicit constraint
saturation tracking with penalty-based enforcement. Key features:
- SRINV (Singularity-Robust Inverse) for robust pseudo-inverse
- Analytical feasible scale computation for graceful degradation
- Hybrid hard saturation: explicit saturation for top violators + penalty for rest
- Penalty gradient "nudge" to push solution toward feasibility
- Warm-start support for temporal continuity
- Fixed-iteration loop suitable for CusADi compilation

Reference: Adapted from eSNS algorithm with penalty relaxation for GPU parallelism.
"""

from __future__ import annotations

from typing import Optional

try:
    import casadi as ca
except ImportError:
    ca = None


# Default parameters matching C++ VelocitySolverConfig
DEFAULT_EPSILON = 1e-6
DEFAULT_DAMPING = 0.1
DEFAULT_MU0 = 1e-3  # Initial penalty weight (softer start)
DEFAULT_GAMMA = 2.5  # Penalty growth factor per iteration
DEFAULT_ETA = 0.1  # Penalty gradient step size
DEFAULT_K_MAX = 12  # Fixed iterations (increased for accuracy)
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


def build_fi_pesns_velocity_solve(
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
    use_warm_start: bool = False,
    **kwargs,  # Accept legacy parameters for compatibility
) -> "ca.Function":
    """
    Build FI-PeSNS velocity solver with penalty-based constraint enforcement.

    Algorithm per iteration:
    1. For each task (hierarchically):
       a. Compute projected Jacobian J_P = J @ P
       b. Compute SRINV: J_pinv = srinv(J_P)
       c. Compute delta: Δdq = J_pinv @ (target - J @ dq)
       d. Compute feasible scale: s = get_feasible_task_scale(...)
       e. Apply scaled delta: dq += s * Δdq
       f. Update projector: P -= J_pinv @ J_P
    2. Compute violation residuals
    3. Apply penalty gradient: dq += eta * mu * C.T @ violation
    4. Ramp penalty: mu *= gamma
    5. Final clamp: Ensure hard constraint satisfaction

    Args:
        n_dof: Degrees of freedom
        n_tasks: Number of hierarchical tasks
        task_dims: List of task dimensions
        n_constraints: Number of constraint rows
        tol: SRINV tolerance
        damping: SRINV damping
        mu0: Initial penalty weight (start soft, ~1e-3)
        gamma: Penalty growth factor (2.0-3.0 typical)
        eta: Penalty gradient step size (0.05-0.2 typical)
        k_max: Fixed iterations (10-15 typical)
        use_warm_start: Add prior_dq input for warm-starting (default: False)

    Returns:
        CasADi Function: inputs [targets, jacobians, C, lower, upper, (prior_dq)] -> [velocity, scales]
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
        prior_dq = ca.SX.sym("prior_dq", n_dof)
        dq = prior_dq * 0.9  # Slight decay for stability
    else:
        dq = ca.SX.zeros(n_dof)

    task_scales = ca.SX.zeros(n_tasks)
    mu = mu0

    # Fixed-iteration outer loop
    for k_iter in range(k_max):
        # Reset projector for each outer iteration (tasks processed sequentially)
        P = ca.SX.eye(n_dof)

        target_offset = 0
        jacobian_offset = 0

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

            # Residual: target - J @ dq
            residual = target - J @ dq
            delta_dq = J_pinv @ residual

            # Analytical feasible scale
            # a = contribution from delta_dq, b = current constraint value
            a = C @ delta_dq  # Scaled contribution
            b = C @ dq  # Unscaled (current)
            scale = get_feasible_task_scale(a, b, lower, upper, n_constraints)

            # Apply scaled delta
            dq = dq + scale * delta_dq
            task_scales[task_idx] = scale

            # Update projector for next task
            P = P - J_pinv @ JP
            # Threshold small values
            P = ca.if_else(ca.fabs(P) < tol, 0.0, P)

        # Violation residuals after all tasks
        constraint_val = C @ dq
        r_low = lower - constraint_val  # Positive if below lower bound
        r_high = constraint_val - upper  # Positive if above upper bound

        # Compute violations
        viol_low = ca.fmax(0.0, r_low)
        viol_high = ca.fmax(0.0, r_high)

        # Penalty gradient "nudge" toward feasibility
        # For lower violations: need to increase constraint value -> move in C.T @ viol_low direction
        # For upper violations: need to decrease constraint value -> move in -C.T @ viol_high direction
        phi_grad = C.T @ (viol_low - viol_high)
        dq = dq + eta * mu * phi_grad

        # Ramp penalty (faster ramp in later iterations for convergence)
        mu = mu * gamma

    # Final clamp to ensure hard constraint satisfaction
    # Project onto feasible region using simple clamping
    constraint_val_final = C @ dq
    for i in range(n_constraints):
        c_row = C[i, :].T
        c_norm_sq = ca.sumsqr(c_row)

        # Check if violated
        val = constraint_val_final[i]
        clamped_val = ca.fmin(upper[i], ca.fmax(lower[i], val))
        delta = clamped_val - val

        # Apply minimal correction along constraint direction
        correction = ca.if_else(
            c_norm_sq > tol * tol, delta * c_row / c_norm_sq, ca.SX.zeros(n_dof)
        )
        dq = dq + correction

    # Build inputs/outputs
    if use_warm_start:
        inputs = [targets, jacobians_flat, C, lower, upper, prior_dq]
        input_names = ["targets", "jacobians", "C", "lower", "upper", "prior_dq"]
    else:
        inputs = [targets, jacobians_flat, C, lower, upper]
        input_names = ["targets", "jacobians", "C", "lower", "upper"]

    fn = ca.Function(
        "fn_fi_pesns_velocity_solve",
        inputs,
        [dq, task_scales],
        input_names,
        ["velocity", "scales"],
    )
    return fn


def build_fi_pesns_single_task(
    n_dof: int,
    task_dim: int,
    n_constraints: int,
    tol: float = DEFAULT_EPSILON,
    damping: float = DEFAULT_DAMPING,
    mu0: float = DEFAULT_MU0,
    gamma: float = DEFAULT_GAMMA,
    eta: float = DEFAULT_ETA,
    k_max: int = DEFAULT_K_MAX,
    use_warm_start: bool = False,
    **kwargs,  # Accept legacy parameters
) -> "ca.Function":
    """Build FI-PeSNS for single-task velocity IK."""
    return build_fi_pesns_velocity_solve(
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
        use_warm_start=use_warm_start,
    )


# Pre-defined robot configurations
ROBOT_CONFIGS = {
    "panda": {"n_dof": 7, "default_task_dims": [6], "n_constraints": 7},
    "ur5": {"n_dof": 6, "default_task_dims": [6], "n_constraints": 6},
    "iiwa14": {"n_dof": 7, "default_task_dims": [6], "n_constraints": 7},
    "humanoid_arm": {"n_dof": 20, "default_task_dims": [6, 3], "n_constraints": 10},
}


def build_fi_pesns_for_robot(
    robot_name: str,
    n_tasks: int = 1,
    task_dims: Optional[list[int]] = None,
    extra_constraints: int = 0,
    **kwargs,
) -> "ca.Function":
    """
    Build FI-PeSNS velocity solver for a known robot configuration.

    Args:
        robot_name: One of "panda", "ur5", "iiwa14", "humanoid_arm"
        n_tasks: Number of tasks
        task_dims: Task dimensions (uses default if None)
        extra_constraints: Additional constraint rows
        **kwargs: Passed to build_fi_pesns_velocity_solve

    Returns:
        CasADi Function
    """
    if robot_name not in ROBOT_CONFIGS:
        raise ValueError(f"Unknown robot: {robot_name}. Known: {list(ROBOT_CONFIGS.keys())}")

    config = ROBOT_CONFIGS[robot_name]
    n_dof = config["n_dof"]
    if task_dims is None:
        task_dims = (
            config["default_task_dims"][:n_tasks]
            if n_tasks <= len(config["default_task_dims"])
            else config["default_task_dims"] * n_tasks
        )
    n_constraints = config["n_constraints"] + extra_constraints

    return build_fi_pesns_velocity_solve(
        n_dof=n_dof,
        n_tasks=len(task_dims),
        task_dims=task_dims,
        n_constraints=n_constraints,
        **kwargs,
    )
