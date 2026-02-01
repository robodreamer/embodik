"""
CasADi symbolic implementation of hierarchical velocity IK solver.

This module builds a CasADi Function that implements a fixed-iteration
approximation of EmbodiK's hierarchical null-space projection solver.

The C++ solver uses while-loops with data-dependent saturation. CasADi
requires fixed iteration counts, so we unroll N iterations with soft
saturation approximations.
"""

from __future__ import annotations

from typing import Optional, Tuple

try:
    import casadi as ca
except ImportError:
    ca = None


def _regularized_inverse(
    J: "ca.SX",
    P: "ca.SX",
    epsilon: float = 1e-6,
    damping: float = 1e-6,
) -> "ca.SX":
    """
    Compute regularized pseudo-inverse of J @ P.
    
    Uses damped least squares: (J P)^+ = (J P)^T @ inv((J P)(J P)^T + lambda*I)
    
    Args:
        J: Task Jacobian (m x n)
        P: Null-space projector (n x n)
        epsilon: Threshold for singular value damping
        damping: Damping factor for regularization
        
    Returns:
        Regularized pseudo-inverse (n x m)
    """
    JP = J @ P
    m = JP.shape[0]
    
    # Gram matrix
    gram = JP @ JP.T
    
    # Add damping for regularization
    gram_reg = gram + damping * ca.SX.eye(m)
    
    # Pseudo-inverse via regularized Gram
    return JP.T @ ca.inv(gram_reg)


def _compute_feasible_scale(
    constraint_eval: "ca.SX",
    contribution: "ca.SX", 
    lower: "ca.SX",
    upper: "ca.SX",
    epsilon: float = 1e-10,
) -> "ca.SX":
    """
    Compute feasible scaling factor for velocity contribution.
    
    For each constraint i, find the maximum scale s in [0, 1] such that:
        lower[i] <= constraint_eval[i] + s * contribution[i] <= upper[i]
    
    Uses soft min/max to make it differentiable.
    
    Args:
        constraint_eval: Current constraint values (k,)
        contribution: Velocity contribution to constraints (k,)
        lower: Lower bounds (k,)
        upper: Upper bounds (k,)
        epsilon: Small value to avoid division by zero
        
    Returns:
        Feasible scale in [0, 1]
    """
    k = constraint_eval.shape[0]
    
    # For each constraint, compute the scale that would hit the bound
    # scale = (bound - current) / contribution
    
    # Initialize with 1.0 (full scale)
    scale = ca.SX.ones(1)
    
    for i in range(k):
        c = contribution[i]
        current = constraint_eval[i]
        lb = lower[i]
        ub = upper[i]
        
        # Compute scale for lower bound violation
        # If c < 0 and current + s*c < lb, then s < (lb - current) / c
        margin_lower = lb - current
        scale_lower = ca.if_else(
            ca.fabs(c) > epsilon,
            ca.if_else(c < 0, margin_lower / c, 1.0),
            1.0
        )
        
        # Compute scale for upper bound violation
        # If c > 0 and current + s*c > ub, then s < (ub - current) / c
        margin_upper = ub - current
        scale_upper = ca.if_else(
            ca.fabs(c) > epsilon,
            ca.if_else(c > 0, margin_upper / c, 1.0),
            1.0
        )
        
        # Take minimum of both
        scale_i = ca.fmin(scale_lower, scale_upper)
        
        # Clamp to [0, 1]
        scale_i = ca.fmax(0.0, ca.fmin(1.0, scale_i))
        
        # Take minimum across all constraints
        scale = ca.fmin(scale, scale_i)
    
    return scale


def _soft_clamp(x: "ca.SX", lower: "ca.SX", upper: "ca.SX") -> "ca.SX":
    """Soft clamp using fmin/fmax."""
    return ca.fmax(lower, ca.fmin(upper, x))


def build_velocity_solve_casadi(
    n_dof: int,
    n_tasks: int,
    task_dims: list[int],
    n_constraints: int,
    n_iterations: int = 10,
    epsilon: float = 1e-6,
    damping: float = 1e-6,
) -> "ca.Function":
    """
    Build a CasADi Function for hierarchical velocity IK.
    
    This implements a fixed-iteration approximation of EmbodiK's
    null-space projection solver with constraint saturation.
    
    Args:
        n_dof: Number of degrees of freedom (joint velocities)
        n_tasks: Number of hierarchical tasks
        task_dims: List of task dimensions [m_0, m_1, ..., m_{n_tasks-1}]
        n_constraints: Number of constraint rows (typically n_dof + additional)
        n_iterations: Number of saturation iterations per task (default 10)
        epsilon: Numerical threshold
        damping: Regularization damping factor
        
    Returns:
        CasADi Function with signature:
            inputs: [objective_targets_flat, objective_jacobians_flat, 
                     constraint_matrix, min_bounds, max_bounds]
            outputs: [velocity_solution, task_scales]
    """
    if ca is None:
        raise RuntimeError("CasADi is required: pip install casadi")
    
    if len(task_dims) != n_tasks:
        raise ValueError(f"task_dims length {len(task_dims)} != n_tasks {n_tasks}")
    
    total_task_dim = sum(task_dims)
    
    # Define symbolic inputs
    # Flattened objective targets: [m_0 + m_1 + ... + m_{n_tasks-1}]
    objective_targets_flat = ca.SX.sym("objective_targets", total_task_dim)
    
    # Flattened Jacobians: [m_0*n_dof + m_1*n_dof + ...]
    total_jacobian_size = sum(d * n_dof for d in task_dims)
    objective_jacobians_flat = ca.SX.sym("jacobians", total_jacobian_size)
    
    # Constraint matrix: (n_constraints x n_dof)
    constraint_matrix = ca.SX.sym("C", n_constraints, n_dof)
    
    # Bounds
    min_bounds = ca.SX.sym("min_bounds", n_constraints)
    max_bounds = ca.SX.sym("max_bounds", n_constraints)
    
    # Initialize solution
    dq = ca.SX.zeros(n_dof)
    
    # Initialize null-space projector as identity
    P = ca.SX.eye(n_dof)
    
    # Task scales output
    task_scales = ca.SX.zeros(n_tasks)
    
    # Extract individual task targets and Jacobians
    target_offset = 0
    jacobian_offset = 0
    
    for task_idx in range(n_tasks):
        task_dim = task_dims[task_idx]
        
        # Extract target for this task
        target = objective_targets_flat[target_offset:target_offset + task_dim]
        target_offset += task_dim
        
        # Extract and reshape Jacobian for this task
        jac_size = task_dim * n_dof
        jacobian_flat = objective_jacobians_flat[jacobian_offset:jacobian_offset + jac_size]
        jacobian_offset += jac_size
        J = ca.reshape(jacobian_flat, task_dim, n_dof)
        
        # Store previous state for saturation loop
        dq_prev = dq
        P_constrained = P
        
        # Saturation iteration loop (fixed iterations)
        scale = ca.SX.ones(1)
        
        for _ in range(n_iterations):
            # Compute regularized inverse in constrained null-space
            J_pinv = _regularized_inverse(J, P_constrained, epsilon, damping)
            
            # Compute velocity contribution
            residual = target - J @ dq_prev
            delta_dq = J_pinv @ residual
            
            # Trial solution
            dq_trial = dq_prev + scale * delta_dq
            
            # Check constraint satisfaction
            constraint_eval = constraint_matrix @ dq_trial
            
            # Compute contribution to constraints from this velocity
            contribution = constraint_matrix @ delta_dq
            
            # Compute feasible scale
            new_scale = _compute_feasible_scale(
                constraint_matrix @ dq_prev,
                contribution,
                min_bounds,
                max_bounds,
                epsilon
            )
            
            # Update scale (take minimum)
            scale = ca.fmin(scale, new_scale)
            
            # Soft clamp scale to [0, 1]
            scale = ca.fmax(0.0, ca.fmin(1.0, scale))
        
        # Apply scaled velocity
        J_pinv = _regularized_inverse(J, P, epsilon, damping)
        dq = dq_prev + scale * (J_pinv @ (target - J @ dq_prev))
        
        # Store task scale
        task_scales[task_idx] = scale
        
        # Update null-space projector for next task
        # P_new = P - J_pinv @ J @ P
        JP = J @ P
        JP_pinv = _regularized_inverse(J, P, epsilon, damping)
        P = P - JP_pinv @ JP
        
        # Threshold small values to zero
        P = ca.if_else(ca.fabs(P) < epsilon, 0.0, P)
    
    # Final constraint clamping (soft)
    final_eval = constraint_matrix @ dq
    for i in range(n_constraints):
        violation_lower = min_bounds[i] - final_eval[i]
        violation_upper = final_eval[i] - max_bounds[i]
        
        # If violated, scale down proportionally
        max_violation = ca.fmax(violation_lower, violation_upper)
        correction_needed = ca.fmax(0.0, max_violation)
        
        # Apply soft correction (reduce velocity magnitude)
        correction_factor = ca.if_else(
            correction_needed > epsilon,
            1.0 / (1.0 + correction_needed),
            1.0
        )
        dq = dq * correction_factor
    
    # Build CasADi Function
    fn = ca.Function(
        "fn_velocity_solve",
        [objective_targets_flat, objective_jacobians_flat, 
         constraint_matrix, min_bounds, max_bounds],
        [dq, task_scales],
        ["targets", "jacobians", "C", "lower", "upper"],
        ["velocity", "scales"]
    )
    
    return fn


def build_velocity_solve_single_task(
    n_dof: int,
    task_dim: int,
    n_constraints: int,
    n_iterations: int = 10,
    epsilon: float = 1e-6,
    damping: float = 1e-6,
) -> "ca.Function":
    """
    Build a simplified CasADi Function for single-task velocity IK.
    
    This is a simpler version for testing and benchmarking with a single task.
    
    Args:
        n_dof: Number of degrees of freedom
        task_dim: Task dimension (e.g., 6 for SE3 task)
        n_constraints: Number of constraint rows
        n_iterations: Number of saturation iterations
        epsilon: Numerical threshold
        damping: Regularization damping
        
    Returns:
        CasADi Function
    """
    return build_velocity_solve_casadi(
        n_dof=n_dof,
        n_tasks=1,
        task_dims=[task_dim],
        n_constraints=n_constraints,
        n_iterations=n_iterations,
        epsilon=epsilon,
        damping=damping,
    )


# Pre-defined configurations for common robots
ROBOT_CONFIGS = {
    "panda": {
        "n_dof": 7,
        "default_task_dims": [6],  # Single EE task
        "n_constraints": 7,  # Joint velocity limits
    },
    "ur5": {
        "n_dof": 6,
        "default_task_dims": [6],
        "n_constraints": 6,
    },
    "iiwa14": {
        "n_dof": 7,
        "default_task_dims": [6],
        "n_constraints": 7,
    },
}


def build_for_robot(
    robot_name: str,
    n_tasks: int = 1,
    task_dims: Optional[list[int]] = None,
    extra_constraints: int = 0,
    **kwargs,
) -> "ca.Function":
    """
    Build a CasADi velocity solver for a known robot configuration.
    
    Args:
        robot_name: One of "panda", "ur5", "iiwa14"
        n_tasks: Number of tasks
        task_dims: Task dimensions (uses default if None)
        extra_constraints: Additional constraint rows beyond joint limits
        **kwargs: Additional arguments to build_velocity_solve_casadi
        
    Returns:
        CasADi Function
    """
    if robot_name not in ROBOT_CONFIGS:
        raise ValueError(f"Unknown robot: {robot_name}. Known: {list(ROBOT_CONFIGS.keys())}")
    
    config = ROBOT_CONFIGS[robot_name]
    n_dof = config["n_dof"]
    
    if task_dims is None:
        task_dims = config["default_task_dims"] * n_tasks
    
    n_constraints = config["n_constraints"] + extra_constraints
    
    return build_velocity_solve_casadi(
        n_dof=n_dof,
        n_tasks=len(task_dims),
        task_dims=task_dims,
        n_constraints=n_constraints,
        **kwargs,
    )
