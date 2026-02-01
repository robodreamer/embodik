"""
CasADi symbolic implementation of velocity IK solver.

This module builds a CasADi Function that closely matches the C++ eSNS
algorithm using components from casadi_components: regularized inverse,
feasible scale computation, and single-objective step. Fixed N iterations
approximate the while-loop for constraint satisfaction.

For GPU acceleration, use CusADi to compile to CUDA kernels.
"""

from __future__ import annotations

from typing import Optional

try:
    import casadi as ca
except ImportError:
    ca = None

from embodik.gpu.casadi_components import (
    compute_regularized_inverse,
    compute_feasible_velocity_scale,
    compute_single_objective_step,
    compute_augmented_projector,
    compute_single_objective_step_with_saturation,
    DEFAULT_EPSILON,
    DEFAULT_REGULARIZATION_FACTOR,
    DEFAULT_MAGNITUDE_LIMIT,
)

# Fixed iterations for saturation loop (CasADi requires fixed structure)
N_SATURATION_ITERATIONS = 15


def build_velocity_solve_casadi(
    n_dof: int,
    n_tasks: int,
    task_dims: list[int],
    n_constraints: int,
    epsilon: float = DEFAULT_EPSILON,
    regularization_factor: float = DEFAULT_REGULARIZATION_FACTOR,
    magnitude_limit: float = DEFAULT_MAGNITUDE_LIMIT,
    n_iterations: int = N_SATURATION_ITERATIONS,
) -> "ca.Function":
    """
    Build a CasADi Function for velocity IK matching C++ eSNS structure.

    Uses regularized inverse, feasible scale per constraint, and fixed
    iterations for constraint satisfaction. Single task: iterative scale
    refinement. Multi-task: hierarchical null-space projection.

    Args:
        n_dof: Number of degrees of freedom (joint velocities)
        n_tasks: Number of hierarchical tasks
        task_dims: List of task dimensions [m_0, m_1, ...]
        n_constraints: Number of constraint rows (typically n_dof)
        epsilon: Numerical tolerance (matches C++ solver_config.epsilon)
        regularization_factor: Regularization factor for pseudo-inverse
        magnitude_limit: Max scaled contribution norm (C++ solver_config.magnitude_limit)
        n_iterations: Fixed iterations for scale/saturation loop

    Returns:
        CasADi Function with signature:
            inputs: [targets, jacobians, C, lower, upper]
            outputs: [velocity, scales]
    """
    if ca is None:
        raise RuntimeError("CasADi is required: pip install casadi")

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

    dq = ca.SX.zeros(n_dof)
    P = ca.SX.eye(n_dof)
    task_scales = ca.SX.zeros(n_tasks)

    target_offset = 0
    jacobian_offset = 0

    for task_idx in range(n_tasks):
        task_dim = task_dims[task_idx]
        target = targets[target_offset:target_offset + task_dim]
        target_offset += task_dim
        jac_size = task_dim * n_dof
        jac_flat = jacobians_flat[jacobian_offset:jacobian_offset + jac_size]
        jacobian_offset += jac_size
        # Row-major flat: flat[i*n_dof + j] = J[i,j]. CasADi reshape is column-major,
        # so reshape(n_dof, task_dim).T gives (task_dim, n_dof) with (i,j) = flat[i*n_dof + j].
        J = ca.reshape(jac_flat, n_dof, task_dim).T

        # Projected Jacobian and its regularized inverse (matches C++)
        JP = J @ P
        J_pinv = compute_regularized_inverse(
            JP, epsilon=epsilon, regularization_factor=regularization_factor
        )

        # Saturation loop: fixed N iterations (C++ while loop) with augmented projector
        scale = ca.SX.ones(1)
        sat_diag = ca.SX.zeros(n_constraints)
        sat_vals = ca.SX.zeros(n_constraints)
        dq_task = dq  # Solution after previous tasks

        for _ in range(n_iterations):
            # C_sat = diag(sat_diag) @ C (C++ saturated_constraint_matrix)
            C_sat = ca.diag(sat_diag) @ C
            P_aug = compute_augmented_projector(
                J_pinv, J, C_sat, P, epsilon=epsilon
            )
            # Unscaled trial (scale=1) with saturation correction (C++ lines 425-431)
            dq_trial = compute_single_objective_step_with_saturation(
                dq_task, J, P, target, ca.SX.ones(1),
                J_pinv, P_aug, sat_vals, C_sat,
            )
            # Constraint evaluation and contribution decomposition (C++ lines 434-447)
            constraint_eval = C @ dq_trial
            scaled_contribution = C @ J_pinv @ target
            unscaled_contribution = constraint_eval - scaled_contribution
            contribution_magnitude = ca.norm_2(scaled_contribution)
            scale_from_constraints = compute_feasible_velocity_scale(
                scaled_contribution,
                unscaled_contribution,
                lower,
                upper,
                sat_diag,
                n_constraints,
            )
            scale = ca.if_else(
                contribution_magnitude < epsilon,
                1.0,
                ca.if_else(
                    contribution_magnitude > magnitude_limit,
                    0.0,
                    scale_from_constraints,
                ),
            )
            # Scaled solution with saturation correction (C++ lines 496-499)
            dq_task = compute_single_objective_step_with_saturation(
                dq_task, J, P, target, scale,
                J_pinv, P_aug, sat_vals, C_sat,
            )
            # Violations on scaled solution (C++ lines 493-516)
            constraint_eval_scaled = C @ dq_task
            # Update saturation: when scale==1, saturate all violating constraints
            sat_diag_next = ca.SX.zeros(n_constraints)
            sat_vals_next = ca.SX.zeros(n_constraints)
            for i in range(n_constraints):
                violated = ca.if_else(
                    (constraint_eval_scaled[i] < lower[i] - epsilon)
                    + (constraint_eval_scaled[i] > upper[i] + epsilon)
                    > 0,
                    1.0,
                    0.0,
                )
                add_sat = ca.fmax(sat_diag[i], violated * ca.if_else(scale > 0.99, 1.0, 0.0))
                sat_diag_next[i] = add_sat
                sat_vals_next[i] = ca.if_else(
                    add_sat > 0.5,
                    ca.fmin(ca.fmax(constraint_eval_scaled[i], lower[i]), upper[i]),
                    sat_vals[i],
                )
            sat_diag = sat_diag_next
            sat_vals = sat_vals_next

        dq = dq_task
        task_scales[task_idx] = scale

        # Update null-space projector for next task (C++ lines 651-656)
        # N = N_prev - (J@N_prev)^# @ (J @ N_prev)
        P = P - J_pinv @ JP
        # Threshold small values
        P = ca.if_else(ca.fabs(P) < epsilon, 0.0, P)

    fn = ca.Function(
        "fn_velocity_solve",
        [targets, jacobians_flat, C, lower, upper],
        [dq, task_scales],
        ["targets", "jacobians", "C", "lower", "upper"],
        ["velocity", "scales"],
    )
    return fn


def build_velocity_solve_single_task(
    n_dof: int,
    task_dim: int,
    n_constraints: int,
    epsilon: float = DEFAULT_EPSILON,
    regularization_factor: float = DEFAULT_REGULARIZATION_FACTOR,
    n_iterations: int = N_SATURATION_ITERATIONS,
) -> "ca.Function":
    """
    Build a CasADi Function for single-task velocity IK.

    Args:
        n_dof: Number of degrees of freedom
        task_dim: Task dimension (e.g., 6 for SE3 task)
        n_constraints: Number of constraint rows
        epsilon: Numerical tolerance
        regularization_factor: Regularization factor
        n_iterations: Saturation loop iterations

    Returns:
        CasADi Function
    """
    return build_velocity_solve_casadi(
        n_dof=n_dof,
        n_tasks=1,
        task_dims=[task_dim],
        n_constraints=n_constraints,
        epsilon=epsilon,
        regularization_factor=regularization_factor,
        n_iterations=n_iterations,
    )


# Pre-defined configurations for common robots
ROBOT_CONFIGS = {
    "panda": {
        "n_dof": 7,
        "default_task_dims": [6],
        "n_constraints": 7,
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
        extra_constraints: Additional constraint rows
        **kwargs: Passed to build_velocity_solve_casadi

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
