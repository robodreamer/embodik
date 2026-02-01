"""
Unit tests for GPU solver components (CasADi building blocks).

Each component is tested in isolation before combining into the full solver.
"""

import pytest
import numpy as np
from typing import Tuple

# Component test tolerances
COMPONENT_ATOL = 1e-8  # Strict for component-level match
RTOL = 1e-5


def regularized_inverse_numpy(
    J: np.ndarray,
    epsilon: float = 1e-6,
    regularization_factor: float = 0.1,
) -> np.ndarray:
    """
    NumPy reference for compute_regularized_inverse (determinant-based + diagonal damping).

    Matches C++ logic: Gram matrix, determinant regularization, then J.T @ inv(G_reg).
    """
    gram = J @ J.T
    m = gram.shape[0]
    threshold_squared = epsilon * epsilon
    det_val = np.linalg.det(gram)
    if det_val < threshold_squared:
        reg = (1.0 - (det_val / threshold_squared) ** 2) * threshold_squared
    else:
        reg = 0.0
    gram_reg = gram + reg * np.eye(m) + regularization_factor * epsilon * np.eye(m)
    J_pinv = J.T @ np.linalg.inv(gram_reg)
    return J_pinv


def feasible_scale_numpy(
    bound_lower: float,
    bound_upper: float,
    coefficient: float,
    k_min: float = 1e-10,
    k_max: float = 1e10,
) -> Tuple[float, float]:
    """
    NumPy reference for ComputeFeasibleScalingRange (C++ lines 116-134).

    Returns (max_allowed_scale, min_allowed_scale) in [0,1] and [0,1].
    """
    max_allowed_scale = 1.0
    min_allowed_scale = 0.0
    if k_min < np.abs(coefficient) < k_max:
        if coefficient < 0.0 and bound_lower < 0.0 and coefficient <= bound_upper:
            max_allowed_scale = bound_lower / coefficient
            min_allowed_scale = bound_upper / coefficient
        elif coefficient > 0.0 and bound_upper > 0.0 and coefficient >= bound_lower:
            max_allowed_scale = bound_upper / coefficient
            min_allowed_scale = bound_lower / coefficient
    return (min(1.0, max_allowed_scale), max(0.0, min_allowed_scale))


# =============================================================================
# Component 1: Regularized Inverse
# =============================================================================


def test_regularized_inverse_matches_numpy():
    """CasADi regularized inverse should match NumPy reference (determinant + damping)."""
    try:
        import casadi as ca
        from embodik.gpu.casadi_components import (
            compute_regularized_inverse,
            DEFAULT_EPSILON,
            DEFAULT_REGULARIZATION_FACTOR,
        )
    except ImportError as e:
        pytest.skip(f"CasADi or components not available: {e}")

    rng = np.random.default_rng(42)
    for _ in range(5):
        m, n = 6, 7  # Underdetermined (task_dim x n_dof)
        J_np = rng.standard_normal((m, n)).astype(np.float64)
        # Ensure full row rank
        J_np = J_np + 0.1 * np.eye(m, n)

        J_sx = ca.SX.sym("J", m, n)
        J_pinv_sx = compute_regularized_inverse(
            J_sx,
            epsilon=DEFAULT_EPSILON,
            regularization_factor=DEFAULT_REGULARIZATION_FACTOR,
        )
        fn = ca.Function("J_pinv", [J_sx], [J_pinv_sx])
        J_pinv_ca = np.array(fn(J_np))

        J_pinv_np = regularized_inverse_numpy(
            J_np,
            epsilon=DEFAULT_EPSILON,
            regularization_factor=DEFAULT_REGULARIZATION_FACTOR,
        )
        np.testing.assert_allclose(J_pinv_ca, J_pinv_np, atol=COMPONENT_ATOL, rtol=RTOL)


def test_regularized_inverse_projection():
    """J_pinv @ J should act as identity on row space of J (for full row rank J)."""
    try:
        import casadi as ca
        from embodik.gpu.casadi_components import compute_regularized_inverse
    except ImportError as e:
        pytest.skip(f"CasADi or components not available: {e}")

    rng = np.random.default_rng(43)
    m, n = 6, 7
    J_np = rng.standard_normal((m, n)).astype(np.float64) + 0.2 * np.eye(m, n)
    x_np = J_np.T @ rng.standard_normal(m).astype(np.float64)  # x in row space of J

    J_sx = ca.SX.sym("J", m, n)
    x_sx = ca.SX.sym("x", n, 1)
    J_pinv_sx = compute_regularized_inverse(J_sx)
    # (J_pinv @ J) @ x should equal x when x is in row space
    proj_sx = J_pinv_sx @ J_sx @ x_sx
    fn = ca.Function("proj", [J_sx, x_sx], [proj_sx])
    proj_ca = np.array(fn(J_np, x_np)).ravel()
    np.testing.assert_allclose(proj_ca, x_np.ravel(), atol=1e-5, rtol=1e-4)


def test_regularized_inverse_simple_matches_full():
    """Simple (determinant-only) regularized inverse should match full when factor=0."""
    try:
        import casadi as ca
        from embodik.gpu.casadi_components import (
            compute_regularized_inverse,
            compute_regularized_inverse_simple,
        )
    except ImportError as e:
        pytest.skip(f"CasADi or components not available: {e}")

    rng = np.random.default_rng(44)
    m, n = 6, 7
    J_np = rng.standard_normal((m, n)).astype(np.float64) + 0.2 * np.eye(m, n)

    J_sx = ca.SX.sym("J", m, n)
    pinv_full = compute_regularized_inverse(J_sx, regularization_factor=0.0)
    pinv_simple = compute_regularized_inverse_simple(J_sx)
    fn_full = ca.Function("pinv_full", [J_sx], [pinv_full])
    fn_simple = ca.Function("pinv_simple", [J_sx], [pinv_simple])
    np.testing.assert_allclose(
        np.array(fn_full(J_np)),
        np.array(fn_simple(J_np)),
        atol=COMPONENT_ATOL,
        rtol=RTOL,
    )


# =============================================================================
# Component 2: Feasible Scale
# =============================================================================


def test_feasible_scale_edge_cases():
    """Feasible scale computation for positive/negative coefficient and tight bounds."""
    try:
        import casadi as ca
        from embodik.gpu.casadi_components import compute_feasible_scale
    except ImportError as e:
        pytest.skip(f"CasADi or components not available: {e}")

    # Build symbolic function
    lb_sx = ca.SX.sym("lb")
    ub_sx = ca.SX.sym("ub")
    c_sx = ca.SX.sym("c")
    scale_sx = compute_feasible_scale(lb_sx, ub_sx, c_sx)
    fn = ca.Function("feasible_scale", [lb_sx, ub_sx, c_sx], [scale_sx])

    test_cases = [
        # (lb, ub, c, expected_max_scale)
        (-1.0, 1.0, 0.5, 1.0),   # c>0, scale 1 achieves 0.5 in [-1,1]
        (-1.0, 1.0, 2.0, 0.5),   # c>0, scale 0.5 achieves 1.0 (upper bound)
        (-1.0, 1.0, -2.0, 0.5),  # c<0, scale 0.5 achieves -1.0 (lower bound)
        (-1.0, 1.0, -0.5, 1.0),  # c<0, scale 1 achieves -0.5 in [-1,1]
        (0.0, 1.0, 0.5, 1.0),
        (-2.0, 2.0, 1.0, 1.0),
    ]
    for lb, ub, c, expected in test_cases:
        scale_ca = float(np.asarray(fn(lb, ub, c)).item())
        max_scale_np, _ = feasible_scale_numpy(lb, ub, c)
        np.testing.assert_allclose(scale_ca, max_scale_np, atol=1e-6)
        np.testing.assert_allclose(scale_ca, expected, atol=1e-6)


def test_feasible_scale_clamped_to_unit():
    """Feasible scale must be in [0, 1]."""
    try:
        import casadi as ca
        from embodik.gpu.casadi_components import compute_feasible_scale
    except ImportError as e:
        pytest.skip(f"CasADi or components not available: {e}")

    lb_sx = ca.SX.sym("lb")
    ub_sx = ca.SX.sym("ub")
    c_sx = ca.SX.sym("c")
    scale_sx = compute_feasible_scale(lb_sx, ub_sx, c_sx)
    fn = ca.Function("feasible_scale", [lb_sx, ub_sx, c_sx], [scale_sx])

    rng = np.random.default_rng(45)
    for _ in range(20):
        lb = rng.uniform(-2, 0)
        ub = rng.uniform(0, 2)
        c = rng.uniform(-2, 2)
        scale = float(np.asarray(fn(lb, ub, c)).item())
        assert 0.0 <= scale <= 1.0, f"scale={scale} for lb={lb}, ub={ub}, c={c}"


# =============================================================================
# Component 3: Single Objective Solve Step
# =============================================================================


def test_single_objective_no_saturation():
    """Single task, no constraint violations: component step matches CPU solver."""
    try:
        import casadi as ca
        import embodik as eik
        from embodik.gpu.casadi_components import compute_single_objective_step
    except ImportError as e:
        pytest.skip(f"CasADi or embodik not available: {e}")

    # 2-DOF, 1 task (dim 2), loose bounds so no saturation
    n_dof = 2
    task_dim = 2
    rng = np.random.default_rng(46)
    J_np = rng.standard_normal((task_dim, n_dof)).astype(np.float64) + 0.5 * np.eye(task_dim, n_dof)
    target_np = rng.standard_normal(task_dim).astype(np.float64) * 0.3
    C = np.eye(n_dof, dtype=np.float64)
    lower = np.array([-10.0, -10.0], dtype=np.float64)
    upper = np.array([10.0, 10.0], dtype=np.float64)

    goals = [target_np]
    jacobians = [J_np]
    cpu_result = eik.computeMultiObjectiveVelocitySolutionEigen(
        goals, jacobians, C, lower, upper
    )
    cpu_dq = np.array(cpu_result.solution, dtype=np.float64)

    # Component: dq = 0 + J_pinv @ (1 * target - J @ 0) = J_pinv @ target
    dq_prev_sx = ca.SX.sym("dq_prev", n_dof, 1)
    J_sx = ca.SX.sym("J", task_dim, n_dof)
    P_sx = ca.SX.sym("P", n_dof, n_dof)
    target_sx = ca.SX.sym("target", task_dim, 1)
    scale_sx = ca.SX.sym("scale", 1, 1)
    dq_new_sx = compute_single_objective_step(
        dq_prev_sx, J_sx, P_sx, target_sx, scale_sx
    )
    fn = ca.Function(
        "single_step",
        [dq_prev_sx, J_sx, P_sx, target_sx, scale_sx],
        [dq_new_sx],
    )
    dq_prev_0 = np.zeros((n_dof, 1), dtype=np.float64)
    P_eye = np.eye(n_dof, dtype=np.float64)
    scale_1 = np.array([[1.0]], dtype=np.float64)
    comp_dq = np.array(fn(dq_prev_0, J_np, P_eye, target_np.reshape(-1, 1), scale_1)).ravel()

    np.testing.assert_allclose(comp_dq, cpu_dq, atol=1e-5, rtol=1e-4)


def test_single_objective_with_scale():
    """Single task with scale < 1: dq = J_pinv @ (scale * target)."""
    try:
        import casadi as ca
        from embodik.gpu.casadi_components import compute_single_objective_step
    except ImportError as e:
        pytest.skip(f"CasADi or components not available: {e}")

    n_dof = 2
    task_dim = 2
    rng = np.random.default_rng(47)
    J_np = rng.standard_normal((task_dim, n_dof)).astype(np.float64) + 0.5 * np.eye(task_dim, n_dof)
    target_np = rng.standard_normal(task_dim).astype(np.float64)
    dq_prev = np.zeros((n_dof, 1), dtype=np.float64)
    P = np.eye(n_dof, dtype=np.float64)
    scale = np.array([[0.5]], dtype=np.float64)

    dq_prev_sx = ca.SX.sym("dq_prev", n_dof, 1)
    J_sx = ca.SX.sym("J", task_dim, n_dof)
    P_sx = ca.SX.sym("P", n_dof, n_dof)
    target_sx = ca.SX.sym("target", task_dim, 1)
    scale_sx = ca.SX.sym("scale", 1, 1)
    dq_new_sx = compute_single_objective_step(
        dq_prev_sx, J_sx, P_sx, target_sx, scale_sx
    )
    fn = ca.Function(
        "single_step",
        [dq_prev_sx, J_sx, P_sx, target_sx, scale_sx],
        [dq_new_sx],
    )
    comp_dq = np.array(fn(dq_prev, J_np, P, target_np.reshape(-1, 1), scale)).ravel()

    # Numerically: dq = J_pinv @ (0.5 * target)
    J_pinv_np = regularized_inverse_numpy(J_np)
    expected_dq = (J_pinv_np @ (0.5 * target_np)).ravel()
    np.testing.assert_allclose(comp_dq, expected_dq, atol=1e-6, rtol=1e-5)


# =============================================================================
# Component 4: Constraint Saturation (velocity scale from constraints)
# =============================================================================


def test_feasible_velocity_scale_no_saturation():
    """When no constraints are saturated, scale = min of per-constraint feasible scales."""
    try:
        import casadi as ca
        from embodik.gpu.casadi_components import compute_feasible_velocity_scale
    except ImportError as e:
        pytest.skip(f"CasADi or components not available: {e}")

    n_constraints = 3
    rng = np.random.default_rng(48)
    scaled_contrib = rng.standard_normal(n_constraints).astype(np.float64) * 0.5
    unscaled = rng.standard_normal(n_constraints).astype(np.float64) * 0.1
    lower = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    upper = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    sat_diag = np.zeros(n_constraints, dtype=np.float64)  # none saturated

    sc_sx = ca.SX.sym("sc", n_constraints, 1)
    unsc_sx = ca.SX.sym("unsc", n_constraints, 1)
    lo_sx = ca.SX.sym("lo", n_constraints, 1)
    hi_sx = ca.SX.sym("hi", n_constraints, 1)
    sat_sx = ca.SX.sym("sat", n_constraints, 1)
    scale_sx = compute_feasible_velocity_scale(
        sc_sx, unsc_sx, lo_sx, hi_sx, sat_sx, n_constraints
    )
    fn = ca.Function(
        "velocity_scale",
        [sc_sx, unsc_sx, lo_sx, hi_sx, sat_sx],
        [scale_sx],
    )
    scale_ca = float(np.asarray(fn(scaled_contrib, unscaled, lower, upper, sat_diag)).item())

    # Reference: min of feasible_scale(lower[i]-unscaled[i], upper[i]-unscaled[i], scaled_contrib[i])
    expected_scales = []
    for i in range(n_constraints):
        max_s, _ = feasible_scale_numpy(
            lower[i] - unscaled[i], upper[i] - unscaled[i], scaled_contrib[i]
        )
        expected_scales.append(max_s)
    expected = min(expected_scales)
    np.testing.assert_allclose(scale_ca, expected, atol=1e-6)
    assert 0.0 <= scale_ca <= 1.0


# =============================================================================
# Component 5: Generalized inverse and augmented projector (saturation)
# =============================================================================


def test_generalized_inverse_matches_numpy():
    """compute_generalized_inverse matches NumPy pinv for full-rank and zero matrix."""
    try:
        import casadi as ca
        from embodik.gpu.casadi_components import compute_generalized_inverse
    except ImportError as e:
        pytest.skip(f"CasADi or components not available: {e}")

    rng = np.random.default_rng(49)
    # Full-rank (3 x 5)
    M_np = rng.standard_normal((3, 5)).astype(np.float64)
    M_sx = ca.SX.sym("M", 3, 5)
    pinv_sx = compute_generalized_inverse(M_sx, epsilon=1e-8)
    fn = ca.Function("pinv", [M_sx], [pinv_sx])
    pinv_ca = np.array(fn(M_np))
    pinv_np = np.linalg.pinv(M_np)
    np.testing.assert_allclose(pinv_ca, pinv_np, atol=1e-5, rtol=1e-4)

    # Zero matrix (same shape): should return zero
    M_zero = np.zeros((3, 5), dtype=np.float64)
    pinv_zero = np.array(fn(M_zero))
    np.testing.assert_allclose(pinv_zero, 0.0, atol=1e-10)


def test_augmented_projector_zero_saturation():
    """When C_sat is zero, augmented projector is zero and step reduces to no-saturation step."""
    try:
        import casadi as ca
        from embodik.gpu.casadi_components import (
            compute_augmented_projector,
            compute_single_objective_step,
            compute_single_objective_step_with_saturation,
            compute_regularized_inverse,
        )
    except ImportError as e:
        pytest.skip(f"CasADi or components not available: {e}")

    n_dof, task_dim, n_constraints = 3, 2, 3
    rng = np.random.default_rng(50)
    J_np = rng.standard_normal((task_dim, n_dof)).astype(np.float64) + 0.5 * np.eye(task_dim, n_dof)
    P_np = np.eye(n_dof, dtype=np.float64)
    C_sat_zero = np.zeros((n_constraints, n_dof), dtype=np.float64)
    dq_prev = np.zeros((n_dof,), dtype=np.float64)
    target = rng.standard_normal(task_dim).astype(np.float64) * 0.3
    scale = np.array([1.0])

    dq_prev_sx = ca.SX.sym("dq", n_dof)
    J_sx = ca.SX.sym("J", task_dim, n_dof)
    P_sx = ca.SX.sym("P", n_dof, n_dof)
    target_sx = ca.SX.sym("t", task_dim)
    scale_sx = ca.SX.sym("s", 1)
    C_sat_sx = ca.SX.sym("C_sat", n_constraints, n_dof)
    sat_vals_sx = ca.SX.sym("sat_vals", n_constraints)

    J_pinv_sx = compute_regularized_inverse(J_sx @ P_sx)
    P_aug_sx = compute_augmented_projector(J_pinv_sx, J_sx, C_sat_sx, P_sx)
    dq_no_sat = compute_single_objective_step(dq_prev_sx, J_sx, P_sx, target_sx, scale_sx)
    dq_with_sat = compute_single_objective_step_with_saturation(
        dq_prev_sx, J_sx, P_sx, target_sx, scale_sx,
        J_pinv_sx, P_aug_sx, sat_vals_sx, C_sat_sx,
    )
    fn_no = ca.Function("step", [dq_prev_sx, J_sx, P_sx, target_sx, scale_sx], [dq_no_sat])
    fn_with = ca.Function(
        "step_sat",
        [dq_prev_sx, J_sx, P_sx, target_sx, scale_sx, sat_vals_sx, C_sat_sx],
        [dq_with_sat],
    )

    dq_no = np.array(fn_no(dq_prev, J_np, P_np, target, scale)).ravel()
    sat_vals_zero = np.zeros(n_constraints, dtype=np.float64)
    dq_with = np.array(fn_with(dq_prev, J_np, P_np, target, scale, sat_vals_zero, C_sat_zero)).ravel()
    np.testing.assert_allclose(dq_no, dq_with, atol=1e-10, err_msg="With C_sat=0, step with saturation should match step without")
