"""
Tests for FI-PeSNS (Fixed-Iteration Penalized eSNS) GPU solver.

Tests validate:
1. SRINV matches regularized inverse
2. Feasible scale computation
3. FI-PeSNS vs CPU solver accuracy
4. Constraint satisfaction
"""

from typing import List, Tuple

import numpy as np
import pytest

# Test tolerances
VELOCITY_ATOL = (
    0.4  # FI-PeSNS vs CPU (penalty approximation allows gap; focus is constraint satisfaction)
)
SIMPLE_ATOL = 1e-6  # No-saturation case
CONSTRAINT_TOL = 1e-3  # Constraint satisfaction tolerance (primary goal)


def generate_random_problem(
    seed: int,
    n_dof: int = 7,
    n_tasks: int = 1,
    task_dims: List[int] = None,
    n_constraints: int = None,
    bound_scale: float = 1.0,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Generate a random IK problem."""
    rng = np.random.default_rng(seed)

    if task_dims is None:
        task_dims = [min(6, n_dof - 1) for _ in range(n_tasks)]

    if n_constraints is None:
        n_constraints = n_dof

    goals = []
    jacobians = []

    for task_dim in task_dims:
        goal = rng.uniform(-0.5, 0.5, task_dim)
        jacobian = rng.uniform(-1.0, 1.0, (task_dim, n_dof))
        goals.append(goal.astype(np.float64))
        jacobians.append(np.asfortranarray(jacobian.astype(np.float64)))

    C = np.asfortranarray(np.eye(n_constraints, n_dof, dtype=np.float64))
    lower = np.full(n_constraints, -bound_scale, dtype=np.float64)
    upper = np.full(n_constraints, bound_scale, dtype=np.float64)

    return goals, jacobians, C, lower, upper


# =============================================================================
# SRINV Tests
# =============================================================================


def _extended_srinv_reference(
    matrix: np.ndarray,
    *,
    epsilon: float = 1e-6,
    damping: float = 0.1,
) -> np.ndarray:
    gram = matrix @ matrix.T
    threshold_squared = epsilon**2
    determinant = float(np.linalg.det(gram))
    global_regularization = (
        (1.0 - (determinant / threshold_squared) ** 2) * threshold_squared
        if determinant < threshold_squared
        else 0.0
    )
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=True)
    per_value_damping = damping * np.maximum(
        1.0 - np.minimum(singular_values / epsilon, 1.0) ** 2,
        0.0,
    )
    regularized = (
        gram
        + global_regularization * np.eye(gram.shape[0])
        + left @ np.diag(per_value_damping) @ left.T
    )
    return matrix.T @ np.linalg.inv(regularized)


def _rotated_matrix(
    singular_values: np.ndarray,
    *,
    velocity_dimension: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    task_dimension = singular_values.size
    left = np.linalg.qr(rng.standard_normal((task_dimension, task_dimension)))[0]
    right = np.linalg.qr(rng.standard_normal((velocity_dimension, task_dimension)))[0]
    return left @ np.diag(singular_values) @ right.T, left, right


def test_gpu_srinv_defaults_match_public_kinematics_solver():
    from embodik.gpu.casadi_fi_pesns import DEFAULT_DAMPING as FI_DAMPING
    from embodik.gpu.casadi_fi_pesns import DEFAULT_EPSILON as FI_EPSILON
    from embodik.gpu.casadi_pph_sns import DEFAULT_DAMPING as PPH_DAMPING
    from embodik.gpu.casadi_pph_sns import DEFAULT_EPSILON as PPH_EPSILON

    assert (FI_EPSILON, FI_DAMPING) == (0.1, 0.1)
    assert (PPH_EPSILON, PPH_DAMPING) == (0.1, 0.1)


def test_srinv_matches_numpy_pinv():
    """SRINV should match NumPy pseudo-inverse for well-conditioned matrices."""
    try:
        import casadi as ca

        from embodik.gpu.casadi_fi_pesns import srinv
    except ImportError as e:
        pytest.skip(f"CasADi or modules not available: {e}")

    rng = np.random.default_rng(100)
    m, n = 4, 7
    A_np = rng.standard_normal((m, n)).astype(np.float64)

    A_sx = ca.SX.sym("A", m, n)
    srinv_sx = srinv(A_sx, tol=1e-6, damping=0.1)
    fn_srinv = ca.Function("srinv", [A_sx], [srinv_sx])

    srinv_result = np.array(fn_srinv(A_np))
    pinv_np = np.linalg.pinv(A_np)

    # With small damping, should be close to NumPy pinv
    np.testing.assert_allclose(srinv_result, pinv_np, atol=1e-4, rtol=1e-3)


def test_srinv_near_singular():
    """SRINV handles near-singular matrices gracefully."""
    try:
        import casadi as ca

        from embodik.gpu.casadi_fi_pesns import srinv
    except ImportError as e:
        pytest.skip(f"CasADi or modules not available: {e}")

    # Near-singular: repeated row
    A_np = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0 + 1e-8, 0.0, 0.0],  # Nearly identical to first row
        ],
        dtype=np.float64,
    )

    A_sx = ca.SX.sym("A", 2, 3)
    srinv_sx = srinv(A_sx, tol=1e-6, damping=0.1)
    fn = ca.Function("srinv", [A_sx], [srinv_sx])

    result = np.array(fn(A_np))
    assert np.all(np.isfinite(result)), "SRINV should handle near-singular matrix"


def test_srinv_does_not_damp_full_rank_singular_values_above_threshold():
    """The symbolic path should match CPU SRINV when spectral damping is inactive."""
    try:
        import casadi as ca

        from embodik.gpu.casadi_fi_pesns import srinv
    except ImportError as e:
        pytest.skip(f"CasADi or modules not available: {e}")

    singular_values = np.array([1.0, 0.3, 0.0032575734])
    matrix = np.diag(singular_values)
    symbolic = ca.SX.sym("matrix", 3, 3)
    function = ca.Function(
        "srinv_above_threshold",
        [symbolic],
        [srinv(symbolic, tol=1e-6, damping=0.1)],
    )

    np.testing.assert_allclose(
        np.asarray(function(matrix)),
        np.linalg.inv(matrix),
        rtol=1e-10,
        atol=1e-10,
    )


def test_srinv_threshold_crossing_damping_is_finite_and_bounded():
    try:
        import casadi as ca

        from embodik.gpu.casadi_fi_pesns import srinv
    except ImportError as e:
        pytest.skip(f"CasADi or modules not available: {e}")

    matrix = np.diag([1.0, 0.75e-6, 1e-9])
    symbolic = ca.SX.sym("matrix", 3, 3)
    function = ca.Function(
        "srinv_threshold_crossing",
        [symbolic],
        [srinv(symbolic, tol=1e-6, damping=0.1)],
    )
    result = np.asarray(function(matrix))

    assert np.all(np.isfinite(result))
    assert np.linalg.norm(result) < 2.0


@pytest.mark.parametrize(
    ("singular_values", "seed", "relative_tolerance"),
    [
        (np.array([1.0, 0.3, 0.0032575734]), 301, 5e-5),
        (np.array([1.0, 0.75e-6, 1e-9]), 302, 5e-5),
        (np.array([1.0, 0.5e-6, 0.25e-6]), 303, 5e-5),
    ],
)
def test_srinv_matches_extended_cpu_law_for_rotated_spectra(
    singular_values: np.ndarray,
    seed: int,
    relative_tolerance: float,
):
    try:
        import casadi as ca

        from embodik.gpu.casadi_fi_pesns import srinv
    except ImportError as e:
        pytest.skip(f"CasADi or modules not available: {e}")

    matrix, _, _ = _rotated_matrix(
        singular_values,
        velocity_dimension=5,
        seed=seed,
    )
    symbolic = ca.SX.sym("rotated_matrix", *matrix.shape)
    function = ca.Function("rotated_srinv", [symbolic], [srinv(symbolic, tol=1e-6, damping=0.1)])

    np.testing.assert_allclose(
        np.asarray(function(matrix)),
        _extended_srinv_reference(matrix),
        rtol=relative_tolerance,
        atol=1e-8,
    )


def test_srinv_threshold_straddle_preserves_healthy_action():
    """Near-equal singular values that straddle epsilon mix the null subspace.

    Per-mode damping is discontinuous at ``tol``, so a tiny left-basis rotation
    swaps a damped direction for an undamped one. CasADi's fixed Jacobi SVD and
    NumPy's SVD can therefore disagree on the inverse matrix across BLAS builds
    even though both apply the same extended-SRINV law. The well-conditioned
    task direction must still match and must not leak into the near-null joint
    subspace.
    """
    try:
        import casadi as ca

        from embodik.gpu.casadi_fi_pesns import srinv
    except ImportError as e:
        pytest.skip(f"CasADi or modules not available: {e}")

    matrix, left, right = _rotated_matrix(
        np.array([1.0, 0.999e-6, 1.001e-6]),
        velocity_dimension=5,
        seed=304,
    )
    symbolic = ca.SX.sym("straddle_matrix", *matrix.shape)
    gpu = np.asarray(
        ca.Function("straddle_srinv", [symbolic], [srinv(symbolic, tol=1e-6, damping=0.1)])(matrix)
    )
    cpu = _extended_srinv_reference(matrix)
    healthy = left[:, 0]
    gpu_healthy = gpu @ healthy

    assert np.isfinite(gpu).all()
    np.testing.assert_allclose(gpu_healthy, cpu @ healthy, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(gpu_healthy, right[:, 0], rtol=1e-4, atol=1e-4)
    assert float(right[:, 0] @ gpu_healthy) == pytest.approx(1.0, rel=1e-4)
    np.testing.assert_allclose(right[:, 1:].T @ gpu_healthy, 0.0, atol=1e-4)


def test_srinv_preserves_healthy_motion_while_damping_singular_directions():
    try:
        import casadi as ca

        from embodik.gpu.casadi_fi_pesns import srinv
    except ImportError as e:
        pytest.skip(f"CasADi or modules not available: {e}")

    matrix, left, right = _rotated_matrix(
        np.array([1.0, 0.75e-6, 1e-9]),
        velocity_dimension=5,
        seed=305,
    )
    symbolic = ca.SX.sym("directional_matrix", *matrix.shape)
    function = ca.Function(
        "directional_srinv",
        [symbolic],
        [srinv(symbolic, tol=1e-6, damping=0.1)],
    )
    inverse = np.asarray(function(matrix))
    gains = right.T @ inverse @ left

    assert gains[0, 0] == pytest.approx(1.0, rel=1e-10)
    assert abs(gains[1, 1]) < 1e-4
    assert abs(gains[2, 2]) < 1e-7
    np.testing.assert_allclose(
        gains - np.diag(np.diag(gains)),
        0.0,
        atol=1e-8,
    )


def test_default_srinv_matches_cpu_extended_law_in_rotated_directions():
    """GPU defaults must reproduce CPU extended SRINV, not a nearby variant."""
    try:
        import casadi as ca

        import embodik as eik
        from embodik.gpu.casadi_fi_pesns import srinv
    except ImportError as e:
        pytest.skip(f"CasADi or modules not available: {e}")

    matrix, _, _ = _rotated_matrix(
        np.array([1.7, 0.8, 0.3, 0.12, 0.075, 0.002]),
        velocity_dimension=7,
        seed=711,
    )
    rng = np.random.default_rng(712)
    goal = rng.uniform(-0.2, 0.2, matrix.shape[0])
    symbolic = ca.SX.sym("default_directional_matrix", *matrix.shape)
    gpu_inverse = ca.Function(
        "default_directional_srinv",
        [symbolic],
        [srinv(symbolic)],
    )

    cpu_result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [goal],
        [matrix],
        np.eye(matrix.shape[1]),
        np.full(matrix.shape[1], -100.0),
        np.full(matrix.shape[1], 100.0),
        sr_tolerance=0.1,
        sr_damping=0.1,
    )
    gpu_solution = np.asarray(gpu_inverse(matrix)) @ goal

    assert cpu_result.status == eik.SolverStatus.SUCCESS
    np.testing.assert_allclose(
        gpu_solution,
        np.asarray(cpu_result.solution),
        rtol=5e-12,
        atol=5e-12,
    )
    np.testing.assert_allclose(
        np.asarray(gpu_inverse(matrix)),
        _extended_srinv_reference(matrix, epsilon=0.1, damping=0.1),
        rtol=5e-12,
        atol=5e-12,
    )


def test_default_fi_task_pass_matches_cpu_extended_srinv_with_loose_bounds():
    """Fixed penalty iterations must not repeatedly weaken CPU SRINV damping."""
    try:
        import embodik as eik
        from embodik.gpu.casadi_fi_pesns import build_fi_pesns_single_task
    except ImportError as e:
        pytest.skip(f"EmbodiK or CasADi not available: {e}")

    matrix, _, _ = _rotated_matrix(
        np.array([1.7, 0.8, 0.3, 0.12, 0.075, 0.002]),
        velocity_dimension=7,
        seed=713,
    )
    goal = np.random.default_rng(714).uniform(-0.2, 0.2, matrix.shape[0])
    lower = np.full(matrix.shape[1], -100.0)
    upper = np.full(matrix.shape[1], 100.0)

    cpu_result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [goal],
        [matrix],
        np.eye(matrix.shape[1]),
        lower,
        upper,
        sr_tolerance=0.1,
        sr_damping=0.1,
    )
    gpu_function = build_fi_pesns_single_task(
        n_dof=matrix.shape[1],
        task_dim=matrix.shape[0],
        n_constraints=matrix.shape[1],
        k_max=2,
    )
    gpu_solution, gpu_scales = gpu_function(
        goal,
        matrix.flatten(),
        np.eye(matrix.shape[1]),
        lower,
        upper,
    )

    assert cpu_result.status == eik.SolverStatus.SUCCESS
    np.testing.assert_allclose(
        np.asarray(gpu_solution).ravel(),
        np.asarray(cpu_result.solution),
        rtol=5e-12,
        atol=5e-12,
    )
    np.testing.assert_allclose(np.asarray(gpu_scales).ravel(), 1.0, atol=0.0)


@pytest.mark.parametrize("seed", [0, 17, 52])
def test_srinv_resolves_rotated_six_row_multiple_singularity(seed: int):
    try:
        import casadi as ca

        from embodik.gpu.casadi_fi_pesns import srinv
    except ImportError as e:
        pytest.skip(f"CasADi or modules not available: {e}")

    matrix, _, _ = _rotated_matrix(
        np.array([2.0, 1.7, 1.1, 0.4, 0.75e-6, 1e-9]),
        velocity_dimension=7,
        seed=seed,
    )
    symbolic = ca.SX.sym("six_row_matrix", *matrix.shape)
    function = ca.Function("six_row_srinv", [symbolic], [srinv(symbolic, tol=1e-6, damping=0.1)])

    np.testing.assert_allclose(
        np.asarray(function(matrix)),
        _extended_srinv_reference(matrix),
        rtol=5e-5,
        atol=1e-8,
    )


def test_gpu_srinv_wrappers_share_formula():
    """FI-PeSNS and PPH-SNS wrappers should route through the same CasADi formula."""
    try:
        import casadi as ca

        from embodik.gpu.casadi_fi_pesns import srinv as fi_srinv
        from embodik.gpu.casadi_pph_sns import srinv as pph_srinv
    except ImportError as e:
        pytest.skip(f"CasADi or modules not available: {e}")

    rng = np.random.default_rng(101)
    matrix = rng.standard_normal((3, 5)).astype(np.float64)

    A_sx = ca.SX.sym("A", 3, 5)
    fn_fi = ca.Function("fi_srinv", [A_sx], [fi_srinv(A_sx, tol=1e-6, damping=0.1)])
    fn_pph = ca.Function("pph_srinv", [A_sx], [pph_srinv(A_sx, tol=1e-6, damping=0.1)])

    np.testing.assert_allclose(np.array(fn_fi(matrix)), np.array(fn_pph(matrix)))


# =============================================================================
# Feasible Scale Tests
# =============================================================================


def test_get_feasible_task_scale_unconstrained():
    """Scale = 1 when solution is within bounds."""
    try:
        import casadi as ca

        from embodik.gpu.casadi_fi_pesns import get_feasible_task_scale
    except ImportError as e:
        pytest.skip(f"CasADi or modules not available: {e}")

    n_constraints = 3
    a = np.array([0.1, -0.2, 0.3], dtype=np.float64)  # Contribution
    b = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # Current position
    d = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    d_bar = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    a_sx = ca.SX.sym("a", n_constraints)
    b_sx = ca.SX.sym("b", n_constraints)
    d_sx = ca.SX.sym("d", n_constraints)
    d_bar_sx = ca.SX.sym("d_bar", n_constraints)

    scale_sx = get_feasible_task_scale(a_sx, b_sx, d_sx, d_bar_sx, n_constraints)
    fn = ca.Function("scale", [a_sx, b_sx, d_sx, d_bar_sx], [scale_sx])

    scale = float(np.array(fn(a, b, d, d_bar)).item())
    assert scale == 1.0, f"Expected scale=1.0, got {scale}"


def test_get_feasible_task_scale_constrained():
    """Scale < 1 when full step would violate bounds."""
    try:
        import casadi as ca

        from embodik.gpu.casadi_fi_pesns import get_feasible_task_scale
    except ImportError as e:
        pytest.skip(f"CasADi or modules not available: {e}")

    n_constraints = 3
    a = np.array([2.0, 0.1, 0.1], dtype=np.float64)  # First would violate
    b = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    d = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    d_bar = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    a_sx = ca.SX.sym("a", n_constraints)
    b_sx = ca.SX.sym("b", n_constraints)
    d_sx = ca.SX.sym("d", n_constraints)
    d_bar_sx = ca.SX.sym("d_bar", n_constraints)

    scale_sx = get_feasible_task_scale(a_sx, b_sx, d_sx, d_bar_sx, n_constraints)
    fn = ca.Function("scale", [a_sx, b_sx, d_sx, d_bar_sx], [scale_sx])

    scale = float(np.array(fn(a, b, d, d_bar)).item())
    # a[0]=2.0, upper=1.0, so max scale = 1.0/2.0 = 0.5
    assert 0.0 < scale < 1.0, f"Expected 0 < scale < 1, got {scale}"
    np.testing.assert_allclose(scale, 0.5, atol=1e-6)


# =============================================================================
# FI-PeSNS Full Solver Tests
# =============================================================================


def test_fi_pesns_builds():
    """FI-PeSNS function builds without error."""
    try:
        from embodik.gpu.casadi_fi_pesns import build_fi_pesns_velocity_solve
    except ImportError as e:
        pytest.skip(f"CasADi or modules not available: {e}")

    fn = build_fi_pesns_velocity_solve(
        n_dof=7,
        n_tasks=2,
        task_dims=[6, 3],
        n_constraints=7,
        k_max=5,
    )

    assert fn is not None
    assert fn.n_in() == 5
    assert fn.n_out() == 2


def test_fi_pesns_evaluates():
    """FI-PeSNS function evaluates without error."""
    try:
        from embodik.gpu.casadi_fi_pesns import build_fi_pesns_single_task
    except ImportError as e:
        pytest.skip(f"CasADi or modules not available: {e}")

    n_dof, task_dim, n_constraints = 3, 2, 3
    fn = build_fi_pesns_single_task(
        n_dof=n_dof,
        task_dim=task_dim,
        n_constraints=n_constraints,
        k_max=5,
    )

    targets = np.array([0.1, 0.1], dtype=np.float64)
    jacobians = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    C = np.eye(n_constraints, dtype=np.float64)
    lower = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    upper = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    result = fn(targets, jacobians, C, lower, upper)
    velocity = np.array(result[0]).flatten()
    scales = np.array(result[1]).flatten()

    assert velocity.shape == (n_dof,)
    assert scales.shape == (1,)
    assert np.all(np.isfinite(velocity))
    assert 0.0 <= scales[0] <= 1.0


def test_fi_pesns_robot_configs():
    """Robot configuration builder works."""
    try:
        from embodik.gpu.casadi_fi_pesns import ROBOT_CONFIGS, build_fi_pesns_for_robot
    except ImportError as e:
        pytest.skip(f"CasADi or modules not available: {e}")

    for robot_name in ROBOT_CONFIGS.keys():
        fn = build_fi_pesns_for_robot(robot_name, k_max=3)
        assert fn is not None
        assert fn.n_in() == 5
        assert fn.n_out() == 2


def test_fi_pesns_matches_cpu_loose_bounds():
    """FI-PeSNS matches CPU when bounds are loose (no saturation)."""
    try:
        import embodik as eik
        from embodik.gpu.casadi_fi_pesns import build_fi_pesns_single_task
    except ImportError as e:
        pytest.skip(f"embodik or CasADi not available: {e}")

    n_dof, task_dim, n_constraints = 7, 6, 7
    fn = build_fi_pesns_single_task(
        n_dof=n_dof,
        task_dim=task_dim,
        n_constraints=n_constraints,
        tol=1e-6,
        damping=0.1,
        k_max=10,
    )

    rng = np.random.default_rng(200)
    target = rng.uniform(-0.3, 0.3, task_dim).astype(np.float64)
    J = rng.uniform(-1.0, 1.0, (task_dim, n_dof)).astype(np.float64)
    C = np.eye(n_dof, dtype=np.float64)
    lower = np.full(n_dof, -10.0, dtype=np.float64)  # Loose bounds
    upper = np.full(n_dof, 10.0, dtype=np.float64)

    # CPU
    cpu_result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [target], [np.asfortranarray(J)], C, lower, upper
    )
    cpu_vel = np.array(cpu_result.solution).ravel()

    # FI-PeSNS
    vel_pesns, _ = fn(target, J.flatten(), C, lower, upper)
    vel_pesns = np.array(vel_pesns).flatten()

    # With loose bounds, FI-PeSNS should match CPU closely
    np.testing.assert_allclose(
        cpu_vel, vel_pesns, atol=SIMPLE_ATOL, err_msg="FI-PeSNS must match CPU with loose bounds"
    )


def test_fi_pesns_vs_cpu_tight_bounds():
    """FI-PeSNS vs CPU with tight bounds (may require scaling/penalty)."""
    try:
        import embodik as eik
        from embodik.gpu.casadi_fi_pesns import build_fi_pesns_single_task
    except ImportError as e:
        pytest.skip(f"embodik or CasADi not available: {e}")

    n_dof, task_dim, n_constraints = 7, 6, 7
    fn = build_fi_pesns_single_task(
        n_dof=n_dof,
        task_dim=task_dim,
        n_constraints=n_constraints,
        tol=1e-6,
        damping=0.1,
        k_max=15,
        mu0=1e-2,
        gamma=2.0,
        eta=0.1,
    )

    rng = np.random.default_rng(201)
    max_vel_err = 0.0
    max_constraint_viol = 0.0

    for _ in range(10):
        target = rng.uniform(-0.5, 0.5, task_dim).astype(np.float64)
        J = rng.uniform(-1.0, 1.0, (task_dim, n_dof)).astype(np.float64)
        C = np.eye(n_dof, dtype=np.float64)
        lower = np.full(n_dof, -1.0, dtype=np.float64)
        upper = np.full(n_dof, 1.0, dtype=np.float64)

        # CPU
        cpu_result = eik.computeMultiObjectiveVelocitySolutionEigen(
            [target], [np.asfortranarray(J)], C, lower, upper
        )
        cpu_vel = np.array(cpu_result.solution).ravel()

        # FI-PeSNS
        vel_pesns, _ = fn(target, J.flatten(), C, lower, upper)
        vel_pesns = np.array(vel_pesns).flatten()

        # Track errors
        err = np.max(np.abs(cpu_vel - vel_pesns))
        max_vel_err = max(max_vel_err, err)

        # Check constraint satisfaction
        constraint_val = C @ vel_pesns
        viol_low = np.maximum(0, lower - constraint_val)
        viol_high = np.maximum(0, constraint_val - upper)
        max_viol = np.max(np.maximum(viol_low, viol_high))
        max_constraint_viol = max(max_constraint_viol, max_viol)

    # FI-PeSNS should be close to CPU (penalty approximation allows some gap)
    assert (
        max_vel_err < VELOCITY_ATOL
    ), f"FI-PeSNS velocity vs CPU max error {max_vel_err} >= {VELOCITY_ATOL}"
    # Constraints should be approximately satisfied
    assert (
        max_constraint_viol < CONSTRAINT_TOL
    ), f"FI-PeSNS max constraint violation {max_constraint_viol} >= {CONSTRAINT_TOL}"


def test_fi_pesns_constraint_satisfaction():
    """FI-PeSNS output should satisfy constraints (with penalty nudge)."""
    try:
        from embodik.gpu.casadi_fi_pesns import build_fi_pesns_single_task
    except ImportError as e:
        pytest.skip(f"CasADi not available: {e}")

    n_dof, task_dim, n_constraints = 7, 6, 7
    fn = build_fi_pesns_single_task(
        n_dof=n_dof,
        task_dim=task_dim,
        n_constraints=n_constraints,
        k_max=20,
        mu0=1e-1,
        gamma=2.0,
        eta=0.2,
    )

    rng = np.random.default_rng(202)
    max_viol = 0.0

    for _ in range(20):
        target = rng.uniform(-1.0, 1.0, task_dim).astype(np.float64)  # Aggressive target
        J = rng.uniform(-1.0, 1.0, (task_dim, n_dof)).astype(np.float64)
        C = np.eye(n_dof, dtype=np.float64)
        lower = np.full(n_dof, -1.0, dtype=np.float64)
        upper = np.full(n_dof, 1.0, dtype=np.float64)

        vel_pesns, _ = fn(target, J.flatten(), C, lower, upper)
        vel_pesns = np.array(vel_pesns).flatten()

        constraint_val = C @ vel_pesns
        viol = np.maximum(0, np.maximum(lower - constraint_val, constraint_val - upper))
        max_viol = max(max_viol, np.max(viol))

    assert (
        max_viol < CONSTRAINT_TOL
    ), f"FI-PeSNS max constraint violation {max_viol} >= {CONSTRAINT_TOL}"


def test_fi_pesns_multi_task():
    """FI-PeSNS handles multiple hierarchical tasks."""
    try:
        from embodik.gpu.casadi_fi_pesns import build_fi_pesns_velocity_solve
    except ImportError as e:
        pytest.skip(f"CasADi not available: {e}")

    n_dof = 7
    task_dims = [6, 3]
    n_constraints = 7
    fn = build_fi_pesns_velocity_solve(
        n_dof=n_dof,
        n_tasks=2,
        task_dims=task_dims,
        n_constraints=n_constraints,
        k_max=10,
    )

    rng = np.random.default_rng(203)
    target1 = rng.uniform(-0.3, 0.3, 6).astype(np.float64)
    target2 = rng.uniform(-0.3, 0.3, 3).astype(np.float64)
    targets = np.concatenate([target1, target2])

    J1 = rng.uniform(-1.0, 1.0, (6, n_dof)).astype(np.float64)
    J2 = rng.uniform(-1.0, 1.0, (3, n_dof)).astype(np.float64)
    jacobians = np.concatenate([J1.flatten(), J2.flatten()])

    C = np.eye(n_constraints, n_dof, dtype=np.float64)
    lower = np.full(n_constraints, -1.0, dtype=np.float64)
    upper = np.full(n_constraints, 1.0, dtype=np.float64)

    vel, scales = fn(targets, jacobians, C, lower, upper)
    vel = np.array(vel).flatten()
    scales = np.array(scales).flatten()

    assert vel.shape == (n_dof,)
    assert scales.shape == (2,)
    assert np.all(np.isfinite(vel))
    assert np.all((scales >= 0) & (scales <= 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
