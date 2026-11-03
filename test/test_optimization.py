"""Standalone mathematical formulations require no robot model or state."""

import numpy as np
import pytest

import embodik as eik


def assert_solution(result, expected, objective=None):
    assert result.success, (result.status, result.message)
    assert result.status == eik.SolveStatus.SOLVED
    np.testing.assert_allclose(result.x, expected, atol=1e-7, rtol=0)
    assert np.isfinite(result.objective)
    assert result.normalized_violation <= 1e-8
    assert result.elapsed_ms >= 0
    if objective is not None:
        assert result.objective == pytest.approx(objective, abs=1e-8)


def test_unconstrained_ls_preserves_zero_ridge_and_rank_deficiency():
    result = eik.solve_ls(np.array([[1.0, 1.0]]), np.array([1.0]))
    assert_solution(result, [0.5, 0.5], 0)
    np.testing.assert_allclose(result.x.sum(), 1, atol=1e-12)


@pytest.mark.parametrize("kind", ["ls", "qp"])
@pytest.mark.parametrize("warm", [None, [0.0, 0.0], [10.0, -20.0]])
def test_projection_and_finite_infeasible_warm_start(kind, warm):
    A, lower, upper = np.eye(2), -np.ones(2), np.ones(2)
    x0 = None if warm is None else np.array(warm)
    if kind == "ls":
        r = eik.solve_ls(np.eye(2), np.array([2.0, -3.0]), A, lower, upper, x0=x0)
        expected_objective = 2.5
    else:
        r = eik.solve_qp(np.eye(2), np.array([-2.0, 3.0]), A, lower, upper, x0=x0)
        expected_objective = -4.0
    assert_solution(r, [1, -1], expected_objective)


def test_qp_linear_term_sign_and_equality():
    P, q = np.array([[4.0, 1.0], [1.0, 2.0]]), -np.ones(2)
    A = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert_solution(
        eik.solve_qp(P, q, A, np.array([1.0, 0.0, 0.0]), np.array([1.0, np.inf, np.inf])),
        [0.25, 0.75],
        -0.125,
    )


@pytest.mark.parametrize("rows", ["original", "scaled", "duplicate", "permuted"])
def test_ls_equivalent_general_constraints(rows):
    A, l, u = np.array([[1.0, 1.0], [1.0, -1.0]]), np.array([-0.5, -2.0]), np.array([0.5, 2.0])
    if rows == "scaled":
        scale = np.array([1e-8, 1e8])
        A, l, u = A * scale[:, None], l * scale, u * scale
    elif rows == "duplicate":
        A, l, u = np.vstack([A, A]), np.r_[l, l], np.r_[u, u]
    elif rows == "permuted":
        A, l, u = A[::-1].copy(), l[::-1].copy(), u[::-1].copy()
    assert_solution(eik.solve_ls(np.eye(2), np.ones(2), A, l, u), [0.25, 0.25], 0.5625)


def test_explicit_ridge_changes_only_the_requested_objective():
    assert_solution(eik.solve_ls(np.eye(2), np.ones(2), ridge=1.0), [0.5, 0.5], 0.5)


def test_stacked_horizon_dynamics_and_control_bounds():
    J = np.diag([1.0, 1.0, np.sqrt(0.1), np.sqrt(0.1)])
    A = np.array(
        [[1.0, 0.0, -1.0, 0.0], [-1.0, 1.0, 0.0, -1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    l, u = np.array([1.0, 0.0, -0.4, -0.4]), np.array([1.0, 0.0, 0.4, 0.4])
    assert_solution(eik.solve_ls(J, np.zeros(4), A, l, u), [0.6, 0.2, -0.4, -0.4], 0.216)


@pytest.mark.parametrize(
    "A,l,u",
    [
        ([[0.0]], [1.0], [2.0]),
        ([[1.0], [1.0]], [1.0, -np.inf], [np.inf, 0.0]),
        ([[1.0]], [2.0], [1.0]),
    ],
)
def test_infeasible_constraints_return_no_candidate(A, l, u):
    r = eik.solve_ls(np.ones((1, 1)), np.zeros(1), np.array(A), np.array(l), np.array(u))
    assert not r.success and r.status == eik.SolveStatus.INFEASIBLE
    assert r.x.size == 0


@pytest.mark.parametrize("field", ["J", "target", "A", "l", "u", "x0", "ridge", "tol"])
def test_nonfinite_invalid_input_is_not_a_success(field):
    args = dict(
        J=np.eye(2),
        target=np.ones(2),
        A=np.eye(2),
        l=-np.ones(2),
        u=np.ones(2),
        x0=np.zeros(2),
        ridge=0.0,
        tol=1e-8,
    )
    if field in ("ridge", "tol"):
        args[field] = np.nan
    else:
        args[field] = args[field].copy()
        args[field].flat[0] = np.nan
    r = eik.solve_ls(**args)
    assert not r.success and r.status == eik.SolveStatus.INVALID_INPUT
    assert r.x.size == 0


def test_invalid_shapes_and_options():
    for options in (
        dict(A=np.ones((1, 3))),
        dict(A=np.eye(2), l=np.zeros(1)),
        dict(x0=np.zeros(3)),
        dict(ridge=-1.0),
        dict(tol=0.0),
        dict(max_iter=0),
    ):
        r = eik.solve_ls(np.eye(2), np.ones(2), **options)
        assert not r.success and r.status == eik.SolveStatus.INVALID_INPUT
        assert r.x.size == 0


@pytest.mark.parametrize("P", [np.zeros((2, 2)), np.ones((2, 2)), np.diag([1.0, -1.0])])
def test_unsupported_hessian_families_are_explicit(P):
    r = eik.solve_qp(P, np.ones(2))
    assert not r.success and r.status == eik.SolveStatus.UNSUPPORTED
    assert r.x.size == 0


def test_asymmetric_qp_input_is_rejected():
    r = eik.solve_qp(np.array([[1.0, 1.0], [0.0, 1.0]]), np.zeros(2))
    assert r.status == eik.SolveStatus.INVALID_INPUT and not r.success


def test_large_asymmetric_qp_is_rejected_without_norm_overflow():
    r = eik.solve_qp(np.array([[1e200, 1e200], [0.0, 1e200]]), np.zeros(2))
    assert r.status == eik.SolveStatus.INVALID_INPUT and not r.success


def test_exact_large_ls_solution_does_not_evaluate_zero_ridge_times_infinity():
    r = eik.solve_ls(np.ones((1, 1)), np.array([1e160]))
    assert r.success, (r.status, r.message)
    assert r.x[0] == pytest.approx(1e160)
    assert r.objective == 0.0


@pytest.mark.parametrize("J", [np.diag([1.0, 1e-20]), np.array([[1.0, 1.0], [0.0, 1e-20]])])
def test_unresolved_weak_objective_direction_is_not_reported_solved(J):
    r = eik.solve_ls(J, np.array([0.0, 1.0]))
    assert not r.success and r.status == eik.SolveStatus.NO_PROGRESS
    assert r.x.size == 0


def test_weak_positive_definite_qp_is_explicitly_uncertified():
    r = eik.solve_qp(np.diag([1.0, 1e-40]), np.array([0.0, -1e-20]))
    assert not r.success and r.status == eik.SolveStatus.NO_PROGRESS
    assert r.x.size == 0


def test_dependent_inconsistent_ls_is_conservatively_uncertified():
    r = eik.solve_ls(np.array([[1.0, 1.0], [2.0, 2.0]]), np.array([1.0, 3.0]))
    assert not r.success and r.status == eik.SolveStatus.NO_PROGRESS
    assert r.x.size == 0


def test_rounded_schur_pivot_is_not_an_exact_rank_certificate():
    a = 0.1
    r = eik.solve_ls(np.array([[1.0, a], [a, a*a]]), np.array([-a, 1.0]))
    assert not r.success and r.status == eik.SolveStatus.NO_PROGRESS
    assert r.x.size == 0


def test_requested_gpu_is_not_silently_run_on_cpu():
    r = eik.solve_ls(np.eye(2), np.ones(2), backend="gpu")
    assert r.status == eik.SolveStatus.UNSUPPORTED and r.backend == "none"
    assert r.x.size == 0


def test_unknown_backend_and_no_input_mutation():
    J, target, A, l, u = np.eye(2), np.ones(2), np.eye(2), -np.ones(2), np.ones(2)
    saved = [x.copy() for x in (J, target, A, l, u)]
    assert_solution(eik.solve_ls(J, target, A, l, u), [1, 1], 0)
    for old, new in zip(saved, (J, target, A, l, u)):
        np.testing.assert_array_equal(old, new)
    r = eik.solve_ls(J, target, backend="missing")
    assert r.status == eik.SolveStatus.INVALID_INPUT
