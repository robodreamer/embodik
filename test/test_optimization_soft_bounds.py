"""Exact affine interval penalties are independent of robot state."""

import numpy as np
import pytest

import embodik as eik


def checked(result, expected, objective=None):
    assert result.success, (result.status, result.message)
    np.testing.assert_allclose(result.x, expected, atol=1e-8, rtol=0)
    assert result.x.size == len(expected)
    assert result.normalized_violation <= 1e-8
    if objective is not None:
        assert result.objective == pytest.approx(objective, abs=1e-9)


def test_inactive_interval_is_zero_cost_and_zero_ridge_stays_zero():
    checked(
        eik.solve_ls(
            np.eye(1),
            np.array([2.0]),
            S=np.eye(1),
            sl=np.array([-1.0]),
            su=np.array([3.0]),
            weights=np.array([9.0]),
        ),
        [2.0],
        0.0,
    )


@pytest.mark.parametrize(
    "sl,su,target,expected",
    [
        ([0.0], [np.inf], -2.0, -0.5),
        ([-np.inf], [-2.0], 0.0, -1.5),
        ([-2.0], [-2.0], 0.0, -1.5),
    ],
)
def test_weighted_lower_upper_and_equality(sl, su, target, expected):
    checked(
        eik.solve_ls(
            np.eye(1),
            np.array([target]),
            S=np.eye(1),
            sl=np.array(sl),
            su=np.array(su),
            weights=np.array([3.0]),
        ),
        [expected],
        1.5,
    )


def test_soft_defaults_are_unit_weight_and_one_sided_infinity():
    checked(eik.solve_ls(np.eye(1), np.array([-2.0]), S=np.eye(1), sl=np.zeros(1)), [-1.0], 1.0)
    checked(eik.solve_ls(np.eye(1), np.array([2.0]), S=np.eye(1), su=np.zeros(1)), [1.0], 1.0)


def test_ridge_applies_to_original_x_without_auxiliary_leak():
    result = eik.solve_ls(np.eye(1), np.zeros(1), ridge=2.0, S=np.eye(1), sl=np.ones(1))
    checked(result, [0.25], 0.375)
    assert result.max_violation == 0.0


def test_incompatible_penalties_are_not_infeasible_and_hard_bounds_stay_hard():
    S = np.ones((2, 1))
    checked(
        eik.solve_ls(
            np.eye(1), np.zeros(1), S=S, sl=np.array([2.0, -np.inf]), su=np.array([np.inf, -2.0])
        ),
        [0.0],
        4.0,
    )
    result = eik.solve_ls(
        np.eye(1), np.zeros(1), A=np.eye(1), u=np.zeros(1), S=np.eye(1), sl=np.array([2.0])
    )
    checked(result, [0.0], 2.0)
    assert result.max_violation <= 1e-10


def test_independent_constructed_kkt_reference_with_coupled_constraints():
    S = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    sl, su = np.array([1.0, -np.inf, -0.5, 0.0]), np.array([np.inf, -0.5, 0.5, 1.0])
    weights = np.array([2.0, 4.0, 3.0, 5.0])
    A = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    l, u = np.array([1.0, 0.5, -np.inf]), np.array([1.0, np.inf, 0.75])
    target = np.array([0.4, 0.075, 1.725])
    result = eik.solve_ls(
        np.eye(3),
        target,
        A,
        l,
        u,
        S=S,
        sl=sl,
        su=su,
        weights=weights,
        ridge=0.5,
        x0=np.array([0.5, -0.25, 0.75]),
        tol=1e-10,
    )
    checked(result, [0.5, -0.25, 0.75])
    distance = S @ result.x - np.clip(S @ result.x, sl, su)
    gradient = 1.5 * result.x - target + S.T @ (weights * distance)
    dual = np.array([0.2, -0.3, 0.4])
    np.testing.assert_allclose(gradient + A.T @ dual, 0.0, atol=1e-9, rtol=0)
    assert result.objective == pytest.approx(
        0.5 * np.sum((result.x - target) ** 2)
        + 0.25 * np.sum(result.x**2)
        + 0.5 * np.sum(weights * distance**2),
        abs=1e-12,
    )
    assert np.max(np.maximum(l - A @ result.x, A @ result.x - u)) <= 1e-10


@pytest.mark.parametrize("disabled", ["zero_weight", "unbounded", "empty"])
def test_disabled_rows_preserve_base_status_primal_objective_and_iterations(disabled):
    J, target = np.array([[1.0, 1.0]]), np.ones(1)
    base = eik.solve_ls(J, target)
    options = dict(S=np.eye(2))
    if disabled == "zero_weight":
        options.update(sl=np.ones(2), weights=np.zeros(2))
    elif disabled == "empty":
        options = dict(S=np.empty((0, 2)))
    result = eik.solve_ls(J, target, **options)
    assert result.status == base.status
    np.testing.assert_array_equal(result.x, base.x)
    assert result.objective == base.objective
    assert result.iterations == base.iterations


def test_disabled_rows_are_eliminated_alongside_an_effective_penalty():
    baseline = eik.solve_ls(np.eye(1), np.array([-2.0]), S=np.eye(1), sl=np.zeros(1))
    result = eik.solve_ls(
        np.eye(1),
        np.array([-2.0]),
        S=np.ones((3, 1)),
        sl=np.array([0.0, 9.0, -np.inf]),
        su=np.full(3, np.inf),
        weights=np.array([1.0, 0.0, 7.0]),
    )
    assert result.status == baseline.status
    np.testing.assert_array_equal(result.x, baseline.x)
    assert result.objective == baseline.objective
    assert result.iterations == baseline.iterations


@pytest.mark.parametrize("scale", [1e-100, 1e-4, 1.0, 1e100])
def test_uniform_objective_units_preserve_soft_weights_and_original_ridge(scale):
    result = eik.solve_ls(
        scale * np.eye(1),
        np.array([-2.0 * scale]),
        S=np.eye(1),
        sl=np.zeros(1),
        weights=np.array([3.0 * scale**2]),
        ridge=2.0 * scale**2,
    )
    checked(result, [-1.0 / 3.0])
    assert result.objective / scale**2 == pytest.approx(5.0 / 3.0, abs=1e-10)


@pytest.mark.parametrize(
    "field,value",
    [
        ("S", np.ones((1, 2))),
        ("S", np.array([[np.nan]])),
        ("sl", np.zeros(2)),
        ("sl", np.array([np.nan])),
        ("sl", np.array([np.inf])),
        ("su", np.array([-np.inf])),
        ("weights", np.ones(2)),
        ("weights", np.empty(0)),
        ("weights", np.array([-1.0])),
        ("weights", np.array([np.inf])),
        ("weights", np.array([np.nan])),
    ],
)
def test_invalid_soft_shapes_coefficients_intervals_and_weights(field, value):
    options = dict(S=np.eye(1), sl=np.zeros(1), su=np.ones(1))
    options[field] = value
    result = eik.solve_ls(np.eye(1), np.zeros(1), **options)
    assert result.status == eik.SolveStatus.INVALID_INPUT and result.x.size == 0


def test_reversed_soft_interval_is_invalid_instead_of_hard_infeasibility():
    result = eik.solve_ls(np.eye(1), np.zeros(1), S=np.eye(1), sl=np.ones(1), su=np.zeros(1))
    assert result.status == eik.SolveStatus.INVALID_INPUT and result.x.size == 0


@pytest.mark.parametrize("field", ["sl", "su", "weights"])
def test_soft_arguments_require_an_affine_matrix(field):
    result = eik.solve_ls(np.eye(1), np.zeros(1), **{field: np.ones(1)})
    assert result.status == eik.SolveStatus.INVALID_INPUT and result.x.size == 0


def test_original_hard_infeasibility_stays_distinct():
    result = eik.solve_ls(
        np.eye(1), np.zeros(1), A=np.zeros((1, 1)), l=np.ones(1), S=np.eye(1), sl=np.zeros(1)
    )
    assert result.status == eik.SolveStatus.INFEASIBLE and result.x.size == 0


def test_exhausted_working_set_budget_returns_no_original_or_auxiliary_candidate():
    result = eik.solve_ls(
        np.eye(2),
        np.array([-2.0, 3.0]),
        A=np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        l=np.array([0.0, -1.0, -1.0]),
        u=np.array([0.0, 1.0, 1.0]),
        S=np.eye(2),
        sl=np.zeros(2),
        max_iter=1,
    )
    assert result.status == eik.SolveStatus.NO_PROGRESS
    assert result.iterations == 1 and result.x.size == 0


def test_soft_gpu_request_is_explicitly_unsupported():
    result = eik.solve_ls(np.eye(1), np.zeros(1), S=np.eye(1), sl=np.ones(1), backend="gpu")
    assert result.status == eik.SolveStatus.UNSUPPORTED and result.backend == "none"
    assert result.x.size == 0


def test_inputs_are_not_mutated_with_infeasible_warm_start():
    J, target, S, sl, su, weights, x0 = (
        np.eye(2),
        np.ones(2),
        np.eye(2),
        np.zeros(2),
        np.ones(2),
        np.array([1.0, 2.0]),
        np.full(2, -10.0),
    )
    arrays = (J, target, S, sl, su, weights, x0)
    copies = [value.copy() for value in arrays]
    checked(eik.solve_ls(J, target, S=S, sl=sl, su=su, weights=weights, x0=x0), [1.0, 1.0], 0.0)
    for before, after in zip(copies, arrays):
        np.testing.assert_array_equal(before, after)
