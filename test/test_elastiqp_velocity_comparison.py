"""Compare legacy velocity solutions with ElastiQP on shared numerical inputs.

Both solvers receive the same Jacobian, target, and box. EmbodiK solves the
legacy uniform-scale velocity problem. ElastiQP solves the damped least squares
problem in that box. The comparison skips when ElastiQP is not importable.
Point ``PYTHONPATH`` at the ElastiQP ``python`` directory and
``ELASTIQP_BUILD_DIR`` at the extension build when running it locally.
"""

from __future__ import annotations

import numpy as np
import pytest

import embodik as eik

elastiqp = pytest.importorskip("elastiqp")

from support_scale_cases import scale_cases  # noqa: E402

BOX_PENALTY = 1e6
RIDGE = 1e-8
MATCH_ATOL = 2e-4
# Catalog velocities are the default-damped command. Undamped legacy stays on
# that command except for the loose linear combination, which moves by ~8e-5.
CATALOG_UNDAMPED_ATOL = 1e-4
GAP_VELOCITY = 0.1
BOUND_ATOL = 1e-8

AGREE = (
    "linear_combination_loose",
    "repeated_row_partial_cap",
    "midcap_uses_the_open_joint",
    "redundant_dependent_n4",
    "skew_two_degrees_loose",
)
# Legacy scale stops at the first feasible uniform scale. The box least
# squares continues to a smaller task residual.
DIVERGE = (
    "split_caps_stop_at_the_first_feasible_scale",
    "both_tight_keeps_uniform_scale",
    "nonzero_lower_keeps_uniform_scale",
    "wide_box_keeps_uniform_scale",
    "linear_combination_inconsistent",
)
DAMPED_INTERIOR = (
    "linear_combination_loose",
    "midcap_uses_the_open_joint",
    "skew_two_degrees_loose",
)


def _cases():
    return {case.name: case for case in scale_cases()}


def _legacy_velocity(case, damping: float):
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [case.target],
        [np.asfortranarray(case.matrix)],
        np.eye(case.matrix.shape[1]),
        case.lower,
        case.upper,
        1e-6,
        1e-10,
        20,
        1e10,
        2,
        1e-6,
        damping,
    )
    return result, np.asarray(result.solution, dtype=float)


def _box_least_squares(case, damping: float):
    jacobian = np.atleast_2d(case.matrix)
    width = jacobian.shape[1]
    gram = jacobian.T @ jacobian + damping * np.eye(width)
    linear = -(jacobian.T @ case.target)
    inequalities = np.vstack((np.eye(width), -np.eye(width)))
    limits = np.concatenate((case.upper, -case.lower))
    solution = elastiqp.solve(
        gram,
        linear,
        inequalities,
        limits,
        np.full(limits.size, BOX_PENALTY),
        eps_abs=1e-10,
    )
    return solution, np.asarray(solution.x, dtype=float)


def _inside_box(velocity, case) -> None:
    assert np.all(np.isfinite(velocity))
    assert np.all(velocity >= case.lower - BOUND_ATOL)
    assert np.all(velocity <= case.upper + BOUND_ATOL)


@pytest.mark.parametrize("name", AGREE)
def test_legacy_scale_matches_box_least_squares_when_the_scale_is_optimal(name):
    case = _cases()[name]
    legacy, legacy_velocity = _legacy_velocity(case, 0.0)
    quadratic, quadratic_velocity = _box_least_squares(case, RIDGE)
    assert legacy.status == eik.SolverStatus.SUCCESS
    assert quadratic.status == elastiqp.Status.Solved
    assert quadratic.converged == 1
    assert float(np.max(np.asarray(quadratic.t))) <= BOUND_ATOL
    _inside_box(legacy_velocity, case)
    _inside_box(quadratic_velocity, case)
    np.testing.assert_allclose(legacy.task_scales[0], case.scale, atol=1e-6)
    np.testing.assert_allclose(legacy_velocity, case.velocity, atol=CATALOG_UNDAMPED_ATOL)
    np.testing.assert_allclose(quadratic_velocity, legacy_velocity, atol=MATCH_ATOL)


@pytest.mark.parametrize("name", DIVERGE)
def test_uniform_scale_stays_apart_from_box_least_squares(name):
    case = _cases()[name]
    legacy, legacy_velocity = _legacy_velocity(case, 0.0)
    quadratic, quadratic_velocity = _box_least_squares(case, RIDGE)
    assert legacy.status == eik.SolverStatus.SUCCESS
    assert quadratic.status == elastiqp.Status.Solved
    assert float(np.max(np.asarray(quadratic.t))) <= BOUND_ATOL
    _inside_box(legacy_velocity, case)
    _inside_box(quadratic_velocity, case)
    np.testing.assert_allclose(legacy_velocity, case.velocity, atol=1e-6)
    assert np.max(np.abs(quadratic_velocity - legacy_velocity)) > GAP_VELOCITY
    legacy_residual = np.linalg.norm(case.matrix @ legacy_velocity - case.target)
    quadratic_residual = np.linalg.norm(case.matrix @ quadratic_velocity - case.target)
    assert quadratic_residual < legacy_residual - 1e-6


@pytest.mark.parametrize("name", DAMPED_INTERIOR)
def test_default_damping_moves_the_box_least_squares_off_the_legacy_velocity(name):
    case = _cases()[name]
    legacy, legacy_velocity = _legacy_velocity(case, 0.1)
    _quadratic, quadratic_velocity = _box_least_squares(case, 0.1)
    np.testing.assert_allclose(legacy_velocity, case.velocity, atol=1e-6)
    _inside_box(quadratic_velocity, case)
    assert np.max(np.abs(quadratic_velocity - legacy_velocity)) > 1e-3
