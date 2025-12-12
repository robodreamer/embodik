"""Cross-validation tests between embodiK and flex_ik_v1 solvers."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pytest

import embodik as sik

# =============================================================================
# Optional flex_ik_v1 dependency detection
# =============================================================================

VelIkSolver = None
SingularityRobustSolver = None
FLEX_SOLVER_AVAILABLE = False
_flex_solver_error = None  # pragma: no cover - diagnostics only

_flex_candidates = []
_flex_env = os.environ.get("FLEX_IK_V1_PATH")
if _flex_env:
    _flex_candidates.append(Path(_flex_env))

_here = Path(__file__).resolve()
_project_root = _here.parents[1]
_workspace_root = _here.parents[2] if len(_here.parents) > 2 else _project_root

_flex_candidates.extend(
    [
        _project_root / "flex_ik_v1",
        _workspace_root / "flex_ik_v1",
        _workspace_root / "flex-ik-new" / "flex_ik_v1",
    ]
)

for _candidate in _flex_candidates:
    if not _candidate:
        continue
    candidate_path = _candidate.expanduser().resolve()
    if not candidate_path.is_dir():
        continue
    if str(candidate_path) not in sys.path:
        sys.path.append(str(candidate_path))
    try:
        from flexible_ik_solver import (  # type: ignore[import]
            VelIkSolver as _VelIkSolver,
            SingularityRobustSolver as _SingularityRobustSolver,
        )
    except ImportError as exc:  # pragma: no cover - diagnostic path
        _flex_solver_error = exc
        continue
    VelIkSolver = _VelIkSolver
    SingularityRobustSolver = _SingularityRobustSolver
    FLEX_SOLVER_AVAILABLE = True
    break

FLEX_SOLVER_MARK = pytest.mark.skipif(
    not FLEX_SOLVER_AVAILABLE,
    reason="flexible_ik_solver (flex_ik_v1) package not available for Swift vs Flex comparison tests",
)

COMPARE_ATOL = 1e-6
COMPARE_RTOL = 1e-6


def _ensure_column(vec: np.ndarray) -> np.ndarray:
    """Ensure vectors are treated as column matrices for flex solver."""
    return vec.reshape((-1, 1))


def _flex_solve(
    goals: Sequence[np.ndarray],
    jacobians: Sequence[np.ndarray],
    C: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    assert FLEX_SOLVER_AVAILABLE and VelIkSolver and SingularityRobustSolver
    solver = VelIkSolver(SingularityRobustSolver())
    solver.set_constraints(C, lower, upper)
    dx_terms = [_ensure_column(goal) for goal in goals]
    dq, s_data, exit_code = solver.solve(dx_terms, list(jacobians))
    assert exit_code.name == "SUCCESS"
    return dq, np.asarray(s_data, dtype=float).reshape(-1)


def _swift_solve(
    goals: Sequence[np.ndarray],
    jacobians: Sequence[np.ndarray],
    C: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> sik.VelocitySolverResult:
    result = sik.computeMultiObjectiveVelocitySolutionEigen(goals, jacobians, C, lower, upper)
    assert result.status == sik.SolverStatus.SUCCESS
    return result


def _assert_close(
    swift_res: sik.VelocitySolverResult,
    flex_solution: np.ndarray,
    flex_scales: np.ndarray,
    *,
    atol: float = COMPARE_ATOL,
    rtol: float = COMPARE_RTOL,
) -> None:
    np.testing.assert_allclose(np.asarray(swift_res.solution), flex_solution, atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(swift_res.task_scales), flex_scales, atol=atol, rtol=rtol)


@FLEX_SOLVER_MARK
def test_secondary_task_full_scale_match():
    """Both solvers should fully realize compatible primary/secondary goals."""
    goals = [np.array([0.5, 0.2]), np.array([0.1])]
    jacobians = [
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0.5, 0.0, 1.0]]),
    ]
    C = np.eye(3)
    lower = -0.5 * np.ones(3)
    upper = 0.5 * np.ones(3)

    swift_res = _swift_solve(goals, jacobians, C, lower, upper)
    flex_solution, flex_scales = _flex_solve(goals, jacobians, C, lower, upper)
    _assert_close(swift_res, flex_solution, flex_scales)


@FLEX_SOLVER_MARK
def test_secondary_task_zero_scale_match():
    """Secondary task should drop out when infeasible under constraint coupling."""
    jacobians = [
        np.array(
            [
                [1.86755799, -0.97727788, 0.95008842, -0.15135721],
                [-0.10321885, 0.4105985, 0.14404357, 1.45427351],
            ]
        ),
        np.array(
            [
                [0.76103773, 0.12167502, 0.44386323, 0.33367433],
                [1.49407907, -0.20515826, 0.3130677, -0.85409574],
            ]
        ),
    ]
    goals = [np.array([-1.06694915, 0.09843529]), np.array([-0.47218792, -1.08871803])]
    C = np.vstack(
        (
            np.eye(4),
            np.array([[1.76405235, 0.40015721, 0.97873798, 2.2408932]]),
        )
    )
    lower = -0.5 * np.ones(C.shape[0])
    upper = 0.5 * np.ones(C.shape[0])

    swift_res = _swift_solve(goals, jacobians, C, lower, upper)
    flex_solution, flex_scales = _flex_solve(goals, jacobians, C, lower, upper)
    _assert_close(swift_res, flex_solution, flex_scales)
    assert swift_res.task_scales[1] == pytest.approx(0.0, abs=COMPARE_ATOL)


@FLEX_SOLVER_MARK
def test_secondary_task_partial_scale_match():
    """Both solvers should agree on partial scaling of secondary objectives."""
    jacobians = [
        np.array(
            [
                [-0.35112678, -0.73447285, 0.08367298, 0.85620461],
                [2.47032824, -0.14610479, -1.36566815, 1.90543329],
            ]
        ),
        np.array(
            [
                [-2.92351247, 0.87447214, -0.50567919, -0.32074459],
                [-0.532777, 0.94809246, -0.41335591, -0.5042479],
            ]
        ),
        np.array(
            [
                [-1.7602144, 0.18315341, 0.17622192, -0.22353617],
                [1.18345042, 0.85456987, -0.06532862, 0.3643191],
            ]
        ),
    ]
    goals = [
        np.array([0.12933523, 0.34524372]),
        np.array([-2.21386487, -0.98346363]),
        np.array([-0.68591816, 0.0958025]),
    ]
    C = np.vstack(
        (
            np.eye(4),
            np.array([[0.17251947, 1.63548253, 0.0373364, -0.88414969]]),
        )
    )
    lower = -0.3 * np.ones(C.shape[0])
    upper = 0.3 * np.ones(C.shape[0])

    swift_res = _swift_solve(goals, jacobians, C, lower, upper)
    flex_solution, flex_scales = _flex_solve(goals, jacobians, C, lower, upper)
    _assert_close(swift_res, flex_solution, flex_scales)
    assert 0.0 < swift_res.task_scales[1] < 1.0
