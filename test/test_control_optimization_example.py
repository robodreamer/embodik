"""The standalone example is runnable and preserves its mathematical meaning."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import embodik

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/control_optimization.py"


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("control_optimization_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bounded_allocation_saturates_and_balances_remaining_effort(demo):
    result = demo.solve("allocation")
    assert result["status"] == "SOLVED"
    expected = np.array([0.8, 70.0 / 201.0, 0.8, 70.0 / 201.0])
    np.testing.assert_allclose(result["x"], expected, atol=1e-10, rtol=0)
    problem = demo.problems()["allocation"]
    np.testing.assert_allclose(result["achieved"], problem["J"] @ expected, atol=1e-10)
    residual = problem["J"] @ expected - problem["target"]
    objective = 0.5 * residual @ residual + 0.5 * problem["ridge"] * (expected @ expected)
    assert result["objective"] == pytest.approx(objective, abs=1e-12)
    assert result["max_hard_violation"] <= 1e-10


def test_projection_preserves_hard_equality_and_nonnegative_command(demo):
    result = demo.solve("projection")
    assert result["status"] == "SOLVED"
    np.testing.assert_allclose(result["x"], [1.0, 0.0], atol=1e-10)
    assert sum(result["x"]) == pytest.approx(1.0, abs=1e-10)
    assert min(result["x"]) >= -1e-10
    # QP objective omits .5 * ||command||², the constant of the projection LS.
    assert result["objective"] == pytest.approx(-1.5, abs=1e-10)


def test_fit_folds_bias_and_square_root_weights_into_both_sides(demo):
    problem = demo.problems()["fit"]
    np.testing.assert_allclose(problem["J"], [[2.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
    np.testing.assert_allclose(problem["target"], [1.6, -0.3, 0.2])
    result = demo.solve("fit")
    assert result["status"] == "SOLVED"
    expected = np.array([0.5, -11.0 / 54.0])
    np.testing.assert_allclose(result["x"], expected, atol=1e-10)
    # Independently evaluate the original affine weighted objective.
    M = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    residual = M @ expected + np.array([0.2, -0.1, 0.3]) - np.array([1.0, -0.4, 0.7])
    objective = 0.5 * np.dot([4.0, 1.0, 0.25], residual**2) + 0.05 * (expected @ expected)
    assert result["objective"] == pytest.approx(objective, abs=1e-12)


def test_weighted_fit_qp_expansion_preserves_solution_and_constant(demo):
    problem = demo.problems()["fit"]
    J, target, ridge = (problem[key] for key in ("J", "target", "ridge"))
    qp = embodik.solve_qp(
        J.T @ J + ridge * np.eye(J.shape[1]),
        -J.T @ target,
        problem["A"],
        problem["l"],
        problem["u"],
    )
    ls = demo.solve("fit")
    assert qp.success, (qp.status, qp.message)
    np.testing.assert_allclose(qp.x, ls["x"], atol=1e-10)
    assert qp.objective + 0.5 * (target @ target) == pytest.approx(ls["objective"], abs=1e-12)


def test_problem_data_is_fresh_for_each_example_invocation(demo):
    altered = demo.problems()
    altered["fit"]["J"][:] = 0.0
    altered["allocation"]["l"][:] = 999.0
    assert demo.problems()["fit"]["J"][0, 0] == 2.0
    assert demo.solve("allocation")["status"] == "SOLVED"


@pytest.mark.parametrize("case", ["all", "fit"])
def test_command_line_outputs_valid_json_and_selected_cases(case):
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE), "--case", case], text=True, capture_output=True, check=True
    )
    reports = json.loads(completed.stdout)
    assert set(reports) == ({"allocation", "projection", "fit"} if case == "all" else {"fit"})
    assert all(report["status"] == "SOLVED" for report in reports.values())


def test_explicit_gpu_request_remains_unsupported_without_cpu_fallback():
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE), "--case", "fit", "--backend", "gpu"],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 1
    result = json.loads(completed.stdout)["fit"]
    assert result["status"] == "UNSUPPORTED"
    assert result["backend"] == "none"
    assert result["x"] == [] and result["objective"] is None
