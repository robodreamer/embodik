"""Nonzero sub-microradian steps still require nonlinear priority validation."""

from __future__ import annotations

import pytest
from test_priority_numerical_slack import (
    MEASUREMENT_ROUNDOFF,
    NUMERICAL_MERIT_TOLERANCE,
    ORIENTATION_TOLERANCE,
    OUTSIDE_BUDGET_ERROR,
    run_merit_budget_steps,
)

import embodik as eik


@pytest.mark.parametrize("mode", [
    eik.TaskSolveMode.MIN_ERROR, eik.TaskSolveMode.SCALE, eik.TaskSolveMode.SCALE_ELASTIC,
])
@pytest.mark.parametrize("initial_error", [ORIENTATION_TOLERANCE, OUTSIDE_BUDGET_ERROR])
def test_tiny_step_cannot_bypass_priority_validation(tmp_path, mode, initial_error):
    maximum_error = max(initial_error, ORIENTATION_TOLERANCE + NUMERICAL_MERIT_TOLERANCE)
    for result in run_merit_budget_steps(tmp_path, mode, initial_error, lock_translation=True):
        # The locked slider leaves only a two-nanoradian yaw step. Even that
        # nonzero motion must respect the protected nonlinear merit budget.
        assert result.orientation_error <= maximum_error + MEASUREMENT_ROUNDOFF
