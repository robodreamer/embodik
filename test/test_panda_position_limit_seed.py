"""Panda: one position IK step from a seed on a joint upper limit vs. validation-style nudge.

Mirrors the validation_robot regression ``embodik_position_limit_seed_test`` (pendulum URDF)
using the Franka Panda from ``robot_descriptions``. Checks that pre-nudging the seed
inward is optional: at-limit and nudged seeds should both succeed and yield nearly
the same ``q_solution``.

See also: ``validation_robot/.../tests/embodik_position_limit_seed_test.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import embodik as eik

pytest.importorskip("robot_descriptions")

_PANDA_DEFAULT_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
_PANDA_EE_FRAME = "panda_hand"
# Interior gripper values so limit-nudge logic only affects the arm joint we pin to hi.
_PANDA_GRIPPER_Q = np.array([0.02, 0.02], dtype=float)

_POSITION_STEP_DT = 0.05
_SOLVER_DAMPING = 0.1
_NUDGE_EPS = 1e-3
_EE_TRANSLATION_OFFSET_X_M = 0.06
_LIMIT_SLACK_EPS = 1e-6
_ATOL_Q_SOLUTION_MATCH = 5e-3
_FRAME_TASK_PRIORITY = 1
_FRAME_TASK_WEIGHT = 1.0
_MAX_LINEAR_SPEED = 2.0
_MAX_ANGULAR_SPEED = 2.0
# Joint index in ``q`` to pin at upper limit (panda_joint1); see test docstring.
_JOINT_AT_UPPER_LIMIT_IDX = 0


def _nudge_joint_positions_inside_limits(
    q: Any,
    q_lo: Any,
    q_up: Any,
    eps: float = _NUDGE_EPS,
) -> tuple[np.ndarray, int]:
    """Same semantics as validation_robots.embodik_helpers.limits.nudge_joint_positions_inside_limits."""
    q_out = np.asarray(q, dtype=float).copy()
    lo = np.asarray(q_lo, dtype=float)
    hi = np.asarray(q_up, dtype=float)

    count = 0
    for i in range(min(len(q_out), len(lo), len(hi))):
        if np.isfinite(lo[i]) and q_out[i] <= lo[i] + eps:
            q_out[i] = lo[i] + eps
            count += 1
        elif np.isfinite(hi[i]) and q_out[i] >= hi[i] - eps:
            q_out[i] = hi[i] - eps
            count += 1

    return q_out, count


def test_panda_solve_position_step_at_joint_limit_vs_nudged_seed():
    from robot_descriptions.panda_description import URDF_PATH

    robot = eik.RobotModel(URDF_PATH, floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = _POSITION_STEP_DT
    solver.enable_position_limits(True)
    solver.set_damping(_SOLVER_DAMPING)

    q_lo, q_hi = robot.get_joint_limits()
    # Pin panda_joint1 at its upper limit. From the default posture, +X EE motion
    # stays feasible here; pinning e.g. joint6 at max can make +X infeasible (INFEASIBLE).
    j_limit = _JOINT_AT_UPPER_LIMIT_IDX
    q_at_hi = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_Q])
    q_at_hi[j_limit] = float(q_hi[j_limit])

    robot.update_configuration(q_at_hi)
    pose0 = robot.get_frame_pose(_PANDA_EE_FRAME)
    target_t = np.asarray(pose0.translation, dtype=float) + np.array(
        [_EE_TRANSLATION_OFFSET_X_M, 0.0, 0.0], dtype=float
    )

    task = solver.add_frame_task("ee", _PANDA_EE_FRAME)
    task.weight = _FRAME_TASK_WEIGHT
    task.priority = _FRAME_TASK_PRIORITY

    opts = eik.PositionStepOptions()
    opts.dt = _POSITION_STEP_DT
    opts.max_steps = 1
    opts.max_linear_speed = _MAX_LINEAR_SPEED
    opts.max_angular_speed = _MAX_ANGULAR_SPEED

    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(pose0.rotation, dtype=float)
    T[:3, 3] = target_t
    targets = [eik.TaskTarget("ee", T)]

    def one_step(q_seed: np.ndarray) -> np.ndarray:
        q = np.asarray(q_seed, dtype=float).copy()
        robot.update_configuration(q)
        result = solver.solve_position_step(q, targets, opts)
        assert result.status == eik.SolverStatus.SUCCESS, (
            f"{result.status=} {getattr(result, 'status_message', '')}"
        )
        return np.asarray(result.q_solution, dtype=float)

    q_sol_limit = one_step(q_at_hi)
    q_nudged, n_count = _nudge_joint_positions_inside_limits(
        q_at_hi, q_lo, q_hi, eps=_NUDGE_EPS
    )
    assert n_count == 1
    q_sol_nudged = one_step(q_nudged)

    assert np.all(q_sol_limit >= q_lo - _LIMIT_SLACK_EPS)
    assert np.all(q_sol_limit <= q_hi + _LIMIT_SLACK_EPS)
    assert np.all(q_sol_nudged >= q_lo - _LIMIT_SLACK_EPS)
    assert np.all(q_sol_nudged <= q_hi + _LIMIT_SLACK_EPS)

    np.testing.assert_allclose(
        q_sol_limit, q_sol_nudged, rtol=0.0, atol=_ATOL_Q_SOLUTION_MATCH
    )
