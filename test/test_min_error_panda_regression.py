#!/usr/bin/env python3
"""Regression tests for EE MIN_ERROR behavior on Panda."""

from __future__ import annotations

import numpy as np
import pytest

import embodik as eik

_PANDA_DEFAULT_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
_PANDA_GRIPPER_EXTRA = np.array([0.05, 0.05])
_PANDA_EE_FRAME = "panda_hand"


def _load_panda() -> tuple[eik.RobotModel, eik.KinematicsSolver]:
    """Load Panda model via robot_descriptions."""
    from robot_descriptions.panda_description import URDF_PATH

    robot = eik.RobotModel(URDF_PATH, floating_base=False)
    q_init = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
    robot.update_configuration(q_init)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    return robot, solver


def _narrow_limits(
    robot: eik.RobotModel, margin: float = 0.20
) -> tuple[np.ndarray, np.ndarray]:
    """Narrow Panda arm limits around the default configuration."""
    q_lower_orig, q_upper_orig = robot.get_joint_limits()
    q_lower = q_lower_orig.copy()
    q_upper = q_upper_orig.copy()
    q_center = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])

    for i in range(len(_PANDA_DEFAULT_Q)):
        q_lower[i] = max(q_lower_orig[i], q_center[i] - margin)
        q_upper[i] = min(q_upper_orig[i], q_center[i] + margin)

    robot.set_joint_limits(q_lower, q_upper)
    return q_lower, q_upper


def _run_tracking_trace(
    solve_mode: eik.TaskSolveMode,
    *,
    allow_fallback: bool = False,
    with_nullspace_bias: bool = False,
    steps: int = 80,
    offset: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float], eik.SolverResult]:
    """Run a short Panda tracking rollout and return error trace."""
    if offset is None:
        offset = np.array([0.08, 0.0, 0.0], dtype=float)

    robot, solver = _load_panda()
    _narrow_limits(robot, margin=0.20)
    solver.dt = 0.01

    q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
    robot.update_kinematics(q)

    task = solver.add_frame_task("ee_task", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSE)
    task.priority = 0
    task.weight = 10.0
    task.solve_mode = solve_mode
    task.allow_min_error_fallback = allow_fallback

    start_pos = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).translation)
    start_rot = np.array(task.current_orientation)
    target_pos = start_pos + offset
    task.set_target_pose(target_pos, start_rot)

    if with_nullspace_bias:
        posture = solver.add_posture_task("nullspace_bias")
        posture.priority = 1
        posture.weight = 0.1
        posture.solve_mode = eik.TaskSolveMode.MIN_ERROR
        q_bias = q.copy()
        q_bias[:7] += np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1])
        posture.set_target_configuration(q_bias)

    errors: list[float] = []
    last_result = None
    for _ in range(steps):
        last_result = solver.solve_velocity(q)
        dq = np.array(last_result.joint_velocities)
        # Examples use explicit Euler integration for velocity solves.
        q = q + dq * solver.dt
        q_lower, q_upper = robot.get_joint_limits()
        q = np.clip(q, q_lower, q_upper)
        robot.update_configuration(q)
        ee_now = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).translation)
        errors.append(float(np.linalg.norm(target_pos - ee_now)))

    assert last_result is not None
    return np.asarray(errors), [float(x) for x in last_result.task_scales], last_result


def test_min_error_mode_keeps_making_progress_on_panda():
    """MIN_ERROR should significantly reduce the EE position error."""
    pytest.importorskip("robot_descriptions.panda_description")
    errors, _, result = _run_tracking_trace(eik.TaskSolveMode.MIN_ERROR)

    # Must reduce error by a meaningful fraction over the rollout.
    assert errors[-1] < 0.90 * errors[0], (
        f"MIN_ERROR progress too small: initial={errors[0]:.5f}, final={errors[-1]:.5f}"
    )
    assert result.task_modes_effective == [eik.TaskSolveMode.MIN_ERROR]


def test_min_error_not_significantly_worse_than_scale_for_same_setup():
    """MIN_ERROR should not be dramatically slower than SCALE in this setup."""
    pytest.importorskip("robot_descriptions.panda_description")

    errors_scale, _, _ = _run_tracking_trace(eik.TaskSolveMode.SCALE)
    errors_min_error, _, result_min_error = _run_tracking_trace(eik.TaskSolveMode.MIN_ERROR)

    # Allow some degradation (different objective behavior), but guard against
    # pathological sluggishness/stalling regressions.
    assert errors_min_error[-1] <= errors_scale[-1] + 0.015, (
        f"MIN_ERROR final error regressed: min_error={errors_min_error[-1]:.5f}, "
        f"scale={errors_scale[-1]:.5f}"
    )
    assert result_min_error.task_modes_effective == [eik.TaskSolveMode.MIN_ERROR]


def test_min_error_with_nullspace_bias_still_progresses():
    """MIN_ERROR EE should still progress when nullspace bias task is active."""
    pytest.importorskip("robot_descriptions.panda_description")
    errors, _, result = _run_tracking_trace(
        eik.TaskSolveMode.MIN_ERROR,
        with_nullspace_bias=True,
    )

    assert errors[-1] < 0.95 * errors[0], (
        f"MIN_ERROR+nullspace progress too small: initial={errors[0]:.5f}, "
        f"final={errors[-1]:.5f}"
    )
    assert result.task_modes_effective[0] == eik.TaskSolveMode.MIN_ERROR
