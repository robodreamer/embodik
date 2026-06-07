#!/usr/bin/env python3
"""Soft per-joint joint-space metric on the PRIMARY velocity solve.

This is the torso-vs-arm contribution knob: a diagonal metric W makes some joints
"more expensive" so they contribute less to the achieved EE motion, WITHOUT
abandoning the EE task (it stays tracked) and WITHOUT a hard exclusion (so the
feasibility override is preserved). Rate-independent, unlike velocity-limit caps.

knob semantics: higher weight on a joint group => that group moves less.
Neutral weights (all ones) must reproduce today's solve exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

import embodik as eik

pytest.importorskip("robot_descriptions")

_PANDA_DEFAULT_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.05, 0.05])
_PANDA_EE_FRAME = "panda_hand"


def _make_solver() -> tuple[eik.KinematicsSolver, np.ndarray, eik.RobotModel]:
    from robot_descriptions.panda_description import URDF_PATH

    robot = eik.RobotModel(URDF_PATH, floating_base=False)
    q = _PANDA_DEFAULT_Q.copy()
    robot.update_configuration(q)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    pose = robot.get_frame_pose(_PANDA_EE_FRAME)
    # 3-DOF POSITION task: leaves ample redundancy so suppressing the proximal
    # group still lets the distal joints achieve the target (the torso-vs-arm
    # situation: suppress one group, the other still has >= task-dim DOF).
    task = solver.add_frame_task("ee", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSITION)
    task.priority = 0
    task.weight = 1.0
    task.set_target_position(
        np.asarray(pose.translation, dtype=float) + np.array([0.04, 0.03, 0.02])
    )
    return solver, q, robot


# Proximal joints play the "torso-like" group to suppress; 3,4,5,6 remain (>3-DOF
# task) so the position target stays reachable.
_GROUP = [0, 1, 2]


def _dq(metric: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    solver, q, robot = _make_solver()
    if metric is not None:
        solver.set_joint_metric_weights(metric)
    res = solver.solve_velocity(q, apply_limits=True)
    dq = np.asarray(res.joint_velocities, dtype=float)
    jac = np.asarray(robot.get_frame_jacobian(_PANDA_EE_FRAME), dtype=float)
    return dq, (jac @ dq)[:3]  # joint velocities and achieved POSITION velocity


def test_neutral_metric_matches_baseline():
    base_dq, _ = _dq(None)
    ones_dq, _ = _dq(np.ones(9))
    np.testing.assert_allclose(ones_dq, base_dq, atol=1e-9)


def test_high_weight_group_contributes_less_but_target_met():
    base_dq, base_twist = _dq(None)
    metric = np.ones(9)
    for j in _GROUP:
        metric[j] = 25.0
    dq, twist = _dq(metric)

    base_share = np.linalg.norm(base_dq[_GROUP]) / (np.linalg.norm(base_dq) + 1e-12)
    share = np.linalg.norm(dq[_GROUP]) / (np.linalg.norm(dq) + 1e-12)
    assert share < base_share, f"suppressed group should move less: {share} !< {base_share}"

    # EE task still achieved: same commanded twist, just redistributed.
    np.testing.assert_allclose(twist, base_twist, atol=5e-3)


def test_monotonic_suppression():
    shares = []
    for w in (1.0, 5.0, 25.0, 100.0):
        metric = np.ones(9)
        for j in _GROUP:
            metric[j] = w
        dq, _ = _dq(metric)
        shares.append(np.linalg.norm(dq[_GROUP]) / (np.linalg.norm(dq) + 1e-12))
    diffs = np.diff(shares)
    assert np.all(diffs <= 1e-6), f"group share should decrease monotonically: {shares}"


def _make_pose_solver() -> tuple[eik.KinematicsSolver, np.ndarray, eik.RobotModel]:
    """6-DOF pose target: the distal joints [3,4,5,6] alone are underactuated for a
    full pose, so meeting it REQUIRES recruiting the proximal group."""
    from robot_descriptions.panda_description import URDF_PATH

    robot = eik.RobotModel(URDF_PATH, floating_base=False)
    q = _PANDA_DEFAULT_Q.copy()
    robot.update_configuration(q)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    pose = robot.get_frame_pose(_PANDA_EE_FRAME)
    task = solver.add_frame_task("ee", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSE)
    task.priority = 0
    task.weight = 1.0
    c, s = np.cos(0.6), np.sin(0.6)
    rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    task.set_target_position(np.asarray(pose.translation, float) + np.array([0.10, 0.07, 0.05]))
    task.set_target_orientation(np.asarray(pose.rotation, float) @ rot_z)
    return solver, q, robot


def _progress_along(dq, robot, bhat):
    jac = np.asarray(robot.get_frame_jacobian(_PANDA_EE_FRAME), float)
    return float((jac @ dq) @ bhat)


def test_feasibility_override_soft_recruits_better_than_hard_lock():
    """A suppressed group must still be RECRUITED when the task needs it (soft
    weight), unlike a hard lock. With a pose target the distal joints can't meet
    alone, the soft metric tracks better than locking the proximal group out — the
    feasibility override, and the reason the knob is a soft weight not an exclusion."""
    solver, q, robot = _make_pose_solver()
    base = np.asarray(solver.solve_velocity(q, apply_limits=True).joint_velocities, float)
    b = np.asarray(robot.get_frame_jacobian(_PANDA_EE_FRAME), float) @ base
    bhat = b / (np.linalg.norm(b) + 1e-12)

    # SOFT: moderate proximal suppression (low contribution, but recruitable).
    s_solver, s_q, s_robot = _make_pose_solver()
    metric = np.ones(9)
    for j in _GROUP:
        metric[j] = 8.0
    s_solver.set_joint_metric_weights(metric)
    soft = np.asarray(s_solver.solve_velocity(s_q, apply_limits=True).joint_velocities, float)

    # HARD: lock the proximal group out (velocity override = 0 -> exclusion).
    h_solver, h_q, h_robot = _make_pose_solver()
    for j in _GROUP:
        h_solver.set_joint_velocity_limit(j, 0.0)
    hard = np.asarray(h_solver.solve_velocity(h_q, apply_limits=True).joint_velocities, float)

    soft_prog = _progress_along(soft, s_robot, bhat)
    hard_prog = _progress_along(hard, h_robot, bhat)
    # Soft has a strictly larger feasible set than a hard lock, so it tracks at
    # least as well; recruiting the group makes it strictly better here. (The
    # *degree* is geometry-dependent and validated strongly on G1 in `limitdemo`.)
    assert soft_prog > hard_prog * 1.02, (
        f"soft metric must recruit + track better than a hard lock: "
        f"soft {soft_prog:.4f} vs hard {hard_prog:.4f}"
    )
    assert np.linalg.norm(soft[_GROUP]) > 1e-3, "suppressed group must still be recruited"
    assert np.linalg.norm(hard[_GROUP]) < 1e-9, "hard lock keeps the group at zero"


def test_metric_composes_with_velocity_lock():
    """Metric + a hard velocity lock coexist: locked joints stay zero, solve ok."""
    solver, q, robot = _make_solver()
    metric = np.ones(9)
    metric[3] = 10.0  # suppress a distal joint
    solver.set_joint_metric_weights(metric)
    solver.set_joint_velocity_limit(0, 0.0)  # hard-lock joint 0
    dq = np.asarray(solver.solve_velocity(q, apply_limits=True).joint_velocities, float)
    assert abs(dq[0]) < 1e-9, "hard-locked joint must stay zero even with a metric"
    assert np.linalg.norm(dq) > 1e-6, "solve should still produce motion"
