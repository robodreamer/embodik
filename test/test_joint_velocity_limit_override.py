#!/usr/bin/env python3
"""Per-joint velocity-limit override on the solver.

This is the reuse lever for the torso-vs-arm contribution knob: shrinking a
joint's velocity cap throttles it in the velocity QP, so the SNS/limit-avoidance
machinery recruits the other joints (e.g. torso) to keep tracking the EE task.
Neutral (no override) must reproduce today's behaviour exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

import embodik as eik

pytest.importorskip("robot_descriptions")

_PANDA_DEFAULT_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.05, 0.05])
_PANDA_EE_FRAME = "panda_hand"


def _make_solver() -> tuple[eik.KinematicsSolver, np.ndarray]:
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
    task.set_target_position(
        np.asarray(pose.translation, dtype=float) + np.array([0.08, 0.05, 0.04])
    )
    task.set_target_orientation(np.asarray(pose.rotation, dtype=float))
    return solver, q


def _baseline_dq() -> np.ndarray:
    solver, q = _make_solver()
    return np.asarray(solver.solve_velocity(q, apply_limits=True).joint_velocities, float)


def test_override_caps_targeted_joint_velocity():
    base = _baseline_dq()
    j = int(np.argmax(np.abs(base)))
    cap = 0.25 * abs(base[j])
    assert cap > 1e-6, "baseline must move the targeted joint for a meaningful test"

    solver, q = _make_solver()
    solver.set_joint_velocity_limit(j, cap)
    dq = np.asarray(solver.solve_velocity(q, apply_limits=True).joint_velocities, float)

    assert abs(dq[j]) <= cap * 1.05, f"joint {j} not throttled: |dq|={abs(dq[j])} > cap={cap}"
    assert abs(dq[j]) < abs(base[j]), "override should reduce the joint's velocity"


def test_neutral_no_override_matches_baseline():
    base = _baseline_dq()
    solver, q = _make_solver()
    j = int(np.argmax(np.abs(base)))
    solver.set_joint_velocity_limit(j, 0.25 * abs(base[j]))
    solver.clear_joint_velocity_limit_overrides()
    dq = np.asarray(solver.solve_velocity(q, apply_limits=True).joint_velocities, float)
    np.testing.assert_allclose(dq, base, atol=1e-9)
