#!/usr/bin/env python3
"""Tests for example runtime IK helpers."""

from __future__ import annotations

import numpy as np

from examples.incubating.sungjoon_port_phase1.robust_ik_runtime import (
    clip_configuration,
    robust_solve_position_step,
)


class _Status:
    def __init__(self, name: str) -> None:
        self.name = name


class _Result:
    def __init__(self, name: str, *, q_solution=None, joint_velocities=None, status_message: str = "") -> None:
        self.status = _Status(name)
        self.q_solution = q_solution
        self.joint_velocities = joint_velocities
        self.status_message = status_message


class _Task:
    def __init__(self) -> None:
        self.pose_targets = []
        self.position_targets = []
        self.orientation_targets = []

    def set_target_pose(self, position, rotation) -> None:
        self.pose_targets.append((np.asarray(position, dtype=float), np.asarray(rotation, dtype=float)))

    def set_target_position(self, position) -> None:
        self.position_targets.append(np.asarray(position, dtype=float))

    def set_target_orientation(self, rotation) -> None:
        self.orientation_targets.append(np.asarray(rotation, dtype=float))


class _Solver:
    def __init__(self, *, step_result: _Result, velocity_result: _Result | None = None) -> None:
        self.dt = 0.1
        self._step_result = step_result
        self._velocity_result = velocity_result
        self.tasks = {"ee_task": _Task()}
        self.calls = []

    def solve_position_step(self, q_current, targets, options):
        self.calls.append(("step", np.asarray(q_current, dtype=float), len(targets), options.max_steps))
        return self._step_result

    def solve_velocity(self, q_current, apply_limits=True):
        self.calls.append(("velocity", np.asarray(q_current, dtype=float), bool(apply_limits)))
        return self._velocity_result

    def get_task(self, name: str) -> _Task:
        return self.tasks[name]


class _Robot:
    is_floating_base = False

    def integrate(self, q_current, dq, dt):
        return np.asarray(q_current, dtype=float) + float(dt) * np.asarray(dq, dtype=float)


class _FloatingRobot:
    is_floating_base = True


class _Target:
    def __init__(self) -> None:
        self.task_name = "ee_task"
        self.target_pose = np.eye(4, dtype=float)
        self.target_pose[:3, 3] = np.array([0.3, -0.1, 0.2], dtype=float)


class _Options:
    def __init__(self) -> None:
        self.max_steps = 3


def test_clip_configuration_normalizes_floating_base_quaternion() -> None:
    robot = _FloatingRobot()
    q = np.array([0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 5.0, -5.0], dtype=float)
    q_lo = np.array([-1.0] * q.size, dtype=float)
    q_hi = np.array([1.0] * q.size, dtype=float)
    clipped = clip_configuration(robot, q, q_lo, q_hi)
    assert np.allclose(clipped[7:], np.array([1.0, -1.0]))
    assert np.isclose(np.linalg.norm(clipped[3:7]), 1.0)


def test_robust_solve_position_step_falls_back_to_velocity_and_zeros_locked_indices() -> None:
    solver = _Solver(
        step_result=_Result("INVALID_INPUT"),
        velocity_result=_Result("SUCCESS", joint_velocities=np.array([1.0, 2.0, 3.0], dtype=float)),
    )
    robot = _Robot()
    target = _Target()
    options = _Options()
    out = robust_solve_position_step(
        robot=robot,
        solver=solver,
        q_current=np.zeros(3, dtype=float),
        targets=[target],
        options=options,
        q_lo=-np.ones(3, dtype=float),
        q_hi=np.ones(3, dtype=float),
        zero_velocity_indices=[1],
        fallback_status_names=("INVALID_INPUT",),
    )
    np.testing.assert_allclose(out.q_next, np.array([0.1, 0.0, 0.3], dtype=float))
    assert out.solver_result.status.name == "SUCCESS"
    assert solver.calls[0][0] == "step"
    assert solver.calls[1][0] == "velocity"
    assert len(solver.tasks["ee_task"].pose_targets) == 1


def test_robust_solve_position_step_holds_previous_configuration_on_non_finite() -> None:
    solver = _Solver(step_result=_Result("NON_FINITE_INPUT"))
    robot = _Robot()
    q_current = np.array([0.2, -0.1], dtype=float)
    out = robust_solve_position_step(
        robot=robot,
        solver=solver,
        q_current=q_current,
        targets=[_Target()],
        options=_Options(),
        q_lo=-np.ones(2, dtype=float),
        q_hi=np.ones(2, dtype=float),
    )
    np.testing.assert_allclose(out.q_next, q_current)
    assert out.solver_calls == 1
