#!/usr/bin/env python3
"""Tests for example runtime IK helpers."""

from __future__ import annotations

import numpy as np

from embodik.interactive_ik import (
    ConstrainedStepGuard,
    ConstraintBoundary,
    clip_configuration,
    robust_solve_position_step,
)


class _Status:
    def __init__(self, name: str) -> None:
        self.name = name


class _Result:
    def __init__(
        self,
        name: str,
        *,
        q_solution=None,
        joint_velocities=None,
        status_message: str = "",
        collision_rejection_count: int = 0,
        stall_escape_count: int = 0,
    ) -> None:
        self.status = _Status(name)
        self.q_solution = q_solution
        self.joint_velocities = joint_velocities
        self.status_message = status_message
        self.collision_rejection_count = collision_rejection_count
        self.stall_escape_count = stall_escape_count


class _Task:
    def __init__(self) -> None:
        self.pose_targets = []
        self.position_targets = []
        self.orientation_targets = []

    def set_target_pose(self, position, rotation) -> None:
        self.pose_targets.append(
            (np.asarray(position, dtype=float), np.asarray(rotation, dtype=float))
        )

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
        self.calls.append(
            ("step", np.asarray(q_current, dtype=float), len(targets), options.max_steps)
        )
        return self._step_result

    def solve_velocity(self, q_current, apply_limits=True):
        self.calls.append(("velocity", np.asarray(q_current, dtype=float), bool(apply_limits)))
        return self._velocity_result

    def get_task(self, name: str) -> _Task:
        return self.tasks[name]


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


def test_robust_solve_position_step_does_not_retry_or_post_clip() -> None:
    solver = _Solver(
        step_result=_Result("INVALID_INPUT", q_solution=np.array([2.0, -2.0, 0.5], dtype=float)),
        velocity_result=_Result("SUCCESS", joint_velocities=np.array([1.0, 2.0, 3.0], dtype=float)),
    )
    target = _Target()
    options = _Options()
    out = robust_solve_position_step(
        solver=solver,
        q_current=np.zeros(3, dtype=float),
        targets=[target],
        options=options,
    )
    np.testing.assert_allclose(out.q_next, np.array([2.0, -2.0, 0.5], dtype=float))
    assert out.solver_result.status.name == "INVALID_INPUT"
    assert [call[0] for call in solver.calls] == ["step"]
    assert len(solver.tasks["ee_task"].pose_targets) == 0


def test_robust_solve_position_step_holds_previous_configuration_on_non_finite() -> None:
    solver = _Solver(step_result=_Result("NON_FINITE_INPUT"))
    q_current = np.array([0.2, -0.1], dtype=float)
    out = robust_solve_position_step(
        solver=solver,
        q_current=q_current,
        targets=[_Target()],
        options=_Options(),
    )
    np.testing.assert_allclose(out.q_next, q_current)
    assert out.solver_calls == 1


def test_robust_solve_position_step_trusts_solver_q_solution_limits() -> None:
    solver = _Solver(step_result=_Result("SUCCESS", q_solution=np.array([1.2, -1.2], dtype=float)))
    out = robust_solve_position_step(
        solver=solver,
        q_current=np.zeros(2, dtype=float),
        targets=[_Target()],
        options=_Options(),
    )
    np.testing.assert_allclose(out.q_next, np.array([1.2, -1.2], dtype=float))
    assert out.solver_calls == 1


def test_robust_solve_position_step_accepts_solver_intervention_without_fallback() -> None:
    solver = _Solver(
        step_result=_Result(
            "INFEASIBLE",
            q_solution=np.array([0.4, -0.2], dtype=float),
            collision_rejection_count=1,
        ),
        velocity_result=_Result("SUCCESS", joint_velocities=np.array([1.0, 1.0], dtype=float)),
    )
    out = robust_solve_position_step(
        solver=solver,
        q_current=np.zeros(2, dtype=float),
        targets=[_Target()],
        options=_Options(),
    )
    np.testing.assert_allclose(out.q_next, np.array([0.4, -0.2], dtype=float))
    assert out.solver_result.status.name == "INFEASIBLE"
    assert [call[0] for call in solver.calls] == ["step"]


def test_robust_solve_position_step_applies_collision_violated_safe_hold() -> None:
    solver = _Solver(
        step_result=_Result(
            "COLLISION_VIOLATED",
            q_solution=np.array([0.15, -0.05], dtype=float),
        ),
    )
    out = robust_solve_position_step(
        solver=solver,
        q_current=np.zeros(2, dtype=float),
        targets=[_Target()],
        options=_Options(),
    )
    np.testing.assert_allclose(out.q_next, np.array([0.15, -0.05], dtype=float))
    assert out.solver_result.status.name == "COLLISION_VIOLATED"
    assert [call[0] for call in solver.calls] == ["step"]


def test_constrained_step_guard_restores_last_safe_on_boundary_violation() -> None:
    guard = ConstrainedStepGuard(np.array([0.1, 0.2], dtype=float))
    decision = guard.evaluate(
        q_candidate=np.array([0.4, 0.5], dtype=float),
        result=_Result("SUCCESS", joint_velocities=np.zeros(2, dtype=float)),
        max_task_error=0.01,
        boundaries=[
            ConstraintBoundary(
                "collision",
                value=0.02,
                minimum=0.035,
                enabled=True,
            )
        ],
        task_deadband=1e-4,
        constraints_enabled=True,
    )
    assert decision.restored_last_safe
    assert decision.restore_labels == ["collision"]
    np.testing.assert_allclose(decision.q_next, np.array([0.1, 0.2], dtype=float))


def test_constrained_step_guard_tracks_zero_motion_snap_threshold() -> None:
    guard = ConstrainedStepGuard(
        np.zeros(2, dtype=float),
        zero_motion_resync_frames=2,
        zero_motion_snap_frames=3,
    )
    result = _Result("SUCCESS", joint_velocities=np.zeros(2, dtype=float))
    boundary = ConstraintBoundary("CoM", value=0.0, minimum=0.0, enabled=True)

    first = guard.evaluate(
        q_candidate=np.zeros(2, dtype=float),
        result=result,
        max_task_error=0.01,
        boundaries=[boundary],
        task_deadband=1e-4,
        constraints_enabled=True,
    )
    second = guard.evaluate(
        q_candidate=np.zeros(2, dtype=float),
        result=result,
        max_task_error=0.01,
        boundaries=[boundary],
        task_deadband=1e-4,
        constraints_enabled=True,
    )
    third = guard.evaluate(
        q_candidate=np.zeros(2, dtype=float),
        result=result,
        max_task_error=0.01,
        boundaries=[boundary],
        task_deadband=1e-4,
        constraints_enabled=True,
    )

    assert not first.zero_motion_resync
    assert second.zero_motion_resync
    assert third.zero_motion_snap
