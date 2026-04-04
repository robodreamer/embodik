#!/usr/bin/env python3
"""Tests for elastic band robustness when starting from violated states.

Simulates hardware startup scenarios where the robot may be in a
configuration that violates joint limits and/or collision constraints.
The elastic band should allow the solver to work immediately and
gradually recover toward nominal constraints.

Scenarios:
1. Joint limit violation: joints beyond their limits
2. Combined: joints at limits + collision margin violated (narrowed limits)
3. Recovery: solver should gradually return to nominal constraints
"""

from __future__ import annotations

import numpy as np
import pytest

import embodik as eik

_PANDA_DEFAULT_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
_PANDA_GRIPPER_EXTRA = np.array([0.05, 0.05])
_PANDA_EE_FRAME = "panda_hand"


def _load_panda():
    from robot_descriptions.panda_description import URDF_PATH

    robot = eik.RobotModel(URDF_PATH, floating_base=False)
    q_init = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
    robot.update_configuration(q_init)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    return robot, solver


def _narrow_limits(robot, margin=0.15):
    q_lower_orig, q_upper_orig = robot.get_joint_limits()
    q_lower = q_lower_orig.copy()
    q_upper = q_upper_orig.copy()
    q_center = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
    for i in range(len(_PANDA_DEFAULT_Q)):
        q_lower[i] = max(q_lower_orig[i], q_center[i] - margin)
        q_upper[i] = min(q_upper_orig[i], q_center[i] + margin)
    robot.set_joint_limits(q_lower, q_upper)
    return q_lower, q_upper


class TestJointLimitViolationStartup:
    """Scenario: robot starts with joints beyond their limits."""

    def test_solver_works_with_joints_beyond_limits(self):
        """Elastic band should let the solver produce motion even when
        joints start beyond their nominal limits."""
        robot, solver = _load_panda()
        q_lower, q_upper = _narrow_limits(robot, margin=0.10)

        # Push joint 0 and 2 beyond their upper limits
        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        q[0] = q_upper[0] + 0.02  # 0.02 rad beyond upper limit
        q[2] = q_upper[2] + 0.03  # 0.03 rad beyond upper limit
        robot.update_configuration(q)
        robot.update_kinematics(q)

        # Without elastic band
        solver.clear_tasks()
        task = solver.add_frame_task("ee", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 10.0
        robot.update_kinematics(q)
        ee_pos = np.array(task.current_position)
        ee_rot = np.array(task.current_orientation)
        # Target back toward center (should be reachable)
        q_center = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q_center)
        robot.update_kinematics(q_center)
        center_pos = np.array(task.current_position)
        center_rot = np.array(task.current_orientation)
        robot.update_configuration(q)
        robot.update_kinematics(q)
        task.set_target_pose(center_pos, center_rot)

        result_no_elastic = solver.solve_velocity(q)
        dq_no = float(np.linalg.norm(result_no_elastic.joint_velocities))

        # With elastic band
        solver.enable_elastic_band(delta_max=0.05)
        result_elastic = solver.solve_velocity(q)
        dq_elastic = float(np.linalg.norm(result_elastic.joint_velocities))

        print(f"\nJoints beyond limits:")
        print(f"  Without elastic: dq={dq_no:.6f}, status={result_no_elastic.status}")
        print(f"  With elastic:    dq={dq_elastic:.6f}, status={result_elastic.status}")

        # Elastic band should produce at least as much motion
        assert dq_elastic >= dq_no * 0.5, (
            f"Elastic band should help: dq_elastic={dq_elastic} vs dq_no={dq_no}"
        )
        assert np.all(np.isfinite(result_elastic.joint_velocities))

        solver.disable_elastic_band()
        solver.clear_tasks()

    def test_joints_recover_toward_limits_over_time(self):
        """Starting beyond limits, the solver should drive joints back
        within limits as the elastic band decays."""
        robot, solver = _load_panda()
        q_lower, q_upper = _narrow_limits(robot, margin=0.10)

        # Start with joint 1 beyond upper limit
        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        q[1] = q_upper[1] + 0.015
        robot.update_configuration(q)
        robot.update_kinematics(q)

        solver.enable_elastic_band(delta_max=0.05)
        solver.clear_tasks()
        task = solver.add_frame_task("ee", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 10.0

        # Target at center — solver should drive toward it
        q_center = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q_center)
        robot.update_kinematics(q_center)
        center_pos = np.array(task.current_position)
        center_rot = np.array(task.current_orientation)
        robot.update_configuration(q)
        robot.update_kinematics(q)
        task.set_target_pose(center_pos, center_rot)

        initial_violation = max(0, q[1] - q_upper[1])
        q_run = q.copy()
        for _ in range(100):
            result = solver.solve_velocity(q_run)
            dq = result.joint_velocities
            q_run = robot.integrate(q_run, dq, solver.dt)
            # Don't clip — let solver handle it naturally
            robot.update_kinematics(q_run)

        final_violation = max(0, q_run[1] - q_upper[1])
        print(f"\nJoint 1 violation: initial={initial_violation:.4f}, "
              f"final={final_violation:.4f}")

        # Violation should decrease (or at least not worsen)
        assert final_violation <= initial_violation + 1e-4, (
            f"Joint should recover: violation went from {initial_violation:.4f} "
            f"to {final_violation:.4f}"
        )

        solver.disable_elastic_band()
        solver.clear_tasks()


class TestCombinedViolationStartup:
    """Scenario: joints at limits + many saturated DOFs simultaneously."""

    def test_elastic_band_helps_with_many_saturated_joints(self):
        """With very narrow limits and many joints saturated, elastic band
        should still allow some task progress."""
        robot, solver = _load_panda()
        _narrow_limits(robot, margin=0.05)  # Very narrow

        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q)
        robot.update_kinematics(q)

        solver.clear_tasks()
        task = solver.add_frame_task("ee", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 10.0
        robot.update_kinematics(q)
        ee_pos = np.array(task.current_position)
        ee_rot = np.array(task.current_orientation)
        task.set_target_pose(ee_pos + np.array([0.03, 0.0, 0.0]), ee_rot)

        q_lower, q_upper = robot.get_joint_limits()

        # Baseline: no elastic band
        infeasible_baseline = 0
        q_run = q.copy()
        for _ in range(50):
            result = solver.solve_velocity(q_run)
            if result.status == eik.SolverStatus.INFEASIBLE:
                infeasible_baseline += 1
            dq = result.joint_velocities
            q_run = robot.integrate(q_run, dq, solver.dt)
            q_run = np.clip(q_run, q_lower, q_upper)
            robot.update_kinematics(q_run)

        # Elastic band
        robot.update_configuration(q)
        robot.update_kinematics(q)
        solver.enable_elastic_band(delta_max=0.05)
        task.set_target_pose(ee_pos + np.array([0.03, 0.0, 0.0]), ee_rot)

        infeasible_elastic = 0
        q_run = q.copy()
        for _ in range(50):
            result = solver.solve_velocity(q_run)
            if result.status == eik.SolverStatus.INFEASIBLE:
                infeasible_elastic += 1
            dq = result.joint_velocities
            q_run = robot.integrate(q_run, dq, solver.dt)
            q_run = np.clip(q_run, q_lower, q_upper)
            robot.update_kinematics(q_run)

        print(f"\nVery narrow limits (±0.05 rad):")
        print(f"  Baseline infeasible: {infeasible_baseline}")
        print(f"  Elastic infeasible:  {infeasible_elastic}")

        assert infeasible_elastic <= infeasible_baseline, (
            f"Elastic should not increase infeasible: "
            f"{infeasible_elastic} > {infeasible_baseline}"
        )

        solver.disable_elastic_band()
        solver.clear_tasks()


class TestGradualRecovery:
    """Verify that elastic band gradually restores nominal constraints."""

    def test_elastic_deltas_converge_to_zero_after_reaching_target(self):
        """Once target is reached, elastic deltas should decay to near-zero."""
        robot, solver = _load_panda()
        _narrow_limits(robot, margin=0.15)

        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q)
        robot.update_kinematics(q)

        solver.enable_elastic_band(delta_max=0.05)
        solver.clear_tasks()
        task = solver.add_frame_task("ee", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 10.0
        robot.update_kinematics(q)
        ee_pos = np.array(task.current_position)
        ee_rot = np.array(task.current_orientation)

        # Phase 1: drive toward challenging target
        task.set_target_pose(ee_pos + np.array([0.08, 0.0, 0.0]), ee_rot)
        q_lower, q_upper = robot.get_joint_limits()
        for _ in range(80):
            result = solver.solve_velocity(q)
            dq = result.joint_velocities
            q = robot.integrate(q, dq, solver.dt)
            q = np.clip(q, q_lower, q_upper)
            robot.update_kinematics(q)

        peak_delta = solver.elastic_band_max_delta()

        # Phase 2: set target to current position (already reached)
        robot.update_kinematics(q)
        task.set_target_pose(np.array(task.current_position),
                             np.array(task.current_orientation))

        for _ in range(50):
            result = solver.solve_velocity(q)
            dq = result.joint_velocities
            q = robot.integrate(q, dq, solver.dt)
            q = np.clip(q, q_lower, q_upper)
            robot.update_kinematics(q)

        final_delta = solver.elastic_band_max_delta()
        print(f"\nDelta convergence: peak={peak_delta:.6f}, final={final_delta:.6f}")

        assert final_delta < peak_delta * 0.5 or final_delta < 0.005, (
            f"Deltas should converge: peak={peak_delta:.6f}, final={final_delta:.6f}"
        )

        solver.disable_elastic_band()
        solver.clear_tasks()

    def test_no_nan_or_crash_with_extreme_initial_state(self):
        """Solver should not crash or produce NaN even with extreme
        initial violations."""
        robot, solver = _load_panda()
        _narrow_limits(robot, margin=0.10)

        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        q_lower, q_upper = robot.get_joint_limits()

        # Push multiple joints well beyond limits
        for i in range(min(5, len(_PANDA_DEFAULT_Q))):
            q[i] = q_upper[i] + 0.05  # 0.05 rad beyond limit

        robot.update_configuration(q)
        robot.update_kinematics(q)

        solver.enable_elastic_band(delta_max=0.1)
        solver.clear_tasks()
        task = solver.add_frame_task("ee", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 10.0
        robot.update_kinematics(q)
        ee_pos = np.array(task.current_position)
        ee_rot = np.array(task.current_orientation)
        task.set_target_pose(ee_pos + np.array([0.05, 0.0, 0.0]), ee_rot)

        # Run many steps — should never crash or produce NaN
        for step in range(100):
            result = solver.solve_velocity(q)
            dq = result.joint_velocities
            assert np.all(np.isfinite(dq)), f"NaN at step {step}"
            q = robot.integrate(q, dq, solver.dt)
            robot.update_kinematics(q)

            deltas = solver.elastic_band_deltas()
            assert np.all(np.isfinite(deltas)), f"NaN in deltas at step {step}"

        solver.disable_elastic_band()
        solver.clear_tasks()
