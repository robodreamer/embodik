#!/usr/bin/env python3
"""Reproduction test for joint-6 self-drive bug under narrowed limits.

Bug: When joint limits are narrowed (e.g. to 50% of original range) and some
joints hit saturation during EE tracking, joint 6 (Panda wrist) starts
rotating toward its upper limit on its own, even when the task direction does
not require it. This breaks task directional consistency.

Root cause hypothesis: kMinBoundFraction (0.10) in calculate_velocity_box_constraint
guarantees 10% velocity headroom *toward* each limit even when a joint is
already at that limit. The SNS solver exploits this headroom on unconstrained
joints (particularly joint 6) to minimize task error, causing spurious drift.
"""

from __future__ import annotations

import numpy as np
import pytest
import embodik as eik

_PANDA_DEFAULT_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
_PANDA_GRIPPER = np.array([0.05, 0.05])
_PANDA_EE_FRAME = "panda_hand"


def _setup(limit_scale: float = 0.4):
    """Load Panda, narrow limits, return (robot, solver, q, q_lower, q_upper)."""
    from robot_descriptions.panda_description import URDF_PATH

    robot = eik.RobotModel(URDF_PATH, floating_base=False)
    q_init = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER])
    robot.update_configuration(q_init)

    q_lo_orig, q_hi_orig = robot.get_joint_limits()
    center = 0.5 * (q_lo_orig + q_hi_orig)
    half = 0.5 * (q_hi_orig - q_lo_orig) * limit_scale
    q_lo = center - half
    q_hi = center + half
    robot.set_joint_limits(q_lo, q_hi)

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)

    return robot, solver, q_init.copy(), q_lo, q_hi


def _track_ee_and_record_joint6(robot, solver, q, q_lo, q_hi, ee_offset, steps=300, gain=60.0):
    """Drive EE toward offset, record per-step joint-6 values and velocities."""
    from embodik.utils import compute_pose_error

    task = solver.add_frame_task("ee", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSE)
    task.priority = 0
    task.weight = 1.0

    robot.update_kinematics(q)
    ee_start = robot.get_frame_pose(_PANDA_EE_FRAME)
    target_pos = ee_start.translation + ee_offset
    target_rot = ee_start.rotation

    j6_values = []
    j6_velocities = []
    saturated_joints_per_step = []

    for _ in range(steps):
        current_pose = robot.get_frame_pose(_PANDA_EE_FRAME)
        target_pose = eik.Rt(R=target_rot, t=target_pos)
        error = compute_pose_error(current_pose, target_pose)
        vel = np.concatenate([gain * error[:3], gain * error[3:]])
        vel[:3] = np.clip(vel[:3], -0.5, 0.5)
        vel[3:] = np.clip(vel[3:], -0.5, 0.5)
        task.set_target_velocity(vel)

        result = solver.solve_velocity(q, apply_limits=True)
        dq = result.joint_velocities
        q = q + dq * solver.dt
        q = np.clip(q, q_lo, q_hi)
        robot.update_configuration(q)

        j6_values.append(q[6])
        j6_velocities.append(dq[6])
        saturated_joints_per_step.append(list(result.saturated_joints))

    solver.clear_tasks()
    return np.array(j6_values), np.array(j6_velocities), saturated_joints_per_step


class TestJoint6DriftBug:
    """Tests that demonstrate joint-6 drifting toward its upper limit."""

    def test_x_positive_drive_causes_j6_drift(self):
        """Drive EE in +X. Joint 6 should stay near its initial value.

        With narrowed limits (40%), other joints hit limits. Under the bug,
        joint 6 drifts monotonically toward its upper limit.
        """
        robot, solver, q, q_lo, q_hi = _setup(limit_scale=0.4)
        j6_init = q[6]

        j6_vals, j6_vels, sat = _track_ee_and_record_joint6(
            robot,
            solver,
            q,
            q_lo,
            q_hi,
            ee_offset=np.array([0.15, 0.0, 0.0]),
            steps=400,
        )

        j6_drift = j6_vals[-1] - j6_init
        j6_max_drift = np.max(np.abs(j6_vals - j6_init))

        # Count steps where j6 is being pushed toward upper limit
        j6_toward_upper = np.sum(j6_vels > 1e-4)

        print(f"\n--- Joint-6 drift diagnostics (+X drive) ---")
        print(f"  j6_init:          {j6_init:.4f}")
        print(f"  j6_final:         {j6_vals[-1]:.4f}")
        print(f"  j6_drift:         {j6_drift:.4f}")
        print(f"  j6_max_drift:     {j6_max_drift:.4f}")
        print(f"  j6_upper_limit:   {q_hi[6]:.4f}")
        print(f"  j6_lower_limit:   {q_lo[6]:.4f}")
        print(f"  steps_toward_upper: {j6_toward_upper} / {len(j6_vels)}")
        print(f"  saturated steps:  {sum(1 for s in sat if len(s) > 0)}")

        # If bug is present: j6_drift will be large and positive
        # (toward upper limit), and j6_toward_upper will be high.
        # This test documents the bug — it should FAIL once the bug is fixed.
        # The assertion below is the DESIRED behavior after fix:
        assert j6_max_drift < 0.15, (
            f"Joint 6 drifted {j6_max_drift:.4f} rad from init — "
            f"expected < 0.15 rad for a pure +X EE target."
        )

    def test_y_positive_drive_causes_j6_drift(self):
        """Drive EE in +Y with narrowed limits."""
        robot, solver, q, q_lo, q_hi = _setup(limit_scale=0.4)
        j6_init = q[6]

        j6_vals, j6_vels, sat = _track_ee_and_record_joint6(
            robot,
            solver,
            q,
            q_lo,
            q_hi,
            ee_offset=np.array([0.0, 0.15, 0.0]),
            steps=400,
        )

        j6_drift = j6_vals[-1] - j6_init
        j6_max_drift = np.max(np.abs(j6_vals - j6_init))
        j6_toward_upper = np.sum(j6_vels > 1e-4)

        print(f"\n--- Joint-6 drift diagnostics (+Y drive) ---")
        print(f"  j6_init:          {j6_init:.4f}")
        print(f"  j6_final:         {j6_vals[-1]:.4f}")
        print(f"  j6_drift:         {j6_drift:.4f}")
        print(f"  j6_max_drift:     {j6_max_drift:.4f}")
        print(f"  j6_upper_limit:   {q_hi[6]:.4f}")
        print(f"  j6_lower_limit:   {q_lo[6]:.4f}")
        print(f"  steps_toward_upper: {j6_toward_upper} / {len(j6_vels)}")

        assert j6_max_drift < 0.15, (
            f"Joint 6 drifted {j6_max_drift:.4f} rad from init — "
            f"expected < 0.15 rad for a pure +Y EE target."
        )

    def test_no_drift_with_original_limits(self):
        """With original limits (scale=1.0), joint 6 should not drift much."""
        robot, solver, q, q_lo, q_hi = _setup(limit_scale=1.0)
        j6_init = q[6]

        j6_vals, j6_vels, _ = _track_ee_and_record_joint6(
            robot,
            solver,
            q,
            q_lo,
            q_hi,
            ee_offset=np.array([0.15, 0.0, 0.0]),
            steps=400,
        )

        j6_max_drift = np.max(np.abs(j6_vals - j6_init))
        print(f"\n--- Joint-6 drift (original limits, +X) ---")
        print(f"  j6_max_drift: {j6_max_drift:.4f}")

        assert (
            j6_max_drift < 0.15
        ), f"Joint 6 drifted {j6_max_drift:.4f} rad even with original limits."


class TestKMinBoundFractionFixed:
    """Verify kMinBoundFraction no longer injects velocity toward limits."""

    def test_no_headroom_toward_upper_limit(self):
        """When at upper limit, velocity bound toward limit must be <= 0."""
        robot, solver, q, q_lo, q_hi = _setup(limit_scale=0.4)

        q[4] = q_hi[4]
        robot.update_configuration(q)

        vel_limits = robot.get_velocity_limits()
        margin_to_upper = q_hi[4] - q[4]  # ~0
        margin_to_lower = q[4] - q_lo[4]

        lower_vel, upper_vel = solver.calculate_velocity_box_constraint(
            margin_to_lower, margin_to_upper, vel_limits[4], 15.0, solver.dt
        )

        print(f"\n--- Velocity bounds at upper limit (joint 4) ---")
        print(f"  upper_vel_bound:  {upper_vel:.6f}")
        print(f"  10% vel_limit:    {0.1 * vel_limits[4]:.6f}")

        assert upper_vel <= 0.0, (
            f"upper_vel={upper_vel:.6f} > 0 when AT upper limit — "
            f"kMinBoundFraction should not provide headroom toward limit."
        )
        assert lower_vel < 0.0, "Should still have headroom AWAY from limit."

    def test_no_headroom_toward_lower_limit(self):
        """When at lower limit, velocity bound toward limit must be >= 0."""
        robot, solver, q, q_lo, q_hi = _setup(limit_scale=0.4)

        q[4] = q_lo[4]
        robot.update_configuration(q)

        vel_limits = robot.get_velocity_limits()
        margin_to_upper = q_hi[4] - q[4]
        margin_to_lower = q[4] - q_lo[4]  # ~0

        lower_vel, upper_vel = solver.calculate_velocity_box_constraint(
            margin_to_lower, margin_to_upper, vel_limits[4], 15.0, solver.dt
        )

        print(f"\n--- Velocity bounds at lower limit (joint 4) ---")
        print(f"  lower_vel_bound:  {lower_vel:.6f}")

        assert lower_vel >= 0.0, (
            f"lower_vel={lower_vel:.6f} < 0 when AT lower limit — "
            f"kMinBoundFraction should not provide headroom toward limit."
        )
        assert upper_vel > 0.0, "Should still have headroom AWAY from limit."

    def test_headroom_in_both_dirs_when_centered(self):
        """When well inside both limits, softening works in both directions."""
        robot, solver, q, q_lo, q_hi = _setup(limit_scale=0.4)
        solver.enable_saturation_exit_behavior(True)

        vel_limits = robot.get_velocity_limits()
        margin_lo = q[4] - q_lo[4]
        margin_hi = q_hi[4] - q[4]

        lower_vel, upper_vel = solver.calculate_velocity_box_constraint(
            margin_lo, margin_hi, vel_limits[4], 15.0, solver.dt
        )

        min_vel = 0.1 * vel_limits[4]
        assert lower_vel <= -min_vel, "Should have full softening AWAY from both limits."
        assert upper_vel >= min_vel, "Should have full softening AWAY from both limits."
