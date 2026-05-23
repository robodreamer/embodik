"""Tests for acceleration-level constraint cascading in velocity IK.

Experiment 2: Add acceleration limits that tighten velocity bounds based on
the previous tick's joint velocities, inspired by RB-Y1 SDK's cascaded
constraint integration: position → velocity → acceleration bounds.

This produces smoother joint velocity profiles (lower jerk) without
degrading tracking accuracy, especially during fast direction changes.
"""

import numpy as np
import pytest

import embodik as eik

PANDA_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])


@pytest.fixture
def panda_setup():
    pytest.importorskip("robot_descriptions.panda_description")
    from robot_descriptions.panda_description import URDF_PATH

    robot = eik.RobotModel(URDF_PATH, floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    return robot, solver


class TestAccelerationConstraints:

    def test_acceleration_limits_api_exists(self, panda_setup):
        """KinematicsSolver should expose acceleration limit methods."""
        _, solver = panda_setup
        assert hasattr(
            solver, "enable_acceleration_limits"
        ), "KinematicsSolver should have enable_acceleration_limits method"
        assert hasattr(
            solver, "set_acceleration_limits"
        ), "KinematicsSolver should have set_acceleration_limits method"

    def test_acceleration_limits_validate_shape_and_values(self, panda_setup):
        """Acceleration limits should reject invalid vectors before solving."""
        robot, solver = panda_setup

        with pytest.raises(ValueError, match="size nv"):
            solver.set_acceleration_limits(np.ones(robot.nv - 1))

        bad_limits = np.full(robot.nv, 1.0)
        bad_limits[0] = np.nan
        with pytest.raises(ValueError, match="finite and non-negative"):
            solver.set_acceleration_limits(bad_limits)

        bad_limits = np.full(robot.nv, 1.0)
        bad_limits[0] = -1.0
        with pytest.raises(ValueError, match="finite and non-negative"):
            solver.set_acceleration_limits(bad_limits)

    def test_acceleration_limits_bound_first_enabled_tick_from_rest(self, panda_setup):
        """Enabling acceleration limits should constrain the first solve from zero velocity."""
        robot, solver = panda_setup
        task = solver.add_frame_task("ee", "panda_hand", eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 1.0
        solver.enable_position_limits(True)

        q = PANDA_HOME.copy()
        robot.update_configuration(q)
        ee_pose = robot.get_frame_pose("panda_hand")
        target_pos = ee_pose.translation.copy()
        target_pos[0] += 0.3
        T = ee_pose.homogeneous()
        task.set_target_pose(target_pos, T[:3, :3])

        a_max = 2.0
        solver.set_acceleration_limits(np.full(robot.nv, a_max))
        solver.enable_acceleration_limits(True)
        result = solver.solve_velocity(q, apply_limits=True)

        assert result.status == eik.SolverStatus.SUCCESS
        assert np.max(np.abs(np.asarray(result.joint_velocities))) <= a_max * solver.dt * 1.02

    def test_acceleration_limits_reduce_jerk(self, panda_setup):
        """With acceleration limits, max joint velocity jump (jerk proxy)
        should be smaller than without.
        """
        robot, solver = panda_setup
        task = solver.add_frame_task("ee", "panda_hand", eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 1.0
        solver.enable_position_limits(True)

        # Target that requires a fast direction change
        q_start = PANDA_HOME.copy()
        robot.update_configuration(q_start)
        ee_pose = robot.get_frame_pose("panda_hand")
        base_pos = ee_pose.translation.copy()
        T = ee_pose.homogeneous()
        base_rot = T[:3, :3]

        n_steps = 150

        def run_trajectory(accel_enabled):
            q = q_start.copy()
            velocities = []
            # Phase 1: reach right
            target_pos = base_pos.copy()
            target_pos[1] += 0.15
            task.set_target_pose(target_pos, base_rot)
            for _ in range(n_steps // 2):
                robot.update_configuration(q)
                result = solver.solve_velocity(q, apply_limits=True)
                if result.status != eik.SolverStatus.SUCCESS:
                    break
                dq = np.array(result.joint_velocities)
                velocities.append(dq.copy())
                q = q + dq * solver.dt

            # Phase 2: sharp reversal — reach left
            target_pos = base_pos.copy()
            target_pos[1] -= 0.15
            task.set_target_pose(target_pos, base_rot)
            for _ in range(n_steps // 2):
                robot.update_configuration(q)
                result = solver.solve_velocity(q, apply_limits=True)
                if result.status != eik.SolverStatus.SUCCESS:
                    break
                dq = np.array(result.joint_velocities)
                velocities.append(dq.copy())
                q = q + dq * solver.dt

            return velocities

        # Without acceleration limits
        solver.enable_acceleration_limits(False)
        vel_no_accel = run_trajectory(False)

        # With acceleration limits (conservative: 10 rad/s^2 per joint)
        solver.enable_acceleration_limits(True)
        nv = robot.nv
        accel_limits = np.full(nv, 10.0)
        solver.set_acceleration_limits(accel_limits)
        vel_with_accel = run_trajectory(True)

        def max_jerk(velocities):
            if len(velocities) < 2:
                return 0.0
            jumps = [
                np.max(np.abs(velocities[i + 1] - velocities[i]))
                for i in range(len(velocities) - 1)
            ]
            return max(jumps)

        jerk_no_accel = max_jerk(vel_no_accel)
        jerk_with_accel = max_jerk(vel_with_accel)

        # Acceleration limits should reduce the max velocity jump
        assert jerk_with_accel < jerk_no_accel, (
            f"Jerk with accel limits ({jerk_with_accel:.4f}) should be less "
            f"than without ({jerk_no_accel:.4f})"
        )

    def test_acceleration_limits_bound_velocity_change(self, panda_setup):
        """Each tick's velocity change per joint should not exceed a_max * dt."""
        robot, solver = panda_setup
        task = solver.add_frame_task("ee", "panda_hand", eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 1.0
        solver.enable_position_limits(True)

        nv = robot.nv
        a_max = 5.0  # rad/s^2
        solver.enable_acceleration_limits(True)
        solver.set_acceleration_limits(np.full(nv, a_max))

        q = PANDA_HOME.copy()
        robot.update_configuration(q)
        ee_pose = robot.get_frame_pose("panda_hand")
        target_pos = ee_pose.translation.copy()
        target_pos[0] += 0.2  # large reach
        T = ee_pose.homogeneous()
        task.set_target_pose(target_pos, T[:3, :3])

        prev_dq = np.zeros(nv)
        dt = solver.dt
        max_accel_violation = 0.0

        for step in range(100):
            robot.update_configuration(q)
            result = solver.solve_velocity(q, apply_limits=True)
            if result.status != eik.SolverStatus.SUCCESS:
                break
            dq = np.array(result.joint_velocities)

            if step > 0:  # skip first step (no previous velocity)
                accel = np.abs(dq - prev_dq) / dt
                max_accel_violation = max(max_accel_violation, np.max(accel) - a_max)

            prev_dq = dq.copy()
            q = q + dq * dt

        # Allow small numerical tolerance
        assert (
            max_accel_violation < 0.1
        ), f"Acceleration violation {max_accel_violation:.4f} exceeds tolerance"

    def test_tracking_not_catastrophically_degraded(self, panda_setup):
        """Acceleration limits should not degrade EE tracking by more than 15%."""
        robot, solver = panda_setup
        task = solver.add_frame_task("ee", "panda_hand", eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 1.0
        solver.enable_position_limits(True)

        q_start = PANDA_HOME.copy()
        robot.update_configuration(q_start)
        ee_pose = robot.get_frame_pose("panda_hand")
        target_pos = ee_pose.translation.copy()
        target_pos[0] += 0.1
        T = ee_pose.homogeneous()
        target_rot = T[:3, :3]
        task.set_target_pose(target_pos, target_rot)

        n_steps = 100

        def run_and_measure_error():
            q = q_start.copy()
            for _ in range(n_steps):
                robot.update_configuration(q)
                result = solver.solve_velocity(q, apply_limits=True)
                if result.status != eik.SolverStatus.SUCCESS:
                    break
                q = q + np.array(result.joint_velocities) * solver.dt
            robot.update_configuration(q)
            final_pos = robot.get_frame_pose("panda_hand").translation
            return np.linalg.norm(final_pos - target_pos)

        # Without acceleration limits
        solver.enable_acceleration_limits(False)
        error_baseline = run_and_measure_error()

        # With acceleration limits
        solver.enable_acceleration_limits(True)
        solver.set_acceleration_limits(np.full(robot.nv, 10.0))
        error_with_accel = run_and_measure_error()

        # Should not be more than 15% worse
        assert error_with_accel < error_baseline * 1.15 + 1e-6, (
            f"Error with accel limits ({error_with_accel:.6f}) is more than "
            f"15% worse than baseline ({error_baseline:.6f})"
        )
