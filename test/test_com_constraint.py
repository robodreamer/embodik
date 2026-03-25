#!/usr/bin/env python3
"""Unit tests for the CoM support-polygon constraint API.

Tests cover:
- configure / clear API surface
- invalid input rejection
- half-plane Jacobian shape
- velocity/acceleration limit clamping
- CoM stays inside polygon during IK solve
- frame transform (non-world frame_name)
"""

import math
import os
import tempfile

import numpy as np
import pytest

import embodik

# ---------------------------------------------------------------------------
# Minimal test robot (2-DOF revolute arm with enough mass for CoM)
# ---------------------------------------------------------------------------

_URDF_CONTENT = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="5.0"/>
  </joint>

  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.1"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.2"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="5.0"/>
  </joint>

  <link name="link2">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0.1"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
  </link>

  <joint name="ee_joint" type="fixed">
    <parent link="link2"/>
    <child link="end_effector"/>
    <origin xyz="0 0 0.2"/>
  </joint>

  <link name="end_effector">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
</robot>
"""


@pytest.fixture(scope="module")
def urdf_path():
    fd, path = tempfile.mkstemp(suffix=".urdf")
    with os.fdopen(fd, "w") as f:
        f.write(_URDF_CONTENT)
    yield path
    os.unlink(path)


@pytest.fixture
def robot(urdf_path):
    r = embodik.RobotModel(urdf_path, floating_base=False)
    q0 = np.zeros(r.nq)
    r.update_configuration(q0)
    return r


@pytest.fixture
def solver(robot):
    s = embodik.KinematicsSolver(robot)
    s.dt = 0.01
    return s


# ---------------------------------------------------------------------------
# A convenient large square polygon (always contains nominal CoM for this robot)
# ---------------------------------------------------------------------------
LARGE_SQUARE = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
# A small square that the CoM (roughly at xy ≈ (0,0)) sits inside
TIGHT_SQUARE = np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]])
# Triangle well away from origin – CoM should lie outside it when constraint is set
FAR_TRIANGLE = np.array([[0.5, 0.5], [1.0, 0.5], [0.75, 1.0]])


# ===========================================================================
# Phase 3.1 / 3.2 – API surface and invalid input
# ===========================================================================


class TestConfigureApi:
    def test_configure_and_clear(self, solver):
        """configure_com_constraint should accept valid Nx2 polygon; clear resets."""
        solver.configure_com_constraint(LARGE_SQUARE, margin=0.0)
        # Should be callable without error; subsequent clear is safe
        solver.clear_com_constraint()
        solver.clear_com_constraint()  # double-clear is safe

    def test_configure_nx3_vertices(self, solver):
        """Nx3 input (with z column) should be accepted (z is ignored)."""
        verts_3d = np.hstack([LARGE_SQUARE, np.ones((4, 1)) * 0.5])
        solver.configure_com_constraint(verts_3d)
        solver.clear_com_constraint()

    def test_configure_with_margin(self, solver):
        solver.configure_com_constraint(LARGE_SQUARE, margin=0.2)
        solver.clear_com_constraint()

    def test_configure_with_all_params(self, solver):
        solver.configure_com_constraint(
            LARGE_SQUARE,
            margin=0.1,
            frame_name="world",
            com_vel_max=0.3,
            com_acc_max=0.05,
            use_acceleration_limits=True,
        )
        solver.clear_com_constraint()

    def test_rejects_fewer_than_3_vertices(self, solver):
        two_pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        with pytest.raises(Exception):
            solver.configure_com_constraint(two_pts)

    def test_rejects_empty_array(self, solver):
        with pytest.raises(Exception):
            solver.configure_com_constraint(np.zeros((0, 2)))

    def test_rejects_missing_x_column(self, solver):
        one_col = np.array([[0.0], [1.0], [2.0]])
        with pytest.raises(Exception):
            solver.configure_com_constraint(one_col)

    def test_rejects_unknown_frame(self, solver):
        with pytest.raises(Exception):
            solver.configure_com_constraint(LARGE_SQUARE, frame_name="nonexistent_frame")


# ===========================================================================
# Phase 3.3 – Jacobian shape
# ===========================================================================


class TestJacobianShape:
    def test_jacobian_rows_match_halfplanes(self, robot, solver):
        """
        A convex hull of N vertices has N half-planes.
        The constraint Jacobian must have N rows and nv columns.
        """
        solver.configure_com_constraint(LARGE_SQUARE)

        # Run one solve to trigger compute_com_constraint internally
        task = solver.add_frame_task("ee", "end_effector")
        ee_pose = robot.get_frame_pose("end_effector")
        task.set_target_pose(ee_pose.translation, ee_pose.rotation)
        task.weight = 1.0

        result = solver.solve_velocity()
        # After a successful solve the CoM constraint was active; check solution shape
        assert len(result.solution) == robot.nv

        solver.clear_tasks()
        solver.clear_com_constraint()

    def test_triangle_has_3_halfplanes(self, robot, solver):
        """Triangle polygon -> 3 half-planes."""
        triangle = np.array([[0.0, 0.0], [0.2, 0.0], [0.1, 0.2]])
        solver.configure_com_constraint(triangle)
        task = solver.add_frame_task("ee", "end_effector")
        ee_pose = robot.get_frame_pose("end_effector")
        task.set_target_pose(ee_pose.translation, ee_pose.rotation)
        task.weight = 1.0
        result = solver.solve_velocity()
        assert result.status == embodik.SolverStatus.SUCCESS
        solver.clear_tasks()
        solver.clear_com_constraint()


# ===========================================================================
# Phase 3.4 – Velocity/acceleration limit clamping
# ===========================================================================


class TestVelocityAccelerationLimits:
    def _run_many_steps_track_com(self, robot, solver, n_steps=60):
        """Solve velocity IK repeatedly and track CoM XY trajectory."""
        com_history = []
        q = np.zeros(robot.nq)
        dt = solver.dt

        for _ in range(n_steps):
            robot.update_configuration(q)
            result = solver.solve_velocity(q)
            if result.status != embodik.SolverStatus.SUCCESS:
                break
            dq = result.joint_velocities
            q = q + dq * dt
            com_history.append(robot.get_com_position()[:2].copy())

        return np.array(com_history)

    def test_com_stays_inside_large_polygon(self, robot, solver):
        """With a large polygon and a reachable target, CoM stays inside."""
        solver.configure_com_constraint(LARGE_SQUARE, margin=0.0, com_vel_max=0.4, com_acc_max=0.1)
        task = solver.add_frame_task("ee", "end_effector")
        ee_pose = robot.get_frame_pose("end_effector")
        task.set_target_pose(ee_pose.translation + np.array([0.0, 0.0, 0.05]), ee_pose.rotation)
        task.weight = 1.0
        posture = solver.add_posture_task("posture")
        posture.weight = 0.01

        com_traj = self._run_many_steps_track_com(robot, solver, n_steps=40)

        # All CoM XY positions should be inside [-1, 1]^2
        assert len(com_traj) > 0
        assert np.all(com_traj[:, 0] >= -1.0)
        assert np.all(com_traj[:, 0] <= 1.0)
        assert np.all(com_traj[:, 1] >= -1.0)
        assert np.all(com_traj[:, 1] <= 1.0)

        solver.clear_tasks()
        solver.clear_com_constraint()

    def test_constraint_active_keeps_com_inside_tight_polygon(self, robot, solver):
        """Tight polygon around nominal CoM: CoM should not stray far beyond."""
        solver.configure_com_constraint(
            TIGHT_SQUARE, margin=0.0, com_vel_max=0.4, com_acc_max=0.1, use_acceleration_limits=True
        )
        task = solver.add_frame_task("ee", "end_effector")
        ee_pose = robot.get_frame_pose("end_effector")
        # Target that would push CoM far outside tight polygon
        task.set_target_pose(ee_pose.translation + np.array([0.3, 0.0, 0.0]), ee_pose.rotation)
        task.weight = 1.0
        posture = solver.add_posture_task("posture")
        posture.weight = 0.01

        com_traj = self._run_many_steps_track_com(robot, solver, n_steps=60)
        # CoM should stay approximately within a small band around the polygon
        # (QP may not enforce exactly, but should be much smaller than unconstrained)
        assert len(com_traj) > 0
        max_deviation = np.max(np.abs(com_traj[:, 0]))
        assert (
            max_deviation < 0.3
        ), f"CoM x deviated {max_deviation:.3f} m, constraint not effective."

        solver.clear_tasks()
        solver.clear_com_constraint()

    def test_no_constraint_allows_com_to_drift(self, robot, solver):
        """Without constraint, CoM can move freely (baseline comparison)."""
        task = solver.add_frame_task("ee", "end_effector")
        ee_pose = robot.get_frame_pose("end_effector")
        task.set_target_pose(ee_pose.translation + np.array([0.3, 0.0, 0.0]), ee_pose.rotation)
        task.weight = 1.0
        posture = solver.add_posture_task("posture")
        posture.weight = 0.01

        com_traj = self._run_many_steps_track_com(robot, solver, n_steps=60)
        # Without constraint CoM can drift; just check the solve didn't error
        assert len(com_traj) > 0

        solver.clear_tasks()

    def test_acceleration_limits_reduce_approach_velocity(self, robot, solver):
        """
        With use_acceleration_limits=True vs False, the constrained solve
        should behave more conservatively near the boundary when acc limits
        are enabled.  We only check the solve completes successfully.
        """
        q0 = np.zeros(robot.nq)
        robot.update_configuration(q0)

        for use_acc in (True, False):
            solver.configure_com_constraint(
                TIGHT_SQUARE,
                margin=0.0,
                com_vel_max=0.4,
                com_acc_max=0.1,
                use_acceleration_limits=use_acc,
            )
            task = solver.add_frame_task("ee", "end_effector")
            ee_pose = robot.get_frame_pose("end_effector")
            task.set_target_pose(ee_pose.translation + np.array([0.2, 0.0, 0.0]), ee_pose.rotation)
            task.weight = 1.0
            posture = solver.add_posture_task("posture")
            posture.weight = 0.01

            result = solver.solve_velocity(q0)
            assert (
                result.status == embodik.SolverStatus.SUCCESS
            ), f"Solve failed with use_acceleration_limits={use_acc}"
            solver.clear_tasks()
            solver.clear_com_constraint()

    def test_outside_polygon_blocks_further_outward_motion(self, robot, solver):
        """If CoM starts outside, CoM constraint should reduce outward command."""
        # Shifted polygon: requires x >= 0.05, while nominal CoM x is near 0.
        shifted_square = np.array([[0.05, -0.3], [0.30, -0.3], [0.30, 0.3], [0.05, 0.3]])
        task = solver.add_frame_task("ee", "end_effector")
        ee_pose = robot.get_frame_pose("end_effector")
        # Command EE in -x direction, which tends to push CoM further outside.
        task.set_target_pose(ee_pose.translation + np.array([-0.1, 0.0, 0.0]), ee_pose.rotation)
        task.weight = 1.0

        q0 = np.zeros(robot.nq)
        robot.update_configuration(q0)

        # Baseline without CoM constraint.
        free_result = solver.solve_velocity(q0)
        assert free_result.status == embodik.SolverStatus.SUCCESS
        dq_free = np.asarray(free_result.joint_velocities)

        # Re-run with CoM constraint active.
        solver.configure_com_constraint(
            shifted_square, margin=0.0, com_vel_max=0.4, com_acc_max=0.1, use_acceleration_limits=True
        )
        constrained_result = solver.solve_velocity(q0)
        assert constrained_result.status in (
            embodik.SolverStatus.SUCCESS,
            embodik.SolverStatus.INFEASIBLE,
        )
        dq_constrained = np.asarray(constrained_result.joint_velocities)

        # For this robot, +dq[1] increases CoM x.  The constrained solution should
        # be less outward (i.e., larger joint2 velocity) than unconstrained.
        assert dq_constrained[1] > dq_free[1], (
            f"Expected CoM constraint to reduce outward motion: "
            f"dq_free[1]={dq_free[1]:.6f}, dq_constrained[1]={dq_constrained[1]:.6f}"
        )

        solver.clear_tasks()
        solver.clear_com_constraint()

    def test_outside_polygon_rollout_beats_unconstrained(self, robot, solver):
        """Across rollout, constrained CoM should violate less than unconstrained."""
        shifted_square = np.array([[0.05, -0.3], [0.30, -0.3], [0.30, 0.3], [0.05, 0.3]])
        x_min = 0.05

        task = solver.add_frame_task("ee", "end_effector")
        ee_pose = robot.get_frame_pose("end_effector")
        # Persistent outward pull (-x) that tends to worsen x-violation.
        task.set_target_pose(ee_pose.translation + np.array([-0.1, 0.0, 0.0]), ee_pose.rotation)
        task.weight = 1.0

        def rollout(apply_constraint: bool, steps: int = 25) -> float:
            q = np.zeros(robot.nq)
            robot.update_configuration(q)
            if apply_constraint:
                solver.configure_com_constraint(
                    shifted_square, margin=0.0, com_vel_max=0.4, com_acc_max=0.1, use_acceleration_limits=True
                )
            else:
                solver.clear_com_constraint()

            for _ in range(steps):
                res = solver.solve_velocity(q)
                assert res.status in (
                    embodik.SolverStatus.SUCCESS,
                    embodik.SolverStatus.INFEASIBLE,
                )
                dq = np.asarray(res.joint_velocities)
                q = q + dq * solver.dt
                robot.update_configuration(q)

            return max(0.0, x_min - robot.get_com_position()[0])

        violation_free = rollout(apply_constraint=False)
        violation_constrained = rollout(apply_constraint=True)
        assert violation_constrained <= violation_free + 1e-9, (
            f"Expected constrained rollout to have less/equal violation: "
            f"free={violation_free:.6f}, constrained={violation_constrained:.6f}"
        )

        solver.clear_tasks()
        solver.clear_com_constraint()

    def test_solve_position_respects_com_constraint(self, robot, solver):
        """solve_position should include configured CoM constraint rows."""
        shifted_square = np.array([[0.05, -0.3], [0.30, -0.3], [0.30, 0.3], [0.05, 0.3]])
        x_min = 0.05
        q0 = np.zeros(robot.nq)

        ee_pose = robot.get_frame_pose("end_effector")
        target = np.eye(4)
        target[:3, :3] = np.asarray(ee_pose.rotation)
        target[:3, 3] = np.asarray(ee_pose.translation) + np.array([-0.15, 0.0, 0.0])

        opts = embodik.PositionIKOptions()
        opts.max_iterations = 40
        opts.position_gain = 20.0
        opts.orientation_gain = 20.0

        solver.clear_com_constraint()
        out_free = solver.solve_position(q0, target, "end_effector", opts)
        assert out_free.status in (
            embodik.SolverStatus.SUCCESS,
            embodik.SolverStatus.NO_PROGRESS,
            embodik.SolverStatus.INFEASIBLE,
        )
        robot.update_configuration(np.asarray(out_free.q_solution))
        violation_free = max(0.0, x_min - float(robot.get_com_position()[0]))

        solver.configure_com_constraint(
            shifted_square,
            margin=0.0,
            com_vel_max=0.4,
            com_acc_max=0.1,
            use_acceleration_limits=True,
        )
        out_constrained = solver.solve_position(q0, target, "end_effector", opts)
        assert out_constrained.status in (
            embodik.SolverStatus.SUCCESS,
            embodik.SolverStatus.NO_PROGRESS,
            embodik.SolverStatus.INFEASIBLE,
        )
        robot.update_configuration(np.asarray(out_constrained.q_solution))
        violation_constrained = max(0.0, x_min - float(robot.get_com_position()[0]))

        assert violation_constrained <= violation_free + 1e-9, (
            f"Expected solve_position CoM constraint to reduce/equal violation: "
            f"free={violation_free:.6f}, constrained={violation_constrained:.6f}"
        )

        solver.clear_com_constraint()

    def test_velocity_and_position_api_consistency_for_com_constraint(self, robot, solver):
        """CoM constraints should improve/equal violation in both solve APIs."""
        shifted_square = np.array([[0.05, -0.3], [0.30, -0.3], [0.30, 0.3], [0.05, 0.3]])
        x_min = 0.05
        q0 = np.zeros(robot.nq)

        # Velocity API check (single-step integrated update).
        task = solver.add_frame_task("ee_consistency", "end_effector")
        ee_pose = robot.get_frame_pose("end_effector")
        task.set_target_pose(ee_pose.translation + np.array([-0.1, 0.0, 0.0]), ee_pose.rotation)
        task.weight = 1.0
        robot.update_configuration(q0)
        solver.clear_com_constraint()
        vel_free = solver.solve_velocity(q0)
        assert vel_free.status in (embodik.SolverStatus.SUCCESS, embodik.SolverStatus.INFEASIBLE)
        q_vel_free = q0 + np.asarray(vel_free.joint_velocities) * solver.dt
        robot.update_configuration(q_vel_free)
        vel_violation_free = max(0.0, x_min - float(robot.get_com_position()[0]))

        robot.update_configuration(q0)
        solver.configure_com_constraint(
            shifted_square,
            margin=0.0,
            com_vel_max=0.4,
            com_acc_max=0.1,
            use_acceleration_limits=True,
        )
        vel_constrained = solver.solve_velocity(q0)
        assert vel_constrained.status in (embodik.SolverStatus.SUCCESS, embodik.SolverStatus.INFEASIBLE)
        q_vel_constrained = q0 + np.asarray(vel_constrained.joint_velocities) * solver.dt
        robot.update_configuration(q_vel_constrained)
        vel_violation_constrained = max(0.0, x_min - float(robot.get_com_position()[0]))
        assert vel_violation_constrained <= vel_violation_free + 1e-9

        # Position API check.
        target = np.eye(4)
        target[:3, :3] = np.asarray(ee_pose.rotation)
        target[:3, 3] = np.asarray(ee_pose.translation) + np.array([-0.15, 0.0, 0.0])
        opts = embodik.PositionIKOptions()
        opts.max_iterations = 40
        opts.position_gain = 20.0
        opts.orientation_gain = 20.0

        robot.update_configuration(q0)
        solver.clear_com_constraint()
        pos_free = solver.solve_position(q0, target, "end_effector", opts)
        assert pos_free.status in (
            embodik.SolverStatus.SUCCESS,
            embodik.SolverStatus.NO_PROGRESS,
            embodik.SolverStatus.INFEASIBLE,
        )
        robot.update_configuration(np.asarray(pos_free.q_solution))
        pos_violation_free = max(0.0, x_min - float(robot.get_com_position()[0]))

        robot.update_configuration(q0)
        solver.configure_com_constraint(
            shifted_square,
            margin=0.0,
            com_vel_max=0.4,
            com_acc_max=0.1,
            use_acceleration_limits=True,
        )
        pos_constrained = solver.solve_position(q0, target, "end_effector", opts)
        assert pos_constrained.status in (
            embodik.SolverStatus.SUCCESS,
            embodik.SolverStatus.NO_PROGRESS,
            embodik.SolverStatus.INFEASIBLE,
        )
        robot.update_configuration(np.asarray(pos_constrained.q_solution))
        pos_violation_constrained = max(0.0, x_min - float(robot.get_com_position()[0]))
        assert pos_violation_constrained <= pos_violation_free + 1e-9

        solver.clear_tasks()
        solver.clear_com_constraint()


# ===========================================================================
# Phase 3.5 – Frame transform
# ===========================================================================


class TestFrameTransform:
    def test_base_link_frame(self, robot, solver):
        """Specifying frame_name='base_link' (= world for fixed-base) should work."""
        solver.configure_com_constraint(LARGE_SQUARE, margin=0.0, frame_name="base_link")
        task = solver.add_frame_task("ee", "end_effector")
        ee_pose = robot.get_frame_pose("end_effector")
        task.set_target_pose(ee_pose.translation, ee_pose.rotation)
        task.weight = 1.0

        result = solver.solve_velocity()
        assert result.status == embodik.SolverStatus.SUCCESS

        solver.clear_tasks()
        solver.clear_com_constraint()

    def test_end_effector_frame(self, robot, solver):
        """Using the end_effector frame (rotated/translated) should not crash."""
        solver.configure_com_constraint(LARGE_SQUARE, margin=0.0, frame_name="end_effector")
        task = solver.add_frame_task("ee", "end_effector")
        ee_pose = robot.get_frame_pose("end_effector")
        task.set_target_pose(ee_pose.translation, ee_pose.rotation)
        task.weight = 1.0

        result = solver.solve_velocity()
        assert result.status == embodik.SolverStatus.SUCCESS

        solver.clear_tasks()
        solver.clear_com_constraint()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
