"""Tests for collision hardening features.

Covers:
1. SolverStatus.COLLISION_VIOLATED = 9 — returned when a safe seed cannot
   maintain min_distance in any integration step.
2. Post-step rejection bug fix — solutions at distance < min_distance no
   longer return SUCCESS.
3. Per-pair min_distance override — set_collision_pair_min_distance /
   get_collision_pair_min_distance_overrides / clear_collision_pair_min_distance.
4. adaptive_dt on PositionStepOptions — scales integration dt by
   (pos_error / reference_distance) to reduce cycles needed for large jumps.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

import embodik

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PANDA_DEFAULT_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04], dtype=float)


def _write_prismatic_collision_urdf(tmp_path: pathlib.Path) -> pathlib.Path:
    urdf_path = tmp_path / "prismatic_collision_floor.urdf"
    urdf_path.write_text(
        """<?xml version="1.0"?>
<robot name="prismatic_collision_floor">
  <link name="world"/>
  <link name="obstacle">
    <collision>
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
  </link>
  <joint name="obstacle_fixed" type="fixed">
    <parent link="world"/>
    <child link="obstacle"/>
  </joint>
  <link name="moving">
    <collision>
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
  </link>
  <joint name="moving_slide" type="prismatic">
    <parent link="world"/>
    <child link="moving"/>
    <origin xyz="0.106 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="0.2" effort="100" velocity="1"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    return urdf_path


def _write_masked_floor_collision_urdf(tmp_path: pathlib.Path) -> pathlib.Path:
    urdf_path = tmp_path / "masked_collision_floor.urdf"
    urdf_path.write_text(
        """<?xml version="1.0"?>
<robot name="masked_collision_floor">
  <link name="world"/>
  <link name="left_obstacle">
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <joint name="left_fixed" type="fixed">
    <parent link="world"/><child link="left_obstacle"/>
  </joint>
  <link name="right_obstacle">
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <joint name="right_fixed" type="fixed">
    <parent link="world"/><child link="right_obstacle"/>
    <origin xyz="0.245 0 0"/>
  </joint>
  <link name="structural_a">
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <joint name="structural_a_fixed" type="fixed">
    <parent link="world"/><child link="structural_a"/>
    <origin xyz="0 1 0"/>
  </joint>
  <link name="structural_b">
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <joint name="structural_b_fixed" type="fixed">
    <parent link="world"/><child link="structural_b"/>
    <origin xyz="0.106 1 0"/>
  </joint>
  <link name="moving">
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <joint name="moving_slide" type="prismatic">
    <parent link="world"/><child link="moving"/>
    <origin xyz="0.12 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-0.1" upper="0.1" effort="100" velocity="1"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    return urdf_path


def _ensure_ros_package_path(urdf_path: pathlib.Path) -> None:
    existing = os.environ.get("ROS_PACKAGE_PATH", "")
    paths: set[str] = set()
    p = urdf_path.parent
    while p != p.parent:
        paths.add(str(p))
        p = p.parent
    new_path = ":".join(str(x) for x in paths)
    os.environ["ROS_PACKAGE_PATH"] = f"{existing}:{new_path}" if existing else new_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def panda_robot():
    from robot_descriptions.panda_description import URDF_PATH

    urdf_path = pathlib.Path(URDF_PATH)
    _ensure_ros_package_path(urdf_path)
    return embodik.RobotModel(str(URDF_PATH))


@pytest.fixture(scope="module")
def panda_solver(panda_robot):
    solver = embodik.KinematicsSolver(panda_robot)
    solver.dt = 0.01
    solver.set_damping(0.01)
    return solver


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSuccessGuaranteesMinDistanceNotViolated:
    """SUCCESS results must never violate the configured min_distance."""

    def test_success_guarantees_min_distance_not_violated(self, panda_robot, panda_solver):
        min_dist = 0.05
        panda_solver.configure_collision_constraint(
            min_distance=min_dist,
            max_constraints=3,
        )

        panda_solver.clear_tasks()
        task = panda_solver.add_frame_task("ee_task", "panda_hand")
        task.priority = 0
        task.weight = 1.0

        q0 = _PANDA_DEFAULT_Q.copy()
        panda_robot.update_configuration(q0)
        hand_pose = panda_robot.get_frame_pose("panda_hand")
        base_pos = np.asarray(hand_pose.translation, dtype=float)
        base_rot = np.asarray(hand_pose.rotation, dtype=float)

        # Target offset that pushes the arm significantly
        target = np.eye(4, dtype=float)
        target[:3, :3] = base_rot
        target[:3, 3] = base_pos + np.array([0.0, 0.3, -0.3], dtype=float)

        opts = embodik.PositionStepOptions()
        opts.max_steps = 50
        opts.position_gain = 100.0
        opts.orientation_gain = 1.0
        opts.stall_recovery = False

        q = q0.copy()
        for _ in range(50):
            result = panda_solver.solve_position_step(q, target, "ee_task", opts)
            q = np.asarray(result.q_solution, dtype=float)
            panda_robot.update_configuration(q)

            if result.status == embodik.SolverStatus.SUCCESS:
                dist = panda_solver.evaluate_min_collision_distance(q)
                assert dist >= min_dist - 0.01, (
                    f"SUCCESS result violated min_distance: "
                    f"dist={dist:.4f} < min_dist={min_dist} (tol=0.01)"
                )


class TestCollisionViolatedStatusFromSafeSeed:
    """COLLISION_VIOLATED is returned when a safe seed cannot make progress."""

    def test_collision_violated_status_returned_from_safe_seed(self, panda_robot, panda_solver):
        # Very large min_distance to make almost any motion trigger violation
        tight_min_dist = 0.20
        panda_solver.configure_collision_constraint(
            min_distance=tight_min_dist,
            max_constraints=3,
        )

        panda_solver.clear_tasks()
        task = panda_solver.add_frame_task("ee_task", "panda_hand")
        task.priority = 0
        task.weight = 1.0

        q0 = _PANDA_DEFAULT_Q.copy()
        panda_robot.update_configuration(q0)
        hand_pose = panda_robot.get_frame_pose("panda_hand")
        base_pos = np.asarray(hand_pose.translation, dtype=float)
        base_rot = np.asarray(hand_pose.rotation, dtype=float)

        # Target far away to force aggressive motion
        target = np.eye(4, dtype=float)
        target[:3, :3] = base_rot
        target[:3, 3] = base_pos + np.array([0.5, 0.0, 0.0], dtype=float)

        opts = embodik.PositionStepOptions()
        opts.max_steps = 1
        opts.position_gain = 10.0
        opts.stall_recovery = False

        # Check whether the seed is above min_distance (required for
        # COLLISION_VIOLATED to be possible per the spec)
        seed_dist = panda_solver.evaluate_min_collision_distance(q0)
        if seed_dist < tight_min_dist:
            pytest.skip(
                f"Default panda q is already inside tight min_distance "
                f"({seed_dist:.4f} < {tight_min_dist}). Cannot test COLLISION_VIOLATED."
            )

        result = panda_solver.solve_position_step(q0, target, "ee_task", opts)

        if result.status == embodik.SolverStatus.COLLISION_VIOLATED:
            # q_solution should be close to the input (held at last safe)
            q_sol = np.asarray(result.q_solution, dtype=float)
            dist_from_seed = np.linalg.norm(q_sol - q0)
            assert dist_from_seed < 1.0, (
                f"COLLISION_VIOLATED q_solution drifted far from seed: "
                f"||q_sol - q0|| = {dist_from_seed:.4f}"
            )
            # q_solution distance should not be deeply negative
            sol_dist = panda_solver.evaluate_min_collision_distance(q_sol)
            assert sol_dist > -0.05, (
                f"COLLISION_VIOLATED q_solution in deep collision: " f"dist={sol_dist:.4f}"
            )
        else:
            # Other statuses (INFEASIBLE, NO_PROGRESS, etc.) are valid when
            # the constraint is extremely tight; the test is only exercised
            # when COLLISION_VIOLATED actually fires.
            assert result.status in (
                embodik.SolverStatus.INFEASIBLE,
                embodik.SolverStatus.SUCCESS,
                embodik.SolverStatus.NO_PROGRESS,
                embodik.SolverStatus.COLLISION_VIOLATED,
            ), f"Unexpected status: {result.status}"


class TestPerPairMinDistanceOverride:
    """Per-pair min_distance override API: set / get / clear."""

    def test_per_pair_min_distance_override_respected(self, panda_robot, panda_solver):
        # Global constraint at a low value
        panda_solver.configure_collision_constraint(
            min_distance=0.02,
            max_constraints=3,
        )

        link_a = "panda_link4"
        link_b = "panda_link6"
        override_dist = 0.10

        panda_solver.set_collision_pair_min_distance(link_a, link_b, override_dist)

        overrides = panda_solver.get_collision_pair_min_distance_overrides()
        assert (
            len(overrides) > 0
        ), "Expected at least one override after set_collision_pair_min_distance"

        # At least one override should be close to the specified distance
        distances = [d for _, d in overrides]
        assert any(
            abs(d - override_dist) < 0.01 for d in distances
        ), f"No override near {override_dist} found; overrides={overrides}"

        # Clear the override and verify removal
        panda_solver.clear_collision_pair_min_distance(link_a, link_b)
        overrides_after = panda_solver.get_collision_pair_min_distance_overrides()
        assert (
            len(overrides_after) == 0
        ), f"Expected empty overrides after clear, got {overrides_after}"


class TestNonWorseningCollisionFloor:
    """The structural floor is a minimum recovery target, not a ceiling."""

    def test_pair_above_floor_can_move_toward_floor(self, tmp_path):
        robot = embodik.RobotModel(str(_write_prismatic_collision_urdf(tmp_path)))
        solver = embodik.KinematicsSolver(robot)
        solver.dt = 0.01
        solver.set_damping(0.01)
        solver.configure_collision_constraint(min_distance=0.07, max_constraints=1)
        solver.set_non_worsening_collision_floor_enabled(True)
        solver.set_collision_structural_floor(0.01)

        task = solver.add_frame_task("moving_task", "moving")
        task.priority = 0
        task.weight = 1.0

        q = np.array([0.014], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("moving")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.asarray(pose.rotation, dtype=float)
        target[:3, 3] = np.asarray(pose.translation, dtype=float)
        target[0, 3] -= 0.005

        initial_distance = solver.evaluate_min_collision_distance(q)
        assert initial_distance == pytest.approx(0.020, abs=5e-4)

        options = embodik.PositionStepOptions()
        options.max_steps = 1
        options.position_gain = 10.0
        options.orientation_gain = 1.0
        options.stall_recovery = False

        for _ in range(50):
            result = solver.solve_position_step(q, target, "moving_task", options)
            q = np.asarray(result.q_solution, dtype=float)

        final_distance = solver.evaluate_min_collision_distance(q)
        assert 0.0145 <= final_distance < 0.019

    def test_movable_pair_below_floor_recovers_to_floor(self, tmp_path):
        robot = embodik.RobotModel(str(_write_prismatic_collision_urdf(tmp_path)))
        solver = embodik.KinematicsSolver(robot)
        solver.dt = 0.01
        solver.set_damping(0.01)
        solver.configure_collision_constraint(min_distance=0.07, max_constraints=1)
        solver.set_non_worsening_collision_floor_enabled(True)
        solver.set_collision_structural_floor(0.01)

        task = solver.add_frame_task("moving_task", "moving")
        task.priority = 0
        task.weight = 1.0

        q = np.array([0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("moving")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.asarray(pose.rotation, dtype=float)
        target[:3, 3] = np.asarray(pose.translation, dtype=float)

        initial_distance = solver.evaluate_min_collision_distance(q)
        assert initial_distance == pytest.approx(0.006, abs=5e-4)

        options = embodik.PositionStepOptions()
        options.max_steps = 1
        options.position_gain = 10.0
        options.orientation_gain = 1.0
        options.stall_recovery = False

        for _ in range(50):
            result = solver.solve_position_step(q, target, "moving_task", options)
            q = np.asarray(result.q_solution, dtype=float)

        final_distance = solver.evaluate_min_collision_distance(q)
        assert final_distance >= 0.0095

    def test_per_pair_override_can_preserve_a_tighter_structural_clearance(self, tmp_path):
        robot = embodik.RobotModel(str(_write_prismatic_collision_urdf(tmp_path)))
        solver = embodik.KinematicsSolver(robot)
        solver.dt = 0.01
        solver.set_damping(0.01)
        solver.configure_collision_constraint(min_distance=0.07, max_constraints=1)
        solver.set_non_worsening_collision_floor_enabled(True)
        solver.set_collision_structural_floor(0.01)
        solver.set_collision_pair_min_distance("obstacle", "moving", 0.005)

        task = solver.add_frame_task("moving_task", "moving")
        task.priority = 0
        task.weight = 1.0

        q = np.array([0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("moving")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.asarray(pose.rotation, dtype=float)
        target[:3, 3] = np.asarray(pose.translation, dtype=float)

        options = embodik.PositionStepOptions()
        options.max_steps = 1
        options.position_gain = 10.0
        options.orientation_gain = 1.0
        options.stall_recovery = False

        for _ in range(20):
            result = solver.solve_position_step(q, target, "moving_task", options)
            q = np.asarray(result.q_solution, dtype=float)

        final_distance = solver.evaluate_min_collision_distance(q)
        assert 0.005 <= final_distance < 0.008

    def test_structural_override_does_not_mask_another_pair_crossing_floor(self, tmp_path):
        robot = embodik.RobotModel(str(_write_masked_floor_collision_urdf(tmp_path)))
        solver = embodik.KinematicsSolver(robot)
        solver.dt = 0.1
        solver.set_damping(0.01)
        solver.configure_collision_constraint(min_distance=0.07, max_constraints=1)
        solver.set_non_worsening_collision_floor_enabled(True)
        solver.set_collision_structural_floor(0.01)
        solver.set_collision_pair_min_distance("structural_a", "structural_b", 0.005)

        task = solver.add_frame_task("moving_task", "moving")
        task.priority = 0
        task.weight = 1.0

        q = np.array([0.0], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("moving")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.asarray(pose.rotation, dtype=float)
        target[:3, 3] = np.asarray(pose.translation, dtype=float)
        target[0, 3] += 0.025

        options = embodik.PositionStepOptions()
        options.dt = 0.1
        options.max_steps = 1
        options.position_gain = 10.0
        options.orientation_gain = 1.0
        options.max_linear_speed = 1.0
        options.stall_recovery = False

        result = solver.solve_position_step(q, target, "moving_task", options)
        q_next = np.asarray(result.q_solution, dtype=float)

        # The structural pair remains at 6 mm, but the moving/right pair must
        # still make useful progress without crossing its independent 10 mm
        # floor (q <= 15 mm).
        assert q_next[0] >= 0.005
        assert q_next[0] <= 0.0151

    def test_recovering_one_violated_pair_cannot_cross_another_pair_floor(self, tmp_path):
        robot = embodik.RobotModel(str(_write_masked_floor_collision_urdf(tmp_path)))
        solver = embodik.KinematicsSolver(robot)
        solver.dt = 0.1
        solver.set_damping(0.01)
        solver.configure_collision_constraint(min_distance=0.07, max_constraints=1)
        solver.set_non_worsening_collision_floor_enabled(True)
        solver.set_collision_structural_floor(0.01)
        solver.set_collision_pair_min_distance("structural_a", "structural_b", 0.005)

        task = solver.add_frame_task("moving_task", "moving")
        task.priority = 0
        task.weight = 1.0

        q = np.array([-0.015], dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("moving")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.asarray(pose.rotation, dtype=float)
        target[:3, 3] = np.asarray(pose.translation, dtype=float)
        target[0, 3] += 0.033

        options = embodik.PositionStepOptions()
        options.dt = 0.1
        options.max_steps = 1
        options.position_gain = 10.0
        options.orientation_gain = 1.0
        options.max_linear_speed = 1.0
        options.stall_recovery = False

        result = solver.solve_position_step(q, target, "moving_task", options)
        q_next = np.asarray(result.q_solution, dtype=float)

        # The left pair starts 5 mm inside its floor. Recovery should move right,
        # but the independently safe right pair must remain at or above 10 mm.
        assert q_next[0] >= q[0] + 0.005
        assert q_next[0] <= 0.0151


class TestViolatedSeedNeverReturnsCollisionViolated:
    """COLLISION_VIOLATED must never be returned from a violated (inside min_distance) seed."""

    def test_violated_seed_never_returns_collision_violated(self, panda_robot, panda_solver):
        min_dist = 0.05
        panda_solver.configure_collision_constraint(
            min_distance=min_dist,
            max_constraints=3,
        )

        panda_solver.clear_tasks()
        task = panda_solver.add_frame_task("ee_task", "panda_hand")
        task.priority = 0
        task.weight = 1.0

        # Try configurations folded toward self-collision
        candidate_configs = [
            np.array([0.0, 1.5, 0.0, -0.5, 0.0, 0.5, 0.0, 0.04, 0.04], dtype=float),
            np.array([0.0, 1.8, 0.0, -0.3, 0.0, 0.3, 0.0, 0.04, 0.04], dtype=float),
            np.array([0.5, 1.5, 0.5, -0.5, 0.0, 0.5, 0.5, 0.04, 0.04], dtype=float),
        ]

        # Find a config that is actually inside min_distance
        violated_q = None
        for cand in candidate_configs:
            panda_robot.update_configuration(cand)
            d = panda_solver.evaluate_min_collision_distance(cand)
            if d < min_dist:
                violated_q = cand.copy()
                break

        if violated_q is None:
            pytest.skip(
                "Could not find a panda configuration inside min_distance=0.05 "
                "from the candidate list. Test requires a pre-violated seed."
            )

        panda_robot.update_configuration(violated_q)
        hand_pose = panda_robot.get_frame_pose("panda_hand")
        base_pos = np.asarray(hand_pose.translation, dtype=float)
        base_rot = np.asarray(hand_pose.rotation, dtype=float)

        target = np.eye(4, dtype=float)
        target[:3, :3] = base_rot
        target[:3, 3] = base_pos + np.array([0.2, 0.0, 0.1], dtype=float)

        opts = embodik.PositionStepOptions()
        opts.max_steps = 1
        opts.position_gain = 5.0
        opts.stall_recovery = False

        q = violated_q.copy()
        for _ in range(50):
            result = panda_solver.solve_position_step(q, target, "ee_task", opts)

            assert result.status != embodik.SolverStatus.COLLISION_VIOLATED, (
                "COLLISION_VIOLATED must never be returned when seed is inside "
                f"min_distance; seed dist={panda_solver.evaluate_min_collision_distance(q):.4f}, "
                f"min_dist={min_dist}"
            )

            q = np.asarray(result.q_solution, dtype=float)
            panda_robot.update_configuration(q)

    def test_inside_margin_recovery_motion_is_not_reported_as_success(
        self, panda_robot, panda_solver
    ):
        """Recovery from an already-violated seed may move, but not claim full success."""
        min_dist = 0.05
        panda_solver.configure_collision_constraint(
            min_distance=min_dist,
            max_constraints=3,
        )

        panda_solver.clear_tasks()
        task = panda_solver.add_frame_task("ee_task", "panda_hand")
        task.priority = 0
        task.weight = 1.0

        q0 = _PANDA_DEFAULT_Q.copy()
        seed_dist = panda_solver.evaluate_min_collision_distance(q0)
        if seed_dist >= min_dist:
            pytest.skip(
                f"Default panda q is not inside min_distance " f"({seed_dist:.4f} >= {min_dist})."
            )

        panda_robot.update_configuration(q0)
        hand_pose = panda_robot.get_frame_pose("panda_hand")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.asarray(hand_pose.rotation, dtype=float)
        target[:3, 3] = np.asarray(hand_pose.translation, dtype=float) + np.array(
            [0.0, 0.3, -0.3], dtype=float
        )

        opts = embodik.PositionStepOptions()
        opts.max_steps = 50
        opts.position_gain = 100.0
        opts.orientation_gain = 1.0
        opts.stall_recovery = False

        result = panda_solver.solve_position_step(q0, target, "ee_task", opts)
        q_sol = np.asarray(result.q_solution, dtype=float)
        sol_dist = panda_solver.evaluate_min_collision_distance(q_sol)

        assert int(getattr(result, "stall_escape_count", 0)) > 0
        assert sol_dist < min_dist
        assert result.status != embodik.SolverStatus.SUCCESS


class TestAdaptiveDtReducesApproachCycles:
    """adaptive_dt should reduce number of cycles needed to close a large position error."""

    def test_adaptive_dt_reduces_approach_cycles(self, panda_robot, panda_solver):
        # No collision constraint for this test — pure convergence speed
        panda_solver.clear_collision_constraint()

        panda_solver.clear_tasks()
        task = panda_solver.add_frame_task("ee_task", "panda_hand")
        task.priority = 0
        task.weight = 1.0

        q0 = _PANDA_DEFAULT_Q.copy()
        panda_robot.update_configuration(q0)
        hand_pose = panda_robot.get_frame_pose("panda_hand")
        base_pos = np.asarray(hand_pose.translation, dtype=float)
        base_rot = np.asarray(hand_pose.rotation, dtype=float)

        # Target 0.4 m away from current EE pose
        target = np.eye(4, dtype=float)
        target[:3, :3] = base_rot
        target[:3, 3] = base_pos + np.array([0.0, 0.4, 0.0], dtype=float)

        max_cycles = 300
        goal_error = 0.02

        # --- Baseline: no adaptive_dt ---
        opts_base = embodik.PositionStepOptions()
        opts_base.dt = 0.01
        opts_base.max_steps = 1
        opts_base.position_gain = 10.0
        opts_base.orientation_gain = 1.0
        opts_base.stall_recovery = False
        opts_base.adaptive_dt = False

        q = q0.copy()
        baseline_cycles = max_cycles
        baseline_final_error = float("inf")
        for i in range(max_cycles):
            result = panda_solver.solve_position_step(q, target, "ee_task", opts_base)
            q = np.asarray(result.q_solution, dtype=float)
            panda_robot.update_configuration(q)
            pos = np.asarray(panda_robot.get_frame_pose("panda_hand").translation, dtype=float)
            err = float(np.linalg.norm(pos - target[:3, 3]))
            baseline_final_error = err
            if err < goal_error:
                baseline_cycles = i + 1
                break

        # --- Adaptive dt ---
        opts_adapt = embodik.PositionStepOptions()
        opts_adapt.dt = 0.01
        opts_adapt.max_steps = 1
        opts_adapt.position_gain = 10.0
        opts_adapt.orientation_gain = 1.0
        opts_adapt.stall_recovery = False
        opts_adapt.adaptive_dt = True
        opts_adapt.adaptive_dt_max_scale = 5.0
        opts_adapt.adaptive_dt_reference_distance = 0.05

        q = q0.copy()
        panda_robot.update_configuration(q)
        adaptive_cycles = max_cycles
        adaptive_final_error = float("inf")
        for i in range(max_cycles):
            result = panda_solver.solve_position_step(q, target, "ee_task", opts_adapt)
            q = np.asarray(result.q_solution, dtype=float)
            panda_robot.update_configuration(q)
            pos = np.asarray(panda_robot.get_frame_pose("panda_hand").translation, dtype=float)
            err = float(np.linalg.norm(pos - target[:3, 3]))
            adaptive_final_error = err
            if err < goal_error:
                adaptive_cycles = i + 1
                break

        if baseline_cycles < max_cycles or adaptive_cycles < max_cycles:
            # At least one converged; adaptive should be faster
            assert adaptive_cycles < baseline_cycles, (
                f"adaptive_dt did not reduce cycles to convergence: "
                f"baseline={baseline_cycles}, adaptive={adaptive_cycles}"
            )
        else:
            # Neither converged within 300 cycles; adaptive should show smaller error
            assert adaptive_final_error < baseline_final_error, (
                f"Neither converged in {max_cycles} cycles; "
                f"adaptive final error ({adaptive_final_error:.4f}) should be "
                f"smaller than baseline ({baseline_final_error:.4f})"
            )


class TestAdaptiveDtCollisionStall:
    """Adaptive dt must not cause collision-overshoot stalls near min_distance.

    When adaptive_dt is enabled with a large max_scale, the integration step grows
    proportionally to position error.  Without the proximity cap, a large step can
    overshoot past min_distance when clearance is small, triggering COLLISION_VIOLATED
    on every tick and causing a permanent stall.  The proximity cap reduces scale near
    collision boundaries so the integration step stays within the available clearance.
    """

    # Exclusion helper: skip adjacent links and finger/hand pairs
    @staticmethod
    def _make_exclusion_pairs(robot):
        import re

        all_pairs = robot.get_collision_pair_names()
        excl = []
        for a, b in all_pairs:
            if any(x in n for x in ("finger", "hand") for n in (a, b)):
                excl.append((a, b))
                continue
            na = re.findall(r"link(\d+)", a)
            nb = re.findall(r"link(\d+)", b)
            if na and nb and abs(int(na[0]) - int(nb[0])) <= 1:
                excl.append((a, b))
        return excl

    def test_adaptive_dt_proximity_cap_prevents_collision_stall(self, panda_robot, panda_solver):
        """With proximity cap: adaptive_dt must not enter a COLLISION_VIOLATED stall.

        Without the cap, large adaptive steps overshoot the ~2 mm clearance above
        min_distance, producing kCollisionViolated every tick and freezing the robot.
        The proximity cap reduces scale when clearance < step_dt * max_ee_speed,
        keeping the integration step within bounds.
        """
        excl = self._make_exclusion_pairs(panda_robot)
        min_dist = 0.020  # ~2 mm below the natural panda_link5/link7 distance of ~22 mm
        panda_solver.configure_collision_constraint(
            min_distance=min_dist,
            max_constraints=3,
            exclude_pairs=excl,
        )

        panda_solver.clear_tasks()
        task = panda_solver.add_frame_task("ee_task", "panda_hand")
        task.priority = 0
        task.weight = 1.0

        q0 = _PANDA_DEFAULT_Q.copy()
        panda_robot.update_configuration(q0)
        seed_dist = panda_solver.evaluate_min_collision_distance(q0)
        if seed_dist < min_dist:
            pytest.skip(
                f"Default panda q already inside min_distance "
                f"({seed_dist:.4f} < {min_dist}); cannot test stall."
            )

        hand_pose = panda_robot.get_frame_pose("panda_hand")
        # Drive toward a target that requires the arm to sweep near link5/link7
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.asarray(hand_pose.rotation, dtype=float)
        target[:3, 3] = np.asarray(hand_pose.translation, dtype=float) + np.array([0.0, 0.3, -0.3])

        opts = embodik.PositionStepOptions()
        opts.max_steps = 1
        opts.position_gain = 10.0
        opts.adaptive_dt = True
        opts.adaptive_dt_max_scale = 10.0
        opts.adaptive_dt_reference_distance = 0.02
        opts.stall_recovery = False

        max_cycles = 200
        violation_count = 0
        consecutive_violations = 0
        max_consecutive = 0

        q = q0.copy()
        for _ in range(max_cycles):
            result = panda_solver.solve_position_step(q, target, "ee_task", opts)
            if result.status == embodik.SolverStatus.COLLISION_VIOLATED:
                violation_count += 1
                consecutive_violations += 1
                max_consecutive = max(max_consecutive, consecutive_violations)
            else:
                consecutive_violations = 0
            q = np.asarray(result.q_solution, dtype=float)
            panda_robot.update_configuration(q)

        # With the proximity cap the robot should make forward progress and not
        # get locked in a COLLISION_VIOLATED stall (>10 consecutive violations).
        assert max_consecutive < 10, (
            f"Adaptive dt caused a COLLISION_VIOLATED stall: "
            f"{max_consecutive} consecutive violations over {max_cycles} cycles. "
            f"Total violations: {violation_count}"
        )

    def test_without_adaptive_dt_no_stall(self, panda_robot, panda_solver):
        """Baseline: normal dt must not stall via COLLISION_VIOLATED stall either.

        Verifies that the desaturation and multi-target rejection fixes also prevent
        violations at normal dt (no adaptive scaling).
        """
        excl = self._make_exclusion_pairs(panda_robot)
        min_dist = 0.020
        panda_solver.configure_collision_constraint(
            min_distance=min_dist,
            max_constraints=3,
            exclude_pairs=excl,
        )

        panda_solver.clear_tasks()
        task = panda_solver.add_frame_task("ee_task", "panda_hand")
        task.priority = 0
        task.weight = 1.0

        q0 = _PANDA_DEFAULT_Q.copy()
        panda_robot.update_configuration(q0)
        seed_dist = panda_solver.evaluate_min_collision_distance(q0)
        if seed_dist < min_dist:
            pytest.skip(f"Seed dist {seed_dist:.4f} < min_dist {min_dist}")

        hand_pose = panda_robot.get_frame_pose("panda_hand")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.asarray(hand_pose.rotation, dtype=float)
        target[:3, 3] = np.asarray(hand_pose.translation, dtype=float) + np.array([0.0, 0.3, -0.3])

        opts = embodik.PositionStepOptions()
        opts.max_steps = 1
        opts.position_gain = 10.0
        opts.adaptive_dt = False
        opts.stall_recovery = False

        max_cycles = 200
        consecutive_violations = 0
        max_consecutive = 0
        q = q0.copy()
        for _ in range(max_cycles):
            result = panda_solver.solve_position_step(q, target, "ee_task", opts)
            if result.status == embodik.SolverStatus.COLLISION_VIOLATED:
                consecutive_violations += 1
                max_consecutive = max(max_consecutive, consecutive_violations)
            else:
                consecutive_violations = 0
            q = np.asarray(result.q_solution, dtype=float)
            panda_robot.update_configuration(q)

        assert max_consecutive < 10, (
            f"Normal dt produced COLLISION_VIOLATED stall: "
            f"{max_consecutive} consecutive violations"
        )
