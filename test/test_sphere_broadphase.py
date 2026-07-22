"""Tests for native sphere broadphase correctness and performance.

Validates that the SphereBroadphase integrated into the solver:
1. Never changes solver decisions (conservativeness)
2. Never introduces penetrations
3. Reduces exact HPP-FCL distance queries
4. Does not regress collision computation time
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import embodik as eik
from examples.utils.robot_models import ensure_ros_package_path

try:
    from robot_descriptions.panda_description import URDF_PATH as PANDA_URDF_PATH
except ImportError:
    pytest.skip("robot_descriptions not installed", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def panda_robot():
    ensure_ros_package_path(Path(PANDA_URDF_PATH))
    return eik.RobotModel(str(PANDA_URDF_PATH), floating_base=False)


@pytest.fixture(scope="module")
def random_configs():
    rng = np.random.RandomState(42)
    lb = np.array([-2.9, -1.8, -2.9, -3.1, -2.9, -0.08, -2.9, 0.0, 0.0])
    ub = np.array([2.9, 1.8, 2.9, 0.08, 2.9, 3.8, 2.9, 0.04, 0.04])
    return [rng.uniform(lb, ub) for _ in range(100)]


def _make_solver(robot, enable_broadphase):
    """Create a deterministic solver with collision constraint."""
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.configure_collision_constraint(min_distance=0.03, max_constraints=3)
    solver.enable_collision_pair_cache(False)
    solver.enable_sphere_broadphase(enable_broadphase)
    return solver


def _make_solver_with_task(robot, enable_broadphase):
    """Create solver with an EE frame task for trajectory tests."""
    solver = _make_solver(robot, enable_broadphase)
    task = solver.add_frame_task("ee", "panda_hand")
    task.priority = 0
    task.weight = 10.0
    return solver, task


def _write_offset_collision_urdf(tmp_path: Path) -> Path:
    urdf_path = tmp_path / "offset_collision.urdf"
    urdf_path.write_text(
        """<?xml version="1.0"?>
<robot name="offset_collision">
  <link name="world"/>
  <link name="obstacle">
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <joint name="obstacle_fixed" type="fixed">
    <parent link="world"/><child link="obstacle"/>
  </joint>
  <link name="moving">
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <joint name="moving_slide" type="prismatic">
    <parent link="world"/><child link="moving"/>
    <origin xyz="0.125 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-0.05" upper="0.05" effort="100" velocity="1"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    return urdf_path


def _write_rotated_box_collision_urdf(tmp_path: Path) -> Path:
    urdf_path = tmp_path / "rotated_box_collision.urdf"
    urdf_path.write_text(
        """<?xml version="1.0"?>
<robot name="rotated_box_collision">
  <link name="world"/>
  <link name="obstacle">
    <collision>
      <origin rpy="0 0 0.7853981633974483"/>
      <geometry><box size="1.0 0.05 0.05"/></geometry>
    </collision>
  </link>
  <joint name="obstacle_fixed" type="fixed">
    <parent link="world"/><child link="obstacle"/>
  </joint>
  <link name="moving">
    <collision>
      <origin rpy="0 0 0.7853981633974483"/>
      <geometry><box size="1.0 0.05 0.05"/></geometry>
    </collision>
  </link>
  <joint name="moving_slide" type="prismatic">
    <parent link="world"/><child link="moving"/>
    <origin xyz="-0.1060660171779821 0.1060660171779821 0"/>
    <axis xyz="-0.7071067811865475 0.7071067811865475 0"/>
    <limit lower="-0.02" upper="0.02" effort="100" velocity="1"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    return urdf_path


def _write_motion_certificate_urdf(tmp_path: Path) -> Path:
    urdf_path = tmp_path / "motion_certificate.urdf"
    urdf_path.write_text(
        """<?xml version="1.0"?>
<robot name="motion_certificate">
  <link name="world"/>
  <link name="obstacle">
    <collision><geometry><cylinder radius="0.05" length="0.5"/></geometry></collision>
  </link>
  <joint name="obstacle_fixed" type="fixed">
    <parent link="world"/><child link="obstacle"/>
  </joint>
  <link name="moving">
    <collision><geometry><cylinder radius="0.05" length="0.5"/></geometry></collision>
  </link>
  <joint name="moving_slide" type="prismatic">
    <parent link="world"/><child link="moving"/>
    <origin xyz="0.09 0.09 0"/><axis xyz="1 0 0"/>
    <limit lower="-0.02" upper="0.02" effort="100" velocity="1"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    return urdf_path


def _run_circular_trajectory(robot, solver, task, n_steps, q0=None):
    """Run a circular EE trajectory and return per-step results and configs."""
    if q0 is None:
        q0 = np.zeros(robot.nq)
    q = np.array(q0, dtype=float)

    # Get initial EE pose for trajectory centre
    robot.update_configuration(q)
    pose = robot.get_frame_pose("panda_hand")
    base_pos = np.array(pose.translation)
    base_rot = np.array(pose.rotation)

    results = []
    configs = [q.copy()]
    for step in range(n_steps):
        phase = 0.09 * step
        target = base_pos + np.array(
            [
                0.04 * math.cos(phase),
                0.04 * math.sin(phase),
                0.02 * math.sin(0.5 * phase),
            ]
        )
        task.set_target_pose(target, base_rot)
        res = solver.solve_velocity(q, apply_limits=True)
        results.append(res)
        q = np.array(robot.integrate(q, np.array(res.joint_velocities) * solver.dt))
        configs.append(q.copy())
    return results, configs


# ---------------------------------------------------------------------------
# Task 4: Correctness tests
# ---------------------------------------------------------------------------


class TestSphereBroadphaseCorrectness:
    """Broadphase must not change solver status."""

    def test_broadphase_does_not_change_status(self, panda_robot, random_configs):
        """Solver status must match with and without broadphase on random configs."""
        solver_off = _make_solver(panda_robot, enable_broadphase=False)
        task_off = solver_off.add_frame_task("ee", "panda_hand")
        task_off.priority = 0
        task_off.weight = 10.0

        solver_on = _make_solver(panda_robot, enable_broadphase=True)
        task_on = solver_on.add_frame_task("ee", "panda_hand")
        task_on.priority = 0
        task_on.weight = 10.0

        mismatches = 0
        for q in random_configs[:50]:
            # Set same target so both solvers see the same problem
            panda_robot.update_configuration(q)
            pose = panda_robot.get_frame_pose("panda_hand")
            target_pos = np.array(pose.translation) + np.array([0.01, 0.0, 0.0])
            target_rot = np.array(pose.rotation)
            task_off.set_target_pose(target_pos, target_rot)
            task_on.set_target_pose(target_pos, target_rot)

            res_off = solver_off.solve_velocity(q, apply_limits=True)
            res_on = solver_on.solve_velocity(q, apply_limits=True)

            if res_off.status != res_on.status:
                mismatches += 1

        assert mismatches == 0, f"Status mismatch in {mismatches}/50 configs"


class TestSphereBroadphaseConservativeness:
    """Broadphase must be conservative: never cause different solver decisions."""

    def test_joint_velocities_identical(self, panda_robot, random_configs):
        """Joint velocities must be identical with and without broadphase."""
        solver_off = _make_solver(panda_robot, enable_broadphase=False)
        task_off = solver_off.add_frame_task("ee", "panda_hand")
        task_off.priority = 0
        task_off.weight = 10.0

        solver_on = _make_solver(panda_robot, enable_broadphase=True)
        task_on = solver_on.add_frame_task("ee", "panda_hand")
        task_on.priority = 0
        task_on.weight = 10.0

        max_diff = 0.0
        for q in random_configs:
            panda_robot.update_configuration(q)
            pose = panda_robot.get_frame_pose("panda_hand")
            target_pos = np.array(pose.translation) + np.array([0.01, 0.0, 0.0])
            target_rot = np.array(pose.rotation)
            task_off.set_target_pose(target_pos, target_rot)
            task_on.set_target_pose(target_pos, target_rot)

            res_off = solver_off.solve_velocity(q, apply_limits=True)
            res_on = solver_on.solve_velocity(q, apply_limits=True)

            diff = np.linalg.norm(
                np.array(res_off.joint_velocities) - np.array(res_on.joint_velocities)
            )
            max_diff = max(max_diff, diff)

        assert max_diff < 1e-6, f"Max velocity norm diff = {max_diff:.2e}, expected < 1e-6"

    def test_offset_geometry_uses_parent_joint_transform(self, tmp_path):
        """A fixed-joint offset must not be applied twice to sphere centres."""
        robot = eik.RobotModel(str(_write_offset_collision_urdf(tmp_path)))
        q = np.zeros(robot.nq, dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("moving")
        target_position = np.asarray(pose.translation, dtype=float) - np.array(
            [0.02, 0.0, 0.0], dtype=float
        )
        target_rotation = np.asarray(pose.rotation, dtype=float)

        solver_off = _make_solver(robot, enable_broadphase=False)
        task_off = solver_off.add_frame_task("moving", "moving")
        task_off.set_target_pose(target_position, target_rotation)
        result_off = solver_off.solve_velocity(q, apply_limits=True)

        solver_on = _make_solver(robot, enable_broadphase=True)
        task_on = solver_on.add_frame_task("moving", "moving")
        task_on.set_target_pose(target_position, target_rotation)
        result_on = solver_on.solve_velocity(q, apply_limits=True)

        assert result_on.collision_sphere_culled_pairs == 0
        assert result_on.collision_exact_distance_queries >= 1
        np.testing.assert_allclose(
            result_on.joint_velocities,
            result_off.joint_velocities,
            atol=1e-9,
            rtol=0.0,
        )

    def test_rotated_box_separation_culls_exact_query(self, tmp_path):
        """Oriented bounds should certify long parallel boxes as separated."""
        robot = eik.RobotModel(str(_write_rotated_box_collision_urdf(tmp_path)))
        q = np.zeros(robot.nq, dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("moving")
        target_position = np.asarray(pose.translation, dtype=float) + np.array(
            [0.005, -0.005, 0.0], dtype=float
        )
        target_rotation = np.asarray(pose.rotation, dtype=float)

        solver_off = _make_solver(robot, enable_broadphase=False)
        task_off = solver_off.add_frame_task("moving", "moving")
        task_off.set_target_pose(target_position, target_rotation)
        result_off = solver_off.solve_velocity(q, apply_limits=True)

        solver_on = _make_solver(robot, enable_broadphase=True)
        task_on = solver_on.add_frame_task("moving", "moving")
        task_on.set_target_pose(target_position, target_rotation)
        result_on = solver_on.solve_velocity(q, apply_limits=True)

        assert result_off.collision_exact_distance_queries >= 1
        assert result_on.collision_exact_distance_queries == 0
        assert result_on.collision_sphere_culled_pairs >= 1
        np.testing.assert_allclose(
            result_on.joint_velocities,
            result_off.joint_velocities,
            atol=1e-9,
            rtol=0.0,
        )

    def test_cached_scan_refines_nearest_points_after_distance_ranking(self, panda_robot):
        """Large cached scans should refine only the selected collision rows."""
        solver = eik.KinematicsSolver(panda_robot)
        solver.dt = 0.01
        solver.configure_collision_constraint(
            min_distance=0.03,
            nearest_points_all_pairs=False,
            max_constraints=3,
        )
        solver.set_collision_tuning_mode(eik.CollisionTuningMode.BALANCED)
        task = solver.add_frame_task("ee", "panda_hand")
        task.priority = 0
        task.weight = 10.0

        q = np.zeros(panda_robot.nq, dtype=float)
        panda_robot.update_configuration(q)
        pose = panda_robot.get_frame_pose("panda_hand")
        task.set_target_pose(
            np.asarray(pose.translation, dtype=float) + np.array([0.01, 0.0, 0.0]),
            np.asarray(pose.rotation, dtype=float),
        )
        solver.solve_velocity(q, apply_limits=True)

        q[0] += 0.02
        panda_robot.update_configuration(q)
        pose = panda_robot.get_frame_pose("panda_hand")
        task.set_target_pose(
            np.asarray(pose.translation, dtype=float) + np.array([0.01, 0.0, 0.0]),
            np.asarray(pose.rotation, dtype=float),
        )
        result = solver.solve_velocity(q, apply_limits=True)

        assert result.collision_pairs_considered > 3
        assert result.collision_exact_distance_queries > result.collision_pairs_considered
        assert result.collision_exact_distance_queries <= result.collision_pairs_considered + 4

    def test_nearby_post_step_uses_rigid_motion_certificate(self, tmp_path):
        """A nearby candidate should reuse an exact distance with a safe bound."""
        robot = eik.RobotModel(str(_write_motion_certificate_urdf(tmp_path)))
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01
        solver.configure_collision_constraint(
            min_distance=0.005,
            nearest_points_all_pairs=False,
            max_constraints=1,
        )
        solver.enable_collision_pair_cache(False, 1, 0.0, 8)
        solver.enable_sphere_broadphase(True)
        solver.set_non_worsening_collision_floor_enabled(True)
        solver.add_frame_task("moving", "moving")

        q = np.zeros(robot.nq, dtype=float)
        robot.update_configuration(q)
        target = robot.get_frame_pose("moving").homogeneous()
        target[0, 3] += 0.001
        options = eik.PositionStepOptions()
        options.dt = solver.dt
        options.max_steps = 1
        options.position_gain = 10.0
        options.orientation_gain = 10.0

        result = solver.solve_position_step(q, [eik.TaskTarget("moving", target)], options)

        assert result.status == eik.SolverStatus.SUCCESS
        assert solver.evaluate_min_collision_distance(q) > 0.02
        assert solver.get_last_post_step_collision_exact_distance_queries() == 1
        assert solver.get_last_post_step_collision_motion_bound_culled_pairs() == 1

    def test_public_post_step_distance_does_not_mutate_solver_state(self, tmp_path):
        """A diagnostic distance query must not seed future recovery state."""
        urdf_path = _write_motion_certificate_urdf(tmp_path)

        def make_solver():
            robot = eik.RobotModel(str(urdf_path))
            solver = eik.KinematicsSolver(robot)
            solver.dt = 0.01
            solver.configure_collision_constraint(
                min_distance=0.005,
                nearest_points_all_pairs=False,
                max_constraints=1,
            )
            solver.enable_collision_pair_cache(False, 1, 0.0, 8)
            solver.enable_sphere_broadphase(True)
            solver.set_non_worsening_collision_floor_enabled(True)
            solver.add_frame_task("moving", "moving")
            return robot, solver

        q = np.zeros(1, dtype=float)
        robot_a, solver_a = make_solver()
        robot_b, solver_b = make_solver()
        robot_a.update_configuration(q)
        target = robot_a.get_frame_pose("moving").homogeneous()
        target[0, 3] += 0.001
        options = eik.PositionStepOptions()
        options.dt = 0.01
        options.max_steps = 1

        assert solver_a.get_last_post_step_collision_exact_distance_queries() == 0
        assert solver_a.evaluate_post_step_collision_distance(q) is not None
        assert solver_a.get_last_post_step_collision_exact_distance_queries() == 0
        assert solver_a.get_last_post_step_collision_motion_bound_culled_pairs() == 0

        result_a = solver_a.solve_position_step(q, [eik.TaskTarget("moving", target)], options)
        result_b = solver_b.solve_position_step(q, [eik.TaskTarget("moving", target)], options)
        assert result_a.status == result_b.status
        np.testing.assert_allclose(result_a.q_solution, result_b.q_solution, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(
            result_a.joint_velocities,
            result_b.joint_velocities,
            atol=1e-12,
            rtol=0.0,
        )

    @pytest.mark.parametrize(
        "query_name",
        ["evaluate_collision_debug", "evaluate_min_collision_distance"],
    )
    def test_public_collision_diagnostic_preserves_robot_state(self, tmp_path, query_name):
        """Explicit collision probes must not replace the solver warm state."""
        robot = eik.RobotModel(str(_write_motion_certificate_urdf(tmp_path)))
        solver = eik.KinematicsSolver(robot)

        warm_q = np.array([-0.01], dtype=float)
        probe_q = np.array([0.02], dtype=float)
        robot.update_configuration(warm_q)

        result = getattr(solver, query_name)(probe_q)

        assert result is not None
        np.testing.assert_allclose(
            robot.get_current_configuration(),
            warm_q,
            atol=0.0,
            rtol=0.0,
        )

    @pytest.mark.parametrize(
        "query_name",
        ["evaluate_collision_debug", "evaluate_min_collision_distance"],
    )
    def test_public_collision_diagnostic_rejects_nonfinite_probe(self, tmp_path, query_name):
        """Invalid collision probes fail explicitly without corrupting warm state."""
        robot = eik.RobotModel(str(_write_motion_certificate_urdf(tmp_path)))
        solver = eik.KinematicsSolver(robot)

        warm_q = np.array([-0.01], dtype=float)
        robot.update_configuration(warm_q)

        with pytest.raises(RuntimeError, match="finite"):
            getattr(solver, query_name)(np.array([np.nan], dtype=float))

        np.testing.assert_allclose(
            robot.get_current_configuration(),
            warm_q,
            atol=0.0,
            rtol=0.0,
        )

    def test_rejected_preferred_lock_restores_post_step_certificates(self, tmp_path):
        """A rejected speculative solve must not affect the fallback cache."""
        urdf_path = _write_motion_certificate_urdf(tmp_path)

        def make_solver():
            robot = eik.RobotModel(str(urdf_path))
            solver = eik.KinematicsSolver(robot)
            solver.dt = 0.01
            solver.configure_collision_constraint(
                min_distance=0.005,
                nearest_points_all_pairs=False,
                max_constraints=1,
            )
            solver.enable_collision_pair_cache(False, 1, 0.0, 8)
            solver.enable_sphere_broadphase(True)
            solver.set_non_worsening_collision_floor_enabled(True)
            solver.add_frame_task("moving", "moving")
            return robot, solver

        q = np.zeros(1, dtype=float)
        robot_a, baseline_solver = make_solver()
        robot_b, preferred_solver = make_solver()
        robot_a.update_configuration(q)
        target = robot_a.get_frame_pose("moving").homogeneous()
        target[0, 3] += 0.001

        baseline_options = eik.PositionStepOptions()
        baseline_options.dt = 0.01
        baseline_options.max_steps = 1
        preferred_options = eik.PositionStepOptions()
        preferred_options.dt = 0.01
        preferred_options.max_steps = 1
        preferred_options.preferred_locked_joint_indices = [0]
        preferred_options.preferred_lock_tracking_tolerance = 1e-9
        preferred_options.preferred_lock_orientation_tolerance = 0.0

        baseline = baseline_solver.solve_position_step(
            q, [eik.TaskTarget("moving", target)], baseline_options
        )
        preferred = preferred_solver.solve_position_step(
            q, [eik.TaskTarget("moving", target)], preferred_options
        )

        assert preferred.preferred_lock_fallback_used is True
        assert preferred.status == baseline.status
        np.testing.assert_allclose(preferred.q_solution, baseline.q_solution, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(
            preferred.joint_velocities,
            baseline.joint_velocities,
            atol=1e-12,
            rtol=0.0,
        )
        assert (
            preferred_solver.get_last_post_step_collision_exact_distance_queries()
            == baseline_solver.get_last_post_step_collision_exact_distance_queries()
        )
        assert (
            preferred_solver.get_last_post_step_collision_motion_bound_culled_pairs()
            == baseline_solver.get_last_post_step_collision_motion_bound_culled_pairs()
        )

    def test_no_penetration_after_solve(self, panda_robot):
        """Broadphase must not introduce penetration beyond what the baseline produces.

        The panda URDF has adjacent-link geometry (hand/finger) that can
        overlap even without broadphase, so we compare broadphase vs baseline
        rather than asserting an absolute threshold.
        """
        n_steps = 200

        # Baseline without broadphase
        solver_off, task_off = _make_solver_with_task(panda_robot, enable_broadphase=False)
        _, configs_off = _run_circular_trajectory(
            panda_robot, solver_off, task_off, n_steps=n_steps
        )
        worst_off = 1.0
        for q in configs_off:
            panda_robot.update_configuration(q)
            worst_off = min(worst_off, panda_robot.compute_min_collision_distance())

        # With broadphase
        solver_on, task_on = _make_solver_with_task(panda_robot, enable_broadphase=True)
        _, configs_on = _run_circular_trajectory(panda_robot, solver_on, task_on, n_steps=n_steps)
        worst_on = 1.0
        for q in configs_on:
            panda_robot.update_configuration(q)
            worst_on = min(worst_on, panda_robot.compute_min_collision_distance())

        # Broadphase must not make penetration meaningfully worse (1mm tolerance)
        assert worst_on >= worst_off - 1e-3, (
            f"Broadphase worsened penetration: "
            f"worst_on={worst_on:.6f} vs worst_off={worst_off:.6f}"
        )


# ---------------------------------------------------------------------------
# Task 5: Performance tests
# ---------------------------------------------------------------------------


class TestSphereBroadphasePerformance:
    """Broadphase should reduce exact queries and not regress timing."""

    def test_sphere_culling_reduces_exact_queries(self, panda_robot):
        """Sphere broadphase must cull at least 30% of collision pairs."""
        solver, task = _make_solver_with_task(panda_robot, enable_broadphase=True)
        solver.enable_timing_breakdown(True)

        results, _ = _run_circular_trajectory(panda_robot, solver, task, n_steps=100)

        total_culled = sum(r.collision_sphere_culled_pairs for r in results)
        total_exact = sum(r.collision_exact_distance_queries for r in results)
        total = total_culled + total_exact

        assert total > 0, "No collision queries observed"
        culling_ratio = total_culled / total
        assert culling_ratio > 0.30, (
            f"Culling ratio = {culling_ratio:.1%}, expected > 30% "
            f"(culled={total_culled}, exact={total_exact})"
        )

    def test_sphere_broadphase_reduces_collision_time(self, panda_robot):
        """Broadphase collision time should not be worse than baseline (20% tolerance)."""
        n_steps = 100

        # --- Baseline (no broadphase) ---
        solver_off, task_off = _make_solver_with_task(panda_robot, enable_broadphase=False)
        solver_off.enable_timing_breakdown(True)
        results_off, _ = _run_circular_trajectory(
            panda_robot, solver_off, task_off, n_steps=n_steps
        )
        times_off = [r.collision_constraint_time_ms for r in results_off]

        # --- With broadphase ---
        solver_on, task_on = _make_solver_with_task(panda_robot, enable_broadphase=True)
        solver_on.enable_timing_breakdown(True)
        results_on, _ = _run_circular_trajectory(panda_robot, solver_on, task_on, n_steps=n_steps)
        times_on = [r.collision_constraint_time_ms for r in results_on]

        median_off = float(np.median(times_off))
        median_on = float(np.median(times_on))

        # Allow 20% tolerance: broadphase should not be significantly slower
        assert median_on < median_off * 1.20, (
            f"Broadphase slower: median_on={median_on:.3f}ms vs "
            f"median_off={median_off:.3f}ms (ratio={median_on/median_off:.2f})"
        )
