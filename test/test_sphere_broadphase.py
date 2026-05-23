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
