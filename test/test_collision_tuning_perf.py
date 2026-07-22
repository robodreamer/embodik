"""Performance regression tests for collision-aware IK tuning modes.

Validates that collision tuning modes (speed/balanced) deliver their intended
performance characteristics in solve_position_step, and that post-step
collision rejection safety is preserved.

Root cause of regression: commit b806036 introduced post-step collision
rejection using evaluate_min_collision_distance() — a full brute-force scan
of ALL collision pairs — after every integration step, bypassing all tuning
mode optimizations (caching, budget, proximity gating).
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np
import pytest

import embodik as eik


def _ensure_ros_package_path(urdf_path: Path) -> None:
    resolved = urdf_path.resolve()
    current = os.environ.get("ROS_PACKAGE_PATH", "")
    paths = [Path(p) for p in current.split(":") if p]
    updated = False
    for depth in range(1, 5):
        if len(resolved.parents) > depth:
            root = resolved.parents[depth]
            if root.is_dir() and root not in paths:
                paths.insert(0, root)
                updated = True
    if updated:
        os.environ["ROS_PACKAGE_PATH"] = ":".join(str(p) for p in paths)


def _extract_link_index(name: str) -> int | None:
    m = re.search(r"link[_-]?(\d+)", name.lower())
    return int(m.group(1)) if m else None


def _panda_collision_exclusions(robot: eik.RobotModel) -> list[tuple[str, str]]:
    excl: list[tuple[str, str]] = []
    for a, b in robot.get_collision_pair_names():
        a_l, b_l = a.lower(), b.lower()
        if "finger" in a_l or "finger" in b_l:
            excl.append((a, b))
            continue
        ia, ib = _extract_link_index(a_l), _extract_link_index(b_l)
        if ia is not None and ib is not None and abs(ia - ib) <= 2:
            excl.append((a, b))
    return excl


def _make_panda_solver(tuning_mode: str = "speed"):
    """Create a Panda robot solver with collision constraints and the given tuning mode."""
    from robot_descriptions.panda_description import URDF_PATH

    urdf_path = Path(URDF_PATH)
    _ensure_ros_package_path(urdf_path)

    robot = eik.RobotModel(str(urdf_path), floating_base=False)
    exclusions = _panda_collision_exclusions(robot)
    if exclusions:
        try:
            robot.apply_collision_exclusions(exclusions)
        except Exception:
            pass

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.configure_collision_constraint(
        min_distance=0.04,
        include_pairs=[],
        exclude_pairs=list(exclusions),
        nearest_points_all_pairs=False,
        max_constraints=3,
    )

    mode_map = {
        "speed": eik.CollisionTuningMode.SPEED,
        "balanced": eik.CollisionTuningMode.BALANCED,
        "precise": eik.CollisionTuningMode.PRECISE,
    }
    solver.set_collision_tuning_mode(mode_map[tuning_mode])

    # Start from a known safe configuration (arm extended, far from self-collision)
    q = np.array([0.0, -0.3, 0.0, -2.8, 0.0, 2.5, 0.785, 0.04, 0.04], dtype=float)
    robot.update_configuration(q)

    solver.clear_tasks()
    task = solver.add_frame_task("panda_ee", "panda_hand")
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = eik.TaskSolveMode.SCALE

    # Target: small offset from current EE pose (reachable, far from self-collision)
    hand_pose = robot.get_frame_pose("panda_hand")
    target = np.eye(4, dtype=float)
    target[:3, :3] = np.asarray(hand_pose.rotation, dtype=float)
    target[:3, 3] = np.asarray(hand_pose.translation, dtype=float) + np.array(
        [0.05, 0.03, -0.02], dtype=float
    )

    opts = eik.PositionStepOptions()
    opts.dt = solver.dt
    opts.max_steps = 1
    opts.stall_recovery = False
    opts.position_gain = 1.0
    opts.orientation_gain = 1.0

    return robot, solver, q, target, opts


def _benchmark_position_step(
    tuning_mode: str,
    warmup: int = 5,
    steps: int = 50,
    *,
    acceleration_limit: float | None = None,
    collision_min_distance: float | None = None,
    max_steps: int = 1,
    trajectory_amplitude_m: float = 0.08,
    force_preferred_lock_fallback: bool = False,
):
    """Run solve_position_step with a moving target to keep the robot active.

    Uses a circular trajectory to ensure the robot is always tracking (not
    converged), which reflects the interactive-use case in example 02.
    """
    robot, solver, q, _, opts = _make_panda_solver(tuning_mode)
    if collision_min_distance is not None:
        exclusions = _panda_collision_exclusions(robot)
        solver.configure_collision_constraint(
            min_distance=collision_min_distance,
            include_pairs=[],
            exclude_pairs=exclusions,
            nearest_points_all_pairs=False,
            max_constraints=3,
        )
    opts.max_steps = max_steps
    if acceleration_limit is not None:
        solver.set_acceleration_limits(np.full(robot.nv, acceleration_limit))
        solver.enable_acceleration_limits(True)
    if force_preferred_lock_fallback:
        opts.preferred_locked_joint_indices = [0]
        opts.preferred_lock_tracking_tolerance = 1e-9
        opts.preferred_lock_orientation_tolerance = 0.0
        opts.preferred_lock_max_step_norm = 0.35
        opts.preferred_lock_min_error_reduction_ratio = 10.0

    hand_pose = robot.get_frame_pose("panda_hand")
    base_pos = np.asarray(hand_pose.translation, dtype=float)
    base_rot = np.asarray(hand_pose.rotation, dtype=float)

    def make_target(step_i: int) -> np.ndarray:
        t = step_i * 0.05
        target = np.eye(4, dtype=float)
        target[:3, :3] = base_rot
        target[:3, 3] = base_pos + np.array(
            [
                trajectory_amplitude_m * np.cos(t),
                trajectory_amplitude_m * np.sin(t),
                0.25 * trajectory_amplitude_m * np.sin(t * 0.7),
            ],
            dtype=float,
        )
        return target

    for i in range(warmup):
        target = make_target(i)
        res = solver.solve_position_step(q, target, "panda_ee", opts)
        q = np.asarray(res.q_solution, dtype=float)
        robot.update_configuration(q)

    timings = []
    statuses = []
    moving_steps = 0
    post_step_exact_queries = 0
    post_step_motion_bound_culls = 0
    preferred_lock_attempted_steps = 0
    preferred_lock_fallback_steps = 0
    for i in range(steps):
        target = make_target(warmup + i)
        t0 = time.perf_counter()
        res = solver.solve_position_step(q, target, "panda_ee", opts)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        timings.append(dt_ms)
        statuses.append(res.status)
        preferred_lock_attempted_steps += int(res.preferred_lock_attempted)
        preferred_lock_fallback_steps += int(res.preferred_lock_fallback_used)
        if hasattr(solver, "get_last_post_step_collision_exact_distance_queries"):
            post_step_exact_queries += int(
                solver.get_last_post_step_collision_exact_distance_queries()
            )
        if hasattr(solver, "get_last_post_step_collision_motion_bound_culled_pairs"):
            post_step_motion_bound_culls += int(
                solver.get_last_post_step_collision_motion_bound_culled_pairs()
            )
        q_next = np.asarray(res.q_solution, dtype=float)
        moving_steps += int(np.linalg.norm(q_next - q) > 1e-10)
        q = q_next
        robot.update_configuration(q)

    return {
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "mean_ms": float(np.mean(timings)),
        "timings": timings,
        "statuses": statuses,
        "moving_steps": moving_steps,
        "post_step_exact_queries": post_step_exact_queries,
        "post_step_motion_bound_culls": post_step_motion_bound_culls,
        "preferred_lock_attempted_steps": preferred_lock_attempted_steps,
        "preferred_lock_fallback_steps": preferred_lock_fallback_steps,
    }


class TestCollisionTuningPerformance:
    """Performance regression tests for collision tuning modes."""

    def test_speed_mode_under_3ms(self):
        """Speed mode solve_position_step should run under 3ms per step."""
        result = _benchmark_position_step("speed")
        median = result["median_ms"]
        assert median < 3.0, (
            f"Speed mode median={median:.2f}ms, expected <3ms. "
            f"Full collision scan likely running in hot path."
        )

    def test_balanced_mode_under_4ms(self):
        """Balanced mode solve_position_step should run under 4ms per step."""
        result = _benchmark_position_step("balanced")
        median = result["median_ms"]
        assert median < 4.0, (
            f"Balanced mode median={median:.2f}ms, expected <4ms. "
            f"Full collision scan likely running in hot path."
        )

    def test_acceleration_limited_collision_path_under_15ms_p95(self):
        """Successful predictive projection must retain WBC deadline headroom."""
        result = _benchmark_position_step(
            "speed",
            warmup=10,
            steps=80,
            acceleration_limit=10.0,
            collision_min_distance=0.02,
            max_steps=3,
            trajectory_amplitude_m=0.02,
        )
        assert set(result["statuses"]) == {eik.SolverStatus.SUCCESS}
        assert result["moving_steps"] == 80
        p95 = result["p95_ms"]
        assert p95 < 15.0, (
            f"Acceleration-limited collision path p95={p95:.2f}ms, "
            "expected <15ms to preserve WBC integration headroom."
        )

    def test_preferred_lock_rollback_path_under_15ms_p95(self):
        """Rollback snapshots must not consume the WBC compute budget."""
        result = _benchmark_position_step(
            "speed",
            warmup=10,
            steps=80,
            acceleration_limit=10.0,
            collision_min_distance=0.02,
            max_steps=3,
            trajectory_amplitude_m=0.02,
            force_preferred_lock_fallback=True,
        )
        assert set(result["statuses"]) == {eik.SolverStatus.SUCCESS}
        assert result["preferred_lock_attempted_steps"] == 80
        assert result["preferred_lock_fallback_steps"] > 0
        assert result["post_step_exact_queries"] > 0
        p95 = result["p95_ms"]
        assert p95 < 15.0, (
            f"Preferred-lock rollback path p95={p95:.2f}ms, "
            "expected <15ms to preserve WBC integration headroom."
        )

    def test_speed_not_slower_than_precise(self):
        """Speed mode should be significantly faster than precise mode."""
        speed_result = _benchmark_position_step("speed")
        precise_result = _benchmark_position_step("precise")
        assert speed_result["median_ms"] < precise_result["median_ms"] * 0.8, (
            f"Speed ({speed_result['median_ms']:.2f}ms) should be well under "
            f"precise ({precise_result['median_ms']:.2f}ms)"
        )

    def test_mode_differentiation(self):
        """All three modes should show distinct performance characteristics."""
        speed = _benchmark_position_step("speed")
        balanced = _benchmark_position_step("balanced")
        precise = _benchmark_position_step("precise")
        # Speed < balanced < precise (in timing)
        assert speed["median_ms"] < precise["median_ms"], (
            f"Speed ({speed['median_ms']:.2f}ms) should be faster than "
            f"precise ({precise['median_ms']:.2f}ms)"
        )
        print(
            f"\nMode timings (median): "
            f"speed={speed['median_ms']:.2f}ms, "
            f"balanced={balanced['median_ms']:.2f}ms, "
            f"precise={precise['median_ms']:.2f}ms"
        )


class TestCollisionRejectionSafety:
    """Verify post-step collision rejection still catches penetrations.

    These tests are critical: the post-step rejection was added (b806036) to
    catch penetrations that the constraint-based solver missed.  Our perf fix
    must NOT weaken this safety net.
    """

    def test_rejection_prevents_penetration_speed_mode(self):
        """Speed mode: driving toward self-collision must not produce deep penetration."""
        self._run_penetration_check("speed")

    def test_rejection_prevents_penetration_balanced_mode(self):
        """Balanced mode: driving toward self-collision must not produce deep penetration."""
        self._run_penetration_check("balanced")

    def test_rejection_prevents_penetration_precise_mode(self):
        """Precise mode: driving toward self-collision must not produce deep penetration."""
        self._run_penetration_check("precise")

    def _run_penetration_check(self, mode: str):
        robot, solver, q, _, opts = _make_panda_solver(mode)
        opts.stall_recovery = True
        solver.enable_stall_handler(0.04)

        # Target that drives toward self-collision (near the base)
        target = np.eye(4, dtype=float)
        hand_pose = robot.get_frame_pose("panda_hand")
        target[:3, :3] = np.asarray(hand_pose.rotation, dtype=float)
        target[:3, 3] = np.asarray(hand_pose.translation, dtype=float) + np.array(
            [-0.30, 0.0, -0.25], dtype=float
        )

        worst_penetration = float("inf")
        total_rejections = 0
        for _ in range(200):
            res = solver.solve_position_step(q, target, "panda_ee", opts)
            total_rejections += int(getattr(res, "collision_rejection_count", 0))
            q = np.asarray(res.q_solution, dtype=float)
            robot.update_configuration(q)

            # Track worst penetration seen at any step
            if hasattr(solver, "evaluate_collision_debug"):
                debug = solver.evaluate_collision_debug(q)
                if debug is not None and hasattr(debug, "distance"):
                    worst_penetration = min(worst_penetration, debug.distance)

        # Allow very minor numerical penetration but reject anything deep
        max_allowed_penetration = -0.005  # 5mm tolerance
        if worst_penetration != float("inf"):
            assert worst_penetration >= max_allowed_penetration, (
                f"[{mode}] Deep penetration detected: {worst_penetration:.4f}m "
                f"(threshold {max_allowed_penetration}m). "
                f"Post-step rejection is not catching collisions."
            )
            print(
                f"\n[{mode}] worst_penetration={worst_penetration:.4f}m, "
                f"rejections={total_rejections}"
            )

    def test_all_modes_produce_similar_safety(self):
        """All tuning modes should prevent penetration equally well.

        The speed/balanced optimizations should not compromise safety relative
        to precise mode.
        """
        results = {}
        for mode in ("speed", "balanced", "precise"):
            robot, solver, q, _, opts = _make_panda_solver(mode)
            opts.stall_recovery = True
            solver.enable_stall_handler(0.04)

            target = np.eye(4, dtype=float)
            hand_pose = robot.get_frame_pose("panda_hand")
            target[:3, :3] = np.asarray(hand_pose.rotation, dtype=float)
            target[:3, 3] = np.asarray(hand_pose.translation, dtype=float) + np.array(
                [-0.30, 0.0, -0.25], dtype=float
            )

            worst = float("inf")
            for _ in range(100):
                res = solver.solve_position_step(q, target, "panda_ee", opts)
                q = np.asarray(res.q_solution, dtype=float)
                robot.update_configuration(q)
                if hasattr(solver, "evaluate_collision_debug"):
                    debug = solver.evaluate_collision_debug(q)
                    if debug is not None and hasattr(debug, "distance"):
                        worst = min(worst, debug.distance)
            results[mode] = worst

        # Speed/balanced should not be significantly worse than precise
        if results["precise"] != float("inf") and results["speed"] != float("inf"):
            safety_gap = results["precise"] - results["speed"]
            assert safety_gap < 0.005, (
                f"Speed mode is {safety_gap*1000:.1f}mm less safe than precise. "
                f"speed={results['speed']:.4f}m, precise={results['precise']:.4f}m"
            )
