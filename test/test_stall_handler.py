"""Tests for the C++ stall handler and PositionStepOptions.stall_recovery."""

import os
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytest

import embodik as eik
from embodik import StallHandler, TaskSolveMode

_PANDA_DEFAULT_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
_PANDA_GRIPPER_EXTRA = np.array([0.05, 0.05])
_PANDA_EE_FRAME = "panda_hand"
_DQ_STALL_EPS = 1e-5

_LINK_INDEX_PATTERN = re.compile(r"link_?([0-9]+)")


def _extract_link_index(name: str) -> Optional[int]:
    match = _LINK_INDEX_PATTERN.search(name)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Panda fixtures
# ---------------------------------------------------------------------------

def _make_panda_solver():
    from robot_descriptions.panda_description import URDF_PATH

    robot = eik.RobotModel(URDF_PATH, floating_base=False)
    q_init = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
    robot.update_configuration(q_init)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    task = solver.add_frame_task("ee", _PANDA_EE_FRAME)
    task.weight = 1.0
    task.solve_mode = TaskSolveMode.SCALE
    return robot, solver, task


def _should_panda_exclude_pair(name_a: str, name_b: str) -> bool:
    a_lower, b_lower = name_a.lower(), name_b.lower()
    ee_tokens = ("finger", "hand")
    a_ee = any(t in a_lower for t in ee_tokens)
    b_ee = any(t in b_lower for t in ee_tokens)
    if a_ee and b_ee:
        return True
    idx_a, idx_b = _extract_link_index(a_lower), _extract_link_index(b_lower)
    if a_ee != b_ee:
        other = idx_b if a_ee else idx_a
        return other is not None and other >= 5
    if idx_a is None or idx_b is None:
        return False
    return abs(idx_a - idx_b) <= 2


def _panda_collision_exclusions(robot):
    return [
        (a, b) for a, b in robot.get_collision_pair_names()
        if _should_panda_exclude_pair(a, b)
    ]


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


def _setup_panda_stall():
    """Set up a Panda robot in a configuration that reliably stalls."""
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
    solver.set_damping(0.1)

    q = np.array([0.0, -0.3, 0.0, -2.8, 0.0, 2.5, 0.785, 0.04, 0.04])
    robot.update_configuration(q)

    dbg = solver.evaluate_collision_debug(q)
    if dbg is None or not np.isfinite(dbg.distance):
        return None

    min_dist = 0.04
    solver.configure_collision_constraint(
        min_distance=min_dist,
        include_pairs=[],
        exclude_pairs=list(exclusions),
    )

    solver.clear_tasks()
    task = solver.add_frame_task("panda_stall", _PANDA_EE_FRAME)
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = eik.TaskSolveMode.SCALE

    hand_pos = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).translation)
    target_pos = hand_pos + np.array([-0.30, 0.0, -0.25])
    target_rot = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).rotation)
    task.set_target_pose(target_pos, target_rot)

    return robot, solver, q, task, target_pos, min_dist


# ---------------------------------------------------------------------------
# Dual-iiwa fixtures
# ---------------------------------------------------------------------------

def _dual_iiwa_exclusions(robot):
    excl = []
    for a, b in robot.get_collision_pair_names():
        a_l, b_l = a.lower(), b.lower()
        same_arm = (("left" in a_l and "left" in b_l) or
                    ("right" in a_l and "right" in b_l))
        if same_arm:
            ia, ib = _extract_link_index(a_l), _extract_link_index(b_l)
            if ia is not None and ib is not None and abs(ia - ib) <= 3:
                excl.append((a, b))
        else:
            ia, ib = _extract_link_index(a_l), _extract_link_index(b_l)
            if ia is not None and ib is not None and ia <= 1 and ib <= 1:
                excl.append((a, b))
    return excl


def _setup_dual_iiwa_stall():
    """Set up dual iiwa in a cross-arm stall config."""
    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))

    try:
        from utils.dual_iiwa_urdf import (
            build_dual_iiwa_urdf,
            get_dual_iiwa_frame_names,
            get_dual_iiwa_default_configuration,
        )
    except ImportError:
        return None

    urdf_str = build_dual_iiwa_urdf()
    with tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False) as f:
        f.write(urdf_str)
        urdf_path = f.name

    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
    except Exception:
        return None
    finally:
        os.unlink(urdf_path)

    excl = _dual_iiwa_exclusions(robot)
    if excl:
        try:
            robot.apply_collision_exclusions(excl)
        except Exception:
            pass

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)

    q = np.array(get_dual_iiwa_default_configuration(), dtype=float)
    robot.update_configuration(q)

    dbg = solver.evaluate_collision_debug(q)
    if dbg is None or not np.isfinite(dbg.distance):
        return None

    min_dist = 0.05
    solver.configure_collision_constraint(
        min_distance=min_dist, include_pairs=[], exclude_pairs=list(excl),
    )

    left_frame, right_frame = get_dual_iiwa_frame_names()

    solver.clear_tasks()
    left_task = solver.add_frame_task("left_stall", left_frame)
    left_task.priority = 0
    left_task.weight = 1.0
    left_task.solve_mode = eik.TaskSolveMode.SCALE

    left_pos = np.array(robot.get_frame_pose(left_frame).translation)
    left_rot = np.array(robot.get_frame_pose(left_frame).rotation)
    left_target = left_pos + np.array([0.15, -0.35, -0.10])
    left_task.set_target_pose(left_target, left_rot)

    right_task = solver.add_frame_task("right_stall", right_frame)
    right_task.priority = 0
    right_task.weight = 1.0
    right_task.solve_mode = eik.TaskSolveMode.SCALE
    right_pos = np.array(robot.get_frame_pose(right_frame).translation)
    right_rot = np.array(robot.get_frame_pose(right_frame).rotation)
    right_target = right_pos + np.array([0.15, 0.35, -0.10])
    right_task.set_target_pose(right_target, right_rot)

    return robot, solver, q, left_frame, left_target, min_dist


# ---------------------------------------------------------------------------
# Helper: run a velocity loop and count stall steps
# ---------------------------------------------------------------------------

def _run_velocity_loop(robot, solver, q, steps):
    """Run solve_velocity for `steps` and return (stall_count, final_q)."""
    stall_count = 0
    for _ in range(steps):
        result = solver.solve_velocity(q, apply_limits=True)
        dq = np.array(result.joint_velocities)
        if np.linalg.norm(dq) < _DQ_STALL_EPS:
            stall_count += 1
        q = robot.integrate(q, dq, solver.dt)
        q_lo, q_hi = robot.get_joint_limits()
        q = np.clip(q, q_lo, q_hi)
        robot.update_configuration(q)
    return stall_count, q


# ===================================================================
# API Tests
# ===================================================================

class TestStallHandlerAPI:
    def test_default_disabled(self):
        _, solver, _ = _make_panda_solver()
        assert not solver.stall_handler_enabled()

    def test_enable_disable_roundtrip(self):
        _, solver, _ = _make_panda_solver()
        solver.enable_stall_handler(0.04)
        assert solver.stall_handler_enabled()
        assert solver.stall_handler_current_min_distance() == pytest.approx(0.04)
        assert not solver.stall_handler_is_relaxed()
        solver.disable_stall_handler()
        assert not solver.stall_handler_enabled()

    def test_configure_stall_handler(self):
        _, solver, _ = _make_panda_solver()
        solver.enable_stall_handler(0.05)
        solver.configure_stall_handler(
            stall_threshold=10,
            restore_rate=0.01, floor_fraction=0.5,
        )
        assert solver.stall_handler_enabled()

    def test_python_wrapper_enables(self):
        _, solver, _ = _make_panda_solver()
        handler = StallHandler(solver, nominal_min_distance=0.04)
        assert handler.enabled
        assert handler.current_min_distance == pytest.approx(0.04)
        assert not handler.is_relaxed
        handler.disable()
        assert not handler.enabled

    def test_consecutive_counter_starts_at_zero(self):
        _, solver, _ = _make_panda_solver()
        solver.enable_stall_handler(0.04)
        assert solver.stall_handler_consecutive_stall_steps() == 0

    def test_position_step_options_stall_recovery_default_false(self):
        opts = eik.PositionStepOptions()
        assert opts.stall_recovery is False

    def test_position_step_options_stall_recovery_settable(self):
        opts = eik.PositionStepOptions()
        opts.stall_recovery = True
        assert opts.stall_recovery is True

    def test_position_ik_options_stall_recovery_default_false(self):
        opts = eik.PositionIKOptions()
        assert opts.stall_recovery is False

    def test_position_ik_options_stall_recovery_settable(self):
        opts = eik.PositionIKOptions()
        opts.stall_recovery = True
        assert opts.stall_recovery is True


# ===================================================================
# Recovery Tests (solve_velocity level)
# ===================================================================

class TestStallHandlerRecovery:
    def test_margin_restores_after_healthy_steps(self):
        """With a target at current pose (no stall), margin stays at nominal."""
        robot, solver, task = _make_panda_solver()
        solver.enable_stall_handler(0.04)
        solver.configure_stall_handler(stall_threshold=2)

        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q)
        robot.update_kinematics(q)
        task.set_target_pose(
            np.array(task.current_position),
            np.array(task.current_orientation),
        )

        for _ in range(20):
            result = solver.solve_velocity(q, apply_limits=True)
            q = robot.integrate(q, result.joint_velocities, solver.dt)
            robot.update_configuration(q)

        assert not solver.stall_handler_is_relaxed()

    def test_handler_no_crash_without_collision(self):
        robot, solver, task = _make_panda_solver()
        solver.enable_stall_handler(0.04)
        solver.configure_stall_handler(stall_threshold=3)

        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q)
        robot.update_kinematics(q)
        target_pos = np.array(task.current_position) + np.array([0.1, 0.05, -0.1])
        task.set_target_pose(target_pos, np.array(task.current_orientation))

        for _ in range(200):
            result = solver.solve_velocity(q, apply_limits=True)
            q = robot.integrate(q, result.joint_velocities, solver.dt)
            robot.update_configuration(q)

        assert solver.stall_handler_enabled()


# ===================================================================
# A/B Stall Scenario Tests
# ===================================================================

class TestPandaStallRecovery:
    """Panda self-collision stall: verify handler does not increase stalls."""

    def test_panda_stall_recovery_does_not_increase_stalls(self):
        setup = _setup_panda_stall()
        if setup is None:
            pytest.skip("Panda collision model not available")

        robot, solver, q0, task, target_pos, min_dist = setup
        steps = 200

        # Baseline run (no handler)
        robot.update_configuration(q0)
        task.set_target_pose(
            target_pos,
            np.array(robot.get_frame_pose(_PANDA_EE_FRAME).rotation),
        )
        baseline_stalls, _ = _run_velocity_loop(robot, solver, q0.copy(), steps)

        # Unstall run (handler enabled)
        robot.update_configuration(q0)
        task.set_target_pose(
            target_pos,
            np.array(robot.get_frame_pose(_PANDA_EE_FRAME).rotation),
        )
        solver.enable_stall_handler(min_dist)
        solver.configure_stall_handler(stall_threshold=5)
        handler_stalls, _ = _run_velocity_loop(robot, solver, q0.copy(), steps)
        solver.disable_stall_handler()

        assert handler_stalls <= baseline_stalls, (
            f"Handler should not increase stalls: {handler_stalls} > {baseline_stalls}"
        )


class TestDualIiwaStallRecovery:
    """Dual-iiwa cross-arm stall: verify handler reduces stalls."""

    def test_dual_iiwa_stall_recovery_reduces_stalls(self):
        setup = _setup_dual_iiwa_stall()
        if setup is None:
            pytest.skip("Dual iiwa model not available")

        robot, solver, q0, left_frame, left_target, min_dist = setup
        steps = 200

        baseline_stalls, _ = _run_velocity_loop(robot, solver, q0.copy(), steps)

        # Reset and re-setup for handler run
        setup2 = _setup_dual_iiwa_stall()
        robot2, solver2, q02, _, _, min_dist2 = setup2
        solver2.enable_stall_handler(min_dist2)
        solver2.configure_stall_handler(stall_threshold=5)
        handler_stalls, _ = _run_velocity_loop(robot2, solver2, q02.copy(), steps)
        solver2.disable_stall_handler()

        assert handler_stalls < baseline_stalls, (
            f"Handler should reduce stalls: {handler_stalls} >= {baseline_stalls}"
        )


# ===================================================================
# PositionStepOptions.stall_recovery integration
# ===================================================================

class TestPositionStepStallRecovery:
    """Verify opts.stall_recovery works end-to-end through solve_position_step."""

    def test_stall_recovery_opt_in_no_crash(self):
        """Calling solve_position_step with stall_recovery=True should not crash
        and the handler should persist across calls."""
        robot, solver, task = _make_panda_solver()

        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q)
        robot.update_kinematics(q)
        target = np.eye(4)
        target[:3, 3] = np.array(task.current_position) + np.array([0.05, 0.02, -0.03])
        target[:3, :3] = np.array(task.current_orientation)

        opts = eik.PositionStepOptions()
        opts.position_gain = 20.0
        opts.orientation_gain = 20.0
        opts.stall_recovery = True

        for _ in range(50):
            result = solver.solve_position_step(q, target, "ee", opts)
            q = result.q_solution

        assert solver.stall_handler_enabled(), (
            "Handler should persist across solve_position_step calls "
            "so stall counts can accumulate in single-step-per-tick loops"
        )
        solver.disable_stall_handler()

    def test_stall_recovery_does_not_interfere_when_off(self):
        """stall_recovery=False (default) should not enable the handler."""
        robot, solver, task = _make_panda_solver()

        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q)
        robot.update_kinematics(q)
        target = np.eye(4)
        target[:3, 3] = np.array(task.current_position) + np.array([0.05, 0, 0])
        target[:3, :3] = np.array(task.current_orientation)

        opts = eik.PositionStepOptions()
        opts.position_gain = 20.0
        opts.orientation_gain = 20.0

        result = solver.solve_position_step(q, target, "ee", opts)
        assert not solver.stall_handler_enabled()

    def test_stall_recovery_respects_already_enabled(self):
        """If stall handler is already enabled externally, stall_recovery=True
        should not disable it afterward."""
        robot, solver, task = _make_panda_solver()

        solver.enable_stall_handler(0.0)

        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q)
        robot.update_kinematics(q)
        target = np.eye(4)
        target[:3, 3] = np.array(task.current_position)
        target[:3, :3] = np.array(task.current_orientation)

        opts = eik.PositionStepOptions()
        opts.stall_recovery = True

        solver.solve_position_step(q, target, "ee", opts)
        assert solver.stall_handler_enabled(), (
            "Should not disable externally-enabled handler"
        )

        solver.disable_stall_handler()

    def test_stall_counter_accumulates_across_single_step_calls(self):
        """Regression: in a teleop-style loop (max_steps=1, one call per tick),
        the stall counter must accumulate across calls so recovery triggers."""
        setup = _setup_panda_stall()
        if setup is None:
            pytest.skip("Panda collision debug unavailable")

        robot, solver, q, task, target_pos, min_dist = setup

        opts = eik.PositionStepOptions()
        opts.max_steps = 1
        opts.stall_recovery = True
        opts.position_gain = 20.0
        opts.orientation_gain = 20.0

        target = np.eye(4)
        target[:3, 3] = target_pos
        target[:3, :3] = np.array(
            robot.get_frame_pose(_PANDA_EE_FRAME).rotation
        )

        stall_counts = []
        for i in range(20):
            result = solver.solve_position_step(q, target, "panda_stall", opts)
            q = np.asarray(result.q_solution, dtype=float)
            stall_counts.append(solver.stall_handler_consecutive_stall_steps())

        assert solver.stall_handler_enabled(), "Handler should persist"

        max_count = max(stall_counts)
        assert max_count > 1, (
            f"Stall counter should accumulate across single-step calls but "
            f"max was {max_count}; counts={stall_counts}"
        )

        solver.disable_stall_handler()


# ===================================================================
# solve_velocity(stall_recovery=True) integration
# ===================================================================

class TestSolveVelocityStallRecovery:
    """Verify the stall_recovery flag on solve_velocity."""

    def test_stall_recovery_enables_handler(self):
        """Passing stall_recovery=True should auto-enable the stall handler."""
        robot, solver, task = _make_panda_solver()
        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q)
        robot.update_kinematics(q)
        target_pos = np.array(task.current_position) + np.array([0.05, 0, 0])
        task.set_target_pose(target_pos, np.array(task.current_orientation))

        assert not solver.stall_handler_enabled()
        solver.solve_velocity(q, apply_limits=True, stall_recovery=True)
        assert solver.stall_handler_enabled(), (
            "stall_recovery=True should enable the handler"
        )
        solver.disable_stall_handler()

    def test_stall_recovery_persists_across_calls(self):
        """Handler stays enabled across multiple solve_velocity calls."""
        robot, solver, task = _make_panda_solver()
        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q)
        robot.update_kinematics(q)
        target_pos = np.array(task.current_position) + np.array([0.05, 0, 0])
        task.set_target_pose(target_pos, np.array(task.current_orientation))

        for _ in range(10):
            result = solver.solve_velocity(q, apply_limits=True, stall_recovery=True)
            q = robot.integrate(q, result.joint_velocities, solver.dt)
            robot.update_configuration(q)

        assert solver.stall_handler_enabled()
        solver.disable_stall_handler()

    def test_stall_recovery_false_does_not_enable(self):
        """stall_recovery=False (default) should not enable the handler."""
        robot, solver, task = _make_panda_solver()
        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q)
        solver.solve_velocity(q, apply_limits=True, stall_recovery=False)
        assert not solver.stall_handler_enabled()

    def test_stall_recovery_respects_already_enabled(self):
        """If handler already enabled, stall_recovery=True is idempotent."""
        robot, solver, task = _make_panda_solver()
        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q)

        solver.enable_stall_handler(0.04)
        solver.solve_velocity(q, apply_limits=True, stall_recovery=True)
        assert solver.stall_handler_enabled()
        assert solver.stall_handler_current_min_distance() == pytest.approx(0.04)
        solver.disable_stall_handler()

    def test_solve_velocity_dq_with_stall_recovery(self):
        """solve_velocity_dq should also accept stall_recovery."""
        robot, solver, task = _make_panda_solver()
        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q)
        robot.update_kinematics(q)
        target_pos = np.array(task.current_position) + np.array([0.05, 0, 0])
        task.set_target_pose(target_pos, np.array(task.current_orientation))

        dq = solver.solve_velocity_dq(q, apply_limits=True, stall_recovery=True)
        assert dq is not None
        assert solver.stall_handler_enabled()
        solver.disable_stall_handler()


# ===================================================================
# solve_position(stall_recovery=True) integration
# ===================================================================

class TestSolvePositionStallRecovery:
    """Verify stall_recovery on PositionIKOptions for solve_position."""

    def test_stall_recovery_no_crash(self):
        """Calling solve_position with stall_recovery=True should not crash."""
        from robot_descriptions.panda_description import URDF_PATH

        robot = eik.RobotModel(URDF_PATH, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01

        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q)

        current_pose = robot.get_frame_pose(_PANDA_EE_FRAME)
        target = np.eye(4)
        target[:3, 3] = np.array(current_pose.translation) + np.array([0.05, 0, 0])
        target[:3, :3] = np.array(current_pose.rotation)

        opts = eik.PositionIKOptions()
        opts.stall_recovery = True
        opts.max_iterations = 50
        result = solver.solve_position(q, target, _PANDA_EE_FRAME, opts)
        assert result.q_solution is not None
        assert not solver.stall_handler_enabled(), (
            "Handler should be auto-disabled after solve_position"
        )

    def test_stall_recovery_does_not_interfere_when_off(self):
        """stall_recovery=False should not enable the handler."""
        from robot_descriptions.panda_description import URDF_PATH

        robot = eik.RobotModel(URDF_PATH, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01

        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q)

        current_pose = robot.get_frame_pose(_PANDA_EE_FRAME)
        target = np.eye(4)
        target[:3, 3] = np.array(current_pose.translation) + np.array([0.05, 0, 0])
        target[:3, :3] = np.array(current_pose.rotation)

        opts = eik.PositionIKOptions()
        opts.max_iterations = 50
        result = solver.solve_position(q, target, _PANDA_EE_FRAME, opts)
        assert not solver.stall_handler_enabled()

    def test_stall_recovery_respects_already_enabled(self):
        """If handler already enabled, it should remain active after solve_position."""
        from robot_descriptions.panda_description import URDF_PATH

        robot = eik.RobotModel(URDF_PATH, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01

        solver.enable_stall_handler(0.0)

        q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
        robot.update_configuration(q)

        current_pose = robot.get_frame_pose(_PANDA_EE_FRAME)
        target = np.eye(4)
        target[:3, 3] = np.array(current_pose.translation)
        target[:3, :3] = np.array(current_pose.rotation)

        opts = eik.PositionIKOptions()
        opts.stall_recovery = True
        opts.max_iterations = 10
        solver.solve_position(q, target, _PANDA_EE_FRAME, opts)
        assert solver.stall_handler_enabled(), (
            "Should not disable externally-enabled handler"
        )
        solver.disable_stall_handler()


# ===================================================================
# Dual-EE body stall: broadened detection, time budget, jump prevention
# ===================================================================

def _pose_to_4x4(frame_pose):
    """Convert a Pinocchio frame pose to a 4x4 numpy matrix."""
    T = np.eye(4)
    T[:3, :3] = np.array(frame_pose.rotation)
    T[:3, 3] = np.array(frame_pose.translation)
    return T


def _setup_dual_iiwa_body_stall():
    """Set up dual iiwa with BOTH EEs targeting deep inside the body.

    Returns (robot, solver, q0, left_target_4x4, right_target_4x4, min_dist)
    or None.
    """
    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))

    try:
        from utils.dual_iiwa_urdf import (
            build_dual_iiwa_urdf,
            get_dual_iiwa_frame_names,
            get_dual_iiwa_default_configuration,
        )
    except ImportError:
        return None

    urdf_str = build_dual_iiwa_urdf()
    with tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False) as f:
        f.write(urdf_str)
        urdf_path = f.name

    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
    except Exception:
        return None
    finally:
        os.unlink(urdf_path)

    excl = _dual_iiwa_exclusions(robot)
    if excl:
        try:
            robot.apply_collision_exclusions(excl)
        except Exception:
            pass

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)

    q = np.array(get_dual_iiwa_default_configuration(), dtype=float)
    robot.update_configuration(q)

    dbg = solver.evaluate_collision_debug(q)
    if dbg is None or not np.isfinite(dbg.distance):
        return None

    min_dist = 0.05
    solver.configure_collision_constraint(
        min_distance=min_dist, include_pairs=[], exclude_pairs=list(excl),
    )

    left_frame, right_frame = get_dual_iiwa_frame_names()

    solver.clear_tasks()
    left_task = solver.add_frame_task("left_body", left_frame)
    left_task.priority = 0
    left_task.weight = 1.0
    left_task.solve_mode = eik.TaskSolveMode.SCALE

    right_task = solver.add_frame_task("right_body", right_frame)
    right_task.priority = 0
    right_task.weight = 1.0
    right_task.solve_mode = eik.TaskSolveMode.SCALE

    left_pose = robot.get_frame_pose(left_frame)
    left_pos = np.array(left_pose.translation)
    left_rot = np.array(left_pose.rotation)
    left_target_pos = left_pos + np.array([0.0, -0.40, -0.10])
    left_task.set_target_pose(left_target_pos, left_rot)

    right_pose = robot.get_frame_pose(right_frame)
    right_pos = np.array(right_pose.translation)
    right_rot = np.array(right_pose.rotation)
    right_target_pos = right_pos + np.array([0.0, 0.40, -0.10])
    right_task.set_target_pose(right_target_pos, right_rot)

    # Build 4x4 target matrices for TaskTarget construction
    left_T = np.eye(4)
    left_T[:3, :3] = left_rot
    left_T[:3, 3] = left_target_pos

    right_T = np.eye(4)
    right_T[:3, :3] = right_rot
    right_T[:3, 3] = right_target_pos

    return robot, solver, q, left_T, right_T, min_dist


def _run_position_step_loop(solver, robot, q, targets, opts, steps):
    """Run solve_position_step with multi-target TaskTargets for N steps.

    Returns (q, times_ms, stall_counters).
    """
    times_ms = []
    stall_counters = []
    for _ in range(steps):
        result = solver.solve_position_step(q, targets, opts)
        q = result.q_solution
        robot.update_configuration(q)
        if result.computation_time_ms is not None:
            times_ms.append(result.computation_time_ms)
        counter = solver.stall_handler_consecutive_stall_steps()
        stall_counters.append(counter)
    return q, times_ms, stall_counters


class TestDualEEBodyStall:
    """Dual-EE body collision stall: detection, margin relaxation, recovery.

    All tests use solve_position_step with the multi-target TaskTarget overload
    to match the validation_robot teleop pattern.
    """

    def test_stall_recovery_progressively_relaxes_margin(self):
        """Stall handler should progressively relax the collision margin.

        Each stall → fallback → brief motion → re-stall cycle should ratchet
        the margin down further. Over enough steps the margin should decrease
        well below the initial nominal value.
        """
        setup = _setup_dual_iiwa_body_stall()
        if setup is None:
            pytest.skip("Dual iiwa model not available")

        robot, solver, q0, left_T, right_T, min_dist = setup
        opts = eik.PositionStepOptions()
        opts.stall_recovery = True
        opts.max_steps = 1

        targets = [
            eik.TaskTarget("left_body", left_T),
            eik.TaskTarget("right_body", right_T),
        ]

        q, _, _ = _run_position_step_loop(
            solver, robot, q0.copy(), targets, opts, 200,
        )

        final_min = solver.stall_handler_current_min_distance()
        assert final_min < min_dist * 0.8, (
            f"Margin only relaxed to {final_min:.4f} from {min_dist}; "
            f"expected at least 20% reduction"
        )
        solver.disable_stall_handler()

    def test_stall_counter_reaches_threshold_repeatedly(self):
        """Stall counter should reach the threshold (5) multiple times,
        triggering margin relaxation each cycle."""
        setup = _setup_dual_iiwa_body_stall()
        if setup is None:
            pytest.skip("Dual iiwa model not available")

        robot, solver, q0, left_T, right_T, min_dist = setup
        opts = eik.PositionStepOptions()
        opts.stall_recovery = True
        opts.max_steps = 1

        targets = [
            eik.TaskTarget("left_body", left_T),
            eik.TaskTarget("right_body", right_T),
        ]

        _, _, stall_counters = _run_position_step_loop(
            solver, robot, q0.copy(), targets, opts, 200,
        )

        max_counter = max(stall_counters)
        threshold_hits = sum(1 for c in stall_counters if c >= 4)
        assert max_counter >= 4, (
            f"Counter never approached threshold ({max_counter} < 4)"
        )
        assert threshold_hits >= 3, (
            f"Threshold hit only {threshold_hits} times; expected multiple "
            f"stall→relax→motion→re-stall cycles"
        )
        solver.disable_stall_handler()

    def test_computation_time_bounded_during_stall(self):
        """Per-step solve time should not spike when deeply stalled.

        Infeasible solves should terminate quickly even without an
        explicit iteration cap.
        """
        setup = _setup_dual_iiwa_body_stall()
        if setup is None:
            pytest.skip("Dual iiwa model not available")

        robot, solver, q0, left_T, right_T, min_dist = setup
        opts = eik.PositionStepOptions()
        opts.stall_recovery = True
        opts.max_steps = 1

        targets = [
            eik.TaskTarget("left_body", left_T),
            eik.TaskTarget("right_body", right_T),
        ]

        _, times_ms, _ = _run_position_step_loop(
            solver, robot, q0.copy(), targets, opts, 60,
        )

        if times_ms:
            max_time = max(times_ms)
            assert max_time < 10.0, (
                f"Max per-step time {max_time:.2f}ms exceeds 10ms budget"
            )
        solver.disable_stall_handler()

    def test_disable_task_no_penetration_jump(self):
        """Setting one EE task weight=0 during stall must NOT cause the
        freed arm to jump into deep body penetration."""
        setup = _setup_dual_iiwa_body_stall()
        if setup is None:
            pytest.skip("Dual iiwa model not available")

        robot, solver, q0, left_T, right_T, min_dist = setup
        opts = eik.PositionStepOptions()
        opts.stall_recovery = True
        opts.max_steps = 1

        both_targets = [
            eik.TaskTarget("left_body", left_T),
            eik.TaskTarget("right_body", right_T),
        ]

        # Drive into stall with both EEs
        q, _, _ = _run_position_step_loop(
            solver, robot, q0.copy(), both_targets, opts, 30,
        )

        # Disable left task (simulate user toggling off one EE)
        left_task = solver.get_task("left_body")
        left_task.weight = 0.0

        # Continue with only right EE target active
        right_only_targets = [eik.TaskTarget("right_body", right_T)]
        q, _, _ = _run_position_step_loop(
            solver, robot, q, right_only_targets, opts, 10,
        )

        dbg = solver.evaluate_collision_debug(q)
        if dbg is not None and np.isfinite(dbg.distance):
            assert dbg.distance > -0.02, (
                f"Collision distance {dbg.distance:.4f}m after disabling task; "
                f"arm jumped into deep penetration"
            )
        solver.disable_stall_handler()

    def test_pull_away_unstalls_quickly(self):
        """After stalling into the body and then reversing EE targets outward,
        the solver should recover quickly: motion should resume within the
        first few recovery steps."""
        setup = _setup_dual_iiwa_body_stall()
        if setup is None:
            pytest.skip("Dual iiwa model not available")

        robot, solver, q0, left_T, right_T, min_dist = setup
        opts = eik.PositionStepOptions()
        opts.stall_recovery = True
        opts.max_steps = 1

        stall_targets = [
            eik.TaskTarget("left_body", left_T),
            eik.TaskTarget("right_body", right_T),
        ]

        # Phase 1: drive into stall for 20 steps
        q, _, stall_counters = _run_position_step_loop(
            solver, robot, q0.copy(), stall_targets, opts, 20,
        )

        # The stall handler should have triggered (some steps should have
        # produced motion via the penetration escape).
        stall_motions = sum(1 for c in stall_counters if c == 0)
        assert stall_motions > 0, "Stall handler never triggered during stall phase"

        # Phase 2: reverse targets — move EEs outward (away from body)
        from utils.dual_iiwa_urdf import get_dual_iiwa_frame_names
        left_frame, right_frame = get_dual_iiwa_frame_names()
        left_pose = robot.get_frame_pose(left_frame)
        right_pose = robot.get_frame_pose(right_frame)

        away_left_T = np.eye(4)
        away_left_T[:3, :3] = np.array(left_pose.rotation)
        away_left_T[:3, 3] = np.array(left_pose.translation) + [0.0, 0.30, 0.15]

        away_right_T = np.eye(4)
        away_right_T[:3, :3] = np.array(right_pose.rotation)
        away_right_T[:3, 3] = np.array(right_pose.translation) + [0.0, -0.30, 0.15]

        away_targets = [
            eik.TaskTarget("left_body", away_left_T),
            eik.TaskTarget("right_body", away_right_T),
        ]

        # Phase 3: run with reversed targets and track per-step dq norms
        recovery_dq_norms = []
        for _ in range(10):
            result = solver.solve_position_step(q, away_targets, opts)
            dq_norm = np.linalg.norm(result.q_solution - q)
            q = result.q_solution
            robot.update_configuration(q)
            recovery_dq_norms.append(dq_norm)

        # Motion should resume within the first few recovery steps.
        # The penetration escape gives the collision constraint enough
        # slack for the solver to find feasible motion.
        early_motion = recovery_dq_norms[:5]
        assert any(dq > 1e-4 for dq in early_motion), (
            f"No meaningful motion in first 5 recovery steps: "
            f"dq norms = {[f'{d:.6f}' for d in early_motion]}"
        )

        # Majority of recovery steps should produce meaningful motion.
        motion_steps = sum(1 for dq in recovery_dq_norms if dq > 1e-4)
        assert motion_steps >= 5, (
            f"Only {motion_steps}/10 recovery steps had motion; "
            f"expected at least 5"
        )

        solver.disable_stall_handler()

    def test_velocity_loop_dual_ee_stall_recovery(self):
        """Full velocity loop: handler should reduce stalls vs baseline."""
        setup1 = _setup_dual_iiwa_body_stall()
        if setup1 is None:
            pytest.skip("Dual iiwa model not available")

        robot1, solver1, q01, _, _, _ = setup1
        baseline_stalls, _ = _run_velocity_loop(robot1, solver1, q01.copy(), 200)

        setup2 = _setup_dual_iiwa_body_stall()
        robot2, solver2, q02, _, _, min_dist2 = setup2
        solver2.enable_stall_handler(min_dist2)
        solver2.configure_stall_handler(stall_threshold=5)
        handler_stalls, _ = _run_velocity_loop(robot2, solver2, q02.copy(), 200)
        solver2.disable_stall_handler()

        assert handler_stalls <= baseline_stalls, (
            f"Handler should not increase stalls: {handler_stalls} > {baseline_stalls}"
        )
