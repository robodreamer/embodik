"""Tests for the C++ stall handler and PositionStepOptions.stall_recovery."""

import os
import re
import sys
import tempfile
import textwrap
from contextlib import contextmanager
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


def _disable_weighted_fallback(solver: eik.KinematicsSolver) -> None:
    cfg = eik.SolverRuntimeConfig()
    cfg.weighted_fallback_enabled = False
    solver.configure_runtime(cfg)


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
    _disable_weighted_fallback(solver)
    task = solver.add_frame_task("ee", _PANDA_EE_FRAME)
    task.weight = 1.0
    task.solve_mode = TaskSolveMode.SCALE
    return robot, solver, task


@contextmanager
def _two_link_collision_solver():
    urdf = textwrap.dedent("""
<robot name="two_link">
  <link name="base_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1.0"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/></inertial>
    <collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
  </link>
  <link name="link1">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.5"/>
    <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/></inertial>
    <collision><origin xyz="0.08 0 0" rpy="0 0 0"/><geometry><box size="0.08 0.08 0.08"/></geometry></collision>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0.05 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit effort="10.0" lower="-1.57" upper="1.57" velocity="1.0"/>
  </joint>
</robot>""")
    fd, path = tempfile.mkstemp(suffix=".urdf")
    with os.fdopen(fd, "w") as f:
        f.write(urdf)
    try:
        robot = eik.RobotModel(path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        _disable_weighted_fallback(solver)
        yield solver
    finally:
        os.unlink(path)


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
    return [(a, b) for a, b in robot.get_collision_pair_names() if _should_panda_exclude_pair(a, b)]


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
        same_arm = ("left" in a_l and "left" in b_l) or ("right" in a_l and "right" in b_l)
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
            get_dual_iiwa_default_configuration,
            get_dual_iiwa_frame_names,
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
    _disable_weighted_fallback(solver)

    q = np.array(get_dual_iiwa_default_configuration(), dtype=float)
    robot.update_configuration(q)

    dbg = solver.evaluate_collision_debug(q)
    if dbg is None or not np.isfinite(dbg.distance):
        return None

    # Keep this scenario in a true stall regime so repeated threshold crossings
    # are observable under strict assertions.
    min_dist = 0.35
    solver.configure_collision_constraint(
        min_distance=min_dist,
        include_pairs=[],
        exclude_pairs=list(excl),
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
            restore_rate=0.01,
            floor_fraction=0.5,
        )
        assert solver.stall_handler_enabled()
        assert solver.stall_handler_threshold() == 10
        assert solver.stall_handler_restore_rate() == pytest.approx(0.01)
        assert solver.stall_handler_floor_fraction() == pytest.approx(0.5)

    def test_collision_constraint_seeds_stall_defaults_without_enabling(self):
        with _two_link_collision_solver() as solver:
            solver.configure_collision_constraint(
                min_distance=0.04,
                include_pairs=[],
                exclude_pairs=[],
            )

            assert not solver.stall_handler_enabled()
            assert solver.stall_handler_threshold() == 3
            assert solver.stall_handler_restore_rate() == pytest.approx(0.2)
            assert solver.stall_handler_floor_fraction() == pytest.approx(0.0)

    def test_collision_constraint_preserves_explicit_stall_config(self):
        with _two_link_collision_solver() as solver:
            solver.enable_stall_handler(0.05)
            solver.configure_stall_handler(
                stall_threshold=9,
                restore_rate=0.03,
                floor_fraction=0.4,
            )
            solver.configure_collision_constraint(
                min_distance=0.04,
                include_pairs=[],
                exclude_pairs=[],
            )

            assert solver.stall_handler_threshold() == 9
            assert solver.stall_handler_restore_rate() == pytest.approx(0.03)
            assert solver.stall_handler_floor_fraction() == pytest.approx(0.4)

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

    def test_position_step_primary_options_match_position_ik_defaults(self):
        step = eik.PositionStepOptions()
        ik = eik.PositionIKOptions()
        assert step.primary_solve_mode == ik.primary_solve_mode
        assert step.primary_allow_min_error_fallback == ik.primary_allow_min_error_fallback

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

        assert (
            handler_stalls <= baseline_stalls
        ), f"Handler should not increase stalls: {handler_stalls} > {baseline_stalls}"


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

        assert (
            handler_stalls < baseline_stalls
        ), f"Handler should reduce stalls: {handler_stalls} >= {baseline_stalls}"


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
        assert solver.stall_handler_enabled(), "Should not disable externally-enabled handler"

        solver.disable_stall_handler()

    def test_stall_counter_accumulates_across_single_step_calls(self):
        """Regression: in a teleop-style loop (max_steps=1, one call per tick),
        the stall counter must accumulate across calls so recovery triggers."""
        setup = _setup_panda_stall()
        if setup is None:
            pytest.skip("Panda collision debug unavailable")

        robot, solver, q, task, target_pos, min_dist = setup
        _disable_weighted_fallback(solver)

        opts = eik.PositionStepOptions()
        opts.max_steps = 1
        opts.stall_recovery = True
        opts.position_gain = 20.0
        opts.orientation_gain = 20.0

        target = np.eye(4)
        target[:3, 3] = target_pos
        target[:3, :3] = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).rotation)

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
        assert solver.stall_handler_enabled(), "stall_recovery=True should enable the handler"
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
        assert (
            not solver.stall_handler_enabled()
        ), "Handler should be auto-disabled after solve_position"

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
        assert solver.stall_handler_enabled(), "Should not disable externally-enabled handler"
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
            get_dual_iiwa_default_configuration,
            get_dual_iiwa_frame_names,
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
    _disable_weighted_fallback(solver)

    q = np.array(get_dual_iiwa_default_configuration(), dtype=float)
    robot.update_configuration(q)

    dbg = solver.evaluate_collision_debug(q)
    if dbg is None or not np.isfinite(dbg.distance):
        return None

    # Keep this scenario in a true stall regime so repeated threshold crossings
    # are observable under strict assertions.
    min_dist = 0.35
    solver.configure_collision_constraint(
        min_distance=min_dist,
        include_pairs=[],
        exclude_pairs=list(excl),
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
    to match the dual-EE teleop pattern.
    """

    def test_stationary_merit_hold_does_not_relax_collision_margin(self):
        """A continuity hold must not spend collision-margin recovery budget."""
        setup = _setup_dual_iiwa_body_stall()
        if setup is None:
            pytest.skip("Dual iiwa model not available")

        robot, solver, q0, left_T, right_T, min_dist = setup
        solver.configure_stall_handler(stall_threshold=5)
        opts = eik.PositionStepOptions()
        opts.stall_recovery = True
        opts.max_steps = 1

        targets = [
            eik.TaskTarget("left_body", left_T),
            eik.TaskTarget("right_body", right_T),
        ]

        q, _, stall_counters = _run_position_step_loop(
            solver,
            robot,
            q0.copy(),
            targets,
            opts,
            200,
        )

        final_min = solver.stall_handler_current_min_distance()
        assert np.all(np.isfinite(q))
        assert max(stall_counters) == 0
        assert final_min == pytest.approx(min_dist)
        solver.disable_stall_handler()

    def test_stationary_merit_hold_does_not_count_as_collision_stall(self):
        """A deliberate continuity hold is not a collision-recovery stall."""
        setup = _setup_dual_iiwa_body_stall()
        if setup is None:
            pytest.skip("Dual iiwa model not available")

        robot, solver, q0, left_T, right_T, min_dist = setup
        solver.configure_stall_handler(stall_threshold=5)
        opts = eik.PositionStepOptions()
        opts.stall_recovery = True
        opts.max_steps = 1

        targets = [
            eik.TaskTarget("left_body", left_T),
            eik.TaskTarget("right_body", right_T),
        ]

        _, _, stall_counters = _run_position_step_loop(
            solver,
            robot,
            q0.copy(),
            targets,
            opts,
            200,
        )

        assert max(stall_counters) == 0
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
            solver,
            robot,
            q0.copy(),
            targets,
            opts,
            60,
        )

        if times_ms:
            max_time = max(times_ms)
            assert max_time < 10.0, f"Max per-step time {max_time:.2f}ms exceeds 10ms budget"
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
            solver,
            robot,
            q0.copy(),
            both_targets,
            opts,
            30,
        )

        # Disable left task (simulate user toggling off one EE)
        left_task = solver.get_task("left_body")
        left_task.weight = 0.0

        # Continue with only right EE target active
        right_only_targets = [eik.TaskTarget("right_body", right_T)]
        q, _, _ = _run_position_step_loop(
            solver,
            robot,
            q,
            right_only_targets,
            opts,
            10,
        )

        dbg = solver.evaluate_collision_debug(q)
        if dbg is not None and np.isfinite(dbg.distance):
            assert dbg.distance > -0.02, (
                f"Collision distance {dbg.distance:.4f}m after disabling task; "
                f"arm jumped into deep penetration"
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

        assert (
            handler_stalls <= baseline_stalls
        ), f"Handler should not increase stalls: {handler_stalls} > {baseline_stalls}"


# ===================================================================
# Joint-limit stall: collision must not prevent pull-away recovery
# ===================================================================


class TestJointLimitCollisionInteraction:
    """Verify that collision constraints do not prevent recovery from
    a joint-limit-induced stall when the gizmo target reverses direction.

    Root cause scenario (panda fixture):
    1. Target drives EE inward until a joint (e.g. joint 3) saturates at
       its lower limit -> solver returns INFEASIBLE with dq=0.
    2. User reverses gizmo to pull the EE *away* from the torso.
    3. The solver must produce SUCCESS with meaningful motion on the very
       first step, regardless of collision tuning mode or proximity gating.
    """

    @staticmethod
    def _drive_to_joint_limit_stall(mode):
        """Drive panda into a joint-limit stall and return frozen state."""
        setup = _setup_panda_stall()
        if setup is None:
            return None
        robot, solver, q, task, target_pos, min_dist = setup
        solver.set_collision_tuning_mode(mode)
        opts = eik.PositionStepOptions()
        opts.max_steps = 1
        opts.position_gain = 20.0
        opts.orientation_gain = 20.0
        rot = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).rotation)
        into = np.eye(4)
        into[:3, :3] = rot
        into[:3, 3] = target_pos
        for _ in range(30):
            result = solver.solve_position_step(q, into, "panda_stall", opts)
            q = np.asarray(result.q_solution, dtype=float)
            robot.update_configuration(q)
        return robot, solver, q, opts, rot

    def test_pull_away_recovers_immediately_all_modes(self):
        """After joint-limit stall, a target in the +X direction must
        produce SUCCESS on the very first step for all tuning modes."""
        for mode in (
            eik.CollisionTuningMode.PRECISE,
            eik.CollisionTuningMode.BALANCED,
            eik.CollisionTuningMode.SPEED,
        ):
            state = self._drive_to_joint_limit_stall(mode)
            if state is None:
                pytest.skip("Panda collision model not available")
            robot, solver, q, opts, rot = state
            ee = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).translation)
            away = np.eye(4)
            away[:3, :3] = rot
            away[:3, 3] = ee + np.array([0.15, 0.10, 0.10])
            result = solver.solve_position_step(q, away, "panda_stall", opts)
            dq = float(np.linalg.norm(np.asarray(result.q_solution) - q))
            assert (
                result.status == eik.SolverStatus.SUCCESS
            ), f"Mode {mode.name}: expected SUCCESS on pull-away, got {result.status.name}"
            assert (
                dq > 1e-4
            ), f"Mode {mode.name}: expected meaningful motion on pull-away, got dq={dq}"

    def test_collision_does_not_worsen_infeasibility(self):
        """With collision ON vs OFF at the joint-limit stall config,
        the set of feasible target directions must be identical."""
        setup = _setup_panda_stall()
        if setup is None:
            pytest.skip("Panda collision model not available")
        robot, solver, q, task, target_pos, min_dist = setup
        solver.set_collision_tuning_mode(eik.CollisionTuningMode.SPEED)
        opts = eik.PositionStepOptions()
        opts.max_steps = 1
        opts.position_gain = 20.0
        opts.orientation_gain = 20.0
        rot = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).rotation)
        into = np.eye(4)
        into[:3, :3] = rot
        into[:3, 3] = target_pos
        for _ in range(30):
            result = solver.solve_position_step(q, into, "panda_stall", opts)
            q = np.asarray(result.q_solution, dtype=float)
            robot.update_configuration(q)

        ee = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).translation)
        excl = _panda_collision_exclusions(robot)
        offsets = [
            np.array([-0.005, 0.0, 0.0]),
            np.array([0.005, 0.0, 0.0]),
            np.array([0.0, 0.01, 0.0]),
            np.array([0.0, 0.0, 0.01]),
            np.array([0.0, 0.0, -0.01]),
            np.array([0.10, 0.10, 0.10]),
        ]
        for off in offsets:
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = ee + off

            q_with = q.copy()
            robot.update_configuration(q_with)
            r_with = solver.solve_position_step(q_with, T, "panda_stall", opts)

            solver.clear_collision_constraint()
            q_without = q.copy()
            robot.update_configuration(q_without)
            r_without = solver.solve_position_step(q_without, T, "panda_stall", opts)

            solver.configure_collision_constraint(
                min_distance=min_dist,
                include_pairs=[],
                exclude_pairs=list(excl),
            )

            assert r_with.status == r_without.status, (
                f"offset={off}: collision ON -> {r_with.status.name} "
                f"but collision OFF -> {r_without.status.name}"
            )

    def test_sideways_motion_at_joint_limit(self):
        """At joint-limit stall, lateral targets (+Y, +Z) must produce
        motion even with collision active."""
        state = self._drive_to_joint_limit_stall(eik.CollisionTuningMode.SPEED)
        if state is None:
            pytest.skip("Panda collision model not available")
        robot, solver, q, opts, rot = state
        ee = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).translation)
        for label, offset in [
            ("+Y", np.array([0.0, 0.10, 0.0])),
            ("+Z", np.array([0.0, 0.0, 0.10])),
            ("+Y+Z", np.array([0.0, 0.07, 0.07])),
        ]:
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = ee + offset
            result = solver.solve_position_step(q, T, "panda_stall", opts)
            dq = float(np.linalg.norm(np.asarray(result.q_solution) - q))
            assert (
                result.status == eik.SolverStatus.SUCCESS
            ), f"{label}: expected SUCCESS, got {result.status.name}"
            assert dq > 1e-4, f"{label}: expected motion, got dq={dq}"

    def test_no_all_direction_trap(self):
        """At the joint-limit stall config, at least some directions must
        be feasible -- the solver must not create an all-direction trap
        where no motion is possible."""
        state = self._drive_to_joint_limit_stall(eik.CollisionTuningMode.SPEED)
        if state is None:
            pytest.skip("Panda collision model not available")
        robot, solver, q, opts, rot = state
        ee = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).translation)
        feasible_count = 0
        for dx in [-0.05, 0.0, 0.05]:
            for dy in [-0.05, 0.0, 0.05]:
                for dz in [-0.05, 0.0, 0.05]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    T = np.eye(4)
                    T[:3, :3] = rot
                    T[:3, 3] = ee + np.array([dx, dy, dz])
                    robot.update_configuration(q.copy())
                    result = solver.solve_position_step(
                        q,
                        T,
                        "panda_stall",
                        opts,
                    )
                    dq = float(np.linalg.norm(np.asarray(result.q_solution) - q))
                    if result.status == eik.SolverStatus.SUCCESS and dq > 1e-5:
                        feasible_count += 1
        assert feasible_count >= 10, (
            f"Only {feasible_count}/26 directions feasible at joint-limit " f"stall; arm is trapped"
        )


# ===================================================================
# Two-joint-limit trap: Jacobian clamping prevents SNS task-scaling
# ===================================================================


class TestTwoJointLimitTrap:
    """Verify that when two joints are simultaneously near their position
    limits, the solver does not become effectively frozen for all
    task directions.

    Root cause scenario (alpha robot):
      right_elbow_pitch at q=-0.326 (lower limit -0.3491, margin 23 mrad)
      right_wrist_roll  at q=-0.436 (lower limit -0.4363, margin 0.3 mrad)
    The wrist roll's velocity-box bound is ~0.02 rad/s, causing the SNS
    solver to scale the entire 6D task to nearly zero even for directions
    that don't kinematically require the saturated joint.

    Fix: Jacobian column clamping zeros out task Jacobian entries that
    would command velocity toward a saturated joint limit, letting the
    SNS solver use the remaining DOFs for partial solutions.
    """

    @staticmethod
    def _setup_two_joint_limit():
        """Panda with joint 4 and joint 6 near their lower limits,
        mimicking the alpha arm's elbow+wrist trap."""
        from robot_descriptions.panda_description import URDF_PATH

        urdf_path = Path(URDF_PATH)
        _ensure_ros_package_path(urdf_path)

        robot = eik.RobotModel(str(urdf_path), floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01
        _disable_weighted_fallback(solver)

        q_min, q_max = robot.get_joint_limits()
        q = np.array([0.0, -0.785, 0.0, -3.05, 0.0, -0.0172, 0.785, 0.04, 0.04])
        robot.update_configuration(q)

        solver.clear_tasks()
        task = solver.add_frame_task("trap_test", _PANDA_EE_FRAME)
        task.priority = 0
        task.weight = 1.0
        task.solve_mode = eik.TaskSolveMode.SCALE

        return robot, solver, q, task

    def test_meaningful_motion_near_two_limits(self):
        """With two joints near their lower limits, all 26 sampled
        directions must produce meaningful motion (dq > 1e-3)."""
        robot, solver, q, task = self._setup_two_joint_limit()
        ee = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).translation)
        rot = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).rotation)

        opts = eik.PositionStepOptions()
        opts.max_steps = 1
        opts.position_gain = 20.0
        opts.orientation_gain = 20.0

        q_min, _ = robot.get_joint_limits()
        assert q[3] - q_min[3] < 0.025, "j4 should be near lower limit"
        assert q[5] - q_min[5] < 0.001, "j6 should be near lower limit"

        feasible = 0
        for dx in [-0.05, 0.0, 0.05]:
            for dy in [-0.05, 0.0, 0.05]:
                for dz in [-0.05, 0.0, 0.05]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    T = np.eye(4)
                    T[:3, :3] = rot
                    T[:3, 3] = ee + np.array([dx, dy, dz])
                    robot.update_configuration(q.copy())
                    result = solver.solve_position_step(
                        q.copy(),
                        T,
                        "trap_test",
                        opts,
                    )
                    dq = float(np.linalg.norm(np.asarray(result.q_solution) - q))
                    if result.status == eik.SolverStatus.SUCCESS and dq > 1e-3:
                        feasible += 1
        assert feasible >= 20, (
            f"Only {feasible}/26 directions produced meaningful motion "
            f"with two joints near limits (expected >= 20)"
        )

    def test_trap_direction_not_frozen(self):
        """The specific [-X, +Z] direction that triggered the trap must
        produce meaningful motion, not the near-zero dq that the
        pre-fix solver returned."""
        robot, solver, q, task = self._setup_two_joint_limit()
        ee = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).translation)
        rot = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).rotation)

        opts = eik.PositionStepOptions()
        opts.max_steps = 1
        opts.position_gain = 20.0
        opts.orientation_gain = 20.0

        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = ee + np.array([-0.15, 0.0, 0.10])
        result = solver.solve_position_step(q.copy(), T, "trap_test", opts)
        dq = float(np.linalg.norm(np.asarray(result.q_solution) - q))
        assert (
            result.status == eik.SolverStatus.SUCCESS
        ), f"Expected SUCCESS for [-X,+Z] direction, got {result.status.name}"
        assert dq > 1e-2, (
            f"Expected meaningful motion (dq > 0.01) for [-X,+Z], got dq={dq:.4e}. "
            f"Joint-limit Jacobian clamping may not be active."
        )

    def test_drive_into_limits_then_recover(self):
        """Drive the arm into joint limits, then verify recovery when
        the target reverses to a feasible direction."""
        from robot_descriptions.panda_description import URDF_PATH

        urdf_path = Path(URDF_PATH)
        _ensure_ros_package_path(urdf_path)

        robot = eik.RobotModel(str(urdf_path), floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01
        _disable_weighted_fallback(solver)

        q = np.array([0.0, -0.785, 0.0, -2.8, 0.0, 0.1, 0.785, 0.04, 0.04])
        robot.update_configuration(q)

        solver.clear_tasks()
        task = solver.add_frame_task("recover", _PANDA_EE_FRAME)
        task.priority = 0
        task.weight = 1.0
        task.solve_mode = eik.TaskSolveMode.SCALE

        ee_pos = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).translation)
        ee_rot = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).rotation)
        target_pos = ee_pos + np.array([-0.30, 0.0, 0.15])

        opts = eik.PositionStepOptions()
        opts.max_steps = 1
        opts.position_gain = 20.0
        opts.orientation_gain = 20.0

        for _ in range(40):
            T = np.eye(4)
            T[:3, :3] = ee_rot
            T[:3, 3] = target_pos
            robot.update_configuration(q.copy())
            result = solver.solve_position_step(q, T, "recover", opts)
            if result.status == eik.SolverStatus.SUCCESS:
                q = np.asarray(result.q_solution, dtype=float)

        q_min, _ = robot.get_joint_limits()
        assert q[5] - q_min[5] < 0.002, "j6 should be near its lower limit after driving inward"

        ee_now = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).translation)
        pull_target = ee_now + np.array([0.05, 0.0, -0.05])
        T_pull = np.eye(4)
        T_pull[:3, :3] = ee_rot
        T_pull[:3, 3] = pull_target
        robot.update_configuration(q.copy())
        result = solver.solve_position_step(q, T_pull, "recover", opts)
        dq = float(np.linalg.norm(np.asarray(result.q_solution) - q))
        assert (
            result.status == eik.SolverStatus.SUCCESS
        ), f"Pull-away after limit stall: expected SUCCESS, got {result.status.name}"
        assert dq > 5e-3, f"Pull-away should produce meaningful motion, got dq={dq:.4e}"


# ===================================================================
# Regression: clamping-induced stall must not relax collision margins
# ===================================================================


class TestClampingDoesNotTriggerStallRelaxation:
    """When Jacobian clamping reduces task motion near joint limits,
    the stall handler must NOT misinterpret this as a collision-caused
    stall and relax ``min_distance``.

    Root cause scenario observed on Alpha robot:
      1. Arm approaches torso → joints near limits get Jacobian columns
         clamped to zero.
      2. Clamped Jacobian produces INFEASIBLE or SUCCESS with tiny ||dq||.
      3. Stall handler sees status != SUCCESS with ||dq|| < dq_stall_eps
         → increments stall counter.
      4. After stall_threshold steps, handler relaxes min_distance.
      5. With relaxed collision margin, the task drives the arm *into*
         the torso — the opposite of intended behavior.
    """

    def test_stall_near_joint_limits_does_not_relax_collision_margin(self):
        """Drive EE toward body with stall_recovery ON and a high enough
        ``min_distance`` that the stall handler fires before joint limits
        are reached. Once joints clamp, the handler must stop relaxing
        collision margin further (it is no longer the bottleneck).

        With ``min_dist=0.08`` the stall-relax-advance cycle runs until
        joint 3 hits its lower limit. After that, further relaxation is
        wasted — the bottleneck is joint limits, not collision. The
        handler must not keep ratcheting ``min_distance`` down once
        collision distance stops decreasing while a joint is clamped.
        """
        setup = _setup_panda_stall()
        if setup is None:
            pytest.skip("Panda collision model not available")

        robot, solver, q, task, target_pos, min_dist_orig = setup

        min_dist = 0.08
        solver.clear_collision_constraint()
        excl = _panda_collision_exclusions(robot)
        solver.configure_collision_constraint(
            min_distance=min_dist,
            include_pairs=[],
            exclude_pairs=list(excl),
        )
        solver.enable_stall_handler(min_dist)
        solver.configure_stall_handler(stall_threshold=5)

        opts = eik.PositionStepOptions()
        opts.stall_recovery = True
        opts.max_steps = 1
        opts.position_gain = 20.0
        opts.orientation_gain = 20.0

        rot = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).rotation)
        into_T = np.eye(4)
        into_T[:3, :3] = rot
        into_T[:3, 3] = target_pos

        collision_distances = []
        stall_min_distances = []
        for _ in range(100):
            result = solver.solve_position_step(q, into_T, "panda_stall", opts)
            q = np.asarray(result.q_solution, dtype=float)
            robot.update_configuration(q)
            dbg = solver.evaluate_collision_debug(q)
            d = float(dbg.distance) if dbg is not None else float("inf")
            collision_distances.append(d)
            stall_min_distances.append(solver.stall_handler_current_min_distance())

        worst_collision = min(collision_distances)
        final_margin = solver.stall_handler_current_min_distance()

        assert worst_collision > -0.01, (
            f"Arm pulled into deep collision (d={worst_collision:.4f}m). "
            f"Stall handler relaxed min_distance from {min_dist} to "
            f"{min(stall_min_distances):.4f}; this is a regression."
        )

        floor_min = min_dist * 0.3
        assert final_margin >= floor_min - 1e-6, (
            f"Stall handler dropped margin below floor " f"({final_margin:.4f} < {floor_min:.4f})"
        )
        solver.disable_stall_handler()

    def test_stall_handler_stops_relaxing_when_joint_limit_is_bottleneck(self):
        """Once a joint is clamped near its limit and collision distance
        stops decreasing, the stall handler must not keep relaxing
        ``min_distance`` on every stall-threshold crossing.

        Specifically: after the arm reaches the joint limit, the margin
        should stabilize (not keep dropping by relax_drop_fraction each
        cycle). We allow a few initial drops during the transition, but
        the total relaxation after joint clamping begins must be bounded.
        """
        setup = _setup_panda_stall()
        if setup is None:
            pytest.skip("Panda collision model not available")

        robot, solver, q, task, target_pos, min_dist_orig = setup

        min_dist = 0.08
        solver.clear_collision_constraint()
        excl = _panda_collision_exclusions(robot)
        solver.configure_collision_constraint(
            min_distance=min_dist,
            include_pairs=[],
            exclude_pairs=list(excl),
        )
        solver.enable_stall_handler(min_dist)
        solver.configure_stall_handler(stall_threshold=5)

        opts = eik.PositionStepOptions()
        opts.stall_recovery = True
        opts.max_steps = 1
        opts.position_gain = 20.0
        opts.orientation_gain = 20.0

        rot = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).rotation)
        into_T = np.eye(4)
        into_T[:3, :3] = rot
        into_T[:3, 3] = target_pos

        q_min, _ = robot.get_joint_limits()
        clamp_margin = 5e-4

        margin_at_first_clamp = None
        margin_after_clamp = None
        for step in range(100):
            near_limit = any(0 < q[i] - q_min[i] < clamp_margin for i in range(min(7, q.size)))

            result = solver.solve_position_step(q, into_T, "panda_stall", opts)
            q = np.asarray(result.q_solution, dtype=float)
            robot.update_configuration(q)

            cur_margin = solver.stall_handler_current_min_distance()
            if near_limit and margin_at_first_clamp is None:
                margin_at_first_clamp = cur_margin
            margin_after_clamp = cur_margin

        if margin_at_first_clamp is not None:
            drop_after_clamp = margin_at_first_clamp - margin_after_clamp
            max_acceptable_drop = 2 * 0.10 * min_dist
            assert drop_after_clamp <= max_acceptable_drop + 1e-6, (
                f"Stall handler kept relaxing after joint limit became "
                f"bottleneck: dropped from {margin_at_first_clamp:.4f} to "
                f"{margin_after_clamp:.4f} "
                f"(delta={drop_after_clamp:.4f} > {max_acceptable_drop:.4f})"
            )
        solver.disable_stall_handler()

    def test_collision_distance_monotonic_near_limits_with_stall_recovery(self):
        """When driving toward the torso, collision distance should not
        suddenly drop (indicating penetration from stall relaxation).
        Small decreases per step are OK; large jumps (> 0.02m) are not."""
        setup = _setup_panda_stall()
        if setup is None:
            pytest.skip("Panda collision model not available")

        robot, solver, q, task, target_pos, min_dist = setup
        solver.enable_stall_handler(min_dist)

        opts = eik.PositionStepOptions()
        opts.stall_recovery = True
        opts.max_steps = 1
        opts.position_gain = 20.0
        opts.orientation_gain = 20.0

        rot = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).rotation)
        into_T = np.eye(4)
        into_T[:3, :3] = rot
        into_T[:3, 3] = target_pos

        prev_d = float(solver.evaluate_collision_debug(q).distance)
        max_drop = 0.0
        for step in range(60):
            result = solver.solve_position_step(q, into_T, "panda_stall", opts)
            q = np.asarray(result.q_solution, dtype=float)
            robot.update_configuration(q)
            dbg = solver.evaluate_collision_debug(q)
            d = float(dbg.distance) if dbg is not None else prev_d
            drop = prev_d - d
            if drop > max_drop:
                max_drop = drop
            prev_d = d

        assert max_drop < 0.02, (
            f"Collision distance dropped by {max_drop:.4f}m in a single "
            f"step; stall handler likely relaxed collision margin "
            f"inappropriately near joint limits."
        )
        solver.disable_stall_handler()


class TestStallHandlerMultiConstraintRegression:
    """Regression tests for K>1 collision rows + stall handling.

    These tests mirror the alpha-wheelbase failure mode at a smaller scale:
    stall recovery plus multiple collision rows should not pull the arm into
    deeper penetration when joint limits dominate.
    """

    @staticmethod
    def _run_trial(*, max_constraints: int, stall_recovery: bool, steps: int = 80):
        setup = _setup_panda_stall()
        if setup is None:
            pytest.skip("Panda collision model not available")

        robot, solver, q, _task, target_pos, _ = setup
        min_dist = 0.08
        solver.clear_collision_constraint()
        excl = _panda_collision_exclusions(robot)
        solver.configure_collision_constraint(
            min_distance=min_dist,
            include_pairs=[],
            exclude_pairs=list(excl),
            max_constraints=max_constraints,
        )
        if stall_recovery:
            solver.enable_stall_handler(min_dist)
            solver.configure_stall_handler(stall_threshold=5)
        else:
            solver.disable_stall_handler()

        opts = eik.PositionStepOptions()
        opts.stall_recovery = stall_recovery
        opts.max_steps = 1
        opts.position_gain = 20.0
        opts.orientation_gain = 20.0

        rot = np.array(robot.get_frame_pose(_PANDA_EE_FRAME).rotation)
        into_T = np.eye(4)
        into_T[:3, :3] = rot
        into_T[:3, 3] = target_pos

        worst_collision = float("inf")
        min_stall_margin = float("inf")
        success_count = 0
        infeasible_count = 0
        max_active_rows = 0
        for _ in range(steps):
            result = solver.solve_position_step(q, into_T, "panda_stall", opts)
            status_str = str(result.status)
            if "SUCCESS" in status_str:
                q = np.asarray(result.q_solution, dtype=float)
                success_count += 1
            elif "INFEASIBLE" in status_str:
                infeasible_count += 1

            robot.update_configuration(q)
            dbg = solver.evaluate_collision_debug(q)
            d = float(dbg.distance) if dbg is not None else float("inf")
            worst_collision = min(worst_collision, d)

            rows = solver.get_last_collision_debug_list()
            max_active_rows = max(max_active_rows, len(rows))
            if stall_recovery:
                min_stall_margin = min(
                    min_stall_margin, float(solver.stall_handler_current_min_distance())
                )

        return {
            "worst_collision": worst_collision,
            "min_stall_margin": min_stall_margin if stall_recovery else None,
            "success_count": success_count,
            "infeasible_count": infeasible_count,
            "max_active_rows": max_active_rows,
        }

    def test_multi_constraint_rows_are_active_with_k2(self):
        m2 = self._run_trial(max_constraints=2, stall_recovery=False, steps=25)
        m1 = self._run_trial(max_constraints=1, stall_recovery=False, steps=25)
        assert (
            m2["max_active_rows"] >= 2
        ), "Expected at least two active collision rows with max_constraints=2"
        assert (
            m1["max_active_rows"] <= 1
        ), "Expected at most one active collision row with max_constraints=1"

    def test_stall_recovery_k2_does_not_worsen_penetration_vs_off(self):
        with_stall = self._run_trial(max_constraints=2, stall_recovery=True, steps=80)
        without_stall = self._run_trial(max_constraints=2, stall_recovery=False, steps=80)

        # Regression guard: enabling stall recovery should not pull into deeper
        # collision than leaving it disabled in this near-limit scenario.
        assert with_stall["worst_collision"] >= without_stall["worst_collision"] - 1e-3, (
            "Stall recovery worsened penetration under K=2:\n"
            f"with_stall={with_stall['worst_collision']:.4f}, "
            f"without_stall={without_stall['worst_collision']:.4f}"
        )

        # Guard against unconditional penetration escape ratcheting margin below
        # zero when the configuration remains limit-dominated.
        assert with_stall["min_stall_margin"] is not None
        assert with_stall["min_stall_margin"] >= -1e-9, (
            "Stall handler dropped collision margin below zero in a limit-dominated "
            f"K=2 run (min_margin={with_stall['min_stall_margin']:.4f})"
        )
