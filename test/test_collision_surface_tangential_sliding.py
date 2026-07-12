"""Geometric regression coverage for sliding along active collision surfaces."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

import numpy as np
import pytest

import embodik as eik

_CONTACT_SEED = np.array([1.72537396, -1.69513454, 0.28872274, 0.0], dtype=float)
_COMMAND_EPOCH_CONTACT_SEED = np.array(
    [0.7848904730053934, 1.4182186491678923, -0.8543876987413632, 0.0],
    dtype=float,
)


@dataclass(frozen=True)
class SlidingTrace:
    desired_normal_motion: float
    desired_tangent_motion: float
    entry_clearance: float
    min_clearance: float
    final_clearance: float
    tangent_progress: float
    achieved_normal_motion: float
    productive_steps: int
    max_zero_motion_streak: int
    tangent_sign_flips: int
    clearance_sign_flips: int
    away_error_reduction: float
    away_normal_progress: float
    away_min_clearance_delta: float
    away_recovery_lag_steps: int
    away_max_zero_motion_streak: int
    max_step_norm: float
    max_acceleration_norm: float
    max_jerk_norm: float


def _write_contact_arm_urdf(tmp_path):
    """Create a redundant planar arm with an upstream collision contact."""
    urdf = textwrap.dedent("""
        <robot name="contact_arm">
          <link name="base">
            <collision><geometry><sphere radius="0.07"/></geometry></collision>
          </link>
          <link name="link1"/>
          <link name="contact_link">
            <collision>
              <origin xyz="0.075 0 0"/>
              <geometry><sphere radius="0.04"/></geometry>
            </collision>
          </link>
          <link name="link3"/>
          <link name="tool"/>
          <joint name="joint1" type="revolute">
            <parent link="base"/><child link="link1"/>
            <axis xyz="0 0 1"/>
            <limit lower="-3.14" upper="3.14" effort="10" velocity="2"/>
          </joint>
          <joint name="joint2" type="revolute">
            <parent link="link1"/><child link="contact_link"/>
            <origin xyz="0.15 0 0"/><axis xyz="0 0 1"/>
            <limit lower="-3.14" upper="3.14" effort="10" velocity="2"/>
          </joint>
          <joint name="joint3" type="revolute">
            <parent link="contact_link"/><child link="link3"/>
            <origin xyz="0.15 0 0"/><axis xyz="0 0 1"/>
            <limit lower="-3.14" upper="3.14" effort="10" velocity="2"/>
          </joint>
          <joint name="joint4" type="prismatic">
            <parent link="link3"/><child link="tool"/>
            <origin xyz="0.15 0 0"/><axis xyz="0 0 1"/>
            <limit lower="-0.30" upper="0.30" effort="10" velocity="2"/>
          </joint>
        </robot>
        """).strip()
    path = tmp_path / "contact_arm.urdf"
    path.write_text(urdf)
    return path


def _write_axis_aligned_contact_urdf(tmp_path):
    """Create an XY stage whose collision normal is exactly one task axis."""
    urdf = textwrap.dedent("""
        <robot name="axis_aligned_contact">
          <link name="base">
            <collision><geometry><sphere radius="0.05"/></geometry></collision>
          </link>
          <link name="x_stage"/>
          <link name="tool">
            <collision><geometry><sphere radius="0.05"/></geometry></collision>
          </link>
          <joint name="slide_x" type="prismatic">
            <parent link="base"/><child link="x_stage"/>
            <origin xyz="0.12 0 0"/><axis xyz="1 0 0"/>
            <limit lower="-0.05" upper="0.20" effort="10" velocity="2"/>
          </joint>
          <joint name="slide_y" type="prismatic">
            <parent link="x_stage"/><child link="tool"/>
            <axis xyz="0 1 0"/>
            <limit lower="-0.20" upper="0.20" effort="10" velocity="2"/>
          </joint>
        </robot>
        """).strip()
    path = tmp_path / "axis_aligned_contact.urdf"
    path.write_text(urdf)
    return path


def _disable_weighted_fallback(solver: eik.KinematicsSolver) -> None:
    config = solver.runtime_config()
    config.weighted_fallback_enabled = False
    solver.configure_runtime(config)


def _max_true_streak(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _count_sign_flips(deltas: np.ndarray, deadband: float) -> int:
    previous = 0
    flips = 0
    for delta in deltas:
        sign = 0 if abs(float(delta)) <= deadband else int(np.sign(delta))
        if sign == 0:
            continue
        if previous != 0 and sign != previous:
            flips += 1
        previous = sign
    return flips


def _run_sliding_trace(tmp_path, margin_offset: float) -> SlidingTrace:
    robot = eik.RobotModel(str(_write_contact_arm_urdf(tmp_path)), floating_base=False)
    q = _CONTACT_SEED.copy()
    robot.update_configuration(q)

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    _disable_weighted_fallback(solver)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    initial_debug = solver.evaluate_collision_debug(q)
    assert initial_debug is not None
    nearest_delta = np.asarray(initial_debug.point_b_world, dtype=float) - np.asarray(
        initial_debug.point_a_world, dtype=float
    )
    active_normal = nearest_delta / np.linalg.norm(nearest_delta)
    tangent = np.array([-active_normal[1], active_normal[0], 0.0], dtype=float)
    tangent /= np.linalg.norm(tangent)

    task = solver.add_frame_task("tool_position", "tool", eik.TaskType.FRAME_POSITION)
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = eik.TaskSolveMode.SCALE
    task.allow_min_error_fallback = False

    vertical_hold = solver.add_joint_task("vertical_hold", "joint4", target_value=float(q[3]))
    vertical_hold.priority = 1
    vertical_hold.weight = 1.0
    vertical_hold.solve_mode = eik.TaskSolveMode.MIN_ERROR
    vertical_hold.set_excluded_joint_indices([0, 1, 2])

    entry_position = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)
    target_position = entry_position - 0.03 * active_normal + 0.08 * tangent
    task.set_target_position(target_position)

    entry_clearance = float(initial_debug.distance)
    solver.configure_collision_constraint(
        min_distance=entry_clearance + margin_offset,
        include_pairs=list(robot.get_collision_pair_names()),
        max_constraints=1,
    )
    solver.set_proximity_gated_collision_activation_enabled(False)

    clearances: list[float] = []
    tangent_progress: list[float] = []
    joint_steps: list[np.ndarray] = []
    zero_motion: list[bool] = []
    for _ in range(24):
        result = solver.solve_velocity(q, apply_limits=True)
        assert result.weighted_fallback_used is False
        assert result.task_modes_effective[0] == eik.TaskSolveMode.SCALE
        velocity = np.asarray(result.solution, dtype=float)
        q_next = q + solver.dt * velocity
        joint_steps.append(q_next - q)
        q = q_next
        robot.update_configuration(q)

        position = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)
        progress = float(np.dot(position - entry_position, tangent))
        tangent_progress.append(progress)
        clearance = solver.evaluate_min_collision_distance(q)
        assert clearance is not None
        clearances.append(float(clearance))
        remaining_error = float(np.linalg.norm(target_position - position))
        zero_motion.append(np.linalg.norm(joint_steps[-1]) < 1e-6 and remaining_error > 0.02)

    mixed_end_position = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)
    mixed_end_clearance = clearances[-1]
    away_target = mixed_end_position + 0.03 * active_normal + 0.02 * tangent
    away_entry_error = float(np.linalg.norm(away_target - mixed_end_position))
    task.set_target_position(away_target)
    away_errors: list[float] = []
    away_normal_progress: list[float] = []
    away_clearances: list[float] = []
    away_zero_motion: list[bool] = []
    for _ in range(8):
        result = solver.solve_velocity(q, apply_limits=True)
        assert result.weighted_fallback_used is False
        assert result.task_modes_effective[0] == eik.TaskSolveMode.SCALE
        velocity = np.asarray(result.solution, dtype=float)
        q_next = q + solver.dt * velocity
        joint_steps.append(q_next - q)
        q = q_next
        robot.update_configuration(q)

        position = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)
        error = float(np.linalg.norm(away_target - position))
        away_errors.append(error)
        away_normal_progress.append(float(np.dot(position - mixed_end_position, active_normal)))
        clearance = solver.evaluate_min_collision_distance(q)
        assert clearance is not None
        away_clearances.append(float(clearance))
        away_zero_motion.append(np.linalg.norm(joint_steps[-1]) < 1e-6 and error > 0.02)

    step_array = np.asarray(joint_steps, dtype=float)
    acceleration = np.diff(step_array, axis=0)
    jerk = np.diff(acceleration, axis=0)
    progress_deltas = np.diff(np.concatenate(([0.0], tangent_progress)))
    clearance_deltas = np.diff(
        np.concatenate(([entry_clearance], np.asarray(clearances, dtype=float)))
    )
    away_recovery_lag = len(away_errors) + 1
    for index, (error, step) in enumerate(zip(away_errors, joint_steps[-8:])):
        if error < away_entry_error - 1e-5 and np.linalg.norm(step) > 1e-6:
            away_recovery_lag = index + 1
            break
    return SlidingTrace(
        desired_normal_motion=float(np.dot(target_position - entry_position, active_normal)),
        desired_tangent_motion=float(np.dot(target_position - entry_position, tangent)),
        entry_clearance=entry_clearance,
        min_clearance=min(clearances),
        final_clearance=clearances[-1],
        tangent_progress=tangent_progress[-1],
        achieved_normal_motion=clearances[-1] - entry_clearance,
        productive_steps=int(np.count_nonzero(progress_deltas > 1e-5)),
        max_zero_motion_streak=_max_true_streak(zero_motion),
        tangent_sign_flips=_count_sign_flips(progress_deltas, 1e-6),
        clearance_sign_flips=_count_sign_flips(clearance_deltas, 1e-7),
        away_error_reduction=away_entry_error - away_errors[-1],
        away_normal_progress=away_normal_progress[-1],
        away_min_clearance_delta=min(away_clearances) - mixed_end_clearance,
        away_recovery_lag_steps=away_recovery_lag,
        away_max_zero_motion_streak=_max_true_streak(away_zero_motion),
        max_step_norm=float(np.max(np.linalg.norm(step_array, axis=1))),
        max_acceleration_norm=float(np.max(np.linalg.norm(acceleration, axis=1))),
        max_jerk_norm=float(np.max(np.linalg.norm(jerk, axis=1))),
    )


@pytest.mark.parametrize("margin_offset", [0.0, 0.0002], ids=["boundary", "violated"])
def test_collision_surface_preserves_tangent_progress_and_clearance(tmp_path, margin_offset):
    trace = _run_sliding_trace(tmp_path, margin_offset)

    assert trace.desired_normal_motion <= -0.029
    assert trace.desired_tangent_motion >= 0.079
    assert trace.min_clearance >= trace.entry_clearance - 1e-6
    assert trace.final_clearance >= trace.entry_clearance - 1e-6
    assert trace.achieved_normal_motion >= -1e-6
    assert trace.tangent_progress >= 0.02
    assert trace.productive_steps >= 12
    assert trace.max_zero_motion_streak <= 2
    assert trace.tangent_sign_flips <= 1
    assert trace.clearance_sign_flips <= 2
    assert trace.away_error_reduction >= 0.003
    assert trace.away_normal_progress >= 0.002
    assert trace.away_min_clearance_delta >= -1e-6
    assert trace.away_recovery_lag_steps <= 2
    assert trace.away_max_zero_motion_streak <= 2
    assert trace.max_step_norm <= 0.08
    assert trace.max_acceleration_norm <= 0.16
    assert trace.max_jerk_norm <= 0.24


def test_axis_aligned_violated_margin_slides_without_weighted_fallback(tmp_path):
    robot = eik.RobotModel(str(_write_axis_aligned_contact_urdf(tmp_path)), floating_base=False)
    q = np.array([0.02, 0.0], dtype=float)
    robot.update_configuration(q)

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    _disable_weighted_fallback(solver)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    debug = solver.evaluate_collision_debug(q)
    assert debug is not None
    nearest_delta = np.asarray(debug.point_b_world) - np.asarray(debug.point_a_world)
    normal = nearest_delta / np.linalg.norm(nearest_delta)
    tangent = np.array([-normal[1], normal[0], 0.0], dtype=float)
    tangent /= np.linalg.norm(tangent)

    task = solver.add_frame_task("tool_position", "tool", eik.TaskType.FRAME_POSITION)
    task.solve_mode = eik.TaskSolveMode.SCALE
    task.allow_min_error_fallback = False
    entry_position = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)
    task.set_target_position(entry_position - 0.03 * normal + 0.08 * tangent)

    entry_clearance = float(debug.distance)
    solver.configure_collision_constraint(
        min_distance=entry_clearance + 0.0002,
        include_pairs=list(robot.get_collision_pair_names()),
        max_constraints=1,
    )
    solver.set_proximity_gated_collision_activation_enabled(False)

    clearances = []
    for _ in range(16):
        result = solver.solve_velocity(q, apply_limits=True)
        assert result.weighted_fallback_used is False
        assert result.task_modes_effective[0] == eik.TaskSolveMode.SCALE
        assert result.task_scales[0] > 0.99
        q = q + solver.dt * np.asarray(result.solution, dtype=float)
        robot.update_configuration(q)
        clearance = solver.evaluate_min_collision_distance(q)
        assert clearance is not None
        clearances.append(float(clearance))

    final_position = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)
    assert float(np.dot(final_position - entry_position, tangent)) >= 0.02
    assert min(clearances) >= entry_clearance - 1e-6
    assert clearances[-1] >= entry_clearance + 0.0002 - 1e-6


@pytest.mark.parametrize(
    ("entry_x", "expected_recovery_clearance_m"),
    ((-0.006, 0.004), (-0.01, 0.0)),
    ids=("above_floor", "at_floor"),
)
def test_adaptive_position_step_uses_effective_recovery_clearance_for_tangent_motion(
    tmp_path,
    entry_x,
    expected_recovery_clearance_m,
):
    robot = eik.RobotModel(str(_write_axis_aligned_contact_urdf(tmp_path)), floating_base=False)
    q = np.array([entry_x, 0.0], dtype=float)
    robot.update_configuration(q)

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    _disable_weighted_fallback(solver)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.set_non_worsening_collision_floor_enabled(True)
    structural_floor_m = 0.01
    nominal_margin_m = 0.07
    solver.set_collision_structural_floor(structural_floor_m)
    solver.configure_collision_constraint(
        min_distance=nominal_margin_m,
        include_pairs=list(robot.get_collision_pair_names()),
        max_constraints=1,
    )
    solver.set_proximity_gated_collision_activation_enabled(False)

    task = solver.add_frame_task("adaptive_tangent", "tool", eik.TaskType.FRAME_POSITION)
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    entry_position = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)
    task.set_target_position(entry_position)
    solver.solve_velocity(q, apply_limits=True)

    target_pose = np.eye(4, dtype=float)
    tangent_offset_m = 0.08
    target_pose[:3, 3] = entry_position + np.array([0.0, tangent_offset_m, 0.0], dtype=float)
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.dt = solver.dt
    options.position_gain = 10.0
    options.orientation_gain = 0.0
    options.max_linear_speed = 0.5
    options.adaptive_dt = True
    options.adaptive_dt_max_scale = 10.0
    options.adaptive_dt_reference_distance = 0.02
    options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
    options.continuity_command_revision = 1

    entry_clearance = float(solver.evaluate_collision_debug(q).distance)
    result = solver.solve_position_step(q, target_pose, "adaptive_tangent", options)
    q_next = np.asarray(result.q_solution, dtype=float)
    robot.update_configuration(q_next)
    final_position = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)
    final_clearance = float(solver.evaluate_collision_debug(q_next).distance)

    tangent_progress_m = float(final_position[1] - entry_position[1])
    recovery_clearance_m = entry_clearance - structural_floor_m
    assert recovery_clearance_m == pytest.approx(expected_recovery_clearance_m, abs=1e-6)
    expected_scale = min(
        tangent_offset_m / options.adaptive_dt_reference_distance,
        options.adaptive_dt_max_scale,
    )
    expected_progress_m = options.dt * options.max_linear_speed * expected_scale
    assert tangent_progress_m >= 0.98 * expected_progress_m
    assert tangent_progress_m <= expected_progress_m + 1e-6
    assert final_clearance >= entry_clearance - 1e-6


def test_changed_tangent_target_preserves_recovered_command_epoch_clearance(tmp_path):
    robot = eik.RobotModel(str(_write_contact_arm_urdf(tmp_path)), floating_base=False)
    q = _COMMAND_EPOCH_CONTACT_SEED.copy()
    robot.update_configuration(q)

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    debug = solver.evaluate_collision_debug(q)
    assert debug is not None
    nearest_delta = np.asarray(debug.point_b_world) - np.asarray(debug.point_a_world)
    normal = nearest_delta / np.linalg.norm(nearest_delta)
    tangent = np.array([-normal[1], normal[0], 0.0], dtype=float)
    tangent /= np.linalg.norm(tangent)
    entry_clearance = float(debug.distance)
    entry_position = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)

    task = solver.add_frame_task("tool_position", "tool", eik.TaskType.FRAME_POSITION)
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = eik.TaskSolveMode.SCALE
    solver.configure_collision_constraint(
        min_distance=entry_clearance - 0.002,
        include_pairs=list(robot.get_collision_pair_names()),
        max_constraints=1,
    )
    solver.set_proximity_gated_collision_activation_enabled(False)

    options = eik.PositionStepOptions()
    options.max_steps = 4
    options.dt = 0.02
    options.position_gain = 10.0
    options.orientation_gain = 0.0
    options.primary_solve_mode = eik.TaskSolveMode.SCALE
    options.continuity_command_revision = 1

    hold_pose = np.eye(4, dtype=float)
    hold_pose[:3, 3] = entry_position
    hold_result = solver.solve_position_step(q, hold_pose, "tool_position", options)
    q = np.asarray(hold_result.q_solution, dtype=float)
    robot.update_configuration(q)

    tangent_pose = np.eye(4, dtype=float)
    tangent_pose[:3, 3] = entry_position + 0.08 * tangent
    options.continuity_command_revision = 2
    clearances: list[float] = []
    progress: list[float] = []
    for _ in range(10):
        result = solver.solve_position_step(q, tangent_pose, "tool_position", options)
        q = np.asarray(result.q_solution, dtype=float)
        robot.update_configuration(q)
        clearance = solver.evaluate_min_collision_distance(q)
        assert clearance is not None
        clearances.append(float(clearance))
        position = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)
        progress.append(float(np.dot(position - entry_position, tangent)))

    assert progress[0] > 1e-4
    assert progress[-1] >= 0.06
    assert min(clearances) >= entry_clearance - 1e-4


def test_stationary_continuity_preserves_collision_rejection_diagnostics(tmp_path):
    robot = eik.RobotModel(str(_write_axis_aligned_contact_urdf(tmp_path)), floating_base=False)
    q = np.array([0.02, 0.0], dtype=float)
    robot.update_configuration(q)

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    pairs = list(robot.get_collision_pair_names())
    solver.configure_collision_constraint(
        min_distance=0.001,
        include_pairs=pairs,
        max_constraints=1,
    )
    solver.set_proximity_gated_collision_activation_enabled(True)
    solver.set_collision_constraint_activation_multiplier(1.0)
    solver.set_collision_pair_min_distance("base", "tool", 0.06, False)

    task = solver.add_frame_task("rejected_contact", "tool", eik.TaskType.FRAME_POSITION)
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    target_pose = np.eye(4, dtype=float)
    target_pose[:3, 3] = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)
    target_pose[:3, 3] += np.array([-0.10, 0.08, 0.0], dtype=float)

    options = eik.PositionStepOptions()
    options.max_steps = 2
    options.dt = 0.02
    options.position_gain = 10.0
    options.orientation_gain = 0.0
    options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
    options.continuity_command_revision = 1

    q_entry = q.copy()
    rejection_counts: list[int] = []
    for _ in range(30):
        result = solver.solve_position_step(q, target_pose, "rejected_contact", options)
        q = np.asarray(result.q_solution, dtype=float)
        rejection_counts.append(int(result.collision_rejection_count))
        assert result.stall_escape_count == 0
        assert "held current configuration" not in result.status_message
        assert "held satisfied stationary target" not in result.status_message

    np.testing.assert_allclose(q, q_entry, atol=1e-12)
    assert min(rejection_counts) >= 1
    clearance = solver.evaluate_min_collision_distance(q)
    assert clearance is not None
    assert clearance >= 0.04 - 1e-9


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "solve_mode",
    (
        eik.TaskSolveMode.SCALE,
        eik.TaskSolveMode.MIN_ERROR,
    ),
    ids=("scale", "min-error"),
)
def test_stationary_collision_bound_target_settles_after_sliding(tmp_path, solve_mode):
    robot = eik.RobotModel(str(_write_axis_aligned_contact_urdf(tmp_path)), floating_base=False)
    q = np.array([0.02, 0.0], dtype=float)
    robot.update_configuration(q)

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    _disable_weighted_fallback(solver)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    debug = solver.evaluate_collision_debug(q)
    assert debug is not None
    nearest_delta = np.asarray(debug.point_b_world) - np.asarray(debug.point_a_world)
    normal = nearest_delta / np.linalg.norm(nearest_delta)
    tangent = np.array([-normal[1], normal[0], 0.0], dtype=float)
    tangent /= np.linalg.norm(tangent)
    entry_position = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)
    target_pose = np.eye(4, dtype=float)
    target_pose[:3, 3] = entry_position - 0.03 * normal + 0.08 * tangent

    task = solver.add_frame_task("stationary_contact", "tool", eik.TaskType.FRAME_POSITION)
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = solve_mode
    task.allow_min_error_fallback = False

    entry_clearance = float(debug.distance)
    structural_clearance = entry_clearance + 0.0002
    solver.configure_collision_constraint(
        min_distance=structural_clearance,
        include_pairs=list(robot.get_collision_pair_names()),
        max_constraints=1,
    )
    solver.set_proximity_gated_collision_activation_enabled(False)

    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.dt = 0.02
    options.position_gain = 10.0
    options.orientation_gain = 0.0
    options.primary_solve_mode = solve_mode
    options.primary_allow_min_error_fallback = False
    options.max_configuration_step_norm = 0.08
    options.continuity_command_revision = 1

    q_trace = [q.copy()]
    tangent_progress: list[float] = []
    clearances: list[float] = []
    errors: list[float] = []
    for _ in range(320):
        result = solver.solve_position_step(q, target_pose, "stationary_contact", options)
        q = np.asarray(result.q_solution, dtype=float)
        assert np.all(np.isfinite(q))
        robot.update_configuration(q)
        q_trace.append(q.copy())
        position = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)
        tangent_progress.append(float(np.dot(position - entry_position, tangent)))
        errors.append(float(np.linalg.norm(target_pose[:3, 3] - position)))
        clearance = solver.evaluate_min_collision_distance(q)
        assert clearance is not None
        clearances.append(float(clearance))

    settle_steps = 100
    q_tail = np.vstack(q_trace[-(settle_steps + 1) :])
    velocities = np.diff(q_tail, axis=0) / options.dt
    speeds = np.linalg.norm(velocities, axis=1)
    acceleration = np.diff(velocities, axis=0) / options.dt
    jerk = np.diff(acceleration, axis=0) / options.dt
    products = np.einsum("ij,ij->i", velocities[:-1], velocities[1:])
    active = (speeds[:-1] > 1e-3) & (speeds[1:] > 1e-3)
    error_tail = np.asarray(errors[-settle_steps:], dtype=float)
    progress_tail = np.asarray(tangent_progress[-settle_steps:], dtype=float)

    assert max(clearances) >= structural_clearance - 1e-6
    assert min(clearances) >= entry_clearance - 1e-6
    assert tangent_progress[-1] >= 0.02
    assert float(np.sqrt(np.mean(speeds**2))) <= 0.005
    assert float(speeds.max(initial=0.0)) <= 0.02
    assert float(np.sum(np.linalg.norm(np.diff(velocities, axis=0), axis=1))) <= 0.05
    assert float(np.linalg.norm(q_tail[-1] - q_tail[0])) <= 0.005
    assert int(np.sum((products < 0.0) & active)) <= 2
    assert int(np.sum(np.diff(error_tail) > 1e-4)) <= 1
    assert int(np.sum(np.diff(progress_tail) < -1e-4)) <= 1
    assert float(np.linalg.norm(acceleration, axis=1).max(initial=0.0)) <= 2.5
    assert float(np.linalg.norm(jerk, axis=1).max(initial=0.0)) <= 250.0
