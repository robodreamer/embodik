"""Acceleration scaling and bounded nominal-position elasticity contracts."""

import numpy as np
import pytest

import embodik

DT = 0.05
LIMIT = 1.0
NOMINAL_LIMIT = 0.5
TARGET = 0.8
TOLERANCE = 1e-9
URDF = """<robot name="elastic_test"><link name="base"/><link name="tip"/>
<joint name="joint" type="prismatic"><parent link="base"/><child link="tip"/>
<axis xyz="1 0 0"/><limit lower="-1" upper="1" velocity="2" effort="10"/>
</joint></robot>"""


@pytest.fixture
def elastic_solver(tmp_path):
    path = tmp_path / "elastic.urdf"
    path.write_text(URDF)
    robot = embodik.RobotModel(str(path), floating_base=False)
    solver = embodik.AccelerationSolver(robot)
    task = solver.add_joint_task("goal", "joint", TARGET)
    task.allow_min_error_fallback = False
    task.solve_mode = embodik.TaskSolveMode.SCALE_ELASTIC
    options = embodik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.array([10.0])
    options.allow_state_box_task_fallback = False
    return solver, task, options


def test_scale_elastic_uses_scaling_backend_without_min_error_alias(elastic_solver):
    solver, _task, options = elastic_solver
    result = solver.solve(np.array([0.0]), np.array([0.0]), DT, options)
    assert result.status == embodik.SolverStatus.SUCCESS, result.status_message
    assert result.task_modes_effective == [embodik.TaskSolveMode.SCALE]
    assert not any(result.task_used_fallback)
    assert result.q_solution[0] > 0.0


def _narrow_nominal(options):
    options.nominal_position_lower_bounds = np.array([-NOMINAL_LIMIT])
    options.nominal_position_upper_bounds = np.array([NOMINAL_LIMIT])


def test_nominal_elastic_band_allows_progress_that_strict_scale_cannot(elastic_solver):
    solver, task, options = elastic_solver
    _narrow_nominal(options)
    seed, velocity = np.array([NOMINAL_LIMIT]), np.array([0.0])
    task.solve_mode = embodik.TaskSolveMode.SCALE
    strict = solver.solve(seed, velocity, DT, options)
    assert strict.status == embodik.SolverStatus.SUCCESS, strict.status_message
    np.testing.assert_allclose(strict.q_solution, seed)
    task.solve_mode = embodik.TaskSolveMode.SCALE_ELASTIC
    elastic = solver.solve(seed, velocity, DT, options)
    assert elastic.status == embodik.SolverStatus.SUCCESS, elastic.status_message
    assert elastic.elastic_band_enabled
    assert elastic.q_solution[0] > strict.q_solution[0]
    assert elastic.q_solution[0] <= NOMINAL_LIMIT + options.elastic_delta_max
    assert elastic.q_solution[0] <= LIMIT
    assert elastic.elastic_band_capacity_upper[0] == options.elastic_delta_max
    assert elastic.task_modes_effective == [embodik.TaskSolveMode.SCALE]
    assert not any(elastic.task_used_fallback)
    assert solver.get_elastic_band_deltas()[0] > 0.0
    solver.reset_elastic_band()
    np.testing.assert_array_equal(solver.get_elastic_band_deltas(), velocity)


def test_elastic_without_separate_nominal_envelope_has_no_physical_expansion_capacity(
    elastic_solver,
):
    solver, task, options = elastic_solver
    task.set_target_value(LIMIT + NOMINAL_LIMIT)
    result = solver.solve(np.array([LIMIT]), np.array([0.0]), DT, options)
    assert result.status == embodik.SolverStatus.SUCCESS, result.status_message
    np.testing.assert_array_equal(result.elastic_band_capacity_lower, np.zeros(1))
    np.testing.assert_array_equal(result.elastic_band_capacity_upper, np.zeros(1))
    assert result.q_solution[0] <= LIMIT


def test_explicit_frozen_wrist_style_velocity_row_remains_hard_under_elastic(elastic_solver):
    solver, _task, options = elastic_solver
    _narrow_nominal(options)
    row = embodik.FrozenNextVelocityConstraint()
    row.source_id = "coupled_safety_facet"
    row.coefficient_matrix = np.ones((1, 1))
    row.affine_bias = np.zeros(1)
    row.lower_bounds = -np.ones(1)
    row.upper_bounds = np.zeros(1)
    options.frozen_next_velocity_constraints = [row]
    seed = np.array([NOMINAL_LIMIT])
    for _step in range(5):
        result = solver.solve(seed, np.zeros(1), DT, options)
        assert result.status == embodik.SolverStatus.SUCCESS, result.status_message
        assert result.q_solution[0] <= seed[0] + TOLERANCE
        assert result.joint_velocities_next[0] <= TOLERANCE
        assert result.elastic_band_delta[0] <= options.elastic_delta_max


@pytest.mark.parametrize(
    "invalid", ["outside_hard", "unordered", "missing_side", "shape", "decay", "nonfinite"]
)
def test_invalid_nominal_envelope_and_elastic_configuration_fail_closed(elastic_solver, invalid):
    solver, _task, options = elastic_solver
    _narrow_nominal(options)
    if invalid == "outside_hard":
        options.nominal_position_upper_bounds = np.array([LIMIT + NOMINAL_LIMIT])
    elif invalid == "unordered":
        options.nominal_position_lower_bounds = np.array([TARGET])
    elif invalid == "missing_side":
        options.nominal_position_upper_bounds = None
    elif invalid == "shape":
        options.nominal_position_upper_bounds = np.zeros(2)
    elif invalid == "decay":
        options.elastic_decay_rate = 2.0
    else:
        options.elastic_expand_rate = float("nan")
    result = solver.solve(np.zeros(1), np.zeros(1), DT, options)
    assert result.status == embodik.SolverStatus.INVALID_INPUT
    assert result.q_solution.size == 0


def test_coupled_two_joint_safety_facets_are_not_expanded_with_nominal_band(tmp_path):
    second_joint = """<link name="tip2"/><joint name="joint2" type="prismatic">
<parent link="tip"/><child link="tip2"/><axis xyz="0 1 0"/>
<limit lower="-1" upper="1" velocity="2" effort="10"/></joint></robot>"""
    path = tmp_path / "coupled.urdf"
    path.write_text(URDF.replace("</robot>", second_joint))
    robot = embodik.RobotModel(str(path), floating_base=False)
    solver = embodik.AccelerationSolver(robot)
    for name in ("joint", "joint2"):
        task = solver.add_joint_task(name, name, TARGET)
        task.solve_mode = embodik.TaskSolveMode.SCALE_ELASTIC
        task.allow_min_error_fallback = False
    options = embodik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.full(robot.nv, 10.0)
    options.allow_state_box_task_fallback = False
    options.nominal_position_lower_bounds = np.full(robot.nq, -NOMINAL_LIMIT)
    options.nominal_position_upper_bounds = np.full(robot.nq, NOMINAL_LIMIT)
    row = embodik.FrozenNextVelocityConstraint()
    row.source_id = "coupled_diamond_facets"
    row.coefficient_matrix = np.array([[1.0, 1.0], [1.0, -1.0]])
    row.affine_bias = np.zeros(2)
    row.lower_bounds = -np.ones(2)
    row.upper_bounds = np.array([0.0, 1.0])
    options.frozen_next_velocity_constraints = [row]
    seed = np.full(robot.nq, NOMINAL_LIMIT)
    for _step in range(5):
        result = solver.solve(seed, np.zeros(robot.nv), DT, options)
        assert result.status == embodik.SolverStatus.SUCCESS, result.status_message
        facet_rates = row.coefficient_matrix @ result.joint_velocities_next
        assert np.all(facet_rates <= row.upper_bounds + TOLERANCE)
        assert np.all(facet_rates >= row.lower_bounds - TOLERANCE)
        assert np.sum(result.q_solution) <= np.sum(seed) + TOLERANCE
        assert np.all(result.elastic_band_delta <= options.elastic_delta_max)


def test_disabled_position_box_accepts_caller_unbounded_model_limits(tmp_path):
    path = tmp_path / "unbounded.urdf"
    path.write_text(URDF)
    robot = embodik.RobotModel(str(path), floating_base=False)
    robot.set_joint_limits(np.array([-np.inf]), np.array([np.inf]))
    solver = embodik.AccelerationSolver(robot)
    task = solver.add_joint_task("goal", "joint", TARGET)
    task.solve_mode = embodik.TaskSolveMode.SCALE_ELASTIC
    options = embodik.AccelerationSolveOptions()
    options.apply_position_limits = False
    options.acceleration_limits_override = np.array([10.0])
    result = solver.solve(np.zeros(1), np.zeros(1), DT, options)
    assert result.status == embodik.SolverStatus.SUCCESS, result.status_message
    assert result.q_solution[0] > 0.0
    np.testing.assert_array_equal(result.elastic_band_capacity_upper, np.zeros(1))


def test_model_hard_limit_updates_recap_existing_nominal_band(tmp_path):
    path = tmp_path / "recap.urdf"
    path.write_text(URDF)
    robot = embodik.RobotModel(str(path), floating_base=False)
    solver = embodik.AccelerationSolver(robot)
    task = solver.add_joint_task("goal", "joint", TARGET)
    task.solve_mode = embodik.TaskSolveMode.SCALE_ELASTIC
    task.allow_min_error_fallback = False
    options = embodik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.array([10.0])
    options.allow_state_box_task_fallback = False
    _narrow_nominal(options)
    seed = np.array([NOMINAL_LIMIT])
    first = solver.solve(seed, np.zeros(1), DT, options)
    assert first.status == embodik.SolverStatus.SUCCESS, first.status_message
    tighter_hard_upper = NOMINAL_LIMIT + options.elastic_expand_rate / 2.0
    robot.set_joint_limits(np.array([-LIMIT]), np.array([tighter_hard_upper]))
    second = solver.solve(seed, np.zeros(1), DT, options)
    assert second.status == embodik.SolverStatus.SUCCESS, second.status_message
    assert second.q_solution[0] <= tighter_hard_upper + TOLERANCE
    np.testing.assert_allclose(
        second.elastic_band_capacity_upper, tighter_hard_upper - NOMINAL_LIMIT
    )
    robot.set_joint_limits(np.array([-LIMIT]), np.array([NOMINAL_LIMIT / 2.0]))
    rejected = solver.solve(np.zeros(1), np.zeros(1), DT, options)
    assert rejected.status == embodik.SolverStatus.INVALID_INPUT
    assert rejected.q_solution.size == 0


@pytest.mark.parametrize("posture_reference", [0.2, 0.0])
@pytest.mark.parametrize("collect_diagnostics", [False, True])
def test_min_error_posture_does_not_expand_a_fully_achieved_elastic_band(
    tmp_path, posture_reference, collect_diagnostics
):
    second_joint = """<link name="tip2"/><joint name="joint2" type="prismatic">
<parent link="tip"/><child link="tip2"/><axis xyz="0 1 0"/>
<limit lower="-1" upper="1" velocity="2" effort="10"/></joint></robot>"""
    path = tmp_path / "mixed_elastic.urdf"
    path.write_text(URDF.replace("</robot>", second_joint))
    robot = embodik.RobotModel(str(path), floating_base=False)
    solver = embodik.AccelerationSolver(robot)
    primary = solver.add_joint_task("primary", "joint", 0.0)
    primary.solve_mode = embodik.TaskSolveMode.SCALE_ELASTIC
    primary.allow_min_error_fallback = False
    reference = embodik.AccelerationTaskReference()
    reference.proportional_gain = reference.derivative_gain = 0.0
    reference.desired_acceleration = np.array([0.2])
    solver.set_task_reference("primary", reference)
    posture = solver.add_posture_task("posture")
    posture.priority = 1
    posture.solve_mode = embodik.TaskSolveMode.MIN_ERROR
    reference = embodik.AccelerationTaskReference()
    reference.proportional_gain = reference.derivative_gain = 0.0
    reference.desired_acceleration = np.array([posture_reference, 0.0])
    solver.set_task_reference("posture", reference)
    options = embodik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.full(2, 10.0)
    options.nominal_position_lower_bounds = np.full(2, -NOMINAL_LIMIT)
    options.nominal_position_upper_bounds = np.full(2, NOMINAL_LIMIT)
    options.allow_state_box_task_fallback = False
    options.collect_task_diagnostics = collect_diagnostics
    assert options.elastic_expand_only_saturated

    # Reuse the same interior state to isolate adaptation from occupied space
    # and braking reserve. Check both the used band and post-solve state.
    for _ in range(3):
        result = solver.solve(np.zeros(2), np.zeros(2), DT, options)
        assert result.status == embodik.SolverStatus.SUCCESS, result.status_message
        np.testing.assert_allclose(result.task_scales, [1.0, -1.0], atol=TOLERANCE)
        np.testing.assert_allclose(
            result.task_errors, [0.0, abs(0.2 - posture_reference)], atol=TOLERANCE
        )
        assert not result.saturated_position_indices
        np.testing.assert_allclose(result.joint_accelerations, [0.2, 0.0], atol=TOLERANCE)
        np.testing.assert_array_equal(result.elastic_band_delta, np.zeros(2))
        np.testing.assert_array_equal(solver.get_elastic_band_deltas(), np.zeros(2))
        np.testing.assert_array_equal(
            result.elastic_effective_position_lower_bounds, np.full(2, -NOMINAL_LIMIT)
        )
        np.testing.assert_array_equal(
            result.elastic_effective_position_upper_bounds, np.full(2, NOMINAL_LIMIT)
        )


@pytest.mark.parametrize(
    "requested_mode", [embodik.TaskSolveMode.SCALE_ELASTIC, embodik.TaskSolveMode.MIN_ERROR]
)
def test_only_min_error_diagnostics_do_not_invent_an_elastic_stall(tmp_path, requested_mode):
    path = tmp_path / "unreachable_direction.urdf"
    path.write_text(URDF)
    robot = embodik.RobotModel(str(path), floating_base=False)
    solver = embodik.AccelerationSolver(robot)
    task = solver.add_frame_task("goal", "tip", embodik.TaskType.FRAME_POSITION)
    task.solve_mode = requested_mode
    task.allow_min_error_fallback = True
    task.set_position_mask(np.array([False, True, False]))
    reference = embodik.AccelerationTaskReference()
    reference.proportional_gain = reference.derivative_gain = 0.0
    reference.desired_acceleration = np.array([0.0, 1.0, 0.0])
    solver.set_task_reference("goal", reference)
    options = embodik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.array([10.0])
    options.allow_state_box_task_fallback = False
    _narrow_nominal(options)
    elastic_requested = requested_mode == embodik.TaskSolveMode.SCALE_ELASTIC

    for _ in range(3):
        result = solver.solve(np.zeros(1), np.zeros(1), DT, options)
        assert result.status == embodik.SolverStatus.SUCCESS, result.status_message
        assert result.elastic_band_enabled == elastic_requested
        assert result.task_scales == [-1.0]
        assert result.task_modes_effective == [embodik.TaskSolveMode.MIN_ERROR]
        assert result.task_used_fallback == [elastic_requested]
        np.testing.assert_array_equal(result.joint_accelerations, np.zeros(1))
        np.testing.assert_array_equal(result.elastic_band_delta, np.zeros(1))
        np.testing.assert_array_equal(solver.get_elastic_band_deltas(), np.zeros(1))
