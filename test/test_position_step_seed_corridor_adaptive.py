"""Seed-relative velocity corridors apply to the whole outer position command."""

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("multiple_targets", [False, True])
@pytest.mark.parametrize("max_steps", [1, 2])
@pytest.mark.parametrize("adaptive", [False, True])
def test_seed_corridor_uses_outer_interval_with_adaptive_integration(
    tmp_path, multiple_targets, max_steps, adaptive
):
    urdf = tmp_path / "slide.urdf"
    urdf.write_text("""<robot name="slide"><link name="base"/><link name="tool"/>
<joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/>
<axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
</robot>""")
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSITION)
    options = eik.PositionStepOptions()
    options.position_gain = 10.0
    options.max_linear_speed = 0.5
    options.max_steps = max_steps
    options.adaptive_dt = adaptive
    options.adaptive_dt_max_scale = 5.0
    options.adaptive_dt_reference_distance = 0.02
    options.limit_change_from_seed = True
    options.record_velocity_candidates = True
    initial = np.array([0.2])
    target = np.eye(4)
    target[0, 3] = 1.0
    if multiple_targets:
        result = solver.solve_position_step(
            initial, [eik.TaskTarget("move", target, 10.0, 1.0)], options
        )
    else:
        result = solver.solve_position_step(initial, target, "move", options)
    assert result.velocity_candidates
    for candidate in result.velocity_candidates:
        assert candidate.integration_dt >= solver.dt
        assert np.max(np.abs(candidate.integrated_velocity)) <= 1.0 + 1e-10
    delta = np.asarray(result.q_solution) - initial
    assert 0.0 < delta[0] <= solver.dt + 1e-10


def _slider_solver(tmp_path, *, initial=0.2, max_steps=2, adaptive=True):
    urdf = tmp_path / "bounded_slide.urdf"
    urdf.write_text("""<robot name="slide"><link name="base"/><link name="tool"/>
<joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/>
<axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
</robot>""")
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSITION)
    options = eik.PositionStepOptions()
    options.position_gain = 10
    options.max_linear_speed = 0.5
    options.max_steps = max_steps
    options.adaptive_dt = adaptive
    options.adaptive_dt_max_scale = 5
    options.adaptive_dt_reference_distance = 0.02
    options.limit_change_from_seed = True
    options.record_velocity_candidates = True
    target = np.eye(4)
    target[0, 3] = 1.0
    return solver, options, np.array([initial]), target


def _solve(solver, options, initial, target, multiple_targets):
    if multiple_targets:
        return solver.solve_position_step(
            initial, [eik.TaskTarget("move", target, 10.0, 1.0)], options
        )
    return solver.solve_position_step(initial, target, "move", options)


@pytest.mark.parametrize("multiple_targets", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("cap", [0.0, 0.2])
def test_seed_corridor_uses_effective_override(tmp_path, multiple_targets, adaptive, cap):
    solver, options, initial, target = _slider_solver(tmp_path, adaptive=adaptive)
    solver.set_joint_velocity_limit(0, cap)
    result = _solve(solver, options, initial, target, multiple_targets)
    assert np.max(np.abs(np.asarray(result.q_solution) - initial)) <= cap * solver.dt + 1e-10
    assert np.max(np.abs(result.joint_velocities)) <= cap + 1e-8


@pytest.mark.parametrize("multiple_targets", [False, True])
@pytest.mark.parametrize("max_steps", [1, 2])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("lower,upper,feasible", [(0.3, 0.4, True), (1.2, 1.4, False)])
def test_original_nonzero_hard_row_survives_seed_corridor(
    tmp_path, multiple_targets, max_steps, adaptive, lower, upper, feasible
):
    solver, options, initial, target = _slider_solver(
        tmp_path, max_steps=max_steps, adaptive=adaptive
    )
    solver.set_linear_velocity_constraints(np.ones((1, 1)), np.array([lower]), np.array([upper]))
    result = _solve(solver, options, initial, target, multiple_targets)
    assert np.max(np.abs(np.asarray(result.q_solution) - initial)) <= solver.dt + 1e-10
    if feasible:
        assert result.q_solution[0] > initial[0]
        for candidate in result.velocity_candidates:
            assert lower - 1e-8 <= candidate.integrated_velocity[0] <= upper + 1e-8
        if adaptive:
            assert result.velocity_candidates[0].integration_dt == pytest.approx(solver.dt)
    else:
        np.testing.assert_array_equal(result.q_solution, initial)
        assert result.status != eik.SolverStatus.SUCCESS


@pytest.mark.parametrize("multiple_targets", [False, True])
@pytest.mark.parametrize("initial", [2.0, 2.1])
def test_zero_override_does_not_publish_boundary_projection(tmp_path, multiple_targets, initial):
    solver, options, q, target = _slider_solver(tmp_path, initial=initial)
    solver.set_joint_velocity_limit(0, 0.0)
    target[0, 3] = 3.0
    result = _solve(solver, options, q, target, multiple_targets)
    np.testing.assert_array_equal(result.q_solution, q)
    np.testing.assert_array_equal(result.joint_velocities, [0.0])


@pytest.mark.parametrize("multiple_targets", [False, True])
def test_disabled_seed_corridor_keeps_iterative_prediction(tmp_path, multiple_targets):
    solver, options, initial, target = _slider_solver(tmp_path, max_steps=1)
    options.limit_change_from_seed = False
    result = _solve(solver, options, initial, target, multiple_targets)
    assert result.q_solution[0] - initial[0] > solver.dt


@pytest.mark.parametrize("multiple_targets", [False, True])
def test_stale_acceleration_cannot_override_zero_seed_cap(tmp_path, multiple_targets):
    solver, options, initial, target = _slider_solver(tmp_path, initial=1.99)
    solver.set_acceleration_limits(np.array([0.1]))
    solver.enable_acceleration_limits(True)
    solver.set_joint_velocity_limit(0, 0.0)
    options.current_joint_velocity = np.array([0.8])
    target[0, 3] = 3.0
    result = _solve(solver, options, initial, target, multiple_targets)
    np.testing.assert_array_equal(result.q_solution, initial)
    np.testing.assert_array_equal(result.joint_velocities, [0.0])


@pytest.mark.parametrize("mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.MIN_ERROR])
def test_lower_auxiliary_task_preserves_primary_image(tmp_path, mode):
    urdf = tmp_path / "branches.urdf"
    urdf.write_text(
        """<robot name="branches"><link name="base"/><link name="tool"/><link name="aux"/>
<joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/>
<axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
<joint name="aux_slide" type="prismatic"><parent link="base"/><child link="aux"/>
<axis xyz="0 1 0"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint></robot>"""
    )
    achieved = []
    for secondary in (False, True):
        robot = eik.RobotModel(
            str(urdf), floating_base=False, actuated_joint_names=["slide", "aux_slide"]
        )
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01
        runtime = solver.runtime_config()
        runtime.weighted_fallback_enabled = False
        solver.configure_runtime(runtime)
        solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSITION)
        primary_index = robot.get_joint_config_index("slide")
        auxiliary_index = robot.get_joint_config_index("aux_slide")
        initial = np.zeros(robot.nq)
        initial[primary_index] = 0.2
        if secondary:
            task = solver.add_posture_task("auxiliary")
            task.priority = 2
            task.solve_mode = mode
            task.set_controlled_joint_indices([auxiliary_index])
            posture_target = initial.copy()
            posture_target[auxiliary_index] = 0.5
            task.set_target_configuration(posture_target)
        options = eik.PositionStepOptions()
        options.max_steps = 2
        options.adaptive_dt = True
        options.limit_change_from_seed = True
        options.primary_solve_mode = eik.TaskSolveMode.SCALE
        options.primary_allow_min_error_fallback = False
        target = np.eye(4)
        target[0, 3] = 1.0
        result = solver.solve_position_step(
            initial, [eik.TaskTarget("move", target, 10.0, 1.0)], options
        )
        achieved.append(np.asarray(result.q_solution).copy())
    assert achieved[0][primary_index] > 0.2
    assert achieved[1][primary_index] == pytest.approx(achieved[0][primary_index], abs=1e-8)
    assert 0 < achieved[1][auxiliary_index] <= 0.01 + 1e-10


def test_continuous_joint_coordinates_are_not_used_as_scalar_corridors(tmp_path):
    urdf = tmp_path / "continuous_slide.urdf"
    urdf.write_text(
        """<robot name="continuous_slide"><link name="base"/><link name="turn"/><link name="tool"/>
<joint name="turn_joint" type="continuous"><parent link="base"/><child link="turn"/>
<axis xyz="0 0 1"/><limit velocity="1" effort="1"/></joint>
<joint name="slide" type="prismatic"><parent link="turn"/><child link="tool"/>
<origin xyz="1 0 0"/><axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint></robot>"""
    )
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSITION)
    initial = np.array([1.0, 0.0, 0.0])
    options = eik.PositionStepOptions()
    options.max_steps = 2
    options.adaptive_dt = True
    options.limit_change_from_seed = True
    target = np.eye(4)
    target[:3, 3] = [1.2, 0.2, 0.0]
    result = solver.solve_position_step(initial, target, "move", options)
    assert np.isfinite(result.q_solution).all()
    assert np.linalg.norm(result.q_solution[:2]) == pytest.approx(1.0, abs=1e-12)
    assert abs(result.q_solution[2]) <= solver.dt + 1e-10


@pytest.mark.parametrize("secondary_mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.MIN_ERROR])
def test_unequal_seed_caps_preserve_diagonal_primary_ray(tmp_path, secondary_mode):
    urdf = tmp_path / "plane.urdf"
    urdf.write_text("""<robot name="plane"><link name="base"/><link name="x"/><link name="tool"/>
<joint name="x_slide" type="prismatic"><parent link="base"/><child link="x"/>
<axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
<joint name="y_slide" type="prismatic"><parent link="x"/><child link="tool"/>
<axis xyz="0 1 0"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint></robot>""")
    endpoints = []
    for secondary in (False, True):
        robot = eik.RobotModel(str(urdf), floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01
        runtime = solver.runtime_config()
        runtime.weighted_fallback_enabled = False
        solver.configure_runtime(runtime)
        solver.set_joint_velocity_limit(0, 0.2)
        task = solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSITION)
        task.solve_mode = eik.TaskSolveMode.SCALE
        task.allow_min_error_fallback = False
        if secondary:
            posture = solver.add_posture_task("secondary")
            posture.priority = 2
            posture.solve_mode = secondary_mode
            posture.set_target_configuration(np.array([1.0, -1.0]))
        options = eik.PositionStepOptions()
        options.position_gain = 10
        options.max_linear_speed = 1
        options.max_steps = 2
        options.adaptive_dt = True
        options.limit_change_from_seed = True
        options.primary_solve_mode = eik.TaskSolveMode.SCALE
        options.primary_allow_min_error_fallback = False
        options.record_velocity_candidates = True
        target = np.eye(4)
        target[:2, 3] = 1.0
        result = solver.solve_position_step(np.zeros(2), target, "move", options)
        endpoints.append(np.asarray(result.q_solution).copy())
        assert result.status == eik.SolverStatus.INFEASIBLE
        assert result.task_scales[0] == 0
        assert result.task_modes_effective[0] == eik.TaskSolveMode.SCALE
        assert not result.weighted_fallback_used
        np.testing.assert_allclose(
            result.q_solution, result.velocity_candidates[0].candidate_q, atol=1e-12, rtol=0
        )
        for candidate in result.velocity_candidates:
            assert candidate.integrated_velocity[0] == pytest.approx(
                candidate.integrated_velocity[1], abs=1e-10
            )
        assert 0 < endpoints[-1][0] <= 0.002 + 1e-10
        assert endpoints[-1][0] == pytest.approx(endpoints[-1][1], abs=1e-10)
    np.testing.assert_allclose(endpoints[0], endpoints[1], atol=1e-10, rtol=0)


@pytest.mark.parametrize("multiple_targets", [False, True])
@pytest.mark.parametrize("secondary_mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.MIN_ERROR])
def test_first_failed_strict_primary_does_not_execute_lower_vector(
    tmp_path, multiple_targets, secondary_mode
):
    urdf = tmp_path / "blocked_plane.urdf"
    urdf.write_text("""<robot name="plane"><link name="base"/><link name="x"/><link name="tool"/>
<joint name="x_slide" type="prismatic"><parent link="base"/><child link="x"/>
<axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
<joint name="y_slide" type="prismatic"><parent link="x"/><child link="tool"/>
<axis xyz="0 1 0"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint></robot>""")
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    runtime = solver.runtime_config()
    runtime.weighted_fallback_enabled = False
    solver.configure_runtime(runtime)
    solver.set_joint_velocity_limit(0, 0)
    solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSITION)
    posture = solver.add_posture_task("secondary")
    posture.priority = 2
    posture.solve_mode = secondary_mode
    posture.set_target_configuration(np.array([0.0, -1.0]))
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.adaptive_dt = True
    options.limit_change_from_seed = True
    options.primary_solve_mode = eik.TaskSolveMode.SCALE
    options.primary_allow_min_error_fallback = False
    target = np.eye(4)
    target[:2, 3] = 1.0
    result = _solve(solver, options, np.zeros(2), target, multiple_targets)
    np.testing.assert_array_equal(result.q_solution, np.zeros(2))
    np.testing.assert_array_equal(result.joint_velocities, np.zeros(2))
    assert result.status == eik.SolverStatus.INFEASIBLE
    assert result.task_scales[0] == 0
    assert result.task_modes_effective[0] == eik.TaskSolveMode.SCALE
