"""Coupled configuration bounds must follow every native state and horizon."""

import numpy as np
import pytest

import embodik as eik


@pytest.fixture
def system(tmp_path):
    path = tmp_path / "coupled_sliders.urdf"
    path.write_text("""<robot name="coupled"><link name="world"/>
<link name="x"/><joint name="x" type="prismatic"><parent link="world"/><child link="x"/>
<axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="100" velocity="1"/></joint>
<link name="tool"/><joint name="y" type="prismatic"><parent link="x"/><child link="tool"/>
<axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="100" velocity="1"/></joint>
</robot>""")
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    task = solver.add_frame_task("tool", "tool", eik.TaskType.FRAME_POSITION)
    C = np.array([[1.0, 1.0], [1.0, -1.0]])
    solver.set_linear_position_constraints(C, np.full(2, -0.5), np.full(2, 0.5))
    return robot, solver, task, C


@pytest.mark.parametrize(
    "mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.SCALE_ELASTIC, eik.TaskSolveMode.MIN_ERROR]
)
@pytest.mark.parametrize("steps,dt,adaptive", [(1, 0.01, False), (4, 0.07, False), (3, 0.02, True)])
@pytest.mark.parametrize("single", [False, True])
def test_corners_and_return_follow_all_inner_states(system, mode, steps, dt, adaptive, single):
    robot, solver, task, C = system
    task.solve_mode = mode
    options = eik.PositionStepOptions()
    options.dt, options.max_steps = dt, steps
    options.position_gain, options.orientation_gain = 10.0, 10.0
    options.adaptive_dt = adaptive
    options.primary_solve_mode = mode
    options.primary_allow_min_error_fallback = True
    options.record_velocity_candidates = True
    # A static user row remains independent and survives native row updates.
    solver.set_linear_velocity_constraints(
        np.array([[1.0, 0.0]]), np.array([-0.8]), np.array([0.8])
    )
    q = np.zeros(2)
    seen = 0
    for sign in (1.0, -1.0, 0.0):
        pose = np.eye(4)
        pose[:2, 3] = sign
        target = eik.TaskTarget.from_se3(
            "tool", eik.SE3(rotation=pose[:3, :3], translation=pose[:3, 3]), 10.0, 0.0
        )
        for _ in range(80):
            before = q.copy()
            result = (
                solver.solve_position_step(q, pose, "tool", options)
                if single
                else solver.solve_position_step(q, [target], options)
            )
            q = np.asarray(result.q_solution).copy()
            assert result.status in (eik.SolverStatus.SUCCESS, eik.SolverStatus.NO_PROGRESS)
            assert np.max(np.abs(C @ q)) <= 0.5 + 1e-9
            assert solver.get_linear_velocity_constraint_rows() == 1
            assert solver.get_linear_position_constraint_rows() == 2
            for candidate in result.velocity_candidates:
                seen += 1
                assert np.max(np.abs(C @ np.asarray(candidate.candidate_q))) <= 0.5 + 1e-9
            assert np.all(np.isfinite(q))
        if sign:
            assert np.dot(q, np.full(2, sign)) > 0.45
        else:
            assert np.linalg.norm(q) < 0.005
    assert seen > 0


def test_outside_seed_moves_inward_without_projection_or_reset(system):
    robot, solver, task, C = system
    q = np.array([0.4, 0.4])
    options = eik.PositionStepOptions()
    options.dt = 0.01
    options.position_gain = 10.0
    options.max_steps = 1
    options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    for _ in range(100):
        before = q.copy()
        result = solver.solve_position_step(q, np.eye(4), "tool", options)
        q = np.asarray(result.q_solution).copy()
        assert np.all(np.abs(q - before) <= 0.01 + 1e-9)
        assert np.all(C @ q <= np.maximum(0.5, C @ before) + 1e-9)
        assert np.all(C @ q >= np.minimum(-0.5, C @ before) - 1e-9)
    assert np.linalg.norm(q) < 0.005


def test_reconfigure_clear_and_owned_inputs(system):
    robot, solver, task, C = system
    C[:] = 0
    assert solver.get_linear_position_constraint_rows() == 2
    solver.clear_linear_position_constraints()
    assert solver.get_linear_position_constraint_rows() == 0
    solver.set_linear_position_constraints(np.eye(2), np.array([-0.1, -0.1]), np.array([0.1, 0.1]))
    assert solver.get_linear_position_constraint_rows() == 2
    with pytest.raises(ValueError):
        solver.set_linear_position_constraints(np.ones((1, 3)), np.array([-1.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        solver.set_linear_position_constraints(
            np.eye(2), np.array([1.0, 0.0]), np.array([-1.0, 1.0])
        )


def test_acceleration_execution_fails_closed_even_if_enabled_after_configuration(system):
    robot, solver, task, C = system
    solver.set_acceleration_limits(np.ones(2))
    solver.enable_acceleration_limits(True)
    result = solver.solve_position_step(np.zeros(2), np.eye(4), "tool")
    assert result.status == eik.SolverStatus.INVALID_INPUT
    np.testing.assert_array_equal(result.q_solution, np.zeros(2))
    result = solver.solve_velocity(np.zeros(2))
    assert result.status == eik.SolverStatus.INVALID_INPUT
    assert "acceleration" in result.status_message


@pytest.mark.parametrize(
    "mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.SCALE_ELASTIC, eik.TaskSolveMode.MIN_ERROR]
)
def test_locked_joint_and_actual_velocity_horizon(system, mode):
    robot, solver, task, C = system
    task.solve_mode = mode
    q = np.array([0.25, 0.25])
    target = np.eye(4)
    target[:2, 3] = [0.25, -0.2]
    options = eik.PositionStepOptions()
    options.dt, options.max_steps = 0.035, 4
    options.position_gain = 10.0
    options.locked_joint_indices = [0]
    options.primary_solve_mode = mode
    for _ in range(40):
        result = solver.solve_position_step(q, target, "tool", options)
        q = np.asarray(result.q_solution).copy()
        assert q[0] == 0.25
        assert np.max(np.abs(C @ q)) <= 0.5 + 1e-9
    assert abs(q[1] + 0.2) < 0.005
    solver.clear_linear_velocity_constraints()
    robot.update_configuration(np.array([0.249, 0.249]))
    task.set_target_pose(np.array([1.0, 1.0, 0.0]), np.eye(3))
    result = solver.solve_velocity(np.array([0.249, 0.249]))
    assert result.status == eik.SolverStatus.SUCCESS
    endpoint = np.array([0.249, 0.249]) + solver.dt * np.asarray(result.joint_velocities)
    assert np.max(np.abs(C @ endpoint)) <= 0.5 + 1e-9


def test_contact_execution_fails_closed(system):
    robot, solver, task, C = system
    solver.add_contact_frame("tool")
    result = solver.solve_position_step(np.zeros(2), np.eye(4), "tool")
    assert result.status == eik.SolverStatus.INVALID_INPUT
    np.testing.assert_array_equal(result.q_solution, np.zeros(2))


def test_collision_lift_rejects_uncertified_coupled_position_rows(system):
    robot, provider, task, C = system
    solver = eik.AccelerationSolver(robot)
    result = solver.solve_with_velocity_collision(
        provider,
        np.zeros(2),
        np.zeros(2),
        0.01,
        eik.AccelerationSolveOptions(),
        eik.VelocityCollisionLiftOptions(),
    )
    assert result.status == eik.SolverStatus.INVALID_INPUT
    assert "position-row stopping" in result.status_message


def test_empty_rows_remove_policy(system):
    robot, solver, task, C = system
    solver.set_linear_position_constraints(np.empty((0, 2)), np.empty(0), np.empty(0))
    assert solver.get_linear_position_constraint_rows() == 0
    solver.enable_acceleration_limits(True)
    solver.set_acceleration_limits(np.ones(2))
    result = solver.solve_position_step(np.zeros(2), np.eye(4), "tool")
    assert result.status != eik.SolverStatus.INVALID_INPUT


def test_backend_rows_use_each_captured_seed_and_actual_horizon(system):
    robot, solver, task, C = system
    solver.enable_problem_tracing(True)
    options = eik.PositionStepOptions()
    options.dt, options.max_steps = 0.03, 4
    options.adaptive_dt = True
    target = np.eye(4)
    target[:2, 3] = 1.0
    solver.solve_position_step(np.zeros(2), target, "tool", options)
    problems = solver.get_last_problem_trace().velocity_problems
    assert len(problems) >= 2
    for problem in problems:
        # Joint box is stacked before the generic linear rows. Locate the
        # two known rows and validate their physical-coordinate bounds.
        matrix = (
            np.asarray(problem.constraint_matrix)
            * np.asarray(problem.backend_column_scale)[None, :]
        )
        value = C @ np.asarray(problem.seed_q)
        for row in range(2):
            matches = np.where(np.max(np.abs(matrix - C[row]), axis=1) < 1e-12)[0]
            assert len(matches) == 1
            index = matches[0]
            assert problem.lower_bounds[index] == pytest.approx(
                (min(-0.5, value[row]) - value[row]) / problem.validation_dt
            )
            assert problem.upper_bounds[index] == pytest.approx(
                (max(0.5, value[row]) - value[row]) / problem.validation_dt
            )
