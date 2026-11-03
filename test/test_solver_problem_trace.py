"""Opt-in snapshots replay the actual nominal hierarchy without changing motion."""

import numpy as np
import pytest

import embodik as eik

URDF = """<robot name="trace"><link name="world"/><link name="x"/>
<joint name="x_joint" type="prismatic"><parent link="world"/><child link="x"/>
<axis xyz="1 0 0"/><limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
<link name="tool"/><joint name="y_joint" type="prismatic"><parent link="x"/>
<child link="tool"/><axis xyz="0 1 0"/>
<limit lower="-1" upper="1" velocity="1" effort="1"/></joint></robot>"""
MODES = (eik.TaskSolveMode.SCALE, eik.TaskSolveMode.MIN_ERROR, eik.TaskSolveMode.SCALE_ELASTIC)


def make_solver(path, mode, metric):
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    task = solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSITION)
    task.solve_mode = mode
    task.allow_min_error_fallback = False
    target = np.eye(4)
    target[:2, 3] = [0.1, 0.05]
    task.set_target_pose(target[:3, 3], target[:3, :3])
    if metric:
        solver.set_joint_metric_weights(np.array([4.0, 0.25]))
    return robot, solver, target


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("metric", [False, True])
@pytest.mark.parametrize("position_step", [False, True])
def test_trace_replays_nominal_backend_and_preserves_result(tmp_path, mode, metric, position_step):
    path = tmp_path / "trace.urdf"
    path.write_text(URDF)
    _, baseline, target = make_solver(path, mode, metric)
    robot, traced, _ = make_solver(path, mode, metric)
    traced.enable_problem_tracing()

    def solve(solver):
        if not position_step:
            return solver.solve_velocity(np.zeros(robot.nq))
        options = eik.PositionStepOptions()
        options.primary_solve_mode = mode
        options.primary_allow_min_error_fallback = False
        return solver.solve_position_step(
            np.zeros(robot.nq), [eik.TaskTarget("move", target, 1.0, 1.0)], options
        )

    expected = solve(baseline)
    actual = solve(traced)
    assert actual.status == expected.status == eik.SolverStatus.SUCCESS
    np.testing.assert_array_equal(actual.joint_velocities, expected.joint_velocities)
    assert not baseline.get_last_problem_trace().velocity_problems
    trace = traced.get_last_problem_trace()
    assert trace.velocity_problems
    problem = trace.velocity_problems[0]
    assert problem.active_task_names == ["move"]
    assert problem.constraint_matrix.shape[1] == robot.nv
    replay = eik.replay_velocity_problem(problem)
    assert replay.status == problem.backend_status
    np.testing.assert_allclose(replay.solution, problem.backend_solution, atol=1e-13, rtol=0)
    np.testing.assert_allclose(
        problem.backend_column_scale * problem.backend_solution,
        problem.returned_velocity,
        atol=1e-13,
        rtol=0,
    )
    if position_step:
        np.testing.assert_array_equal(actual.q_solution, expected.q_solution)
        np.testing.assert_array_equal(trace.position_result_q, actual.q_solution)
        assert trace.position_status == actual.status
        assert trace.continuity_decisions
        for decision in trace.continuity_decisions:
            assert decision.reason
            assert decision.seed_q.shape == decision.candidate_q.shape == (robot.nq,)
            assert np.isfinite(decision.initial_merit)
            assert np.isfinite(decision.final_merit)
            assert len(decision.initial_block_merits) == len(decision.final_block_merits)
        snapshot = trace.continuity_decisions[0]
        original_reason = snapshot.reason
        snapshot.reason = "caller edit"
        assert traced.get_last_problem_trace().continuity_decisions[0].reason == original_reason
    else:
        assert trace.position_result_q.size == 0

    # Snapshots remain usable after the next call; callers cannot edit storage.
    problem.lower_bounds = np.full(problem.lower_bounds.shape, -999.0)
    assert not np.all(traced.get_last_problem_trace().velocity_problems[0].lower_bounds == -999.0)
    solve(traced)
    assert len(traced.get_last_problem_trace().velocity_problems) == len(trace.velocity_problems)
    traced.enable_problem_tracing(False)
    solve(traced)
    assert not traced.get_last_problem_trace().velocity_problems


def test_priority_trace_identifies_protected_tasks_and_final_result(tmp_path):
    path = tmp_path / "priority.urdf"
    path.write_text(URDF)
    robot, solver, target = make_solver(path, eik.TaskSolveMode.MIN_ERROR, False)
    solver.add_frame_task("secondary", "world", eik.TaskType.FRAME_POSITION).priority = 1
    solver.enable_problem_tracing()
    result = solver.solve_position_step(
        np.zeros(robot.nq),
        [eik.TaskTarget("move", target), eik.TaskTarget("secondary", np.eye(4))],
        eik.PositionStepOptions(),
    )
    trace = solver.get_last_problem_trace()
    assert trace.priority_candidates
    candidate = trace.priority_candidates[0]
    assert candidate.task_names == ["move", "secondary"]
    assert candidate.task_priorities == [0, 1]
    assert len(candidate.baseline_block_merits) == 2
    assert candidate.accepted_fraction == 1.0
    np.testing.assert_array_equal(trace.position_result_q, result.q_solution)
