"""A fresh hierarchy can reduce finite-step curvature without scaling hard rows."""

import numpy as np
import pytest

import embodik as eik


def make_curved_solver(tmp_path, direction, lower_mode=None):
    slide_lower, slide_upper = (0, 2) if direction < 0 else (-2, 0)
    bend_lower, bend_upper = (-10, 0) if direction < 0 else (0, 10)
    auxiliary = (
        ""
        if lower_mode is None
        else """<link name="auxiliary"/>
<joint name="zz_aux" type="prismatic"><parent link="base"/><child link="auxiliary"/>
<axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint>"""
    )
    path = tmp_path / "curved_tool.urdf"
    path.write_text(f"""<robot name="curved_tool"><link name="base"/><link name="slide"/>
<link name="arm"/><link name="tool"/>
<joint name="slide" type="prismatic"><parent link="base"/><child link="slide"/>
<axis xyz="0 1 0"/><limit lower="{slide_lower}" upper="{slide_upper}" velocity="1" effort="1"/></joint>
<joint name="bend" type="revolute"><parent link="slide"/><child link="arm"/>
<axis xyz="0 0 1"/><limit lower="{bend_lower}" upper="{bend_upper}" velocity="4" effort="1"/></joint>
<joint name="counter" type="revolute"><parent link="arm"/><child link="tool"/>
<origin xyz="0.01 0 0"/><axis xyz="0 0 1"/><limit lower="-10" upper="10" velocity="4" effort="1"/></joint>
{auxiliary}
</robot>""")
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 1.0
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    runtime = solver.runtime_config()
    runtime.refine_partial_scale = True
    runtime.weighted_fallback_enabled = False
    solver.configure_runtime(runtime)
    task = solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSE)
    if lower_mode is not None:
        auxiliary_task = solver.add_joint_task("auxiliary", "zz_aux", 1.0)
        auxiliary_task.priority = 2
        auxiliary_task.weight = 1.0
        auxiliary_task.solve_mode = lower_mode
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.primary_solve_mode = eik.TaskSolveMode.SCALE_ELASTIC
    options.primary_allow_min_error_fallback = False
    options.record_velocity_candidates = True
    target = np.eye(4)
    target[:2, 3] = [0.01, direction * 0.08]
    return robot, solver, task, options, target


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("lower_mode", [None, eik.TaskSolveMode.SCALE, eik.TaskSolveMode.MIN_ERROR])
def test_curved_ray_retries_primary_request_and_refreshes_lower_metadata(
    tmp_path, direction, multiple, lower_mode
):
    robot, solver, task, options, target = make_curved_solver(tmp_path, direction, lower_mode)
    q = np.zeros(robot.nq)
    if multiple:
        result = solver.solve_position_step(q, [eik.TaskTarget("move", target, 1.0, 1.0)], options)
    else:
        result = solver.solve_position_step(q, target, "move", options)
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    assert "primary goal backoff" in result.status_message
    # The maximum local ray rotates four radians in this deliberately large
    # interval and worsens position error. Re-solving a half request rotates
    # two radians, preserves orientation, and reduces the finite position error.
    assert result.task_scales[0] == pytest.approx(0.25, abs=1e-8)
    expected_q = np.zeros(robot.nq)
    expected_q[robot.get_joint_config_index("bend")] = direction * 2.0
    expected_q[robot.get_joint_config_index("counter")] = -direction * 2.0
    if lower_mode is not None:
        expected_q[robot.get_joint_config_index("zz_aux")] = 1.0
    np.testing.assert_allclose(result.q_solution, expected_q, atol=1e-8)
    robot.update_configuration(np.asarray(result.q_solution))
    pose = robot.get_frame_pose("tool")
    assert np.linalg.norm(target[:3, 3] - pose.translation) < 0.08
    np.testing.assert_allclose(pose.rotation, np.eye(3), atol=1e-8)
    if lower_mode is not None:
        # The independent lower joint retains its full requested motion.
        # Halving the old velocity would incorrectly halve this achieved image.
        auxiliary_index = robot.get_joint_config_index("zz_aux")
        assert result.q_solution[auxiliary_index] / solver.dt == pytest.approx(1.0, abs=1e-8)
        assert result.task_modes_effective[1] == lower_mode
        expected = 1.0 if lower_mode == eik.TaskSolveMode.SCALE else -1.0
        assert result.task_scales[1] == pytest.approx(expected, abs=1e-8)
        assert not result.task_used_fallback[1]
        assert result.task_errors[1] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("lower_mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.MIN_ERROR])
def test_backoff_retains_coupled_posture_image_and_best_effort_metadata(tmp_path, lower_mode):
    robot, solver, task, options, target = make_curved_solver(tmp_path, -1.0)
    posture = solver.add_posture_task("posture")
    posture.priority = 2
    posture.weight = 0.1
    posture.solve_mode = lower_mode
    posture_target = np.array([0.0, -2.0, 2.0])
    posture.set_target_configuration(posture_target)
    result = solver.solve_position_step(np.zeros(3), target, "move", options)
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    assert "primary goal backoff" in result.status_message
    np.testing.assert_allclose(result.q_solution, posture_target, atol=1e-8)
    actual_velocity = np.asarray(result.q_solution) / solver.dt
    expected_error = np.linalg.norm(actual_velocity - 0.1 * posture_target)
    assert result.task_errors[1] == pytest.approx(expected_error, abs=1e-8)
    assert result.task_modes_effective[1] == lower_mode
    assert result.task_scales[1] == pytest.approx(
        0.0 if lower_mode == eik.TaskSolveMode.SCALE else -1.0, abs=1e-8
    )
    assert not result.task_used_fallback[1]


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_primary_backoff_cannot_scale_origin_excluding_hard_row(tmp_path, direction):
    robot, solver, task, options, target = make_curved_solver(tmp_path, direction)
    # An origin-excluding row requires the full bend velocity. Halving the old
    # velocity would violate it. Fresh solves keep this row, and their nonlinear
    # endpoints remain unacceptable even if the slide compensates linearly.
    solver.set_linear_velocity_constraints(
        np.array([[0.0, 1.0, 0.0]]), np.array([direction * 4.0]), np.array([direction * 4.0])
    )
    result = solver.solve_position_step(np.zeros(3), target, "move", options)
    assert "primary goal backoff" not in result.status_message
    for candidate in result.velocity_candidates:
        assert candidate.solver_velocity[1] == pytest.approx(direction * 4.0, abs=1e-8)
    assert result.status != eik.SolverStatus.SUCCESS


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_explicit_pose_budget_is_honored_by_physical_ray_preference(tmp_path, direction):
    from scipy.spatial.transform import Rotation

    robot, solver, task, options, target = make_curved_solver(tmp_path, direction)
    target[:3, :3] = Rotation.from_rotvec([0.0, 0.0, direction * 0.1]).as_matrix()
    request = eik.TaskTarget("move", target, 1.0, 1.0)
    request.priority_position_tolerance = 0.1
    options.per_target_priority_budgets = True
    result = solver.solve_position_step(np.zeros(robot.nq), [request], options)
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    # The full bounded ray rotates4rad and stays inside the explicit10cm zone.
    # A hidden strict-per-block guard must not unnecessarily halve its rate.
    assert result.task_scales[0] >= 0.49
    robot.update_configuration(np.asarray(result.q_solution))
    pose = robot.get_frame_pose("tool")
    assert np.linalg.norm(target[:3, 3] - pose.translation) <= 0.1 + 1e-9
    assert (
        np.linalg.norm(Rotation.from_matrix(target[:3, :3].T @ pose.rotation).as_rotvec()) < 0.051
    )
    lower, upper = robot.get_joint_limits()
    assert np.all(result.q_solution >= lower) and np.all(result.q_solution <= upper)
    assert np.all(np.abs(result.q_solution) / solver.dt <= robot.get_velocity_limits() + 1e-10)
