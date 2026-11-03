"""Authorized finite-zone recovery should serve the first unsatisfied task."""

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("budgets", [False, True])
def test_recovery_uses_declared_zone_without_changing_task_policy(tmp_path, weighted, budgets):
    path = tmp_path / "positive_stage.urdf"
    path.write_text(
        '<robot name="positive_stage"><link name="world"/><link name="right"/>'
        '<joint name="x" type="prismatic"><parent link="world"/><child link="right"/>'
        '<axis xyz="1 0 0"/><limit lower="0" upper="1" effort="100" velocity="1"/>'
        '</joint><link name="left"/><joint name="y" type="prismatic">'
        '<parent link="right"/><child link="left"/><axis xyz="1 0 0"/>'
        '<limit lower="0" upper="1" effort="100" velocity="1"/></joint></robot>'
    )
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    config = solver.runtime_config()
    config.weighted_fallback_enabled = weighted
    solver.configure_runtime(config)
    targets = []
    seed = np.full(2, 0.0001)
    for priority, frame, position in [(0, "left", seed.sum()), (1, "right", 0.1)]:
        task = solver.add_frame_task(frame, frame, eik.TaskType.FRAME_POSITION)
        task.priority = priority
        task.solve_mode = eik.TaskSolveMode.MIN_ERROR
        pose = np.eye(4)
        pose[0, 3] = position
        target = eik.TaskTarget(frame, pose, 1.0, 1.0)
        target.priority_position_tolerance = 0.002
        target.priority_orientation_tolerance = 0.002
        targets.append(target)
    options = eik.PositionStepOptions()
    options.dt = solver.dt
    options.max_steps = 1
    options.per_target_priority_budgets = budgets
    options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
    options.primary_allow_min_error_fallback = False
    result = solver.solve_position_step(seed, targets, options)
    q = np.asarray(result.q_solution)
    assert np.all(q >= -1e-9)
    assert np.max(q - seed) <= solver.dt + 1e-9
    assert q.sum() - seed.sum() <= 0.002 + 1e-9
    if weighted and budgets:
        assert q[0] > 0.001
    else:
        np.testing.assert_allclose(q, seed, atol=1e-9)
    for priority, frame in enumerate(("left", "right")):
        task = solver.get_task(frame)
        assert task.priority == priority
        assert task.solve_mode == eik.TaskSolveMode.MIN_ERROR
        assert task.weight == 1.0
    # The transient recovery objective must not leak into a later velocity call.
    following = solver.solve_velocity(seed)
    assert np.linalg.norm(following.joint_velocities) <= 1e-7
