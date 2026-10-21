"""Finite-step merit retries must shrink even when their linear bound stays active."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import embodik as eik


@pytest.mark.parametrize(
    "mode", [eik.TaskSolveMode.MIN_ERROR, eik.TaskSolveMode.SCALE, eik.TaskSolveMode.SCALE_ELASTIC]
)
@pytest.mark.parametrize("yaw", [0.03214285714285714, 0.1, 0.15])
def test_rejected_retry_contracts_and_retains_feasible_motion(tmp_path, mode, yaw):
    urdf = tmp_path / "merit_retry.urdf"
    urdf.write_text(
        '<robot name="merit_retry"><link name="world"/><link name="slide"/>'
        '<joint name="slide" type="prismatic"><parent link="world"/>'
        '<child link="slide"/><axis xyz="0 1 0"/>'
        '<limit lower="-2" upper="2" effort="100" velocity="2"/></joint>'
        '<link name="yaw"/><joint name="yaw" type="revolute">'
        '<parent link="slide"/><child link="yaw"/><axis xyz="0 0 1"/>'
        '<limit lower="-2" upper="2" effort="100" velocity="2"/></joint>'
        '<link name="ee"/><joint name="offset" type="fixed">'
        '<parent link="yaw"/><child link="ee"/><origin xyz="1 0 0"/></joint></robot>'
    )
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.enable_problem_tracing(True)
    # Exercise constrained re-solving, rather than unconstrained direct backoff.
    solver.set_linear_velocity_constraints(np.eye(2), np.full(2, -2.0), np.full(2, 2.0))
    q = np.array([0.0, yaw])
    robot.update_configuration(q)
    task = solver.add_frame_task("hand", "ee")
    task.priority = 0
    pose = robot.get_frame_pose("ee")
    translation = np.array(pose.translation) + [0.0019, 0.0, 0.0]
    target = eik.TaskTarget.from_se3(
        "hand",
        eik.SE3(rotation=Rotation.from_rotvec([0, 0, 1.0]).as_matrix(), translation=translation),
        1.0,
        1.0,
    )
    target.priority_position_tolerance = 0.002
    options = eik.PositionStepOptions()
    options.dt = 0.02
    options.max_steps = 1
    options.per_target_priority_budgets = True
    options.primary_solve_mode = mode
    result = solver.solve_position_step(q, [target], options)
    trace = solver.get_last_problem_trace()
    problems = trace.velocity_problems
    assert problems and 0.9 < trace.priority_candidates[0].accepted_fraction < 1.0
    assert len(problems) <= 9  # Original solve plus the unchanged eight-retry cap.
    for previous, retry in zip(problems, problems[1:]):
        if retry.validation_dt == previous.validation_dt:
            # A new finite-budget correction changes the subproblem; unchanged
            # near-one proposals must still contract instead of repeating.
            assert retry.constraint_matrix.shape[0] > previous.constraint_matrix.shape[0]
        else:
            assert retry.validation_dt <= 0.5 * previous.validation_dt
    assert np.linalg.norm(np.asarray(result.q_solution) - q) > 1e-6
    robot.update_configuration(result.q_solution)
    actual = robot.get_frame_pose("ee")
    assert np.linalg.norm(np.asarray(actual.translation) - translation) <= 0.002 + 1e-9
    # The returned step still obeys the original outer physical velocity bound.
    assert np.max(np.abs(np.asarray(result.q_solution) - q) / options.dt) <= 2.0 + 1e-9
