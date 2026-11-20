"""A feasible translation must not stall on noncommuting wrist curvature."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import embodik as eik


def _make_wrist(tmp_path, offset=False):
    axes = ["1 0 0", "0 1 0", "0 0 1", "1 0 0", "0 1 0", "0 0 1", "1 0 0", "0 1 0"]
    parts = ['<robot name="redundant_wrist"><link name="world"/>']
    parent = "world"
    for index, axis in enumerate(axes):
        child = f"link{index}"
        kind = "prismatic" if index < 3 else "revolute"
        parts.append(
            f'<link name="{child}"/><joint name="joint{index}" type="{kind}">'
            f'<parent link="{parent}"/><child link="{child}"/><axis xyz="{axis}"/>'
            '<limit lower="-2" upper="2" effort="100" velocity="4"/></joint>'
        )
        parent = child
    if offset:
        parts.append(
            f'<link name="tool"/><joint name="offset" type="fixed">'
            f'<parent link="{parent}"/><child link="tool"/><origin xyz="1 0.2 0.3"/></joint>'
        )
        parent = "tool"
    path = tmp_path / "redundant_wrist.urdf"
    path.write_text("".join(parts) + "</robot>")
    robot = eik.RobotModel(str(path), floating_base=False)
    return robot, parent


def test_reduced_budget_command_cannot_discard_mandatory_velocity_row(tmp_path):
    path = tmp_path / "mandatory_turn.urdf"
    path.write_text("""<robot name="mandatory_turn"><link name="world"/>
<link name="arm"/><joint name="yaw" type="revolute"><parent link="world"/>
<child link="arm"/><axis xyz="0 0 1"/>
<limit lower="-1" upper="1" velocity="1" effort="100"/></joint>
<link name="tool"/><joint name="offset" type="fixed"><parent link="arm"/>
<child link="tool"/><origin xyz="1 0 0"/></joint></robot>""")
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.1
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    runtime = solver.runtime_config()
    runtime.weighted_fallback_enabled = False
    solver.configure_runtime(runtime)
    solver.set_linear_velocity_constraints(np.ones((1, 1)), np.array([0.8]), np.array([1.0]))
    turn = solver.add_frame_task("turn", "tool", eik.TaskType.FRAME_ORIENTATION)
    turn.priority = 0
    hold = solver.add_frame_task("hold", "tool", eik.TaskType.FRAME_POSITION)
    hold.priority, hold.solve_mode = 1, eik.TaskSolveMode.MIN_ERROR
    pose = eik.SE3(
        rotation=Rotation.from_rotvec([0, 0, 0.2]).as_matrix(),
        translation=np.array([1.0, 0.0, 0.0]),
    )
    target = eik.TaskTarget.from_se3("turn", pose, 0.0, 10.0)
    protected = eik.TaskTarget.from_se3("hold", pose, 1.0, 0.0)
    protected.priority_position_tolerance = 0.002
    options = eik.PositionStepOptions()
    options.dt, options.max_steps = 0.1, 1
    options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
    options.per_target_priority_budgets = True
    options.stall_recovery = False
    result = solver.solve_position_step(np.zeros(1), [target, protected], options)
    # Any allowed velocity travels >=0.08rad, whose chord exceeds the 2mm zone.
    # A smaller geometrically safe fraction must not override the caller row.
    assert 2 * np.sin(0.08 / 2) > 0.002
    assert result.status != eik.SolverStatus.SUCCESS
    np.testing.assert_allclose(result.q_solution, np.zeros(1), atol=1e-12, rtol=0)
    assert solver.get_linear_velocity_constraint_rows() == 1


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("coupled_rows", [False, True])
@pytest.mark.parametrize(
    "mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.SCALE_ELASTIC, eik.TaskSolveMode.MIN_ERROR]
)
def test_orientation_budget_correction_retains_full_translation_rate(
    tmp_path, seed, mode, coupled_rows
):
    robot, parent = _make_wrist(tmp_path)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.03
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    if coupled_rows:
        row = np.zeros((1, robot.nv))
        row[0, -2:] = 1.0
        solver.set_linear_position_constraints(row, np.array([-4.0]), np.array([4.0]))
    rng = np.random.default_rng(20261023 + seed)
    q = np.r_[np.zeros(3), rng.uniform(-0.5, 0.5, 5)]
    robot.update_configuration(q)
    initial_pose = robot.get_frame_pose(parent)
    rotation = np.asarray(initial_pose.rotation) @ Rotation.from_rotvec([0, 0, 0.00199]).as_matrix()
    target_pose = eik.SE3(
        rotation=rotation, translation=np.asarray(initial_pose.translation) + [0.5, 0, 0]
    )
    primary = solver.add_frame_task("move", parent, eik.TaskType.FRAME_POSITION)
    primary.priority, primary.solve_mode = 0, mode
    orientation = solver.add_frame_task("hold", parent, eik.TaskType.FRAME_ORIENTATION)
    orientation.priority, orientation.solve_mode = 1, eik.TaskSolveMode.MIN_ERROR
    posture = solver.add_posture_task("posture")
    posture.priority, posture.solve_mode, posture.weight = 2, eik.TaskSolveMode.MIN_ERROR, 20.0
    posture.set_target_configuration(np.r_[np.zeros(3), rng.uniform(-1.5, 1.5, 5)])
    target = eik.TaskTarget.from_se3("move", target_pose, 10.0, 0.0)
    protected = eik.TaskTarget.from_se3("hold", target_pose, 0.0, 10.0)
    protected.priority_orientation_tolerance = 0.002
    options = eik.PositionStepOptions()
    options.dt, options.max_steps, options.max_linear_speed = 0.03, 1, 0.5
    options.per_target_priority_budgets = True
    options.primary_solve_mode = mode

    # Independent witness: translate only the first slider at 0.5 m/s and
    # retain all wrist coordinates. Its orientation stays at the initial
    # 1.99 mrad error. Numerical correction may optimize posture too, but must
    # not sacrifice this feasible primary rate or publish an over-budget pose.
    for tick in range(10):
        previous = q.copy()
        result = solver.solve_position_step(q, [target, protected], options)
        assert result.status == eik.SolverStatus.SUCCESS, result.status_message
        q = np.asarray(result.q_solution).copy()
        assert q[0] == pytest.approx(0.015 * (tick + 1), abs=1e-9)
        assert np.all(np.abs(q) <= 2 + 1e-10)
        assert np.all(np.abs(q - previous) <= 4 * options.dt + 1e-9)
        if coupled_rows:
            assert abs(q[-2] + q[-1]) <= 4.0 + 1e-9
        robot.update_configuration(q)
        error = Rotation.from_matrix(
            rotation @ np.asarray(robot.get_frame_pose(parent).rotation).T
        ).magnitude()
        assert error <= 0.002 + 1e-9
        assert solver.get_linear_velocity_constraint_rows() == 0
        assert not solver.runtime_config().refine_partial_scale


def _gradient_capture(tmp_path, orientation_only, mask, excluded, seed, outside):
    robot, frame = _make_wrist(tmp_path, offset=True)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.03
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    rng = np.random.default_rng(seed)
    q = np.r_[np.zeros(3), rng.uniform(-0.5, 0.5, 5)]
    robot.update_configuration(q)
    actual = robot.get_frame_pose(frame)
    mask = np.asarray(mask)
    direction = np.array([0.2, 0.3, 0.4])
    delta = (0.7 if outside else 0.00199) * direction / np.linalg.norm(mask * direction)
    position = np.asarray(actual.translation).copy()
    rotation = np.asarray(actual.rotation).copy()
    if orientation_only:
        rotation = Rotation.from_rotvec(delta).as_matrix() @ rotation
    else:
        position += delta
    primary = solver.add_frame_task("move", "link2", eik.TaskType.FRAME_POSITION)
    primary.priority = 0
    kind = eik.TaskType.FRAME_ORIENTATION if orientation_only else eik.TaskType.FRAME_POSITION
    hold = solver.add_frame_task("hold", frame, kind)
    hold.priority, hold.solve_mode = 1, eik.TaskSolveMode.MIN_ERROR
    if orientation_only:
        hold.set_orientation_mask(mask)
    else:
        hold.set_position_mask(mask)
    if excluded:
        hold.set_excluded_joint_indices([3])
    posture = solver.add_posture_task("posture")
    posture.priority, posture.solve_mode, posture.weight = 2, eik.TaskSolveMode.MIN_ERROR, 20.0
    posture.set_target_configuration(np.r_[np.zeros(3), rng.uniform(-1.5, 1.5, 5)])
    move = eik.TaskTarget.from_se3(
        "move", eik.SE3(rotation=np.eye(3), translation=np.array([0.5, 0.0, 0.0])), 10.0, 0.0
    )
    protected = eik.TaskTarget.from_se3(
        "hold",
        eik.SE3(rotation=rotation, translation=position),
        0.0 if orientation_only else (0.0001 if outside else 10.0),
        (0.0001 if outside else 10.0) if orientation_only else 0.0,
    )
    if orientation_only:
        protected.priority_orientation_tolerance = 0.002
    else:
        protected.priority_position_tolerance = 0.002
    options = eik.PositionStepOptions()
    options.dt, options.max_steps, options.max_linear_speed = 0.03, 1, 0.5
    options.per_target_priority_budgets = True
    solver.enable_problem_tracing(True)
    solver.solve_position_step(q, [move, protected], options)
    trace = solver.get_last_problem_trace()
    return robot, solver, trace, frame, position, rotation, mask


@pytest.mark.parametrize("outside", [False, True])
@pytest.mark.parametrize("orientation_only", [False, True])
@pytest.mark.parametrize("mask", [[1.0, 1.0, 1.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.25]])
@pytest.mark.parametrize("excluded", [False, True])
def test_budget_gradient_matches_independent_physical_difference(
    tmp_path, orientation_only, mask, excluded, outside
):
    # Select the first deterministic state that actually needs a finite cut.
    # Some mask/seed combinations stay within budget without correction.
    new_rows = []
    for seed in range(20261023, 20261055):
        robot, solver, trace, frame, position, rotation, mask = _gradient_capture(
            tmp_path, orientation_only, mask, excluded, seed, outside
        )
        initial = np.asarray(trace.velocity_problems[0].constraint_matrix)
        initial = initial / np.asarray(trace.velocity_problems[0].backend_column_scale)
        for problem in trace.velocity_problems[1:]:
            physical = np.asarray(problem.constraint_matrix) / np.asarray(
                problem.backend_column_scale
            )
            new_rows = [
                row.copy()
                for row in physical
                if not any(np.allclose(row, old, atol=1e-12, rtol=1e-10) for old in initial)
            ]
            if new_rows:
                break
        if new_rows:
            break
    assert new_rows, "No finite-budget correction was exercised"
    candidate = np.asarray(trace.priority_candidates[0].proposed_q).copy()

    def merit(state):
        robot.update_configuration(state)
        pose = robot.get_frame_pose(frame)
        error = (
            Rotation.from_matrix(rotation @ np.asarray(pose.rotation).T).as_rotvec()
            if orientation_only
            else position - np.asarray(pose.translation)
        )
        return np.linalg.norm(mask * error)

    step = 1e-7
    reference = np.empty(robot.nv)
    for column in range(robot.nv):
        perturbation = np.zeros(robot.nv)
        perturbation[column] = step
        reference[column] = (merit(candidate + perturbation) - merit(candidate - perturbation)) / (
            2 * step
        )
    gradient = min(new_rows, key=lambda row: np.linalg.norm(row - reference))
    np.testing.assert_allclose(gradient, reference, atol=2e-7, rtol=2e-6)
    if excluded:
        assert abs(reference[3]) > 1e-6
        assert abs(gradient[3]) > 1e-6
