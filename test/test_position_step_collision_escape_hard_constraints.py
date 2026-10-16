"""Post-solve collision escapes must preserve position-step hard constraints."""

import numpy as np
import pytest

import embodik as eik


@pytest.fixture
def escape_solver(tmp_path):
    path = tmp_path / "escape_stage.urdf"
    path.write_text("""<robot name="escape_stage">
<link name="base"><collision><geometry><sphere radius="0.05"/></geometry></collision></link>
<link name="stage"/>
<link name="tool"><collision><geometry><sphere radius="0.05"/></geometry></collision></link>
<joint name="normal" type="prismatic"><parent link="base"/><child link="stage"/>
<origin xyz="0.09 0.01 0"/><axis xyz="1 0 0"/>
<limit lower="-0.05" upper="0.3" effort="10" velocity="1"/></joint>
<joint name="tangent" type="prismatic"><parent link="stage"/><child link="tool"/>
<axis xyz="0 1 0"/><limit lower="-0.2" upper="0" effort="10" velocity="1"/></joint>
</robot>""")
    robot = eik.RobotModel(str(path))
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.configure_collision_constraint(min_distance=0.05, max_constraints=1)
    task = solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSITION)
    task.allow_min_error_fallback = False
    runtime = solver.runtime_config()
    runtime.weighted_fallback_enabled = False
    solver.configure_runtime(runtime)
    return solver


@pytest.mark.parametrize("multi_target", [False, True])
@pytest.mark.parametrize("declared_floor", [False, True])
@pytest.mark.parametrize(
    "restriction", ["locked", "excluded", "integration_zero", "velocity_limit", "hard_row"]
)
def test_collision_escape_preserves_hard_joint_restrictions(
    escape_solver, multi_target, declared_floor, restriction
):
    solver = escape_solver
    if declared_floor:
        # The geometric overlap is explicitly permitted for this pair. A
        # stalled task must not try to escape its raw negative distance.
        solver.set_collision_pair_min_distance("base", "tool", -0.02)
    options = eik.PositionStepOptions()
    options.dt = 0.01
    options.max_steps = 1
    if restriction == "locked":
        options.locked_joint_indices = [0]
    elif restriction == "excluded":
        options.excluded_joint_indices = [0]
    elif restriction == "integration_zero":
        options.integration_zero_velocity_indices = [0]
    elif restriction == "velocity_limit":
        solver.set_joint_velocity_limit(0, 0.0)
    else:
        solver.set_linear_velocity_constraints(np.array([[1.0, 0.0]]), np.zeros(1), np.zeros(1))
    options.primary_allow_min_error_fallback = False
    options.record_velocity_candidates = True
    target = np.eye(4)
    target[:3, 3] = [0.13, 0.01, 0.0]
    initial = np.zeros(2)
    if multi_target:
        result = solver.solve_position_step(
            initial, [eik.TaskTarget("move", target, 10.0, 10.0)], options
        )
    else:
        result = solver.solve_position_step(initial, target, "move", options)
    assert result.velocity_candidates
    assert abs(result.velocity_candidates[0].integrated_velocity[0]) <= 1e-12
    assert abs(result.q_solution[0]) <= 1e-12
    assert result.stall_escape_count == 0
    assert np.isfinite(result.q_solution).all()


@pytest.mark.parametrize("multi_target", [False, True])
def test_noncollision_desaturation_retains_feasible_inward_motion(tmp_path, multi_target):
    links = [f'<link name="link{i}"/>' for i in range(5)]
    joints = [f"""<joint name="slide{i}" type="prismatic">
<parent link="link{i}"/><child link="link{i + 1}"/><axis xyz="1 0 0"/>
<limit lower="-2" upper="0" velocity="1" effort="1"/></joint>""" for i in range(4)]
    path = tmp_path / "desaturation_stage.urdf"
    path.write_text('<robot name="desaturation">' + "".join(links + joints) + "</robot>")
    robot = eik.RobotModel(str(path))
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.add_frame_task("move", "link4", eik.TaskType.FRAME_POSE)
    initial = np.full(4, -1e-4)
    target = np.eye(4)
    target[0, 3] = initial.sum() + 0.04
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.primary_allow_min_error_fallback = False
    if multi_target:
        result = solver.solve_position_step(
            initial, [eik.TaskTarget("move", target, 10.0, 10.0)], options
        )
    else:
        result = solver.solve_position_step(initial, target, "move", options)
    assert result.stall_escape_count > 0
    assert np.all(result.q_solution < initial)
    assert np.max(np.abs((result.q_solution - initial) / solver.dt)) <= 1.0


@pytest.mark.parametrize("multi_target", [False, True])
def test_inconsistent_original_hard_rows_hold_instead_of_escaping(escape_solver, multi_target):
    solver = escape_solver
    # Each row is a valid interval, but their intersection is empty: xdot
    # cannot be both >=0.1 and <=-0.1. No projected escape can satisfy them.
    solver.set_linear_velocity_constraints(
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        np.array([0.1, -1.0]),
        np.array([1.0, -0.1]),
    )
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.primary_allow_min_error_fallback = False
    target = np.eye(4)
    target[:3, 3] = [0.13, 0.01, 0.0]
    initial = np.zeros(2)
    if multi_target:
        result = solver.solve_position_step(
            initial, [eik.TaskTarget("move", target, 10.0, 10.0)], options
        )
    else:
        result = solver.solve_position_step(initial, target, "move", options)
    np.testing.assert_allclose(result.q_solution, initial, atol=1e-12, rtol=0)
    assert result.stall_escape_count == 0
    assert result.status in {eik.SolverStatus.INFEASIBLE, eik.SolverStatus.NUMERICAL_ERROR}


def test_priority_rejected_desaturation_reports_hold_without_accepted_escape(tmp_path):
    links = [f'<link name="link{i}"/>' for i in range(5)]
    joints = [f"""<joint name="slide{i}" type="prismatic">
<parent link="link{i}"/><child link="link{i + 1}"/><axis xyz="1 0 0"/>
<limit lower="-2" upper="0" velocity="1" effort="1"/></joint>""" for i in range(4)]
    path = tmp_path / "priority_desaturation.urdf"
    path.write_text('<robot name="priority">' + "".join(links + joints) + "</robot>")
    robot = eik.RobotModel(str(path))
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    position = solver.add_frame_task("position", "link4", eik.TaskType.FRAME_POSE)
    position.priority = 0
    orientation = solver.add_frame_task("orientation", "link4", eik.TaskType.FRAME_ORIENTATION)
    orientation.priority = 1
    initial = np.full(4, -1e-4)
    target = np.eye(4)
    target[0, 3] = initial.sum() + 0.04
    orientation_target = target.copy()
    angle = 0.1
    orientation_target[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.primary_allow_min_error_fallback = False
    result = solver.solve_position_step(
        initial,
        [
            eik.TaskTarget("position", target, 10.0, 1.0),
            eik.TaskTarget("orientation", orientation_target, 1.0, 3.0),
        ],
        options,
    )
    np.testing.assert_allclose(result.q_solution, initial, atol=1e-12, rtol=0)
    assert result.stall_escape_count == 0
    assert result.status == eik.SolverStatus.NO_PROGRESS
    assert result.position_step_hold_active
