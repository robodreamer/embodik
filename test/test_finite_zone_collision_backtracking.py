"""A checked fraction must respect both curved clearance and a tight pose zone."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import embodik as eik


def _case(tmp_path, direction, weighted, mandatory=False):
    path = tmp_path / "curved_fraction.urdf"
    path.write_text(f"""<robot name="curved_fraction"><link name="world"/>
<link name="obstacle"><collision><geometry><sphere radius="0.5"/></geometry></collision></link>
<joint name="obstacle_fixed" type="fixed"><parent link="world"/><child link="obstacle"/>
<origin xyz="-1 0 0"/></joint>
<link name="x"/><joint name="x" type="prismatic"><parent link="world"/><child link="x"/>
<axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="2" effort="100"/></joint>
<link name="y"/><joint name="y" type="prismatic"><parent link="x"/><child link="y"/>
<axis xyz="0 1 0"/><limit lower="-2" upper="2" velocity="2" effort="100"/></joint>
<link name="yaw"/><joint name="yaw" type="revolute"><parent link="y"/><child link="yaw"/>
<axis xyz="0 0 1"/><limit lower="-2" upper="2" velocity="1" effort="100"/></joint>
<link name="tool"><collision><origin xyz="0.1 {0.3 * direction} 0"/>
<geometry><sphere radius="0.5"/></geometry></collision></link>
<joint name="offset" type="fixed"><parent link="yaw"/><child link="tool"/>
<origin xyz="1 0 0"/></joint></robot>""")
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    seed = np.array([-0.001999, 0.0, 0.0])
    floor = np.hypot(2.1 + seed[0], 0.3) - 1.0
    solver.configure_collision_constraint(floor, [], [], True, 4)
    solver.set_collision_trajectory_floor_anchor(seed, 1e-9)
    C = (
        np.array([[1.0, 0.0, 0.0], [0.0, direction, 0.0]])
        if mandatory
        else np.array([[1.0, 0.0, 0.0]])
    )
    solver.set_linear_velocity_constraints(
        C,
        np.array([0.0, 0.1]) if mandatory else np.zeros(1),
        np.array([0.0, 2.0]) if mandatory else np.zeros(1),
    )
    config = solver.runtime_config()
    config.weighted_fallback_enabled = weighted
    solver.configure_runtime(config)
    hold = solver.add_frame_task("hold", "tool", eik.TaskType.FRAME_POSITION)
    hold.priority, hold.solve_mode = 0, eik.TaskSolveMode.MIN_ERROR
    turn = solver.add_frame_task("turn", "tool", eik.TaskType.FRAME_ORIENTATION)
    turn.priority, turn.solve_mode = 1, eik.TaskSolveMode.MIN_ERROR
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec([0, 0, direction]).as_matrix()
    pose[:3, 3] = [1, 0, 0]
    protected = eik.TaskTarget("hold", pose, 1.0, 0.0)
    protected.priority_position_tolerance = 0.002
    target = eik.TaskTarget("turn", pose, 0.0, 10.0)
    options = eik.PositionStepOptions()
    options.dt, options.max_steps = solver.dt, 1
    options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
    options.primary_allow_min_error_fallback = False
    options.per_target_priority_budgets = True
    return solver, options, [protected, target], seed, floor


def _check(q, before, direction, floor):
    theta = q[2]
    tool = np.array([q[0] + np.cos(theta), q[1] + np.sin(theta)])
    assert np.linalg.norm(tool - [1, 0]) <= 0.002 + 1e-9
    center = np.array(
        [
            1 + q[0] + 1.1 * np.cos(theta) - 0.3 * direction * np.sin(theta),
            q[1] + 1.1 * np.sin(theta) + 0.3 * direction * np.cos(theta),
        ]
    )
    assert np.linalg.norm(center) - 1 >= floor - 1e-9
    assert abs(q[0] - before[0]) <= 1e-10
    assert np.all(abs(q - before) <= np.array([2, 2, 1]) * 0.02 + 1e-9)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("weighted", [False, True])
def test_finite_zone_collision_fraction_retains_hard_rows(tmp_path, direction, weighted):
    solver, options, targets, seed, floor = _case(tmp_path, direction, weighted)
    # Independent feasible endpoint; never supplied to the solver.
    witness = seed + np.array([0, direction * 2.2e-5, direction * 2e-5])
    _check(witness, seed, direction, floor)
    result = solver.solve_position_step(seed, targets, options)
    q = np.array(result.q_solution)
    _check(q, seed, direction, floor)
    assert solver.get_linear_velocity_constraint_rows() == 1
    if weighted:
        assert direction * q[2] > 1e-5
    else:
        np.testing.assert_allclose(q, seed, atol=1e-10, rtol=0)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_backtracking_does_not_erase_mandatory_velocity(tmp_path, direction):
    solver, options, targets, seed, _ = _case(tmp_path, direction, True, mandatory=True)
    result = solver.solve_position_step(seed, targets, options)
    assert solver.get_linear_velocity_constraint_rows() == 2
    assert result.status != eik.SolverStatus.SUCCESS
