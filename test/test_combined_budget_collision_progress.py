"""A finite pose budget and collision floor must retain feasible redundancy."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import embodik as eik


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize(
    "mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.SCALE_ELASTIC, eik.TaskSolveMode.MIN_ERROR]
)
def test_turn_with_redundant_clearance_and_finite_position_budget(tmp_path, mode, direction):
    path = tmp_path / "combined_clearance.urdf"
    path.write_text(f"""<robot name="combined_clearance"><link name="world"/>
<link name="obstacle"><collision><geometry><sphere radius="0.5"/></geometry></collision></link>
<joint name="obstacle_slide" type="prismatic"><parent link="world"/><child link="obstacle"/>
<origin xyz="-1 0 0"/><axis xyz="-1 0 0"/><limit lower="-2" upper="2" effort="100" velocity="2"/></joint>
<link name="x"/><joint name="x" type="prismatic"><parent link="world"/><child link="x"/>
<axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="100" velocity="2"/></joint>
<link name="y"/><joint name="y" type="prismatic"><parent link="x"/><child link="y"/>
<axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="100" velocity="2"/></joint>
<link name="yaw"/><joint name="yaw" type="revolute"><parent link="y"/><child link="yaw"/>
<axis xyz="0 0 1"/><limit lower="-2" upper="2" effort="100" velocity="1"/></joint>
<link name="tool"><collision><origin xyz="0.1 {0.3 * direction} 0"/>
<geometry><sphere radius="0.5"/></geometry></collision></link>
<joint name="offset" type="fixed"><parent link="yaw"/><child link="tool"/>
<origin xyz="1 0 0"/></joint></robot>""")
    robot = eik.RobotModel(str(path), floating_base=False)
    assert list(robot.get_joint_names()) == ["obstacle_slide", "x", "y", "yaw"]
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    radius_squared = (2.1 - 0.001999) ** 2 + 0.3**2
    floor = np.sqrt(radius_squared) - 1.0
    solver.configure_collision_constraint(floor, [], [], True, 4)
    q = np.array([0.0, -0.001999, 0.0, 0.0])
    solver.set_collision_trajectory_floor_anchor(q, 1e-9)
    velocity_caps = np.array([2.0, 2.0, 2.0, 1.0])

    def check_configuration(configuration):
        robot.update_configuration(configuration)
        position = np.asarray(robot.get_frame_pose("tool").translation)
        assert np.linalg.norm(position - [1.0, 0.0, 0.0]) <= 0.002 + 1e-9
        distances = solver.evaluate_collision_pair_distances(configuration)
        assert len(distances) == 1
        assert distances[0].distance >= floor - 1e-9
        assert np.all(np.abs(configuration) <= 2.0 + 1e-9)

    # Independently exhibit a bounded nonlinear route. Keep the tool position
    # fixed and move the obstacle's redundant slider to retain exact clearance.
    # The witness only checks feasibility; it is never supplied to the solver.
    previous = q.copy()
    for theta in direction * np.linspace(0.0, 1.0, 101):
        collision_y = 0.1 * np.sin(theta) + 0.3 * direction * np.cos(theta)
        obstacle_shift = np.sqrt(radius_squared - collision_y**2) - (
            1.998001 + 0.1 * np.cos(theta) - 0.3 * direction * np.sin(theta)
        )
        witness = np.array([obstacle_shift, 0.998001 - np.cos(theta), -np.sin(theta), theta])
        check_configuration(witness)
        assert np.all(np.abs(witness - previous) <= velocity_caps * 0.01 + 1e-9)
        previous = witness

    turn = solver.add_frame_task("turn", "tool", eik.TaskType.FRAME_ORIENTATION)
    turn.priority, turn.solve_mode = 0, mode
    hold = solver.add_frame_task("hold", "tool", eik.TaskType.FRAME_POSITION)
    hold.priority, hold.solve_mode = 1, eik.TaskSolveMode.MIN_ERROR
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec([0.0, 0.0, direction]).as_matrix()
    pose[:3, 3] = [1.0, 0.0, 0.0]
    target = eik.TaskTarget("turn", pose, 0.0, 10.0)
    protected = eik.TaskTarget("hold", pose, 1.0, 0.0)
    protected.priority_position_tolerance = 0.002
    options = eik.PositionStepOptions()
    options.dt, options.max_steps = 0.02, 1
    options.primary_solve_mode = mode
    options.per_target_priority_budgets = True
    options.stall_recovery = False
    for _ in range(100):
        before = q.copy()
        result = solver.solve_position_step(q, [target, protected], options)
        q = np.asarray(result.q_solution).copy()
        assert solver.get_linear_velocity_constraint_rows() == 0
        assert not solver.runtime_config().refine_partial_scale
        check_configuration(q)
        assert np.all(np.abs(q - before) <= velocity_caps * options.dt + 1e-9)
    assert direction * q[3] >= 0.99
