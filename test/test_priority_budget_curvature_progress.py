"""Tight finite budgets must not throttle a feasible curved redundant motion."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import embodik as eik


@pytest.mark.parametrize(
    "mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.SCALE_ELASTIC, eik.TaskSolveMode.MIN_ERROR]
)
def test_redundant_sliders_support_full_rate_rotation_inside_position_budget(tmp_path, mode):
    path = tmp_path / "curved_budget.urdf"
    path.write_text("""<robot name="curved_budget"><link name="world"/>
<link name="x"/><joint name="x" type="prismatic"><parent link="world"/><child link="x"/>
<axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="100" velocity="2"/></joint>
<link name="y"/><joint name="y" type="prismatic"><parent link="x"/><child link="y"/>
<axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="100" velocity="2"/></joint>
<link name="yaw"/><joint name="yaw" type="revolute"><parent link="y"/><child link="yaw"/>
<axis xyz="0 0 1"/><limit lower="-2" upper="2" effort="100" velocity="1"/></joint>
<link name="tool"/><joint name="offset" type="fixed"><parent link="yaw"/><child link="tool"/>
<origin xyz="1 0 0"/></joint></robot>""")
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.set_linear_velocity_constraints(np.eye(3), np.full(3, -2.0), np.full(3, 2.0))
    turn = solver.add_frame_task("turn", "tool", eik.TaskType.FRAME_ORIENTATION)
    turn.priority, turn.solve_mode = 0, mode
    hold = solver.add_frame_task("hold", "tool", eik.TaskType.FRAME_POSITION)
    hold.priority, hold.solve_mode = 1, eik.TaskSolveMode.MIN_ERROR
    pose = eik.SE3(
        rotation=Rotation.from_rotvec([0.0, 0.0, 1.0]).as_matrix(),
        translation=np.array([1.0, 0.0, 0.0]),
    )
    target = eik.TaskTarget.from_se3("turn", pose, 0.0, 10.0)
    protected = eik.TaskTarget.from_se3("hold", pose, 1.0, 0.0)
    protected.priority_position_tolerance = 0.002
    options = eik.PositionStepOptions()
    options.dt, options.max_steps = 0.02, 1
    options.per_target_priority_budgets = True
    options.primary_solve_mode = mode
    q = np.array([-0.001999, 0.0, 0.0])
    for _ in range(100):
        before = q.copy()
        result = solver.solve_position_step(q, [target, protected], options)
        assert solver.get_linear_velocity_constraint_rows() == 3
        assert not solver.runtime_config().refine_partial_scale
        q = np.asarray(result.q_solution).copy()
        robot.update_configuration(q)
        assert (
            np.linalg.norm(np.asarray(robot.get_frame_pose("tool").translation) - [1.0, 0.0, 0.0])
            <= 0.002 + 1e-9
        )
        assert np.all(np.abs(q - before) / 0.02 <= np.array([2.0, 2.0, 1.0]) + 1e-9)
    # An exact bounded witness exists: x=1-cos(yaw), y=-sin(yaw).
    # At 1 rad/s this command has ample time to settle in 100 control ticks.
    assert q[2] >= 0.99
