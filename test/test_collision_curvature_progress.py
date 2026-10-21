"""Redundant translation should support a rotation tangent to a clearance floor."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import embodik as eik


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("lock_slide", [False, True])
@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize(
    "mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.SCALE_ELASTIC, eik.TaskSolveMode.MIN_ERROR]
)
def test_finite_collision_curvature_uses_redundancy_instead_of_holding(
    tmp_path, mode, single, lock_slide, direction
):
    urdf = tmp_path / "collision_curvature.urdf"
    urdf.write_text("""<robot name="collision_curvature"><link name="world"/>
<link name="obstacle"><collision><geometry><sphere radius="0.5"/></geometry></collision></link>
<joint name="obstacle_fixed" type="fixed"><parent link="world"/><child link="obstacle"/><origin xyz="-1 0 0"/></joint>
<link name="slide"/><joint name="slide" type="prismatic"><parent link="world"/><child link="slide"/>
<axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="100" velocity="1"/></joint>
<link name="yaw"/><joint name="yaw" type="revolute"><parent link="slide"/><child link="yaw"/>
<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="100" velocity="1"/></joint>
<link name="tool"><collision><geometry><sphere radius="0.5"/></geometry></collision></link>
<joint name="offset" type="fixed"><parent link="yaw"/><child link="tool"/><origin xyz="1 0 0"/></joint>
</robot>""")
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.configure_collision_constraint(1.0, [], [], True, 4)
    q = np.zeros(2)
    solver.set_collision_trajectory_floor_anchor(q, 1e-9)
    task = solver.add_frame_task("turn", "tool", eik.TaskType.FRAME_ORIENTATION)
    task.solve_mode = mode
    target = np.eye(4)
    target[:3, :3] = Rotation.from_rotvec([0.0, 0.0, direction * 0.5]).as_matrix()
    options = eik.PositionStepOptions()
    options.dt, options.max_steps = 0.02, 1
    options.primary_solve_mode = mode
    if lock_slide:
        options.locked_joint_indices = [0]
    options.orientation_gain = 10.0
    options.stall_recovery = False
    solver.set_linear_velocity_constraints(
        np.array([[1.0, 0.0]]), np.array([-0.8]), np.array([0.8])
    )
    for _ in range(100):
        before = q.copy()
        result = (
            solver.solve_position_step(q, target, "turn", options)
            if single
            else solver.solve_position_step(q, [eik.TaskTarget("turn", target, 0.0, 10.0)], options)
        )
        assert solver.get_linear_velocity_constraint_rows() == 1
        assert not solver.runtime_config().refine_partial_scale
        q = np.asarray(result.q_solution).copy()
        distances = solver.evaluate_collision_pair_distances(q)
        assert len(distances) == 1
        assert distances[0].distance >= 1.0 - 1e-9
        assert np.all(abs(q - before) <= 0.02 + 1e-9)
    # A bounded witness exists: x=1-cos(yaw), which moves the tool away from
    # the obstacle while yaw advances at up to1rad/s. No manual reset required.
    if lock_slide:
        assert abs(q[1]) < 0.001
    else:
        assert direction * q[1] >= 0.49
