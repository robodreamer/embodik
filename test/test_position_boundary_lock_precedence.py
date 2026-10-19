"""A valid locked joint at a true boundary must stay fixed during integration."""

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.SCALE_ELASTIC])
@pytest.mark.parametrize("lock_kind", ["explicit", "task_exclusion"])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("outside_seed", [False, True])
def test_boundary_lock_stays_fixed_with_feasible_other_joint(
    tmp_path, mode, lock_kind, direction, outside_seed
):
    lower, upper = (-2, 0) if direction > 0 else (0, 2)
    urdf = tmp_path / "locked_slides.urdf"
    urdf.write_text(f"""<robot name="locked_slides"><link name="base"/>
<link name="middle"/><link name="tool"/>
<joint name="first" type="prismatic"><parent link="base"/><child link="middle"/>
<axis xyz="1 0 0"/><limit lower="{lower}" upper="{upper}" velocity="1" effort="1"/></joint>
<joint name="second" type="prismatic"><parent link="middle"/><child link="tool"/>
<axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
</robot>""")
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    task = solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSITION)
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.position_gain = 10.0
    options.primary_solve_mode = mode
    options.primary_allow_min_error_fallback = False
    if lock_kind == "explicit":
        options.locked_joint_indices = [0]
    else:
        task.set_excluded_joint_indices([0])
    target = np.eye(4)
    target[0, 3] = direction * 0.01
    seed = np.zeros(2)
    if outside_seed:
        seed[0] = direction * 2e-4
    result = solver.solve_position_step(seed, target, "move", options)
    if outside_seed:
        assert lower <= result.q_solution[0] <= upper, result.status_message
    else:
        assert abs(result.q_solution[0]) < 1e-12, result.status_message
    assert direction * result.q_solution[1] > 1e-3, result.status_message


@pytest.mark.parametrize("mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.SCALE_ELASTIC])
@pytest.mark.parametrize("upper_boundary", [False, True])
@pytest.mark.parametrize("inset_distance", [0.0, 2e-5])
@pytest.mark.parametrize("inward_target_distance", [0.0, 1e-5])
def test_unlocked_stationary_joint_inside_margin_preserves_coordinated_motion(
    tmp_path, mode, upper_boundary, inset_distance, inward_target_distance
):
    """A margin must cap motion, without injecting a new task-space command."""
    lower, upper = (-2, 0) if upper_boundary else (0, 2)
    urdf = tmp_path / "orthogonal_slides.urdf"
    urdf.write_text(f"""<robot name="orthogonal_slides"><link name="base"/>
<link name="middle"/><link name="tool"/>
<joint name="first" type="prismatic"><parent link="base"/><child link="middle"/>
<axis xyz="1 0 0"/><limit lower="{lower}" upper="{upper}" velocity="1" effort="1"/></joint>
<joint name="second" type="prismatic"><parent link="middle"/><child link="tool"/>
<axis xyz="0 1 0"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
</robot>""")
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSITION)
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.position_gain = 10.0
    options.primary_solve_mode = mode
    options.primary_allow_min_error_fallback = False
    seed = np.array([-inset_distance if upper_boundary else inset_distance, 0.0])
    target = np.eye(4)
    inward_sign = -1.0 if upper_boundary else 1.0
    target[:2, 3] = [seed[0] + inward_sign * inward_target_distance, 0.01]
    result = solver.solve_position_step(seed, target, "move", options)
    inward_motion = inward_sign * (result.q_solution[0] - seed[0])
    assert -1e-12 <= inward_motion <= inward_target_distance + 1e-12
    if inward_target_distance == 0.0:
        assert result.q_solution[0] == pytest.approx(seed[0], abs=1e-12)
    assert result.q_solution[1] > 1e-3, result.status_message
