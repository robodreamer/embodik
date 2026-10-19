"""A lower preference must preserve the physical primary motion near a limit."""

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("multiple_targets", [False, True])
def test_full_pose_keeps_physical_primary_image_near_lower_limit(tmp_path, multiple_targets):
    urdf = tmp_path / "opposed_sliders.urdf"
    urdf.write_text(
        '<robot name="opposed_sliders"><link name="base"/>'
        '<link name="middle"/><link name="tool"/>'
        '<joint name="first" type="prismatic"><parent link="base"/>'
        '<child link="middle"/><axis xyz="-1 0 0"/>'
        '<limit lower="0" upper="1" velocity="10" effort="1"/></joint>'
        '<joint name="second" type="prismatic"><parent link="middle"/>'
        '<child link="tool"/><axis xyz="1 0 0"/>'
        '<limit lower="-1" upper="1" velocity="10" effort="1"/></joint>'
        "</robot>"
    )
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.set_acceleration_limits(np.full(robot.nv, 15.0))
    solver.enable_acceleration_limits(True)

    pose = solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSE)
    pose.priority = 0
    pose.solve_mode = eik.TaskSolveMode.SCALE
    pose.allow_min_error_fallback = True
    posture = solver.add_posture_task("preference")
    posture.priority = 2
    posture.weight = 1.0
    posture.set_controlled_joint_indices([0])
    posture.set_target_configuration(np.array([0.3, 0.0]))
    posture.solve_mode = eik.TaskSolveMode.SCALE

    initial = np.array([0.0002, 0.0])
    robot.update_configuration(initial)
    target = robot.get_frame_pose("tool").homogeneous()
    target[0, 3] += 0.01
    pose.set_target_pose(target[:3, 3], target[:3, :3])
    options = eik.PositionStepOptions()
    options.dt = solver.dt
    options.max_steps = 1
    options.position_gain = 10.0
    options.orientation_gain = 10.0
    options.primary_solve_mode = eik.TaskSolveMode.SCALE
    options.primary_allow_min_error_fallback = True
    options.adaptive_dt = False

    if multiple_targets:
        result = solver.solve_position_step(
            initial, [eik.TaskTarget("move", target, 10.0, 10.0)], options
        )
    else:
        result = solver.solve_position_step(initial, target, "move", options)

    applied = (np.asarray(result.q_solution) - initial) / options.dt
    assert not any(result.task_used_fallback), (
        result.task_modes_effective,
        result.task_scales,
        applied,
    )
    # Both joints remain free: the first can move inward while the second
    # compensates to preserve the primary frame's physical [-1, 1] Jacobian.
    assert result.status == eik.SolverStatus.SUCCESS
    assert applied[0] > 1e-4
    assert np.array([-1.0, 1.0]) @ applied > 1e-3
    assert np.max(np.abs(applied)) <= 15.0 * options.dt + 1e-9
    assert result.q_solution[0] >= 0.0
    robot.update_configuration(result.q_solution)
    final_position = robot.get_frame_pose("tool").translation
    assert np.linalg.norm(final_position - target[:3, 3]) < 0.01 - 1e-6
    np.testing.assert_allclose(result.joint_velocities, applied, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(
        solver.get_previous_joint_velocities(), applied, atol=1e-10, rtol=0.0
    )
