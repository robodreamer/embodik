"""Near-limit bounds must retain an orientation task's legal inward direction."""

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("axis", [-1.0, 1.0])
@pytest.mark.parametrize("multiple_targets", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("full_pose", [False, True])
def test_orientation_retains_inward_motion_at_limit(
    tmp_path, axis, multiple_targets, adaptive, full_pose
):
    lower, upper = (-2, 0) if axis > 0 else (0, 2)
    urdf = tmp_path / "coupled_yaw.urdf"
    urdf.write_text(
        f"""<robot name="coupled_yaw"><link name="base"/><link name="middle"/><link name="tool"/>
<joint name="first" type="revolute"><parent link="base"/><child link="middle"/>
<axis xyz="0 0 {axis}"/><limit lower="{lower}" upper="{upper}" velocity="1" effort="1"/></joint>
<joint name="second" type="revolute"><parent link="middle"/><child link="tool"/>
<axis xyz="0 0 {axis}"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint></robot>"""
    )
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01 if adaptive else 0.05
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.add_frame_task(
        "rotate", "tool", eik.TaskType.FRAME_POSE if full_pose else eik.TaskType.FRAME_ORIENTATION
    )
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.position_gain = options.orientation_gain = 10.0
    options.adaptive_dt = adaptive
    options.adaptive_dt_max_scale = 5.0
    options.adaptive_dt_reference_distance = 0.002
    options.locked_joint_indices = [1]
    options.primary_solve_mode = eik.TaskSolveMode.SCALE
    options.primary_allow_min_error_fallback = False
    q = np.array([-axis * 1e-4, 0.0])
    target = np.eye(4)
    angle = -0.01
    target[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    if multiple_targets:
        result = solver.solve_position_step(
            q, [eik.TaskTarget("rotate", target, 10.0, 10.0)], options
        )
    else:
        result = solver.solve_position_step(q, target, "rotate", options)
    actual = np.asarray(result.q_solution)
    assert abs(actual[1]) <= 1e-12
    assert -axis * (actual[0] - q[0]) > 1e-4
    assert abs(angle - axis * actual.sum()) < abs(angle - axis * q.sum())


@pytest.mark.parametrize("axis", [-1.0, 1.0])
@pytest.mark.parametrize("allow_fallback", [False, True])
def test_full_pose_recovery_respects_authorized_min_error(tmp_path, axis, allow_fallback):
    """An unreachable roll must not discard an authorized inward yaw fallback."""
    lower, upper = (-2, 0) if axis > 0 else (0, 2)
    urdf = tmp_path / "yaw_with_unreachable_roll.urdf"
    urdf.write_text(
        f"""<robot name="coupled_yaw"><link name="base"/><link name="middle"/><link name="tool"/>
<joint name="first" type="revolute"><parent link="base"/><child link="middle"/>
<axis xyz="0 0 {axis}"/><limit lower="{lower}" upper="{upper}" velocity="1" effort="1"/></joint>
<joint name="second" type="revolute"><parent link="middle"/><child link="tool"/>
<axis xyz="0 0 {axis}"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint></robot>"""
    )
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.05
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    config = solver.runtime_config()
    config.enable_auto_task_layout = False
    config.weighted_fallback_enabled = False
    solver.configure_runtime(config)
    solver.add_frame_task("pose", "tool", eik.TaskType.FRAME_POSE)
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.adaptive_dt = False
    options.position_gain = options.orientation_gain = 10.0
    options.locked_joint_indices = [1]
    options.primary_solve_mode = eik.TaskSolveMode.SCALE
    options.primary_allow_min_error_fallback = allow_fallback
    q = np.array([-axis * 1e-4, 0.0])
    yaw, roll = -0.01, 0.02
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    target = np.eye(4)
    target[:3, :3] = rz @ rx
    result = solver.solve_position_step(q, target, "pose", options)
    actual = np.asarray(result.q_solution)
    assert abs(actual[1]) <= 1e-12
    q_min, q_max = map(np.asarray, robot.get_joint_limits())
    assert np.all(actual >= q_min) and np.all(actual <= q_max)
    if allow_fallback:
        assert -axis * (actual[0] - q[0]) > 1e-4, (
            result.status,
            result.status_message,
            result.task_scales,
            result.task_modes_effective,
            result.task_used_fallback,
        )
        assert result.task_scales[0] == -1.0
        assert result.task_modes_effective[0] == eik.TaskSolveMode.MIN_ERROR
        assert result.task_used_fallback[0]
        errors = []
        for configuration in (q, actual):
            robot.update_configuration(configuration)
            rotation = np.asarray(robot.get_frame_pose("tool").rotation)
            cosine = (np.trace(target[:3, :3].T @ rotation) - 1.0) / 2.0
            errors.append(np.arccos(np.clip(cosine, -1.0, 1.0)))
        assert errors[1] < errors[0] - 1e-4
    else:
        np.testing.assert_allclose(actual, q, atol=1e-12, rtol=0)
