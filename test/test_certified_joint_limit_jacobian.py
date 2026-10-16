"""Cartesian frame tasks preserve physical inward motion near joint limits."""

from __future__ import annotations

import numpy as np
import pytest

import embodik as eik

STEP_DT = 0.02
POSITION_LOWER = 0.0
POSITION_UPPER = 1.0
INITIAL_LIMIT_MARGIN = 1e-4
VELOCITY_LIMIT = 0.2
REQUESTED_JOINT_SPEED = 0.1
JOINT_EFFORT = 100.0
NUMERIC_TOLERANCE = 1e-9
MINIMUM_ACHIEVED_FRACTION = 0.5
ELASTIC_MAXIMUM_EXPANSION = 0.0


def _build_case(tmp_path, axis_sign, near_lower, mode, full_pose=False):
    urdf_path = tmp_path / "physical_joint_limit.urdf"
    urdf_path.write_text(f"""<robot name="physical_joint_limit">
<link name="world"/><link name="moving"/>
<joint name="slide" type="prismatic"><parent link="world"/><child link="moving"/>
<axis xyz="{axis_sign} 0 0"/>
<limit lower="{POSITION_LOWER}" upper="{POSITION_UPPER}" effort="{JOINT_EFFORT}" velocity="{VELOCITY_LIMIT}"/>
</joint></robot>""")
    robot = eik.RobotModel(str(urdf_path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = STEP_DT
    config = solver.runtime_config()
    config.weighted_fallback_enabled = False
    config.enable_auto_task_layout = False
    solver.configure_runtime(config)
    solver.enable_elastic_band(ELASTIC_MAXIMUM_EXPANSION)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    task = solver.add_frame_task(
        "translation",
        "moving",
        eik.TaskType.FRAME_POSE if full_pose else eik.TaskType.FRAME_POSITION,
    )
    task.solve_mode = mode
    task.allow_min_error_fallback = False
    configuration = np.array(
        [
            (
                POSITION_LOWER + INITIAL_LIMIT_MARGIN
                if near_lower
                else POSITION_UPPER - INITIAL_LIMIT_MARGIN
            ),
        ]
    )
    inward_speed = REQUESTED_JOINT_SPEED if near_lower else -REQUESTED_JOINT_SPEED
    task.set_target_position_velocity(np.array([axis_sign * inward_speed, 0.0, 0.0]))
    return robot, solver, configuration, inward_speed


@pytest.mark.parametrize(
    "mode",
    [
        eik.TaskSolveMode.MIN_ERROR,
        eik.TaskSolveMode.SCALE,
        eik.TaskSolveMode.SCALE_ELASTIC,
    ],
)
@pytest.mark.parametrize("near_lower", [False, True])
@pytest.mark.parametrize("axis_sign", [-1.0, 1.0])
@pytest.mark.parametrize("full_pose", [False, True])
def test_near_limit_jacobian_clamp_preserves_inward_motion(
    tmp_path, mode, near_lower, axis_sign, full_pose
):
    robot, solver, configuration, inward_speed = _build_case(
        tmp_path, axis_sign, near_lower, mode, full_pose
    )
    interior = np.full(robot.nv, (POSITION_LOWER + POSITION_UPPER) / 2)
    interior_result = solver.solve_velocity(interior, apply_limits=True)
    interior_speed = np.asarray(interior_result.joint_velocities)[0]
    assert abs(interior_speed) > MINIMUM_ACHIEVED_FRACTION * abs(inward_speed)
    result = solver.solve_velocity(configuration, apply_limits=True)
    velocity = np.asarray(result.joint_velocities)
    np.testing.assert_allclose(velocity, [interior_speed], atol=NUMERIC_TOLERANCE, rtol=0.0)
    candidate = configuration + STEP_DT * velocity
    assert POSITION_LOWER - NUMERIC_TOLERANCE <= candidate[0] <= POSITION_UPPER + NUMERIC_TOLERANCE
    assert abs(velocity[0]) <= VELOCITY_LIMIT + NUMERIC_TOLERANCE


@pytest.mark.parametrize(
    "mode",
    [
        eik.TaskSolveMode.MIN_ERROR,
        eik.TaskSolveMode.SCALE,
        eik.TaskSolveMode.SCALE_ELASTIC,
    ],
)
@pytest.mark.parametrize("near_lower", [False, True])
@pytest.mark.parametrize("full_pose", [False, True])
def test_outward_velocity_request_stays_inside_position_bounds(
    tmp_path, mode, near_lower, full_pose
):
    _robot, solver, configuration, inward_speed = _build_case(
        tmp_path, -1.0, near_lower, mode, full_pose
    )
    task = solver.get_task("translation")
    task.set_target_position_velocity(np.array([inward_speed, 0.0, 0.0]))
    result = solver.solve_velocity(configuration, apply_limits=True)
    candidate = configuration + STEP_DT * np.asarray(result.joint_velocities)
    assert POSITION_LOWER - NUMERIC_TOLERANCE <= candidate[0] <= POSITION_UPPER + NUMERIC_TOLERANCE


@pytest.mark.parametrize("axis_sign", [-1.0, 1.0])
@pytest.mark.parametrize("near_lower", [False, True])
@pytest.mark.parametrize("multi_target", [False, True])
@pytest.mark.parametrize("full_pose", [False, True])
def test_position_step_can_leave_near_limit_inward(
    tmp_path, axis_sign, near_lower, multi_target, full_pose
):
    robot, solver, configuration, inward_speed = _build_case(
        tmp_path, axis_sign, near_lower, eik.TaskSolveMode.SCALE, full_pose
    )
    robot.update_configuration(configuration)
    target = np.eye(4)
    target[:3, 3] = np.asarray(robot.get_frame_pose("moving").translation)
    target[0, 3] += axis_sign * np.sign(inward_speed) * 0.01
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.position_gain = 10.0
    options.primary_solve_mode = eik.TaskSolveMode.SCALE
    options.primary_allow_min_error_fallback = False
    if multi_target:
        result = solver.solve_position_step(
            configuration, [eik.TaskTarget("translation", target, 10.0, 10.0)], options
        )
    else:
        result = solver.solve_position_step(configuration, target, "translation", options)
    motion = result.q_solution[0] - configuration[0]
    assert np.sign(inward_speed) * motion > 1e-3, result.status_message
    assert POSITION_LOWER <= result.q_solution[0] <= POSITION_UPPER


@pytest.mark.parametrize("axis_sign", [-1.0, 1.0])
@pytest.mark.parametrize("near_lower", [False, True])
def test_full_pose_inward_motion_with_uncombined_objectives(tmp_path, axis_sign, near_lower):
    _robot, solver, configuration, inward_speed = _build_case(
        tmp_path, axis_sign, near_lower, eik.TaskSolveMode.SCALE, True
    )
    orientation = solver.add_frame_task("orientation", "moving", eik.TaskType.FRAME_ORIENTATION)
    orientation.priority = solver.get_task("translation").priority
    orientation.solve_mode = eik.TaskSolveMode.SCALE
    # Different fallback policies force separate objective assembly.
    orientation.allow_min_error_fallback = True
    orientation.set_target_angular_velocity(np.zeros(3))
    result = solver.solve_velocity(configuration, apply_limits=True)
    velocity = np.asarray(result.joint_velocities)
    assert np.sign(inward_speed) * velocity[0] > MINIMUM_ACHIEVED_FRACTION * abs(inward_speed)
    assert len(result.task_modes_effective) == 2
    assert not result.weighted_fallback_used


@pytest.mark.parametrize("mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.SCALE_ELASTIC])
def test_full_pose_coupled_inward_motion_preserves_physical_jacobian(tmp_path, mode):
    """A column opposing the goal can be necessary for a feasible combined ray."""
    urdf_path = tmp_path / "coupled_slides.urdf"
    urdf_path.write_text("""<robot name="coupled_slides">
<link name="world"/><link name="first"/><link name="tool"/>
<joint name="first_slide" type="prismatic"><parent link="world"/><child link="first"/>
<axis xyz="1 0 0"/><limit lower="0" upper="1" effort="100" velocity="1"/></joint>
<joint name="second_slide" type="prismatic"><parent link="first"/><child link="tool"/>
<axis xyz="0.8944271909999159 0.4472135954999579 0"/>
<limit lower="-1" upper="1" effort="100" velocity="1"/></joint></robot>""")
    robot = eik.RobotModel(str(urdf_path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = STEP_DT
    config = solver.runtime_config()
    config.damping = 1e-12
    config.weighted_fallback_enabled = False
    config.enable_auto_task_layout = False
    solver.configure_runtime(config)
    # Isolate Jacobian filtering from deliberate pseudoinverse damping.
    solver.set_regularization_epsilon(1e-9)
    solver.enable_elastic_band(0.0)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    task = solver.add_frame_task("pose", "tool", eik.TaskType.FRAME_POSE)
    task.solve_mode = mode
    task.allow_min_error_fallback = False
    desired = np.array([-0.1, -0.1, 0.0])
    task.set_target_position_velocity(desired)
    configuration = np.array([INITIAL_LIMIT_MARGIN, 0.0])
    result = solver.solve_velocity(configuration, apply_limits=True)
    velocity = np.asarray(result.joint_velocities)
    physical_translation_jacobian = np.array(
        [[1.0, 2.0 / np.sqrt(5.0)], [0.0, 1.0 / np.sqrt(5.0)], [0.0, 0.0]]
    )
    achieved = physical_translation_jacobian @ velocity
    scale = achieved[0] / desired[0]
    assert scale > MINIMUM_ACHIEVED_FRACTION
    assert scale <= 1.0 + NUMERIC_TOLERANCE
    np.testing.assert_allclose(achieved, scale * desired, atol=NUMERIC_TOLERANCE, rtol=0.0)
    assert velocity[0] > 0.0
    candidate = configuration + STEP_DT * velocity
    assert np.all(candidate >= np.array([0.0, -1.0]) - NUMERIC_TOLERANCE)
    assert np.all(candidate <= np.array([1.0, 1.0]) + NUMERIC_TOLERANCE)
    assert not result.weighted_fallback_used


@pytest.mark.parametrize("near_lower", [False, True])
@pytest.mark.parametrize(
    "mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.SCALE_ELASTIC, eik.TaskSolveMode.MIN_ERROR]
)
def test_direct_posture_task_can_move_inward_from_either_limit(tmp_path, near_lower, mode):
    _robot, solver, q, inward = _build_case(tmp_path, 1.0, near_lower, mode)
    solver.clear_tasks()
    posture = solver.add_posture_task("guide")
    posture.solve_mode = mode
    posture.allow_min_error_fallback = False
    posture.set_target_velocity(np.array([inward]))
    result = solver.solve_velocity(q, apply_limits=True)
    assert np.sign(inward) * result.joint_velocities[0] > 0.5 * abs(inward)
    assert not result.weighted_fallback_used


def test_separate_posture_objective_preserves_inward_motion(tmp_path):
    _robot, solver, q, inward = _build_case(tmp_path, 1.0, False, eik.TaskSolveMode.SCALE)
    solver.clear_tasks()
    posture = solver.add_posture_task("guide")
    posture.allow_min_error_fallback = False
    posture.solve_mode = eik.TaskSolveMode.SCALE
    posture.set_target_velocity(np.array([inward]))
    orientation = solver.add_frame_task("orientation", "moving", eik.TaskType.FRAME_ORIENTATION)
    orientation.solve_mode = eik.TaskSolveMode.SCALE
    orientation.priority = posture.priority
    orientation.allow_min_error_fallback = True
    orientation.set_target_angular_velocity(np.zeros(3))
    result = solver.solve_velocity(q, apply_limits=True)
    assert result.joint_velocities[0] < -0.05
    assert not result.weighted_fallback_used
