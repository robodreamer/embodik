"""Certified explicit tasks retain their physical Jacobians at joint limits."""

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


def _build_case(tmp_path, axis_sign, near_lower, mode, certified):
    urdf_path = tmp_path / 'physical_joint_limit.urdf'
    urdf_path.write_text(f'''<robot name="physical_joint_limit">
<link name="world"/><link name="moving"/>
<joint name="slide" type="prismatic"><parent link="world"/><child link="moving"/><axis xyz="{axis_sign} 0 0"/><limit lower="{POSITION_LOWER}" upper="{POSITION_UPPER}" effort="{JOINT_EFFORT}" velocity="{VELOCITY_LIMIT}"/></joint></robot>''')
    robot = eik.RobotModel(str(urdf_path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = STEP_DT
    solver.enable_elastic_band(ELASTIC_MAXIMUM_EXPANSION)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.enable_velocity_task_mode_certification(certified)
    task = solver.add_frame_task('translation', 'moving', eik.TaskType.FRAME_POSITION)
    task.solve_mode = mode
    task.allow_min_error_fallback = False
    configuration = np.array([
        POSITION_LOWER + INITIAL_LIMIT_MARGIN if near_lower
        else POSITION_UPPER - INITIAL_LIMIT_MARGIN,
    ])
    inward_speed = REQUESTED_JOINT_SPEED if near_lower else -REQUESTED_JOINT_SPEED
    task.set_target_position_velocity(np.array([axis_sign * inward_speed, 0.0, 0.0]))
    return robot, solver, configuration, inward_speed


@pytest.mark.parametrize('mode', [
    eik.TaskSolveMode.MIN_ERROR, eik.TaskSolveMode.SCALE, eik.TaskSolveMode.SCALE_ELASTIC,
])
@pytest.mark.parametrize('near_lower', [False, True])
@pytest.mark.parametrize('axis_sign', [-1.0, 1.0])
@pytest.mark.parametrize('certified', [False, True])
def test_physical_inward_motion_is_retained_for_certified_tasks(
    tmp_path, mode, near_lower, axis_sign, certified,
):
    robot, solver, configuration, inward_speed = _build_case(
        tmp_path, axis_sign, near_lower, mode, certified,
    )
    interior = np.full(robot.nv, (POSITION_LOWER + POSITION_UPPER) / 2)
    interior_result = solver.solve_velocity(interior, apply_limits=True)
    interior_speed = np.asarray(interior_result.joint_velocities)[0]
    assert abs(interior_speed) > MINIMUM_ACHIEVED_FRACTION * abs(inward_speed)
    result = solver.solve_velocity(configuration, apply_limits=True)
    velocity = np.asarray(result.joint_velocities)
    clipped_by_legacy_heuristic = (near_lower and axis_sign < 0.0) or (
        not near_lower and axis_sign > 0.0
    )
    expected_speed = interior_speed if certified or not clipped_by_legacy_heuristic else 0.0
    np.testing.assert_allclose(velocity, [expected_speed], atol=NUMERIC_TOLERANCE, rtol=0.0)
    # The scalar physical derivative is axis_sign; an inward velocity is valid
    # regardless of the sign of the task-space Jacobian coefficient.
    if certified:
        assert axis_sign * velocity[0] == pytest.approx(
            axis_sign * interior_speed, abs=NUMERIC_TOLERANCE,
        )
    candidate = configuration + STEP_DT * velocity
    assert POSITION_LOWER - NUMERIC_TOLERANCE <= candidate[0] <= (
        POSITION_UPPER + NUMERIC_TOLERANCE
    )
    assert abs(velocity[0]) <= VELOCITY_LIMIT + NUMERIC_TOLERANCE


@pytest.mark.parametrize('mode', [
    eik.TaskSolveMode.MIN_ERROR, eik.TaskSolveMode.SCALE, eik.TaskSolveMode.SCALE_ELASTIC,
])
@pytest.mark.parametrize('near_lower', [False, True])
def test_certified_physical_jacobian_keeps_outward_position_bounds(tmp_path, mode, near_lower):
    robot, solver, configuration, inward_speed = _build_case(
        tmp_path, -1.0, near_lower, mode, True,
    )
    task = solver.get_task('translation')
    task.set_target_position_velocity(np.array([inward_speed, 0.0, 0.0]))
    result = solver.solve_velocity(configuration, apply_limits=True)
    candidate = configuration + STEP_DT * np.asarray(result.joint_velocities)
    assert POSITION_LOWER - NUMERIC_TOLERANCE <= candidate[0] <= (
        POSITION_UPPER + NUMERIC_TOLERANCE
    )
