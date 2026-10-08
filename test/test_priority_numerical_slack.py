"""Numerical merit slack is an absolute budget, not a control-tick allowance."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import embodik as eik

STEP_DT = 0.02
ORIENTATION_TOLERANCE = 0.002
NUMERICAL_MERIT_TOLERANCE = 1e-9
MEASUREMENT_ROUNDOFF = 1e-14
STEP_COUNT = 50
GOAL_DISTANCE = 0.15
SLIDER_SPEED = 0.01
YAW_SPEED = 1e-7
JOINT_LIMIT = 1.0
JOINT_EFFORT = 100.0
LEVER_LENGTH = 1.0
OUTSIDE_BUDGET_ERROR = 2 * ORIENTATION_TOLERANCE
URDF = f'''<robot name="numerical_merit_budget">
<link name="world"/><link name="slider"/>
<joint name="slide" type="prismatic"><parent link="world"/><child link="slider"/><axis xyz="0 1 0"/><limit lower="{-JOINT_LIMIT}" upper="{JOINT_LIMIT}" effort="{JOINT_EFFORT}" velocity="{SLIDER_SPEED}"/></joint>
<link name="wrist"/>
<joint name="yaw" type="revolute"><parent link="slider"/><child link="wrist"/><axis xyz="0 0 1"/><limit lower="{-JOINT_LIMIT}" upper="{JOINT_LIMIT}" effort="{JOINT_EFFORT}" velocity="{YAW_SPEED}"/></joint>
<link name="ee"/><joint name="offset" type="fixed"><parent link="wrist"/><child link="ee"/><origin xyz="{LEVER_LENGTH} 0 0"/></joint></robot>'''


@pytest.mark.parametrize('mode', [
    eik.TaskSolveMode.MIN_ERROR, eik.TaskSolveMode.SCALE, eik.TaskSolveMode.SCALE_ELASTIC,
])
@pytest.mark.parametrize('initial_error', [ORIENTATION_TOLERANCE, OUTSIDE_BUDGET_ERROR])
def test_priority_numerical_allowance_does_not_ratchet(tmp_path, mode, initial_error):
    urdf_path = tmp_path / 'numerical_merit_budget.urdf'
    urdf_path.write_text(URDF)
    robot = eik.RobotModel(str(urdf_path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = STEP_DT
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    primary = solver.add_frame_task('primary', 'ee')
    primary.priority = 0
    secondary = solver.add_frame_task('secondary', 'world')
    secondary.priority = 1
    configuration = np.zeros(robot.nv)
    robot.update_configuration(configuration)
    pose = robot.get_frame_pose('ee')
    target = eik.TaskTarget.from_se3(
        'primary', eik.SE3(
            rotation=Rotation.from_rotvec([0.0, 0.0, -initial_error]).as_matrix(),
            translation=np.asarray(pose.translation) + np.array([0.0, GOAL_DISTANCE, 0.0]),
        ), 1.0, 1.0,
    )
    target.priority_orientation_tolerance = ORIENTATION_TOLERANCE
    secondary_target = eik.TaskTarget.from_se3(
        'secondary', robot.get_frame_pose('world'), 1.0, 1.0,
    )
    options = eik.PositionStepOptions()
    options.dt = STEP_DT
    options.max_steps = 1
    options.primary_solve_mode = mode
    options.primary_allow_min_error_fallback = False
    maximum_error = max(initial_error, ORIENTATION_TOLERANCE + NUMERICAL_MERIT_TOLERANCE)
    for _ in range(STEP_COUNT):
        result = solver.solve_position_step(configuration, [target, secondary_target], options)
        configuration = np.asarray(result.q_solution).copy()
        # Translation keeps the configuration step above the no-motion shortcut,
        # while the slow yaw hits the numerical allowance rather than a large
        # violation. Reusing it every tick was enough to break the fixed budget.
        assert result.orientation_error <= maximum_error + MEASUREMENT_ROUNDOFF
