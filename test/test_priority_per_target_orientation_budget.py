"""Certified ordinary CPU velocity steps protect each explicit angular budget.

Acceleration and the optional ordinary guard are not required by this contract.
Legacy certification-disabled callers retain aggregate priority semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import embodik as eik

STEP_DT = 0.02
JOINT_SPEED = 0.05
JOINT_LIMIT = 1.0
JOINT_EFFORT = 100.0
LEVER_LENGTH = 1.0
GOAL_TRANSLATION = 0.15
ORIENTATION_BUDGET = 0.002
OUTSIDE_BUDGET_ERROR = 2 * ORIENTATION_BUDGET
TICK_COUNT = 8
STEPS_PER_TICK = 1
NUMERICAL_SLACK = 1e-9
MEASUREMENT_ROUNDOFF = 1e-14
PRIMARY_PRIORITY = 0
SECONDARY_PRIORITY = 1
ACCELERATION_LIMIT = 10.0
OUTWARD_DRIVE = 1.0
RECOVERY_DRIVE = -1.0
LEFT_SIDE = 'left'
RIGHT_SIDE = 'right'
SIDES = (LEFT_SIDE, RIGHT_SIDE)
YAW_AXIS = np.array([0.0, 0.0, 1.0])
TRANSLATION_AXIS = np.array([0.0, 1.0, 0.0])
TRANSLATION_MASK = np.array([False, True, False])
ORIENTATION_MASKS = {
    'full': np.array([True, True, True]),
    'yaw_only': np.array([False, False, True]),
    'yaw_uncommanded': np.array([True, False, False]),
    'none': np.array([False, False, False]),
}
TASK_MODES = (
    eik.TaskSolveMode.MIN_ERROR,
    eik.TaskSolveMode.SCALE,
    eik.TaskSolveMode.SCALE_ELASTIC,
)


@dataclass
class BudgetObservation:
    left_error: float
    right_error: float
    raw_left_error: float
    step_norm: float


def run_case(
    tmp_path, *, certified, mode, initial_error, mask_name,
    acceleration=False, position_direction=OUTWARD_DRIVE, orientation_gain=1.0,
):
    branches = ''.join(
        f'<link name="{side}_wrist"/>'
        f'<joint name="{side}_yaw" type="revolute">'
        f'<parent link="world"/><child link="{side}_wrist"/><axis xyz="0 0 1"/>'
        f'<limit lower="{-JOINT_LIMIT}" upper="{JOINT_LIMIT}" '
        f'effort="{JOINT_EFFORT}" velocity="{JOINT_SPEED}"/></joint>'
        f'<link name="{side}_ee"/>'
        f'<joint name="{side}_offset" type="fixed">'
        f'<parent link="{side}_wrist"/><child link="{side}_ee"/>'
        f'<origin xyz="{LEVER_LENGTH} 0 0"/></joint>'
        for side in SIDES
    )
    path = tmp_path / 'two_frame_budget.urdf'
    path.write_text(f'<robot name="two_frame_budget"><link name="world"/>{branches}</robot>')
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.enable_velocity_task_mode_certification(certified)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.dt = STEP_DT
    if acceleration:
        solver.set_acceleration_limits(np.full(robot.nv, ACCELERATION_LIMIT))
        solver.enable_acceleration_limits(True)
        solver.set_previous_joint_velocities(np.zeros(robot.nv))
    configuration = np.zeros(robot.nv)
    robot.update_configuration(configuration)
    targets = []
    desired_rotations = {
        LEFT_SIDE: Rotation.from_rotvec(-initial_error * YAW_AXIS).as_matrix(),
        RIGHT_SIDE: np.eye(YAW_AXIS.size),
    }
    orientation_mask = ORIENTATION_MASKS[mask_name]
    for side in SIDES:
        task = solver.add_frame_task(side, f'{side}_ee')
        task.priority = PRIMARY_PRIORITY
        task.set_position_mask(TRANSLATION_MASK)
        task.set_orientation_mask(orientation_mask)
        pose = robot.get_frame_pose(f'{side}_ee')
        translation_offset = (
            position_direction * GOAL_TRANSLATION if side == LEFT_SIDE else 0.0
        )
        desired_translation = np.asarray(pose.translation) + translation_offset * TRANSLATION_AXIS
        target = eik.TaskTarget.from_se3(
            side, eik.SE3(rotation=desired_rotations[side], translation=desired_translation),
            1.0, orientation_gain,
        )
        target.priority_orientation_tolerance = ORIENTATION_BUDGET
        targets.append(target)
    secondary = solver.add_frame_task('secondary', 'world')
    secondary.priority = SECONDARY_PRIORITY
    targets.append(eik.TaskTarget.from_se3(
        'secondary', robot.get_frame_pose('world'), 1.0, 1.0,
    ))
    options = eik.PositionStepOptions()
    options.dt = STEP_DT
    options.max_steps = STEPS_PER_TICK
    options.primary_solve_mode = mode
    options.primary_allow_min_error_fallback = False
    for _ in range(TICK_COUNT):
        previous_configuration = configuration.copy()
        result = solver.solve_position_step(configuration, targets, options)
        configuration = np.asarray(result.q_solution).copy()
        robot.update_configuration(configuration)
        left_rotation = np.asarray(robot.get_frame_pose('left_ee').rotation)
        right_rotation = np.asarray(robot.get_frame_pose('right_ee').rotation)
        left_error = Rotation.from_matrix(
            desired_rotations[LEFT_SIDE].T @ left_rotation,
        ).as_rotvec()
        right_error = Rotation.from_matrix(
            desired_rotations[RIGHT_SIDE].T @ right_rotation,
        ).as_rotvec()
        yield BudgetObservation(
            left_error=(
                float(np.linalg.norm(left_error * orientation_mask))
                if orientation_gain > 0.0 else 0.0
            ),
            right_error=(
                float(np.linalg.norm(right_error * orientation_mask))
                if orientation_gain > 0.0 else 0.0
            ),
            raw_left_error=float(np.linalg.norm(left_error)),
            step_norm=float(np.linalg.norm(configuration - previous_configuration)),
        )


@pytest.mark.parametrize('mode', TASK_MODES)
@pytest.mark.parametrize('certified', [False, True])
@pytest.mark.parametrize('initial_error', [ORIENTATION_BUDGET, OUTSIDE_BUDGET_ERROR])
@pytest.mark.parametrize('mask_name', tuple(ORIENTATION_MASKS))
def test_explicit_per_frame_budget_cannot_be_spent_by_another_frame(
    tmp_path, mode, certified, initial_error, mask_name,
):
    observations = list(run_case(
        tmp_path, certified=certified, mode=mode, initial_error=initial_error, mask_name=mask_name,
    ))
    left_initial_merit = initial_error if ORIENTATION_MASKS[mask_name][-1] else 0.0
    left_allowed = max(left_initial_merit, ORIENTATION_BUDGET + NUMERICAL_SLACK)
    right_allowed = ORIENTATION_BUDGET + NUMERICAL_SLACK
    yaw_is_commanded = bool(ORIENTATION_MASKS[mask_name][-1])
    if certified or not yaw_is_commanded:
        for observation in observations:
            assert observation.left_error <= left_allowed + MEASUREMENT_ROUNDOFF
            assert observation.right_error <= right_allowed + MEASUREMENT_ROUNDOFF
    else:
        # Default legacy policy intentionally permits aggregate error trading.
        assert max(item.left_error for item in observations) > left_allowed
    if not yaw_is_commanded:
        # Ignoring a yaw component must remain useful, even though raw rotation
        # leaves the configured angular budget. It is not a commanded merit.
        assert max(item.raw_left_error for item in observations) > left_allowed
        assert max(item.step_norm for item in observations) > MEASUREMENT_ROUNDOFF


@pytest.mark.parametrize('mode', TASK_MODES)
def test_certified_outside_budget_seed_can_recover(tmp_path, mode):
    observations = list(run_case(
        tmp_path, certified=True, mode=mode, initial_error=OUTSIDE_BUDGET_ERROR,
        mask_name='full', position_direction=RECOVERY_DRIVE,
    ))
    assert observations[-1].left_error < OUTSIDE_BUDGET_ERROR
    assert max(item.left_error for item in observations) <= (
        OUTSIDE_BUDGET_ERROR + MEASUREMENT_ROUNDOFF
    )


@pytest.mark.parametrize('mode', TASK_MODES)
def test_disabled_orientation_gain_does_not_protect_raw_rotation(tmp_path, mode):
    observations = list(run_case(
        tmp_path, certified=True, mode=mode, initial_error=OUTSIDE_BUDGET_ERROR,
        mask_name='full', orientation_gain=0.0,
    ))
    assert all(item.left_error == 0.0 and item.right_error == 0.0 for item in observations)
    assert observations[-1].raw_left_error > OUTSIDE_BUDGET_ERROR
    assert max(item.step_norm for item in observations) > MEASUREMENT_ROUNDOFF
