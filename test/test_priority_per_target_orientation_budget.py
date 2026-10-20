"""Same-priority frame tasks share one orientation merit."""

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
OUTWARD_DRIVE = 1.0
LEFT_SIDE = "left"
RIGHT_SIDE = "right"
SIDES = (LEFT_SIDE, RIGHT_SIDE)
YAW_AXIS = np.array([0.0, 0.0, 1.0])
TRANSLATION_AXIS = np.array([0.0, 1.0, 0.0])
TRANSLATION_MASK = np.array([False, True, False])
ORIENTATION_MASKS = {
    "full": np.array([True, True, True]),
    "yaw_only": np.array([False, False, True]),
    "yaw_uncommanded": np.array([True, False, False]),
    "none": np.array([False, False, False]),
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
    translation_error: float


def run_case(
    tmp_path,
    *,
    mode,
    initial_error,
    mask_name,
    position_direction=OUTWARD_DRIVE,
    orientation_gain=1.0,
    per_target=False,
    translation_joint=False,
    acceleration_limits=False,
    budget_schedule=None,
):
    branches = "".join(
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
    path = tmp_path / "two_frame_budget.urdf"
    if translation_joint:
        for side in SIDES:
            branches = branches.replace(
                f'<parent link="world"/><child link="{side}_wrist"/>',
                f'<parent link="{side}_slide"/><child link="{side}_wrist"/>',
            )
            branches += (
                f'<link name="{side}_slide"/><joint name="{side}_translate" type="prismatic">'
                f'<parent link="world"/><child link="{side}_slide"/><axis xyz="0 1 0"/>'
                f'<limit lower="-1" upper="1" effort="100" velocity="1"/></joint>'
            )
    path.write_text(f'<robot name="two_frame_budget"><link name="world"/>{branches}</robot>')
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.dt = STEP_DT
    if acceleration_limits:
        solver.set_acceleration_limits(np.full(robot.nv, 1.0))
        solver.enable_acceleration_limits(True)
    configuration = np.zeros(robot.nv)
    robot.update_configuration(configuration)
    targets = []
    desired_rotations = {
        LEFT_SIDE: Rotation.from_rotvec(-initial_error * YAW_AXIS).as_matrix(),
        RIGHT_SIDE: np.eye(YAW_AXIS.size),
    }
    orientation_mask = ORIENTATION_MASKS[mask_name]
    for side in SIDES:
        task = solver.add_frame_task(side, f"{side}_ee")
        task.priority = PRIMARY_PRIORITY
        task.set_position_mask(TRANSLATION_MASK)
        task.set_orientation_mask(orientation_mask)
        pose = robot.get_frame_pose(f"{side}_ee")
        translation_offset = position_direction * GOAL_TRANSLATION if side == LEFT_SIDE else 0.0
        desired_translation = np.asarray(pose.translation) + translation_offset * TRANSLATION_AXIS
        target = eik.TaskTarget.from_se3(
            side,
            eik.SE3(rotation=desired_rotations[side], translation=desired_translation),
            1.0,
            orientation_gain,
        )
        target.priority_orientation_tolerance = ORIENTATION_BUDGET
        targets.append(target)
    secondary = solver.add_frame_task("secondary", "world")
    secondary.priority = SECONDARY_PRIORITY
    targets.append(
        eik.TaskTarget.from_se3(
            "secondary",
            robot.get_frame_pose("world"),
            1.0,
            1.0,
        )
    )
    options = eik.PositionStepOptions()
    options.dt = STEP_DT
    options.max_steps = STEPS_PER_TICK
    options.primary_solve_mode = mode
    options.primary_allow_min_error_fallback = False
    if per_target:
        options.per_target_priority_budgets = True
    for tick in range(TICK_COUNT):
        if budget_schedule is not None:
            for target in targets[:2]:
                target.priority_orientation_tolerance = budget_schedule(tick)
        previous_configuration = configuration.copy()
        result = solver.solve_position_step(configuration, targets, options)
        configuration = np.asarray(result.q_solution).copy()
        robot.update_configuration(configuration)
        left_rotation = np.asarray(robot.get_frame_pose("left_ee").rotation)
        right_rotation = np.asarray(robot.get_frame_pose("right_ee").rotation)
        left_error = Rotation.from_matrix(
            desired_rotations[LEFT_SIDE].T @ left_rotation,
        ).as_rotvec()
        right_error = Rotation.from_matrix(
            desired_rotations[RIGHT_SIDE].T @ right_rotation,
        ).as_rotvec()
        yield BudgetObservation(
            left_error=(
                float(np.linalg.norm(left_error * orientation_mask))
                if orientation_gain > 0.0
                else 0.0
            ),
            right_error=(
                float(np.linalg.norm(right_error * orientation_mask))
                if orientation_gain > 0.0
                else 0.0
            ),
            raw_left_error=float(np.linalg.norm(left_error)),
            step_norm=float(np.linalg.norm(configuration - previous_configuration)),
            translation_error=float(
                np.linalg.norm(
                    robot.get_frame_pose("left_ee").translation
                    - np.asarray(targets[0].target_pose)[:3, 3]
                )
            ),
        )


@pytest.mark.parametrize("mode", TASK_MODES)
@pytest.mark.parametrize("initial_error", [ORIENTATION_BUDGET, OUTSIDE_BUDGET_ERROR])
@pytest.mark.parametrize("mask_name", tuple(ORIENTATION_MASKS))
def test_same_priority_frames_share_one_orientation_merit(
    tmp_path,
    mode,
    initial_error,
    mask_name,
):
    observations = list(
        run_case(
            tmp_path,
            mode=mode,
            initial_error=initial_error,
            mask_name=mask_name,
        )
    )
    left_initial_merit = initial_error if ORIENTATION_MASKS[mask_name][-1] else 0.0
    left_allowed = max(left_initial_merit, ORIENTATION_BUDGET + NUMERICAL_SLACK)
    right_allowed = ORIENTATION_BUDGET + NUMERICAL_SLACK
    yaw_is_commanded = bool(ORIENTATION_MASKS[mask_name][-1])
    if not yaw_is_commanded:
        for observation in observations:
            assert observation.left_error <= left_allowed + MEASUREMENT_ROUNDOFF
            assert observation.right_error <= right_allowed + MEASUREMENT_ROUNDOFF
        # A masked yaw component stays out of the commanded merit.
        assert max(item.raw_left_error for item in observations) > left_allowed
        assert max(item.step_norm for item in observations) > MEASUREMENT_ROUNDOFF
    else:
        assert max(item.left_error for item in observations) > left_allowed


@pytest.mark.parametrize("mode", TASK_MODES)
def test_disabled_orientation_gain_does_not_protect_raw_rotation(tmp_path, mode):
    observations = list(
        run_case(
            tmp_path,
            mode=mode,
            initial_error=OUTSIDE_BUDGET_ERROR,
            mask_name="full",
            orientation_gain=0.0,
        )
    )
    assert all(item.left_error == 0.0 and item.right_error == 0.0 for item in observations)
    assert observations[-1].raw_left_error > OUTSIDE_BUDGET_ERROR
    assert max(item.step_norm for item in observations) > MEASUREMENT_ROUNDOFF


@pytest.mark.parametrize("mode", TASK_MODES)
@pytest.mark.parametrize("initial_error", [0.0, ORIENTATION_BUDGET, OUTSIDE_BUDGET_ERROR])
@pytest.mark.parametrize("mask_name", tuple(ORIENTATION_MASKS))
@pytest.mark.parametrize("acceleration_limits", [False, True])
def test_explicit_individual_budgets_do_not_spend_another_hand_tolerance(
    tmp_path, mode, initial_error, mask_name, acceleration_limits
):
    observations = list(
        run_case(
            tmp_path,
            mode=mode,
            initial_error=initial_error,
            mask_name=mask_name,
            per_target=True,
            acceleration_limits=acceleration_limits,
        )
    )
    initial_merit = initial_error if ORIENTATION_MASKS[mask_name][-1] else 0.0
    allowed = max(initial_merit, ORIENTATION_BUDGET + NUMERICAL_SLACK)
    assert all(item.left_error <= allowed + MEASUREMENT_ROUNDOFF for item in observations)
    assert all(
        item.right_error <= ORIENTATION_BUDGET + NUMERICAL_SLACK + MEASUREMENT_ROUNDOFF
        for item in observations
    )


@pytest.mark.parametrize("mode", TASK_MODES)
def test_individual_budgets_still_generate_feasible_translation(tmp_path, mode):
    observations = list(
        run_case(
            tmp_path,
            mode=mode,
            initial_error=ORIENTATION_BUDGET,
            mask_name="full",
            per_target=True,
            translation_joint=True,
        )
    )
    assert observations[-1].translation_error < GOAL_TRANSLATION - 1e-3
    assert max(item.step_norm for item in observations) > 1e-3
    assert all(item.left_error <= ORIENTATION_BUDGET + NUMERICAL_SLACK for item in observations)


@pytest.mark.parametrize("mode", TASK_MODES)
def test_inflating_individual_budget_online_releases_motion_without_target_reset(tmp_path, mode):
    observations = list(
        run_case(
            tmp_path,
            mode=mode,
            initial_error=ORIENTATION_BUDGET,
            mask_name="full",
            per_target=True,
            budget_schedule=lambda tick: ORIENTATION_BUDGET if tick < TICK_COUNT // 2 else 0.02,
        )
    )
    assert (
        max(item.left_error for item in observations[: TICK_COUNT // 2])
        <= ORIENTATION_BUDGET + NUMERICAL_SLACK
    )
    assert observations[-1].left_error > ORIENTATION_BUDGET + 1e-4
    assert observations[-1].left_error <= 0.02 + NUMERICAL_SLACK
