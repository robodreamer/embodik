#!/usr/bin/env python3
"""Arm-joint classifier coverage for alpha-wheelbase naming.

The shared bimanual app buckets joints into left/right arm velocity index lists
to drive per-arm freezing and nullspace control. The classifiers must recognise
alpha-wheelbase arm joints (``left_shoulder_*`` etc.) while staying correct for
the legacy FFW (``arm_l_*``) naming and excluding non-arm joints.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.example_helpers.common_bimanual_teleop_app import (
    DEFAULT_AUTO_TORSO_CONTRIBUTION,
    DEFAULT_MAX_ANGULAR_SPEED,
    DEFAULT_MAX_LINEAR_SPEED,
    _apply_bimanual_task_dof_ownership,
    _apply_position_step_speed_caps,
    _apply_torso_control_priority_policy,
    _collect_torso_arm_contribution_indices,
    _effective_torso_contribution,
    _is_arm_joint,
    _is_left_arm_joint,
    _is_right_arm_joint,
    _torso_arm_contribution_metric_weights,
)

ALPHA_LEFT_ARM = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
]
ALPHA_RIGHT_ARM = [name.replace("left_", "right_") for name in ALPHA_LEFT_ARM]
ALPHA_NON_ARM = [
    "base_yaw_joint",
    "base_pitch_joint",
    "knee_pitch_joint",
    "hip_pitch_joint",
    "torso_yaw_joint",
    "neck_yaw_joint",
    "neck_pitch_joint",
    "left_robotiq_85_left_knuckle_joint",
    "right_robotiq_85_right_finger_tip_joint",
]


def test_alpha_left_arm_joints_classified_left() -> None:
    for name in ALPHA_LEFT_ARM:
        assert _is_left_arm_joint(name), name
        assert not _is_right_arm_joint(name), name
        assert _is_arm_joint(name), name


def test_alpha_right_arm_joints_classified_right() -> None:
    for name in ALPHA_RIGHT_ARM:
        assert _is_right_arm_joint(name), name
        assert not _is_left_arm_joint(name), name
        assert _is_arm_joint(name), name


def test_alpha_non_arm_joints_not_classified_as_arm() -> None:
    for name in ALPHA_NON_ARM:
        assert not _is_arm_joint(name), name


def test_legacy_ffw_arm_naming_still_classified() -> None:
    assert _is_left_arm_joint("arm_l_joint3")
    assert _is_right_arm_joint("arm_r_joint3")
    assert _is_left_arm_joint("gripper_l_joint2")
    assert _is_right_arm_joint("gripper_r_joint2")


class _FakeRobot:
    def __init__(self, names: list[str]) -> None:
        self._indices = {name: idx for idx, name in enumerate(names)}

    def get_joint_velocity_index(self, joint_name: str) -> int:
        return self._indices[joint_name]

    def get_joint_velocity_size(self, _joint_name: str) -> int:
        return 1


class _FakeTask:
    def __init__(self) -> None:
        self.excluded: list[int] = []
        self.priority = 0

    def set_excluded_joint_indices(self, indices: list[int]) -> None:
        self.excluded = list(indices)

    def clear_excluded_joint_indices(self) -> None:
        self.excluded = []


class _FakePostureTask:
    def __init__(self) -> None:
        self.controlled: list[int] = []

    def set_controlled_joint_indices(self, indices: list[int]) -> None:
        self.controlled = list(indices)


def test_bimanual_task_dof_ownership_keeps_arms_available_to_torso_marker() -> None:
    right_task = _FakeTask()
    left_task = _FakeTask()
    torso_task = _FakeTask()

    _apply_bimanual_task_dof_ownership(
        right_task=right_task,
        left_task=left_task,
        torso_task=torso_task,
        right_active=True,
        left_active=True,
        torso_active=True,
        decouple_torso_and_arms=True,
        left_arm_velocity_indices=[10, 11],
        right_arm_velocity_indices=[20, 21],
        torso_velocity_indices=[0, 1, 2],
    )

    assert right_task.excluded == [0, 1, 2, 10, 11]
    assert left_task.excluded == [0, 1, 2, 20, 21]
    assert torso_task.excluded == []


def test_bimanual_task_dof_ownership_shared_mode_preserves_torso_sharing() -> None:
    right_task = _FakeTask()
    left_task = _FakeTask()
    torso_task = _FakeTask()

    _apply_bimanual_task_dof_ownership(
        right_task=right_task,
        left_task=left_task,
        torso_task=torso_task,
        right_active=True,
        left_active=True,
        torso_active=True,
        decouple_torso_and_arms=False,
        left_arm_velocity_indices=[10, 11],
        right_arm_velocity_indices=[20, 21],
        torso_velocity_indices=[0, 1, 2],
    )

    assert right_task.excluded == [10, 11]
    assert left_task.excluded == [20, 21]
    assert torso_task.excluded == []


def test_torso_control_shared_mode_is_secondary_and_removes_lift_bias() -> None:
    torso_task = _FakeTask()
    posture_task = _FakePostureTask()

    torso_secondary = _apply_torso_control_priority_policy(
        torso_task=torso_task,
        posture_task=posture_task,
        torso_active=True,
        decouple_torso_and_arms=False,
        posture_controlled_indices=[0, 1, 2],
        lift_posture_indices=[0],
    )

    assert torso_secondary is True
    assert torso_task.priority == 1
    assert posture_task.controlled == [1, 2]


def test_torso_control_decoupled_mode_stays_primary_and_keeps_posture_bias() -> None:
    torso_task = _FakeTask()
    posture_task = _FakePostureTask()

    torso_secondary = _apply_torso_control_priority_policy(
        torso_task=torso_task,
        posture_task=posture_task,
        torso_active=True,
        decouple_torso_and_arms=True,
        posture_controlled_indices=[0, 1, 2],
        lift_posture_indices=[0],
    )

    assert torso_secondary is False
    assert torso_task.priority == 0
    assert posture_task.controlled == [0, 1, 2]


def test_alpha_torso_contribution_metric_groups_torso_and_arms() -> None:
    names = [
        "base_yaw_joint",
        "base_pitch_joint",
        "knee_pitch_joint",
        "hip_pitch_joint",
        "torso_yaw_joint",
        *ALPHA_LEFT_ARM,
        *ALPHA_RIGHT_ARM,
        "neck_yaw_joint",
        "left_robotiq_85_left_knuckle_joint",
    ]
    torso, arms, nv = _collect_torso_arm_contribution_indices(
        _FakeRobot(names), names, lock_joint_names=set(ALPHA_NON_ARM[:5])
    )

    assert torso == list(range(5))
    assert arms == list(range(5, 19))
    assert nv == len(names)

    arms_do_work = _torso_arm_contribution_metric_weights(
        0.0,
        torso_velocity_indices=torso,
        arm_velocity_indices=arms,
        nv=nv,
    )
    torso_does_work = _torso_arm_contribution_metric_weights(
        1.0,
        torso_velocity_indices=torso,
        arm_velocity_indices=arms,
        nv=nv,
    )

    assert arms_do_work is not None
    assert torso_does_work is not None
    assert arms_do_work[torso[0]] > arms_do_work[arms[0]]
    assert torso_does_work[arms[0]] > torso_does_work[torso[0]]
    assert (
        _torso_arm_contribution_metric_weights(
            0.5,
            torso_velocity_indices=torso,
            arm_velocity_indices=arms,
            nv=nv,
        )
        is None
    )


def test_auto_torso_contribution_caps_torso_bias() -> None:
    assert _effective_torso_contribution(0.75, torso_prefer_locked=False) == 0.75
    assert (
        _effective_torso_contribution(0.75, torso_prefer_locked=True)
        == DEFAULT_AUTO_TORSO_CONTRIBUTION
    )
    assert _effective_torso_contribution(0.2, torso_prefer_locked=True) == 0.2


def test_common_bimanual_default_speed_caps_apply_to_position_step_options() -> None:
    class _Opts:
        max_linear_speed = 0.0
        max_angular_speed = 0.0

    opts = _Opts()
    _apply_position_step_speed_caps(
        opts,
        max_linear_speed=DEFAULT_MAX_LINEAR_SPEED,
        max_angular_speed=DEFAULT_MAX_ANGULAR_SPEED,
    )

    assert opts.max_linear_speed == DEFAULT_MAX_LINEAR_SPEED
    assert opts.max_angular_speed == DEFAULT_MAX_ANGULAR_SPEED


def test_common_bimanual_speed_caps_are_backward_compatible_with_old_options() -> None:
    _apply_position_step_speed_caps(object(), max_linear_speed=0.5, max_angular_speed=1.0)
