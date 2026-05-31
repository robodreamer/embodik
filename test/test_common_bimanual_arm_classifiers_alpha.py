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
    _is_arm_joint,
    _is_left_arm_joint,
    _is_right_arm_joint,
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
