#!/usr/bin/env python3
"""Tests for ROBOTIS AI worker example helpers."""

from __future__ import annotations

from examples.incubating.example_helpers.robotis_ai_worker_utils import (
    default_worker_allowed_joint_names,
    resolve_ai_worker_frames,
)


def test_resolve_ai_worker_frames_prefers_gripper_base_frames() -> None:
    frame_map = resolve_ai_worker_frames(
        [
            "base_link",
            "arm_base_link",
            "head_link2",
            "arm_r_link7",
            "arm_l_link7",
            "gripper_r_rh_p12_rn_base",
            "gripper_l_rh_p12_rn_base",
        ]
    )
    assert frame_map["base"] == "base_link"
    assert frame_map["arm_base"] == "arm_base_link"
    assert frame_map["head"] == "head_link2"
    assert frame_map["right_tool"] == "gripper_r_rh_p12_rn_base"
    assert frame_map["left_tool"] == "gripper_l_rh_p12_rn_base"


def test_resolve_ai_worker_frames_falls_back_to_link7_and_camera() -> None:
    frame_map = resolve_ai_worker_frames(
        [
            "base_link",
            "lift_link",
            "head_link1",
            "arm_r_link7",
            "camera_l_link",
        ]
    )
    assert frame_map["arm_base"] == "lift_link"
    assert frame_map["head"] == "head_link1"
    assert frame_map["right_tool"] == "arm_r_link7"
    assert frame_map["left_tool"] == "camera_l_link"


def test_default_worker_allowed_joint_names_excludes_wheels() -> None:
    allowed = default_worker_allowed_joint_names(
        [
            "lift_joint",
            "head_joint1",
            "arm_r_joint3",
            "gripper_l_joint2",
            "left_wheel_steer",
            "rear_wheel_drive",
        ]
    )
    assert "lift_joint" in allowed
    assert "head_joint1" in allowed
    assert "arm_r_joint3" in allowed
    assert "gripper_l_joint2" in allowed
    assert "left_wheel_steer" not in allowed
    assert "rear_wheel_drive" not in allowed
