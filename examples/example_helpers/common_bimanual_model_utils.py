#!/usr/bin/env python3
"""Shared model helpers for bimanual mobile manipulators."""

from __future__ import annotations

from typing import Iterable


def resolve_common_bimanual_frames(frame_names: Iterable[str]) -> dict[str, str]:
    """Resolve key task frames for supported bimanual mobile-manipulator URDFs."""
    names = set(frame_names)

    def pick(candidates: list[str], label: str) -> str:
        for name in candidates:
            if name in names:
                return name
        raise ValueError(f"Could not resolve frame for {label}. Tried: {candidates}")

    return {
        "base": pick(["base_link", "base", "base_body"], "base"),
        "arm_base": pick(["arm_base_link", "lift_link", "link_torso_5", "torso_link"], "arm_base"),
        "head": pick(
            ["head_link2", "head_link1", "link_head_1", "head_base", "head_pitch_link"], "head"
        ),
        "right_tool": pick(
            [
                "gripper_r_rh_p12_rn_base",
                "arm_r_link7",
                "camera_r_link",
                "ee_right_tip",
                "ee_finger_r1",
                "ee_right",
                "tool_right",
                "right_gripper_frame",
                "right_finger_frame",
            ],
            "right_tool",
        ),
        "left_tool": pick(
            [
                "gripper_l_rh_p12_rn_base",
                "arm_l_link7",
                "camera_l_link",
                "ee_left_tip",
                "ee_finger_l1",
                "ee_left",
                "tool_left",
                "left_gripper_frame",
                "left_finger_frame",
            ],
            "left_tool",
        ),
    }


# Prefixes whose joints are always part of the reduced IK set (legacy FFW/RB-Y1).
_IK_INCLUDE_PREFIXES = ("lift_", "arm_", "torso_", "left_arm_", "right_arm_")
# Substrings that identify side-prefixed arm + torso-chain joints. Head/neck are
# intentionally omitted (they do not help bimanual reaching, matching the FFW setup).
_IK_INCLUDE_TOKENS = ("shoulder", "elbow", "wrist", "base_yaw", "base_pitch", "knee", "hip")
# Substrings that exclude a joint regardless of any include match (drive train + grippers).
_IK_EXCLUDE_TOKENS = ("wheel_", "world_fixed", "caster", "robotiq", "neck_")


def default_common_bimanual_ik_joint_names(joint_names: Iterable[str]) -> list[str]:
    """Return a reduced active IK set for supported bimanual models."""
    allowed: list[str] = []
    for joint_name in joint_names:
        if any(token in joint_name for token in _IK_EXCLUDE_TOKENS):
            continue
        if joint_name in {"left_wheel", "right_wheel"}:
            continue
        if joint_name.startswith(_IK_INCLUDE_PREFIXES) or any(
            token in joint_name for token in _IK_INCLUDE_TOKENS
        ):
            allowed.append(joint_name)
    return allowed


def default_common_bimanual_allowed_joint_names(joint_names: Iterable[str]) -> set[str]:
    """Return the active IK joint names as a set for tests and filters."""
    return set(default_common_bimanual_ik_joint_names(joint_names))
