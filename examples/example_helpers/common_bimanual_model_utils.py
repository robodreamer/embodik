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
        "base": pick(["base_link", "base"], "base"),
        "arm_base": pick(["arm_base_link", "lift_link", "link_torso_5"], "arm_base"),
        "head": pick(["head_link2", "head_link1", "link_head_1", "head_base"], "head"),
        "right_tool": pick(
            [
                "gripper_r_rh_p12_rn_base",
                "arm_r_link7",
                "camera_r_link",
                "ee_right_tip",
                "ee_finger_r1",
                "ee_right",
                "tool_right",
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
            ],
            "left_tool",
        ),
    }


def default_common_bimanual_ik_joint_names(joint_names: Iterable[str]) -> list[str]:
    """Return a reduced active IK set for supported bimanual models."""
    allowed: list[str] = []
    for joint_name in joint_names:
        if any(token in joint_name for token in ("wheel_", "world_fixed")):
            continue
        if joint_name in {"left_wheel", "right_wheel"}:
            continue
        if joint_name.startswith(("lift_", "arm_", "torso_", "left_arm_", "right_arm_")):
            allowed.append(joint_name)
    return allowed


def default_common_bimanual_allowed_joint_names(joint_names: Iterable[str]) -> set[str]:
    """Return the active IK joint names as a set for tests and filters."""
    return set(default_common_bimanual_ik_joint_names(joint_names))
