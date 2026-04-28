#!/usr/bin/env python3
"""Shared model helpers for AI Worker-style dual-arm mobile manipulators."""

from __future__ import annotations

from typing import Iterable


def resolve_ai_worker_frames(frame_names: Iterable[str]) -> dict[str, str]:
    """Resolve key task frames for ROBOTIS AI Worker-style URDFs."""
    names = set(frame_names)

    def pick(candidates: list[str], label: str) -> str:
        for name in candidates:
            if name in names:
                return name
        raise ValueError(f"Could not resolve frame for {label}. Tried: {candidates}")

    return {
        "base": pick(["base_link"], "base"),
        "arm_base": pick(["arm_base_link", "lift_link"], "arm_base"),
        "head": pick(["head_link2", "head_link1"], "head"),
        "right_tool": pick(
            ["gripper_r_rh_p12_rn_base", "arm_r_link7", "camera_r_link"],
            "right_tool",
        ),
        "left_tool": pick(
            ["gripper_l_rh_p12_rn_base", "arm_l_link7", "camera_l_link"],
            "left_tool",
        ),
    }


def default_ai_worker_ik_joint_names(joint_names: Iterable[str]) -> list[str]:
    """Return a reduced active IK set for AI Worker follower-style models."""
    allowed: list[str] = []
    for joint_name in joint_names:
        if any(token in joint_name for token in ("wheel_", "world_fixed")):
            continue
        if joint_name.startswith("lift_") or joint_name.startswith("arm_"):
            allowed.append(joint_name)
    return allowed


def default_ai_worker_allowed_joint_names(joint_names: Iterable[str]) -> set[str]:
    """Return the active IK joint names as a set for tests and filters."""
    return set(default_ai_worker_ik_joint_names(joint_names))
