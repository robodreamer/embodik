#!/usr/bin/env python3
"""Helpers for ROBOTIS AI worker example scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def resolve_ffw_urdf_path(variant: str) -> Path:
    """Resolve a local ROBOTIS FFW URDF path for the requested variant."""
    variant = str(variant).strip().lower()
    if variant not in {"sg2", "bg2"}:
        raise ValueError(f"Unsupported FFW variant '{variant}'. Expected one of: bg2, sg2")

    override_key = f"EMBODIK_FFW_{variant.upper()}_URDF"
    for env_key in (override_key, "EMBODIK_FFW_URDF"):
        env_value = os.environ.get(env_key, "").strip()
        if env_value:
            path = Path(env_value).expanduser().resolve()
            if path.is_file():
                return path
            raise FileNotFoundError(f"{env_key} does not exist: {path}")

    candidates = {
        "sg2": [
            Path("/path/to/local/Projects/robot_models_urdf/ffw_sg2_mobile_robot/ffw_sg2_mobile_robot.urdf"),
        ],
        "bg2": [
            Path("/path/to/local/Projects/robot_models_urdf/ffw_bg2_mobile_robot/ffw_bg2_mobile_robot.urdf"),
        ],
    }
    found = _first_existing(candidates[variant])
    if found is not None:
        return found

    raise FileNotFoundError(
        "Could not find a ROBOTIS FFW URDF for variant "
        f"'{variant}'. Set {override_key}=/absolute/path/to/ffw_{variant}.urdf"
    )


def resolve_ai_worker_frames(frame_names: Iterable[str]) -> dict[str, str]:
    """Resolve key task frames for the ROBOTIS AI worker URDFs."""
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


def default_worker_ik_joint_names(joint_names: Iterable[str]) -> list[str]:
    """Return the reduced IK joint set for the worker demo."""
    allowed: list[str] = []
    for joint_name in joint_names:
        if any(token in joint_name for token in ("wheel_", "world_fixed")):
            continue
        if joint_name.startswith("lift_") or joint_name.startswith("arm_"):
            allowed.append(joint_name)
    return allowed


def default_worker_allowed_joint_names(joint_names: Iterable[str]) -> set[str]:
    """Backward-compatible alias for the worker demo's active IK joints."""
    return set(default_worker_ik_joint_names(joint_names))
