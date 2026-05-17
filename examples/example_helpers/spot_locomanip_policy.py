"""Spot locomanipulation policy helpers shared by the mjviser example and tests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from operator import sub
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

SPOT_LEG_JOINT_NAMES: tuple[str, ...] = (
    "fl_hx",
    "fl_hy",
    "fl_kn",
    "fr_hx",
    "fr_hy",
    "fr_kn",
    "hl_hx",
    "hl_hy",
    "hl_kn",
    "hr_hx",
    "hr_hy",
    "hr_kn",
)
MJCF_ARM_JOINT_NAMES: tuple[str, ...] = (
    "arm_sh0",
    "arm_sh1",
    "arm_el0",
    "arm_el1",
    "arm_wr0",
    "arm_wr1",
)
MJCF_GRIPPER_JOINT_NAME = "arm_f1x"
POLICY_ARM_JOINT_NAMES: tuple[str, ...] = tuple(
    f"arm0_{name.removeprefix('arm_')}" for name in MJCF_ARM_JOINT_NAMES
)
POLICY_GRIPPER_JOINT_NAME = "arm0_f1x"

DEFAULT_LEG_JOINT_SPOT = np.array(
    [0.12, 0.5, -1.0, -0.12, 0.5, -1.0, 0.12, 0.5, -1.0, -0.12, 0.5, -1.0],
    dtype=np.float64,
)
DEFAULT_STAND_LEG_JOINTS = np.array(
    [0.12, 0.72, -1.45, -0.12, 0.72, -1.45, 0.12, 0.72, -1.45, -0.12, 0.72, -1.45],
    dtype=np.float64,
)
DEFAULT_STAND_BASE_HEIGHT = 0.525
DEFAULT_ARM_COMMAND = np.array([0.0, -3.12, 3.13, 1.57, 0.0, -1.57, -0.054], dtype=np.float64)
DEFAULT_BODY_ROLL_TRIM = -0.04
DEFAULT_BODY_PITCH_TRIM = 0.16
DEFAULT_BODY_ROLL_PITCH_HEIGHT = np.array(
    [DEFAULT_BODY_ROLL_TRIM, DEFAULT_BODY_PITCH_TRIM, 0.64],
    dtype=np.float64,
)
DEFAULT_VELOCITY_COMMAND = np.zeros(3, dtype=np.float64)
POLICY_ACTION_SCALE = 0.2

# Locomanipulation UI/control defaults.
INITIAL_ARM_COMMAND = np.array([0.0, -0.9, 1.8, 0.0, -0.9, 0.0, -1.54], dtype=np.float64)
POSE_COMMAND_RANGE = (-2.0, 2.0)
POSE_COMMAND_STEP = 0.1
YAW_COMMAND_RANGE = (-3.14, 3.14)
YAW_COMMAND_STEP = 0.1
VELOCITY_COMMAND_RANGE = (-1.0, 1.0)
VELOCITY_COMMAND_STEP = 0.1
ROLL_PITCH_RANGE = (-0.5, 0.5)
ROLL_PITCH_STEP = 0.02
HEIGHT_RANGE = (0.3, 0.65)
HEIGHT_STEP = 0.02
POSE_KP_LINEAR = 2.5
POSE_KI_LINEAR = 0.5
POSE_KP_YAW = 3.0
POSE_KI_YAW = 0.4
POSE_MAX_LINEAR_SPEED = 1.0
POSE_MAX_YAW_SPEED = 1.0
POSE_MAX_INTEGRAL = 1.0


class LocomanipPolicy(Enum):
    """Vendored Spot locomanipulation checkpoints."""

    LOCOMANIP = "locomanip"
    LOCOMANIP_STATIONARY = "locomanip-stationary"


@dataclass(frozen=True)
class PolicySpec:
    name: LocomanipPolicy
    filename: str
    description: str


POLICIES: dict[LocomanipPolicy, PolicySpec] = {
    LocomanipPolicy.LOCOMANIP: PolicySpec(
        LocomanipPolicy.LOCOMANIP,
        "locomanip_policy.onnx",
        "Locomanipulation policy with base velocity commands.",
    ),
    LocomanipPolicy.LOCOMANIP_STATIONARY: PolicySpec(
        LocomanipPolicy.LOCOMANIP_STATIONARY,
        "locomanip_stationary_policy.onnx",
        "Stationary locomanipulation policy.",
    ),
}


def policy_asset_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "spot_policies"


def policy_checkpoint_path(policy: LocomanipPolicy) -> Path:
    return policy_asset_dir() / POLICIES[policy].filename


def parse_policy(value: str) -> LocomanipPolicy:
    try:
        return LocomanipPolicy(value)
    except ValueError as exc:
        choices = ", ".join(policy.value for policy in LocomanipPolicy)
        raise ValueError(f"Unknown policy {value!r}; expected one of: {choices}") from exc


def normalize_mjcf_joint_name(name: str) -> str:
    """Map policy arm names to MuJoCo Menagerie arm names."""

    if name.startswith("arm0_"):
        return "arm_" + name.removeprefix("arm0_")
    return name


def _orbit_joint_order(include_arm: bool = True) -> list[str]:
    parts = (("arm0", ("sh0", "sh1", "el0", "el1", "wr0", "wr1", "f1x")),) if include_arm else ()
    leg_parts = (
        ("fl", ("hx", "hy", "kn")),
        ("fr", ("hx", "hy", "kn")),
        ("hl", ("hx", "hy", "kn")),
        ("hr", ("hx", "hy", "kn")),
    )
    all_parts = parts + leg_parts
    max_levels = 7 if include_arm else 3
    return [
        f"{part}_{joint_names[level]}"
        for level in range(max_levels)
        for part, joint_names in all_parts
        if level < len(joint_names)
    ]


SPOT_JOINT_ORDER: tuple[str, ...] = (
    *SPOT_LEG_JOINT_NAMES,
    *POLICY_ARM_JOINT_NAMES,
    POLICY_GRIPPER_JOINT_NAME,
)
ORBIT_JOINT_ORDER: tuple[str, ...] = tuple(_orbit_joint_order(include_arm=True))
ORBIT_LEG_JOINT_ORDER: tuple[str, ...] = tuple(_orbit_joint_order(include_arm=False))
SPOT_TO_ORBIT_INDEX: tuple[int, ...] = tuple(
    ORBIT_JOINT_ORDER.index(name) for name in SPOT_JOINT_ORDER
)
SPOT_TO_ORBIT_LEG_INDEX: tuple[int, ...] = tuple(
    ORBIT_LEG_JOINT_ORDER.index(name) for name in SPOT_LEG_JOINT_NAMES
)
DEFAULT_JOINT_SPOT = np.concatenate([DEFAULT_LEG_JOINT_SPOT, DEFAULT_ARM_COMMAND])
DEFAULT_JOINT_ORBIT = np.array(
    [DEFAULT_JOINT_SPOT[SPOT_JOINT_ORDER.index(name)] for name in ORBIT_JOINT_ORDER],
    dtype=np.float64,
)
DEFAULT_LEG_JOINT_ORBIT = np.array(
    [DEFAULT_LEG_JOINT_SPOT[SPOT_LEG_JOINT_NAMES.index(name)] for name in ORBIT_LEG_JOINT_ORDER],
    dtype=np.float64,
)


def spot_to_orbit(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (len(SPOT_TO_ORBIT_INDEX),):
        raise ValueError(f"Expected {len(SPOT_TO_ORBIT_INDEX)} values, got {arr.shape}")
    out = np.zeros_like(arr)
    for spot_idx, orbit_idx in enumerate(SPOT_TO_ORBIT_INDEX):
        out[orbit_idx] = arr[spot_idx]
    return out


def orbit_to_spot_leg(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < len(SPOT_TO_ORBIT_LEG_INDEX):
        raise ValueError(
            f"Expected at least {len(SPOT_TO_ORBIT_LEG_INDEX)} leg values, got {arr.size}"
        )
    out = np.zeros(len(SPOT_TO_ORBIT_LEG_INDEX), dtype=np.float64)
    for spot_idx, orbit_idx in enumerate(SPOT_TO_ORBIT_LEG_INDEX):
        out[spot_idx] = arr[orbit_idx]
    return out


def build_policy_observation(
    sim_observation: Mapping[str, np.ndarray],
    last_output: Sequence[float],
    velocity_command: Sequence[float] = DEFAULT_VELOCITY_COMMAND,
    arm_joint_command: Sequence[float] = DEFAULT_ARM_COMMAND,
    body_roll_pitch_height_command: Sequence[float] = DEFAULT_BODY_ROLL_PITCH_HEIGHT,
) -> np.ndarray:
    """Build the locomanipulation observation vector used by the policy."""

    joint_pos = np.concatenate(
        [
            np.asarray(sim_observation["joint_pos"], dtype=np.float64),
            np.asarray(sim_observation["arm_state"], dtype=np.float64)[:6],
            np.asarray(sim_observation["gripper_state"], dtype=np.float64)[:1],
        ]
    )
    joint_vel = np.concatenate(
        [
            np.asarray(sim_observation["joint_vel"], dtype=np.float64),
            np.asarray(sim_observation["arm_state"], dtype=np.float64)[6:],
            np.asarray(sim_observation["gripper_state"], dtype=np.float64)[1:],
        ]
    )
    if joint_pos.size != 19 or joint_vel.size != 19:
        raise ValueError(
            f"Expected 19 joint positions/velocities, got {joint_pos.size}/{joint_vel.size}"
        )

    processed = [
        *np.asarray(sim_observation["base_lin_vel"], dtype=np.float64).tolist(),
        *np.asarray(sim_observation["base_ang_vel"], dtype=np.float64).tolist(),
        *np.asarray(sim_observation["projected_gravity"], dtype=np.float64).tolist(),
        *np.asarray(velocity_command, dtype=np.float64).tolist(),
        *np.asarray(arm_joint_command, dtype=np.float64).tolist(),
        *np.asarray(body_roll_pitch_height_command, dtype=np.float64).tolist(),
        *map(sub, spot_to_orbit(joint_pos), DEFAULT_JOINT_ORBIT),
        *spot_to_orbit(joint_vel).tolist(),
        *np.asarray(last_output, dtype=np.float64).tolist(),
    ]
    return np.asarray(processed, dtype=np.float32)


def leg_targets_from_policy_output(policy_output: Sequence[float]) -> np.ndarray:
    """Convert raw policy output to 12 Spot-order leg position targets."""

    output = np.asarray(policy_output, dtype=np.float64).reshape(-1)
    if output.size not in (12, 19):
        raise ValueError(f"Expected a 12- or 19-element policy output, got {output.size}")
    shifted_orbit = output[:12] * POLICY_ACTION_SCALE + DEFAULT_LEG_JOINT_ORBIT
    return orbit_to_spot_leg(shifted_orbit)


def policy_action_to_mujoco_ctrl(
    policy_output: Sequence[float],
    arm_command: Sequence[float] = DEFAULT_ARM_COMMAND,
) -> dict[str, float]:
    """Return MuJoCo actuator targets for the policy output and arm command."""

    leg_targets = leg_targets_from_policy_output(policy_output)
    arm = np.asarray(arm_command, dtype=np.float64)
    if arm.shape != (7,):
        raise ValueError(f"Expected 7 arm/gripper command values, got {arm.shape}")
    targets = {name: float(value) for name, value in zip(SPOT_LEG_JOINT_NAMES, leg_targets)}
    targets.update({name: float(value) for name, value in zip(MJCF_ARM_JOINT_NAMES, arm[:6])})
    targets[MJCF_GRIPPER_JOINT_NAME] = float(arm[6])
    return targets


def available_policy_choices() -> tuple[str, ...]:
    return tuple(policy.value for policy in LocomanipPolicy)
