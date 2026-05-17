#!/usr/bin/env python3
"""Headless hardening harness for Spot locomanip arm-vs-locomotion IK allocation.

The harness sweeps reachable and locomotion-sized wrist-frame targets and reports
how much motion is allocated to the arm versus x/y/yaw locomotion commands. It is
intended to catch regressions where small targets that the arm can reach start
triggering base locomotion too early.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import embodik  # noqa: E402
from example_helpers.spot_locomanip_policy import (  # noqa: E402
    DEFAULT_BODY_ROLL_PITCH_HEIGHT,
    DEFAULT_STAND_BASE_HEIGHT,
    DEFAULT_STAND_LEG_JOINTS,
    INITIAL_ARM_COMMAND,
)
from example_helpers.spot_whole_body_ik import (  # noqa: E402
    OptionalSpotWholeBodyIK,
    SPOT_URDF_ENV_VAR,
    _commanded_base_pose_wxyz,
    resolve_spot_ik_urdf,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    offset: tuple[float, float, float]
    yaw_deg: float = 0.0
    expect_locomotion: bool = False


@dataclass
class ScenarioResult:
    name: str
    offset: tuple[float, float, float]
    yaw_deg: float
    arm_delta_norm: float
    locomotion_delta_norm: float
    locomotion_delta: tuple[float, float, float]
    final_position_error_m: float
    final_orientation_error_rad: float
    mean_solve_ms: float
    p95_solve_ms: float
    status: str
    passed: bool


def _base_observation() -> dict[str, np.ndarray]:
    return {
        "base_pose": np.array(
            [0.0, 0.0, DEFAULT_STAND_BASE_HEIGHT, 1.0, 0.0, 0.0, 0.0],
            dtype=float,
        ),
        "joint_pos": DEFAULT_STAND_LEG_JOINTS.copy(),
        "arm_state": np.concatenate([INITIAL_ARM_COMMAND[:6], np.zeros(6, dtype=float)]),
        "gripper_state": np.array([INITIAL_ARM_COMMAND[6], 0.0], dtype=float),
    }


def _target_pose(current_pose, offset: tuple[float, float, float], yaw_deg: float):
    yaw = np.deg2rad(float(yaw_deg))
    c = np.cos(yaw)
    s = np.sin(yaw)
    yaw_rotation = np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return embodik.Rt(
        R=yaw_rotation @ np.asarray(current_pose.rotation, dtype=float),
        t=np.asarray(current_pose.translation, dtype=float) + np.asarray(offset, dtype=float),
    )


def run_scenario(ik: OptionalSpotWholeBodyIK, scenario: Scenario, *, steps: int) -> ScenarioResult:
    ik.reset_reference()
    observation = _base_observation()
    arm = INITIAL_ARM_COMMAND.copy()
    body = DEFAULT_BODY_ROLL_PITCH_HEIGHT.copy()
    desired_pose = np.zeros(3, dtype=float)
    ik._sync_configuration(
        observation,
        arm,
        base_pose_override=_commanded_base_pose_wxyz(body, desired_pose),
    )
    target_pose = _target_pose(ik.robot.get_frame_pose(ik.tool_frame), scenario.offset, scenario.yaw_deg)

    solve_times: list[float] = []
    final_position_error = float("inf")
    final_orientation_error = float("inf")
    status = "not-run"
    for _ in range(steps):
        result = ik.solve_command(
            observation,
            arm,
            target_pose=target_pose,
            body_command=body,
            desired_pose_command=desired_pose,
        )
        if not result.success:
            status = result.message
            break
        arm = result.arm_command
        body = result.body_command
        desired_pose = result.desired_pose_command
        solve_times.append(float(result.solve_time_ms))
        final_position_error = float(result.position_error)
        final_orientation_error = float(result.orientation_error)
        status = result.message
        observation["arm_state"] = np.concatenate([arm[:6], np.zeros(6, dtype=float)])
        observation["gripper_state"] = np.array([arm[6], 0.0], dtype=float)
        observation["base_pose"] = _commanded_base_pose_wxyz(body, desired_pose)

    locomotion_delta = np.asarray(desired_pose, dtype=float)
    locomotion_delta_norm = float(
        np.linalg.norm([locomotion_delta[0], locomotion_delta[1], 0.5 * locomotion_delta[2]])
    )
    arm_delta_norm = float(np.linalg.norm(np.asarray(arm[:6]) - INITIAL_ARM_COMMAND[:6]))
    mean_solve_ms = float(np.mean(solve_times)) if solve_times else float("inf")
    p95_solve_ms = float(np.percentile(solve_times, 95)) if solve_times else float("inf")
    if scenario.expect_locomotion:
        passed = locomotion_delta_norm > 0.05 and final_position_error < 5e-3
    else:
        passed = locomotion_delta_norm < 5e-3 and arm_delta_norm > 1e-2 and final_position_error < 8e-3
    return ScenarioResult(
        name=scenario.name,
        offset=scenario.offset,
        yaw_deg=scenario.yaw_deg,
        arm_delta_norm=arm_delta_norm,
        locomotion_delta_norm=locomotion_delta_norm,
        locomotion_delta=tuple(float(v) for v in locomotion_delta),
        final_position_error_m=final_position_error,
        final_orientation_error_rad=final_orientation_error,
        mean_solve_ms=mean_solve_ms,
        p95_solve_ms=p95_solve_ms,
        status=status,
        passed=passed,
    )


def default_scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario("reachable_forward_4cm", (0.04, 0.0, 0.0)),
        Scenario("reachable_forward_8cm", (0.08, 0.0, 0.0)),
        Scenario("reachable_forward_12cm", (0.12, 0.0, 0.0)),
        Scenario("reachable_lateral_4cm", (0.0, 0.04, 0.0)),
        Scenario("reachable_yaw_15deg", (0.0, 0.0, 0.0), yaw_deg=15.0),
        Scenario("locomotion_forward_25cm", (0.25, 0.0, 0.0), expect_locomotion=True),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=None, help="Spot whole-body URDF path.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.0,
        help="Locomotion sensitivity scale: higher engages locomotion earlier; lower keeps arm-first longer.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--no-gate", action="store_true", help="Report metrics without failing gates.")
    args = parser.parse_args()

    urdf = args.urdf or resolve_spot_ik_urdf()
    if urdf is None:
        raise SystemExit(f"Spot whole-body URDF not available; set {SPOT_URDF_ENV_VAR} or pass --urdf")
    ik = OptionalSpotWholeBodyIK(urdf, dt=0.01)
    ik.locomotion_sensitivity = float(args.sensitivity)
    results = [run_scenario(ik, scenario, steps=int(args.steps)) for scenario in default_scenarios()]
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        for result in results:
            print(
                f"{result.name:24s} pass={result.passed} "
                f"arm={result.arm_delta_norm:.4f} "
                f"loco={result.locomotion_delta_norm:.4f} "
                f"err={result.final_position_error_m * 1e3:.2f}mm "
                f"mean={result.mean_solve_ms:.3f}ms p95={result.p95_solve_ms:.3f}ms "
                f"status={result.status}"
            )
    return 0 if args.no_gate or all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
