#!/usr/bin/env python3
"""Spot locomanipulation policy in MuJoCo through mjviser.

The MuJoCo model comes from the public MuJoCo Menagerie package exposed by
``robot_descriptions.spot_mj_description``. The ONNX checkpoints are vendored
under ``examples/assets/spot_policies`` so the example does not need access to
any external policy repository.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Mapping

import numpy as np

_EXAMPLES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLES_DIR.parent
_PYTHON_DIR = _REPO_ROOT / "python"
for _path in (_PYTHON_DIR, _EXAMPLES_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from example_helpers.ik_common import COLLISION_TUNING_OPTIONS, DEFAULT_VISER_PORT
from example_helpers.policy_runtime import DEFAULT_POLICY_RATE_HZ, RateLimitedOnnxPolicy
from example_helpers.seer_teleop import (
    DEFAULT_TELEOP_SCALE_FACTOR,
    SeerController,
    apply_controller_delta,
)
from example_helpers.seer_teleop import (
    gripper_command_from_trigger_fraction as _shared_gripper_command_from_trigger_fraction,
)
from example_helpers.seer_teleop import (
    pose_position_wxyz,
)
from example_helpers.spot_locomanip_policy import (
    DEFAULT_ARM_COMMAND,
    DEFAULT_BODY_ROLL_PITCH_HEIGHT,
    HEIGHT_RANGE,
    HEIGHT_STEP,
    INITIAL_ARM_COMMAND,
    POLICIES,
    POSE_COMMAND_RANGE,
    POSE_COMMAND_STEP,
    POSE_KI_LINEAR,
    POSE_KI_YAW,
    POSE_KP_LINEAR,
    POSE_KP_YAW,
    POSE_MAX_INTEGRAL,
    POSE_MAX_LINEAR_SPEED,
    POSE_MAX_YAW_SPEED,
    ROLL_PITCH_RANGE,
    ROLL_PITCH_STEP,
    VELOCITY_COMMAND_RANGE,
    VELOCITY_COMMAND_STEP,
    YAW_COMMAND_RANGE,
    YAW_COMMAND_STEP,
    LocomanipPolicy,
    available_policy_choices,
    build_policy_observation,
    parse_policy,
    policy_action_to_mujoco_ctrl,
    policy_checkpoint_path,
)
from example_helpers.spot_mjviser_adapter import (
    DEFAULT_ACTUATOR_GAINS,
    DEFAULT_LEG_GAIN_SCALE,
    LEG_GAIN_SCALE_MAX,
    LEG_GAIN_SCALE_MIN,
    MANIPULATION_OBJECT_VISUAL_SPECS,
    SpotMujocoAdapter,
    apply_default_actuator_gains,
    load_spot_mujoco_model,
    resolve_spot_scene_arm_xml,
)
from example_helpers.spot_whole_body_ik import (
    OptionalSpotWholeBodyIK,
    resolve_spot_ik_urdf,
    target_pose_from_wxyz,
)

ARM_UNSTOW_INTERPOLATION_SECONDS = 2.0
AUTO_UNSTOW_DELAY_SECONDS = 1.0
RESET_SETTLE_SECONDS = 0.5
DEFAULT_ASYNC_IK_RATE_HZ = DEFAULT_POLICY_RATE_HZ
DEFAULT_ARM_GRAVITY_COMPENSATION_ENABLED = True
GRIPPER_OPEN_COMMAND = float(INITIAL_ARM_COMMAND[6])
GRIPPER_CLOSED_COMMAND = 0.0
IK_TARGET_VISUAL_FRAME = "arm_link_wr1"


def _collision_object_link_candidates(collision_object: str) -> tuple[str, ...]:
    name = str(collision_object)
    base = name[:-2] if name.endswith("_0") else name
    candidates = [base]
    if base.startswith("arm0_"):
        candidates.append(f"arm_{base.removeprefix('arm0_')}")
    return tuple(dict.fromkeys(candidates))


def _mjviser_body_candidates_for_ik_link(link_name: str) -> tuple[str, ...]:
    candidates = [link_name]
    if link_name.startswith("arm0_"):
        candidates.append(f"arm_{link_name.removeprefix('arm0_')}")
    return tuple(dict.fromkeys(candidates))


def _mjviser_body_name_for_collision_object(
    adapter, collision_object: str
) -> str | None:
    """Map an EmbodiK collision object name to the rendered MuJoCo body name."""

    mujoco = getattr(adapter, "_mujoco", None)
    model = getattr(adapter, "model", None)
    if mujoco is None or model is None:
        return None
    for ik_link_name in _collision_object_link_candidates(collision_object):
        for body_name in _mjviser_body_candidates_for_ik_link(ik_link_name):
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) >= 0:
                return body_name
    return None


def _mjviser_body_geom_ids(
    adapter, body_name: str, *, collision_only: bool = False
) -> list[int]:
    """Return all visual/collision geoms attached to a rendered MuJoCo body."""

    mujoco = getattr(adapter, "_mujoco", None)
    model = getattr(adapter, "model", None)
    if mujoco is None or model is None:
        return []
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return []
    return [
        int(geom_id)
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == int(body_id)
        and (
            not collision_only
            or int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )
    ]


def _mjviser_nearest_body_distance(
    adapter,
    data,
    body_a: str,
    body_b: str,
    *,
    distmax: float = 2.0,
    collision_only: bool = True,
) -> SimpleNamespace | None:
    """Compute the nearest distance between rendered MuJoCo geoms on two bodies."""

    mujoco = getattr(adapter, "_mujoco", None)
    model = getattr(adapter, "model", None)
    if mujoco is None or model is None or not hasattr(mujoco, "mj_geomDistance"):
        return None
    best: SimpleNamespace | None = None
    for geom_a in _mjviser_body_geom_ids(
        adapter, body_a, collision_only=collision_only
    ):
        for geom_b in _mjviser_body_geom_ids(
            adapter, body_b, collision_only=collision_only
        ):
            fromto = np.zeros(6, dtype=np.float64)
            try:
                distance = float(
                    mujoco.mj_geomDistance(
                        model,
                        data,
                        int(geom_a),
                        int(geom_b),
                        float(distmax),
                        fromto,
                    )
                )
            except Exception:
                continue
            if not np.isfinite(distance):
                continue
            if best is None or distance < float(best.distance):
                best = SimpleNamespace(
                    distance=distance,
                    point_a_world=fromto[:3].copy(),
                    point_b_world=fromto[3:].copy(),
                    geom_a=int(geom_a),
                    geom_b=int(geom_b),
                    body_a=body_a,
                    body_b=body_b,
                )
    return best


def _mjviser_collision_debug_rows(
    adapter,
    data,
    include_pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    *,
    max_rows: int,
) -> list[SimpleNamespace]:
    """Return nearest-distance rows evaluated on the currently rendered MJCF model."""

    rows: list[SimpleNamespace] = []
    for object_a, object_b in include_pairs:
        body_a = _mjviser_body_name_for_collision_object(adapter, object_a)
        body_b = _mjviser_body_name_for_collision_object(adapter, object_b)
        if body_a is None or body_b is None:
            continue
        nearest = _mjviser_nearest_body_distance(
            adapter,
            data,
            body_a,
            body_b,
            collision_only=True,
        )
        if nearest is None:
            continue
        rows.append(
            SimpleNamespace(
                object_a=str(object_a),
                object_b=str(object_b),
                body_a=body_a,
                body_b=body_b,
                distance=float(nearest.distance),
                point_a_world=np.asarray(nearest.point_a_world, dtype=float),
                point_b_world=np.asarray(nearest.point_b_world, dtype=float),
                geom_a=int(nearest.geom_a),
                geom_b=int(nearest.geom_b),
            )
        )
    return sorted(rows, key=lambda row: float(row.distance))[: int(max_rows)]


def _collision_pairs_from_solver_debug_rows(
    rows: list[object],
    supported_pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...],
) -> list[tuple[str, str]]:
    """Extract curated pairs from solver debug rows while preserving support filtering."""

    supported_by_key = {
        frozenset(pair): (str(pair[0]), str(pair[1])) for pair in supported_pairs
    }
    selected: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()
    for row in rows:
        key = frozenset((str(row.object_a), str(row.object_b)))
        pair = supported_by_key.get(key)
        if pair is None or key in seen:
            continue
        selected.append(pair)
        seen.add(key)
    return selected


def _mjviser_collision_supported_pairs(
    adapter,
    include_pairs: list[tuple[str, str]] | tuple[tuple[str, str], ...],
) -> list[tuple[str, str]]:
    """Keep curated pairs that also have collision geoms in the rendered MJCF."""

    supported: list[tuple[str, str]] = []
    for object_a, object_b in include_pairs:
        body_a = _mjviser_body_name_for_collision_object(adapter, object_a)
        body_b = _mjviser_body_name_for_collision_object(adapter, object_b)
        if body_a is None or body_b is None:
            continue
        if not _mjviser_body_geom_ids(adapter, body_a, collision_only=True):
            continue
        if not _mjviser_body_geom_ids(adapter, body_b, collision_only=True):
            continue
        supported.append((str(object_a), str(object_b)))
    return supported


@dataclass
class CommandState:
    velocity: np.ndarray
    arm: np.ndarray
    body: np.ndarray
    desired_pose: np.ndarray
    pose_error_integral: np.ndarray


def gripper_command_from_trigger_fraction(trigger_fraction: float) -> float:
    """Map analog trigger travel to the Spot gripper command position."""

    return _shared_gripper_command_from_trigger_fraction(
        trigger_fraction,
        open_command=GRIPPER_OPEN_COMMAND,
        closed_command=GRIPPER_CLOSED_COMMAND,
    )


def _yaw_from_wxyz(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in quat_wxyz]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        return 0.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _roll_pitch_from_wxyz(quat_wxyz: np.ndarray) -> tuple[float, float]:
    w, x, y, z = [float(v) for v in quat_wxyz]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        return 0.0, 0.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_arg)
    return roll, pitch


def _target_pose_from_matrix(position: np.ndarray, rotation: np.ndarray):
    from embodik import Rt

    return Rt(R=np.asarray(rotation, dtype=float), t=np.asarray(position, dtype=float))


def schedule_interactive_unstow(controller: "SpotLocomanipController") -> None:
    """Start the app stowed, then unstow after the policy has stabilized."""

    controller.schedule_arm_unstow_after_delay(AUTO_UNSTOW_DELAY_SECONDS)


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _format_condition_number(value: float) -> str:
    if not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.0f}"


def compute_velocity_from_pose(
    current_pose: np.ndarray,
    desired_pose: np.ndarray,
    error_integral: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Pose slider controller: PID in x/y/yaw, clipped to +/-1."""

    ex = float(desired_pose[0] - current_pose[0])
    ey = float(desired_pose[1] - current_pose[1])
    yaw_cur = _yaw_from_wxyz(np.asarray(current_pose[3:7], dtype=float))
    yaw_error = _wrap_to_pi(float(desired_pose[2]) - yaw_cur)

    error_integral[:] = np.clip(
        error_integral + np.array([ex, ey, yaw_error], dtype=float) * dt,
        -POSE_MAX_INTEGRAL,
        POSE_MAX_INTEGRAL,
    )
    vx = POSE_KP_LINEAR * ex + POSE_KI_LINEAR * error_integral[0]
    vy = POSE_KP_LINEAR * ey + POSE_KI_LINEAR * error_integral[1]
    vyaw = POSE_KP_YAW * yaw_error + POSE_KI_YAW * error_integral[2]
    return np.array(
        [
            np.clip(vx, -POSE_MAX_LINEAR_SPEED, POSE_MAX_LINEAR_SPEED),
            np.clip(vy, -POSE_MAX_LINEAR_SPEED, POSE_MAX_LINEAR_SPEED),
            np.clip(vyaw, -POSE_MAX_YAW_SPEED, POSE_MAX_YAW_SPEED),
        ],
        dtype=float,
    )


class SpotLocomanipController:
    def __init__(
        self,
        model,
        *,
        policy: LocomanipPolicy,
        policy_checkpoint: Path | None = None,
        spot_urdf: Path | None = None,
        dt: float = 0.01,
        policy_rate_hz: float = DEFAULT_POLICY_RATE_HZ,
        async_ik_rate_hz: float = DEFAULT_ASYNC_IK_RATE_HZ,
        async_ik: bool = False,
        gpu_args: argparse.Namespace | None = None,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - user-facing dependency error.
            raise RuntimeError("onnxruntime is required for this example") from exc

        self.model = model
        self.adapter = SpotMujocoAdapter(model)
        self._ort = ort
        self._policy_checkpoint_override = policy_checkpoint
        self._sessions: dict[LocomanipPolicy, object] = {}
        self.policy = policy
        self.checkpoint = policy_checkpoint or policy_checkpoint_path(policy)
        self.session = None
        self.input_name = ""
        self.output_name = ""
        self.output_len = 12
        self.policy_runtime = RateLimitedOnnxPolicy(
            rate_hz=policy_rate_hz, output_len=self.output_len
        )
        self.set_policy(policy)
        self.commands = CommandState(
            velocity=np.zeros(3, dtype=float),
            arm=DEFAULT_ARM_COMMAND.copy(),
            body=DEFAULT_BODY_ROLL_PITCH_HEIGHT.copy(),
            desired_pose=np.zeros(3, dtype=float),
            pose_error_integral=np.zeros(3, dtype=float),
        )
        self.gpu_wbc = bool(
            gpu_args is not None and getattr(gpu_args, "gpu_wbc", False)
        )
        if self.gpu_wbc:
            from example_helpers.gpu_spot_locomanip import GpuSpotLocomanipIK

            self.ik = GpuSpotLocomanipIK(spot_urdf, args=gpu_args, dt=dt)
        else:
            self.ik = OptionalSpotWholeBodyIK(spot_urdf, dt=dt)
        if self.ik.collision_include_pairs:
            self.ik._collision_include_pairs = _mjviser_collision_supported_pairs(
                self.adapter,
                self.ik.collision_include_pairs,
            )
        self.step_count = 0
        self.last_ik_status = self.ik.message
        self.last_ik_solve_time_ms = 0.0
        self.last_ik_collision_constraint_time_ms = 0.0
        self.last_ik_condition_number = float("nan")
        self._last_arm_gravity_torque: np.ndarray | None = None
        # GPU arm-overlay acceptance must refer to this physics step. Async
        # command replay needs a contact/history validation contract first.
        self._async_ik_enabled = bool(async_ik and not self.gpu_wbc)
        self._async_ik_dt = 1.0 / max(float(async_ik_rate_hz), 1e-6)
        self._async_ik_elapsed = self._async_ik_dt
        self._ik_lock = threading.RLock()
        self._ik_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="spot-locomanip-ik")
            if self._async_ik_enabled
            else None
        )
        self._ik_future: Future | None = None
        self._ik_generation = 0
        self._pending_ik_request: (
            tuple[
                Mapping[str, np.ndarray],
                np.ndarray,
                object,
                np.ndarray,
                np.ndarray,
            ]
            | None
        ) = None
        self._arm_unstow_elapsed = 0.0
        self._arm_unstow_active = False
        self._arm_unstow_start = DEFAULT_ARM_COMMAND.copy()
        self._arm_unstow_target = INITIAL_ARM_COMMAND.copy()
        self._auto_unstow_delay_remaining: float | None = None
        self._reset_settle_remaining = 0.0

    def set_policy(self, policy: LocomanipPolicy) -> None:
        self.policy = policy
        self.checkpoint = self._policy_checkpoint_override or policy_checkpoint_path(
            policy
        )
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Policy checkpoint not found: {self.checkpoint}")
        if policy not in self._sessions or self._policy_checkpoint_override is not None:
            self._sessions[policy] = self._ort.InferenceSession(
                str(self.checkpoint),
                providers=["CPUExecutionProvider"],
            )
        self.session = self._sessions[policy]
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        output_shape = self.session.get_outputs()[0].shape
        self.output_len = (
            int(output_shape[-1]) if isinstance(output_shape[-1], int) else 12
        )
        self.policy_runtime.configure(
            self.session,
            input_name=self.input_name,
            output_name=self.output_name,
            output_len=self.output_len,
        )

    @property
    def last_output(self) -> np.ndarray:
        return self.policy_runtime.last_output

    @last_output.setter
    def last_output(self, value: np.ndarray) -> None:
        self.policy_runtime.last_output = np.asarray(value, dtype=np.float32)

    @property
    def policy_dt(self) -> float:
        return self.policy_runtime.policy_dt

    @property
    def policy_step_count(self) -> int:
        return self.policy_runtime.step_count

    @policy_step_count.setter
    def policy_step_count(self, value: int) -> None:
        self.policy_runtime.step_count = int(value)

    @property
    def last_policy_inference_time_ms(self) -> float:
        return self.policy_runtime.last_inference_time_ms

    def reset(self, model, data, *, settle_seconds: float = 0.0) -> None:
        self.adapter.reset_home(data)
        self._ik_generation += 1
        self._pending_ik_request = None
        self._async_ik_elapsed = self._async_ik_dt
        with self._ik_lock:
            self.ik.reset_reference()
        self.policy_runtime.reset()
        self.commands.pose_error_integral.fill(0.0)
        self.commands.velocity.fill(0.0)
        self.commands.desired_pose.fill(0.0)
        self.commands.arm[:] = DEFAULT_ARM_COMMAND
        self.commands.body[:] = DEFAULT_BODY_ROLL_PITCH_HEIGHT
        self._arm_unstow_elapsed = 0.0
        self._arm_unstow_active = False
        self._arm_unstow_start[:] = DEFAULT_ARM_COMMAND
        self._arm_unstow_target[:] = INITIAL_ARM_COMMAND
        self._auto_unstow_delay_remaining = None
        self._reset_settle_remaining = max(float(settle_seconds), 0.0)
        self.step_count = 0
        self.last_ik_solve_time_ms = 0.0
        self.last_ik_collision_constraint_time_ms = 0.0
        self.last_ik_condition_number = float("nan")
        self._last_arm_gravity_torque = None

    @property
    def ik_solve_in_flight(self) -> bool:
        return bool(self._ik_future is not None and not self._ik_future.done())

    def close(self) -> None:
        if self._ik_executor is not None:
            self._ik_executor.shutdown(wait=False, cancel_futures=True)
            self._ik_executor = None
        self._ik_future = None
        self._ik_generation += 1
        self._pending_ik_request = None

    def command_tool_pose(
        self,
        observation: Mapping[str, np.ndarray],
        arm_command: np.ndarray,
        *,
        body_command: np.ndarray,
        desired_pose_command: np.ndarray,
    ):
        with self._ik_lock:
            return self.ik.command_tool_pose(
                observation,
                arm_command,
                body_command=body_command,
                desired_pose_command=desired_pose_command,
            )

    def collision_debug_rows(self) -> list[object]:
        if self.ik_solve_in_flight:
            return []
        with self._ik_lock:
            return self.ik.collision_debug_rows()

    def sync_measured_configuration_for_debug(
        self,
        observation: Mapping[str, np.ndarray],
        arm_command: np.ndarray,
    ) -> np.ndarray:
        if self.ik_solve_in_flight:
            return np.array([], dtype=float)
        with self._ik_lock:
            return self.ik.sync_measured_configuration_for_debug(
                observation, arm_command
            )

    @property
    def arm_unstow_active(self) -> bool:
        return self._arm_unstow_active

    @property
    def arm_command_needs_gui_sync(self) -> bool:
        return (
            self._auto_unstow_delay_remaining is not None
            or self._arm_unstow_active
            or self._arm_unstow_elapsed > 0.0
        )

    def finish_arm_interpolation(self, target: np.ndarray) -> None:
        self.commands.arm[:] = target
        self._arm_unstow_start[:] = target
        self._arm_unstow_elapsed = ARM_UNSTOW_INTERPOLATION_SECONDS
        self._arm_unstow_active = False

    def start_arm_interpolation(self, target: np.ndarray) -> None:
        self._arm_unstow_start[:] = self.commands.arm
        self._arm_unstow_target = np.asarray(target, dtype=float).copy()
        self._arm_unstow_elapsed = 0.0
        self._arm_unstow_active = True

    def finish_arm_unstow_interpolation(self) -> None:
        self.finish_arm_interpolation(INITIAL_ARM_COMMAND)

    def start_arm_stow_interpolation(self) -> None:
        self.start_arm_interpolation(DEFAULT_ARM_COMMAND)

    def start_arm_unstow_interpolation(self) -> None:
        self.start_arm_interpolation(INITIAL_ARM_COMMAND)

    def schedule_arm_unstow_after_delay(self, delay_seconds: float) -> None:
        self._auto_unstow_delay_remaining = max(float(delay_seconds), 0.0)

    def _update_auto_unstow(self, dt: float) -> None:
        if self._auto_unstow_delay_remaining is None or self._arm_unstow_active:
            return
        self._auto_unstow_delay_remaining = max(
            0.0,
            self._auto_unstow_delay_remaining - max(float(dt), 0.0),
        )
        if self._auto_unstow_delay_remaining <= 0.0:
            self._auto_unstow_delay_remaining = None
            self.start_arm_unstow_interpolation()

    def _update_arm_unstow_interpolation(self, dt: float) -> None:
        if not self._arm_unstow_active:
            return
        self._arm_unstow_elapsed = min(
            ARM_UNSTOW_INTERPOLATION_SECONDS,
            self._arm_unstow_elapsed + max(float(dt), 0.0),
        )
        alpha = (
            1.0
            if ARM_UNSTOW_INTERPOLATION_SECONDS <= 0.0
            else self._arm_unstow_elapsed / ARM_UNSTOW_INTERPOLATION_SECONDS
        )
        self.commands.arm[:] = (
            1.0 - alpha
        ) * self._arm_unstow_start + alpha * self._arm_unstow_target
        if alpha >= 1.0:
            self._arm_unstow_active = False

    def _copy_ik_observation(
        self, observation: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        return {
            key: np.asarray(value, dtype=float).copy()
            for key, value in observation.items()
        }

    def _async_ik_due(self, dt: float) -> bool:
        self._async_ik_elapsed += max(float(dt), 0.0)
        if self._async_ik_elapsed + 1e-12 < self._async_ik_dt:
            return False
        self._async_ik_elapsed = math.fmod(self._async_ik_elapsed, self._async_ik_dt)
        return True

    def _run_policy(
        self,
        observation: Mapping[str, np.ndarray],
    ) -> None:
        policy_obs = build_policy_observation(
            observation,
            self.last_output,
            velocity_command=self.commands.velocity,
            arm_joint_command=self.commands.arm,
            body_roll_pitch_height_command=self.commands.body,
        )
        self.policy_runtime.run(policy_obs)

    def _run_ik_request(
        self,
        generation: int,
        request: tuple[
            Mapping[str, np.ndarray], np.ndarray, object, np.ndarray, np.ndarray
        ],
    ):
        observation, arm_command, target_pose, body_command, desired_pose_command = (
            request
        )
        with self._ik_lock:
            result = self.ik.solve_command(
                observation,
                arm_command,
                target_pose=target_pose,
                body_command=body_command,
                desired_pose_command=desired_pose_command,
            )
        return generation, result

    def _submit_ik_request(
        self,
        request: tuple[
            Mapping[str, np.ndarray], np.ndarray, object, np.ndarray, np.ndarray
        ],
    ) -> None:
        if self._ik_executor is None:
            return
        self._ik_future = self._ik_executor.submit(
            self._run_ik_request,
            self._ik_generation,
            request,
        )

    def _apply_ik_result(
        self, ik_result, *, gravity_compensation_enabled: bool
    ) -> np.ndarray | None:
        streamed_gripper_command = float(self.commands.arm[6])
        if ik_result.arm_command is not None:
            self.commands.arm[:6] = np.asarray(ik_result.arm_command, dtype=float)[:6]
            self.commands.arm[6] = streamed_gripper_command
        if ik_result.body_command is not None:
            self.commands.body[:] = np.clip(
                ik_result.body_command,
                [ROLL_PITCH_RANGE[0], ROLL_PITCH_RANGE[0], HEIGHT_RANGE[0]],
                [ROLL_PITCH_RANGE[1], ROLL_PITCH_RANGE[1], HEIGHT_RANGE[1]],
            )
        if ik_result.desired_pose_command is not None:
            self.commands.desired_pose[:] = np.clip(
                ik_result.desired_pose_command,
                [POSE_COMMAND_RANGE[0], POSE_COMMAND_RANGE[0], YAW_COMMAND_RANGE[0]],
                [POSE_COMMAND_RANGE[1], POSE_COMMAND_RANGE[1], YAW_COMMAND_RANGE[1]],
            )
        arm_gravity_torque = (
            getattr(ik_result, "arm_gravity_torque", None)
            if gravity_compensation_enabled
            else None
        )
        self.last_ik_status = ik_result.message
        self.last_ik_solve_time_ms = float(getattr(ik_result, "solve_time_ms", 0.0))
        self.last_ik_collision_constraint_time_ms = float(
            getattr(ik_result, "collision_constraint_time_ms", 0.0)
        )
        self.last_ik_condition_number = float(
            getattr(ik_result, "condition_number", float("nan"))
        )
        self._last_arm_gravity_torque = arm_gravity_torque
        return arm_gravity_torque

    def _poll_async_ik(
        self, *, gravity_compensation_enabled: bool
    ) -> np.ndarray | None:
        arm_gravity_torque = (
            self._last_arm_gravity_torque if gravity_compensation_enabled else None
        )
        if self._ik_future is not None and self._ik_future.done():
            future = self._ik_future
            self._ik_future = None
            try:
                generation, ik_result = future.result()
            except Exception as exc:
                self.last_ik_status = f"IK worker error: {exc}"
                self.last_ik_solve_time_ms = 0.0
                self.last_ik_collision_constraint_time_ms = 0.0
                self.last_ik_condition_number = float("nan")
                self._last_arm_gravity_torque = None
                arm_gravity_torque = None
            else:
                if generation == self._ik_generation:
                    arm_gravity_torque = self._apply_ik_result(
                        ik_result,
                        gravity_compensation_enabled=gravity_compensation_enabled,
                    )
        return arm_gravity_torque

    def _schedule_async_ik(
        self,
        observation: Mapping[str, np.ndarray],
        arm_command: np.ndarray,
        *,
        target_pose,
        body_command: np.ndarray,
        desired_pose_command: np.ndarray,
    ) -> None:
        if target_pose is None:
            self.last_ik_status = "ready: enable IK and drag gripper target"
            return
        request = (
            self._copy_ik_observation(observation),
            np.asarray(arm_command, dtype=float).copy(),
            target_pose,
            np.asarray(body_command, dtype=float).copy(),
            np.asarray(desired_pose_command, dtype=float).copy(),
        )
        if self._ik_future is None:
            self._submit_ik_request(request)
        else:
            self._pending_ik_request = request
            if self.ik_solve_in_flight:
                self.last_ik_status = "IK solving asynchronously"

    def step(
        self,
        model,
        data,
        *,
        control_mode: str = "pose",
        ik_enabled: bool = False,
        ik_target_pose=None,
        gravity_compensation_enabled: bool = False,
    ) -> None:
        if self._reset_settle_remaining > 0.0:
            dt = float(model.opt.timestep)
            self._reset_settle_remaining = max(
                0.0,
                self._reset_settle_remaining - dt,
            )
            self.commands.velocity.fill(0.0)
            self.commands.desired_pose.fill(0.0)
            self.commands.pose_error_integral.fill(0.0)
            self.commands.arm[:] = DEFAULT_ARM_COMMAND
            self.commands.body[:] = DEFAULT_BODY_ROLL_PITCH_HEIGHT
            observation = self.adapter.read_policy_observation(data)
            if self.policy_runtime.due(dt):
                self._run_policy(observation)
            self.adapter.clear_applied_forces(data)
            self.adapter.apply_ctrl_targets(
                data,
                policy_action_to_mujoco_ctrl(self.last_output, self.commands.arm),
            )
            self.last_ik_status = "reset settling"
            self.step_count += 1
            return
        dt = float(model.opt.timestep)
        self._update_auto_unstow(dt)
        self._update_arm_unstow_interpolation(dt)
        observation = self.adapter.read_policy_observation(data)
        if self.gpu_wbc:
            from example_helpers.gpu_spot_locomanip import stationary_contact_state

            # Observe contacts and commands at the same measured physics state.
            # The policy may walk or change support; synchronous arm IK locks
            # its base/leg coordinates and reanchors all reference rows here.
            locomotion_request = (
                self.commands.velocity.copy()
                if control_mode == "velocity"
                else self.commands.desired_pose.copy()
            )
            observation.update(
                stationary_contact_state(
                    self.adapter,
                    data,
                    locomotion_request,
                    objects_enabled=bool(self.adapter.manipulation_objects_enabled()),
                )
            )
        arm_gravity_torque = None
        # The gripper is streamed as an external joint command. It is not part
        # of the whole-body IK output, even when the IK model internally syncs
        # its current configuration from this joint angle.
        if ik_enabled:
            if self._async_ik_enabled:
                arm_gravity_torque = self._poll_async_ik(
                    gravity_compensation_enabled=gravity_compensation_enabled
                )
                if self._async_ik_due(dt):
                    self._schedule_async_ik(
                        observation,
                        self.commands.arm,
                        target_pose=ik_target_pose,
                        body_command=self.commands.body,
                        desired_pose_command=self.commands.desired_pose,
                    )
            else:
                with self._ik_lock:
                    ik_result = self.ik.solve_command(
                        observation,
                        self.commands.arm,
                        target_pose=ik_target_pose,
                        body_command=self.commands.body,
                        desired_pose_command=self.commands.desired_pose,
                    )
                arm_gravity_torque = self._apply_ik_result(
                    ik_result,
                    gravity_compensation_enabled=gravity_compensation_enabled,
                )
                if gravity_compensation_enabled and arm_gravity_torque is None:
                    with self._ik_lock:
                        arm_gravity_torque = self.ik.compute_arm_gravity_torque(
                            observation, self.commands.arm
                        )
                    self._last_arm_gravity_torque = arm_gravity_torque
        else:
            self._ik_generation += 1
            self._pending_ik_request = None
            if gravity_compensation_enabled:
                with self._ik_lock:
                    arm_gravity_torque = self.ik.compute_arm_gravity_torque(
                        observation, self.commands.arm
                    )
                self._last_arm_gravity_torque = arm_gravity_torque
            self.last_ik_status = (
                "IK solving asynchronously"
                if self.ik_solve_in_flight
                else self.ik.message
            )
            self.last_ik_solve_time_ms = 0.0
            self.last_ik_collision_constraint_time_ms = 0.0
            self.last_ik_condition_number = float("nan")
        if control_mode == "pose":
            self.commands.velocity[:] = compute_velocity_from_pose(
                observation["base_pose"],
                self.commands.desired_pose,
                self.commands.pose_error_integral,
                dt,
            )
        if self.policy_runtime.due(dt):
            self._run_policy(observation)
        targets = policy_action_to_mujoco_ctrl(self.last_output, self.commands.arm)
        self.adapter.clear_applied_forces(data)
        self.adapter.apply_ctrl_targets(data, targets)
        self.adapter.apply_arm_joint_torques(data, arm_gravity_torque)
        self.step_count += 1


@dataclass
class GuiSync:
    read_commands: Callable[[], None]
    process_teleop: Callable[[], None]
    update_status: Callable[[], None]
    sync_programmatic_controls: Callable[[], None]
    reset_to_default: Callable[[], None]
    reset_serial: Callable[[], int]
    refresh_ik_target_display: Callable[[], None]
    control_mode: Callable[[], str]
    ik_enabled: Callable[[], bool]
    ik_target_pose: Callable[[], object | None]
    gravity_compensation_enabled: Callable[[], bool]


def mujoco_world_to_mjviser_scene_position(
    world_position: np.ndarray,
    scene_offset: np.ndarray,
) -> np.ndarray:
    """Convert a MuJoCo world position to mjviser's displayed scene coordinates."""

    return np.asarray(world_position, dtype=float) + np.asarray(
        scene_offset, dtype=float
    )


def mjviser_scene_to_mujoco_world_position(
    scene_position: np.ndarray,
    scene_offset: np.ndarray,
) -> np.ndarray:
    """Convert an mjviser displayed scene position back to MuJoCo world coordinates."""

    return np.asarray(scene_position, dtype=float) - np.asarray(
        scene_offset, dtype=float
    )


class WorldFixedIkTarget:
    """Keep a viser transform control anchored in MuJoCo world coordinates."""

    def __init__(
        self,
        world_position: np.ndarray,
        quat_wxyz: np.ndarray,
        scene_offset: Callable[[], np.ndarray],
    ):
        self._scene_offset = scene_offset
        self.world_position = np.asarray(world_position, dtype=float).copy()
        self.wxyz = np.asarray(quat_wxyz, dtype=float).copy()

    def set_from_world(self, world_position: np.ndarray, quat_wxyz: np.ndarray) -> None:
        self.world_position = np.asarray(world_position, dtype=float).copy()
        self.wxyz = np.asarray(quat_wxyz, dtype=float).copy()

    def update_from_handle(self, handle) -> None:
        self.world_position = mjviser_scene_to_mujoco_world_position(
            np.asarray(handle.position, dtype=float),
            self._scene_offset(),
        )
        self.wxyz = np.asarray(handle.wxyz, dtype=float).copy()

    def apply_to_handle(self, handle) -> None:
        handle.position = tuple(
            float(v)
            for v in mujoco_world_to_mjviser_scene_position(
                self.world_position,
                self._scene_offset(),
            )
        )
        handle.wxyz = tuple(float(v) for v in self.wxyz)


def _add_gui(
    server,
    controller: SpotLocomanipController,
    data,
    *,
    initial_control_mode: str,
    initial_manipulation_objects_enabled: bool = False,
    scene_offset: Callable[[], np.ndarray] | None = None,
    with_sim_lock: Callable[[Callable[[], None]], None] | None = None,
    teleop_controller: SeerController | None = None,
    initial_teleop_scale: float = DEFAULT_TELEOP_SCALE_FACTOR,
) -> GuiSync:
    def run_with_sim_lock(action: Callable[[], None]) -> None:
        if with_sim_lock is None:
            action()
            return
        with_sim_lock(action)

    def current_scene_offset() -> np.ndarray:
        if scene_offset is None:
            return np.zeros(3, dtype=float)
        return np.asarray(scene_offset(), dtype=float)

    def world_to_scene_position(world_position: np.ndarray) -> np.ndarray:
        return mujoco_world_to_mjviser_scene_position(
            world_position, current_scene_offset()
        )

    def manipulation_visual_pose(
        geom_name: str,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        pos, wxyz = controller.adapter.geom_pose_wxyz(data, geom_name)
        return (
            tuple(float(v) for v in world_to_scene_position(pos)),
            tuple(float(v) for v in wxyz),
        )

    manipulation_object_handles = {}
    for name, spec in MANIPULATION_OBJECT_VISUAL_SPECS.items():
        position, wxyz = manipulation_visual_pose(str(spec["geom_name"]))
        manipulation_object_handles[name] = server.scene.add_box(
            f"/spot_manipulation_objects/{name}",
            dimensions=spec["dimensions"],
            color=spec["color"],
            position=position,
            wxyz=wxyz,
            visible=initial_manipulation_objects_enabled,
        )

    def sync_manipulation_object_visuals(visible: bool) -> None:
        for name, spec in MANIPULATION_OBJECT_VISUAL_SPECS.items():
            handle = manipulation_object_handles[name]
            if visible:
                handle.position, handle.wxyz = manipulation_visual_pose(
                    str(spec["geom_name"])
                )
            handle.visible = visible

    with server.gui.add_folder("Locomanip policy"):
        policy_dropdown = server.gui.add_dropdown(
            "Policy",
            options=available_policy_choices(),
            initial_value=controller.policy.value,
        )
        checkpoint_text = server.gui.add_text(
            "Checkpoint", initial_value=controller.checkpoint.name, disabled=True
        )
        policy_note = server.gui.add_text(
            "Policy note",
            initial_value=POLICIES[controller.policy].description,
            disabled=True,
        )
        control_mode = server.gui.add_dropdown(
            "Command mode",
            options=("pose", "velocity"),
            initial_value=initial_control_mode,
        )
        reset_sim_btn = server.gui.add_button("Reset sim to default")
        manipulation_objects_checkbox = server.gui.add_checkbox(
            "Spawn manipulation objects",
            initial_value=initial_manipulation_objects_enabled,
        )
        if not controller.adapter.has_manipulation_objects():
            manipulation_objects_checkbox.disabled = True
        leg_gain_scale_slider = server.gui.add_slider(
            "Leg gain scale",
            LEG_GAIN_SCALE_MIN,
            LEG_GAIN_SCALE_MAX,
            step=0.05,
            initial_value=DEFAULT_LEG_GAIN_SCALE,
        )
    with server.gui.add_folder("Torso pose command"):
        x_des = server.gui.add_slider(
            "x-des",
            POSE_COMMAND_RANGE[0],
            POSE_COMMAND_RANGE[1],
            step=POSE_COMMAND_STEP,
            initial_value=0.0,
        )
        y_des = server.gui.add_slider(
            "y-des",
            POSE_COMMAND_RANGE[0],
            POSE_COMMAND_RANGE[1],
            step=POSE_COMMAND_STEP,
            initial_value=0.0,
        )
        yaw_des = server.gui.add_slider(
            "yaw-des",
            YAW_COMMAND_RANGE[0],
            YAW_COMMAND_RANGE[1],
            step=YAW_COMMAND_STEP,
            initial_value=0.0,
        )
        zero_pose_btn = server.gui.add_button("Set pose command to zero")
        hold_pose_btn = server.gui.add_button("Stop: hold current torso pose")
    with server.gui.add_folder("Velocity command"):
        x_cmd = server.gui.add_slider(
            "x-command",
            VELOCITY_COMMAND_RANGE[0],
            VELOCITY_COMMAND_RANGE[1],
            step=VELOCITY_COMMAND_STEP,
            initial_value=0.0,
        )
        y_cmd = server.gui.add_slider(
            "y-command",
            VELOCITY_COMMAND_RANGE[0],
            VELOCITY_COMMAND_RANGE[1],
            step=VELOCITY_COMMAND_STEP,
            initial_value=0.0,
        )
        yaw_cmd = server.gui.add_slider(
            "yaw-command",
            VELOCITY_COMMAND_RANGE[0],
            VELOCITY_COMMAND_RANGE[1],
            step=VELOCITY_COMMAND_STEP,
            initial_value=0.0,
        )
        zero_velocity_btn = server.gui.add_button("Set velocity command to zero")
    with server.gui.add_folder("Arm command"):
        arm_sliders = [
            server.gui.add_slider(
                "arm_sh0",
                -2.61799,
                3.14159,
                step=0.02,
                initial_value=float(controller.commands.arm[0]),
            ),
            server.gui.add_slider(
                "arm_sh1",
                -3.14159,
                0.523599,
                step=0.02,
                initial_value=float(controller.commands.arm[1]),
            ),
            server.gui.add_slider(
                "arm_el0",
                0.0,
                3.14159,
                step=0.02,
                initial_value=float(controller.commands.arm[2]),
            ),
            server.gui.add_slider(
                "arm_el1",
                -2.79253,
                2.79253,
                step=0.02,
                initial_value=float(controller.commands.arm[3]),
            ),
            server.gui.add_slider(
                "arm_wr0",
                -1.8326,
                1.8326,
                step=0.02,
                initial_value=float(controller.commands.arm[4]),
            ),
            server.gui.add_slider(
                "arm_wr1",
                -2.87979,
                2.87979,
                step=0.02,
                initial_value=float(controller.commands.arm[5]),
            ),
            server.gui.add_slider(
                "arm_f1x",
                -1.57,
                0.0,
                step=0.02,
                initial_value=float(controller.commands.arm[6]),
            ),
        ]
    with server.gui.add_folder("Body command"):
        roll = server.gui.add_slider(
            "roll",
            ROLL_PITCH_RANGE[0],
            ROLL_PITCH_RANGE[1],
            step=ROLL_PITCH_STEP,
            initial_value=float(controller.commands.body[0]),
        )
        pitch = server.gui.add_slider(
            "pitch",
            ROLL_PITCH_RANGE[0],
            ROLL_PITCH_RANGE[1],
            step=ROLL_PITCH_STEP,
            initial_value=float(controller.commands.body[1]),
        )
        height = server.gui.add_slider(
            "height",
            HEIGHT_RANGE[0],
            HEIGHT_RANGE[1],
            step=HEIGHT_STEP,
            initial_value=float(controller.commands.body[2]),
        )
        reset_body_btn = server.gui.add_button("Reset body command")
    gripper_pos, gripper_wxyz = controller.adapter.body_pose_wxyz(
        data, IK_TARGET_VISUAL_FRAME
    )
    ik_target_state = WorldFixedIkTarget(
        gripper_pos, gripper_wxyz, current_scene_offset
    )
    ik_visual_reference_position = gripper_pos.copy()
    ik_visual_reference_rotation = np.asarray(
        target_pose_from_wxyz(gripper_pos, gripper_wxyz).rotation,
        dtype=float,
    )
    ik_command_reference_pose = None
    ik_target = server.scene.add_transform_controls(
        "/spot_gripper_ik_target",
        scale=0.18,
        position=tuple(float(v) for v in world_to_scene_position(gripper_pos)),
        wxyz=tuple(float(v) for v in gripper_wxyz),
    )
    ik_target.visible = False
    ik_target_armed = False
    suppress_ik_target_update = False
    last_ik_target_scene_offset = current_scene_offset().copy()
    reset_serial = 0
    with server.gui.add_folder("EmbodiK IK"):
        ik_checkbox = server.gui.add_checkbox(
            "Enable gripper IK",
            initial_value=False,
        )
        gravity_checkbox = server.gui.add_checkbox(
            "Enable arm gravity torque",
            initial_value=DEFAULT_ARM_GRAVITY_COMPENSATION_ENABLED,
        )
        base_assist_slider = server.gui.add_slider(
            "Base assist",
            0.25,
            4.0,
            step=0.05,
            initial_value=float(controller.ik.locomotion_sensitivity),
        )
        arm_recovery_bias_slider = server.gui.add_slider(
            "Arm recovery bias",
            1.0,
            6.0,
            step=0.25,
            initial_value=float(
                getattr(controller.ik, "condition_arm_weight_scale", 3.0)
            ),
        )
        collision_available = bool(getattr(controller.ik, "collision_available", False))
        collision_debug_available = bool(
            controller.ik.enabled and controller.ik.collision_include_pairs
        )
        collision_enable = server.gui.add_checkbox(
            "Enable collision constraint",
            initial_value=bool(controller.ik.enable_collision),
            disabled=not collision_available,
        )
        collision_min_dist_mm = server.gui.add_slider(
            "Collision min dist (mm)",
            0.0,
            100.0,
            step=1.0,
            initial_value=float(getattr(controller.ik, "collision_min_distance", 0.05))
            * 1e3,
        )
        collision_max_rows = server.gui.add_slider(
            "Closest collision checks",
            1,
            8,
            step=1,
            initial_value=int(getattr(controller.ik, "collision_max_constraints", 3)),
        )
        collision_tuning = server.gui.add_dropdown(
            "Collision tuning",
            options=COLLISION_TUNING_OPTIONS,
            initial_value=str(
                getattr(controller.ik, "collision_tuning_mode", "balanced")
            ),
        )
        show_collision_debug = server.gui.add_checkbox(
            "Show collision debug",
            initial_value=False,
            disabled=not collision_debug_available,
        )
        collision_log_mode = server.gui.add_checkbox(
            "Collision debug logging",
            initial_value=False,
            disabled=not collision_debug_available,
        )
        ik_position_gain = server.gui.add_slider(
            "IK position gain",
            1.0,
            120.0,
            step=1.0,
            initial_value=float(getattr(controller.ik, "position_gain", 60.0)),
        )
        ik_orientation_gain = server.gui.add_slider(
            "IK orientation gain",
            1.0,
            120.0,
            step=1.0,
            initial_value=float(getattr(controller.ik, "orientation_gain", 60.0)),
        )
        ik_solve_mode = server.gui.add_dropdown(
            "IK target solve mode",
            options=(
                ("GPU_SRINV",)
                if controller.gpu_wbc
                else ("SCALE_ELASTIC", "MIN_ERROR", "SCALE")
            ),
            initial_value=str(
                getattr(controller.ik, "target_solve_mode", "SCALE_ELASTIC")
            ),
        )
        ik_dt = server.gui.add_slider(
            "IK dt (s)",
            0.002,
            0.03,
            step=0.001,
            initial_value=float(getattr(controller.ik, "dt", 0.01)),
        )
        ik_max_steps = server.gui.add_slider(
            "IK steps per frame",
            1,
            10,
            step=1,
            initial_value=int(getattr(controller.ik, "max_steps", 1)),
        )
        ik_adaptive_dt = server.gui.add_checkbox(
            "IK adaptive dt",
            initial_value=bool(getattr(controller.ik, "adaptive_dt", True)),
        )
        ik_adaptive_dt_max_scale = server.gui.add_slider(
            "IK adaptive dt max scale",
            1.0,
            10.0,
            step=0.5,
            initial_value=float(getattr(controller.ik, "adaptive_dt_max_scale", 3.0)),
        )
        ik_adaptive_dt_ref_dist = server.gui.add_slider(
            "IK adaptive dt ref dist",
            0.005,
            0.2,
            step=0.005,
            initial_value=float(
                getattr(controller.ik, "adaptive_dt_reference_distance", 0.04)
            ),
        )
        if not controller.ik.enabled:
            ik_checkbox.disabled = True
            gravity_checkbox.disabled = True
            base_assist_slider.disabled = True
            arm_recovery_bias_slider.disabled = True
            for handle in (
                ik_position_gain,
                ik_orientation_gain,
                ik_solve_mode,
                ik_dt,
                ik_max_steps,
                ik_adaptive_dt,
                ik_adaptive_dt_max_scale,
                ik_adaptive_dt_ref_dist,
            ):
                handle.disabled = True
        collision_min_dist_mm.disabled = not collision_available
        collision_max_rows.disabled = not collision_available or controller.gpu_wbc
        collision_tuning.disabled = not collision_available or controller.gpu_wbc
        if controller.gpu_wbc:
            base_assist_slider.disabled = True
            arm_recovery_bias_slider.disabled = True
            server.gui.add_markdown(
                "GPU: synchronous arm IK during walking/contact transitions, self-collision, posture, "
                "torso bounds and adaptive dt. Base/legs remain policy-owned and locked in IK. "
                "Torso commands remain policy-owned. CoM and object dynamics hold IK; async is disabled; "
                "collision capacity is fixed at launch. CPU collision tuning does not apply."
            )
        arm_command_group = server.gui.add_button_group(
            "Arm command",
            options=("Stow", "Unstow"),
        )
        seed_ik_target_btn = server.gui.add_button("Move IK target to gripper")
    teleop_connected = bool(
        teleop_controller is not None and teleop_controller.connected
    )
    with server.gui.add_folder("Seer teleop", expand_by_default=teleop_connected):
        teleop_enabled_checkbox = server.gui.add_checkbox(
            "Enable teleop",
            initial_value=teleop_connected,
            disabled=not teleop_connected,
        )
        teleop_connected_text = server.gui.add_text(
            "Controller",
            initial_value="connected" if teleop_connected else "browser only",
            disabled=True,
        )
        teleop_streaming_text = server.gui.add_text(
            "Streaming", initial_value="OFF", disabled=True
        )
        teleop_gripper_text = server.gui.add_text(
            "Gripper", initial_value="0% closed", disabled=True
        )
        teleop_reset_text = server.gui.add_text(
            "Reset", initial_value="Button B", disabled=True
        )
        teleop_scale_slider = server.gui.add_slider(
            "Position Scale",
            0.5,
            3.0,
            step=0.1,
            initial_value=float(initial_teleop_scale),
            disabled=not teleop_connected,
        )
        teleop_scale_slider.disabled = not (
            teleop_connected and bool(teleop_enabled_checkbox.value)
        )
    with server.gui.add_folder("Status"):
        step_text = server.gui.add_text(
            "Policy steps", initial_value="0", disabled=True
        )
        ik_text = server.gui.add_text(
            "IK overlay", initial_value=controller.last_ik_status, disabled=True
        )
        velocity_text = server.gui.add_text(
            "Policy velocity command", initial_value="0, 0, 0", disabled=True
        )
        target_error_text = server.gui.add_text(
            "IK target delta", initial_value="disabled", disabled=True
        )
        collision_debug_text = server.gui.add_text(
            "Minimum collision vector",
            initial_value="Collision: --",
            disabled=True,
        )
        timing_text = server.gui.add_text(
            "Computation time",
            initial_value="IK 0.00 ms | collision 0.00 ms | cond -- | debug off",
            disabled=True,
        )
        collision_status_text = server.gui.add_text(
            "Collision pairs",
            initial_value=(
                str(len(controller.ik.collision_include_pairs))
                if controller.ik.enabled
                else "disabled"
            ),
            disabled=True,
        )
        actuator_gains_text = server.gui.add_text(
            "Actuator gains",
            initial_value=(
                "example PD: "
                f"legs {DEFAULT_ACTUATOR_GAINS['fl_hx'][0]:g}/"
                f"{DEFAULT_ACTUATOR_GAINS['fl_hx'][1]:g}, "
                f"arm {DEFAULT_ACTUATOR_GAINS['arm_sh0'][0]:g}/"
                f"{DEFAULT_ACTUATOR_GAINS['arm_sh0'][1]:g}, "
                f"gripper {DEFAULT_ACTUATOR_GAINS['arm_f1x'][0]:g}/"
                f"{DEFAULT_ACTUATOR_GAINS['arm_f1x'][1]:g}"
            ),
            disabled=True,
        )

    last_collision_log_key = None
    last_collision_log_time = 0.0
    last_collision_debug_update_time = 0.0
    last_collision_debug_time_ms = 0.0
    collision_debug_colors = (
        ((1.0, 0.2, 0.2), (0.2, 0.8, 0.2)),
        ((1.0, 0.5, 0.0), (0.3, 0.7, 1.0)),
        ((0.9, 0.2, 0.9), (0.2, 0.9, 0.9)),
        ((1.0, 0.9, 0.2), (0.2, 0.6, 1.0)),
    )
    collision_debug_points_a = [
        server.scene.add_icosphere(
            f"/spot_collision_debug/point_a_{i}",
            radius=0.012,
            color=color_a,
            visible=False,
        )
        for i, (color_a, _color_b) in enumerate(collision_debug_colors)
    ]
    collision_debug_points_b = [
        server.scene.add_icosphere(
            f"/spot_collision_debug/point_b_{i}",
            radius=0.012,
            color=color_b,
            visible=False,
        )
        for i, (_color_a, color_b) in enumerate(collision_debug_colors)
    ]
    collision_debug_lines = [None for _ in collision_debug_colors]

    def update_actuator_gains_text(leg_gain_scale: float) -> None:
        leg_kp = 60.0 * leg_gain_scale
        leg_kd = 1.5 * leg_gain_scale
        actuator_gains_text.value = (
            "example PD: "
            f"legs {leg_kp:g}/{leg_kd:g} "
            f"(scale {leg_gain_scale:.2f}), "
            f"arm {DEFAULT_ACTUATOR_GAINS['arm_sh0'][0]:g}/"
            f"{DEFAULT_ACTUATOR_GAINS['arm_sh0'][1]:g}, "
            f"gripper {DEFAULT_ACTUATOR_GAINS['arm_f1x'][0]:g}/"
            f"{DEFAULT_ACTUATOR_GAINS['arm_f1x'][1]:g}"
        )

    def sync_actuator_gains() -> None:
        leg_gain_scale = float(leg_gain_scale_slider.value)
        apply_default_actuator_gains(controller.model, leg_gain_scale=leg_gain_scale)
        update_actuator_gains_text(leg_gain_scale)

    def sync_solver_preference_sliders() -> None:
        controller.ik.locomotion_sensitivity = float(base_assist_slider.value)
        if hasattr(controller.ik, "condition_arm_weight_scale"):
            controller.ik.condition_arm_weight_scale = float(
                arm_recovery_bias_slider.value
            )

    def sync_ik_runtime_options() -> None:
        controller.ik.position_gain = float(ik_position_gain.value)
        controller.ik.orientation_gain = float(ik_orientation_gain.value)
        controller.ik.target_solve_mode = str(ik_solve_mode.value)
        controller.ik.dt = float(ik_dt.value)
        controller.ik.max_steps = int(ik_max_steps.value)
        controller.ik.adaptive_dt = bool(ik_adaptive_dt.value)
        controller.ik.adaptive_dt_max_scale = float(ik_adaptive_dt_max_scale.value)
        controller.ik.adaptive_dt_reference_distance = float(
            ik_adaptive_dt_ref_dist.value
        )
        controller.ik.apply_runtime_options()

    def sync_collision_options() -> None:
        controller.ik.enable_collision = bool(
            collision_available and collision_enable.value
        )
        controller.ik.collision_min_distance = float(collision_min_dist_mm.value) * 1e-3
        controller.ik.collision_max_constraints = int(collision_max_rows.value)
        controller.ik.collision_tuning_mode = str(collision_tuning.value)
        collision_status_text.value = (
            f"{len(controller.ik.collision_include_pairs)}"
            if controller.ik.enable_collision and controller.ik.collision_include_pairs
            else f"off ({len(controller.ik.collision_include_pairs)} available)"
        )

    def clear_collision_debug() -> None:
        nonlocal last_collision_debug_time_ms
        for point in collision_debug_points_a + collision_debug_points_b:
            point.visible = False
        for line in collision_debug_lines:
            if line is not None:
                line.visible = False
        collision_debug_text.value = "Collision: --"
        last_collision_debug_time_ms = 0.0

    def update_collision_debug(*, force: bool = False) -> None:
        nonlocal collision_debug_lines, last_collision_log_key, last_collision_log_time
        nonlocal last_collision_debug_update_time, last_collision_debug_time_ms
        if controller.gpu_wbc:
            controller.ik.include_collision_debug = bool(show_collision_debug.value)
        if not (
            show_collision_debug.value
            and ik_checkbox.value
            and controller.ik.enabled
            and controller.ik.collision_include_pairs
        ):
            clear_collision_debug()
            return
        now = time.time()
        if not force and now - last_collision_debug_update_time < 0.2:
            return
        last_collision_debug_update_time = now
        debug_start = time.perf_counter()
        solver_debug_rows = (
            controller.collision_debug_rows() if controller.ik.enable_collision else []
        )
        active_pairs = _collision_pairs_from_solver_debug_rows(
            solver_debug_rows,
            controller.ik.collision_include_pairs,
        )
        if not active_pairs:
            active_pairs = list(controller.ik.collision_include_pairs)
        max_visible_rows = max(
            1,
            min(
                len(collision_debug_colors),
                int(collision_max_rows.value),
            ),
        )
        debug_rows = (
            solver_debug_rows[:max_visible_rows]
            if controller.gpu_wbc
            else _mjviser_collision_debug_rows(
                controller.adapter,
                data,
                active_pairs,
                max_rows=max_visible_rows,
            )
        )
        last_collision_debug_time_ms = (time.perf_counter() - debug_start) * 1e3
        if not debug_rows:
            clear_collision_debug()
            return
        summaries: list[str] = []
        scene_offset_now = current_scene_offset()
        visible_rows = debug_rows[: len(collision_debug_colors)]
        for i, row in enumerate(visible_rows):
            p_a_world = np.asarray(row.point_a_world, dtype=float)
            p_b_world = np.asarray(row.point_b_world, dtype=float)
            p_a = mujoco_world_to_mjviser_scene_position(p_a_world, scene_offset_now)
            p_b = mujoco_world_to_mjviser_scene_position(p_b_world, scene_offset_now)
            collision_debug_points_a[i].position = tuple(float(v) for v in p_a)
            collision_debug_points_b[i].position = tuple(float(v) for v in p_b)
            collision_debug_points_a[i].visible = True
            collision_debug_points_b[i].visible = True
            if collision_debug_lines[i] is not None:
                collision_debug_lines[i].remove()
            segment = np.zeros((1, 2, 3), dtype=float)
            segment[0, 0, :] = p_a
            segment[0, 1, :] = p_b
            color_a, color_b = collision_debug_colors[i]
            collision_debug_lines[i] = server.scene.add_line_segments(
                f"/spot_collision_debug/segment_{i}",
                points=segment,
                colors=np.array([[color_a, color_b]], dtype=float),
                line_width=3.0,
                visible=True,
            )
            vector = p_b_world - p_a_world
            summaries.append(
                f"{row.object_a} <-> {row.object_b} | "
                f"{getattr(row, 'body_a', row.object_a)} <-> {getattr(row, 'body_b', row.object_b)}, "
                f"d={float(row.distance):.4f} m, "
                f"v=[{vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f}]"
            )
        for i in range(len(visible_rows), len(collision_debug_colors)):
            collision_debug_points_a[i].visible = False
            collision_debug_points_b[i].visible = False
            if collision_debug_lines[i] is not None:
                collision_debug_lines[i].visible = False
        collision_debug_text.value = " || ".join(summaries)
        first = visible_rows[0]
        log_key = (
            str(first.object_a),
            str(first.object_b),
            round(float(first.distance), 4),
            len(debug_rows),
        )
        if collision_log_mode.value and (
            log_key != last_collision_log_key or now - last_collision_log_time > 1.0
        ):
            print(
                "[spot-locomanip][collision]",
                f"rows={len(debug_rows)}",
                f"{first.object_a} <-> {first.object_b}",
                f"d={float(first.distance):.4f} m",
                f"vector={np.asarray(first.point_b_world, dtype=float) - np.asarray(first.point_a_world, dtype=float)}",
            )
            last_collision_log_key = log_key
            last_collision_log_time = now

    def sync_manipulation_objects() -> None:
        enabled = bool(manipulation_objects_checkbox.value)
        controller.adapter.set_manipulation_objects_enabled(
            data,
            enabled,
        )
        sync_manipulation_object_visuals(enabled)

    def read_commands() -> None:
        controller.commands.velocity[:] = [x_cmd.value, y_cmd.value, yaw_cmd.value]
        if not ik_checkbox.value:
            controller.commands.desired_pose[:] = [
                x_des.value,
                y_des.value,
                yaw_des.value,
            ]
            if not controller.arm_unstow_active:
                controller.commands.arm[:] = [slider.value for slider in arm_sliders]
            controller.commands.body[:] = [roll.value, pitch.value, height.value]

    def sync_arm_body_controls() -> None:
        x_des.value = float(controller.commands.desired_pose[0])
        y_des.value = float(controller.commands.desired_pose[1])
        yaw_des.value = float(controller.commands.desired_pose[2])
        for slider, value in zip(arm_sliders, controller.commands.arm):
            slider.value = float(value)
        roll.value = float(controller.commands.body[0])
        pitch.value = float(controller.commands.body[1])
        height.value = float(controller.commands.body[2])

    def update_status() -> None:
        nonlocal ik_target_armed
        step_text.value = str(controller.step_count)
        ik_text.value = controller.last_ik_status
        checkpoint_text.value = controller.checkpoint.name
        policy_note.value = POLICIES[controller.policy].description
        velocity_text.value = ", ".join(
            f"{value:+.3f}" for value in controller.commands.velocity
        )
        if ik_checkbox.value and ik_target_armed:
            current_pos, _ = controller.adapter.body_pose_wxyz(
                data, IK_TARGET_VISUAL_FRAME
            )
            target_pos = ik_target_state.world_position
            target_error_text.value = (
                f"dx={target_pos[0] - current_pos[0]:+.3f}, "
                f"dy={target_pos[1] - current_pos[1]:+.3f}, "
                f"dz={target_pos[2] - current_pos[2]:+.3f}, "
                f"|d|={np.linalg.norm(target_pos - current_pos):.3f} m"
            )
        elif ik_checkbox.value:
            target_error_text.value = "move target to start IK"
        else:
            target_error_text.value = "disabled"
        if ik_checkbox.value or controller.arm_command_needs_gui_sync:
            sync_arm_body_controls()
        update_collision_debug()
        collision_debug_label = (
            f"{last_collision_debug_time_ms:.2f} ms"
            if show_collision_debug.value and controller.ik.collision_include_pairs
            else "off"
        )
        timing_text.value = (
            f"policy {controller.last_policy_inference_time_ms:.2f} ms @ "
            f"{1.0 / controller.policy_dt:.0f} Hz | "
            f"IK {controller.last_ik_solve_time_ms:.2f} ms | "
            f"collision {controller.last_ik_collision_constraint_time_ms:.2f} ms | "
            f"cond {_format_condition_number(controller.last_ik_condition_number)} | "
            f"debug {collision_debug_label}"
        )

    def sync_programmatic_controls() -> None:
        """Mirror non-slider-owned command changes before read_commands can overwrite them."""

        if ik_checkbox.value or controller.arm_command_needs_gui_sync:
            sync_arm_body_controls()

    def switch_policy(value: str) -> None:
        controller.set_policy(parse_policy(value))
        controller.commands.pose_error_integral[:] = 0.0
        update_status()

    def set_pose_zero() -> None:
        x_des.value = 0.0
        y_des.value = 0.0
        yaw_des.value = 0.0
        controller.commands.pose_error_integral[:] = 0.0
        read_commands()

    def set_velocity_zero() -> None:
        x_cmd.value = 0.0
        y_cmd.value = 0.0
        yaw_cmd.value = 0.0
        controller.commands.pose_error_integral[:] = 0.0
        read_commands()

    def hold_current_pose() -> None:
        base_pose = controller.adapter.read_policy_observation(data)["base_pose"]
        x_des.value = float(base_pose[0])
        y_des.value = float(base_pose[1])
        yaw_des.value = _yaw_from_wxyz(base_pose[3:7])
        set_velocity_zero()

    def run_arm_command(command: str) -> None:
        read_commands()
        if command == "Stow":
            controller.start_arm_stow_interpolation()
        else:
            controller.start_arm_unstow_interpolation()
        sync_arm_body_controls()

    def reset_body() -> None:
        roll.value = float(DEFAULT_BODY_ROLL_PITCH_HEIGHT[0])
        pitch.value = float(DEFAULT_BODY_ROLL_PITCH_HEIGHT[1])
        height.value = float(DEFAULT_BODY_ROLL_PITCH_HEIGHT[2])
        read_commands()

    def reset_sim_to_default() -> None:
        nonlocal ik_target_armed, reset_serial
        reset_serial += 1
        ik_checkbox.value = False
        ik_target.visible = False
        ik_target_armed = False
        teleop_enabled = bool(
            teleop_controller is not None
            and teleop_controller.connected
            and teleop_enabled_checkbox.value
        )
        if teleop_controller is not None:
            teleop_controller.set_enabled(teleop_enabled)
            teleop_controller.reset_reference()
            teleop_controller.reset_runtime_state()
        teleop_streaming_text.value = "OFF"
        teleop_gripper_text.value = "0% closed" if teleop_enabled else "disabled"
        teleop_reset_text.value = "Button B" if teleop_enabled else "enable teleop"
        teleop_scale_slider.disabled = not teleop_enabled
        controller.ik.reset_reference()
        controller.reset(controller.model, data, settle_seconds=RESET_SETTLE_SECONDS)
        sync_actuator_gains()
        sync_manipulation_objects()
        x_des.value = 0.0
        y_des.value = 0.0
        yaw_des.value = 0.0
        x_cmd.value = 0.0
        y_cmd.value = 0.0
        yaw_cmd.value = 0.0
        controller.commands.velocity.fill(0.0)
        controller.commands.desired_pose.fill(0.0)
        schedule_interactive_unstow(controller)
        controller.commands.body[:] = DEFAULT_BODY_ROLL_PITCH_HEIGHT
        controller.adapter.apply_ctrl_targets(
            data,
            policy_action_to_mujoco_ctrl(
                controller.last_output, controller.commands.arm
            ),
        )
        seed_ik_target_from_gripper(sync_arm_command=False)
        ik_target.visible = False
        sync_arm_body_controls()
        update_status()

    def seed_ik_target_from_gripper(*, sync_arm_command: bool = True) -> None:
        nonlocal ik_target_armed, ik_visual_reference_position, ik_visual_reference_rotation
        nonlocal ik_command_reference_pose
        if sync_arm_command:
            controller.commands.arm[:] = controller.adapter.read_measured_arm_command(
                data
            )
        pos, wxyz = controller.adapter.body_pose_wxyz(data, IK_TARGET_VISUAL_FRAME)
        ik_target_state.set_from_world(pos, wxyz)
        visual_reference_pose = target_pose_from_wxyz(pos, wxyz)
        ik_visual_reference_position = np.asarray(pos, dtype=float).copy()
        ik_visual_reference_rotation = np.asarray(
            visual_reference_pose.rotation, dtype=float
        )
        ik_command_reference_pose = controller.command_tool_pose(
            controller.adapter.read_policy_observation(data),
            controller.commands.arm,
            body_command=controller.commands.body,
            desired_pose_command=controller.commands.desired_pose,
        )
        apply_ik_target_handle()
        ik_target_armed = False
        controller.ik.reset_reference()

    def apply_ik_target_handle() -> None:
        nonlocal last_ik_target_scene_offset, suppress_ik_target_update
        suppress_ik_target_update = True
        try:
            ik_target_state.apply_to_handle(ik_target)
            last_ik_target_scene_offset = current_scene_offset().copy()
        finally:
            suppress_ik_target_update = False

    def refresh_ik_target_display() -> None:
        nonlocal last_ik_target_scene_offset
        scene_offset_now = current_scene_offset()
        scene_offset_changed = not np.allclose(
            scene_offset_now, last_ik_target_scene_offset
        )
        sync_manipulation_object_visuals(bool(manipulation_objects_checkbox.value))
        if ik_checkbox.value and scene_offset_changed:
            apply_ik_target_handle()
        elif scene_offset_changed:
            last_ik_target_scene_offset = scene_offset_now.copy()
        update_collision_debug()

    def set_ik_enabled() -> None:
        ik_target.visible = bool(ik_checkbox.value)
        if ik_checkbox.value:
            seed_ik_target_from_gripper()
        else:
            controller.ik.reset_reference()

    def current_ik_target_pose():
        if not ik_checkbox.value or not ik_target_armed:
            return None
        visual_target = target_pose_from_wxyz(
            ik_target_state.world_position,
            ik_target_state.wxyz,
        )
        if ik_command_reference_pose is None:
            return visual_target
        visual_position = np.asarray(visual_target.translation, dtype=float)
        visual_rotation = np.asarray(visual_target.rotation, dtype=float)
        command_reference_position = np.asarray(
            ik_command_reference_pose.translation, dtype=float
        )
        command_reference_rotation = np.asarray(
            ik_command_reference_pose.rotation, dtype=float
        )
        visual_delta_rotation = visual_rotation @ ik_visual_reference_rotation.T
        return _target_pose_from_matrix(
            command_reference_position
            + (visual_position - ik_visual_reference_position),
            visual_delta_rotation @ command_reference_rotation,
        )

    def update_ik_target_from_handle() -> None:
        nonlocal ik_target_armed, last_ik_target_scene_offset
        if suppress_ik_target_update:
            return
        if (
            teleop_controller is not None
            and bool(teleop_enabled_checkbox.value)
            and teleop_controller.streaming
        ):
            # While A is held, the controller owns the IK target. Ignore any
            # simultaneous browser-gizmo update so the two input modes do not
            # fight over the same target pose.
            apply_ik_target_handle()
            return
        ik_target_state.update_from_handle(ik_target)
        last_ik_target_scene_offset = current_scene_offset().copy()
        ik_target_armed = True

    teleop_stream_start_pose = target_pose_from_wxyz(
        ik_target_state.world_position,
        ik_target_state.wxyz,
    )

    def set_ik_target_from_pose(pose) -> None:
        nonlocal ik_target_armed
        position, wxyz = pose_position_wxyz(pose)
        ik_target_state.set_from_world(position, wxyz)
        apply_ik_target_handle()
        ik_target_armed = True

    def start_teleop_streaming() -> None:
        nonlocal teleop_stream_start_pose
        if teleop_controller is None:
            return
        if ik_checkbox.disabled:
            return
        if not ik_checkbox.value:
            ik_checkbox.value = True
            set_ik_enabled()
        teleop_controller.reset_reference()
        teleop_stream_start_pose = target_pose_from_wxyz(
            ik_target_state.world_position,
            ik_target_state.wxyz,
        )
        teleop_streaming_text.value = "ON"

    def stop_teleop_streaming() -> None:
        teleop_streaming_text.value = "OFF"

    def set_teleop_enabled() -> None:
        enabled = (
            bool(teleop_enabled_checkbox.value)
            and teleop_controller is not None
            and teleop_controller.connected
        )
        teleop_scale_slider.disabled = not enabled
        if teleop_controller is not None:
            teleop_controller.set_enabled(enabled)
            if enabled:
                teleop_controller.reset_reference()
        if not enabled:
            teleop_streaming_text.value = "OFF"

    def process_teleop() -> None:
        if teleop_controller is None:
            return
        teleop_connected_text.value = (
            "connected" if teleop_controller.connected else "browser only"
        )
        if not (teleop_controller.connected and bool(teleop_enabled_checkbox.value)):
            teleop_controller.set_enabled(False)
            teleop_gripper_text.value = "disabled"
            teleop_reset_text.value = "enable teleop"
            return
        teleop_controller.set_enabled(True)
        teleop_controller.process_buttons()
        trigger_fraction = float(getattr(teleop_controller, "trigger_fraction", 0.0))
        teleop_gripper_text.value = f"{100.0 * trigger_fraction:.0f}% closed"
        teleop_reset_text.value = "Button B"
        gripper_command = gripper_command_from_trigger_fraction(trigger_fraction)
        controller.commands.arm[6] = gripper_command
        arm_sliders[6].value = gripper_command
        if not (teleop_controller.streaming and ik_checkbox.value):
            return
        delta = teleop_controller.relative_pose()
        if delta is None:
            return
        delta_pos, delta_wxyz = delta
        pose = apply_controller_delta(
            teleop_stream_start_pose,
            delta_pos,
            delta_wxyz,
            float(teleop_scale_slider.value),
        )
        set_ik_target_from_pose(pose)

    if teleop_controller is not None:
        teleop_controller.on_stream_start = start_teleop_streaming
        teleop_controller.on_stream_stop = stop_teleop_streaming
        teleop_controller.on_reset = reset_sim_to_default

    policy_dropdown.on_update(lambda _: switch_policy(str(policy_dropdown.value)))
    for handle in [
        x_des,
        y_des,
        yaw_des,
        x_cmd,
        y_cmd,
        yaw_cmd,
        *arm_sliders,
        roll,
        pitch,
        height,
    ]:
        handle.on_update(lambda _: read_commands())
    zero_pose_btn.on_click(lambda _: set_pose_zero())
    hold_pose_btn.on_click(lambda _: hold_current_pose())
    zero_velocity_btn.on_click(lambda _: set_velocity_zero())
    reset_body_btn.on_click(lambda _: reset_body())
    reset_sim_btn.on_click(lambda _: run_with_sim_lock(reset_sim_to_default))
    manipulation_objects_checkbox.on_update(
        lambda _: run_with_sim_lock(sync_manipulation_objects)
    )
    leg_gain_scale_slider.on_update(lambda _: run_with_sim_lock(sync_actuator_gains))
    base_assist_slider.on_update(lambda _: sync_solver_preference_sliders())
    arm_recovery_bias_slider.on_update(lambda _: sync_solver_preference_sliders())
    for handle in (
        ik_position_gain,
        ik_orientation_gain,
        ik_solve_mode,
        ik_dt,
        ik_max_steps,
        ik_adaptive_dt,
        ik_adaptive_dt_max_scale,
        ik_adaptive_dt_ref_dist,
    ):
        handle.on_update(lambda _: sync_ik_runtime_options())
    for handle in (
        collision_enable,
        collision_min_dist_mm,
        collision_max_rows,
        collision_tuning,
    ):
        handle.on_update(
            lambda _: (sync_collision_options(), update_collision_debug(force=True))
        )
    show_collision_debug.on_update(lambda _: update_collision_debug(force=True))
    ik_checkbox.on_update(lambda _: set_ik_enabled())
    teleop_enabled_checkbox.on_update(lambda _: set_teleop_enabled())
    ik_target.on_update(lambda _: update_ik_target_from_handle())
    arm_command_group.on_click(lambda _: run_arm_command(str(arm_command_group.value)))
    seed_ik_target_btn.on_click(lambda _: seed_ik_target_from_gripper())

    read_commands()
    sync_solver_preference_sliders()
    sync_ik_runtime_options()
    sync_collision_options()
    sync_actuator_gains()
    sync_manipulation_objects()
    set_teleop_enabled()
    set_ik_enabled()
    update_status()
    return GuiSync(
        read_commands=read_commands,
        process_teleop=process_teleop,
        update_status=update_status,
        sync_programmatic_controls=sync_programmatic_controls,
        reset_to_default=reset_sim_to_default,
        reset_serial=lambda: reset_serial,
        refresh_ik_target_display=refresh_ik_target_display,
        control_mode=lambda: str(control_mode.value),
        ik_enabled=lambda: bool(ik_checkbox.value),
        ik_target_pose=current_ik_target_pose,
        gravity_compensation_enabled=lambda: bool(gravity_checkbox.value),
    )


def run_headless(args: argparse.Namespace) -> None:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "mujoco is required for --headless. From a Pixi checkout, run this "
            "through the mjviser environment, for example "
            "`pixi run -e mjviser spot-locomanip-mjviser --headless`."
        ) from exc

    gpu_wbc = bool(getattr(args, "gpu_wbc", False))
    model = load_spot_mujoco_model(
        Path(args.scene) if args.scene else None,
        apply_default_gains=args.default_gains,
    )
    data = mujoco.MjData(model)
    controller = SpotLocomanipController(
        model,
        policy=parse_policy(args.policy),
        policy_checkpoint=(
            Path(args.policy_checkpoint) if args.policy_checkpoint else None
        ),
        spot_urdf=_resolve_optional_spot_urdf(args.spot_urdf),
        dt=args.dt,
        policy_rate_hz=float(getattr(args, "policy_rate_hz", DEFAULT_POLICY_RATE_HZ)),
        async_ik_rate_hz=float(
            getattr(args, "async_ik_rate_hz", DEFAULT_ASYNC_IK_RATE_HZ)
        ),
        gpu_args=args,
    )
    controller.reset(model, data)
    controller.adapter.set_manipulation_objects_enabled(
        data,
        bool(args.spawn_manipulation_objects),
    )
    target = None
    if gpu_wbc:
        observation = controller.adapter.read_policy_observation(data)
        target = controller.command_tool_pose(
            observation,
            controller.commands.arm,
            body_command=controller.commands.body,
            desired_pose_command=controller.commands.desired_pose,
        )
        target = target_pose_from_wxyz(
            np.asarray(target.translation)
            + np.array([args.gpu_wbc_target_offset, 0.0, 0.0]),
            np.asarray(controller.ik._eik.r2q(target.rotation, order="xyzs"))[
                [3, 0, 1, 2]
            ],
        )
    gpu_accepted_steps = 0
    initial_arm_command = controller.commands.arm.copy()
    maximum_arm_command_motion = 0.0
    for _ in range(args.steps):
        controller.step(
            model,
            data,
            control_mode=args.control_mode,
            ik_enabled=gpu_wbc,
            ik_target_pose=target,
        )
        if gpu_wbc and controller.last_ik_status.startswith(
            ("GPU SUCCESS", "GPU SAFE_STEP")
        ):
            gpu_accepted_steps += 1
        maximum_arm_command_motion = max(
            maximum_arm_command_motion,
            float(np.linalg.norm(controller.commands.arm - initial_arm_command)),
        )
        mujoco.mj_step(model, data)
        if (
            not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or not np.isfinite(data.ctrl).all()
        ):
            raise RuntimeError("MuJoCo state became non-finite")
    gpu_summary = ""
    if gpu_wbc:
        if gpu_accepted_steps == 0:
            raise RuntimeError(
                f"GPU IK never accepted a step: {controller.last_ik_status}"
            )
        result = getattr(controller.ik, "last_result", None)
        collision_distance = getattr(result, "minimum_collision_distance_m", None)
        gpu_summary = (
            f", gpu_accepted_steps={gpu_accepted_steps}, "
            f"arm_command_motion={maximum_arm_command_motion:.6g}"
            + (
                ""
                if collision_distance is None
                else f", collision_distance={float(collision_distance):.4f}m"
            )
        )
    print(
        f"Ran {args.steps} headless steps with policy={controller.policy.value}, "
        f"checkpoint={controller.checkpoint.name}, scene={resolve_spot_scene_arm_xml().name}, "
        f"gains={'example PD' if args.default_gains else 'menagerie raw'}, "
        f"objects={'on' if args.spawn_manipulation_objects else 'off'}, "
        f"IK={controller.last_ik_status}{gpu_summary}"
    )


def configure_spot_viewer_defaults(viewer) -> None:
    """Use a fixed world camera by default for gripper IK teleoperation."""

    viewer.scene.camera_tracking_enabled = False


def run_viewer(args: argparse.Namespace) -> None:
    try:
        import mujoco
        import viser
        from mjviser import Viewer
    except ImportError as exc:
        raise RuntimeError(
            "mujoco, viser, and mjviser are required for interactive mode. "
            "From a Pixi checkout, run "
            "`pixi run -e mjviser spot-locomanip-mjviser` or, for Seer teleop, "
            "`pixi run -e mjviser-teleop spot-locomanip-mjviser --enable-teleop`."
        ) from exc

    model = load_spot_mujoco_model(
        Path(args.scene) if args.scene else None,
        apply_default_gains=args.default_gains,
    )
    data = mujoco.MjData(model)
    controller = SpotLocomanipController(
        model,
        policy=parse_policy(args.policy),
        policy_checkpoint=(
            Path(args.policy_checkpoint) if args.policy_checkpoint else None
        ),
        spot_urdf=_resolve_optional_spot_urdf(args.spot_urdf),
        dt=args.dt,
        policy_rate_hz=float(getattr(args, "policy_rate_hz", DEFAULT_POLICY_RATE_HZ)),
        async_ik_rate_hz=float(
            getattr(args, "async_ik_rate_hz", DEFAULT_ASYNC_IK_RATE_HZ)
        ),
        async_ik=not bool(args.sync_ik),
        gpu_args=args,
    )
    controller.reset(model, data)
    controller.adapter.set_manipulation_objects_enabled(
        data,
        bool(args.spawn_manipulation_objects),
    )
    schedule_interactive_unstow(controller)
    teleop_controller = SeerController(
        args.controller_port if args.enable_teleop else None
    )
    teleop_controller.connect()
    server = viser.ViserServer(port=args.port, label="EmbodiK Spot locomanipulation")

    last_status_update = 0.0
    gui_sync: GuiSync | None = None

    def reset_viewer_runtime_state() -> None:
        try:
            viewer._step_count = 0
            viewer._budget = 0.0
            viewer._last_tick = time.perf_counter()
            viewer._time_until_next_render = 0.0
            viewer._dirty = True
        except NameError:
            pass

    def reset_fn(model, data) -> None:
        if gui_sync is None:
            controller.reset(model, data)
            reset_viewer_runtime_state()
            return
        gui_sync.reset_to_default()
        reset_viewer_runtime_state()

    def step_fn(model, data) -> None:
        nonlocal last_status_update
        if gui_sync is None:
            return
        reset_serial_before = gui_sync.reset_serial()
        gui_sync.read_commands()
        gui_sync.process_teleop()
        if gui_sync.reset_serial() != reset_serial_before:
            reset_viewer_runtime_state()
            mujoco.mj_forward(model, data)
            gui_sync.refresh_ik_target_display()
            gui_sync.update_status()
            return
        controller.step(
            model,
            data,
            control_mode=gui_sync.control_mode(),
            ik_enabled=gui_sync.ik_enabled(),
            ik_target_pose=gui_sync.ik_target_pose(),
            gravity_compensation_enabled=gui_sync.gravity_compensation_enabled(),
        )
        gui_sync.sync_programmatic_controls()
        mujoco.mj_step(model, data)
        gui_sync.refresh_ik_target_display()
        now = time.time()
        if now - last_status_update > 0.2:
            last_status_update = now
            gui_sync.update_status()

    viewer = Viewer(
        model,
        data,
        step_fn=step_fn,
        reset_fn=reset_fn,
        server=server,
    )
    configure_spot_viewer_defaults(viewer)

    def scene_offset() -> np.ndarray:
        tracked_body_id = getattr(viewer.scene, "_tracked_body_id", None)
        if viewer.scene.camera_tracking_enabled and tracked_body_id is not None:
            return -np.asarray(data.xpos[int(tracked_body_id)], dtype=float).copy()
        return np.zeros(3, dtype=float)

    def with_viewer_lock(action: Callable[[], None]) -> None:
        with viewer._lock:
            action()
            reset_viewer_runtime_state()
            viewer._render()
            viewer._update_status_display()
        viewer._sync_sliders()

    gui_sync = _add_gui(
        server,
        controller,
        data,
        initial_control_mode=args.control_mode,
        initial_manipulation_objects_enabled=args.spawn_manipulation_objects,
        scene_offset=scene_offset,
        with_sim_lock=with_viewer_lock,
        teleop_controller=teleop_controller,
        initial_teleop_scale=args.teleop_scale,
    )
    try:
        viewer.run()
    finally:
        controller.close()
        teleop_controller.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=available_policy_choices(),
        default=LocomanipPolicy.LOCOMANIP.value,
        help="Locomanipulation checkpoint variant to run.",
    )
    parser.add_argument(
        "--policy-checkpoint", type=str, default=None, help="Override ONNX path."
    )
    parser.add_argument(
        "--scene", type=str, default=None, help="Override MuJoCo scene XML path."
    )
    parser.add_argument(
        "--spot-urdf",
        type=str,
        default=None,
        help=(
            "Optional compatible Spot URDF path for the EmbodiK IK overlay. "
            "If omitted, EMBODIK_SPOT_IK_URDF is used when set."
        ),
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run without Viser/mjviser UI."
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Headless simulation steps."
    )
    parser.add_argument(
        "--dt", type=float, default=0.01, help="Controller timestep for IK hooks."
    )
    parser.add_argument(
        "--policy-rate-hz",
        type=float,
        default=DEFAULT_POLICY_RATE_HZ,
        help="ONNX policy inference rate. The default is 50 Hz; MuJoCo still steps at its model rate.",
    )
    parser.add_argument(
        "--async-ik-rate-hz",
        type=float,
        default=DEFAULT_ASYNC_IK_RATE_HZ,
        help=(
            "Background IK solve request rate. The default keeps collision-constrained IK "
            "at the policy rate while preventing solve backlogs from saturating the simulation/render thread."
        ),
    )
    parser.add_argument(
        "--spawn-manipulation-objects",
        action="store_true",
        help="Start with the optional manipulation table, cube, and obstacle blocks enabled.",
    )
    parser.set_defaults(default_gains=True)
    parser.add_argument(
        "--default-gains",
        action="store_true",
        help="Use the example PD gains. This is the default.",
    )
    parser.add_argument(
        "--raw-menagerie-gains",
        dest="default_gains",
        action="store_false",
        help="Keep MuJoCo Menagerie's raw actuator gains for comparison.",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_VISER_PORT, help="Viser server port."
    )
    parser.add_argument(
        "--enable-teleop",
        action="store_true",
        help="Enable Seer/xvisio controller teleoperation. Without this flag, xvisio is not imported.",
    )
    parser.add_argument(
        "--controller-port",
        default="/dev/ttyUSB0",
        help="Seer/xvisio controller serial port used with --enable-teleop.",
    )
    parser.add_argument(
        "--scale",
        "--teleop-scale",
        dest="teleop_scale",
        type=float,
        default=DEFAULT_TELEOP_SCALE_FACTOR,
        help="Position scale for Seer controller relative motion.",
    )
    parser.add_argument(
        "--control-mode",
        choices=("pose", "velocity"),
        default="pose",
        help="Use pose sliders or expose direct velocity commands.",
    )
    parser.add_argument(
        "--sync-ik",
        action="store_true",
        help="Run IK synchronously on the viewer thread instead of the default background worker.",
    )
    parser.add_argument(
        "--gpu-wbc",
        action="store_true",
        help="Synchronous GPU arm overlay during policy walking/contact transitions; CoM and object dynamics hold IK; async disabled.",
    )
    parser.add_argument("--gpu-wbc-manifest", type=Path)
    parser.add_argument("--gpu-wbc-cache-dir", type=Path)
    parser.add_argument(
        "--gpu-wbc-collision",
        action="store_true",
        help="Enable GPU self-collision constraints at launch.",
    )
    parser.add_argument(
        "--gpu-wbc-target-offset",
        type=float,
        default=0.02,
        help="Headless GPU tool target x offset in metres.",
    )
    return parser


def _resolve_optional_spot_urdf(value: str | None) -> Path | None:
    return resolve_spot_ik_urdf(value)


def main() -> None:
    args = build_parser().parse_args()
    if args.headless:
        run_headless(args)
    else:
        run_viewer(args)


if __name__ == "__main__":
    main()
