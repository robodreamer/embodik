"""Dual-controller Seer teleop state + target mapping for the bimanual app.

One Seer device exposes a (left, right) controller pair; this maps each side's
relative pose onto a per-arm target pose. No Viser dependency, so it is unit
testable with a fake device and plain Rt poses. Gripper is display-only.

Button mapping used by the shared bimanual controller layout:
  - right: hold Button A to stream
  - left: hold Button B to stream; release Button A to reset robot + targets
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from embodik import Rt, q2r, r2q

from .seer_teleop import (
    BUTTON_A,
    BUTTON_B,
    RESET_DEBOUNCE_S,
    TRIGGER_MAX_VALUE,
    TRIGGER_THRESHOLD,
    SeerController,
    apply_controller_delta,
)

SIDES = ("left", "right")

# Shared bimanual stream / reset controls.
STREAM_BUTTON_BY_SIDE = {"right": BUTTON_A, "left": BUTTON_B}
ROBOT_RESET_BUTTON_BY_SIDE = {"left": BUTTON_A}


def _relative_wxyz_from_origin(origin_wxyz: np.ndarray, current_wxyz: np.ndarray) -> np.ndarray:
    """Rotation delta (wxyz) from a per-side stream origin to the current reading."""

    origin_xyzw = np.array(
        [origin_wxyz[1], origin_wxyz[2], origin_wxyz[3], origin_wxyz[0]], dtype=float
    )
    current_xyzw = np.array(
        [current_wxyz[1], current_wxyz[2], current_wxyz[3], current_wxyz[0]], dtype=float
    )
    delta_R = q2r(current_xyzw, order="xyzs") @ q2r(origin_xyzw, order="xyzs").T
    delta_xyzw = r2q(delta_R, order="xyzs")
    return np.array([delta_xyzw[3], delta_xyzw[0], delta_xyzw[1], delta_xyzw[2]], dtype=float)


@dataclass
class ArmTeleopState:
    side: str
    enabled: bool = True
    connected: bool = False
    streaming: bool = False
    base_pose: Optional[Rt] = None  # arm target pose captured at stream start
    rel_origin_pos: Optional[np.ndarray] = None  # set when 2nd side streams without global reset
    rel_origin_wxyz: Optional[np.ndarray] = None
    trigger_fraction: float = 0.0
    gripper_closed: bool = False
    buttons: tuple[int, int, int] = (0, 0, 0)
    new_pose: Optional[Rt] = None  # set per step(); None means "hold"
    _prev_key: int = 0
    _last_reset_t: float = -1.0e9


@dataclass
class BimanualSeerTeleop:
    """Owns one Seer device read per-side and maps deltas to per-arm targets."""

    scale: float = 1.5
    port: Optional[str] = None
    controller: SeerController = field(default=None)  # type: ignore[assignment]
    arms: dict[str, ArmTeleopState] = field(default_factory=dict)
    on_reset_robot: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if self.controller is None:
            self.controller = SeerController(port=self.port)
        self.arms = {s: ArmTeleopState(side=s) for s in SIDES}

    @property
    def connected(self) -> bool:
        return self.controller.connected

    def is_side_connected(self, side: str) -> bool:
        return bool(self.arms[side].connected)

    def connect_side(self, side: str) -> bool:
        """Open the shared Seer device (if needed) and mark ``side`` as connected."""
        if not self.controller.connected:
            if not self.controller.connect():
                return False
        self.arms[side].connected = True
        return True

    def disconnect_side(self, side: str) -> None:
        """Mark ``side`` disconnected; close the device when no sides remain."""
        st = self.arms[side]
        st.connected = False
        st.streaming = False
        st.base_pose = None
        st.rel_origin_pos = None
        st.rel_origin_wxyz = None
        if not any(arm.connected for arm in self.arms.values()):
            self.controller.disconnect()

    def connect(self) -> bool:
        """Connect both controller sides on the shared Seer device."""
        if self.controller.connected:
            for side in SIDES:
                self.arms[side].connected = True
            return True
        if not self.controller.connect():
            return False
        for side in SIDES:
            self.arms[side].connected = True
        return True

    def disconnect(self) -> None:
        """Disconnect both sides and close the device."""
        for side in SIDES:
            self.arms[side].connected = False
            self.arms[side].streaming = False
            self.arms[side].base_pose = None
            self.arms[side].rel_origin_pos = None
            self.arms[side].rel_origin_wxyz = None
        self.controller.disconnect()

    def set_arm_enabled(self, side: str, enabled: bool) -> None:
        self.arms[side].enabled = bool(enabled)
        if not enabled:
            self.arms[side].streaming = False
            self.arms[side].rel_origin_pos = None
            self.arms[side].rel_origin_wxyz = None

    def _any_other_streaming(self, side: str) -> bool:
        return any(st.streaming for other, st in self.arms.items() if other != side)

    def begin_streaming(self, side: str, base_pose: Rt) -> None:
        st = self.arms[side]
        if self._any_other_streaming(side):
            rel = self.controller.relative_pose_side(side)
            if rel is not None:
                st.rel_origin_pos = np.asarray(rel[0], dtype=float).copy()
                st.rel_origin_wxyz = np.asarray(rel[1], dtype=float).copy()
            else:
                st.rel_origin_pos = np.zeros(3, dtype=float)
                st.rel_origin_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            self.controller.reset_reference()
            st.rel_origin_pos = None
            st.rel_origin_wxyz = None
        st.streaming = True
        st.base_pose = base_pose

    def reset_references(self) -> None:
        self.controller.reset_reference()

    def step(self, *, left_target: Rt, right_target: Rt, now: float = 0.0) -> dict[str, ArmTeleopState]:
        targets = {"left": left_target, "right": right_target}
        for side in SIDES:
            st = self.arms[side]
            st.new_pose = None
            if not st.enabled or not st.connected:
                continue
            self._process_side_buttons(st, now=now, current_target=targets[side])
            if st.streaming and st.base_pose is not None:
                rel = self.controller.relative_pose_side(side)
                if rel is not None:
                    delta_pos = np.asarray(rel[0], dtype=float)
                    delta_wxyz = np.asarray(rel[1], dtype=float)
                    if st.rel_origin_pos is not None and st.rel_origin_wxyz is not None:
                        delta_pos = delta_pos - st.rel_origin_pos
                        delta_wxyz = _relative_wxyz_from_origin(st.rel_origin_wxyz, delta_wxyz)
                    st.new_pose = apply_controller_delta(
                        st.base_pose, delta_pos, delta_wxyz, float(self.scale)
                    )
        return self.arms

    def _process_side_buttons(self, st: ArmTeleopState, *, now: float, current_target: Rt) -> None:
        raw = self.controller.raw_buttons_side(st.side)
        if raw is None:
            return
        trigger, _side, key = raw

        stream_btn = STREAM_BUTTON_BY_SIDE[st.side]
        stream_pressed = bool(key & stream_btn)
        prev_stream_pressed = bool(st._prev_key & stream_btn)
        if stream_pressed and not prev_stream_pressed:
            self.begin_streaming(st.side, current_target)
        elif prev_stream_pressed and not stream_pressed:
            st.streaming = False
            st.rel_origin_pos = None
            st.rel_origin_wxyz = None

        reset_btn = ROBOT_RESET_BUTTON_BY_SIDE.get(st.side)
        if reset_btn is not None:
            reset_pressed = bool(key & reset_btn)
            prev_reset_pressed = bool(st._prev_key & reset_btn)
            if (
                prev_reset_pressed
                and not reset_pressed
                and (now - st._last_reset_t) > RESET_DEBOUNCE_S
                and self.on_reset_robot is not None
            ):
                self.on_reset_robot()
                st._last_reset_t = now

        st.trigger_fraction = float(np.clip(trigger, 0.0, TRIGGER_MAX_VALUE)) / TRIGGER_MAX_VALUE
        st.gripper_closed = trigger > TRIGGER_THRESHOLD
        st.buttons = (int(trigger), int(_side), int(key))
        st._prev_key = key
