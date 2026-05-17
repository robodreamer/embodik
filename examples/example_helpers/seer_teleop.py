"""Shared xvisio/Seer teleoperation helpers for interactive examples."""

from __future__ import annotations

import importlib
import time
from typing import Any, Callable, Optional, Tuple

import numpy as np

from embodik import Rt, q2r, r2q

DEFAULT_TELEOP_SCALE_FACTOR = 1.5
BUTTON_A = 16
BUTTON_B = 32
TRIGGER_THRESHOLD = 30
# xvisio reports the Seer analog trigger on the same 0-90-ish scale used by
# the original teleop thresholding path, not as an 8-bit 0-255 axis.
TRIGGER_MAX_VALUE = 90.0
RESET_DEBOUNCE_S = 2.0


class SeerController:
    """Thin adapter from Seer controller events to reusable teleop state."""

    def __init__(self, port: str | None):
        self.port = port
        self.device: Optional[Any] = None
        self.enabled = True
        self.streaming = False
        self.gripper_closed = False
        self.trigger_value = 0.0
        self.trigger_fraction = 0.0
        self._prev_trigger = 0
        self._prev_side = 0
        self._prev_key = 0
        self._last_button_time = 0.0
        self.on_stream_start: Callable[[], None] = lambda: None
        self.on_stream_stop: Callable[[], None] = lambda: None
        self.on_reset: Callable[[], None] = lambda: None

    @property
    def connected(self) -> bool:
        return self.device is not None

    def connect(self) -> bool:
        if not self.port:
            return False

        try:
            xvisio = importlib.import_module("xvisio")
        except Exception as exc:
            print(f"xvisio not available ({exc}); using browser transform controls.")
            return False

        try:
            controllers = xvisio.discover_controllers()
            if not controllers:
                print("No Seer controllers found; using browser transform controls.")
                return False

            print(f"Found controller: {controllers[0]}")
            self.device = xvisio.open_controller(port=self.port)
            time.sleep(0.5)
            self.reset_reference()
            return True
        except Exception as exc:  # pragma: no cover - hardware-dependent.
            print(f"Controller connection failed: {exc}")
            self.device = None
            return False

    def disconnect(self) -> None:
        if self.device is not None:
            self.device.close()
            self.device = None

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            was_streaming = self.streaming
            self.reset_runtime_state()
            if was_streaming:
                self.on_stream_stop()

    def reset_reference(self) -> None:
        if self.device is not None:
            self.device.reset_controller_reference()

    def reset_runtime_state(self) -> None:
        """Clear latched button/trigger state without touching the hardware reference."""

        self.streaming = False
        self.gripper_closed = False
        self.trigger_value = 0.0
        self.trigger_fraction = 0.0
        self._prev_trigger = 0
        self._prev_side = 0
        self._prev_key = 0

    def relative_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.device is None:
            return None
        try:
            left, right = self.device.controller_relative()
            c = right if right is not None else left
            if c is None:
                return None
            return (
                np.asarray(c.position, dtype=float),
                np.asarray(c.quat_wxyz, dtype=float),
            )
        except Exception:  # pragma: no cover - hardware-dependent.
            return None

    def raw_buttons(self) -> Optional[Tuple[int, int, int]]:
        if self.device is None:
            return None
        try:
            left, right = self.device.controller()
            c = right if right is not None else left
            if c is None:
                return None
            return int(c.key_trigger), int(c.key_side), int(c.key)
        except Exception:  # pragma: no cover - hardware-dependent.
            return None

    def process_buttons(self) -> None:
        if not self.enabled:
            self.set_enabled(False)
            return

        raw = self.raw_buttons()
        if raw is None:
            return

        trigger, side, key = raw
        now = time.time()

        a_pressed = bool(key & BUTTON_A)
        b_pressed = bool(key & BUTTON_B)
        prev_a_pressed = bool(self._prev_key & BUTTON_A)
        prev_b_pressed = bool(self._prev_key & BUTTON_B)

        if not prev_a_pressed and a_pressed:
            self.streaming = True
            self.on_stream_start()
        elif prev_a_pressed and not a_pressed:
            self.streaming = False
            self.on_stream_stop()

        self.trigger_value = float(np.clip(trigger, 0.0, TRIGGER_MAX_VALUE))
        self.trigger_fraction = self.trigger_value / TRIGGER_MAX_VALUE
        self.gripper_closed = trigger > TRIGGER_THRESHOLD

        if (
            not prev_b_pressed
            and b_pressed
            and now - self._last_button_time > RESET_DEBOUNCE_S
        ):
            self.on_reset()
            self._last_button_time = now

        self._prev_trigger = trigger
        self._prev_side = side
        self._prev_key = key


def gripper_command_from_trigger_fraction(
    trigger_fraction: float,
    *,
    open_command: float,
    closed_command: float = 0.0,
) -> float:
    """Map analog trigger travel to a continuous gripper joint command."""

    alpha = float(np.clip(trigger_fraction, 0.0, 1.0))
    return (1.0 - alpha) * float(open_command) + alpha * float(closed_command)


def pose_from_transform_control(control: Any) -> Rt:
    """Convert a Viser transform-control pose to an ``Rt`` pose."""

    wxyz = np.asarray(control.wxyz, dtype=float)
    xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=float)
    return Rt(R=q2r(xyzw, order="xyzs"), t=np.asarray(control.position, dtype=float))


def apply_controller_delta(
    base_pose: Any,
    delta_pos: np.ndarray,
    delta_wxyz: np.ndarray,
    scale: float,
) -> Rt:
    """Apply a Seer relative pose delta to a robot/world target pose."""

    delta_xyzw = np.array(
        [delta_wxyz[1], delta_wxyz[2], delta_wxyz[3], delta_wxyz[0]],
        dtype=float,
    )
    delta_R = q2r(delta_xyzw, order="xyzs")
    return Rt(
        R=delta_R @ np.asarray(base_pose.rotation, dtype=float),
        t=np.asarray(base_pose.translation, dtype=float) + float(scale) * np.asarray(delta_pos, dtype=float),
    )


def pose_position_wxyz(pose: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(position, wxyz)`` for an ``Rt``/Pinocchio-style pose."""

    quat_xyzw = np.asarray(r2q(np.asarray(pose.rotation, dtype=float), order="xyzs"), dtype=float)
    return (
        np.asarray(pose.translation, dtype=float),
        np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float),
    )


def set_transform_control_pose(control: Any, pose: Any) -> None:
    """Write an ``Rt``/Pinocchio-style pose to a Viser transform control."""

    position, wxyz = pose_position_wxyz(pose)
    control.position = tuple(float(v) for v in position)
    control.wxyz = tuple(float(v) for v in wxyz)
