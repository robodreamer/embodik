"""Tests for shared Seer/xvisio teleoperation helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

pytest.importorskip("embodik")

from example_helpers import seer_teleop  # noqa: E402


def test_seer_controller_does_not_import_xvisio_without_explicit_port(monkeypatch) -> None:
    calls: list[str] = []

    def fake_import_module(name: str):
        calls.append(name)
        raise AssertionError("xvisio should not be imported when no controller port is provided")

    monkeypatch.setattr(seer_teleop.importlib, "import_module", fake_import_module)

    controller = seer_teleop.SeerController(None)

    assert controller.connect(auto_detect_port=False) is False
    assert calls == []


def test_connect_without_port_can_auto_detect_serial_devices(monkeypatch) -> None:
    monkeypatch.setattr(seer_teleop, "_available_serial_ports", lambda: ["/dev/ttyUSB1"])
    monkeypatch.setattr(seer_teleop, "_port_accessible", lambda port: (True, "OK"))

    class FakeDevice:
        def reset_controller_reference(self):
            pass

        def close(self):
            pass

    fake_xvisio = SimpleNamespace(
        discover_controllers=lambda: ["seer"],
        open_controller=lambda port: FakeDevice(),
    )
    monkeypatch.setattr(seer_teleop.importlib, "import_module", lambda name: fake_xvisio)

    controller = seer_teleop.SeerController(None)
    assert controller.connect() is True
    assert controller.port == "/dev/ttyUSB1"


def test_candidate_controller_ports_prefers_requested_port() -> None:
    ports = seer_teleop._candidate_controller_ports(
        "/dev/ttyUSB0",
        auto_detect_port=True,
    )
    assert ports == ["/dev/ttyUSB0"]


def test_seer_controller_imports_xvisio_only_when_port_is_provided(monkeypatch) -> None:
    class FakeDevice:
        def __init__(self):
            self.reset_count = 0
            self.closed = False

        def reset_controller_reference(self):
            self.reset_count += 1

        def close(self):
            self.closed = True

    fake_device = FakeDevice()
    fake_xvisio = SimpleNamespace(
        discover_controllers=lambda: ["seer"],
        open_controller=lambda port: fake_device,
    )
    calls: list[str] = []

    def fake_import_module(name: str):
        calls.append(name)
        assert name == "xvisio"
        return fake_xvisio

    monkeypatch.setattr(seer_teleop, "_port_accessible", lambda port: (True, "OK"))
    monkeypatch.setattr(seer_teleop.importlib, "import_module", fake_import_module)

    controller = seer_teleop.SeerController("/dev/ttyUSB0")

    assert controller.connect() is True
    assert calls == ["xvisio"]
    assert controller.connected
    assert fake_device.reset_count == 1
    controller.disconnect()
    assert fake_device.closed


def test_connect_is_idempotent_when_already_connected(monkeypatch) -> None:
    open_count = 0

    class FakeDevice:
        def reset_controller_reference(self):
            pass

        def close(self):
            pass

    def open_controller(port):
        nonlocal open_count
        open_count += 1
        return FakeDevice()

    fake_xvisio = SimpleNamespace(
        discover_controllers=lambda: ["seer"],
        open_controller=open_controller,
    )
    monkeypatch.setattr(seer_teleop, "_port_accessible", lambda port: (True, "OK"))
    monkeypatch.setattr(seer_teleop.importlib, "import_module", lambda name: fake_xvisio)

    controller = seer_teleop.SeerController("/dev/ttyUSB0")
    assert controller.connect() is True
    assert controller.connect() is True
    assert open_count == 1


def test_connect_rejects_while_connection_in_progress(monkeypatch) -> None:
    class FakeDevice:
        def reset_controller_reference(self):
            pass

        def close(self):
            pass

    fake_xvisio = SimpleNamespace(
        discover_controllers=lambda: ["seer"],
        open_controller=lambda port: FakeDevice(),
    )
    monkeypatch.setattr(seer_teleop, "_port_accessible", lambda port: (True, "OK"))
    monkeypatch.setattr(seer_teleop.importlib, "import_module", lambda name: fake_xvisio)

    controller = seer_teleop.SeerController("/dev/ttyUSB0")
    controller._connecting = True
    assert controller.connect() is False
    assert controller.last_connect_error == "Connection already in progress"


def test_seer_controller_handles_xvisio_native_load_failure(monkeypatch) -> None:
    def fake_import_module(name: str):
        assert name == "xvisio"
        raise OSError("libxvsdk.so: cannot open shared object file")

    monkeypatch.setattr(seer_teleop.importlib, "import_module", fake_import_module)

    controller = seer_teleop.SeerController("/dev/ttyUSB0")

    assert controller.connect() is False
    assert controller.connected is False


def test_seer_button_mapping_uses_a_for_stream_trigger_fraction_and_b_for_reset(
    monkeypatch,
) -> None:
    states = iter(
        [
            (0, 0, 0),
            (0, 0, seer_teleop.BUTTON_A),
            (seer_teleop.TRIGGER_THRESHOLD + 1, 0, seer_teleop.BUTTON_A),
            (
                seer_teleop.TRIGGER_THRESHOLD + 1,
                0,
                seer_teleop.BUTTON_A | seer_teleop.BUTTON_B,
            ),
            (0, 0, seer_teleop.BUTTON_A),
            (seer_teleop.TRIGGER_THRESHOLD + 1, 0, seer_teleop.BUTTON_A),
            (0, 0, 0),
        ]
    )

    class FakeDevice:
        def controller(self):
            trigger, side, key = next(states)
            return None, SimpleNamespace(key_trigger=trigger, key_side=side, key=key)

    monkeypatch.setattr(seer_teleop.time, "time", lambda: 10.0)
    controller = seer_teleop.SeerController("/dev/ttyUSB0")
    controller.device = FakeDevice()
    events: list[str] = []
    controller.on_stream_start = lambda: events.append("stream_start")
    controller.on_stream_stop = lambda: events.append("stream_stop")
    controller.on_reset = lambda: events.append("reset")

    controller.process_buttons()
    assert controller.streaming is False
    assert controller.gripper_closed is False

    controller.process_buttons()
    assert controller.streaming is True
    assert controller.gripper_closed is False

    controller.process_buttons()
    assert controller.streaming is True
    assert controller.gripper_closed is True
    assert controller.trigger_fraction == pytest.approx(
        (seer_teleop.TRIGGER_THRESHOLD + 1) / seer_teleop.TRIGGER_MAX_VALUE
    )

    controller.process_buttons()
    assert events == ["stream_start", "reset"]

    controller.process_buttons()
    assert controller.streaming is True
    assert controller.gripper_closed is False
    assert controller.trigger_fraction == pytest.approx(0.0)

    controller.process_buttons()
    assert controller.streaming is True
    assert controller.gripper_closed is True
    assert controller.trigger_fraction == pytest.approx(
        (seer_teleop.TRIGGER_THRESHOLD + 1) / seer_teleop.TRIGGER_MAX_VALUE
    )

    controller.process_buttons()
    assert controller.streaming is False
    assert controller.trigger_fraction == pytest.approx(0.0)
    assert events == ["stream_start", "reset", "stream_stop"]


def test_seer_controller_disabled_gates_stream_reset_and_trigger(monkeypatch) -> None:
    class FakeDevice:
        def controller(self):
            return None, SimpleNamespace(
                key_trigger=seer_teleop.TRIGGER_MAX_VALUE,
                key_side=0,
                key=seer_teleop.BUTTON_A | seer_teleop.BUTTON_B,
            )

    monkeypatch.setattr(seer_teleop.time, "time", lambda: 10.0)
    controller = seer_teleop.SeerController("/dev/ttyUSB0")
    controller.device = FakeDevice()
    controller.set_enabled(False)
    events: list[str] = []
    controller.on_stream_start = lambda: events.append("stream_start")
    controller.on_stream_stop = lambda: events.append("stream_stop")
    controller.on_reset = lambda: events.append("reset")

    controller.process_buttons()

    assert controller.streaming is False
    assert controller.trigger_fraction == pytest.approx(0.0)
    assert events == []


def test_seer_trigger_fraction_reaches_closed_at_controller_full_press(monkeypatch) -> None:
    class FakeDevice:
        def controller(self):
            return None, SimpleNamespace(
                key_trigger=int(seer_teleop.TRIGGER_MAX_VALUE),
                key_side=0,
                key=0,
            )

    controller = seer_teleop.SeerController("/dev/ttyUSB0")
    controller.device = FakeDevice()

    controller.process_buttons()

    assert controller.trigger_fraction == pytest.approx(1.0)
    assert controller.gripper_closed is True


def test_seer_trigger_fraction_clamps_overrange_values(monkeypatch) -> None:
    class FakeDevice:
        def controller(self):
            return None, SimpleNamespace(
                key_trigger=int(seer_teleop.TRIGGER_MAX_VALUE) + 100,
                key_side=0,
                key=0,
            )

    controller = seer_teleop.SeerController("/dev/ttyUSB0")
    controller.device = FakeDevice()

    controller.process_buttons()

    assert controller.trigger_fraction == pytest.approx(1.0)


def test_gripper_trigger_fraction_maps_continuously_to_command() -> None:
    assert seer_teleop.gripper_command_from_trigger_fraction(
        0.0, open_command=-1.57, closed_command=0.0
    ) == pytest.approx(-1.57)
    assert seer_teleop.gripper_command_from_trigger_fraction(
        1.0, open_command=-1.57, closed_command=0.0
    ) == pytest.approx(0.0)
    assert seer_teleop.gripper_command_from_trigger_fraction(
        0.5, open_command=-1.57, closed_command=0.0
    ) == pytest.approx(-0.785)


def test_controller_delta_pose_helpers_round_trip_transform_control_pose() -> None:
    base_control = SimpleNamespace(
        position=(1.0, 2.0, 3.0),
        wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    base_pose = seer_teleop.pose_from_transform_control(base_control)

    pose = seer_teleop.apply_controller_delta(
        base_pose,
        np.array([0.1, -0.2, 0.3]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        scale=2.0,
    )

    target_control = SimpleNamespace(position=(0.0, 0.0, 0.0), wxyz=(0.0, 0.0, 0.0, 1.0))
    seer_teleop.set_transform_control_pose(target_control, pose)

    np.testing.assert_allclose(target_control.position, np.array([1.2, 1.6, 3.6]))
    np.testing.assert_allclose(target_control.wxyz, np.array([1.0, 0.0, 0.0, 0.0]))
