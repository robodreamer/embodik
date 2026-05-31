import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.example_helpers.seer_teleop import SeerController  # noqa: E402


class _FakeCtrl:
    def __init__(self, pos, quat_wxyz, key_trigger=0, key_side=0, key=0):
        self.position = np.asarray(pos, dtype=float)
        self.quat_wxyz = np.asarray(quat_wxyz, dtype=float)
        self.key_trigger = key_trigger
        self.key_side = key_side
        self.key = key


class _FakeDevice:
    """Mimics an xvisio device exposing (left, right) controllers."""

    def __init__(self, left, right):
        self._left, self._right = left, right
        self.reset_count = 0

    def controller_relative(self):
        return (self._left, self._right)

    def controller(self):
        return (self._left, self._right)

    def reset_controller_reference(self):
        self.reset_count += 1

    def close(self):
        pass


def test_relative_pose_side_selects_requested_controller():
    left = _FakeCtrl([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    right = _FakeCtrl([0.0, 2.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    ctrl = SeerController(port=None)
    ctrl.device = _FakeDevice(left, right)

    lpos, _ = ctrl.relative_pose_side("left")
    rpos, _ = ctrl.relative_pose_side("right")
    assert np.allclose(lpos, [1.0, 0.0, 0.0])
    assert np.allclose(rpos, [0.0, 2.0, 0.0])


def test_raw_buttons_side_selects_requested_controller():
    left = _FakeCtrl([0, 0, 0], [1, 0, 0, 0], key_trigger=5, key=16)
    right = _FakeCtrl([0, 0, 0], [1, 0, 0, 0], key_trigger=40, key=32)
    ctrl = SeerController(port=None)
    ctrl.device = _FakeDevice(left, right)
    assert ctrl.raw_buttons_side("left") == (5, 0, 16)
    assert ctrl.raw_buttons_side("right") == (40, 0, 32)


from examples.example_helpers.bimanual_seer_teleop import (  # noqa: E402
    ArmTeleopState,
    BimanualSeerTeleop,
)
from embodik import Rt  # noqa: E402


def _identity_target():
    return Rt(R=np.eye(3), t=np.zeros(3))


def test_streaming_side_moves_its_target_other_holds():
    # Right controller reports a +y relative translation; left reports nothing.
    right = _FakeCtrl([0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], key=16)  # A held -> streaming
    left = _FakeCtrl([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], key=0)
    teleop = BimanualSeerTeleop(scale=2.0)
    teleop.controller.device = _FakeDevice(left, right)
    teleop.arms["right"].connected = True
    teleop.set_arm_enabled("left", True)
    teleop.set_arm_enabled("right", True)
    teleop.begin_streaming("right", _identity_target())

    out = teleop.step(left_target=_identity_target(), right_target=_identity_target())
    assert out["right"].new_pose is not None
    assert np.allclose(out["right"].new_pose.translation, [0.0, 2.0, 0.0])
    assert out["left"].new_pose is None


def test_both_sides_stream_moves_both_targets():
    teleop = BimanualSeerTeleop(scale=1.0)
    teleop.arms["left"].connected = True
    teleop.arms["right"].connected = True
    teleop.set_arm_enabled("left", True)
    teleop.set_arm_enabled("right", True)

    right = _FakeCtrl([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], key=16)
    left = _FakeCtrl([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], key=0)
    teleop.controller.device = _FakeDevice(left, right)
    teleop.step(left_target=_identity_target(), right_target=_identity_target(), now=0.0)
    assert teleop.arms["right"].streaming is True

    right = _FakeCtrl([0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], key=16)
    left = _FakeCtrl([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], key=32)
    teleop.controller.device = _FakeDevice(left, right)
    teleop.step(left_target=_identity_target(), right_target=_identity_target(), now=0.1)
    assert teleop.arms["left"].streaming is True

    right = _FakeCtrl([0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], key=16)
    left = _FakeCtrl([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], key=32)
    teleop.controller.device = _FakeDevice(left, right)
    out = teleop.step(left_target=_identity_target(), right_target=_identity_target(), now=0.2)
    assert np.allclose(out["left"].new_pose.translation, [1.0, 0.0, 0.0])
    assert np.allclose(out["right"].new_pose.translation, [0.0, 1.0, 0.0])


def test_left_streams_on_button_b_right_on_button_a():
    right = _FakeCtrl([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], key=0)
    left = _FakeCtrl([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], key=32)
    teleop = BimanualSeerTeleop(scale=1.0)
    teleop.controller.device = _FakeDevice(left, right)
    teleop.arms["left"].connected = True
    teleop.arms["right"].connected = True
    teleop.set_arm_enabled("left", True)
    teleop.set_arm_enabled("right", True)

    teleop.step(left_target=_identity_target(), right_target=_identity_target(), now=0.0)
    assert teleop.arms["left"].streaming is True
    assert teleop.arms["right"].streaming is False

    right = _FakeCtrl([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], key=16)
    left = _FakeCtrl([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], key=0)
    teleop.controller.device = _FakeDevice(left, right)
    teleop.step(left_target=_identity_target(), right_target=_identity_target(), now=0.1)
    assert teleop.arms["right"].streaming is True


def test_second_stream_does_not_global_reset_while_first_is_streaming():
    teleop = BimanualSeerTeleop(scale=2.0)
    teleop.arms["left"].connected = True
    teleop.arms["right"].connected = True
    teleop.set_arm_enabled("left", True)
    teleop.set_arm_enabled("right", True)

    right = _FakeCtrl([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], key=16)
    left = _FakeCtrl([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], key=0)
    device = _FakeDevice(left, right)
    teleop.controller.device = device
    teleop.step(left_target=_identity_target(), right_target=_identity_target(), now=0.0)
    assert device.reset_count == 1
    assert teleop.arms["right"].streaming is True

    right = _FakeCtrl([0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], key=16)
    left = _FakeCtrl([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], key=32)
    device = _FakeDevice(left, right)
    device.reset_count = 1
    teleop.controller.device = device
    teleop.step(left_target=_identity_target(), right_target=_identity_target(), now=0.1)
    assert device.reset_count == 1
    assert teleop.arms["left"].streaming is True
    assert teleop.arms["right"].streaming is True

    out = teleop.step(left_target=_identity_target(), right_target=_identity_target(), now=0.2)
    assert np.allclose(out["right"].new_pose.translation, [0.0, 2.0, 0.0])
    assert np.allclose(out["left"].new_pose.translation, [0.0, 0.0, 0.0])


def test_left_button_a_release_invokes_robot_reset_callback():
    teleop = BimanualSeerTeleop(scale=1.0)
    teleop.controller.device = _FakeDevice(
        _FakeCtrl([0, 0, 0], [1, 0, 0, 0], key=0),
        _FakeCtrl([0, 0, 0], [1, 0, 0, 0], key=0),
    )
    teleop.arms["left"].connected = True
    teleop.set_arm_enabled("left", True)
    calls = {"count": 0}
    teleop.on_reset_robot = lambda: calls.__setitem__("count", calls["count"] + 1)

    left_pressed = _FakeCtrl([0, 0, 0], [1, 0, 0, 0], key=16)
    left_released = _FakeCtrl([0, 0, 0], [1, 0, 0, 0], key=0)
    teleop.controller.device = _FakeDevice(left_pressed, _FakeCtrl([0, 0, 0], [1, 0, 0, 0], key=0))
    teleop.step(left_target=_identity_target(), right_target=_identity_target(), now=0.0)
    teleop.controller.device = _FakeDevice(left_released, _FakeCtrl([0, 0, 0], [1, 0, 0, 0], key=0))
    teleop.step(left_target=_identity_target(), right_target=_identity_target(), now=1.0)
    assert calls["count"] == 1


def test_disconnect_side_closes_device_when_both_disconnected():
    teleop = BimanualSeerTeleop(scale=1.0)
    teleop.controller.device = _FakeDevice(
        _FakeCtrl([0, 0, 0], [1, 0, 0, 0]),
        _FakeCtrl([0, 0, 0], [1, 0, 0, 0]),
    )
    teleop.arms["left"].connected = True
    teleop.arms["right"].connected = True
    teleop.disconnect_side("right")
    assert teleop.connected
    assert teleop.is_side_connected("left")
    assert not teleop.is_side_connected("right")
    teleop.disconnect_side("left")
    assert not teleop.connected
    assert teleop.controller.device is None
