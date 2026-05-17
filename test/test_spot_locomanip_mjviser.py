"""Tests for the Spot locomanipulation mjviser example helpers."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

try:
    import mujoco
    import onnxruntime
except ImportError:  # Optional Spot environment dependencies.
    onnxruntime = None  # type: ignore[assignment]
    mujoco = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    onnxruntime is None or mujoco is None,
    reason="Spot locomanipulation tests require the optional spot environment",
)

from example_helpers.spot_locomanip_policy import (
    DEFAULT_ARM_COMMAND,
    DEFAULT_BODY_ROLL_PITCH_HEIGHT,
    HEIGHT_RANGE,
    HEIGHT_STEP,
    INITIAL_ARM_COMMAND,
    MJCF_ARM_JOINT_NAMES,
    MJCF_GRIPPER_JOINT_NAME,
    POSE_COMMAND_RANGE,
    POSE_COMMAND_STEP,
    POSE_KI_LINEAR,
    POSE_KI_YAW,
    POSE_KP_LINEAR,
    POSE_KP_YAW,
    ROLL_PITCH_RANGE,
    ROLL_PITCH_STEP,
    SPOT_LEG_JOINT_NAMES,
    DEFAULT_STAND_BASE_HEIGHT,
    DEFAULT_STAND_LEG_JOINTS,
    VELOCITY_COMMAND_RANGE,
    VELOCITY_COMMAND_STEP,
    YAW_COMMAND_RANGE,
    YAW_COMMAND_STEP,
    LocomanipPolicy,
    build_policy_observation,
    leg_targets_from_policy_output,
    normalize_mjcf_joint_name,
    policy_action_to_mujoco_ctrl,
    policy_checkpoint_path,
)
from example_helpers.policy_runtime import RateLimitedOnnxPolicy
from example_helpers.spot_mjviser_adapter import (
    DEFAULT_ACTUATOR_GAINS,
    DEFAULT_LEG_GAIN_SCALE,
    LEG_GAIN_SCALE_MAX,
    LEG_GAIN_SCALE_MIN,
    MANIPULATION_OBJECT_BODY_POSES,
    MANIPULATION_OBJECT_GEOM_NAMES,
    MANIPULATION_OBJECT_HIDDEN_Z,
    MANIPULATION_OBJECT_VISUAL_SPECS,
    SpotMujocoAdapter,
    apply_default_actuator_gains,
    load_spot_mujoco_model,
    resolve_spot_scene_arm_xml,
)
from example_helpers.seer_teleop import DEFAULT_TELEOP_SCALE_FACTOR


def _load_example_module():
    path = _EXAMPLES_DIR / "15_spot_locomanip_mjviser.py"
    spec = importlib.util.spec_from_file_location("spot_locomanip_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _zero_policy_observation(last_output_len: int = 12) -> dict[str, np.ndarray]:
    return {
        "base_lin_vel": np.zeros(3),
        "base_ang_vel": np.zeros(3),
        "projected_gravity": np.array([0.0, 0.0, -1.0]),
        "joint_pos": np.zeros(12),
        "joint_vel": np.zeros(12),
        "arm_state": np.zeros(12),
        "gripper_state": np.zeros(2),
        "base_pose": np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]),
    }


@pytest.mark.parametrize("policy", list(LocomanipPolicy))
def test_locomanip_checkpoints_load_with_expected_io(policy: LocomanipPolicy) -> None:
    checkpoint = policy_checkpoint_path(policy)
    assert checkpoint.is_file()
    session = onnxruntime.InferenceSession(str(checkpoint), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, 72]
    assert session.get_outputs()[0].shape == [1, 12]


def test_policy_observation_and_action_mapping_shapes() -> None:
    observation = build_policy_observation(
        _zero_policy_observation(),
        np.zeros(12),
        arm_joint_command=DEFAULT_ARM_COMMAND,
    )
    assert observation.shape == (72,)

    leg_targets = leg_targets_from_policy_output(np.zeros(12))
    assert leg_targets.shape == (12,)

    ctrl = policy_action_to_mujoco_ctrl(np.zeros(12), DEFAULT_ARM_COMMAND)
    assert len(ctrl) == 19
    assert ctrl["arm_sh1"] == pytest.approx(DEFAULT_ARM_COMMAND[1])


def test_rate_limited_onnx_policy_reuses_output_between_50hz_ticks() -> None:
    class CountingSession:
        def __init__(self):
            self.run_count = 0

        def run(self, *_args, **_kwargs):
            self.run_count += 1
            return [np.full((1, 3), self.run_count, dtype=np.float32)]

    session = CountingSession()
    runtime = RateLimitedOnnxPolicy(rate_hz=50.0, output_len=3)
    runtime.configure(session, input_name="input", output_name="output", output_len=3)
    policy_input = np.ones(4, dtype=np.float32)

    np.testing.assert_allclose(runtime.maybe_run(0.005, policy_input), np.ones(3))
    np.testing.assert_allclose(runtime.maybe_run(0.005, policy_input), np.ones(3))
    np.testing.assert_allclose(runtime.maybe_run(0.009, policy_input), np.ones(3))
    assert session.run_count == 1
    assert runtime.step_count == 1

    np.testing.assert_allclose(runtime.maybe_run(0.001, policy_input), np.full(3, 2.0))
    assert session.run_count == 2
    assert runtime.step_count == 2
    assert runtime.last_inference_time_ms >= 0.0


def test_gripper_trigger_fraction_maps_continuously_to_command() -> None:
    module = _load_example_module()

    assert module.gripper_command_from_trigger_fraction(0.0) == pytest.approx(INITIAL_ARM_COMMAND[6])
    assert module.gripper_command_from_trigger_fraction(1.0) == pytest.approx(0.0)
    assert module.gripper_command_from_trigger_fraction(0.5) == pytest.approx(0.5 * INITIAL_ARM_COMMAND[6])
    assert module.gripper_command_from_trigger_fraction(-1.0) == pytest.approx(INITIAL_ARM_COMMAND[6])
    assert module.gripper_command_from_trigger_fraction(2.0) == pytest.approx(0.0)


def test_policy_arm_names_normalize_to_menagerie_names() -> None:
    assert normalize_mjcf_joint_name("arm0_sh0") == "arm_sh0"
    assert normalize_mjcf_joint_name("arm0_f1x") == "arm_f1x"
    assert normalize_mjcf_joint_name("fl_hx") == "fl_hx"


def test_locomanip_command_ranges_and_gains_are_defined() -> None:
    assert POSE_COMMAND_RANGE == (-2.0, 2.0)
    assert POSE_COMMAND_STEP == pytest.approx(0.1)
    assert VELOCITY_COMMAND_RANGE == (-1.0, 1.0)
    assert VELOCITY_COMMAND_STEP == pytest.approx(0.1)
    assert YAW_COMMAND_RANGE == (-3.14, 3.14)
    assert YAW_COMMAND_STEP == pytest.approx(0.1)
    assert ROLL_PITCH_RANGE == (-0.5, 0.5)
    assert ROLL_PITCH_STEP == pytest.approx(0.02)
    assert HEIGHT_RANGE == (0.3, 0.65)
    assert HEIGHT_STEP == pytest.approx(0.02)
    assert POSE_KP_LINEAR == pytest.approx(2.5)
    assert POSE_KI_LINEAR == pytest.approx(0.5)
    assert POSE_KP_YAW == pytest.approx(3.0)
    assert POSE_KI_YAW == pytest.approx(0.4)
    assert DEFAULT_ACTUATOR_GAINS["fl_hx"] == pytest.approx((75.0, 1.875))
    assert DEFAULT_LEG_GAIN_SCALE == pytest.approx(1.25)
    assert LEG_GAIN_SCALE_MIN == pytest.approx(1.0)
    assert LEG_GAIN_SCALE_MAX == pytest.approx(2.0)
    assert DEFAULT_ACTUATOR_GAINS["arm_sh0"] == pytest.approx((320.0, 8.0))
    assert DEFAULT_ACTUATOR_GAINS["arm_sh1"] == pytest.approx((150.0, 15.3))
    assert DEFAULT_ACTUATOR_GAINS["arm_el0"] == pytest.approx((120.0, 5.2))
    assert DEFAULT_ACTUATOR_GAINS["arm_f1x"] == pytest.approx((16.0, 1.0))


def test_mujoco_actuator_gains_default_to_example_values() -> None:
    model = load_spot_mujoco_model()

    for name, (kp, kd) in DEFAULT_ACTUATOR_GAINS.items():
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert actuator_id >= 0
        assert model.actuator_gainprm[actuator_id, 0] == pytest.approx(kp)
        assert model.actuator_biasprm[actuator_id, 1] == pytest.approx(-kp)
        assert model.actuator_biasprm[actuator_id, 2] == pytest.approx(-kd)


def test_apply_default_actuator_gains_restores_example_values() -> None:
    model = load_spot_mujoco_model(apply_default_gains=False)

    apply_default_actuator_gains(model)

    for name, (kp, kd) in DEFAULT_ACTUATOR_GAINS.items():
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert model.actuator_gainprm[actuator_id, 0] == pytest.approx(kp)
        assert model.actuator_biasprm[actuator_id, 1] == pytest.approx(-kp)
        assert model.actuator_biasprm[actuator_id, 2] == pytest.approx(-kd)


def test_leg_actuator_gains_are_runtime_scalable_between_reference_and_double() -> None:
    model = load_spot_mujoco_model(apply_default_gains=False)

    apply_default_actuator_gains(model, leg_gain_scale=LEG_GAIN_SCALE_MIN)

    for name in SPOT_LEG_JOINT_NAMES:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert model.actuator_gainprm[actuator_id, 0] == pytest.approx(60.0)
        assert model.actuator_biasprm[actuator_id, 1] == pytest.approx(-60.0)
        assert model.actuator_biasprm[actuator_id, 2] == pytest.approx(-1.5)
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm_sh0")
    assert model.actuator_gainprm[actuator_id, 0] == pytest.approx(320.0)
    assert model.actuator_biasprm[actuator_id, 2] == pytest.approx(-8.0)

    apply_default_actuator_gains(model, leg_gain_scale=LEG_GAIN_SCALE_MAX)

    for name in SPOT_LEG_JOINT_NAMES:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert model.actuator_gainprm[actuator_id, 0] == pytest.approx(120.0)
        assert model.actuator_biasprm[actuator_id, 1] == pytest.approx(-120.0)
        assert model.actuator_biasprm[actuator_id, 2] == pytest.approx(-3.0)


def test_raw_menagerie_actuator_gains_can_be_enabled_for_comparison() -> None:
    model = load_spot_mujoco_model(apply_default_gains=False)

    for name in DEFAULT_ACTUATOR_GAINS:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert actuator_id >= 0
        assert model.actuator_gainprm[actuator_id, 0] == pytest.approx(500.0)
        assert model.actuator_biasprm[actuator_id, 1] == pytest.approx(-500.0)
        assert model.actuator_biasprm[actuator_id, 2] == pytest.approx(-40.0)


def test_raw_menagerie_gains_keep_headless_policy_height_bounded() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model(apply_default_gains=False)
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(model, policy=module.LocomanipPolicy.LOCOMANIP)
    controller.reset(model, data)

    for _ in range(250):
        controller.step(model, data, control_mode="pose", ik_enabled=False)
        mujoco.mj_step(model, data)

    assert np.isfinite(data.qpos).all()
    assert data.qpos[2] > 0.5


def test_raw_menagerie_gains_keep_current_gripper_ik_height_bounded() -> None:
    module = _load_example_module()
    spot_urdf = module._resolve_optional_spot_urdf(None)
    if spot_urdf is None:
        pytest.skip("Spot whole-body URDF is not available")
    model = load_spot_mujoco_model(apply_default_gains=False)
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(
        model,
        policy=module.LocomanipPolicy.LOCOMANIP,
        spot_urdf=spot_urdf,
    )
    controller.reset(model, data)
    pos, wxyz = controller.adapter.body_pose_wxyz(data, "arm_link_fngr")
    controller.commands.arm[:] = controller.adapter.read_measured_arm_command(data)
    controller.ik.reset_reference()
    target_pose = module.target_pose_from_wxyz(pos, wxyz)

    for _ in range(250):
        controller.step(
            model,
            data,
            control_mode="pose",
            ik_enabled=True,
            ik_target_pose=target_pose,
            gravity_compensation_enabled=False,
        )
        mujoco.mj_step(model, data)

    assert np.isfinite(data.qpos).all()
    assert data.qpos[2] > 0.5


def test_pose_controller_applies_clipping_behavior() -> None:
    module = _load_example_module()
    current_pose = np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
    desired_pose = np.array([2.0, -2.0, 3.14])
    integral = np.zeros(3)

    velocity = module.compute_velocity_from_pose(current_pose, desired_pose, integral, dt=0.02)

    np.testing.assert_allclose(velocity, np.array([1.0, -1.0, 1.0]))
    np.testing.assert_allclose(integral, np.array([0.04, -0.04, 0.0628]), atol=1e-6)


def test_controller_reset_clears_pose_integral() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(model, policy=module.LocomanipPolicy.LOCOMANIP)
    controller.commands.pose_error_integral[:] = [1.0, -1.0, 0.5]

    controller.reset(model, data)

    np.testing.assert_allclose(controller.commands.pose_error_integral, np.zeros(3))


def test_controller_reset_matches_locomanip_initial_configuration() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(model, policy=module.LocomanipPolicy.LOCOMANIP)

    controller.reset(model, data)

    assert data.qpos[2] == pytest.approx(DEFAULT_STAND_BASE_HEIGHT)
    leg_positions = controller.adapter.read_joint_positions(data, SPOT_LEG_JOINT_NAMES)
    np.testing.assert_allclose(leg_positions, DEFAULT_STAND_LEG_JOINTS)
    arm_positions = controller.adapter.read_joint_positions(
        data, (*MJCF_ARM_JOINT_NAMES, MJCF_GRIPPER_JOINT_NAME)
    )
    np.testing.assert_allclose(arm_positions, DEFAULT_ARM_COMMAND)
    ctrl = controller.adapter.current_ctrl_targets(data)
    for name, value in zip((*MJCF_ARM_JOINT_NAMES, MJCF_GRIPPER_JOINT_NAME), DEFAULT_ARM_COMMAND):
        assert ctrl[name] == pytest.approx(value)


def test_controller_reset_is_independent_of_adversarial_runtime_state() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model()
    baseline_data = mujoco.MjData(model)
    adversarial_data = mujoco.MjData(model)
    baseline = module.SpotLocomanipController(model, policy=module.LocomanipPolicy.LOCOMANIP)
    controller = module.SpotLocomanipController(model, policy=module.LocomanipPolicy.LOCOMANIP)

    baseline.reset(model, baseline_data)
    controller.reset(model, adversarial_data)
    module.schedule_interactive_unstow(controller)
    controller.start_arm_unstow_interpolation()

    # Poison every mutable runtime surface that has caused reset coupling in the
    # interactive app: fallen pose, non-unit quaternion, high velocities, stale
    # controls, applied forces, warm-start acceleration, IK/policy state, and GUI
    # command state.  A reset must not depend on any of these values.
    adversarial_data.time = 42.0
    adversarial_data.qpos[:] = np.linspace(-1.5, 1.5, model.nq)
    adversarial_data.qpos[:7] = [1.2, -0.7, 0.08, 0.2, 0.3, -0.4, 0.5]
    adversarial_data.qvel[:] = np.linspace(8.0, -8.0, model.nv)
    adversarial_data.ctrl[:] = np.linspace(-3.0, 3.0, model.nu)
    adversarial_data.qfrc_applied[:] = np.linspace(4.0, -4.0, model.nv)
    adversarial_data.xfrc_applied[:] = np.linspace(2.0, -2.0, adversarial_data.xfrc_applied.size).reshape(
        adversarial_data.xfrc_applied.shape
    )
    adversarial_data.qacc_warmstart[:] = np.linspace(-5.0, 5.0, model.nv)
    if getattr(adversarial_data, "act", None) is not None and adversarial_data.act.size:
        adversarial_data.act[:] = np.linspace(-1.0, 1.0, adversarial_data.act.size)
    controller.last_output[:] = 7.0
    controller.step_count = 1234
    controller.last_ik_status = "stale"
    controller.commands.velocity[:] = [1.0, -1.0, 0.5]
    controller.commands.desired_pose[:] = [2.0, -2.0, 1.0]
    controller.commands.body[:] = [0.4, -0.4, 0.3]
    controller.commands.arm[:] = INITIAL_ARM_COMMAND
    controller.commands.pose_error_integral[:] = [0.3, -0.2, 0.1]

    controller.reset(model, adversarial_data)

    np.testing.assert_allclose(adversarial_data.qpos, baseline_data.qpos, atol=1e-12)
    np.testing.assert_allclose(adversarial_data.qvel, baseline_data.qvel, atol=1e-12)
    np.testing.assert_allclose(adversarial_data.ctrl, baseline_data.ctrl, atol=1e-12)
    np.testing.assert_allclose(adversarial_data.qfrc_applied, baseline_data.qfrc_applied, atol=1e-12)
    np.testing.assert_allclose(adversarial_data.xfrc_applied, baseline_data.xfrc_applied, atol=1e-12)
    np.testing.assert_allclose(
        adversarial_data.qacc_warmstart,
        baseline_data.qacc_warmstart,
        atol=1e-12,
    )
    if getattr(adversarial_data, "act", None) is not None and adversarial_data.act.size:
        np.testing.assert_allclose(adversarial_data.act, baseline_data.act, atol=1e-12)
    np.testing.assert_allclose(controller.last_output, np.zeros_like(controller.last_output))
    np.testing.assert_allclose(controller.commands.velocity, np.zeros(3))
    np.testing.assert_allclose(controller.commands.desired_pose, np.zeros(3))
    np.testing.assert_allclose(controller.commands.pose_error_integral, np.zeros(3))
    np.testing.assert_allclose(controller.commands.body, DEFAULT_BODY_ROLL_PITCH_HEIGHT)
    np.testing.assert_allclose(controller.commands.arm, DEFAULT_ARM_COMMAND)
    assert controller.step_count == 0
    assert not controller.arm_unstow_active
    assert controller._auto_unstow_delay_remaining is None


def test_controller_reset_from_fallen_motion_is_repeatable_and_stable() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(model, policy=module.LocomanipPolicy.LOCOMANIP)
    startup_steps = int(
        (module.AUTO_UNSTOW_DELAY_SECONDS + module.ARM_UNSTOW_INTERPOLATION_SECONDS + 1.0)
        / float(model.opt.timestep)
    )

    fallen_states = (
        ([0.4, -0.2, 0.05, 0.5, 0.5, -0.5, 0.5], [8.0, -4.0, 2.0]),
        ([-0.9, 0.6, 0.10, 0.1, -0.7, 0.2, 0.6], [-5.0, 3.0, -2.0]),
        ([1.7, 1.2, 0.03, -0.4, 0.3, 0.8, -0.2], [2.0, 7.0, 4.0]),
    )

    for base_qpos, command in fallen_states:
        controller.reset(model, data)
        controller.commands.velocity[:] = command
        controller.commands.body[:] = [0.4, -0.4, 0.3]
        controller.commands.arm[:] = INITIAL_ARM_COMMAND
        for _ in range(80):
            controller.step(model, data, control_mode="velocity", ik_enabled=False)
            mujoco.mj_step(model, data)

        data.time = 99.0
        data.qpos[:7] = base_qpos
        data.qvel[:] = np.linspace(10.0, -10.0, model.nv)
        data.ctrl[:] = np.linspace(-6.0, 6.0, model.nu)
        data.qfrc_applied[:] = np.linspace(3.0, -3.0, model.nv)
        data.xfrc_applied[:] = np.linspace(1.5, -1.5, data.xfrc_applied.size).reshape(
            data.xfrc_applied.shape
        )
        data.qacc_warmstart[:] = np.linspace(-2.0, 2.0, model.nv)

        controller.reset(model, data)
        module.schedule_interactive_unstow(controller)
        controller.adapter.apply_ctrl_targets(
            data,
            policy_action_to_mujoco_ctrl(controller.last_output, controller.commands.arm),
        )

        assert data.time == pytest.approx(0.0)
        assert data.qpos[2] == pytest.approx(DEFAULT_STAND_BASE_HEIGHT)
        np.testing.assert_allclose(data.qvel, np.zeros(model.nv), atol=1e-12)
        min_height = float(data.qpos[2])
        for _ in range(startup_steps):
            controller.step(model, data, control_mode="pose", ik_enabled=False)
            mujoco.mj_step(model, data)
            min_height = min(min_height, float(data.qpos[2]))
            assert np.isfinite(data.qpos).all()
            assert np.isfinite(data.qvel).all()
            assert np.isfinite(data.ctrl).all()
            assert np.isfinite(data.qfrc_applied).all()
        assert min_height > 0.35
        assert not controller.arm_unstow_active
        np.testing.assert_allclose(controller.commands.arm, INITIAL_ARM_COMMAND, atol=1e-6)


def test_button_reset_settle_barrier_holds_home_before_unstow() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(model, policy=module.LocomanipPolicy.LOCOMANIP)
    controller.reset(model, data, settle_seconds=module.RESET_SETTLE_SECONDS)
    module.schedule_interactive_unstow(controller)

    settle_steps = max(1, int(module.RESET_SETTLE_SECONDS / float(model.opt.timestep)) - 1)
    for _ in range(settle_steps):
        controller.commands.arm[:] = INITIAL_ARM_COMMAND
        controller.commands.velocity[:] = [1.0, -1.0, 0.5]
        controller.step(model, data, control_mode="velocity", ik_enabled=True, ik_target_pose=object())
        mujoco.mj_step(model, data)
        np.testing.assert_allclose(controller.commands.arm, DEFAULT_ARM_COMMAND)
        np.testing.assert_allclose(controller.commands.velocity, np.zeros(3))
        assert np.isfinite(controller.last_output).all()
        assert controller.last_ik_status == "reset settling"

    # The auto-unstow countdown starts only after the settle barrier has elapsed.
    assert controller._auto_unstow_delay_remaining == pytest.approx(module.AUTO_UNSTOW_DELAY_SECONDS)


def test_controller_interpolates_arm_command_from_current_state_when_requested() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(model, policy=module.LocomanipPolicy.LOCOMANIP)
    controller.reset(model, data)

    controller.step(model, data, control_mode="pose", ik_enabled=False)

    np.testing.assert_allclose(controller.commands.arm, DEFAULT_ARM_COMMAND)
    assert not controller.arm_unstow_active

    start = DEFAULT_ARM_COMMAND.copy()
    start[0] = -0.4
    controller.commands.arm[:] = start
    controller.start_arm_unstow_interpolation()
    controller.step(model, data, control_mode="pose", ik_enabled=False)

    alpha = float(model.opt.timestep) / module.ARM_UNSTOW_INTERPOLATION_SECONDS
    expected = (1.0 - alpha) * start + alpha * INITIAL_ARM_COMMAND
    np.testing.assert_allclose(controller.commands.arm, expected)
    assert controller.arm_unstow_active

    controller._arm_unstow_elapsed = (
        module.ARM_UNSTOW_INTERPOLATION_SECONDS - float(model.opt.timestep)
    )
    controller.step(model, data, control_mode="pose", ik_enabled=False)

    np.testing.assert_allclose(controller.commands.arm, INITIAL_ARM_COMMAND)
    assert not controller.arm_unstow_active

    start = INITIAL_ARM_COMMAND.copy()
    start[1] = -0.7
    controller.commands.arm[:] = start
    controller.start_arm_stow_interpolation()
    controller.step(model, data, control_mode="pose", ik_enabled=False)

    alpha = float(model.opt.timestep) / module.ARM_UNSTOW_INTERPOLATION_SECONDS
    expected = (1.0 - alpha) * start + alpha * DEFAULT_ARM_COMMAND
    np.testing.assert_allclose(controller.commands.arm, expected)
    assert controller.arm_unstow_active


def test_mjviser_scene_position_conversion_accounts_for_camera_tracking_offset() -> None:
    module = _load_example_module()
    world_position = np.array([0.475, 0.0, 0.805])
    scene_offset = np.array([0.0, 0.0, -0.525])

    scene_position = module.mujoco_world_to_mjviser_scene_position(
        world_position,
        scene_offset,
    )

    np.testing.assert_allclose(scene_position, np.array([0.475, 0.0, 0.280]))
    np.testing.assert_allclose(
        module.mjviser_scene_to_mujoco_world_position(scene_position, scene_offset),
        world_position,
    )


def test_world_fixed_ik_target_keeps_authoritative_mujoco_world_pose() -> None:
    module = _load_example_module()
    scene_offset = np.array([0.0, 0.0, -0.5])
    target = module.WorldFixedIkTarget(
        np.array([0.4, -0.1, 0.8]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        lambda: scene_offset,
    )
    handle = SimpleNamespace(position=(0.0, 0.0, 0.0), wxyz=(1.0, 0.0, 0.0, 0.0))

    target.apply_to_handle(handle)
    np.testing.assert_allclose(handle.position, np.array([0.4, -0.1, 0.3]))

    scene_offset[:] = [0.2, 0.0, -0.6]
    target.apply_to_handle(handle)
    np.testing.assert_allclose(target.world_position, np.array([0.4, -0.1, 0.8]))
    np.testing.assert_allclose(handle.position, np.array([0.6, -0.1, 0.2]))

    handle.position = (0.9, 0.1, 0.25)
    handle.wxyz = (0.0, 1.0, 0.0, 0.0)
    target.update_from_handle(handle)
    np.testing.assert_allclose(target.world_position, np.array([0.7, 0.1, 0.85]))
    np.testing.assert_allclose(target.wxyz, np.array([0.0, 1.0, 0.0, 0.0]))


def test_collision_object_link_candidates_map_urdf_names_to_mjcf_visual_bodies() -> None:
    module = _load_example_module()

    assert module._collision_object_link_candidates("arm0_link_el1_0") == (
        "arm0_link_el1",
        "arm_link_el1",
    )
    assert module._collision_object_link_candidates("body_0") == ("body",)
    assert module._mjviser_body_candidates_for_ik_link("arm0_link_sh1") == (
        "arm0_link_sh1",
        "arm_link_sh1",
    )


def test_mjviser_collision_debug_rows_use_rendered_mujoco_geometries() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    adapter = SpotMujocoAdapter(model)
    adapter.reset_home(data)

    rows = module._mjviser_collision_debug_rows(
        adapter,
        data,
        (("body_0", "arm0_link_el0_0"),),
        max_rows=1,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.body_a == "body"
    assert row.body_b == "arm_link_el0"
    assert 0.05 < row.distance < 0.08
    np.testing.assert_allclose(
        np.linalg.norm(row.point_b_world - row.point_a_world),
        row.distance,
        atol=1e-9,
    )
    assert ("arm0_link_sh1_0", "arm0_link_el1_0") not in module._mjviser_collision_supported_pairs(
        adapter,
        (("arm0_link_sh1_0", "arm0_link_el1_0"),),
    )


def test_collision_pairs_from_solver_debug_rows_stay_curated_and_supported() -> None:
    module = _load_example_module()
    rows = [
        SimpleNamespace(object_a="arm0_link_el0_0", object_b="body_0"),
        SimpleNamespace(object_a="not_curated", object_b="body_0"),
        SimpleNamespace(object_a="body_0", object_b="arm0_link_el0_0"),
    ]
    supported = [
        ("body_0", "arm0_link_el0_0"),
        ("body_0", "arm0_link_wr0_0"),
    ]

    assert module._collision_pairs_from_solver_debug_rows(rows, supported) == [
        ("body_0", "arm0_link_el0_0")
    ]


def test_spot_viewer_defaults_disable_camera_tracking_for_ik_teleop() -> None:
    module = _load_example_module()
    viewer = SimpleNamespace(scene=SimpleNamespace(camera_tracking_enabled=True))

    module.configure_spot_viewer_defaults(viewer)

    assert viewer.scene.camera_tracking_enabled is False


def test_cli_default_policy_is_locomanipulation() -> None:
    module = _load_example_module()

    args = module.build_parser().parse_args([])

    assert args.policy == module.LocomanipPolicy.LOCOMANIP.value
    assert args.default_gains is True
    assert args.enable_teleop is False
    assert args.controller_port == "/dev/ttyUSB0"
    assert args.teleop_scale == pytest.approx(DEFAULT_TELEOP_SCALE_FACTOR)
    assert args.policy_rate_hz == pytest.approx(50.0)
    assert args.async_ik_rate_hz == pytest.approx(50.0)
    assert module.DEFAULT_ARM_GRAVITY_COMPENSATION_ENABLED is True


def test_controller_can_switch_between_locomanip_policies() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model()
    controller = module.SpotLocomanipController(
        model, policy=module.LocomanipPolicy.LOCOMANIP_STATIONARY
    )
    stationary_checkpoint = controller.checkpoint

    controller.last_output[:] = 1.0
    controller.set_policy(module.LocomanipPolicy.LOCOMANIP)

    assert controller.policy == module.LocomanipPolicy.LOCOMANIP
    assert controller.checkpoint != stationary_checkpoint
    assert controller.checkpoint.name == "locomanip_policy.onnx"
    np.testing.assert_allclose(controller.last_output, np.zeros(12))


def test_controller_uses_ik_outputs_as_policy_commands() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(model, policy=module.LocomanipPolicy.LOCOMANIP)
    controller.reset(model, data)

    arm_goal = np.array([0.1, -0.8, 1.7, 0.2, -0.6, 0.1, -0.7], dtype=float)
    body_goal = np.array([0.08, -0.06, 0.55], dtype=float)
    pose_goal = np.array([0.25, -0.15, 0.3], dtype=float)
    arm_gravity_torque = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=float)

    class FakeIK:
        enabled = True
        message = "fake ready"

        def solve_command(
            self,
            observation,
            arm_command,
            *,
            target_pose,
            body_command,
            desired_pose_command,
        ):
            self.observation = observation
            self.arm_command_in = arm_command.copy()
            self.target_pose = target_pose
            self.body_command_in = body_command.copy()
            self.desired_pose_command_in = desired_pose_command.copy()
            return SimpleNamespace(
                arm_command=arm_goal.copy(),
                body_command=body_goal.copy(),
                desired_pose_command=pose_goal.copy(),
                arm_gravity_torque=arm_gravity_torque.copy(),
                message="fake solved",
            )

    fake_ik = FakeIK()
    controller.ik = fake_ik
    target_pose = object()

    controller.step(
        model,
        data,
        control_mode="pose",
        ik_enabled=True,
        ik_target_pose=target_pose,
        gravity_compensation_enabled=True,
    )

    assert fake_ik.target_pose is target_pose
    np.testing.assert_allclose(fake_ik.arm_command_in, DEFAULT_ARM_COMMAND)
    np.testing.assert_allclose(fake_ik.body_command_in, DEFAULT_BODY_ROLL_PITCH_HEIGHT)
    np.testing.assert_allclose(fake_ik.desired_pose_command_in, np.zeros(3))
    expected_arm_command = arm_goal.copy()
    expected_arm_command[6] = DEFAULT_ARM_COMMAND[6]
    np.testing.assert_allclose(controller.commands.arm, expected_arm_command)
    np.testing.assert_allclose(controller.commands.body, body_goal)
    np.testing.assert_allclose(controller.commands.desired_pose, pose_goal)
    assert controller.last_ik_status == "fake solved"
    assert controller.commands.velocity[0] > 0.0
    assert controller.commands.velocity[1] < 0.0

    ctrl = controller.adapter.current_ctrl_targets(data)
    for name, value in zip((*MJCF_ARM_JOINT_NAMES, MJCF_GRIPPER_JOINT_NAME), expected_arm_command):
        assert ctrl[name] == pytest.approx(value)
    for name, value in zip(MJCF_ARM_JOINT_NAMES, arm_gravity_torque):
        assert data.qfrc_applied[controller.adapter._joint_addresses[name].qvel] == pytest.approx(
            value
        )


def test_controller_async_ik_does_not_block_policy_step() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(
        model,
        policy=module.LocomanipPolicy.LOCOMANIP,
        async_ik=True,
    )
    controller.reset(model, data)

    arm_goal = np.array([0.1, -0.8, 1.7, 0.2, -0.6, 0.1, -0.7], dtype=float)
    body_goal = np.array([0.08, -0.06, 0.55], dtype=float)
    pose_goal = np.array([0.25, -0.15, 0.3], dtype=float)

    class SlowIK:
        enabled = True
        message = "fake ready"

        def solve_command(
            self,
            observation,
            arm_command,
            *,
            target_pose,
            body_command,
            desired_pose_command,
        ):
            self.observation = observation
            self.arm_command_in = arm_command.copy()
            time.sleep(0.2)
            return SimpleNamespace(
                arm_command=arm_goal.copy(),
                body_command=body_goal.copy(),
                desired_pose_command=pose_goal.copy(),
                arm_gravity_torque=None,
                solve_time_ms=200.0,
                collision_constraint_time_ms=12.0,
                condition_number=42.0,
                message="fake async solved",
            )

        def compute_arm_gravity_torque(self, observation, arm_command):
            return None

    controller.ik = SlowIK()
    try:
        start = time.perf_counter()
        controller.step(
            model,
            data,
            control_mode="pose",
            ik_enabled=True,
            ik_target_pose=object(),
        )
        elapsed_ms = (time.perf_counter() - start) * 1e3

        assert elapsed_ms < 150.0
        np.testing.assert_allclose(controller.commands.arm, DEFAULT_ARM_COMMAND)
        assert controller.ik_solve_in_flight

        deadline = time.perf_counter() + 2.0
        while controller.ik_solve_in_flight and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert not controller.ik_solve_in_flight

        controller.step(
            model,
            data,
            control_mode="pose",
            ik_enabled=True,
            ik_target_pose=object(),
        )

        expected_arm_command = arm_goal.copy()
        expected_arm_command[6] = DEFAULT_ARM_COMMAND[6]
        np.testing.assert_allclose(controller.commands.arm, expected_arm_command)
        np.testing.assert_allclose(controller.commands.body, body_goal)
        np.testing.assert_allclose(controller.commands.desired_pose, pose_goal)
        assert controller.last_ik_status == "fake async solved"
        assert controller.last_ik_solve_time_ms == pytest.approx(200.0)
        assert controller.last_ik_collision_constraint_time_ms == pytest.approx(12.0)
        assert controller.last_ik_condition_number == pytest.approx(42.0)
    finally:
        controller.close()


def test_controller_keeps_policy_at_50hz_while_collision_ik_is_in_flight() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(
        model,
        policy=module.LocomanipPolicy.LOCOMANIP,
        async_ik=True,
    )
    controller.reset(model, data)

    class CountingSession:
        def __init__(self):
            self.run_count = 0

        def run(self, *_args, **_kwargs):
            self.run_count += 1
            return [np.full((1, controller.output_len), self.run_count, dtype=np.float32)]

    class SlowCollisionIK:
        enabled = True
        message = "fake collision IK ready"

        def solve_command(self, *_args, **_kwargs):
            time.sleep(0.25)
            return SimpleNamespace(
                arm_command=controller.commands.arm.copy(),
                body_command=controller.commands.body.copy(),
                desired_pose_command=controller.commands.desired_pose.copy(),
                arm_gravity_torque=None,
                solve_time_ms=250.0,
                collision_constraint_time_ms=200.0,
                condition_number=1.0,
                message="fake collision IK solved",
            )

        def compute_arm_gravity_torque(self, _observation, _arm_command):
            return None

    session = CountingSession()
    controller.policy_runtime.session = session
    controller.ik = SlowCollisionIK()
    simulated_seconds = 0.1
    steps = max(1, int(round(simulated_seconds / float(model.opt.timestep))))
    expected_policy_runs = 1 + int((steps * float(model.opt.timestep)) // controller.policy_dt)

    try:
        start = time.perf_counter()
        for _ in range(steps):
            controller.step(
                model,
                data,
                control_mode="pose",
                ik_enabled=True,
                ik_target_pose=object(),
            )
            mujoco.mj_step(model, data)
        elapsed_ms = (time.perf_counter() - start) * 1e3

        assert elapsed_ms < 100.0
        assert controller.ik_solve_in_flight
        assert session.run_count == expected_policy_runs
        assert controller.policy_step_count == expected_policy_runs
        assert controller.step_count == steps
    finally:
        controller.close()


def test_controller_async_ik_ignores_result_after_ik_disabled() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(
        model,
        policy=module.LocomanipPolicy.LOCOMANIP,
        async_ik=True,
    )
    controller.reset(model, data)

    arm_goal = INITIAL_ARM_COMMAND.copy()
    arm_goal[:6] += 0.3

    class SlowIK:
        enabled = True
        message = "fake ready"

        def solve_command(self, *args, **kwargs):
            time.sleep(0.05)
            return SimpleNamespace(
                arm_command=arm_goal.copy(),
                body_command=DEFAULT_BODY_ROLL_PITCH_HEIGHT.copy(),
                desired_pose_command=np.zeros(3, dtype=float),
                arm_gravity_torque=None,
                solve_time_ms=50.0,
                collision_constraint_time_ms=0.0,
                condition_number=1.0,
                message="stale async solved",
            )

        def compute_arm_gravity_torque(self, observation, arm_command):
            return None

    controller.ik = SlowIK()
    try:
        controller.step(model, data, control_mode="pose", ik_enabled=True, ik_target_pose=object())
        deadline = time.perf_counter() + 2.0
        while controller.ik_solve_in_flight and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert not controller.ik_solve_in_flight

        controller.step(model, data, control_mode="pose", ik_enabled=False)
        np.testing.assert_allclose(controller.commands.arm, DEFAULT_ARM_COMMAND)
        assert controller.last_ik_status == "fake ready"
    finally:
        controller.close()


def test_enabling_gui_ik_does_not_reseed_body_command_from_measured_pose() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(model, policy=module.LocomanipPolicy.LOCOMANIP)
    controller.reset(model, data)

    for _ in range(120):
        controller.step(model, data, control_mode="pose", ik_enabled=False)
        mujoco.mj_step(model, data)

    measured_pose = controller.adapter.read_policy_observation(data)["base_pose"]
    assert measured_pose[2] < DEFAULT_BODY_ROLL_PITCH_HEIGHT[2] - 0.05
    body_before = controller.commands.body.copy()
    desired_before = controller.commands.desired_pose.copy()

    class FakeHandle:
        def __init__(self, value=None):
            self.value = value
            self.visible = True
            self.disabled = False
            self.position = (0.0, 0.0, 0.0)
            self.wxyz = (1.0, 0.0, 0.0, 0.0)
            self._update = None
            self._click = None

        def on_update(self, callback):
            self._update = callback

        def on_click(self, callback):
            self._click = callback

        def trigger_update(self):
            assert self._update is not None
            self._update(None)

        def trigger_click(self):
            assert self._click is not None
            self._click(None)

    class FakeGui:
        def __init__(self):
            self.handles = {}
            self.folders = {}

        def add_folder(self, name, **kwargs):
            self.folders[name] = kwargs
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def _handle(self, name, value=None, **kwargs):
            handle = FakeHandle(value)
            handle.disabled = bool(kwargs.get("disabled", False))
            self.handles[name] = handle
            return handle

        def add_dropdown(self, name, *, initial_value, **_kwargs):
            return self._handle(name, initial_value, **_kwargs)

        def add_slider(self, name, *_args, initial_value, **_kwargs):
            return self._handle(name, initial_value, **_kwargs)

        def add_checkbox(self, name, *, initial_value, **_kwargs):
            return self._handle(name, initial_value, **_kwargs)

        def add_text(self, name, *, initial_value, **_kwargs):
            return self._handle(name, initial_value, **_kwargs)

        def add_button(self, name, **_kwargs):
            return self._handle(name, **_kwargs)

        def add_button_group(self, name, *, options, **_kwargs):
            return self._handle(name, options[0], **_kwargs)

    class FakeScene:
        def __init__(self):
            self.boxes = {}
            self.icospheres = {}
            self.line_segments = {}

        def add_box(self, name, *, position, visible, wxyz=None, **_kwargs):
            handle = FakeHandle()
            handle.position = position
            handle.wxyz = wxyz
            handle.visible = visible
            self.boxes[name] = handle
            return handle

        def add_transform_controls(self, name, *, position, wxyz, **_kwargs):
            handle = FakeHandle()
            handle.position = position
            handle.wxyz = wxyz
            self.handle = handle
            return handle

        def add_icosphere(self, name, *, visible, **_kwargs):
            handle = FakeHandle()
            handle.visible = visible
            self.icospheres[name] = handle
            return handle

        def add_line_segments(self, name, *, points, visible, **_kwargs):
            handle = FakeHandle()
            handle.points = points
            handle.visible = visible
            handle.remove = lambda: setattr(handle, "visible", False)
            self.line_segments[name] = handle
            return handle

    class FakeServer:
        def __init__(self):
            self.gui = FakeGui()
            self.scene = FakeScene()

    class FakeTeleop:
        connected = True
        streaming = False
        gripper_closed = False
        trigger_fraction = 0.5

        def __init__(self):
            self.process_count = 0
            self.reset_count = 0
            self.on_stream_start = lambda: None
            self.on_stream_stop = lambda: None
            self.on_reset = lambda: None

        def reset_reference(self):
            self.reset_count += 1

        def reset_runtime_state(self):
            self.streaming = False
            self.gripper_closed = False
            self.trigger_fraction = 0.0

        def set_enabled(self, enabled):
            self.enabled = bool(enabled)
            if not self.enabled:
                self.reset_runtime_state()

        def process_buttons(self):
            self.process_count += 1

        def relative_pose(self):
            return np.array([0.02, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])

    server = FakeServer()
    teleop = FakeTeleop()
    gui_sync = module._add_gui(
        server,
        controller,
        data,
        initial_control_mode="pose",
        teleop_controller=teleop,
    )
    leg_gain_scale = server.gui.handles["Leg gain scale"]
    assert leg_gain_scale.value == pytest.approx(DEFAULT_LEG_GAIN_SCALE)
    assert server.gui.handles["Controller"].value == "connected"
    assert server.gui.folders["Seer teleop"]["expand_by_default"] is True
    assert server.gui.handles["Enable teleop"].value is True
    assert teleop.enabled
    assert server.gui.handles["Position Scale"].value == pytest.approx(DEFAULT_TELEOP_SCALE_FACTOR)
    assert server.gui.handles["Position Scale"].disabled is False
    locomotion_sensitivity = server.gui.handles["Base assist"]
    assert locomotion_sensitivity.value == pytest.approx(1.0)
    locomotion_sensitivity.value = 1.5
    locomotion_sensitivity.trigger_update()
    assert controller.ik.locomotion_sensitivity == pytest.approx(1.5)
    arm_recovery_bias = server.gui.handles["Arm recovery bias"]
    assert arm_recovery_bias.value == pytest.approx(3.0)
    arm_recovery_bias.value = 4.5
    arm_recovery_bias.trigger_update()
    assert controller.ik.condition_arm_weight_scale == pytest.approx(4.5)
    collision_min_dist = server.gui.handles["Collision min dist (mm)"]
    collision_rows = server.gui.handles["Closest collision checks"]
    collision_tuning = server.gui.handles["Collision tuning"]
    collision_enable = server.gui.handles["Enable collision constraint"]
    assert collision_enable.disabled is True
    assert collision_enable.value is False
    assert collision_min_dist.disabled is True
    assert collision_rows.disabled is True
    assert collision_tuning.disabled is True
    assert server.gui.handles["Show collision debug"].disabled is True
    assert server.gui.handles["Show collision debug"].value is False
    assert server.gui.handles["Collision pairs"].value == "off (0 available)"
    assert server.gui.handles["Computation time"].value.endswith(
        "ms @ 50 Hz | IK 0.00 ms | collision 0.00 ms | cond -- | debug off"
    )
    for name in (
        "IK position gain",
        "IK orientation gain",
        "IK target solve mode",
        "IK dt (s)",
        "IK steps per frame",
        "IK adaptive dt",
        "IK adaptive dt max scale",
        "IK adaptive dt ref dist",
    ):
        assert server.gui.handles[name].disabled is True

    browser_only_server = FakeServer()
    browser_only_controller = module.SpotLocomanipController(
        model,
        policy=module.LocomanipPolicy.LOCOMANIP,
    )
    browser_only_controller.reset(model, data)
    browser_only_teleop = FakeTeleop()
    browser_only_teleop.connected = False
    module._add_gui(
        browser_only_server,
        browser_only_controller,
        data,
        initial_control_mode="pose",
        teleop_controller=browser_only_teleop,
    )
    assert browser_only_server.gui.folders["Seer teleop"]["expand_by_default"] is False
    assert browser_only_server.gui.handles["Enable teleop"].value is False
    assert browser_only_server.gui.handles["Enable teleop"].disabled is True

    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fl_hx")
    assert model.actuator_gainprm[actuator_id, 0] == pytest.approx(75.0)
    assert model.actuator_biasprm[actuator_id, 2] == pytest.approx(-1.875)
    leg_gain_scale.value = LEG_GAIN_SCALE_MIN
    leg_gain_scale.trigger_update()
    assert model.actuator_gainprm[actuator_id, 0] == pytest.approx(60.0)
    assert model.actuator_biasprm[actuator_id, 2] == pytest.approx(-1.5)
    assert "scale 1.00" in server.gui.handles["Actuator gains"].value
    objects_checkbox = server.gui.handles["Spawn manipulation objects"]
    assert not controller.adapter.manipulation_objects_enabled()
    assert all(not handle.visible for handle in server.scene.boxes.values())
    objects_checkbox.value = True
    objects_checkbox.trigger_update()
    assert controller.adapter.manipulation_objects_enabled()
    assert set(server.scene.boxes) == {
        f"/spot_manipulation_objects/{name}" for name in MANIPULATION_OBJECT_VISUAL_SPECS
    }
    assert all(handle.visible for handle in server.scene.boxes.values())
    objects_checkbox.value = False
    objects_checkbox.trigger_update()
    assert not controller.adapter.manipulation_objects_enabled()
    assert all(not handle.visible for handle in server.scene.boxes.values())

    ik_checkbox = server.gui.handles["Enable gripper IK"]
    assert not ik_checkbox.value
    ik_checkbox.disabled = False
    teleop_start = np.asarray(server.scene.handle.position, dtype=float).copy()
    teleop.streaming = True
    gui_sync.process_teleop()
    assert teleop.process_count == 1
    np.testing.assert_allclose(server.scene.handle.position, teleop_start, atol=1e-9)
    assert server.gui.handles["Gripper"].value == "50% closed"

    teleop_enable = server.gui.handles["Enable teleop"]
    teleop_enable.value = True
    teleop_enable.trigger_update()
    assert server.gui.handles["Position Scale"].disabled is False
    teleop.on_stream_start()
    assert ik_checkbox.value
    teleop.trigger_fraction = 0.5
    teleop.gripper_closed = True
    teleop_start = np.asarray(server.scene.handle.position, dtype=float).copy()
    process_count_before = teleop.process_count
    wrist_roll_command_before = float(controller.commands.arm[4])
    wrist_pitch_command_before = float(controller.commands.arm[5])
    wrist_roll_slider_before = float(server.gui.handles["arm_wr0"].value)
    wrist_pitch_slider_before = float(server.gui.handles["arm_wr1"].value)
    teleop.streaming = True
    gui_sync.process_teleop()
    assert teleop.process_count == process_count_before + 1
    np.testing.assert_allclose(
        server.scene.handle.position,
        teleop_start + np.array([0.02 * DEFAULT_TELEOP_SCALE_FACTOR, 0.0, 0.0]),
        atol=1e-9,
    )
    assert controller.commands.arm[4] == pytest.approx(wrist_roll_command_before)
    assert controller.commands.arm[5] == pytest.approx(wrist_pitch_command_before)
    assert server.gui.handles["arm_wr0"].value == pytest.approx(wrist_roll_slider_before)
    assert server.gui.handles["arm_wr1"].value == pytest.approx(wrist_pitch_slider_before)
    assert controller.commands.arm[6] == pytest.approx(0.5 * INITIAL_ARM_COMMAND[6])
    arm_slider = server.gui.handles["arm_f1x"]
    assert arm_slider.value == pytest.approx(0.5 * INITIAL_ARM_COMMAND[6])
    assert server.gui.handles["Gripper"].value == "50% closed"
    assert gui_sync.ik_target_pose() is not None
    ik_checkbox.value = True
    ik_checkbox.trigger_update()
    assert server.scene.handle.visible

    np.testing.assert_allclose(controller.commands.body, body_before)
    np.testing.assert_allclose(controller.commands.desired_pose, desired_before)

    ik_checkbox.value = False
    for name, value in zip(
        (*MJCF_ARM_JOINT_NAMES, MJCF_GRIPPER_JOINT_NAME),
        INITIAL_ARM_COMMAND,
    ):
        server.gui.handles[name].value = float(value)
    server.gui.handles["roll"].value = 0.25
    server.gui.handles["pitch"].value = -0.25
    server.gui.handles["height"].value = 0.35
    gui_sync.read_commands()
    assert not np.allclose(controller.commands.arm, DEFAULT_ARM_COMMAND)
    assert not np.allclose(controller.commands.body, DEFAULT_BODY_ROLL_PITCH_HEIGHT)

    ik_checkbox.value = True
    ik_checkbox.trigger_update()
    data.time = 12.3
    data.qpos[:7] = [0.8, -0.4, 0.06, 0.4, -0.2, 0.3, 0.1]
    for name, value in zip(SPOT_LEG_JOINT_NAMES, np.linspace(-1.0, 1.0, len(SPOT_LEG_JOINT_NAMES))):
        data.qpos[controller.adapter._joint_addresses[name].qpos] = float(value)
    for name, value in zip(
        (*MJCF_ARM_JOINT_NAMES, MJCF_GRIPPER_JOINT_NAME),
        np.linspace(-2.0, 2.0, len(MJCF_ARM_JOINT_NAMES) + 1),
    ):
        data.qpos[controller.adapter._joint_addresses[name].qpos] = float(value)
    data.qvel[:] = 1.0
    data.ctrl[:] = 9.0
    data.qfrc_applied[:] = 2.0
    data.xfrc_applied[:] = 3.0
    data.qacc_warmstart[:] = 4.0
    server.gui.handles["Reset sim to default"].trigger_click()

    assert not ik_checkbox.value
    assert teleop_enable.value
    assert teleop.enabled
    assert not teleop.streaming
    assert teleop.trigger_fraction == pytest.approx(0.0)
    assert server.gui.handles["Position Scale"].disabled is False
    assert server.gui.handles["Gripper"].value == "0% closed"
    assert not server.scene.handle.visible
    np.testing.assert_allclose(controller.commands.arm, DEFAULT_ARM_COMMAND)
    np.testing.assert_allclose(controller.commands.body, DEFAULT_BODY_ROLL_PITCH_HEIGHT)
    for name, value in zip(
        (*MJCF_ARM_JOINT_NAMES, MJCF_GRIPPER_JOINT_NAME),
        DEFAULT_ARM_COMMAND,
    ):
        assert server.gui.handles[name].value == pytest.approx(value)
    assert server.gui.handles["roll"].value == pytest.approx(DEFAULT_BODY_ROLL_PITCH_HEIGHT[0])
    assert server.gui.handles["pitch"].value == pytest.approx(DEFAULT_BODY_ROLL_PITCH_HEIGHT[1])
    assert server.gui.handles["height"].value == pytest.approx(DEFAULT_BODY_ROLL_PITCH_HEIGHT[2])
    assert data.time == pytest.approx(0.0)
    assert data.qpos[2] == pytest.approx(DEFAULT_STAND_BASE_HEIGHT)
    np.testing.assert_allclose(
        controller.adapter.read_joint_positions(data, SPOT_LEG_JOINT_NAMES),
        DEFAULT_STAND_LEG_JOINTS,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        controller.adapter.read_joint_positions(
            data, (*MJCF_ARM_JOINT_NAMES, MJCF_GRIPPER_JOINT_NAME)
        ),
        DEFAULT_ARM_COMMAND,
        atol=1e-12,
    )
    np.testing.assert_allclose(data.qvel, np.zeros(model.nv), atol=1e-12)
    np.testing.assert_allclose(data.qfrc_applied, np.zeros(model.nv), atol=1e-12)
    np.testing.assert_allclose(data.xfrc_applied, np.zeros_like(data.xfrc_applied), atol=1e-12)
    np.testing.assert_allclose(data.qacc_warmstart, np.zeros(model.nv), atol=1e-12)
    min_height = float(data.qpos[2])
    total_reset_steps = int(
        (
            module.RESET_SETTLE_SECONDS
            + module.AUTO_UNSTOW_DELAY_SECONDS
            + module.ARM_UNSTOW_INTERPOLATION_SECONDS
            + 0.2
        )
        / float(model.opt.timestep)
    )
    for _ in range(total_reset_steps):
        gui_sync.read_commands()
        controller.step(model, data, control_mode="pose", ik_enabled=bool(ik_checkbox.value))
        gui_sync.sync_programmatic_controls()
        mujoco.mj_step(model, data)
        min_height = min(min_height, float(data.qpos[2]))
        assert np.isfinite(data.qpos).all()
        assert np.isfinite(data.qvel).all()
        assert np.isfinite(data.ctrl).all()
        assert np.isfinite(data.qfrc_applied).all()
    assert min_height > 0.35
    assert not controller.arm_unstow_active
    np.testing.assert_allclose(controller.commands.arm, INITIAL_ARM_COMMAND, atol=1e-6)


def test_controller_delayed_unstow_progresses_while_ik_waits_for_target() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(model, policy=module.LocomanipPolicy.LOCOMANIP)
    controller.reset(model, data)
    module.schedule_interactive_unstow(controller)

    for _ in range(int(module.AUTO_UNSTOW_DELAY_SECONDS / float(model.opt.timestep))):
        controller.step(model, data, control_mode="pose", ik_enabled=True, ik_target_pose=None)
    assert np.max(np.abs(controller.commands.arm - DEFAULT_ARM_COMMAND)) > 0.0
    assert controller.arm_unstow_active


def test_reset_after_startup_unstow_restarts_stowed_and_unstows_again() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(model, policy=module.LocomanipPolicy.LOCOMANIP)
    controller.reset(model, data)
    module.schedule_interactive_unstow(controller)
    startup_steps = int(
        (module.AUTO_UNSTOW_DELAY_SECONDS + module.ARM_UNSTOW_INTERPOLATION_SECONDS + 0.5)
        / float(model.opt.timestep)
    )
    for _ in range(startup_steps):
        controller.step(model, data, control_mode="pose", ik_enabled=False)
        mujoco.mj_step(model, data)
    np.testing.assert_allclose(controller.commands.arm, INITIAL_ARM_COMMAND, atol=1e-6)
    assert not controller.arm_unstow_active

    controller.reset(model, data)
    module.schedule_interactive_unstow(controller)
    controller.adapter.apply_ctrl_targets(
        data,
        policy_action_to_mujoco_ctrl(controller.last_output, controller.commands.arm),
    )

    assert data.qpos[2] == pytest.approx(DEFAULT_STAND_BASE_HEIGHT)
    np.testing.assert_allclose(data.qvel, np.zeros(model.nv), atol=1e-12)
    np.testing.assert_allclose(data.qfrc_applied, np.zeros(model.nv), atol=1e-12)
    np.testing.assert_allclose(controller.commands.arm, DEFAULT_ARM_COMMAND)

    min_height = float(data.qpos[2])
    for _ in range(startup_steps):
        controller.step(model, data, control_mode="pose", ik_enabled=False)
        mujoco.mj_step(model, data)
        min_height = min(min_height, float(data.qpos[2]))
        assert np.isfinite(data.qpos).all()
        assert np.isfinite(data.qvel).all()
        assert np.isfinite(data.ctrl).all()
    assert min_height > 0.35
    assert not controller.arm_unstow_active
    np.testing.assert_allclose(controller.commands.arm, INITIAL_ARM_COMMAND, atol=1e-6)


def test_interactive_start_configuration_waits_before_unstowing_arm() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(model, policy=module.LocomanipPolicy.LOCOMANIP)
    controller.reset(model, data)

    module.schedule_interactive_unstow(controller)

    np.testing.assert_allclose(controller.commands.arm, DEFAULT_ARM_COMMAND)
    assert not controller.arm_unstow_active
    for _ in range(int(module.AUTO_UNSTOW_DELAY_SECONDS / float(model.opt.timestep)) - 1):
        controller.step(model, data, control_mode="pose", ik_enabled=False)
        np.testing.assert_allclose(controller.commands.arm, DEFAULT_ARM_COMMAND)
        assert not controller.arm_unstow_active

    controller.step(model, data, control_mode="pose", ik_enabled=False)

    assert controller.arm_unstow_active
    assert np.max(np.abs(controller.commands.arm - DEFAULT_ARM_COMMAND)) > 0.0


def test_unstowed_ik_current_target_uses_command_space_not_visual_pose() -> None:
    module = _load_example_module()
    spot_urdf = module._resolve_optional_spot_urdf(None)
    if spot_urdf is None:
        pytest.skip("Spot whole-body URDF is not available")
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(
        model,
        policy=module.LocomanipPolicy.LOCOMANIP,
        spot_urdf=spot_urdf,
    )
    controller.reset(model, data)
    controller.start_arm_unstow_interpolation()
    for _ in range(int(2.5 / float(model.opt.timestep))):
        controller.step(
            model,
            data,
            control_mode="pose",
            ik_enabled=False,
            gravity_compensation_enabled=True,
        )
        mujoco.mj_step(model, data)

    controller.commands.arm[:] = controller.adapter.read_measured_arm_command(data)
    visual_pos, visual_wxyz = controller.adapter.body_pose_wxyz(data, "arm_link_fngr")
    visual_target = module.target_pose_from_wxyz(visual_pos, visual_wxyz)
    command_target = controller.ik.command_tool_pose(
        controller.adapter.read_policy_observation(data),
        controller.commands.arm,
        body_command=controller.commands.body,
        desired_pose_command=controller.commands.desired_pose,
    )
    assert command_target is not None
    visual_to_command_error = np.linalg.norm(
        np.asarray(command_target.translation, dtype=float)
        - np.asarray(visual_target.translation, dtype=float)
    )
    assert visual_to_command_error > 0.01

    arm_before = controller.commands.arm.copy()
    body_before = controller.commands.body.copy()
    controller.step(
        model,
        data,
        control_mode="pose",
        ik_enabled=True,
        ik_target_pose=command_target,
        gravity_compensation_enabled=True,
    )

    np.testing.assert_allclose(controller.commands.arm, arm_before, atol=1e-9)
    np.testing.assert_allclose(controller.commands.body, body_before, atol=1e-9)
    assert controller.last_ik_status.startswith("SUCCESS")

    upward_target = module._target_pose_from_matrix(
        np.asarray(command_target.translation, dtype=float) + np.array([0.0, 0.0, 0.05]),
        np.asarray(command_target.rotation, dtype=float),
    )
    for _ in range(30):
        controller.step(
            model,
            data,
            control_mode="pose",
            ik_enabled=True,
            ik_target_pose=upward_target,
            gravity_compensation_enabled=True,
        )
        mujoco.mj_step(model, data)

    assert np.max(np.abs(controller.commands.arm - arm_before)) > 0.02
    np.testing.assert_allclose(controller.commands.body, body_before, atol=1e-9)
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
    assert controller.adapter.read_policy_observation(data)["base_pose"][2] > 0.35


def test_locomanip_ik_base_assist_tunes_solver_native_xy_yaw_motion() -> None:
    module = _load_example_module()
    spot_urdf = module._resolve_optional_spot_urdf(None)
    if spot_urdf is None:
        pytest.skip("Spot whole-body URDF is not available")
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(
        model,
        policy=module.LocomanipPolicy.LOCOMANIP,
        spot_urdf=spot_urdf,
    )
    controller.reset(model, data)
    controller.commands.arm[:] = INITIAL_ARM_COMMAND
    for _ in range(50):
        controller.step(model, data, control_mode="pose", ik_enabled=False)
        mujoco.mj_step(model, data)

    observation = controller.adapter.read_policy_observation(data)
    arm_before = controller.adapter.read_measured_arm_command(data)
    body_before = controller.commands.body.copy()
    desired_before = controller.commands.desired_pose.copy()
    command_target = controller.ik.command_tool_pose(
        observation,
        arm_before,
        body_command=body_before,
        desired_pose_command=desired_before,
    )
    assert command_target is not None

    def solve_offset(
        translation_offset: np.ndarray,
        yaw_offset: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        c = np.cos(yaw_offset)
        s = np.sin(yaw_offset)
        yaw_rotation = np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        target = module._target_pose_from_matrix(
            np.asarray(command_target.translation, dtype=float) + translation_offset,
            yaw_rotation @ np.asarray(command_target.rotation, dtype=float),
        )
        arm = arm_before.copy()
        body = body_before.copy()
        desired = desired_before.copy()
        status = ""
        for _ in range(20):
            result = controller.ik.solve_command(
                observation,
                arm,
                target_pose=target,
                body_command=body,
                desired_pose_command=desired,
            )
            arm = result.arm_command
            body = result.body_command
            desired = result.desired_pose_command
            status = result.message
        return arm - arm_before, desired - desired_before, status

    x_arm_delta, x_pose_delta, x_status = solve_offset(np.array([0.02, 0.0, 0.0]))
    x_large_arm_delta, x_large_pose_delta, x_large_status = solve_offset(np.array([0.12, 0.0, 0.0]))
    y_arm_delta, y_pose_delta, y_status = solve_offset(np.array([0.0, 0.04, 0.0]))
    yaw_arm_delta, yaw_pose_delta, yaw_status = solve_offset(
        np.zeros(3),
        np.deg2rad(15.0),
    )
    default_arm_delta, default_pose_delta, default_status = solve_offset(np.array([0.08, 0.0, 0.0]))

    assert x_status.startswith("SUCCESS")
    assert x_large_status.startswith("SUCCESS")
    assert y_status.startswith("SUCCESS")
    assert yaw_status.startswith("SUCCESS")
    assert default_status.startswith("SUCCESS")
    assert abs(float(x_pose_delta[0])) > 0.015
    assert abs(float(x_large_pose_delta[0])) > abs(float(x_pose_delta[0])) + 0.05
    assert abs(float(y_pose_delta[1])) > 0.02
    assert abs(float(yaw_pose_delta[2])) > 0.10
    assert abs(float(default_pose_delta[0])) > 0.07
    assert np.linalg.norm(default_arm_delta[:6]) < 0.05
    assert np.linalg.norm(x_arm_delta[:6]) > 1e-3
    assert np.linalg.norm(x_large_arm_delta[:6]) > np.linalg.norm(x_arm_delta[:6])
    assert np.linalg.norm(y_arm_delta[:6]) > 1e-3
    assert np.linalg.norm(yaw_arm_delta[:6]) > 0.02

    controller.ik.locomotion_sensitivity = 0.25
    controller.ik.reset_reference()
    low_compliance_arm_delta, low_compliance_pose_delta, low_compliance_status = solve_offset(
        np.array([0.08, 0.0, 0.0])
    )
    assert low_compliance_status.startswith("SUCCESS")

    controller.ik.locomotion_sensitivity = 4.0
    controller.ik.reset_reference()
    high_compliance_arm_delta, high_compliance_pose_delta, high_compliance_status = solve_offset(
        np.array([0.08, 0.0, 0.0])
    )
    assert high_compliance_status.startswith("SUCCESS")
    assert np.linalg.norm(low_compliance_arm_delta[:6]) > 5.0 * np.linalg.norm(
        high_compliance_arm_delta[:6]
    )
    assert abs(float(high_compliance_pose_delta[0])) > abs(float(low_compliance_pose_delta[0])) + 0.02

    controller.ik.locomotion_sensitivity = 1.0
    controller.ik.reset_reference()
    locomotion_arm_delta, locomotion_pose_delta, locomotion_status = solve_offset(
        np.array([0.25, 0.0, 0.0])
    )
    assert locomotion_status.startswith("SUCCESS")
    assert abs(locomotion_pose_delta[0]) > 0.05
    assert abs(locomotion_pose_delta[0]) > np.linalg.norm(locomotion_arm_delta[:6])


def test_locomanip_ik_configures_curated_collision_constraint_when_available() -> None:
    module = _load_example_module()
    spot_urdf = module._resolve_optional_spot_urdf(None)
    if spot_urdf is None:
        pytest.skip("Spot whole-body URDF is not available")
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(
        model,
        policy=module.LocomanipPolicy.LOCOMANIP,
        spot_urdf=spot_urdf,
    )
    if not controller.ik.collision_available:
        pytest.skip("EmbodiK collision constraint API or Spot collision pairs unavailable")
    assert controller.ik.enable_collision is False
    controller.reset(model, data)
    controller.commands.arm[:] = INITIAL_ARM_COMMAND
    for _ in range(50):
        controller.step(model, data, control_mode="pose", ik_enabled=False)
        mujoco.mj_step(model, data)

    observation = controller.adapter.read_policy_observation(data)
    arm_before = controller.adapter.read_measured_arm_command(data)
    target = controller.ik.command_tool_pose(
        observation,
        arm_before,
        body_command=controller.commands.body,
        desired_pose_command=controller.commands.desired_pose,
    )
    assert target is not None
    assert controller.ik.collision_debug_rows() == []
    controller.ik.enable_collision = True
    controller.ik.collision_min_distance = 0.04
    controller.ik.collision_max_constraints = 2
    controller.ik.collision_tuning_mode = "speed"

    result = controller.ik.solve_command(
        observation,
        arm_before,
        target_pose=target,
        body_command=controller.commands.body,
        desired_pose_command=controller.commands.desired_pose,
    )

    assert result.status_name.startswith("SUCCESS")
    assert controller.ik.collision_include_pairs
    assert controller.ik._collision_config_key == (True, 0.04, 2, "speed")
    controller.commands.desired_pose[:] = [1.0, -1.0, 0.5]
    measured_debug_q = controller.ik.sync_measured_configuration_for_debug(
        observation,
        arm_before,
    )
    np.testing.assert_allclose(measured_debug_q[:3], observation["base_pose"][:3], atol=1e-12)
    rows = controller.ik.collision_debug_rows()
    assert isinstance(rows, list)


def test_locomanip_collision_constraint_configuration_is_cached() -> None:
    from example_helpers.spot_whole_body_ik import OptionalSpotWholeBodyIK

    class FakeSolver:
        def __init__(self) -> None:
            self.configures: list[tuple[float, tuple[tuple[str, str], ...], bool, int]] = []
            self.clears = 0

        def configure_collision_constraint(
            self,
            *,
            min_distance,
            include_pairs,
            exclude_pairs,
            nearest_points_all_pairs,
            max_constraints,
        ) -> None:
            self.configures.append(
                (
                    float(min_distance),
                    tuple(include_pairs),
                    bool(nearest_points_all_pairs),
                    int(max_constraints),
                )
            )

        def clear_collision_constraint(self) -> None:
            self.clears += 1

    ik = OptionalSpotWholeBodyIK.__new__(OptionalSpotWholeBodyIK)
    ik.solver = FakeSolver()
    ik.enable_collision = True
    ik.collision_min_distance = 0.04
    ik.collision_max_constraints = 1
    ik.collision_tuning_mode = "speed"
    ik._collision_include_pairs = [("arm0_link_wr0_0", "body_0")]
    ik._collision_config_key = None

    ik._configure_collision_constraint()
    ik._configure_collision_constraint()

    assert len(ik.solver.configures) == 1
    assert ik.solver.configures[0] == (0.04, (("arm0_link_wr0_0", "body_0"),), False, 1)

    ik.collision_max_constraints = 2
    ik._configure_collision_constraint()

    assert len(ik.solver.configures) == 2
    assert ik.solver.configures[-1][-1] == 2

    ik.enable_collision = False
    ik._configure_collision_constraint()
    ik._configure_collision_constraint()

    assert ik.solver.clears == 1


def test_locomanip_controller_preserves_streamed_gripper_across_ik_output() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(model, policy=module.LocomanipPolicy.LOCOMANIP)
    controller.reset(model, data)
    controller.commands.arm[:] = INITIAL_ARM_COMMAND
    controller.commands.arm[6] = -0.25

    class FakeIK:
        enabled = True
        message = "fake ready"

        def solve_command(
            self,
            observation,
            arm_command,
            *,
            target_pose,
            body_command,
            desired_pose_command,
        ):
            self.arm_command_in = arm_command.copy()
            bad_output = arm_command.copy()
            bad_output[:6] += np.array([0.01, -0.02, 0.03, -0.04, 0.05, -0.06])
            bad_output[6] = 0.0
            return SimpleNamespace(
                arm_command=bad_output,
                body_command=body_command.copy(),
                desired_pose_command=desired_pose_command.copy(),
                arm_gravity_torque=None,
                message="fake success",
            )

    fake_ik = FakeIK()
    controller.ik = fake_ik

    controller.step(model, data, control_mode="pose", ik_enabled=True, ik_target_pose=object())

    np.testing.assert_allclose(fake_ik.arm_command_in, np.r_[INITIAL_ARM_COMMAND[:6], -0.25])
    np.testing.assert_allclose(
        controller.commands.arm[:6],
        INITIAL_ARM_COMMAND[:6] + np.array([0.01, -0.02, 0.03, -0.04, 0.05, -0.06]),
    )
    assert controller.commands.arm[6] == pytest.approx(-0.25)


def test_headless_gripper_close_tracks_without_arm_instability() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(model, policy=module.LocomanipPolicy.LOCOMANIP)
    controller.reset(model, data)
    module.schedule_interactive_unstow(controller)

    startup_steps = int(
        (module.AUTO_UNSTOW_DELAY_SECONDS + module.ARM_UNSTOW_INTERPOLATION_SECONDS + 0.5)
        / float(model.opt.timestep)
    )
    min_height = float(data.qpos[2])
    for _ in range(startup_steps):
        controller.step(model, data, control_mode="pose", ik_enabled=False)
        mujoco.mj_step(model, data)
        min_height = min(min_height, float(data.qpos[2]))
        assert np.isfinite(data.qpos).all()
        assert np.isfinite(data.qvel).all()
        assert np.isfinite(data.ctrl).all()
        assert np.isfinite(data.qfrc_applied).all()

    assert min_height > 0.35
    assert not controller.arm_unstow_active
    np.testing.assert_allclose(controller.commands.arm, INITIAL_ARM_COMMAND, atol=1e-6)

    open_measured = controller.adapter.read_measured_arm_command(data)
    _, open_finger_wxyz = controller.adapter.body_pose_wxyz(data, "arm_link_fngr")
    arm_before_close = open_measured[:6].copy()
    open_gripper = float(open_measured[6])
    assert abs(open_gripper - module.GRIPPER_OPEN_COMMAND) < abs(
        module.GRIPPER_OPEN_COMMAND - module.GRIPPER_CLOSED_COMMAND
    )

    controller.commands.arm[:] = INITIAL_ARM_COMMAND
    controller.commands.arm[6] = module.GRIPPER_CLOSED_COMMAND
    for _ in range(int(0.75 / float(model.opt.timestep))):
        controller.step(model, data, control_mode="pose", ik_enabled=False)
        mujoco.mj_step(model, data)
        min_height = min(min_height, float(data.qpos[2]))
        assert np.isfinite(data.qpos).all()
        assert np.isfinite(data.qvel).all()
        assert np.isfinite(data.ctrl).all()
        assert np.isfinite(data.qfrc_applied).all()

    closed_measured = controller.adapter.read_measured_arm_command(data)
    _, closed_finger_wxyz = controller.adapter.body_pose_wxyz(data, "arm_link_fngr")
    finger_quat_dot = abs(float(np.dot(open_finger_wxyz, closed_finger_wxyz)))
    finger_rotation = 2.0 * np.arccos(np.clip(finger_quat_dot, -1.0, 1.0))
    closed_gripper = float(closed_measured[6])
    assert abs(closed_gripper - module.GRIPPER_CLOSED_COMMAND) < 0.2
    assert finger_rotation > 1.0
    assert abs(closed_gripper - module.GRIPPER_CLOSED_COMMAND) < 0.25 * abs(
        open_gripper - module.GRIPPER_CLOSED_COMMAND
    )
    assert abs(closed_measured[4] - arm_before_close[4]) < 0.05
    assert abs(closed_measured[5] - arm_before_close[5]) < 0.05
    assert np.max(np.abs(closed_measured[:6] - arm_before_close)) < 0.25
    assert min_height > 0.35
    assert float(np.linalg.norm(data.qvel)) < 40.0


def test_controller_feeds_gravity_torque_when_ik_is_disabled() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(model, policy=module.LocomanipPolicy.LOCOMANIP)
    controller.reset(model, data)

    arm_gravity_torque = np.array([0.5, -0.75, 1.25, -1.5, 2.0, -2.25], dtype=float)

    class FakeIK:
        enabled = True
        message = "fake gravity ready"

        def compute_arm_gravity_torque(self, observation, arm_command):
            self.observation = observation
            self.arm_command_in = arm_command.copy()
            return arm_gravity_torque.copy()

    fake_ik = FakeIK()
    controller.ik = fake_ik

    controller.step(model, data, control_mode="pose", ik_enabled=False)

    assert not hasattr(fake_ik, "arm_command_in")
    np.testing.assert_allclose(data.qfrc_applied, np.zeros(model.nv))

    controller.step(
        model,
        data,
        control_mode="pose",
        ik_enabled=False,
        gravity_compensation_enabled=True,
    )

    np.testing.assert_allclose(fake_ik.arm_command_in, DEFAULT_ARM_COMMAND)
    assert controller.last_ik_status == "fake gravity ready"
    for name, value in zip(MJCF_ARM_JOINT_NAMES, arm_gravity_torque):
        assert data.qfrc_applied[controller.adapter._joint_addresses[name].qvel] == pytest.approx(
            value
        )


def test_controller_feeds_gravity_torque_when_ik_has_no_armed_target() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(model, policy=module.LocomanipPolicy.LOCOMANIP)
    controller.reset(model, data)

    arm_gravity_torque = np.array([0.25, -0.5, 0.75, -1.0, 1.25, -1.5], dtype=float)

    class FakeIK:
        enabled = True
        message = "fake ready"

        def solve_command(
            self,
            observation,
            arm_command,
            *,
            target_pose,
            body_command,
            desired_pose_command,
        ):
            self.solve_target_pose = target_pose
            return SimpleNamespace(
                arm_command=None,
                body_command=None,
                desired_pose_command=None,
                arm_gravity_torque=None,
                message="fake waiting for target",
            )

        def compute_arm_gravity_torque(self, observation, arm_command):
            self.gravity_arm_command_in = arm_command.copy()
            return arm_gravity_torque.copy()

    fake_ik = FakeIK()
    controller.ik = fake_ik

    controller.step(
        model,
        data,
        control_mode="pose",
        ik_enabled=True,
        ik_target_pose=None,
        gravity_compensation_enabled=True,
    )

    assert fake_ik.solve_target_pose is None
    np.testing.assert_allclose(fake_ik.gravity_arm_command_in, DEFAULT_ARM_COMMAND)
    for name, value in zip(MJCF_ARM_JOINT_NAMES, arm_gravity_torque):
        assert data.qfrc_applied[controller.adapter._joint_addresses[name].qvel] == pytest.approx(
            value
        )


def test_adapter_applies_and_clears_arm_joint_torques() -> None:
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    adapter = SpotMujocoAdapter(model)
    torques = np.array([1.0, -1.5, 2.0, -2.5, 3.0, -3.5], dtype=float)

    adapter.apply_arm_joint_torques(data, torques)

    for name, value in zip(MJCF_ARM_JOINT_NAMES, torques):
        assert data.qfrc_applied[adapter._joint_addresses[name].qvel] == pytest.approx(value)

    adapter.clear_applied_forces(data)

    np.testing.assert_allclose(data.qfrc_applied, np.zeros(model.nv))


def test_public_menagerie_spot_scene_loads_and_maps_controls() -> None:
    scene = resolve_spot_scene_arm_xml()
    assert scene.name == "scene_arm.xml"
    model = load_spot_mujoco_model(scene)
    data = mujoco.MjData(model)
    adapter = SpotMujocoAdapter(model)
    adapter.reset_home(data)

    obs = adapter.read_policy_observation(data)
    assert obs["joint_pos"].shape == (12,)
    assert obs["arm_state"].shape == (12,)
    assert obs["gripper_state"].shape == (2,)

    ctrl = policy_action_to_mujoco_ctrl(np.zeros(12), DEFAULT_ARM_COMMAND)
    adapter.apply_ctrl_targets(data, ctrl)
    assert np.isfinite(data.ctrl).all()


def test_manipulation_objects_toggle_visibility_and_contact() -> None:
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    adapter = SpotMujocoAdapter(model)
    adapter.reset_home(data)
    adapter.set_manipulation_objects_enabled(data, False)

    assert adapter.has_manipulation_objects()
    assert not adapter.manipulation_objects_enabled()

    for name in MANIPULATION_OBJECT_BODY_POSES:
        address = adapter._manipulation_freejoint_addresses[name]
        np.testing.assert_allclose(
            data.qpos[address.qpos : address.qpos + 3],
            [0.0, 0.0, MANIPULATION_OBJECT_HIDDEN_Z],
        )
        np.testing.assert_allclose(data.qvel[address.qvel : address.qvel + 6], 0.0)
    for name in MANIPULATION_OBJECT_GEOM_NAMES:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_contype[geom_id] == 0
        assert model.geom_conaffinity[geom_id] == 0
        assert model.geom_rgba[geom_id, 3] == pytest.approx(0.0)

    adapter.set_manipulation_objects_enabled(data, True)

    assert adapter.manipulation_objects_enabled()
    for name, pose in MANIPULATION_OBJECT_BODY_POSES.items():
        address = adapter._manipulation_freejoint_addresses[name]
        np.testing.assert_allclose(
            data.qpos[address.qpos : address.qpos + 3],
            np.asarray(pose, dtype=float),
        )
        np.testing.assert_allclose(
            data.qpos[address.qpos + 3 : address.qpos + 7],
            [1.0, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(data.qvel[address.qvel : address.qvel + 6], 0.0)
    for name in MANIPULATION_OBJECT_GEOM_NAMES:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_contype[geom_id] == 1
        assert model.geom_conaffinity[geom_id] == 1
        assert model.geom_rgba[geom_id, 3] == pytest.approx(1.0)

    min_manip_distance = 0.0
    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        if geom1 in MANIPULATION_OBJECT_GEOM_NAMES or geom2 in MANIPULATION_OBJECT_GEOM_NAMES:
            min_manip_distance = min(min_manip_distance, float(contact.dist))
    assert min_manip_distance > -1e-4

    cube0 = adapter._manipulation_freejoint_addresses["manip_small_cube_0"]
    cube1 = adapter._manipulation_freejoint_addresses["manip_small_cube_1"]
    data.qpos[cube1.qpos : cube1.qpos + 3] = data.qpos[cube0.qpos : cube0.qpos + 3]
    mujoco.mj_forward(model, data)
    object_object_contact_count = 0
    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        if {geom1, geom2} == {"manip_small_cube_0_geom", "manip_small_cube_1_geom"}:
            object_object_contact_count += 1
    assert object_object_contact_count > 0

    adapter.set_manipulation_objects_enabled(data, False)

    assert not adapter.manipulation_objects_enabled()
    for name in MANIPULATION_OBJECT_BODY_POSES:
        address = adapter._manipulation_freejoint_addresses[name]
        np.testing.assert_allclose(
            data.qpos[address.qpos : address.qpos + 3],
            [0.0, 0.0, MANIPULATION_OBJECT_HIDDEN_Z],
        )
        np.testing.assert_allclose(data.qvel[address.qvel : address.qvel + 6], 0.0)
    for name in MANIPULATION_OBJECT_GEOM_NAMES:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_contype[geom_id] == 0
        assert model.geom_conaffinity[geom_id] == 0
        assert model.geom_rgba[geom_id, 3] == pytest.approx(0.0)


def test_manipulation_cube_is_dynamic_under_mujoco_physics() -> None:
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    adapter = SpotMujocoAdapter(model)
    adapter.reset_home(data)
    adapter.set_manipulation_objects_enabled(data, True)

    address = adapter._manipulation_freejoint_addresses["manip_cube"]
    data.qpos[address.qpos : address.qpos + 3] = [1.0, 0.0, 1.5]
    data.qpos[address.qpos + 3 : address.qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[address.qvel : address.qvel + 6] = 0.0
    mujoco.mj_forward(model, data)
    start_z = float(data.qpos[address.qpos + 2])

    for _ in range(30):
        mujoco.mj_step(model, data)

    assert data.qpos[address.qpos + 2] < start_z - 0.01
    assert np.linalg.norm(data.qvel[address.qvel : address.qvel + 3]) > 0.01


def test_locomanip_rollout_stays_finite_with_manipulation_obstacles_enabled() -> None:
    module = _load_example_module()
    model = load_spot_mujoco_model()
    data = mujoco.MjData(model)
    controller = module.SpotLocomanipController(model, policy=module.LocomanipPolicy.LOCOMANIP)
    controller.reset(model, data)
    controller.adapter.set_manipulation_objects_enabled(data, True)
    fl_foot_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "FL")
    address = controller.adapter._manipulation_freejoint_addresses["manip_cube"]
    data.qpos[address.qpos : address.qpos + 3] = data.geom_xpos[fl_foot_geom_id].copy()
    data.qpos[address.qpos + 3 : address.qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[address.qvel : address.qvel + 6] = 0.0
    mujoco.mj_forward(model, data)
    assert controller.adapter.manipulation_object_contact_count(data) > 0

    min_height = float(data.qpos[2])
    contact_steps = 0
    for _ in range(80):
        controller.step(model, data, control_mode="pose", ik_enabled=False)
        mujoco.mj_step(model, data)
        min_height = min(min_height, float(data.qpos[2]))
        contact_steps += int(controller.adapter.manipulation_object_contact_count(data) > 0)
        assert np.isfinite(data.qpos).all()
        assert np.isfinite(data.qvel).all()
        assert np.isfinite(data.ctrl).all()

    assert min_height > 0.32
    assert contact_steps > 0


@pytest.mark.parametrize("policy", list(LocomanipPolicy))
def test_headless_example_runs_for_each_locomanip_policy(policy: LocomanipPolicy) -> None:
    module = _load_example_module()
    args = argparse.Namespace(
        policy=policy.value,
        policy_checkpoint=None,
        scene=None,
        spot_urdf=None,
        headless=True,
        steps=3,
        dt=0.01,
        spawn_manipulation_objects=False,
        default_gains=True,
        port=8080,
        control_mode="pose",
    )
    module.run_headless(args)


def test_headless_example_runs_with_manipulation_objects_enabled() -> None:
    module = _load_example_module()
    args = argparse.Namespace(
        policy=LocomanipPolicy.LOCOMANIP.value,
        policy_checkpoint=None,
        scene=None,
        spot_urdf=None,
        headless=True,
        steps=5,
        dt=0.01,
        spawn_manipulation_objects=True,
        default_gains=True,
        port=8080,
        control_mode="pose",
    )
    module.run_headless(args)
