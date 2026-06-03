"""Headless tests for the optional EmbodiK Spot whole-body IK adapter."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

embodik = pytest.importorskip("embodik")

from example_helpers.spot_locomanip_policy import (  # noqa: E402
    DEFAULT_ARM_COMMAND,
    DEFAULT_BODY_ROLL_PITCH_HEIGHT,
    DEFAULT_STAND_BASE_HEIGHT,
    DEFAULT_STAND_LEG_JOINTS,
    HEIGHT_RANGE,
    INITIAL_ARM_COMMAND,
)
from example_helpers.spot_whole_body_ik import OptionalSpotWholeBodyIK  # noqa: E402
from example_helpers.spot_whole_body_ik import (  # noqa: E402
    SPOT_COLLISION_MIN_DISTANCE_M,
    SPOT_URDF_ENV_VAR,
    STANDARD_FULL_BODY_TORSO_POSE_HALF_RANGE,
    SpotFullBodyIK,
    SpotFullBodyIKConfig,
    SpotFullBodyIKMode,
    _roll_pitch_from_rotation,
    _rotation_from_rpy,
    _rotation_to_xyzw,
    _yaw_from_rotation,
    resolve_spot_ik_urdf,
    spot_collision_pairs_from_references,
)


def _resolve_spot_ik_urdf() -> Path:
    if path := resolve_spot_ik_urdf():
        return path
    pytest.skip(
        f"Spot whole-body URDF not available; set {SPOT_URDF_ENV_VAR} to run "
        "the headless Spot IK tests"
    )


def _load_full_body_viser_module():
    path = _EXAMPLES_DIR / "08_spot_full_body_ik_viser.py"
    spec = importlib.util.spec_from_file_location("spot_full_body_ik_viser_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _policy_observation() -> dict[str, np.ndarray]:
    return {
        "base_pose": np.array(
            [0.0, 0.0, DEFAULT_STAND_BASE_HEIGHT, 1.0, 0.0, 0.0, 0.0],
            dtype=float,
        ),
        "joint_pos": DEFAULT_STAND_LEG_JOINTS.copy(),
    }


def _joint_value(ik: OptionalSpotWholeBodyIK, q: np.ndarray, joint_name: str) -> float:
    idx = int(ik.robot.get_joint_config_index(joint_name))
    return float(q[idx])


def _commanded_base_pose(
    body_command: np.ndarray = DEFAULT_BODY_ROLL_PITCH_HEIGHT,
    desired_pose: np.ndarray | None = None,
) -> np.ndarray:
    desired = np.zeros(3, dtype=float) if desired_pose is None else np.asarray(desired_pose)
    body = np.asarray(body_command, dtype=float)
    quat_xyzw = _rotation_to_xyzw(_rotation_from_rpy(body[0], body[1], desired[2]))
    return np.array(
        [desired[0], desired[1], body[2], quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=float,
    )


def _pose_matrix(pose) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.asarray(pose.rotation, dtype=float)
    matrix[:3, 3] = np.asarray(pose.translation, dtype=float)
    return matrix


def _max_joint_delta(robot, q_a: np.ndarray, q_b: np.ndarray, joint_names: list[str]) -> float:
    max_delta = 0.0
    for joint_name in joint_names:
        idx = int(robot.get_joint_config_index(joint_name))
        max_delta = max(max_delta, abs(float(q_a[idx] - q_b[idx])))
    return max_delta


def _max_arm_command_delta(command_a: np.ndarray, command_b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(command_a[:6]) - np.asarray(command_b[:6]))))


def test_spot_whole_body_ik_loads_backup_urdf_and_resolves_frames() -> None:
    ik = OptionalSpotWholeBodyIK(_resolve_spot_ik_urdf(), dt=0.01)

    assert ik.enabled
    assert ik.robot is not None
    assert ik.solver is not None
    assert ik.body_frame == "body"
    assert ik.tool_frame in {"arm0_link_wr1", "arm_link_wr1"}
    assert ik.message.startswith("ready:")


def test_spot_whole_body_ik_uses_bundled_public_urdf_by_default(monkeypatch) -> None:
    monkeypatch.delenv(SPOT_URDF_ENV_VAR, raising=False)

    path = resolve_spot_ik_urdf()

    assert path == (_EXAMPLES_DIR / "assets" / "spot_description" / "urdf" / "spot_with_arm.urdf")


def test_spot_whole_body_gripper_command_does_not_move_wrist_tool_frame() -> None:
    ik = OptionalSpotWholeBodyIK(_resolve_spot_ik_urdf(), dt=0.01)
    observation = _policy_observation()
    body = DEFAULT_BODY_ROLL_PITCH_HEIGHT.copy()
    desired_pose = np.zeros(3, dtype=float)
    open_arm = INITIAL_ARM_COMMAND.copy()
    closed_arm = INITIAL_ARM_COMMAND.copy()
    closed_arm[6] = 0.0

    assert ik.tool_frame in {"arm0_link_wr1", "arm_link_wr1"}
    ik._sync_configuration(
        observation,
        open_arm,
        base_pose_override=_commanded_base_pose(body, desired_pose),
    )
    open_tool = ik.robot.get_frame_pose(ik.tool_frame)
    ik._sync_configuration(
        observation,
        closed_arm,
        base_pose_override=_commanded_base_pose(body, desired_pose),
    )
    closed_tool = ik.robot.get_frame_pose(ik.tool_frame)

    np.testing.assert_allclose(closed_tool.translation, open_tool.translation, atol=1e-10)
    np.testing.assert_allclose(closed_tool.rotation, open_tool.rotation, atol=1e-10)

    result = ik.solve_command(
        observation,
        closed_arm,
        target_pose=open_tool,
        body_command=body,
        desired_pose_command=desired_pose,
    )

    assert result.success
    assert result.arm_command is not None
    np.testing.assert_allclose(result.arm_command[:6], closed_arm[:6], atol=1e-9)
    assert result.arm_command[6] == pytest.approx(closed_arm[6])


def test_spot_full_body_viser_parser_supports_optional_seer_controller() -> None:
    module = _load_full_body_viser_module()

    args = module.build_parser().parse_args(
        ["--enable-teleop", "--controller-port", "/dev/ttyUSB1", "--teleop-scale", "2.0"]
    )

    assert args.enable_teleop is True
    assert args.controller_port == "/dev/ttyUSB1"
    assert args.teleop_scale == pytest.approx(2.0)


def test_spot_full_body_ik_initializes_regular_viser_backend() -> None:
    backend = SpotFullBodyIK(_resolve_spot_ik_urdf(), config=SpotFullBodyIKConfig())

    assert backend.body_frame == "body"
    assert backend.tool_frame in {"arm0_link_wr1", "arm_link_wr1"}
    assert backend.foot_frames in {
        ("fl_foot", "fr_foot", "hl_foot", "hr_foot"),
        (
            "front_left_foot",
            "front_right_foot",
            "rear_left_foot",
            "rear_right_foot",
        ),
        (
            "front_left_foot_center",
            "front_right_foot_center",
            "rear_left_foot_center",
            "rear_right_foot_center",
        ),
    }
    assert backend.solver.has_contact_frames() is False
    np.testing.assert_allclose(
        backend.config.torso_pose_half_range,
        STANDARD_FULL_BODY_TORSO_POSE_HALF_RANGE,
    )
    assert backend.config.enable_collision is True


def test_spot_locomanip_collision_defaults_match_async_runtime_budget() -> None:
    ik = OptionalSpotWholeBodyIK(None, dt=0.01)

    assert ik.enable_collision is False
    assert ik.collision_max_constraints == 3
    assert ik.collision_tuning_mode == "balanced"


def test_spot_full_body_arm_torso_stage_locks_legs() -> None:
    backend = SpotFullBodyIK(_resolve_spot_ik_urdf(), config=SpotFullBodyIKConfig(max_steps=2))
    q0 = backend.q.copy()
    current = backend.current_tool_pose()
    target = embodik.Rt(
        R=np.asarray(current.rotation, dtype=float),
        t=np.asarray(current.translation, dtype=float) + np.array([0.03, 0.0, 0.0]),
    )

    result = backend.solve_arm_torso(target)

    assert result.status.name in {"SUCCESS", "INFEASIBLE", "NO_PROGRESS", "NUMERICAL_ERROR"}
    leg_names = list(backend._leg_joint_map.values())
    assert _max_joint_delta(backend.robot, q0, backend.q, leg_names) < 1e-9
    assert np.linalg.norm(backend.q - q0) > 1e-6


def test_spot_full_body_arm_torso_mode_uses_arm_and_bounded_base() -> None:
    backend = SpotFullBodyIK(
        _resolve_spot_ik_urdf(),
        config=SpotFullBodyIKConfig(
            max_steps=1,
            enable_collision=False,
            use_contact_projection=False,
            base_velocity_limit=0.1,
            base_acceleration_limit=0.1,
        ),
    )
    q0 = backend.q.copy()
    current = backend.current_tool_pose()
    target = embodik.Rt(
        R=np.asarray(current.rotation, dtype=float),
        t=np.asarray(current.translation, dtype=float) + np.array([0.04, 0.0, 0.0]),
    )

    result = backend.solve_arm_torso(target)

    assert result.status.name in {"SUCCESS", "INFEASIBLE", "NO_PROGRESS", "NUMERICAL_ERROR"}
    arm_delta = _max_joint_delta(
        backend.robot,
        q0,
        backend.q,
        list(backend._arm_joint_map.values()),
    )
    base_delta = float(np.linalg.norm(backend.q[:3] - q0[:3]))
    leg_delta = _max_joint_delta(
        backend.robot,
        q0,
        backend.q,
        list(backend._leg_joint_map.values()),
    )
    assert arm_delta > 1e-5
    assert base_delta > 1e-5
    assert base_delta <= 2e-3
    assert leg_delta < 1e-9


def test_spot_full_body_gripper_configuration_updates_only_gripper() -> None:
    backend = SpotFullBodyIK(_resolve_spot_ik_urdf(), config=SpotFullBodyIKConfig())
    if not backend._gripper_joint_name:
        pytest.skip("Spot gripper joint is unavailable in this URDF")
    q0 = backend.q.copy()
    backend.set_gripper_configuration(0.0)

    idx = int(backend.robot.get_joint_config_index(backend._gripper_joint_name))
    assert backend.q[idx] == pytest.approx(0.0)
    changed = np.flatnonzero(np.abs(backend.q - q0) > 1e-9)
    assert changed.tolist() == [idx]


def test_spot_full_body_torso_only_stage_moves_legs_and_locks_arm() -> None:
    backend = SpotFullBodyIK(_resolve_spot_ik_urdf(), config=SpotFullBodyIKConfig(max_steps=3))
    q0 = backend.q.copy()
    torso = backend.current_torso_pose()
    target = _pose_matrix(torso)
    target[0, 3] += 0.02

    result = backend.solve_torso_only(target)

    assert result.status.name in {"SUCCESS", "INFEASIBLE", "NO_PROGRESS", "NUMERICAL_ERROR"}
    arm_names = list(backend._arm_joint_map.values())
    leg_names = list(backend._leg_joint_map.values())
    assert _max_joint_delta(backend.robot, q0, backend.q, arm_names) < 1e-9
    assert _max_joint_delta(backend.robot, q0, backend.q, leg_names) > 1e-7
    assert backend.solver.has_contact_frames() is True
    assert backend.foot_anchor_error() < 1e-4


def test_spot_full_body_two_stage_reduces_tool_position_error() -> None:
    backend = SpotFullBodyIK(_resolve_spot_ik_urdf(), config=SpotFullBodyIKConfig(max_steps=2))
    q0 = backend.q.copy()
    current = backend.current_tool_pose()
    target = embodik.Rt(
        R=np.asarray(current.rotation, dtype=float),
        t=np.asarray(current.translation, dtype=float) + np.array([0.04, 0.0, 0.0]),
    )
    initial_error = float(
        np.linalg.norm(np.asarray(current.translation) - np.asarray(target.translation))
    )

    for _ in range(5):
        backend.solve_two_stage(target)
    final = backend.current_tool_pose()
    final_error = float(
        np.linalg.norm(np.asarray(final.translation) - np.asarray(target.translation))
    )

    assert final_error < initial_error
    assert (
        _max_joint_delta(backend.robot, q0, backend.q, list(backend._arm_joint_map.values())) > 1e-7
    )
    assert backend.solver.has_contact_frames() is True
    assert backend.foot_anchor_error() < 1e-4


def test_spot_full_body_two_stage_uses_two_solves_and_arm_residual_torso_target() -> None:
    backend = SpotFullBodyIK(_resolve_spot_ik_urdf(), config=SpotFullBodyIKConfig(max_steps=1))
    current = backend.current_tool_pose()
    torso_initial = backend.current_torso_pose()
    target = embodik.Rt(
        R=np.asarray(current.rotation, dtype=float),
        t=np.asarray(current.translation, dtype=float) + np.array([0.12, 0.02, 0.0]),
    )
    tool_calls: list[tuple[np.ndarray, bool]] = []
    torso_calls: list[np.ndarray] = []
    original_tool = backend._solve_tool_arm_torso_from
    original_torso = backend._solve_torso_with_legs_from

    def wrapped_tool(q: np.ndarray, target_pose, *, maintain_contacts: bool, lock_base: bool):
        tool_calls.append((np.asarray(q, dtype=float).copy(), bool(lock_base)))
        return original_tool(
            q, target_pose, maintain_contacts=maintain_contacts, lock_base=lock_base
        )

    def wrapped_torso(q: np.ndarray, torso_target_pose):
        torso_calls.append(np.asarray(torso_target_pose, dtype=float).copy())
        return original_torso(q, torso_target_pose)

    backend._solve_tool_arm_torso_from = wrapped_tool  # type: ignore[method-assign]
    backend._solve_torso_with_legs_from = wrapped_torso  # type: ignore[method-assign]

    backend.solve_two_stage(target)

    assert [lock_base for _q, lock_base in tool_calls] == [True]
    assert len(torso_calls) == 1
    assert torso_calls[0][0, 3] > _pose_matrix(torso_initial)[0, 3]
    assert torso_calls[0][2, 3] == pytest.approx(backend._torso_bias_pose[2, 3], abs=1e-4)
    np.testing.assert_allclose(torso_calls[0][:3, :3], backend._torso_bias_pose[:3, :3])


def test_spot_full_body_two_stage_restores_torso_bias_while_arm_compensates() -> None:
    backend = SpotFullBodyIK(
        _resolve_spot_ik_urdf(),
        config=SpotFullBodyIKConfig(max_steps=1, enable_collision=False),
    )
    q_drifted = backend.q.copy()
    q_drifted[0] += 0.04
    q_drifted[1] -= 0.03
    q_drifted[2] += 0.07
    q_drifted[3:7] = _rotation_to_xyzw(_rotation_from_rpy(0.12, -0.10, 0.18))
    backend.q = q_drifted
    backend.robot.update_configuration(backend.q)
    backend.reanchor_feet_and_torso()
    target = backend.current_tool_pose()

    def torso_bias_error() -> np.ndarray:
        pose = backend.current_torso_pose()
        position_error = np.asarray(pose.translation, dtype=float) - backend._torso_bias_pose[:3, 3]
        roll, pitch = _roll_pitch_from_rotation(np.asarray(pose.rotation, dtype=float))
        yaw = _yaw_from_rotation(np.asarray(pose.rotation, dtype=float))
        return np.array([*position_error, roll, pitch, yaw], dtype=float)

    initial_error = torso_bias_error()
    q_initial = backend.q.copy()

    for _ in range(30):
        backend.solve_two_stage(target)

    final_error = torso_bias_error()
    assert np.linalg.norm(final_error) < 0.65 * np.linalg.norm(initial_error)
    assert (
        _max_joint_delta(backend.robot, q_initial, backend.q, list(backend._arm_joint_map.values()))
        > 1e-3
    )
    assert backend.foot_anchor_error() < 1e-3


def test_spot_full_body_mode_restores_torso_bias_like_two_stage() -> None:
    backend = SpotFullBodyIK(
        _resolve_spot_ik_urdf(),
        config=SpotFullBodyIKConfig(max_steps=1, enable_collision=False),
    )
    q_drifted = backend.q.copy()
    q_drifted[0] += 0.04
    q_drifted[1] -= 0.03
    q_drifted[2] += 0.07
    q_drifted[3:7] = _rotation_to_xyzw(_rotation_from_rpy(0.12, -0.10, 0.18))
    backend.q = q_drifted
    backend.robot.update_configuration(backend.q)
    backend.reanchor_feet_and_torso()
    target = backend.current_tool_pose()

    def torso_bias_error() -> np.ndarray:
        pose = backend.current_torso_pose()
        position_error = np.asarray(pose.translation, dtype=float) - backend._torso_bias_pose[:3, 3]
        roll, pitch = _roll_pitch_from_rotation(np.asarray(pose.rotation, dtype=float))
        yaw = _yaw_from_rotation(np.asarray(pose.rotation, dtype=float))
        return np.array([*position_error, roll, pitch, yaw], dtype=float)

    initial_error = torso_bias_error()

    for _ in range(120):
        backend.solve_full_body(target)

    final_error = torso_bias_error()
    assert np.linalg.norm(final_error) < 0.9 * np.linalg.norm(initial_error)
    assert backend.foot_anchor_error() < 1e-3


def test_spot_full_body_methods_use_secondary_torso_bias_task() -> None:
    backend = SpotFullBodyIK(
        _resolve_spot_ik_urdf(),
        config=SpotFullBodyIKConfig(max_steps=1, enable_collision=False),
    )
    target = backend.current_tool_pose()
    captured_task_names: list[list[str]] = []

    def capture_solve(q: np.ndarray, targets: list[object], opts: object):
        captured_task_names.append([str(target.task_name) for target in targets])
        return np.asarray(q, dtype=float), types.SimpleNamespace(
            status=types.SimpleNamespace(name="SUCCESS")
        )

    backend._solve_position_step_guarded = capture_solve  # type: ignore[method-assign]

    backend._solve_tool_arm_torso_from(
        backend.q.copy(),
        target,
        maintain_contacts=False,
        lock_base=True,
    )
    backend._solve_full_body_from(backend.q.copy(), target)

    assert backend._torso_task.priority == 0
    assert backend._torso_bias_task.priority == 1
    assert captured_task_names == [
        [backend._tool_task_name, backend._torso_bias_task_name],
        [backend._tool_task_name, backend._torso_bias_task_name],
    ]


def test_spot_full_body_step_options_match_interactive_robust_defaults() -> None:
    backend = SpotFullBodyIK(_resolve_spot_ik_urdf(), config=SpotFullBodyIKConfig())

    opts = backend._step_options()

    assert opts.max_linear_speed == pytest.approx(1.8)
    assert opts.max_angular_speed == pytest.approx(2.5)
    assert opts.adaptive_dt is True
    assert opts.adaptive_dt_reference_distance == pytest.approx(0.04)
    assert opts.adaptive_dt_max_scale == pytest.approx(3.0)
    assert getattr(opts, "stall_recovery", False) is True
    if hasattr(opts, "no_progress_max_steps"):
        assert opts.no_progress_max_steps == 5
    if hasattr(opts, "no_progress_error_tolerance"):
        assert opts.no_progress_error_tolerance == pytest.approx(1e-5)
    if hasattr(opts, "no_progress_dq_norm_tolerance"):
        assert opts.no_progress_dq_norm_tolerance == pytest.approx(1e-6)


def test_packaged_spot_asset_two_stage_collision_recovery_uses_arm() -> None:
    packaged_urdf = _EXAMPLES_DIR / "assets" / "spot_description" / "urdf" / "spot_with_arm.urdf"
    backend = SpotFullBodyIK(
        packaged_urdf,
        config=SpotFullBodyIKConfig(
            enable_collision=True,
            collision_min_distance=SPOT_COLLISION_MIN_DISTANCE_M,
        ),
    )
    initial_tool = backend.current_tool_pose()
    initial_torso = backend.current_torso_pose()
    rotation = np.asarray(initial_tool.rotation, dtype=float)
    initial_tool_position = np.asarray(initial_tool.translation, dtype=float)
    torso_position = np.asarray(initial_torso.translation, dtype=float)
    toward_torso = torso_position - initial_tool_position
    toward_torso /= np.linalg.norm(toward_torso)

    inward_target = embodik.Rt(
        R=rotation,
        t=initial_tool_position + 0.45 * toward_torso,
    )
    for _ in range(30):
        backend.solve(SpotFullBodyIKMode.TWO_STAGE, inward_target, inward_target)

    folded_arm = np.array([backend.q[int(index) + 1] for index in backend._arm_velocity_indices])
    outward_target = embodik.Rt(
        R=rotation,
        t=initial_tool_position - 0.10 * toward_torso,
    )
    for _ in range(80):
        result = backend.solve(SpotFullBodyIKMode.TWO_STAGE, outward_target, outward_target)

    recovered_arm = np.array([backend.q[int(index) + 1] for index in backend._arm_velocity_indices])
    tool_error = float(
        np.linalg.norm(
            np.asarray(backend.current_tool_pose().translation, dtype=float)
            - np.asarray(outward_target.translation, dtype=float)
        )
    )

    assert result[0].status.name == "SUCCESS"
    assert result[1].status.name == "SUCCESS"
    assert np.linalg.norm(recovered_arm - folded_arm) > 0.5
    assert tool_error < 1e-3


def test_spot_collision_pairs_are_curated_for_reference_model() -> None:
    backend = SpotFullBodyIK(_resolve_spot_ik_urdf(), config=SpotFullBodyIKConfig())

    pairs = spot_collision_pairs_from_references(backend.robot)

    assert 0 < len(pairs) <= 60
    assert set(pairs).issubset(set(backend.robot.get_collision_pair_names()))
    assert ("body_1", "arm0_link_wr1_0") in pairs
    assert ("body_2", "arm0_link_fngr_0") in pairs


def test_spot_collision_constraint_configures_curated_pairs() -> None:
    backend = SpotFullBodyIK(
        _resolve_spot_ik_urdf(),
        config=SpotFullBodyIKConfig(
            enable_collision=True,
            collision_min_distance=SPOT_COLLISION_MIN_DISTANCE_M,
        ),
    )
    current = backend.current_tool_pose()
    target = embodik.Rt(
        R=np.asarray(current.rotation, dtype=float),
        t=np.asarray(current.translation, dtype=float) + np.array([0.01, 0.0, 0.0]),
    )

    result = backend.solve_full_body(target)

    assert result.status.name in {"SUCCESS", "INFEASIBLE", "NO_PROGRESS", "NUMERICAL_ERROR"}
    assert backend._collision_config_key is not None
    if hasattr(backend.solver, "sphere_broadphase_enabled"):
        assert backend.solver.sphere_broadphase_enabled() is True
    if hasattr(backend.solver, "get_collision_min_distance"):
        assert backend.solver.get_collision_min_distance() == pytest.approx(
            SPOT_COLLISION_MIN_DISTANCE_M
        )


def test_spot_collision_debug_exposes_minimum_distance_vector() -> None:
    backend = SpotFullBodyIK(_resolve_spot_ik_urdf(), config=SpotFullBodyIKConfig())
    current = backend.current_tool_pose()
    target = embodik.Rt(
        R=np.asarray(current.rotation, dtype=float),
        t=np.asarray(current.translation, dtype=float) + np.array([0.01, 0.0, 0.0]),
    )

    backend.solve_full_body(target)

    rows = []
    if hasattr(backend.solver, "get_last_collision_debug_list"):
        rows = list(backend.solver.get_last_collision_debug_list())
    if not rows and hasattr(backend.solver, "get_last_collision_debug"):
        debug = backend.solver.get_last_collision_debug()
        rows = [] if debug is None else [debug]
    if not rows and hasattr(backend.solver, "evaluate_collision_debug"):
        debug = backend.solver.evaluate_collision_debug(np.asarray(backend.q, dtype=float))
        rows = [] if debug is None else [debug]

    assert rows
    row = rows[0]
    point_a = np.asarray(row.point_a_world, dtype=float)
    point_b = np.asarray(row.point_b_world, dtype=float)
    vector = point_b - point_a
    assert np.isfinite(float(row.distance))
    assert np.isfinite(point_a).all()
    assert np.isfinite(point_b).all()
    assert np.linalg.norm(vector) == pytest.approx(abs(float(row.distance)), abs=1e-5)


def test_floating_base_posture_weights_apply_to_base_and_arm_rows() -> None:
    backend = SpotFullBodyIK(_resolve_spot_ik_urdf(), config=SpotFullBodyIKConfig())
    task = embodik.PostureTask(
        "floating_base_weight_regression",
        backend.robot,
        [2, backend._arm_velocity_indices[0]],
    )
    q_target = backend.q.copy()
    q_target[2] += 0.02
    arm_q_idx = int(backend._arm_velocity_indices[0]) + 1
    q_target[arm_q_idx] += 0.03

    task.set_target_configuration(q_target)
    task.set_controlled_joint_weights(np.array([10.0, 7.0], dtype=float))
    task.update(backend.robot)

    error = task.get_error()
    jacobian = task.get_jacobian()
    np.testing.assert_allclose(error, np.array([0.2, 0.21]), atol=1e-10)
    assert jacobian[0, 2] == pytest.approx(10.0)
    assert jacobian[1, backend._arm_velocity_indices[0]] == pytest.approx(7.0)


def test_spot_full_body_posture_bias_is_not_retargeted_to_current_state() -> None:
    backend = SpotFullBodyIK(_resolve_spot_ik_urdf(), config=SpotFullBodyIKConfig())
    q_current = backend.q.copy()
    q_current[2] -= 0.02
    q_current[int(backend._arm_velocity_indices[0]) + 1] += 0.03
    backend.robot.update_configuration(q_current)

    backend._set_posture_bias_scope(stage1=True)
    backend._update_posture_target()
    backend._posture_task.update(backend.robot)
    error = backend._posture_task.get_error()

    assert np.linalg.norm(error) > 0.0
    assert abs(float(error[2])) > 1e-3


def test_spot_full_body_biases_remain_at_initial_configuration_after_reanchor_and_arm_buttons() -> (
    None
):
    backend = SpotFullBodyIK(_resolve_spot_ik_urdf(), config=SpotFullBodyIKConfig())
    initial_bias = backend._posture_bias_q.copy()
    initial_torso_bias = backend._torso_bias_pose.copy()

    backend.set_arm_configuration(DEFAULT_ARM_COMMAND)
    backend.reanchor_feet_and_torso()
    backend.reset()

    np.testing.assert_allclose(backend._posture_bias_q, initial_bias)
    np.testing.assert_allclose(backend._torso_bias_pose, initial_torso_bias)


def test_spot_ik_urdf_resolver_uses_explicit_env_and_local_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "explicit_spot.urdf"
    env_urdf = tmp_path / "env_spot.urdf"
    explicit.write_text("<robot name='spot_explicit'/>")
    env_urdf.write_text("<robot name='spot_env'/>")

    monkeypatch.setenv(SPOT_URDF_ENV_VAR, str(env_urdf))

    assert resolve_spot_ik_urdf(explicit) == explicit
    assert resolve_spot_ik_urdf(None, include_local_fallbacks=False) == env_urdf
    monkeypatch.delenv(SPOT_URDF_ENV_VAR)
    assert resolve_spot_ik_urdf(None, include_local_fallbacks=False) is None


def test_spot_ik_urdf_resolver_uses_robot_descriptions_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import robot_descriptions

    urdf = tmp_path / "spot_whole_body.urdf"
    urdf.write_text("<robot name='spot'/>")
    module = types.SimpleNamespace(URDF_PATH=str(urdf))
    description = types.SimpleNamespace(has_urdf=True, robot="Spot")

    monkeypatch.delenv(SPOT_URDF_ENV_VAR, raising=False)
    monkeypatch.setitem(sys.modules, "robot_descriptions.spot_description", module)
    monkeypatch.setitem(robot_descriptions.DESCRIPTIONS, "spot_description", description)

    assert resolve_spot_ik_urdf(None, include_local_fallbacks=False) == urdf


def test_spot_whole_body_ik_zero_error_target_returns_policy_commands() -> None:
    ik = OptionalSpotWholeBodyIK(_resolve_spot_ik_urdf(), dt=0.01)
    observation = _policy_observation()
    q0 = ik._sync_configuration(observation, DEFAULT_ARM_COMMAND)
    target_pose = ik.robot.get_frame_pose(ik.tool_frame)

    result = ik.solve_command(observation, DEFAULT_ARM_COMMAND, target_pose=target_pose)

    assert result.success
    assert result.arm_command is not None
    assert result.body_command is not None
    assert result.desired_pose_command is not None
    assert result.arm_gravity_torque is not None
    assert np.isfinite(result.arm_command).all()
    assert np.isfinite(result.body_command).all()
    assert np.isfinite(result.desired_pose_command).all()
    assert np.isfinite(result.arm_gravity_torque).all()
    assert result.arm_gravity_torque.shape == (6,)
    np.testing.assert_allclose(result.arm_command, DEFAULT_ARM_COMMAND, atol=1e-6)
    np.testing.assert_allclose(
        result.body_command,
        [0.0, 0.0, DEFAULT_STAND_BASE_HEIGHT],
        atol=1e-6,
    )
    np.testing.assert_allclose(result.desired_pose_command, [0.0, 0.0, 0.0], atol=1e-6)

    q_after = np.asarray(ik.robot.get_current_configuration(), dtype=float)
    np.testing.assert_allclose(q_after, q0, atol=1e-6)


def test_spot_whole_body_ik_uses_commanded_torso_state_for_integration() -> None:
    ik = OptionalSpotWholeBodyIK(_resolve_spot_ik_urdf(), dt=0.01)
    observation = _policy_observation()
    observation["base_pose"] = observation["base_pose"].copy()
    observation["base_pose"][2] = DEFAULT_STAND_BASE_HEIGHT - 0.04
    ik._sync_configuration(
        observation,
        DEFAULT_ARM_COMMAND,
        base_pose_override=_commanded_base_pose(),
    )
    target_pose = ik.robot.get_frame_pose(ik.tool_frame)

    result = ik.solve_command(
        observation,
        DEFAULT_ARM_COMMAND,
        target_pose=target_pose,
        body_command=DEFAULT_BODY_ROLL_PITCH_HEIGHT,
        desired_pose_command=np.zeros(3),
    )

    assert result.success
    assert result.body_command is not None
    np.testing.assert_allclose(result.body_command, DEFAULT_BODY_ROLL_PITCH_HEIGHT, atol=1e-6)


def test_spot_whole_body_ik_uses_commanded_arm_state_for_integration() -> None:
    ik = OptionalSpotWholeBodyIK(_resolve_spot_ik_urdf(), dt=0.01)
    observation = _policy_observation()
    observation["arm_state"] = np.concatenate([DEFAULT_ARM_COMMAND[:6], np.zeros(6, dtype=float)])
    observation["gripper_state"] = np.array([DEFAULT_ARM_COMMAND[6], 0.0], dtype=float)
    ik._sync_configuration(
        observation,
        INITIAL_ARM_COMMAND,
        base_pose_override=_commanded_base_pose(),
    )
    target_pose = ik.robot.get_frame_pose(ik.tool_frame)

    result = ik.solve_command(
        observation,
        INITIAL_ARM_COMMAND,
        target_pose=target_pose,
        body_command=DEFAULT_BODY_ROLL_PITCH_HEIGHT,
        desired_pose_command=np.zeros(3),
    )

    assert result.success
    assert result.arm_command is not None
    assert result.body_command is not None
    assert result.desired_pose_command is not None
    np.testing.assert_allclose(result.arm_command[:6], INITIAL_ARM_COMMAND[:6], atol=2e-3)
    np.testing.assert_allclose(result.body_command, DEFAULT_BODY_ROLL_PITCH_HEIGHT, atol=1e-6)
    np.testing.assert_allclose(result.desired_pose_command, np.zeros(3), atol=1e-6)


def test_spot_whole_body_ik_repeated_solves_can_walk_base_pose_toward_target() -> None:
    ik = OptionalSpotWholeBodyIK(_resolve_spot_ik_urdf(), dt=0.01)
    observation = _policy_observation()
    observation["base_pose"] = _commanded_base_pose()
    ik._sync_configuration(observation, INITIAL_ARM_COMMAND)
    current_tool = ik.robot.get_frame_pose(ik.tool_frame)
    target_pose = embodik.Rt(
        R=np.asarray(current_tool.rotation, dtype=float),
        t=np.asarray(current_tool.translation, dtype=float) + np.array([0.04, 0.0, 0.0]),
    )

    arm = INITIAL_ARM_COMMAND.copy()
    body = DEFAULT_BODY_ROLL_PITCH_HEIGHT.copy()
    desired_pose = np.zeros(3, dtype=float)
    initial_error = float("inf")
    final_error = float("inf")
    for i in range(30):
        result = ik.solve_command(
            observation,
            arm,
            target_pose=target_pose,
            body_command=body,
            desired_pose_command=desired_pose,
        )
        assert result.arm_command is not None
        assert result.body_command is not None
        assert result.desired_pose_command is not None
        arm = result.arm_command
        body = result.body_command
        desired_pose = result.desired_pose_command
        observation["arm_state"] = np.concatenate([arm[:6], np.zeros(6, dtype=float)])
        observation["gripper_state"] = np.array([arm[6], 0.0], dtype=float)
        observation["base_pose"] = _commanded_base_pose(body, desired_pose)
        ik._sync_configuration(observation, arm)
        current_position = np.asarray(ik.robot.get_frame_pose(ik.tool_frame).translation)
        final_error = np.linalg.norm(np.asarray(target_pose.translation) - current_position)
        if i == 0:
            initial_error = final_error

    assert final_error < initial_error


def test_spot_whole_body_ik_locomanip_tracking_harness_records_performance() -> None:
    ik = OptionalSpotWholeBodyIK(_resolve_spot_ik_urdf(), dt=0.01)
    observation = _policy_observation()
    ik._sync_configuration(
        observation,
        INITIAL_ARM_COMMAND,
        base_pose_override=_commanded_base_pose(),
    )
    current_tool = ik.robot.get_frame_pose(ik.tool_frame)
    target_pose = embodik.Rt(
        R=np.asarray(current_tool.rotation, dtype=float),
        t=np.asarray(current_tool.translation, dtype=float) + np.array([0.08, 0.02, 0.0]),
    )

    arm = INITIAL_ARM_COMMAND.copy()
    body = DEFAULT_BODY_ROLL_PITCH_HEIGHT.copy()
    desired_pose = np.zeros(3, dtype=float)
    solve_times = []
    initial_error = None
    final_error = float("inf")
    for i in range(60):
        result = ik.solve_command(
            observation,
            arm,
            target_pose=target_pose,
            body_command=body,
            desired_pose_command=desired_pose,
        )
        assert result.success
        assert result.arm_command is not None
        assert result.body_command is not None
        assert result.desired_pose_command is not None
        assert result.status_name in {"SUCCESS", "INFEASIBLE", "NO_PROGRESS", "NUMERICAL_ERROR"}
        assert result.solve_time_ms > 0.0
        solve_times.append(result.solve_time_ms)
        arm = result.arm_command
        body = result.body_command
        desired_pose = result.desired_pose_command
        observation["arm_state"] = np.concatenate([arm[:6], np.zeros(6, dtype=float)])
        observation["gripper_state"] = np.array([arm[6], 0.0], dtype=float)
        observation["base_pose"] = _commanded_base_pose(body, desired_pose)
        ik._sync_configuration(observation, arm)
        current_position = np.asarray(ik.robot.get_frame_pose(ik.tool_frame).translation)
        final_error = float(np.linalg.norm(np.asarray(target_pose.translation) - current_position))
        if i == 0:
            initial_error = final_error

    assert initial_error is not None
    assert final_error < 2e-3
    assert final_error < 0.1 * initial_error
    assert float(np.mean(solve_times)) < 5.0
    assert float(np.percentile(solve_times, 95)) < 10.0


def test_spot_whole_body_ik_forward_target_uses_base_assist_and_locks_legs() -> None:
    ik = OptionalSpotWholeBodyIK(_resolve_spot_ik_urdf(), dt=0.01)
    observation = _policy_observation()
    ik._sync_configuration(observation, INITIAL_ARM_COMMAND)
    current_tool = ik.robot.get_frame_pose(ik.tool_frame)
    target_pose = embodik.Rt(
        R=np.asarray(current_tool.rotation, dtype=float),
        t=np.asarray(current_tool.translation, dtype=float)
        + np.array([0.04, 0.0, 0.0], dtype=float),
    )

    result = ik.solve_command(observation, INITIAL_ARM_COMMAND, target_pose=target_pose)

    assert result.success
    assert result.arm_command is not None
    assert result.body_command is not None
    assert result.desired_pose_command is not None
    assert result.desired_pose_command[0] > 0.015
    assert result.desired_pose_command[0] < 0.04
    np.testing.assert_allclose(result.desired_pose_command[1:], np.zeros(2), atol=1e-6)
    assert _max_arm_command_delta(result.arm_command, INITIAL_ARM_COMMAND) > 1e-5
    assert np.isfinite(result.arm_command).all()
    assert ik._task.solve_mode == embodik.TaskSolveMode.SCALE_ELASTIC
    assert ik._step_opts.adaptive_dt is True
    assert ik._step_opts.adaptive_dt_reference_distance == pytest.approx(0.04)
    assert ik._step_opts.adaptive_dt_max_scale == pytest.approx(3.0)
    assert ik._step_opts.max_linear_speed == pytest.approx(0.0)
    assert ik._step_opts.max_angular_speed == pytest.approx(0.0)
    np.testing.assert_allclose(
        ik._step_opts.torso_constraint.velocity_limits,
        np.full(6, 0.5),
    )
    np.testing.assert_allclose(
        ik._step_opts.torso_constraint.acceleration_limits,
        np.full(6, 0.5),
    )
    np.testing.assert_allclose(
        ik._step_opts.torso_constraint.pose_upper_bounds[[3, 4]],
        np.deg2rad([15.0, 15.0]),
    )
    assert ik._step_opts.torso_constraint.pose_upper_bounds[5] == pytest.approx(np.deg2rad(60.0))
    assert HEIGHT_RANGE[0] <= result.body_command[2] <= HEIGHT_RANGE[1]
    assert list(ik._step_opts.locked_joint_indices) == []
    excluded = sorted(set(ik._step_opts.excluded_joint_indices))
    expected_excluded = sorted(
        set(ik._velocity_indices((*ik._leg_joint_map.values(), ik._gripper_joint_name)))
    )
    assert excluded == expected_excluded
    locked = sorted(set(ik._step_opts.integration_zero_velocity_indices))
    assert locked == expected_excluded

    q_after = np.asarray(ik.robot.get_current_configuration(), dtype=float)
    for mjcf_name, urdf_name in ik._leg_joint_map.items():
        expected = DEFAULT_STAND_LEG_JOINTS[tuple(ik._leg_joint_map.keys()).index(mjcf_name)]
        assert _joint_value(ik, q_after, urdf_name) == pytest.approx(expected, abs=1e-9)
    assert _joint_value(ik, q_after, ik._gripper_joint_name) == pytest.approx(
        INITIAL_ARM_COMMAND[6], abs=1e-9
    )


def test_spot_locomanip_arm_torso_ik_uses_arm_when_base_is_bounded() -> None:
    ik = OptionalSpotWholeBodyIK(_resolve_spot_ik_urdf(), dt=0.01)
    observation = _policy_observation()
    ik._step_opts.torso_constraint.velocity_limits = np.full(6, 0.1, dtype=float)
    ik._step_opts.torso_constraint.acceleration_limits = np.full(6, 0.1, dtype=float)
    ik._sync_configuration(
        observation,
        INITIAL_ARM_COMMAND,
        base_pose_override=_commanded_base_pose(),
    )
    current_tool = ik.robot.get_frame_pose(ik.tool_frame)
    target_pose = embodik.Rt(
        R=np.asarray(current_tool.rotation, dtype=float),
        t=np.asarray(current_tool.translation, dtype=float) + np.array([0.04, 0.0, 0.0]),
    )

    result = ik.solve_command(
        observation,
        INITIAL_ARM_COMMAND,
        target_pose=target_pose,
        body_command=DEFAULT_BODY_ROLL_PITCH_HEIGHT.copy(),
        desired_pose_command=np.zeros(3, dtype=float),
    )

    assert result.success
    assert result.arm_command is not None
    assert result.desired_pose_command is not None
    assert _max_arm_command_delta(result.arm_command, INITIAL_ARM_COMMAND) > 1e-5
    assert 0.0 < result.desired_pose_command[0] < 0.02
    np.testing.assert_allclose(result.desired_pose_command[1:], np.zeros(2), atol=1e-3)


def test_spot_whole_body_ik_nullspace_bias_keeps_neutral_roll_pitch_height() -> None:
    ik = OptionalSpotWholeBodyIK(_resolve_spot_ik_urdf(), dt=0.01)
    observation = _policy_observation()
    initial_body = np.array([0.02, -0.03, DEFAULT_STAND_BASE_HEIGHT + 0.02], dtype=float)
    initial_desired = np.array([0.0, 0.0, 0.4], dtype=float)
    q_initial = ik._sync_configuration(
        observation,
        INITIAL_ARM_COMMAND,
        base_pose_override=_commanded_base_pose(initial_body, initial_desired),
    )
    target_pose = ik.robot.get_frame_pose(ik.tool_frame)

    result = ik.solve_command(
        observation,
        INITIAL_ARM_COMMAND,
        target_pose=target_pose,
        body_command=initial_body,
        desired_pose_command=initial_desired,
    )
    assert result.success

    drifted_body = np.array([-0.18, 0.16, DEFAULT_STAND_BASE_HEIGHT - 0.08], dtype=float)
    drifted_desired = np.array([0.25, -0.15, -0.8], dtype=float)
    q_drifted = ik._sync_configuration(
        observation,
        INITIAL_ARM_COMMAND,
        base_pose_override=_commanded_base_pose(drifted_body, drifted_desired),
    )
    drifted_torso_pose = ik.robot.get_frame_pose(ik.body_frame)
    q_target = ik._neutral_torso_posture_target(
        q_drifted,
        np.asarray(drifted_torso_pose.rotation, dtype=float),
    )

    assert q_target[0] == pytest.approx(q_drifted[0])
    assert q_target[1] == pytest.approx(q_drifted[1])
    assert q_target[2] == pytest.approx(initial_body[2])
    target_rotation = embodik.q2r(q_target[3:7], order="xyzs")
    roll, pitch = _roll_pitch_from_rotation(target_rotation)
    yaw = _yaw_from_rotation(target_rotation)
    assert roll == pytest.approx(initial_body[0], abs=1e-6)
    assert pitch == pytest.approx(initial_body[1], abs=1e-6)
    assert yaw == pytest.approx(drifted_desired[2], abs=1e-6)
    assert ik._posture_task.weight == pytest.approx(1.0)
    assert list(ik._ik_opts.nullspace_active_joints[:6]) == [0, 1, 2, 3, 4, 5]
    np.testing.assert_allclose(
        ik._ik_opts.nullspace_joint_weights[:6],
        ik._locomanip_base_gains(),
    )
    np.testing.assert_allclose(q_initial[:2], np.zeros(2), atol=1e-12)


def test_spot_whole_body_ik_nullspace_bias_targets_initial_arm_posture() -> None:
    ik = OptionalSpotWholeBodyIK(_resolve_spot_ik_urdf(), dt=0.01)
    observation = _policy_observation()
    initial_body = DEFAULT_BODY_ROLL_PITCH_HEIGHT.copy()
    initial_desired = np.zeros(3, dtype=float)
    q_initial = ik._sync_configuration(
        observation,
        INITIAL_ARM_COMMAND,
        base_pose_override=_commanded_base_pose(initial_body, initial_desired),
    )
    target_pose = ik.robot.get_frame_pose(ik.tool_frame)

    result = ik.solve_command(
        observation,
        INITIAL_ARM_COMMAND,
        target_pose=target_pose,
        body_command=initial_body,
        desired_pose_command=initial_desired,
    )
    assert result.success
    assert ik._posture_reference_q is not None

    drifted_arm = DEFAULT_ARM_COMMAND.copy()
    q_drifted = ik._sync_configuration(
        observation,
        drifted_arm,
        base_pose_override=_commanded_base_pose(initial_body, initial_desired),
    )
    torso_pose = ik.robot.get_frame_pose(ik.body_frame)
    q_target = ik._neutral_torso_posture_target(
        q_drifted,
        np.asarray(torso_pose.rotation, dtype=float),
    )

    max_drift_correction = 0.0
    for joint_name in ik._arm_joint_map.values():
        idx = int(ik.robot.get_joint_config_index(joint_name))
        assert q_target[idx] == pytest.approx(q_initial[idx])
        max_drift_correction = max(max_drift_correction, abs(float(q_target[idx] - q_drifted[idx])))
    assert max_drift_correction > 1e-3


def test_spot_whole_body_ik_nullspace_bias_restores_stretched_arm() -> None:
    ik = OptionalSpotWholeBodyIK(_resolve_spot_ik_urdf(), dt=0.01)
    observation = _policy_observation()
    body = DEFAULT_BODY_ROLL_PITCH_HEIGHT.copy()
    desired_pose = np.zeros(3, dtype=float)
    ik._sync_configuration(
        observation,
        INITIAL_ARM_COMMAND,
        base_pose_override=_commanded_base_pose(body, desired_pose),
    )
    target_pose = ik.robot.get_frame_pose(ik.tool_frame)
    result = ik.solve_command(
        observation,
        INITIAL_ARM_COMMAND,
        target_pose=target_pose,
        body_command=body,
        desired_pose_command=desired_pose,
    )
    assert result.success

    stretched_arm = INITIAL_ARM_COMMAND.copy()
    stretched_arm[:6] = np.array([0.0, -0.05, 0.10, 0.0, -0.05, 0.0], dtype=float)
    ik._sync_configuration(
        observation,
        stretched_arm,
        base_pose_override=_commanded_base_pose(body, desired_pose),
    )
    held_target = ik.robot.get_frame_pose(ik.tool_frame)
    initial_distance = np.linalg.norm(stretched_arm[:6] - INITIAL_ARM_COMMAND[:6])

    restored = ik.solve_command(
        observation,
        stretched_arm,
        target_pose=held_target,
        body_command=body,
        desired_pose_command=desired_pose,
    )

    assert restored.success
    assert restored.arm_command is not None
    step_delta = restored.arm_command[:6] - stretched_arm[:6]
    reference_delta = INITIAL_ARM_COMMAND[:6] - stretched_arm[:6]
    assert np.linalg.norm(step_delta) > 1e-4
    assert np.dot(step_delta, reference_delta) > 0.0
    assert np.linalg.norm(restored.arm_command[:6] - INITIAL_ARM_COMMAND[:6]) < initial_distance


def test_spot_whole_body_ik_condition_number_hands_off_small_nudge_to_locomotion() -> None:
    ik = OptionalSpotWholeBodyIK(_resolve_spot_ik_urdf(), dt=0.01)
    observation = _policy_observation()
    body = DEFAULT_BODY_ROLL_PITCH_HEIGHT.copy()
    desired_pose = np.zeros(3, dtype=float)
    ik._sync_configuration(
        observation,
        INITIAL_ARM_COMMAND,
        base_pose_override=_commanded_base_pose(body, desired_pose),
    )
    current_tool = ik.robot.get_frame_pose(ik.tool_frame)
    target_pose = embodik.Rt(
        R=np.asarray(current_tool.rotation, dtype=float),
        t=np.asarray(current_tool.translation, dtype=float) + np.array([0.02, 0.0, 0.0]),
    )
    posture_indices_by_solve: list[list[int]] = []

    def high_condition_solve(q: np.ndarray, target_matrix: np.ndarray):
        posture_indices_by_solve.append(list(ik._posture_velocity_indices))
        return np.asarray(q, dtype=float), types.SimpleNamespace(
            status=types.SimpleNamespace(name="SUCCESS"),
            position_error=0.0,
            orientation_error=0.0,
            condition_number=250.0,
        )

    ik._solve_position_step_guarded = high_condition_solve  # type: ignore[method-assign]

    result = ik.solve_command(
        observation,
        INITIAL_ARM_COMMAND,
        target_pose=target_pose,
        body_command=body,
        desired_pose_command=desired_pose,
    )

    assert result.success
    assert result.condition_number == pytest.approx(250.0)
    assert "cond=250" in result.message
    assert len(posture_indices_by_solve) == 2
    assert 0 in posture_indices_by_solve[0]
    assert 0 not in posture_indices_by_solve[1]
    assert ik._condition_locomotion_active


def test_spot_whole_body_ik_no_motion_high_condition_does_not_latch_protection() -> None:
    ik = OptionalSpotWholeBodyIK(_resolve_spot_ik_urdf(), dt=0.01)
    observation = _policy_observation()
    body = DEFAULT_BODY_ROLL_PITCH_HEIGHT.copy()
    desired_pose = np.zeros(3, dtype=float)
    ik._sync_configuration(
        observation,
        INITIAL_ARM_COMMAND,
        base_pose_override=_commanded_base_pose(body, desired_pose),
    )
    target_pose = ik.robot.get_frame_pose(ik.tool_frame)
    posture_indices_by_solve: list[list[int]] = []

    def high_condition_solve(q: np.ndarray, target_matrix: np.ndarray):
        posture_indices_by_solve.append(list(ik._posture_velocity_indices))
        return np.asarray(q, dtype=float), types.SimpleNamespace(
            status=types.SimpleNamespace(name="SUCCESS"),
            position_error=0.0,
            orientation_error=0.0,
            condition_number=250.0,
        )

    ik._solve_position_step_guarded = high_condition_solve  # type: ignore[method-assign]

    first = ik.solve_command(
        observation,
        INITIAL_ARM_COMMAND,
        target_pose=target_pose,
        body_command=body,
        desired_pose_command=desired_pose,
    )
    second = ik.solve_command(
        observation,
        INITIAL_ARM_COMMAND,
        target_pose=target_pose,
        body_command=body,
        desired_pose_command=desired_pose,
    )

    assert first.success
    assert second.success
    assert len(posture_indices_by_solve) == 2
    assert all(indices[:6] == [0, 1, 2, 3, 4, 5] for indices in posture_indices_by_solve)
    assert not ik._condition_locomotion_active


def test_spot_whole_body_ik_condition_protection_biases_stretched_arm_toward_reference() -> None:
    observation = _policy_observation()
    body = DEFAULT_BODY_ROLL_PITCH_HEIGHT.copy()
    desired_pose = np.zeros(3, dtype=float)
    stretched_arm = INITIAL_ARM_COMMAND.copy()
    stretched_arm[:6] = np.array([0.0, -0.05, 0.10, 0.0, -0.05, 0.0], dtype=float)

    def solve_with_condition_state(active: bool, arm_recovery_bias: float = 3.0):
        ik = OptionalSpotWholeBodyIK(_resolve_spot_ik_urdf(), dt=0.01)
        ik.condition_arm_weight_scale = arm_recovery_bias
        ik._sync_configuration(
            observation,
            INITIAL_ARM_COMMAND,
            base_pose_override=_commanded_base_pose(body, desired_pose),
        )
        target_pose = ik.robot.get_frame_pose(ik.tool_frame)
        result = ik.solve_command(
            observation,
            INITIAL_ARM_COMMAND,
            target_pose=target_pose,
            body_command=body.copy(),
            desired_pose_command=desired_pose.copy(),
        )
        assert result.success

        ik._sync_configuration(
            observation,
            stretched_arm,
            base_pose_override=_commanded_base_pose(body, desired_pose),
        )
        current_tool = ik.robot.get_frame_pose(ik.tool_frame)
        forward_target = embodik.Rt(
            R=np.asarray(current_tool.rotation, dtype=float),
            t=np.asarray(current_tool.translation, dtype=float)
            + np.array([0.02, 0.0, 0.0], dtype=float),
        )
        ik._condition_locomotion_active = active
        return ik.solve_command(
            observation,
            stretched_arm,
            target_pose=forward_target,
            body_command=body.copy(),
            desired_pose_command=desired_pose.copy(),
        )

    arm_first = solve_with_condition_state(False)
    weak_protected = solve_with_condition_state(True, arm_recovery_bias=1.0)
    protected = solve_with_condition_state(True, arm_recovery_bias=3.0)

    assert arm_first.success
    assert weak_protected.success
    assert protected.success
    assert arm_first.arm_command is not None
    assert weak_protected.arm_command is not None
    assert protected.arm_command is not None
    assert arm_first.desired_pose_command is not None
    assert weak_protected.desired_pose_command is not None
    assert protected.desired_pose_command is not None

    arm_first_distance = np.linalg.norm(arm_first.arm_command[:6] - INITIAL_ARM_COMMAND[:6])
    weak_protected_distance = np.linalg.norm(
        weak_protected.arm_command[:6] - INITIAL_ARM_COMMAND[:6]
    )
    protected_distance = np.linalg.norm(protected.arm_command[:6] - INITIAL_ARM_COMMAND[:6])
    initial_distance = np.linalg.norm(stretched_arm[:6] - INITIAL_ARM_COMMAND[:6])
    arm_first_delta = arm_first.arm_command[:6] - stretched_arm[:6]
    protected_delta = protected.arm_command[:6] - stretched_arm[:6]
    reference_delta = INITIAL_ARM_COMMAND[:6] - stretched_arm[:6]
    protected_pose_x = abs(float(protected.desired_pose_command[0] - desired_pose[0]))
    arm_first_pose_x = abs(float(arm_first.desired_pose_command[0] - desired_pose[0]))
    weak_protected_pose_x = abs(float(weak_protected.desired_pose_command[0] - desired_pose[0]))

    assert arm_first_distance < initial_distance
    assert protected_distance < weak_protected_distance
    assert protected_distance < arm_first_distance
    assert np.dot(arm_first_delta, reference_delta) > 0.0
    assert np.dot(protected_delta, reference_delta) > np.dot(arm_first_delta, reference_delta)
    assert protected_pose_x == pytest.approx(arm_first_pose_x, abs=1e-3)
    assert protected_pose_x == pytest.approx(weak_protected_pose_x, abs=1e-3)
    assert protected.position_error <= arm_first.position_error + 1e-4


def test_spot_whole_body_ik_small_forward_target_uses_arm_after_pose_nudge() -> None:
    ik = OptionalSpotWholeBodyIK(_resolve_spot_ik_urdf(), dt=0.01)
    observation = _policy_observation()
    ik._sync_configuration(
        observation,
        INITIAL_ARM_COMMAND,
        base_pose_override=_commanded_base_pose(),
    )
    current_tool = ik.robot.get_frame_pose(ik.tool_frame)
    target_pose = embodik.Rt(
        R=np.asarray(current_tool.rotation, dtype=float),
        t=np.asarray(current_tool.translation, dtype=float)
        + np.array([0.02, 0.0, 0.0], dtype=float),
    )

    arm = INITIAL_ARM_COMMAND.copy()
    body = DEFAULT_BODY_ROLL_PITCH_HEIGHT.copy()
    desired_pose = np.zeros(3, dtype=float)
    first_x = None
    for i in range(5):
        result = ik.solve_command(
            observation,
            arm,
            target_pose=target_pose,
            body_command=body,
            desired_pose_command=desired_pose,
        )
        assert result.arm_command is not None
        assert result.body_command is not None
        assert result.desired_pose_command is not None
        arm = result.arm_command
        body = result.body_command
        desired_pose = result.desired_pose_command
        if i == 0:
            first_x = desired_pose[0]

    assert first_x is not None
    assert first_x < 0.017
    assert np.linalg.norm(arm[:6] - INITIAL_ARM_COMMAND[:6]) > 0.002


def test_spot_whole_body_ik_computes_arm_gravity_torque() -> None:
    ik = OptionalSpotWholeBodyIK(_resolve_spot_ik_urdf(), dt=0.01)
    torque = ik.compute_arm_gravity_torque(_policy_observation(), DEFAULT_ARM_COMMAND)

    assert torque is not None
    assert torque.shape == (6,)
    assert np.isfinite(torque).all()
