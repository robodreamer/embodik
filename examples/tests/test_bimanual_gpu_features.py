"""Headless feature wiring regressions; no CUDA or downloaded robot is required."""

import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from examples.example_helpers import common_bimanual_teleop_app as app


def test_content_cache_survives_transformed_temporary_names(tmp_path):
    payload = b'<robot name="test"><link name="body"/></robot>'
    first, second = tmp_path / "random_a.urdf", tmp_path / "random_b.urdf"
    first.write_bytes(payload)
    second.write_bytes(payload)
    cached = app._gpu_cached_urdf(first, tmp_path / "cache")
    assert cached == app._gpu_cached_urdf(second, tmp_path / "cache")
    assert cached.read_bytes() == payload
    assert cached.parent.name == hashlib.sha256(payload).hexdigest()
    second.write_bytes(payload.replace(b"body", b"arm"))
    assert cached != app._gpu_cached_urdf(second, tmp_path / "cache")


def test_cache_keeps_relative_mesh_model_at_source(tmp_path):
    source = tmp_path / "robot.urdf"
    source.write_text(
        '<robot name="r"><link name="a"><visual><geometry>'
        '<mesh filename="meshes/a.stl"/></geometry></visual></link></robot>'
    )
    cached = app._gpu_cached_urdf(source, tmp_path / "cache")
    assert cached == source.resolve()
    assert 'filename="meshes/a.stl"' in cached.read_text()
    assert app._gpu_cached_urdf(cached, tmp_path / "cache") == cached


def test_collision_pairs_use_loaded_names_and_exclusions(tmp_path, monkeypatch):
    source = tmp_path / "robot.urdf"
    source.write_text(
        '<robot name="r"><link name="a"/><link name="b"/><link name="c"/>'
        '<joint name="ab" type="fixed"><parent link="a"/><child link="b"/></joint>'
        '<joint name="bc" type="fixed"><parent link="b"/><child link="c"/></joint></robot>'
    )
    robot = SimpleNamespace(
        get_collision_pair_names=lambda: [("a_0", "b_0"), ("a_0", "c_0")]
    )
    monkeypatch.setattr(
        app, "COMMON_BIMANUAL_COLLISION_LINK_PAIRS", [("a", "b"), ("a", "c")]
    )
    assert app._generate_common_bimanual_collision_include_pairs(
        robot, source, [("b_0", "a_0")]
    ) == [("a_0", "c_0")]


def test_posture_rows_follow_source_velocity_spans(monkeypatch):
    monkeypatch.setattr(app, "_is_arm_joint", lambda name: name == "tool_joint")
    robot = SimpleNamespace(
        get_joint_velocity_index=lambda name: {
            "body_joint": 8,
            "tool_joint": 2,
            "unused": 15,
        }[name],
        get_joint_velocity_size=lambda name: {
            "body_joint": 2,
            "tool_joint": 3,
            "unused": 1,
        }[name],
    )
    rows, kinds = app._gpu_posture_rows(
        robot, ("body_joint", "tool_joint", "unused"), {"body_joint"}
    )
    assert rows == (8, 9, 2, 3, 4)
    assert app._gpu_regularizer_weights(kinds, 0.2, 0.7) == (0.2, 0.2, 0.7, 0.7, 0.7)
    with pytest.raises(ValueError, match="separate GPU tasks"):
        app._gpu_regularizer_weights(((True, True),), 3, 4)


@pytest.mark.parametrize("collision_enabled", [False, True])
def test_headless_runtime_features_and_gpu_debug(collision_enabled):
    solver = SimpleNamespace(
        frames=("right_tool", "left_tool"),
        collision_supported=True,
        configure_runtime=Mock(),
        extract_active_configuration=lambda q: np.asarray(q)[[4, 1, 7]],
    )
    opts = SimpleNamespace(
        max_steps=3,
        position_gain=7.0,
        orientation_gain=2.0,
        adaptive_dt=True,
        adaptive_dt_max_scale=4.0,
        adaptive_dt_reference_distance=0.07,
    )
    target = np.eye(4)
    target[:3, 3] = (0.1, 0.2, 0.3)
    app._configure_gpu_bimanual_features(
        solver,
        options=opts,
        collision_enabled=collision_enabled,
        collision_distance=0.03,
        posture_target=np.arange(9),
        posture_kinds=((True, False), (False, True), (False, True)),
        posture_weight=0.2,
        arm_weight=0.4,
        torso_target=target,
        torso_enabled=False,
    )
    values = solver.configure_runtime.call_args.kwargs
    assert values["collision_enabled"] is collision_enabled
    assert values["collision_min_distance_m"] == 0.03
    assert values["adaptive_dt"] is True
    assert values["adaptive_dt_max_scale"] == 4.0
    assert values["acceleration_limits_enabled"] is False
    assert values["max_joint_acceleration_rad_s2"] == 15.0
    assert values["iterations"] == 3
    assert values["frame_position_gains"] == (7.0, 7.0)
    assert values["posture_target_configuration"] == (4, 1, 7)
    assert values["posture_weights"] == (0.2, 0.4, 0.4)
    assert values["secondary_frame_weights"] == (0.0,)
    assert values["secondary_frame_target_poses_wxyz"] == (
        (0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0),
    )
    cpu = Mock()
    row = SimpleNamespace(object_a="a", object_b="b", distance=0.02)
    gpu = SimpleNamespace(collision_debug=row)
    assert app._collision_debug_rows(
        cpu, gpu, gpu_mode=True, enabled=collision_enabled, visible=True
    ) == ([row] if collision_enabled else [])
    assert (
        app._collision_debug_rows(cpu, gpu, gpu_mode=True, enabled=True, visible=False)
        == []
    )
    assert cpu.mock_calls == []


def test_torso_secondary_does_not_merge_cpu_priority_bands():
    solver = SimpleNamespace(
        frames=("r", "l"), collision_supported=False, configure_runtime=Mock()
    )
    kwargs = dict(
        options=SimpleNamespace(
            max_steps=1,
            position_gain=3,
            orientation_gain=1,
            adaptive_dt=False,
            adaptive_dt_max_scale=5,
            adaptive_dt_reference_distance=0.05,
        ),
        collision_enabled=False,
        collision_distance=0.03,
        posture_target=(),
        posture_kinds=(),
        posture_weight=0,
        arm_weight=0,
        torso_target=np.eye(4),
        torso_enabled=True,
    )
    app._configure_gpu_bimanual_features(solver, **kwargs)
    assert solver.configure_runtime.call_args.kwargs["secondary_frame_weights"] == (
        1.0,
    )
    assert "collision_min_distance_m" not in solver.configure_runtime.call_args.kwargs
    app._configure_gpu_bimanual_features(solver, **dict(kwargs, posture_weight=0.2))
    assert solver.configure_runtime.call_args.kwargs["secondary_frame_weights"] == (
        1.0,
    )


def test_bimanual_constructor_selects_tertiary_posture(monkeypatch):
    core = SimpleNamespace(configure_task_hierarchy=Mock())
    monkeypatch.setattr(
        app.GpuWbcMultiFrameSolver,
        "__init__",
        lambda self, **kwargs: setattr(self, "_solver", core),
    )
    app._BimanualGpuSolver()
    core.configure_task_hierarchy.assert_called_once_with(posture_priority=2)


@pytest.mark.parametrize(
    "active, priorities, expected",
    [
        ((True, True), (0, 0), ((0, 1), ())),
        ((True, True), (0, 1), ((0,), (1,))),
        ((True, True), (1, 0), ((1,), (0,))),
        ((False, True), (0, 0), ((1,), ())),
        ((True, False), (0, 0), ((0,), ())),
    ],
)
def test_tool_layout_preserves_held_arm_priority(active, priorities, expected):
    assert app._gpu_bimanual_task_layout(active, priorities) == expected


def test_tool_layout_rejects_missing_primary_and_unknown_band():
    with pytest.raises(ValueError, match="primary"):
        app._gpu_bimanual_task_layout((False, False), (0, 0))
    with pytest.raises(ValueError, match="priority"):
        app._gpu_bimanual_task_layout((True, True), (0, 2))


def test_posture_row_selection_matches_shared_torso_and_single_arm():
    kinds = app._gpu_bimanual_posture_kinds(
        (18, 3, 10, 7),
        ((True, False), (False, True), (False, True), (True, False)),
        torso_enabled=True,
        lift_rows=(18,),
        active_arm_rows=(10,),
    )
    assert app._gpu_regularizer_weights(kinds, 0.2, 0.7) == (0, 0, 0.7, 0.2)


def test_moving_arm_and_torso_share_secondary_gains_and_targets():
    solver = SimpleNamespace(
        frames=("held_tool",),
        collision_supported=False,
        configure_runtime=Mock(),
        extract_active_configuration=lambda q: q,
    )
    moving, torso = np.eye(4), np.eye(4)
    moving[0, 3], torso[2, 3] = 0.4, 0.9
    app._configure_gpu_bimanual_features(
        solver,
        options=SimpleNamespace(
            max_steps=2,
            position_gain=6,
            orientation_gain=3,
            adaptive_dt=False,
            adaptive_dt_max_scale=2,
            adaptive_dt_reference_distance=0.05,
        ),
        collision_enabled=False,
        collision_distance=0.03,
        posture_target=(0.2,),
        posture_kinds=((True, False),),
        posture_weight=0.4,
        arm_weight=0,
        torso_target=torso,
        torso_enabled=True,
        secondary_targets=(moving,),
    )
    values = solver.configure_runtime.call_args.kwargs
    assert values["frame_position_gains"] == (6,)
    assert values["secondary_frame_position_gains"] == (6, 6)
    assert values["secondary_frame_orientation_gains"] == (3, 3)
    assert values["secondary_frame_weights"] == (1, 1)
    assert values["secondary_frame_target_poses_wxyz"][0][0] == 0.4
    assert values["secondary_frame_target_poses_wxyz"][1][2] == 0.9
    assert values["posture_weights"] == (0.4,)


def test_gpu_bimanual_forwards_com_polygon_and_cpu_bound_settings():
    solver = SimpleNamespace(
        frames=("right", "left"),
        collision_supported=False,
        configure_runtime=Mock(),
        extract_active_configuration=lambda q: q,
    )
    polygon = np.array([[-0.2, -0.1], [0.2, -0.1], [0.2, 0.1], [-0.2, 0.1]])
    app._configure_gpu_bimanual_features(
        solver,
        options=SimpleNamespace(
            max_steps=2,
            position_gain=6,
            orientation_gain=3,
            adaptive_dt=True,
            adaptive_dt_max_scale=2,
            adaptive_dt_reference_distance=0.05,
        ),
        collision_enabled=False,
        collision_distance=0.03,
        posture_target=(),
        posture_kinds=(),
        posture_weight=0,
        arm_weight=0,
        torso_target=np.eye(4),
        torso_enabled=False,
        com_enabled=True,
        com_support_polygon=polygon,
        com_margin=0.05,
        com_vel_max=0.3,
        com_acc_max=0.2,
        com_use_acceleration_limits=False,
        com_proximity_fraction=0.0,
    )
    values = solver.configure_runtime.call_args.kwargs
    assert values["com_enabled"] is True
    np.testing.assert_array_equal(values["com_support_polygon_xy"], polygon)
    assert values["com_margin"] == 0.05
    assert values["com_vel_max"] == 0.3
    assert values["com_acc_max"] == 0.2
    assert values["com_use_acceleration_limits"] is False
    assert values["com_proximity_fraction"] == 0.0


def test_cpu_debug_still_uses_cpu_rows():
    cpu = Mock()
    cpu.get_last_collision_debug_list.return_value = ["cpu row"]
    assert app._collision_debug_rows(
        cpu, None, gpu_mode=False, enabled=True, visible=True
    ) == ["cpu row"]


@pytest.mark.parametrize(
    "policy, reason",
    [
        (app.TORSO_POLICY_DECOUPLED, "per-task excluded joint columns"),
        (app.TORSO_POLICY_LOCKED, "MIN_ERROR"),
        (app.TORSO_POLICY_AUTO, "preferred-lock candidate search"),
    ],
)
def test_missing_torso_policies_remain_fail_closed(policy, reason):
    holds = app._gpu_bimanual_policy_holds(policy, 0.5, "SCALE", False)
    assert len(holds) == 1
    assert reason in holds[0]


def test_solver_policy_holds_follow_effective_fallback_and_metric_neutrality():
    assert (
        app._gpu_bimanual_policy_holds(app.TORSO_POLICY_FREE, 0.5, "SCALE", False) == []
    )
    assert (
        app._gpu_bimanual_policy_holds(
            app.TORSO_POLICY_FREE, 0.50000001, "SCALE", False
        )
        == []
    )
    holds = app._gpu_bimanual_policy_holds(
        app.TORSO_POLICY_FREE, 0.7, "MIN_ERROR", True
    )
    assert len(holds) == 3
    assert "weighted joint metric" in holds[0]
    assert "MIN_ERROR solve policy" in holds[1]
    assert "candidate solve" in holds[2]
