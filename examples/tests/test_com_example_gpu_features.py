"""Focused regressions for Example 04's optional GPU path."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

example = importlib.import_module("04_com_constraint_example")


def run_step(gpu, **overrides):
    options = dict(
        com_enabled=False,
        collision_enabled=True,
        collision_min_distance=0.045,
        show_debug=False,
        posture_gain=0.7,
        adaptive_dt=True,
        position_gain=12.0,
        orientation_gain=8.0,
        iterations=3,
        dt=0.02,
    )
    options.update(overrides)
    return example.gpu_step(
        gpu, np.array([4.0, 5.0, 6.0, 7.0]), (2, 0), "pose", **options
    )


def test_com_enabled_configures_and_solves():
    gpu = Mock()
    gpu.solve_step.return_value = SimpleNamespace(joints=np.array([8.0, 9.0]))
    q, result = run_step(gpu, com_enabled=True)
    np.testing.assert_array_equal(q, [9, 5, 8, 7])
    assert result is gpu.solve_step.return_value
    assert gpu.configure_runtime.call_args.kwargs["com_enabled"]
    gpu.solve_step.assert_called_once()


@pytest.mark.parametrize(
    "collision,debug,expected",
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_runtime_controls_lazy_debug_and_noncontiguous_joint_merge(
    collision, debug, expected
):
    gpu = Mock(collision_supported=True)
    gpu.solve_step.return_value = SimpleNamespace(joints=np.array([8.0, 9.0]))
    q, result = run_step(gpu, collision_enabled=collision, show_debug=debug)
    np.testing.assert_array_equal(q, [9, 5, 8, 7])
    np.testing.assert_array_equal(gpu.solve_step.call_args.args[0], [6, 4])
    assert gpu.solve_step.call_args.args[1] == ("pose",)
    assert gpu.solve_step.call_args.kwargs == {"include_collision_debug": expected}
    gpu.configure_runtime.assert_called_once_with(
        dt=0.02,
        iterations=3,
        frame_position_gains=(12.0,),
        frame_orientation_gains=(8.0,),
        collision_enabled=collision,
        collision_min_distance_m=0.045,
        posture_gain=0.7,
        adaptive_dt=True,
        com_enabled=False,
        com_support_polygon_xy=example.DEFAULT_POLYGON,
        com_margin=0.0,
        com_vel_max=0.4,
        com_acc_max=0.1,
        com_use_acceleration_limits=True,
        com_proximity_fraction=0.05,
    )
    assert result is gpu.solve_step.return_value


def test_missing_collision_is_explicit_failure_and_disabled_collision_runs():
    gpu = Mock(collision_supported=False)
    with pytest.raises(RuntimeError, match="self-collision"):
        run_step(gpu)
    gpu.solve_step.assert_not_called()
    gpu.solve_step.return_value = SimpleNamespace(joints=np.array([6.0, 4.0]))
    run_step(gpu, collision_enabled=False)
    gpu.solve_step.assert_called_once()
    assert gpu.configure_runtime.call_args.kwargs["collision_min_distance_m"] is None


def test_gpu_fault_propagates_without_modifying_configuration():
    gpu = Mock(collision_supported=True)
    gpu.solve_step.side_effect = RuntimeError("CUDA fault")
    with pytest.raises(RuntimeError, match="CUDA fault"):
        run_step(gpu)


def test_collision_pairs_follow_topology_and_order_independent_exclusions(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot><joint><parent link="root"/><child link="arm"/></joint></robot>'
    )
    robot = SimpleNamespace(
        get_collision_geometries=lambda: [
            {"name": name, "parent_frame": parent}
            for name, parent in [
                ("a", "root"),
                ("b", "root"),
                ("c", "arm"),
                ("d", "tool"),
                ("e", "other"),
            ]
        ],
        get_collision_pair_names=lambda: [
            ("a", "b"),
            ("a", "c"),
            ("a", "d"),
            ("c", "e"),
        ],
    )
    assert example.gpu_collision_pairs(robot, urdf, [("d", "a")]) == (("c", "e"),)


@pytest.mark.parametrize("count", [2, 5, 11])
def test_constructor_derives_layout_from_model(monkeypatch, count):
    names = tuple(f"joint_{i}" for i in range(count))
    robot = SimpleNamespace(get_joint_config_index=lambda name: 2 * names.index(name))
    gpu_robot = SimpleNamespace(get_joint_velocity_index=lambda name: names.index(name))
    factory = Mock(return_value=Mock())
    monkeypatch.setattr(example, "derive_frame_active_joint_names", lambda *args: names)
    monkeypatch.setattr(example.embodik, "RobotModel", Mock(return_value=gpu_robot))
    monkeypatch.setattr(example, "load_robot_presets", lambda: {"custom": {}})
    monkeypatch.setattr(
        example, "gpu_collision_pairs", lambda *args, **kwargs: (("a", "b"),)
    )
    monkeypatch.setattr(example, "GpuWbcMultiFrameSolver", factory)
    args = SimpleNamespace(gpu_wbc_manifest="spec", gpu_wbc_cache_dir="cache")
    config = dict(
        robot=robot,
        key="custom",
        target_link="tcp",
        urdf_path="robot.urdf",
        default_configuration=np.arange(count * 2, dtype=float),
    )
    gpu, indices = example.create_gpu_solver(args, config)
    options = factory.call_args.kwargs
    assert indices == tuple(range(0, count * 2, 2))
    assert options["active_joint_names"] == names
    assert options["posture_velocity_indices"] == tuple(range(count))
    assert options["posture_target_configuration"] == indices
    assert options["posture_weights"] == (1.0,) * count
    assert options["solver_backend"] == "torch_srinv"
    gpu.configure_runtime.assert_called_once_with(collision_enabled=False)
    assert gpu is factory.return_value


def test_auto_exclusions_use_two_topological_edges(tmp_path):
    urdf = tmp_path / "chain.urdf"
    urdf.write_text(
        '<robot><joint><parent link="base"/><child link="middle"/></joint>'
        '<joint><parent link="middle"/><child link="wrist"/></joint>'
        '<joint type="fixed"><parent link="wrist"/><child link="tip"/></joint></robot>'
    )
    robot = SimpleNamespace(
        get_collision_geometries=lambda: [
            {"name": name, "parent_frame": name} for name in ("base", "tip", "remote")
        ],
        get_collision_pair_names=lambda: [("base", "tip"), ("base", "remote")],
    )
    assert len(example.gpu_collision_pairs(robot, urdf)) == 2
    assert example.gpu_collision_pairs(robot, urdf, auto=True) == (("base", "remote"),)
