"""Model mapping and runtime regression checks for public Examples 01 and 03."""

import importlib
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

basic = importlib.import_module("01_basic_ik_simple")


@pytest.mark.parametrize("count", [4, 9])
def test_constructor_uses_source_indices_and_filtered_geometry(monkeypatch, count):
    names = tuple(f"joint_{i}" for i in range(count))
    indices = tuple(range(1, 2 * count, 2))
    velocities = tuple(range(2, count + 2))
    robot = SimpleNamespace(get_joint_config_index=lambda name: indices[names.index(name)])
    gpu_robot = SimpleNamespace(
        get_joint_velocity_index=lambda name: velocities[names.index(name)],
        get_collision_pair_names=lambda: [("a", "b"), ("c", "d")],
    )
    monkeypatch.setattr(basic, "derive_frame_active_joint_names", lambda *_: names)
    model = Mock(return_value=gpu_robot)
    monkeypatch.setattr(basic.embodik, "RobotModel", model)
    constructor = Mock()
    monkeypatch.setattr(basic, "GpuWbcMultiFrameSolver", constructor)
    args = SimpleNamespace(gpu_wbc_manifest="manifest", gpu_wbc_cache_dir="cache")
    q = np.arange(count * 2, dtype=float)
    solver, mapping, supported = basic.create_gpu_solver(
        args,
        robot,
        "robot.urdf",
        "arbitrary",
        "tool",
        q,
        exclusions=[("b", "a")],
        collision_enabled=False,
    )
    options = constructor.call_args.kwargs
    assert mapping == indices
    assert options["active_joint_names"] == names
    assert options["posture_velocity_indices"] == velocities
    assert options["posture_target_configuration"] == tuple(q[list(indices)])
    assert options["default_configuration"] == tuple(q[list(indices)])
    assert options["collision_pairs"] == (("c", "d"),)
    assert options["solver_backend"] == "torch_srinv"
    assert supported
    solver.configure_runtime.assert_called_once_with(collision_enabled=False)
    model.assert_called_once_with("robot.urdf", actuated_joint_names=names, floating_base=False)


def make_controls():
    gui = SimpleNamespace(
        add_folder=lambda *_: nullcontext(),
        add_checkbox=lambda _, **kw: SimpleNamespace(value=kw["initial_value"]),
        add_slider=lambda _, **kw: SimpleNamespace(value=kw["initial_value"]),
    )
    scene = SimpleNamespace(add_line_segments=Mock(return_value=SimpleNamespace(visible=True)))
    return basic.GpuControls(gui, scene, True, True, 0.03), scene


@pytest.mark.parametrize(
    "collision,debug", [(False, False), (False, True), (True, False), (True, True)]
)
def test_runtime_features_and_lazy_debug(collision, debug):
    controls, scene = make_controls()
    controls.collision.value = collision
    controls.debug.value = debug
    controls.distance.value = 0.045
    controls.posture.value = 0.6
    controls.pos.value = 12.0
    controls.rot.value = 7.0
    controls.adaptive.value = True
    result = SimpleNamespace(collision_debug=None)
    solver = SimpleNamespace(configure_runtime=Mock(), solve_step=Mock(return_value=result))
    q, target = object(), object()
    assert controls.solve(solver, q, target) is result
    solver.solve_step.assert_called_once_with(
        q, (target,), include_collision_debug=collision and debug
    )
    solver.configure_runtime.assert_called_once_with(
        collision_enabled=collision,
        collision_min_distance_m=0.045,
        posture_gain=0.6,
        adaptive_dt=True,
        frame_position_gains=(12.0,),
        frame_orientation_gains=(7.0,),
    )
    scene.add_line_segments.assert_not_called()


def test_debug_reuses_geometry_and_hides_stale_pair():
    controls, scene = make_controls()
    controls.debug.value = True
    debug = SimpleNamespace(point_a_world=(0, 0, 0), point_b_world=(1, 0, 0))
    controls.show_debug(debug)
    controls.show_debug(debug)
    assert scene.add_line_segments.call_count == 1
    controls.show_debug()
    assert not controls.line.visible


def test_runtime_failure_propagates_without_cpu_fallback():
    controls, _ = make_controls()
    solver = SimpleNamespace(
        configure_runtime=Mock(), solve_step=Mock(side_effect=RuntimeError("GPU"))
    )
    with pytest.raises(RuntimeError, match="GPU"):
        controls.solve(solver, object(), object())


def test_geometry_free_robot_does_not_configure_collision_distance():
    controls, _ = make_controls()
    controls.collision_supported = False
    solver = SimpleNamespace(
        configure_runtime=Mock(),
        solve_step=Mock(return_value=SimpleNamespace(collision_debug=None)),
    )
    controls.solve(solver, object(), object())
    assert solver.configure_runtime.call_args.kwargs["collision_enabled"] is False
    assert solver.configure_runtime.call_args.kwargs["collision_min_distance_m"] is None


def test_teleop_imports_shared_generalized_path():
    teleop = importlib.import_module("03_teleop_ik")
    assert teleop._basic_gpu.create_gpu_solver is basic.create_gpu_solver
