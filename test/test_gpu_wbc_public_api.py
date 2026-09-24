from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embodik.gpu.wbc import (
    GPU_WBC_CAPABILITIES,
    GpuWbcFloatingMultiFrameSolver,
    GpuWbcMultiFrameSolver,
    derive_frames_active_joint_names,
    derive_frames_active_velocity_indices,
    floating_pose_model_parameters_from_embodik,
    pose_model_parameters_from_embodik,
)


@dataclass(frozen=True)
class _RuntimeConfig:
    max_joint_acceleration_rad_s2: float | None = None
    collision_query_distance_m: float = 0.1
    standalone_cuda_graph_enabled: bool = False


def test_gpu_wbc_capability_contract_is_explicit() -> None:
    assert GPU_WBC_CAPABILITIES.collision_constraints
    assert GPU_WBC_CAPABILITIES.posture_nullspace
    assert GPU_WBC_CAPABILITIES.torso_bounds
    assert GPU_WBC_CAPABILITIES.torso_staging
    assert GPU_WBC_CAPABILITIES.adaptive_dt
    assert GPU_WBC_CAPABILITIES.acceleration_limits
    assert GPU_WBC_CAPABILITIES.com_support_polygon
    assert GPU_WBC_CAPABILITIES.capture_point_constraints
    assert GPU_WBC_CAPABILITIES.velocity_zmp_constraints
    assert GPU_WBC_CAPABILITIES.centroidal_momentum_tasks
    assert GPU_WBC_CAPABILITIES.independent_world_reset
    assert GPU_WBC_CAPABILITIES.per_world_status


class _UnseenFixedRobot:
    nq = 5
    nv = 5
    is_floating_base = False

    _names = ("wrist_roll", "rail", "elbow", "camera_tilt", "shoulder")
    _indices = {"rail": 0, "shoulder": 1, "elbow": 2, "wrist_roll": 3, "camera_tilt": 4}

    def get_joint_names(self):
        return self._names

    def get_frame_names(self):
        return ("base", "inspection_tool", "camera")

    def get_collision_geometry_names(self):
        return ("base_collision", "tool_collision")

    def get_joint_config_index(self, name):
        return self._indices[name]

    def get_joint_velocity_index(self, name):
        return self._indices[name]

    def get_joint_config_size(self, _name):
        return 1

    def get_joint_velocity_size(self, _name):
        return 1

    def get_joint_limits(self):
        return np.full(5, -2.0), np.full(5, 2.0)

    def get_velocity_limits(self):
        return np.arange(1.0, 6.0)

    def get_frame_jacobian(self, frame):
        jacobian = np.zeros((6, self.nv))
        columns = (0, 1, 2, 3) if frame == "inspection_tool" else (0, 4)
        jacobian[0, list(columns)] = 1.0
        return jacobian


class _UnseenFloatingRobot:
    nq = 10
    nv = 9
    is_floating_base = True

    _names = ("free_root", "mast", "tool_slide", "sensor_pan")
    _q = {"free_root": 0, "mast": 7, "tool_slide": 8, "sensor_pan": 9}
    _v = {"free_root": 0, "mast": 6, "tool_slide": 7, "sensor_pan": 8}

    def get_joint_names(self):
        return self._names

    def get_frame_names(self):
        return ("body", "novel_tool")

    def get_collision_geometry_names(self):
        return ("body_0", "novel_tool_0")

    def get_joint_config_index(self, name):
        return self._q[name]

    def get_joint_velocity_index(self, name):
        return self._v[name]

    def get_joint_config_size(self, name):
        return 7 if name == "free_root" else 1

    def get_joint_velocity_size(self, name):
        return 6 if name == "free_root" else 1

    def get_joint_limits(self):
        return np.full(self.nq, -2.0), np.full(self.nq, 2.0)

    def get_velocity_limits(self):
        return np.full(self.nv, 3.0)

    def get_frame_jacobian(self, frame):
        assert frame == "novel_tool"
        jacobian = np.zeros((6, self.nv))
        jacobian[:, :6] = np.eye(6)
        jacobian[0, 7] = 1.0
        return jacobian


def test_unseen_fixed_model_derives_order_and_dimensions_without_family_table():
    robot = _UnseenFixedRobot()

    names = derive_frames_active_joint_names(robot, ("inspection_tool", "camera"))
    parameters = pose_model_parameters_from_embodik(
        robot,
        name="never-seen-before",
        model_hash="synthetic-fixed-hash",
        task_frames=("inspection_tool", "camera"),
        active_joint_names=names,
        default_configuration=np.zeros(robot.nq),
    )

    assert names == ("rail", "shoulder", "elbow", "wrist_roll", "camera_tilt")
    assert parameters.robot_spec.configuration_dim == 5
    assert parameters.robot_spec.velocity_dim == 5
    assert parameters.robot_spec.active_velocity_indices == (0, 1, 2, 3, 4)
    assert parameters.robot_spec.task_frames == ("inspection_tool", "camera")
    assert parameters.joint_velocity_limits == (1.0, 2.0, 3.0, 4.0, 5.0)


def test_unseen_floating_model_derives_root_and_noncontiguous_active_coordinates():
    robot = _UnseenFloatingRobot()
    active = derive_frames_active_velocity_indices(robot, ("novel_tool",))

    parameters = floating_pose_model_parameters_from_embodik(
        robot,
        name="novel-mobile-manipulator",
        model_hash="synthetic-floating-hash",
        task_frames=("novel_tool",),
        active_velocity_indices=active,
        default_configuration=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        base_velocity_limits=(0.5, 0.5, 0.5, 1.0, 1.0, 1.0),
    )

    assert active == (0, 1, 2, 3, 4, 5, 7)
    assert parameters.robot_spec.configuration_dim == 10
    assert parameters.robot_spec.velocity_dim == 9
    assert parameters.robot_spec.active_velocity_indices == active
    assert parameters.robot_spec.task_frames == ("novel_tool",)


def test_from_robot_rejects_a_missing_or_wrong_base_model():
    with pytest.raises(ValueError, match="from_robot requires robot="):
        GpuWbcMultiFrameSolver.from_robot(Path("novel.urdf"), Path("cache"))
    with pytest.raises(ValueError, match="requires a floating-base"):
        GpuWbcFloatingMultiFrameSolver.from_robot(
            Path("novel.urdf"),
            Path("cache"),
            robot=_UnseenFixedRobot(),
        )
    with pytest.raises(ValueError, match="requires a fixed-base"):
        GpuWbcMultiFrameSolver.from_robot(
            Path("novel.urdf"),
            Path("cache"),
            robot=_UnseenFloatingRobot(),
        )


def test_fixed_factory_keeps_constraint_only_joints_and_needs_no_manifest(monkeypatch):
    captured = {}

    def fake_init(self, *args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(GpuWbcMultiFrameSolver, "__init__", fake_init)
    robot = _UnseenFixedRobot()

    GpuWbcMultiFrameSolver.from_robot(
        Path("novel.urdf"),
        Path("cache"),
        robot=robot,
        robot_name="novel",
        frames=("inspection_tool",),
        default_configuration=np.zeros(robot.nq),
    )

    assert captured["args"] == (None, Path("novel.urdf"), Path("cache"))
    assert captured["kwargs"]["solver_backend"] == "warp_srinv"
    assert captured["kwargs"]["active_joint_names"] == (
        "rail",
        "shoulder",
        "elbow",
        "wrist_roll",
        "camera_tilt",
    )


def test_floating_factory_keeps_all_model_velocities_by_default(monkeypatch):
    captured = {}

    def fake_init(self, *args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(GpuWbcFloatingMultiFrameSolver, "__init__", fake_init)
    robot = _UnseenFloatingRobot()

    GpuWbcFloatingMultiFrameSolver.from_robot(
        Path("novel-floating.urdf"),
        Path("cache"),
        robot=robot,
        robot_name="novel-floating",
        frames=("novel_tool",),
        frame_task_dimensions=(6,),
        frame_position_gains=(8.0,),
        frame_orientation_gains=(4.0,),
        default_configuration=np.zeros(robot.nq),
        base_velocity_limits=(0.5,) * 6,
    )

    assert captured["args"] == (None, Path("novel-floating.urdf"), Path("cache"))
    assert captured["kwargs"]["solver_backend"] == "warp_srinv"
    assert captured["kwargs"]["active_velocity_indices"] == tuple(range(robot.nv))


def test_directional_srinv_accepts_overdetermined_model_tasks():
    torch = pytest.importorskip("torch")

    from embodik.gpu.wbc._runtime.multi_pose_solver import _directional_srinv

    jacobian = torch.randn(3, 6, 5, dtype=torch.float64)
    inverse = _directional_srinv(torch, jacobian, tolerance=0.1, damping=0.1)

    assert inverse.shape == (3, 5, 6)
    assert torch.isfinite(inverse).all()


def test_device_batch_keeps_velocity_history_on_device():
    torch = pytest.importorskip("torch")

    accepted = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    expected = SimpleNamespace(accepted_velocity=accepted)
    adapter = GpuWbcFloatingMultiFrameSolver.__new__(GpuWbcFloatingMultiFrameSolver)
    adapter._previous_velocity = torch.zeros_like(accepted)
    adapter._solver = SimpleNamespace(
        solve=lambda q, target, history, current_velocity=None: expected
    )

    result = adapter.solve_device_batch(torch.zeros((8, 4)), torch.zeros((8, 1, 7)))

    assert result is expected
    torch.testing.assert_close(adapter._previous_velocity, accepted)


def test_device_batch_routes_measured_velocity_separately_from_history():
    torch = pytest.importorskip("torch")

    calls = []
    accepted = torch.zeros((2, 3), dtype=torch.float32)
    adapter = GpuWbcFloatingMultiFrameSolver.__new__(GpuWbcFloatingMultiFrameSolver)
    adapter._previous_velocity = torch.ones_like(accepted)
    adapter._solver = SimpleNamespace(
        solve=lambda q, target, history, current: (
            calls.append((history, current)) or SimpleNamespace(accepted_velocity=accepted)
        )
    )
    measured = torch.full_like(accepted, 2.0)

    adapter.solve_device_batch(
        torch.zeros((2, 4)), torch.zeros((2, 1, 7)), current_velocity=measured
    )

    assert calls[0][0] is adapter._previous_velocity
    assert calls[0][1] is measured


@pytest.mark.parametrize("adapter_type", (GpuWbcMultiFrameSolver, GpuWbcFloatingMultiFrameSolver))
def test_device_body_pose_api_preserves_device_tensor(adapter_type):
    torch = pytest.importorskip("torch")

    q = torch.zeros((8, 4), dtype=torch.float32)
    poses = torch.zeros((8, 6, 7), dtype=torch.float32)
    kinematics = SimpleNamespace(
        body_names=("base", "shoulder", "tool"),
        evaluate_body_poses=lambda value: poses if value is q else None,
    )
    adapter = adapter_type.__new__(adapter_type)
    adapter._solver = SimpleNamespace(kinematics=kinematics)

    assert adapter.body_names == ("base", "shoulder", "tool")
    assert adapter.evaluate_body_poses_device(q) is poses


def test_runtime_routes_centroidal_controls_without_cpu_fallback():
    calls = {}

    class _Core:
        posture_task_enabled = False
        torso_constraint_enabled = False
        secondary_frame_tasks_enabled = False
        collision = None
        _momentum_priority = 2
        _momentum_excluded = np.array([False, True, False], dtype=bool)

        def configure_capture_point_constraint(self, **options):
            calls["capture"] = options

        def configure_velocity_zmp_constraint(self, **options):
            calls["zmp"] = options

        def configure_centroidal_momentum(self, **options):
            calls["momentum"] = options

    torch = pytest.importorskip("torch")
    core = _Core()
    core._momentum_excluded = torch.as_tensor(core._momentum_excluded)
    core._frame_position_gains = torch.ones(1)
    core._frame_orientation_gains = torch.ones(1)
    core.config = _RuntimeConfig()
    adapter = GpuWbcFloatingMultiFrameSolver.__new__(GpuWbcFloatingMultiFrameSolver)
    adapter._solver = core
    adapter._torch = torch
    adapter.frames = ("tool",)
    adapter.configuration_dim = 3
    adapter.active_velocity_indices = (4, 7, 9)
    adapter._last_runtime_option_signature = None
    adapter._previous_velocity = torch.zeros((1, 3))
    adapter._last_target = None
    adapter._last_target_host = None

    polygon = np.array([[-0.2, -0.1], [0.2, -0.1], [0.2, 0.1], [-0.2, 0.1]])
    adapter.configure_runtime(
        capture_point_enabled=True,
        capture_point_support_polygon_xy=polygon,
        capture_point_margin=0.05,
        capture_point_omega=3.0,
        velocity_zmp_enabled=True,
        velocity_zmp_support_polygon_xy=polygon,
        velocity_zmp_fz_min=12.0,
        centroidal_momentum_enabled=True,
        centroidal_momentum_target=(0.0,) * 6,
        centroidal_momentum_axis_mask=(True, True, False, False, False, False),
        centroidal_momentum_weight=0.02,
        centroidal_momentum_priority=2,
        centroidal_momentum_excluded_velocity_indices=(7,),
    )

    assert calls["capture"]["enabled"] is True
    np.testing.assert_array_equal(calls["capture"]["support_polygon_xy"], polygon)
    assert calls["zmp"]["fz_min"] == 12.0
    assert calls["momentum"]["axis_mask"][:2] == (True, True)
    assert calls["momentum"]["weight"] == 0.02


def test_runtime_rejects_shape_changing_momentum_priority_and_exclusions():
    torch = pytest.importorskip("torch")
    adapter = GpuWbcFloatingMultiFrameSolver.__new__(GpuWbcFloatingMultiFrameSolver)
    adapter._torch = torch
    adapter.frames = ("tool",)
    adapter.configuration_dim = 2
    adapter.active_velocity_indices = (3, 8)
    adapter._last_runtime_option_signature = None
    adapter._previous_velocity = torch.zeros((1, 2))
    adapter._last_target = None
    adapter._last_target_host = None
    adapter._solver = SimpleNamespace(
        collision=None,
        posture_task_enabled=False,
        torso_constraint_enabled=False,
        secondary_frame_tasks_enabled=False,
        _momentum_priority=1,
        _momentum_excluded=torch.tensor([False, True]),
        _frame_position_gains=torch.ones(1),
        _frame_orientation_gains=torch.ones(1),
        config=_RuntimeConfig(),
    )

    with pytest.raises(ValueError, match="priority requires rebuilding"):
        adapter.configure_runtime(centroidal_momentum_priority=2)
    adapter._last_runtime_option_signature = None
    with pytest.raises(ValueError, match="excluded velocities requires rebuilding"):
        adapter.configure_runtime(centroidal_momentum_excluded_velocity_indices=(3,))


def test_device_batch_applies_reset_and_valid_masks_without_rejecting_the_batch():
    torch = pytest.importorskip("torch")

    calls = []
    accepted = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    stored = torch.ones_like(accepted)
    adapter = GpuWbcFloatingMultiFrameSolver.__new__(GpuWbcFloatingMultiFrameSolver)
    adapter._torch = torch
    adapter.batch_size = 3
    adapter._previous_velocity = stored.clone()
    adapter._solver = SimpleNamespace(
        device=stored.device,
        solve=lambda q, target, history, current_velocity=None, **options: (
            calls.append((history.clone(), options))
            or SimpleNamespace(accepted_velocity=accepted.clone())
        ),
    )
    reset = torch.tensor([True, False, False])
    valid = torch.tensor([True, True, False])

    result = adapter.solve_device_batch(
        torch.zeros((3, 4)),
        torch.zeros((3, 1, 7)),
        reset_mask=reset,
        valid_mask=valid,
    )

    history, options = calls[0]
    assert torch.equal(history[0], torch.zeros(2))
    assert torch.equal(history[1], torch.ones(2))
    assert options["reset_mask"].tolist() == [True, False, False]
    assert options["valid_mask"].tolist() == [True, True, False]
    torch.testing.assert_close(adapter._previous_velocity[0], accepted[0])
    torch.testing.assert_close(adapter._previous_velocity[1], accepted[1])
    torch.testing.assert_close(adapter._previous_velocity[2], torch.ones(2))
    assert result.accepted_velocity is not None


def test_reset_state_can_clear_selected_worlds():
    torch = pytest.importorskip("torch")

    adapter = GpuWbcFloatingMultiFrameSolver.__new__(GpuWbcFloatingMultiFrameSolver)
    adapter._torch = torch
    adapter.batch_size = 2
    adapter._solver = SimpleNamespace(device=torch.device("cpu"))
    adapter._previous_velocity = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    adapter._last_target = object()
    adapter.reset_state(torch.tensor([True, False]))

    torch.testing.assert_close(adapter._previous_velocity[0], torch.zeros(2))
    torch.testing.assert_close(adapter._previous_velocity[1], torch.tensor([2.0, 2.0]))
    assert adapter._last_target is not None

    adapter.reset_state()
    torch.testing.assert_close(adapter._previous_velocity, torch.zeros((2, 2)))
    assert adapter._last_target is None


def test_world_status_encodes_inactive_invalid_and_hold_without_reducing_the_batch():
    torch = pytest.importorskip("torch")

    from embodik.gpu.wbc._runtime.multi_pose_solver import (
        WORLD_STATUS_HELD,
        WORLD_STATUS_INACTIVE,
        WORLD_STATUS_INVALID_INPUT,
        WORLD_STATUS_NUMERICAL_FAILURE,
        WORLD_STATUS_SUCCESS,
        _encode_world_status,
        _floating_position_limits_valid,
    )

    enabled = torch.tensor([True, True, True, False])
    input_valid = torch.tensor([True, False, True, False])
    spectral_ok = torch.tensor([True, True, False, True])
    published = torch.tensor([True, True, False, True])
    status = _encode_world_status(torch, enabled, input_valid, spectral_ok, published)

    assert status.tolist() == [
        WORLD_STATUS_SUCCESS,
        WORLD_STATUS_INVALID_INPUT,
        WORLD_STATUS_NUMERICAL_FAILURE,
        WORLD_STATUS_INACTIVE,
    ]
    assert WORLD_STATUS_HELD == 2

    q = torch.tensor([[0.0, 0.0], [3.0, 0.0], [-0.5, 0.1]])
    lower = torch.tensor([-1.0, -1.0])
    upper = torch.tensor([1.0, 1.0])
    valid = _floating_position_limits_valid(q, lower, upper, (0, 1), ())
    assert valid.tolist() == [True, False, True]


def _stub_core(torch, *, floating: bool = False):
    from embodik.gpu.wbc._runtime.multi_pose_solver import DeviceResidentMultiFramePoseSolver

    core = DeviceResidentMultiFramePoseSolver.__new__(DeviceResidentMultiFramePoseSolver)
    core.torch = torch
    core.device = torch.device("cpu")
    core.batch_size = 3
    core.configuration_dim = 7 if floating else 2
    core.velocity_dim = 6 if floating else 2
    core.frame_count = 1
    core.velocity_zmp_constraint_enabled = False
    core._true_world_mask = torch.ones(core.batch_size, dtype=torch.bool)
    core._false_world_mask = torch.zeros(core.batch_size, dtype=torch.bool)
    core.robot_spec = SimpleNamespace(
        floating_base=floating,
        active_velocity_indices=(0, 3, 5) if not floating else (0, 1, 2, 3, 4, 5),
    )
    if floating:
        core._position_configuration_indices = []
        core._position_velocity_indices = []
        core._locked_active_columns = ()
        core._joint_lower = torch.zeros(0)
        core._joint_upper = torch.zeros(0)
        core._root_quat_start = 3
    else:
        core._joint_lower = torch.tensor([-1.0, -1.0])
        core._joint_upper = torch.tensor([1.0, 1.0])
    return core


def test_world_input_valid_isolates_nan_and_limit_failures_per_world():
    torch = pytest.importorskip("torch")
    core = _stub_core(torch)
    q = torch.tensor([[0.0, 0.0], [float("nan"), 0.0], [2.0, 0.0]], dtype=torch.float32)
    target = torch.zeros((3, 1, 7), dtype=torch.float32)
    target[..., 3] = 1.0
    valid = core._world_input_valid(q, target, None, None)
    assert valid.tolist() == [True, False, False]


def test_layout_validation_does_not_host_reduce_invalid_values():
    torch = pytest.importorskip("torch")
    core = _stub_core(torch)
    q = torch.tensor([[0.0, 0.0], [float("nan"), 0.0], [0.0, 0.0]], dtype=torch.float32)
    target = torch.zeros((3, 1, 7), dtype=torch.float32)
    target[..., 3] = 1.0
    core._validate_layout(q, target, None, None, None, None)
    with pytest.raises(ValueError, match="must have shape"):
        core._validate_layout(q[:2], target, None, None, None, None)


def test_floating_world_input_valid_rejects_degenerate_root_quaternion():
    torch = pytest.importorskip("torch")
    core = _stub_core(torch, floating=True)
    q = torch.zeros((3, 7), dtype=torch.float32)
    q[:, 6] = 1.0
    q[1, 3:] = 0.0
    target = torch.zeros((3, 1, 7), dtype=torch.float32)
    target[..., 3] = 1.0
    valid = core._world_input_valid(q, target, None, None)
    assert valid.tolist() == [True, False, True]


def test_configure_posture_binds_torch_through_the_solver_instance():
    torch = pytest.importorskip("torch")
    from embodik.gpu.wbc._runtime.multi_pose_solver import DeviceResidentMultiFramePoseSolver

    core = DeviceResidentMultiFramePoseSolver.__new__(DeviceResidentMultiFramePoseSolver)
    core.torch = torch
    core.device = torch.device("cpu")
    core.posture_task_enabled = True
    core._posture_runtime_enabled = False
    core._posture_gain = 1.0
    core._posture_gain_device = torch.tensor(1.0)
    core._posture_weights = torch.ones(2)
    core._posture_jacobian = torch.zeros((2, 3), dtype=torch.float32)
    core._posture_velocity_indices = (3, 5)
    core.robot_spec = SimpleNamespace(active_velocity_indices=(1, 3, 5))
    core._graph = object()

    core.configure_posture(weights=(0.5, 0.25))

    torch.testing.assert_close(core._posture_weights, torch.tensor([0.5, 0.25]))
    torch.testing.assert_close(core._posture_jacobian[0], torch.tensor([0.0, 0.5, 0.0]))
    torch.testing.assert_close(core._posture_jacobian[1], torch.tensor([0.0, 0.0, 0.25]))
    assert core._graph is None


def test_measure_device_batch_reports_dispatch_sync_and_allocator_snapshot():
    torch = pytest.importorskip("torch")

    class _Cuda:
        def synchronize(self, _device=None):
            return None

        def memory_allocated(self, _device=None):
            return 128

        def memory_reserved(self, _device=None):
            return 256

    adapter = GpuWbcFloatingMultiFrameSolver.__new__(GpuWbcFloatingMultiFrameSolver)
    adapter._torch = SimpleNamespace(cuda=_Cuda())
    adapter._solver = SimpleNamespace(device="cuda:0")
    adapter.batch_size = 1
    adapter._previous_velocity = torch.zeros((1, 2))
    adapter.solve_device_batch = lambda *args, **kwargs: SimpleNamespace(
        accepted_velocity=None, kernel_time_ms=1.5
    )

    from embodik.gpu.wbc import GpuWbcResourceReport

    report = GpuWbcFloatingMultiFrameSolver.measure_device_batch(
        adapter, torch.zeros((1, 2)), torch.zeros((1, 1, 7))
    )
    assert isinstance(report, GpuWbcResourceReport)
    assert report.kernel_time_ms == pytest.approx(1.5)
    assert report.allocated_bytes == 128
    assert report.reserved_bytes == 256
    assert report.synchronization_ms >= 0.0
    assert report.host_dispatch_ms >= 0.0


def test_fixed_adapter_reset_and_device_batch_delegate_masks():
    torch = pytest.importorskip("torch")
    calls = []
    adapter = GpuWbcMultiFrameSolver.__new__(GpuWbcMultiFrameSolver)
    adapter._torch = torch
    adapter.batch_size = 2
    adapter._solver = SimpleNamespace(device=torch.device("cpu"))
    adapter._previous_velocity = torch.ones((2, 2))
    adapter._last_target = object()
    adapter._last_target_host = object()
    adapter._last_runtime_option_signature = "keep"
    adapter.reset_state(torch.tensor([False, True]))
    assert adapter._last_runtime_option_signature == "keep"
    torch.testing.assert_close(adapter._previous_velocity[0], torch.ones(2))
    torch.testing.assert_close(adapter._previous_velocity[1], torch.zeros(2))

    adapter._solver = SimpleNamespace(
        device=torch.device("cpu"),
        solve=lambda q, target, history, current=None, **options: (
            calls.append(options) or SimpleNamespace(accepted_velocity=torch.zeros((2, 2)))
        ),
    )
    adapter.solve_device_batch(
        torch.zeros((2, 3)),
        torch.zeros((2, 1, 7)),
        valid_mask=torch.tensor([True, False]),
    )
    assert calls[0]["valid_mask"].tolist() == [True, False]
