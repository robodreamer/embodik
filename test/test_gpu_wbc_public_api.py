from __future__ import annotations

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
def test_gpu_wbc_capability_contract_is_explicit() -> None:
    assert GPU_WBC_CAPABILITIES.collision_constraints
    assert GPU_WBC_CAPABILITIES.posture_nullspace
    assert GPU_WBC_CAPABILITIES.torso_bounds
    assert GPU_WBC_CAPABILITIES.torso_staging
    assert GPU_WBC_CAPABILITIES.adaptive_dt
    assert GPU_WBC_CAPABILITIES.acceleration_limits
    assert GPU_WBC_CAPABILITIES.com_support_polygon
    assert not GPU_WBC_CAPABILITIES.capture_point_constraints
    assert not GPU_WBC_CAPABILITIES.velocity_zmp_constraints
    assert not GPU_WBC_CAPABILITIES.centroidal_momentum_tasks


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
    adapter._solver = SimpleNamespace(solve=lambda q, target, history: expected)

    result = adapter.solve_device_batch(
        torch.zeros((8, 4)), torch.zeros((8, 1, 7))
    )

    assert result is expected
    torch.testing.assert_close(adapter._previous_velocity, accepted)
