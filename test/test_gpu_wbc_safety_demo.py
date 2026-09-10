from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from embodik.gpu.wbc import (
    GpuWbcFloatingMultiFrameSolver,
    GpuWbcMultiFrameSolver,
    derive_frame_active_joint_names,
    derive_frames_active_joint_names,
    gpu_wbc_collision_control_availability,
)


def test_frame_active_joint_derivation_uses_jacobian_and_model_indices() -> None:
    class Robot:
        def get_frame_jacobian(self, frame):
            assert frame == "tool"
            jacobian = np.zeros((6, 4))
            jacobian[0, 1] = 1.0
            jacobian[2, 3] = 0.5
            return jacobian

        def get_joint_names(self):
            return ("fixed_effect", "joint_b", "joint_a", "gripper")

        def get_joint_config_size(self, name):
            return 1

        def get_joint_velocity_size(self, name):
            return 1

        def get_joint_velocity_index(self, name):
            return {
                "fixed_effect": 0,
                "joint_b": 3,
                "joint_a": 1,
                "gripper": 2,
            }[name]

    assert derive_frame_active_joint_names(Robot(), "tool") == (
        "joint_a",
        "joint_b",
    )


def test_multi_frame_active_joint_derivation_returns_ordered_union() -> None:
    class Robot:
        def get_frame_jacobian(self, frame):
            jacobian = np.zeros((6, 4))
            jacobian[0, 1 if frame == "left_tool" else 3] = 1.0
            return jacobian

        def get_joint_names(self):
            return ("fixed_effect", "joint_b", "joint_a", "gripper")

        def get_joint_config_size(self, name):
            return 1

        def get_joint_velocity_size(self, name):
            return 1

        def get_joint_velocity_index(self, name):
            return {
                "fixed_effect": 0,
                "joint_b": 3,
                "joint_a": 1,
                "gripper": 2,
            }[name]

    assert derive_frames_active_joint_names(Robot(), ("right_tool", "left_tool")) == (
        "joint_a",
        "joint_b",
    )


def test_gpu_collision_controls_do_not_fall_through_to_cpu_capabilities() -> None:
    assert gpu_wbc_collision_control_availability(
        gpu_active=True,
        cpu_constraint_available=True,
        cpu_debug_available=True,
    ) == (False, False)
    assert gpu_wbc_collision_control_availability(
        gpu_active=False,
        cpu_constraint_available=True,
        cpu_debug_available=True,
    ) == (True, True)
    assert gpu_wbc_collision_control_availability(
        gpu_active=True,
        cpu_constraint_available=False,
        cpu_debug_available=False,
        gpu_constraint_available=True,
        gpu_debug_available=True,
    ) == (True, True)


@pytest.mark.parametrize(
    ("adapter_type", "joint_count"),
    (
        (GpuWbcMultiFrameSolver, 23),
        (GpuWbcFloatingMultiFrameSolver, 26),
    ),
)
def test_public_gpu_adapters_shape_joint_inputs_from_model_dimension(
    adapter_type: type, joint_count: int
) -> None:
    class ArrayTorch:
        def as_tensor(self, values, *, device):
            assert device == "cuda:0"
            return np.asarray(values)

    adapter = adapter_type.__new__(adapter_type)
    adapter.configuration_dim = joint_count
    adapter.batch_size = 1
    adapter._torch = ArrayTorch()
    adapter._solver = SimpleNamespace(device="cuda:0")

    tensor = adapter._q_tensor(np.arange(joint_count, dtype=np.float32))

    assert tensor.shape == (1, joint_count)
