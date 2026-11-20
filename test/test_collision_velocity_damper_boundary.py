"""Exercise the actual Torch damper without constructing a robot or GPU solver."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")


from embodik.gpu.wbc._runtime.multi_pose_solver import DeviceResidentMultiFramePoseSolver


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_deadband_retains_rest_buffer_and_true_recovery(device: str, dtype: Any) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = SimpleNamespace(
        collision_min_distance_m=0.05,
        collision_tolerance_m=1e-4,
        collision_repulsion_deadband_m=0.0,
        collision_recovery_scale=0.2,
        collision_max_separation_speed_nonpenetrating_m_s=0.15,
        collision_min_recovery_speed_m_s=0.05,
        collision_max_separation_speed_m_s=0.5,
        dt=0.01,
    )
    distance = torch.tensor(
        [[0.05, 0.05005, 0.052, 0.049, -0.01], [0.0502, 0.0501, 0.048, 0.04, -0.5]],
        dtype=dtype,
        device=device,
    )
    expected = torch.tensor(
        [[0.0, 0.0, -0.19, 0.022, 0.5], [-0.01, 0.0, 0.042, 0.15, 0.5]],
        dtype=dtype,
        device=device,
    )
    before = distance.clone()
    lower = DeviceResidentMultiFramePoseSolver._collision_lower_bound(
        SimpleNamespace(torch=torch, config=config), distance
    )
    assert torch.all(lower[:, :2] <= 0), {
        "device": device,
        "dtype": str(dtype),
        "distance": distance.cpu().tolist(),
        "lower": lower.cpu().tolist(),
    }
    tolerance = 5e-7 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(lower, expected, rtol=2e-6, atol=tolerance)
    torch.testing.assert_close(distance, before, rtol=0, atol=0)
    assert lower.shape == distance.shape
    assert lower.dtype == dtype and lower.device == distance.device
