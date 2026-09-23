"""Small graph-capturable pose operations shared by the GPU WBC runtime."""

from __future__ import annotations

from typing import Any


def _quaternion_multiply(first: Any, second: Any) -> Any:
    import torch

    x1, y1, z1, w1 = first.unbind(-1)
    x2, y2, z2, w2 = second.unbind(-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def _orientation_error(target_wxyz: Any, current_xyzw: Any) -> tuple[Any, Any]:
    import torch

    target_xyzw = torch.cat((target_wxyz[:, 1:], target_wxyz[:, :1]), dim=-1)
    target_xyzw = target_xyzw / torch.linalg.vector_norm(target_xyzw, dim=-1, keepdim=True)
    current_xyzw = current_xyzw / torch.linalg.vector_norm(current_xyzw, dim=-1, keepdim=True)
    conjugate = torch.cat((-current_xyzw[:, :3], current_xyzw[:, 3:]), dim=-1)
    error = _quaternion_multiply(target_xyzw, conjugate)
    error = torch.where(error[:, 3:4] < 0, -error, error)
    vector_norm = torch.linalg.vector_norm(error[:, :3], dim=-1)
    angle = 2.0 * torch.atan2(vector_norm, error[:, 3].clamp_min(1e-12))
    axis = error[:, :3] / vector_norm.clamp_min(1e-12).unsqueeze(-1)
    return axis * angle.unsqueeze(-1), angle


def _clamp_norm(values: Any, maximum: float) -> Any:
    import torch

    norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    return values * torch.clamp(maximum / norm.clamp_min(1e-12), max=1.0)
