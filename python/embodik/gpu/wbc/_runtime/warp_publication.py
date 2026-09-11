"""One-launch compact publication for multi-frame GPU solver results."""

from __future__ import annotations

from functools import lru_cache

import torch
import warp as wp

from .multi_pose_solver import COMPACT_PUBLICATION_SCALARS


@lru_cache(None)
def _make_publication_kernel(configuration_dim: int, frame_count: int):
    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        q: wp.array2d(dtype=wp.float32),
        position: wp.array2d(dtype=wp.float32),
        orientation: wp.array2d(dtype=wp.float32),
        converged: wp.array(dtype=wp.bool),
        collision_distance: wp.array(dtype=wp.float32),
        collision_active: wp.array(dtype=wp.bool),
        collision_accepted: wp.array(dtype=wp.bool),
        collision_overflow: wp.array(dtype=wp.bool),
        collision_applied: wp.array(dtype=wp.bool),
        effective_dt: wp.array(dtype=wp.float32),
        com_applied: wp.array(dtype=wp.bool),
        com_feasible: wp.array(dtype=wp.bool),
        com_slack: wp.array(dtype=wp.float32),
        torso_applied: wp.array(dtype=wp.bool),
        torso_feasible: wp.array(dtype=wp.bool),
        posture_applied: wp.array(dtype=wp.bool),
        posture_primary: wp.array(dtype=wp.float32),
        posture_before: wp.array(dtype=wp.float32),
        posture_after: wp.array(dtype=wp.float32),
        secondary_applied: wp.array(dtype=wp.bool),
        secondary_primary: wp.array(dtype=wp.float32),
        secondary_before: wp.array(dtype=wp.float32),
        secondary_after: wp.array(dtype=wp.float32),
        spectral_ok: wp.array(dtype=wp.bool),
        collision_clear_state_certified: wp.array(dtype=wp.bool),
        capture_applied: wp.array(dtype=wp.bool),
        capture_feasible: wp.array(dtype=wp.bool),
        capture_slack: wp.array(dtype=wp.float32),
        zmp_applied: wp.array(dtype=wp.bool),
        zmp_feasible: wp.array(dtype=wp.bool),
        zmp_slack: wp.array(dtype=wp.float32),
        zmp_force: wp.array(dtype=wp.float32),
        momentum_applied: wp.array(dtype=wp.bool),
        output: wp.array2d(dtype=wp.float32),
    ):
        b = wp.tid()
        for index in range(configuration_dim):
            output[b, index] = q[b, index]
        offset = configuration_dim
        for index in range(frame_count):
            output[b, offset + index] = position[b, index]
            output[b, offset + frame_count + index] = orientation[b, index]
        offset = configuration_dim + 2 * frame_count
        output[b, offset + 0] = wp.float32(converged[b])
        output[b, offset + 1] = collision_distance[b]
        output[b, offset + 2] = wp.float32(collision_active[b])
        output[b, offset + 3] = wp.float32(collision_accepted[b])
        output[b, offset + 4] = wp.float32(collision_overflow[b])
        output[b, offset + 5] = wp.float32(collision_applied[b])
        output[b, offset + 6] = effective_dt[b]
        output[b, offset + 7] = wp.float32(com_applied[b])
        output[b, offset + 8] = wp.float32(com_feasible[b])
        output[b, offset + 9] = com_slack[b]
        output[b, offset + 10] = wp.float32(torso_applied[b])
        output[b, offset + 11] = wp.float32(torso_feasible[b])
        output[b, offset + 12] = wp.float32(posture_applied[b])
        output[b, offset + 13] = posture_primary[b]
        output[b, offset + 14] = posture_before[b]
        output[b, offset + 15] = posture_after[b]
        output[b, offset + 16] = wp.float32(secondary_applied[b])
        output[b, offset + 17] = secondary_primary[b]
        output[b, offset + 18] = secondary_before[b]
        output[b, offset + 19] = secondary_after[b]
        output[b, offset + 20] = wp.float32(spectral_ok[b])
        output[b, offset + 21] = wp.float32(collision_clear_state_certified[b])
        output[b, offset + 22] = wp.float32(capture_applied[b])
        output[b, offset + 23] = wp.float32(capture_feasible[b])
        output[b, offset + 24] = capture_slack[b]
        output[b, offset + 25] = wp.float32(zmp_applied[b])
        output[b, offset + 26] = wp.float32(zmp_feasible[b])
        output[b, offset + 27] = zmp_slack[b]
        output[b, offset + 28] = zmp_force[b]
        output[b, offset + 29] = wp.float32(momentum_applied[b])

    wp.set_module_options({"max_unroll": 1}, module=kernel.module)
    return kernel


class WarpCompactPublication:
    """Reusable model-shape specialization with fixed fallback scalar buffers."""

    def __init__(
        self,
        batch_capacity: int,
        configuration_dim: int,
        frame_count: int,
        *,
        device: str | torch.device = "cuda",
    ) -> None:
        self.batch_capacity = batch_capacity
        self.configuration_dim = configuration_dim
        self.frame_count = frame_count
        self.device = torch.device(device)
        width = configuration_dim + 2 * frame_count + len(COMPACT_PUBLICATION_SCALARS)
        self.output = torch.empty((batch_capacity, width), dtype=torch.float32, device=self.device)
        self._zero_float = torch.zeros(batch_capacity, dtype=torch.float32, device=self.device)
        self._one_float = torch.ones_like(self._zero_float)
        self._nan_float = torch.full_like(self._zero_float, float("nan"))
        self._false = torch.zeros(batch_capacity, dtype=torch.bool, device=self.device)
        self._true = torch.ones_like(self._false)
        self._kernel = _make_publication_kernel(configuration_dim, frame_count)
        wp.load_module(module=self._kernel.module, device=str(self.device))

    def pack(
        self,
        *,
        q,
        position,
        orientation,
        converged,
        collision_distance=None,
        collision_active=None,
        collision_accepted=None,
        collision_overflow=None,
        collision_applied=None,
        effective_dt=None,
        com_applied=None,
        com_feasible=None,
        com_slack=None,
        torso_applied=None,
        torso_feasible=None,
        posture_applied=None,
        posture_primary=None,
        posture_before=None,
        posture_after=None,
        secondary_applied=None,
        secondary_primary=None,
        secondary_before=None,
        secondary_after=None,
        spectral_ok=None,
        collision_clear_state_certified=None,
        capture_applied=None,
        capture_feasible=None,
        capture_slack=None,
        zmp_applied=None,
        zmp_feasible=None,
        zmp_slack=None,
        zmp_force=None,
        momentum_applied=None,
    ) -> torch.Tensor:
        torch_stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        warp_stream = wp.get_stream(str(self.device))
        stream = warp_stream if warp_stream.is_capturing else torch_stream
        tensors = (
            q,
            position,
            orientation,
            converged,
            self._nan_float if collision_distance is None else collision_distance,
            self._false if collision_active is None else collision_active,
            self._true if collision_accepted is None else collision_accepted,
            self._false if collision_overflow is None else collision_overflow,
            self._false if collision_applied is None else collision_applied,
            self._nan_float if effective_dt is None else effective_dt,
            self._false if com_applied is None else com_applied,
            self._true if com_feasible is None else com_feasible,
            self._nan_float if com_slack is None else com_slack,
            self._false if torso_applied is None else torso_applied,
            self._true if torso_feasible is None else torso_feasible,
            self._false if posture_applied is None else posture_applied,
            self._nan_float if posture_primary is None else posture_primary,
            self._nan_float if posture_before is None else posture_before,
            self._nan_float if posture_after is None else posture_after,
            self._false if secondary_applied is None else secondary_applied,
            self._nan_float if secondary_primary is None else secondary_primary,
            self._nan_float if secondary_before is None else secondary_before,
            self._nan_float if secondary_after is None else secondary_after,
            self._true if spectral_ok is None else spectral_ok,
            (
                self._false
                if collision_clear_state_certified is None
                else collision_clear_state_certified
            ),
            self._false if capture_applied is None else capture_applied,
            self._true if capture_feasible is None else capture_feasible,
            self._nan_float if capture_slack is None else capture_slack,
            self._false if zmp_applied is None else zmp_applied,
            self._true if zmp_feasible is None else zmp_feasible,
            self._nan_float if zmp_slack is None else zmp_slack,
            self._nan_float if zmp_force is None else zmp_force,
            self._false if momentum_applied is None else momentum_applied,
        )
        wp.launch(
            self._kernel,
            dim=q.shape[0],
            inputs=[wp.from_torch(value) for value in tensors],
            outputs=[wp.from_torch(self.output)],
            block_dim=32,
            stream=stream,
        )
        return self.output[: q.shape[0]]
