"""Graph-safe ordered-priority velocity updates built on Warp small inverses."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from .warp_directional_srinv import WarpDirectionalSRINV


class WarpPriorityResult(NamedTuple):
    """One lower-priority update plus device-resident diagnostics."""

    velocity: torch.Tensor
    primary_residual_increase: torch.Tensor
    secondary_residual_before: torch.Tensor
    secondary_residual_after: torch.Tensor
    delta_norm: torch.Tensor
    applied: torch.Tensor
    spectral_ok: torch.Tensor


class WarpSecondaryPriority:
    """Reusable model-derived specialization for one ordered priority band.

    The protected-task projector uses the same relative-rank undamped inverse as
    the Torch reference. The projected lower-priority solve uses the exact
    directional extended-SRINV spectrum. No robot name or family is involved;
    the specialization is keyed only by matrix dimensions and capacities.
    """

    def __init__(
        self,
        primary_rows: int,
        secondary_rows: int,
        velocity_dim: int,
        *,
        batch_capacity: int = 1,
        srinv_tolerance: float = 0.1,
        srinv_damping: float = 0.1,
        relative_rank_tolerance: float = 1.0e-6,
        device: str = "cuda",
    ) -> None:
        dimensions = (primary_rows, secondary_rows, velocity_dim, batch_capacity)
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("priority dimensions and batch capacity must be positive")
        if not math.isfinite(relative_rank_tolerance) or relative_rank_tolerance < 0.0:
            raise ValueError("relative rank tolerance must be finite and nonnegative")
        self.primary_rows = primary_rows
        self.secondary_rows = secondary_rows
        self.velocity_dim = velocity_dim
        self.batch_capacity = batch_capacity
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("device must be CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._primary_inverse = WarpDirectionalSRINV(
            primary_rows,
            velocity_dim,
            batch_capacity=batch_capacity,
            inverse_mode="undamped_relative",
            relative_rank_tolerance=relative_rank_tolerance,
            device=device,
        )
        self._secondary_inverse = WarpDirectionalSRINV(
            secondary_rows,
            velocity_dim,
            batch_capacity=batch_capacity,
            tolerance=srinv_tolerance,
            damping=srinv_damping,
            device=device,
        )
        self._identity = torch.eye(velocity_dim, dtype=torch.float32, device=self.device).expand(
            batch_capacity, -1, -1
        )

    def solve(
        self,
        primary_velocity: torch.Tensor,
        primary_jacobian: torch.Tensor,
        secondary_jacobian: torch.Tensor,
        secondary_goal: torch.Tensor,
        lower_velocity: torch.Tensor,
        upper_velocity: torch.Tensor,
        locked_velocity_mask: torch.Tensor | None = None,
        protected_inverse: torch.Tensor | None = None,
        protected_spectral_ok: torch.Tensor | None = None,
    ) -> WarpPriorityResult:
        batch = primary_velocity.shape[0]
        expected = (
            (primary_velocity, (batch, self.velocity_dim)),
            (primary_jacobian, (batch, self.primary_rows, self.velocity_dim)),
            (secondary_jacobian, (batch, self.secondary_rows, self.velocity_dim)),
            (secondary_goal, (batch, self.secondary_rows)),
            (lower_velocity, (batch, self.velocity_dim)),
            (upper_velocity, (batch, self.velocity_dim)),
        )
        if not 1 <= batch <= self.batch_capacity:
            raise ValueError("batch exceeds priority specialization capacity")
        for tensor, shape in expected:
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != shape
                or tensor.device != self.device
                or tensor.dtype != torch.float32
            ):
                raise ValueError("priority inputs must match the configured CUDA float32 shapes")
        if locked_velocity_mask is None:
            free = torch.ones(self.velocity_dim, dtype=torch.float32, device=self.device)
        else:
            if (
                tuple(locked_velocity_mask.shape) != (self.velocity_dim,)
                or locked_velocity_mask.device != self.device
                or locked_velocity_mask.dtype != torch.bool
            ):
                raise ValueError("locked velocity mask has the wrong layout")
            free = (~locked_velocity_mask).to(torch.float32)

        protected = (primary_jacobian * free[None, None, :]).contiguous()
        if protected_inverse is None:
            protected_inverse = self._primary_inverse.solve(protected)
            primary_spectral_ok = self._primary_inverse.status[:batch] == 0
        else:
            if (
                tuple(protected_inverse.shape) != (batch, self.velocity_dim, self.primary_rows)
                or protected_inverse.device != self.device
                or protected_inverse.dtype != torch.float32
                or locked_velocity_mask is not None
            ):
                raise ValueError("precomputed protected inverse has an incompatible layout")
            primary_spectral_ok = (
                torch.ones(batch, dtype=torch.bool, device=self.device)
                if protected_spectral_ok is None
                else protected_spectral_ok
            )
            if (
                tuple(primary_spectral_ok.shape) != (batch,)
                or primary_spectral_ok.device != self.device
                or primary_spectral_ok.dtype != torch.bool
            ):
                raise ValueError("precomputed protected status has the wrong layout")
        projector = self._identity[:batch] - protected_inverse @ protected
        projector = projector * free[None, :, None] * free[None, None, :]
        residual = secondary_goal - (secondary_jacobian @ primary_velocity.unsqueeze(-1)).squeeze(
            -1
        )
        projected_jacobian = (secondary_jacobian @ projector).contiguous()
        tangent_delta = self._secondary_inverse.solve(projected_jacobian, residual.contiguous())
        delta = (projector @ tangent_delta.unsqueeze(-1)).squeeze(-1)

        moving = delta != 0.0
        available = torch.where(
            delta > 0.0,
            upper_velocity - primary_velocity,
            primary_velocity - lower_velocity,
        )
        ratios = torch.where(
            moving,
            available / torch.clamp(delta.abs(), min=torch.finfo(delta.dtype).tiny),
            torch.full_like(delta, torch.inf),
        )
        scale = torch.clamp(ratios.amin(dim=-1), min=0.0, max=1.0)
        scaled_delta = scale[:, None] * delta
        spectral_ok = primary_spectral_ok & (self._secondary_inverse.status[:batch] == 0)
        velocity = torch.where(
            spectral_ok[:, None],
            primary_velocity + scaled_delta,
            primary_velocity,
        )
        primary_change = (primary_jacobian @ scaled_delta.unsqueeze(-1)).squeeze(-1)
        secondary_after = secondary_goal - (secondary_jacobian @ velocity.unsqueeze(-1)).squeeze(-1)
        delta_norm = torch.linalg.vector_norm(scaled_delta, dim=-1)
        return WarpPriorityResult(
            velocity=velocity,
            primary_residual_increase=torch.linalg.vector_norm(primary_change, dim=-1),
            secondary_residual_before=torch.linalg.vector_norm(residual, dim=-1),
            secondary_residual_after=torch.linalg.vector_norm(secondary_after, dim=-1),
            delta_norm=delta_norm,
            applied=spectral_ok & (scale > 0.0) & (delta_norm > 0.0),
            spectral_ok=spectral_ok,
        )
