"""Graph-capturable ordered priority updates using direct cuSOLVER Jacobi SVD."""

from __future__ import annotations

import torch

from .cusolver_gesvdj import CuSolverDirectionalSRINV
from .warp_priority import WarpPriorityResult


class CuSolverSecondaryPriority:
    """One model-derived priority specialization for the B1 solver path."""

    def __init__(
        self,
        primary_rows: int,
        secondary_rows: int,
        velocity_dim: int,
        *,
        srinv_tolerance: float = 0.1,
        srinv_damping: float = 0.1,
        relative_rank_tolerance: float = 1.0e-6,
        specialized_outputs_enabled: bool = False,
        fused_status_enabled: bool = False,
        compiled_postprocess_enabled: bool = False,
        device: str = "cuda",
    ) -> None:
        self.primary_rows = primary_rows
        self.secondary_rows = secondary_rows
        self.velocity_dim = velocity_dim
        self.device = torch.device(device)
        self._primary_uses_gram = primary_rows > velocity_dim
        primary_factor_rows = velocity_dim if self._primary_uses_gram else primary_rows
        self._primary = CuSolverDirectionalSRINV(
            primary_factor_rows,
            velocity_dim,
            # For tall protected stacks, eigenvalues of J.T@J are squared
            # singular values. Squaring the relative cutoff preserves the
            # same rank decision as an SVD of J itself.
            relative_rank_tolerance=(
                relative_rank_tolerance * relative_rank_tolerance
                if self._primary_uses_gram
                else relative_rank_tolerance
            ),
            output_mode=(
                "action_undamped" if specialized_outputs_enabled else "full"
            ),
            fused_status_enabled=fused_status_enabled,
            profile_label="priority_primary",
            device=device,
        )
        self._secondary = CuSolverDirectionalSRINV(
            secondary_rows,
            velocity_dim,
            tolerance=srinv_tolerance,
            damping=srinv_damping,
            relative_rank_tolerance=relative_rank_tolerance,
            output_mode="action" if specialized_outputs_enabled else "full",
            fused_status_enabled=fused_status_enabled,
            profile_label="priority_secondary",
            device=device,
        )
        self._primary_rhs = torch.zeros(
            (1, primary_factor_rows), dtype=torch.float32, device=self.device
        )
        self._identity = torch.eye(
            velocity_dim, dtype=torch.float32, device=self.device
        )[None]
        compile_options = {"triton.cudagraphs": False}
        self._compiled_scale_delta = (
            torch.compile(
                self._scale_delta_eager,
                fullgraph=True,
                dynamic=False,
                options=compile_options,
            )
            if compiled_postprocess_enabled
            else None
        )
        self._compiled_diagnostics = (
            torch.compile(
                self._diagnostics_eager,
                fullgraph=True,
                dynamic=False,
                options=compile_options,
            )
            if compiled_postprocess_enabled
            else None
        )

    @staticmethod
    def _scale_delta_eager(
        delta: torch.Tensor,
        primary_velocity: torch.Tensor,
        lower_velocity: torch.Tensor,
        upper_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return scale[:, None] * delta, scale

    @staticmethod
    def _diagnostics_eager(
        primary_jacobian: torch.Tensor,
        secondary_jacobian: torch.Tensor,
        secondary_goal: torch.Tensor,
        velocity: torch.Tensor,
        residual: torch.Tensor,
        scaled_delta: torch.Tensor,
        spectral_ok: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        primary_change = (primary_jacobian @ scaled_delta.unsqueeze(-1)).squeeze(-1)
        secondary_after = secondary_goal - (
            secondary_jacobian @ velocity.unsqueeze(-1)
        ).squeeze(-1)
        delta_norm = torch.linalg.vector_norm(scaled_delta, dim=-1)
        return (
            torch.linalg.vector_norm(primary_change, dim=-1),
            torch.linalg.vector_norm(residual, dim=-1),
            torch.linalg.vector_norm(secondary_after, dim=-1),
            delta_norm,
            spectral_ok & (scale > 0.0) & (delta_norm > 0.0),
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
        if primary_velocity.shape != (1, self.velocity_dim):
            raise ValueError("cuSOLVER priority currently requires a B1 velocity")
        free = (
            torch.ones(self.velocity_dim, dtype=torch.float32, device=self.device)
            if locked_velocity_mask is None
            else (~locked_velocity_mask).to(torch.float32)
        )
        protected = (primary_jacobian * free[None, None, :]).contiguous()
        if protected_inverse is None:
            if self._primary_uses_gram:
                protected_gram = (
                    protected.transpose(-2, -1) @ protected
                ).contiguous()
                primary_outputs = self._primary.solve(
                    protected_gram, self._primary_rhs
                )
                protected_inverse = (
                    primary_outputs[2] @ protected.transpose(-2, -1)
                )
            else:
                primary_outputs = self._primary.solve(protected, self._primary_rhs)
                protected_inverse = primary_outputs[2]
            primary_ok = primary_outputs[3] == 0
        else:
            primary_ok = (
                torch.ones(1, dtype=torch.bool, device=self.device)
                if protected_spectral_ok is None
                else protected_spectral_ok
            )
        projector = self._identity - protected_inverse @ protected
        projector = projector * free[None, :, None] * free[None, None, :]
        residual = secondary_goal - (
            secondary_jacobian @ primary_velocity.unsqueeze(-1)
        ).squeeze(-1)
        projected = (secondary_jacobian @ projector).contiguous()
        secondary_outputs = self._secondary.solve(projected, residual.contiguous())
        tangent_delta = secondary_outputs[0]
        delta = (projector @ tangent_delta.unsqueeze(-1)).squeeze(-1)

        scale_delta = self._compiled_scale_delta or self._scale_delta_eager
        scaled_delta, scale = scale_delta(
            delta, primary_velocity, lower_velocity, upper_velocity
        )
        spectral_ok = primary_ok & (secondary_outputs[3] == 0)
        velocity = torch.where(
            spectral_ok[:, None],
            primary_velocity + scaled_delta,
            primary_velocity,
        )
        diagnostics = self._compiled_diagnostics or self._diagnostics_eager
        (
            primary_residual_increase,
            secondary_residual_before,
            secondary_residual_after,
            delta_norm,
            applied,
        ) = diagnostics(
            primary_jacobian,
            secondary_jacobian,
            secondary_goal,
            velocity,
            residual,
            scaled_delta,
            spectral_ok,
            scale,
        )
        return WarpPriorityResult(
            velocity=velocity,
            primary_residual_increase=primary_residual_increase,
            secondary_residual_before=secondary_residual_before,
            secondary_residual_after=secondary_residual_after,
            delta_norm=delta_norm,
            applied=applied,
            spectral_ok=spectral_ok,
        )
