"""Experimental graph-safe equality elimination for a primary SCALE LP.

Complete GPU QR isolates null coordinates. Finite/rank/conditioning and final
original-row/ray checks gate the reduced result. Other lanes use the full native
LP. No FK, geometry, acceleration, secondary hierarchy or publication is added.
"""

from __future__ import annotations

import torch

from .warp_nullspace import WarpNullspace
from .warp_polytope import WarpPolytopeScale


class WarpReducedPolytopeScale:
    def __init__(
        self,
        rows: int,
        columns: int,
        hard_rows: int,
        *,
        batch: int,
        device: str = "cuda",
        native_qr: bool = True,
    ):
        if rows > columns:
            raise ValueError("Reduction requires task rows <= velocity columns")
        self.full = WarpPolytopeScale(rows, columns, hard_rows, batch=batch, device=device)
        self.device = self.full.device
        self.shape = self.full.shape
        self.free = columns - rows
        self.qr = (
            WarpNullspace(rows, columns, batch=batch, device=self.device) if native_qr else None
        )
        self.fast = WarpPolytopeScale(
            1, self.free + 1, columns + hard_rows, batch=batch, device=device
        )
        kw = dict(dtype=torch.float64, device=self.device)
        self.safe_J = torch.eye(rows, columns, **kw).expand(batch, rows, columns)
        self.identity = torch.eye(rows, **kw).expand(batch, rows, rows)
        self.fast_J = torch.zeros((batch, 1, self.free + 1), **kw)
        self.fast_J[:, 0, -1] = 1.0
        self.fast_goal = torch.ones((batch, 1), **kw)
        self.zero = torch.zeros((batch, 1), **kw)
        self.one = torch.ones_like(self.zero)
        self.velocity = torch.empty((batch, columns), **kw)
        self.scales = torch.empty(batch, **kw)
        self.status = torch.empty(batch, dtype=torch.int32, device=self.device)
        self.used_fallback = torch.empty(batch, dtype=torch.bool, device=self.device)
        self.condition_bound = torch.empty(batch, **kw)

    @property
    def iterations(self):
        return self.fast.iterations + self.full.iterations

    def solve(self, J, goal, C, lower, upper, row_lower, row_upper):
        b, m, n, r = self.shape
        for name, value, shape in zip(
            ("J", "goal", "C", "lower", "upper", "row_lower", "row_upper"),
            (J, goal, C, lower, upper, row_lower, row_upper),
            ((b, m, n), (b, m), (b, r, n), (b, n), (b, n), (b, r), (b, r)),
        ):
            if value.shape != shape or value.device != self.device or value.dtype != torch.float64:
                raise ValueError(f"{name} requires{shape}, float64 on{self.device}")
        if self.qr is not None:
            w, N, regular, condition = self.qr.solve(J, goal)
        else:
            finite = torch.isfinite(J).all(dim=(-2, -1)) & torch.isfinite(goal).all(-1)
            safe_J = torch.where(finite[:, None, None], J, self.safe_J)
            safe_goal = torch.where(finite[:, None], goal, torch.zeros_like(goal))
            Q, Rfull = torch.linalg.qr(safe_J.transpose(-2, -1), mode="complete")
            R = Rfull[:, :m, :]
            diag = R.diagonal(dim1=-2, dim2=-1).abs()
            regular = (
                finite
                & torch.isfinite(R).all(dim=(-2, -1))
                & (diag.amin(-1) > 1e-12 * R.abs().amax(dim=(-2, -1)))
            )
            safe_R = torch.where(regular[:, None, None], R, self.identity)
            inv_R = torch.linalg.solve_triangular(safe_R, self.identity, upper=True)
            condition = safe_R.abs().sum(-1).amax(-1) * inv_R.abs().sum(-1).amax(-1)
            regular = regular & torch.isfinite(condition) & (condition <= 1e8)
            y = torch.linalg.solve_triangular(
                safe_R.transpose(-2, -1), safe_goal[:, :, None], upper=False
            )
            w = (Q[:, :, :m] @ y).squeeze(-1)
            N = Q[:, :, m:]
        transform = torch.cat((N, w[:, :, None]), dim=-1)
        # Every original velocity-box point maps into this conservative z box.
        # The original velocity box is ALSO enforced as transformed hard rows.
        positive, negative = N.clamp_min(0), N.clamp_max(0)
        zlo = (
            positive.transpose(-2, -1) @ lower[:, :, None]
            + negative.transpose(-2, -1) @ upper[:, :, None]
        ).squeeze(-1)
        zhi = (
            positive.transpose(-2, -1) @ upper[:, :, None]
            + negative.transpose(-2, -1) @ lower[:, :, None]
        ).squeeze(-1)
        orth_error = (N.transpose(-2, -1) @ w[:, :, None]).squeeze(-1)
        zlo = zlo - orth_error.clamp_min(0)
        zhi = zhi - orth_error.clamp_max(0)
        padding = 1e-10 * (1 + torch.maximum(zlo.abs(), zhi.abs()))
        fast_lower = torch.cat((zlo - padding, self.zero), -1)
        fast_upper = torch.cat((zhi + padding, self.one), -1)
        hard = torch.cat((transform, C @ transform), dim=-2)
        hard_lower, hard_upper = torch.cat((lower, row_lower), -1), torch.cat(
            (upper, row_upper), -1
        )
        reduced, alpha, status = self.fast.solve(
            self.fast_J,
            self.fast_goal,
            hard,
            fast_lower,
            fast_upper,
            hard_lower,
            hard_upper,
            enabled=regular,
        )
        velocity = (transform @ reduced[:, :, None]).squeeze(-1)
        image = (J @ velocity[:, :, None]).squeeze(-1)
        residual = (image - alpha[:, None] * goal).norm(dim=-1)
        hard_image = (C @ velocity[:, :, None]).squeeze(-1)
        accepted = regular & (status == 0) & torch.isfinite(velocity).all(-1)
        accepted = (
            accepted
            & torch.isfinite(residual)
            & (residual <= 1e-8 * torch.maximum(torch.ones_like(alpha), goal.norm(dim=-1)))
        )
        accepted = (
            accepted & (velocity >= lower - 1e-8).all(-1) & (velocity <= upper + 1e-8).all(-1)
        )
        accepted = (
            accepted
            & torch.isfinite(hard_image).all(-1)
            & (hard_image >= row_lower - 1e-8).all(-1)
            & (hard_image <= row_upper + 1e-8).all(-1)
        )
        full_v, full_alpha, full_status = self.full.solve(
            J, goal, C, lower, upper, row_lower, row_upper, enabled=~accepted
        )
        self.velocity.copy_(torch.where(accepted[:, None], velocity, full_v))
        self.scales.copy_(torch.where(accepted, alpha, full_alpha))
        self.status.copy_(torch.where(accepted, status, full_status))
        self.used_fallback.copy_(~accepted)
        self.condition_bound.copy_(condition)
        return self.velocity, self.scales, self.status
