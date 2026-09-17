"""Single-launch graph-safe cyclic row projection for model-derived shapes."""

from __future__ import annotations

from functools import lru_cache
import math

import torch
import warp as wp

from .gpu_constraints import CyclicProjectionResult


@lru_cache(None)
def _make_projection_kernel(
    rows: int,
    columns: int,
    iterations: int,
    fixed_point_early_exit: bool,
):
    vector = wp.types.vector(length=columns, dtype=wp.float64)

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        velocity: wp.array2d(dtype=wp.float32),
        row_matrix: wp.array3d(dtype=wp.float64),
        row_lower: wp.array2d(dtype=wp.float64),
        row_upper: wp.array2d(dtype=wp.float64),
        box_lower: wp.array2d(dtype=wp.float64),
        box_upper: wp.array2d(dtype=wp.float64),
        active: wp.array2d(dtype=wp.bool),
        tolerance: wp.float64,
        corrected: wp.array2d(dtype=wp.float64),
        feasible: wp.array(dtype=wp.bool),
        max_row_violation: wp.array(dtype=wp.float64),
        max_velocity_violation: wp.array(dtype=wp.float64),
        lower_residual: wp.array2d(dtype=wp.float64),
        upper_residual: wp.array2d(dtype=wp.float64),
    ):
        batch = wp.tid()
        work = vector()
        for column in range(columns):
            work[column] = wp.clamp(
                wp.float64(velocity[batch, column]),
                box_lower[batch, column],
                box_upper[batch, column],
            )

        for iteration in range(iterations):
            for row in range(rows):
                value = wp.float64(0.0)
                denominator = wp.float64(0.0)
                for column in range(columns):
                    coefficient = row_matrix[batch, row, column]
                    value += coefficient * work[column]
                    denominator += coefficient * coefficient
                correction = wp.float64(0.0)
                if active[batch, row] and denominator > wp.float64(1.0e-30):
                    if value < row_lower[batch, row]:
                        correction = (row_lower[batch, row] - value) / denominator
                    elif value > row_upper[batch, row]:
                        correction = (row_upper[batch, row] - value) / denominator
                for column in range(columns):
                    work[column] = wp.clamp(
                        work[column] + correction * row_matrix[batch, row, column],
                        box_lower[batch, column],
                        box_upper[batch, column],
                    )

            # A complete cyclic pass is a fixed point exactly when every
            # active row is inside its interval.  In that case every later
            # pass computes a zero correction and cannot change ``work``.
            # Check only after pass one: failures run the original remaining
            # budget without paying another certificate scan per pass.
            if fixed_point_early_exit and iteration == 0 and iterations > 1:
                fixed_point = bool(True)
                for row in range(rows):
                    value = wp.float64(0.0)
                    for column in range(columns):
                        value += row_matrix[batch, row, column] * work[column]
                    if active[batch, row] and (
                        value < row_lower[batch, row]
                        or value > row_upper[batch, row]
                    ):
                        fixed_point = False
                if fixed_point:
                    break

        row_max = wp.float64(0.0)
        for row in range(rows):
            value = wp.float64(0.0)
            for column in range(columns):
                value += row_matrix[batch, row, column] * work[column]
            low = wp.float64(0.0)
            high = wp.float64(0.0)
            if active[batch, row]:
                low = wp.max(row_lower[batch, row] - value, wp.float64(0.0))
                high = wp.max(value - row_upper[batch, row], wp.float64(0.0))
            lower_residual[batch, row] = low
            upper_residual[batch, row] = high
            row_max = wp.max(row_max, wp.max(low, high))

        box_max = wp.float64(0.0)
        for column in range(columns):
            corrected[batch, column] = work[column]
            low = wp.max(box_lower[batch, column] - work[column], wp.float64(0.0))
            high = wp.max(work[column] - box_upper[batch, column], wp.float64(0.0))
            box_max = wp.max(box_max, wp.max(low, high))
        max_row_violation[batch] = row_max
        max_velocity_violation[batch] = box_max
        feasible[batch] = row_max <= tolerance and box_max <= tolerance

    wp.set_module_options({"max_unroll": 1}, module=kernel.module)
    return kernel


class WarpCyclicRowProjection:
    """Reusable exact-policy projection keyed only by matrix shape and budget."""

    def __init__(
        self,
        rows: int,
        columns: int,
        *,
        iterations: int = 16,
        batch_capacity: int = 1,
        feasibility_tolerance: float = 1.0e-8,
        fixed_point_early_exit: bool = True,
        device: str = "cuda",
    ) -> None:
        for name, value in {
            "rows": rows,
            "columns": columns,
            "iterations": iterations,
            "batch_capacity": batch_capacity,
        }.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(feasibility_tolerance) or feasibility_tolerance < 0.0:
            raise ValueError("feasibility_tolerance must be finite and nonnegative")
        if type(fixed_point_early_exit) is not bool:
            raise ValueError("fixed_point_early_exit must be a boolean")
        self.rows = rows
        self.columns = columns
        self.iterations = iterations
        self.batch_capacity = batch_capacity
        self.feasibility_tolerance = feasibility_tolerance
        self.fixed_point_early_exit = fixed_point_early_exit
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("device must be CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        wp.init()
        self.velocity = torch.empty(
            (batch_capacity, columns), dtype=torch.float64, device=self.device
        )
        self.feasible = torch.empty(
            batch_capacity, dtype=torch.bool, device=self.device
        )
        self.max_row_violation = torch.empty(
            batch_capacity, dtype=torch.float64, device=self.device
        )
        self.max_velocity_violation = torch.empty_like(self.max_row_violation)
        self.lower_row_residual = torch.empty(
            (batch_capacity, rows), dtype=torch.float64, device=self.device
        )
        self.upper_row_residual = torch.empty_like(self.lower_row_residual)
        self._outputs = tuple(
            wp.from_torch(value)
            for value in (
                self.velocity,
                self.feasible,
                self.max_row_violation,
                self.max_velocity_violation,
                self.lower_row_residual,
                self.upper_row_residual,
            )
        )
        self._kernel = _make_projection_kernel(
            rows,
            columns,
            iterations,
            fixed_point_early_exit,
        )
        wp.load_module(module=self._kernel.module, device=str(self.device))

    def solve(
        self,
        velocity: torch.Tensor,
        rows: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        velocity_lower: torch.Tensor,
        velocity_upper: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> CyclicProjectionResult:
        batch = velocity.shape[0]
        expected = (
            ("velocity", velocity, (batch, self.columns), torch.float32),
            ("rows", rows, (batch, self.rows, self.columns), torch.float64),
            ("lower", lower, (batch, self.rows), torch.float64),
            ("upper", upper, (batch, self.rows), torch.float64),
            ("velocity_lower", velocity_lower, (batch, self.columns), torch.float64),
            ("velocity_upper", velocity_upper, (batch, self.columns), torch.float64),
            ("active_mask", active_mask, (batch, self.rows), torch.bool),
        )
        if not 1 <= batch <= self.batch_capacity:
            raise ValueError("batch exceeds projection capacity or is empty")
        for name, tensor, shape, dtype in expected:
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != shape
                or tensor.device != self.device
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must match the configured contiguous CUDA layout "
                    f"{shape} {dtype} on {self.device}"
                )
        torch_stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        warp_stream = wp.get_stream(str(self.device))
        stream = warp_stream if warp_stream.is_capturing else torch_stream
        wp.launch(
            self._kernel,
            dim=batch,
            inputs=[
                wp.from_torch(velocity),
                wp.from_torch(rows),
                wp.from_torch(lower),
                wp.from_torch(upper),
                wp.from_torch(velocity_lower),
                wp.from_torch(velocity_upper),
                wp.from_torch(active_mask),
                self.feasibility_tolerance,
            ],
            outputs=self._outputs,
            block_dim=32,
            stream=stream,
        )
        return CyclicProjectionResult(
            velocity=self.velocity[:batch],
            feasible=self.feasible[:batch],
            max_row_violation=self.max_row_violation[:batch],
            max_velocity_violation=self.max_velocity_violation[:batch],
            lower_row_residual=self.lower_row_residual[:batch],
            upper_row_residual=self.upper_row_residual[:batch],
        )
