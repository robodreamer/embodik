"""Isolated, graph-capturable directional extended-SRINV prototype.

J+ = J.T U diag(1 / (eigenvalue + global + directional)) U.T.
This is the spectrum used by multi_pose_solver._directional_srinv, not a
truncated inverse. Local double precision Gram/Jacobi arithmetic protects small
directions; inputs/outputs are float32. Consequently parity is numerical, not
bitwise with Torch's float32 Gram determinant. Very ill-conditioned, large-scale
matrices can expose substantial differences in that determinant.

One CUDA thread per world, one launch. Shape specialization is model-independent.
Outputs are reused and overwritten on the next call; instances are not reentrant.
Construct and warm up outside capture. A graph retains input/output addresses.
Status 0 means success, 1 nonfinite/unsafe arithmetic, 2 Jacobi nonconvergence;
failed worlds return zero. Status is device-resident (never downloaded here).
"""

import math
from functools import lru_cache

import torch
import warp as wp


@lru_cache(None)
def _make_kernel(rows, columns, inverse_mode):
    mat = wp.types.matrix(shape=(rows, rows), dtype=wp.float64)
    vec = wp.types.vector(length=rows, dtype=wp.float64)

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        j: wp.array3d(dtype=wp.float32),
        rhs: wp.array2d(dtype=wp.float32),
        use_rhs: int,
        tolerance: wp.float64,
        damping: wp.float64,
        relative_rank_tolerance: wp.float64,
        sweeps: int,
        convergence: wp.float64,
        inverse: wp.array3d(dtype=wp.float32),
        undamped_inverse: wp.array3d(dtype=wp.float32),
        solution: wp.array2d(dtype=wp.float32),
        status: wp.array(dtype=wp.int32),
    ):
        b = wp.tid()
        a = mat()
        u = mat()
        bad = int(0)
        scale = wp.float64(0.0)
        for r in range(rows):
            u[r, r] = wp.float64(1.0)
            for c in range(rows):
                value = wp.float64(0.0)
                for k in range(columns):
                    x = wp.float64(j[b, r, k])
                    y = wp.float64(j[b, c, k])
                    if not wp.isfinite(x) or not wp.isfinite(y):
                        bad = 1
                        x = wp.float64(0.0)
                        y = wp.float64(0.0)
                    value += x * y
                a[r, c] = value
            scale = wp.max(scale, wp.abs(a[r, r]))

        # Pivoted LU preserves determinant sign (do not substitute a product of
        # clamped eigenvalues or introduce a different regularization policy).
        lu = a
        determinant = wp.float64(1.0)
        for p in range(rows):
            pivot = p
            largest = wp.abs(lu[p, p])
            for r in range(p + 1, rows):
                if wp.abs(lu[r, p]) > largest:
                    largest = wp.abs(lu[r, p])
                    pivot = r
            if pivot != p:
                for c in range(rows):
                    tmp = lu[p, c]
                    lu[p, c] = lu[pivot, c]
                    lu[pivot, c] = tmp
                determinant = -determinant
            d = lu[p, p]
            determinant *= d
            if d != wp.float64(0.0):
                for r in range(p + 1, rows):
                    factor = lu[r, p] / d
                    for c in range(p + 1, rows):
                        lu[r, c] -= factor * lu[p, c]

        # Cyclic symmetric Jacobi, with a relative global residual criterion.
        limit = convergence * wp.max(scale, wp.float64(1.0e-300))
        for sweep in range(sweeps):
            for p in range(rows):
                for q in range(p + 1, rows):
                    apq = a[p, q]
                    if wp.abs(apq) > limit:
                        tau = (a[q, q] - a[p, p]) / (wp.float64(2.0) * apq)
                        sign = wp.float64(1.0)
                        if tau < wp.float64(0.0):
                            sign = wp.float64(-1.0)
                        t = sign / (wp.abs(tau) + wp.sqrt(wp.float64(1.0) + tau * tau))
                        cs = wp.float64(1.0) / wp.sqrt(wp.float64(1.0) + t * t)
                        sn = t * cs
                        a[p, p] -= t * apq
                        a[q, q] += t * apq
                        a[p, q] = wp.float64(0.0)
                        a[q, p] = wp.float64(0.0)
                        for k in range(rows):
                            if k != p and k != q:
                                x = a[k, p]
                                y = a[k, q]
                                a[k, p] = cs * x - sn * y
                                a[p, k] = a[k, p]
                                a[k, q] = sn * x + cs * y
                                a[q, k] = a[k, q]
                            x = u[k, p]
                            y = u[k, q]
                            u[k, p] = cs * x - sn * y
                            u[k, q] = sn * x + cs * y
            residual = wp.float64(0.0)
            for p in range(rows):
                for q in range(p + 1, rows):
                    residual = wp.max(residual, wp.abs(a[p, q]))
            if residual <= limit:
                break
        for p in range(rows):
            for q in range(p + 1, rows):
                if wp.abs(a[p, q]) > limit and bad == 0:
                    bad = 2
        threshold = tolerance * tolerance
        global_reg = wp.float64(0.0)
        if determinant < threshold:
            ratio = determinant / threshold
            global_reg = (wp.float64(1.0) - ratio * ratio) * threshold
        if not wp.isfinite(determinant) or not wp.isfinite(global_reg):
            bad = 1

        # Reuse LU scratch for the regularized inverse Gram matrix.
        weights = vec()
        undamped_weights = vec()
        maximum_eigenvalue = wp.float64(0.0)
        for k in range(rows):
            maximum_eigenvalue = wp.max(maximum_eigenvalue, a[k, k])
        relative_cutoff = (
            relative_rank_tolerance
            * relative_rank_tolerance
            * wp.max(maximum_eigenvalue, wp.float64(0.0))
        )
        for k in range(rows):
            eigenvalue = wp.max(a[k, k], wp.float64(0.0))
            if eigenvalue > relative_cutoff:
                undamped_weights[k] = wp.float64(1.0) / eigenvalue
            else:
                undamped_weights[k] = wp.float64(0.0)
            if inverse_mode == 1:
                weights[k] = undamped_weights[k]
            else:
                directional = damping * wp.max(
                    wp.float64(1.0) - eigenvalue / threshold, wp.float64(0.0)
                )
                denominator = eigenvalue + global_reg + directional
                if denominator > wp.float64(0.0):
                    weights[k] = wp.float64(1.0) / denominator
                else:
                    bad = 1
        for r in range(rows):
            for c in range(rows):
                value = wp.float64(0.0)
                undamped_value = wp.float64(0.0)
                for k in range(rows):
                    basis = u[c, k] * u[r, k]
                    value += basis * weights[k]
                    undamped_value += basis * undamped_weights[k]
                lu[r, c] = value
                a[r, c] = undamped_value
        for v in range(columns):
            answer = wp.float64(0.0)
            for r in range(rows):
                value = wp.float64(0.0)
                undamped_value = wp.float64(0.0)
                for c in range(rows):
                    source = wp.float64(j[b, c, v])
                    value += source * lu[r, c]
                    undamped_value += source * a[r, c]
                if not wp.isfinite(wp.float32(value)):
                    bad = 1
                inverse[b, v, r] = wp.float32(value)
                undamped_inverse[b, v, r] = wp.float32(undamped_value)
                if use_rhs != 0:
                    answer += value * wp.float64(rhs[b, r])
            if not wp.isfinite(wp.float32(answer)):
                bad = 1
            solution[b, v] = wp.float32(answer)
        if bad != 0:
            for v in range(columns):
                solution[b, v] = 0.0
                for r in range(rows):
                    inverse[b, v, r] = 0.0
                    undamped_inverse[b, v, r] = 0.0
        status[b] = bad

    # Small matrices benefit greatly from scalarization. Larger specializations
    # retain loops to avoid an explosion in compiler memory and compilation time.
    wp.set_module_options({"max_unroll": 9 if rows <= 9 else 1}, module=kernel.module)
    return kernel


class WarpDirectionalSRINV:
    """Reusable specialization; ``solve(J, rhs=None)`` returns inverse or J+ rhs.

    J has [B, rows, columns], optional vector RHS [B, rows], 1 <= B <=
    batch_capacity. All tensors must be contiguous CUDA float32 on this device.
    Capacities are explicit resource limits, never robot DOF assumptions.
    Current Torch stream is used, including during CUDA graph capture.
    """

    def __init__(
        self,
        rows,
        columns,
        *,
        batch_capacity=1,
        row_capacity=64,
        column_capacity=128,
        tolerance=0.1,
        damping=0.1,
        max_sweeps=32,
        convergence=1e-12,
        inverse_mode="directional",
        relative_rank_tolerance=1e-6,
        device="cuda",
    ):
        for name, value in dict(
            rows=rows,
            columns=columns,
            batch_capacity=batch_capacity,
            row_capacity=row_capacity,
            column_capacity=column_capacity,
            max_sweeps=max_sweeps,
        ).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if rows > row_capacity or columns > column_capacity:
            raise ValueError("shape exceeds configured capacity")
        if (
            not math.isfinite(tolerance)
            or tolerance <= 0
            or not 0 < tolerance * tolerance < math.inf
        ):
            raise ValueError("tolerance must be finite and positive")
        if not math.isfinite(damping) or damping < 0:
            raise ValueError("damping must be finite and nonnegative")
        if not math.isfinite(convergence) or not 0 < convergence < 1:
            raise ValueError("convergence must be in (0, 1)")
        if inverse_mode not in {"directional", "undamped_relative"}:
            raise ValueError("inverse_mode must be directional or undamped_relative")
        if not math.isfinite(relative_rank_tolerance) or relative_rank_tolerance < 0:
            raise ValueError("relative_rank_tolerance must be finite and nonnegative")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("device must be CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.rows, self.columns, self.batch_capacity = rows, columns, batch_capacity
        self.tolerance, self.damping = tolerance, damping
        self.inverse_mode = inverse_mode
        self.relative_rank_tolerance = relative_rank_tolerance
        self.max_sweeps, self.convergence = max_sweeps, convergence
        wp.init()
        self.inverse = torch.empty(
            (batch_capacity, columns, rows), dtype=torch.float32, device=self.device
        )
        self.solution = torch.empty(
            (batch_capacity, columns), dtype=torch.float32, device=self.device
        )
        self.status = torch.empty(batch_capacity, dtype=torch.int32, device=self.device)
        self.undamped_inverse = torch.empty_like(self.inverse)
        self._dummy_rhs = torch.empty(
            (batch_capacity, rows), dtype=torch.float32, device=self.device
        )
        self._outputs = [
            wp.from_torch(x)
            for x in (
                self.inverse,
                self.undamped_inverse,
                self.solution,
                self.status,
            )
        ]
        self._kernel = _make_kernel(rows, columns, 1 if inverse_mode == "undamped_relative" else 0)
        wp.load_module(module=self._kernel.module, device=str(self.device))

    def solve(self, matrix, rhs=None):
        if not isinstance(matrix, torch.Tensor):
            raise TypeError("matrix must be a Torch tensor")
        if matrix.ndim != 3 or tuple(matrix.shape[1:]) != (self.rows, self.columns):
            raise ValueError("matrix shape must be [B, rows, columns]")
        batch = matrix.shape[0]
        if not 1 <= batch <= self.batch_capacity:
            raise ValueError("batch exceeds capacity or is empty")
        if rhs is not None and (
            not isinstance(rhs, torch.Tensor) or tuple(rhs.shape) != (batch, self.rows)
        ):
            raise ValueError("rhs shape must be [B, rows]")
        for tensor in (matrix,) if rhs is None else (matrix, rhs):
            if (
                tensor.device != self.device
                or tensor.dtype != torch.float32
                or not tensor.is_contiguous()
            ):
                raise ValueError("inputs must be contiguous CUDA float32 on the configured device")
        torch_stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        # A whole-solver parent graph wraps Torch capture in Warp's external
        # ScopedCapture. In that case Warp must receive the registered capture
        # stream object, not a second wrapper around the same CUDA pointer.
        # Standalone Torch graph capture has no registered Warp stream and uses
        # the Torch wrapper directly.
        warp_stream = wp.get_stream(str(self.device))
        stream = warp_stream if warp_stream.is_capturing else torch_stream
        wp.launch(
            self._kernel,
            dim=batch,
            inputs=[
                wp.from_torch(matrix),
                wp.from_torch(self._dummy_rhs if rhs is None else rhs),
                int(rhs is not None),
                self.tolerance,
                self.damping,
                self.relative_rank_tolerance,
                self.max_sweeps,
                self.convergence,
            ],
            outputs=self._outputs,
            block_dim=32,
            stream=stream,
        )
        return (self.inverse if rhs is None else self.solution)[:batch]
