"""Experimental graph-safe per-world complete QR without per-matrix host calls."""

from functools import lru_cache

import torch
import warp as wp


@lru_cache(None)
def make_nullspace_kernel(rows: int, columns: int):
    vector = wp.types.vector(length=columns, dtype=wp.float64)
    small = wp.types.vector(length=rows, dtype=wp.float64)
    basis_type = wp.types.matrix(shape=(columns, columns), dtype=wp.float64)
    triangular = wp.types.matrix(shape=(rows, rows), dtype=wp.float64)

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        J: wp.array3d(dtype=wp.float64),
        goal: wp.array2d(dtype=wp.float64),
        w: wp.array2d(dtype=wp.float64),
        N: wp.array3d(dtype=wp.float64),
        regular: wp.array(dtype=wp.bool),
        condition: wp.array(dtype=wp.float64),
    ):
        lane = wp.tid()
        regular[lane] = False
        condition[lane] = wp.float64(1e300)
        maximum = wp.float64(0.0)
        for col in range(columns):
            w[lane, col] = wp.float64(0.0)
            for k in range(columns - rows):
                N[lane, col, k] = wp.float64(0.0)
        for row in range(rows):
            norm = wp.float64(0.0)
            if not wp.isfinite(goal[lane, row]):
                return
            for col in range(columns):
                value = J[lane, row, col]
                if not wp.isfinite(value):
                    return
                norm += value * value
            if not wp.isfinite(norm):
                return
            maximum = wp.max(maximum, wp.sqrt(norm))
        if maximum <= wp.float64(1e-30):
            return
        Q, R = basis_type(), triangular()
        for row in range(rows):
            candidate = vector()
            for col in range(columns):
                candidate[col] = J[lane, row, col]
            coefficients = small()
            for repeat in range(2):
                for previous in range(row):
                    projection = wp.float64(0.0)
                    for col in range(columns):
                        projection += Q[previous, col] * candidate[col]
                    coefficients[previous] += projection
                    for col in range(columns):
                        candidate[col] -= projection * Q[previous, col]
            norm = wp.sqrt(wp.dot(candidate, candidate))
            if not wp.isfinite(norm) or norm <= wp.float64(1e-12) * maximum:
                return
            for previous in range(row):
                R[previous, row] = coefficients[previous]
            R[row, row] = norm
            inverse_norm = wp.float64(1.0) / norm
            for col in range(columns):
                Q[row, col] = candidate[col] * inverse_norm
        # Compute ||R||inf * ||inv(R)||inf without materializing inv(R).
        sums = small()
        norm_R = wp.float64(0.0)
        for row in range(rows):
            row_sum = wp.float64(0.0)
            for col in range(rows):
                row_sum += wp.abs(R[row, col])
            norm_R = wp.max(norm_R, row_sum)
        for rhs in range(rows):
            inverse_column = small()
            for reverse in range(rows):
                row = rows - reverse - 1
                value = wp.float64(0.0)
                if row == rhs:
                    value = wp.float64(1.0)
                for col in range(row + 1, rows):
                    value -= R[row, col] * inverse_column[col]
                inverse_column[row] = value / R[row, row]
                sums[row] += wp.abs(inverse_column[row])
        inverse_norm = wp.float64(0.0)
        for row in range(rows):
            inverse_norm = wp.max(inverse_norm, sums[row])
        bound = norm_R * inverse_norm
        condition[lane] = bound
        if not wp.isfinite(bound) or bound > wp.float64(1e8):
            return
        rank = int(rows)
        for seed in range(columns):
            if rank < columns:
                candidate = vector()
                candidate[seed] = wp.float64(1.0)
                for repeat in range(2):
                    for previous in range(rank):
                        projection = wp.float64(0.0)
                        for col in range(columns):
                            projection += Q[previous, col] * candidate[col]
                        for col in range(columns):
                            candidate[col] -= projection * Q[previous, col]
                norm = wp.sqrt(wp.dot(candidate, candidate))
                if norm > wp.float64(1e-6):
                    inverse_norm = wp.float64(1.0) / norm
                    for col in range(columns):
                        Q[rank, col] = candidate[col] * inverse_norm
                    rank += 1
        if rank != columns:
            return
        y = small()
        for row in range(rows):
            value = goal[lane, row]
            for previous in range(row):
                value -= R[previous, row] * y[previous]
            y[row] = value / R[row, row]
        for col in range(columns):
            value = wp.float64(0.0)
            for row in range(rows):
                value += Q[row, col] * y[row]
            if not wp.isfinite(value):
                return
            w[lane, col] = value
            for k in range(columns - rows):
                N[lane, col, k] = Q[rows + k, col]
        # Independently verify the actual complete basis and primary equations.
        error = wp.float64(0.0)
        for a in range(columns):
            for b in range(columns):
                dot = wp.float64(0.0)
                for col in range(columns):
                    dot += Q[a, col] * Q[b, col]
                target = wp.float64(0.0)
                if a == b:
                    target = wp.float64(1.0)
                error = wp.max(error, wp.abs(dot - target))
        if error > wp.float64(1e-10):
            return
        for row in range(rows):
            image = wp.float64(0.0)
            for col in range(columns):
                image += J[lane, row, col] * w[lane, col]
            if wp.abs(image - goal[lane, row]) > wp.float64(1e-10) * wp.max(
                wp.float64(1.0), wp.abs(goal[lane, row])
            ):
                return
            for k in range(columns - rows):
                image = wp.float64(0.0)
                for col in range(columns):
                    image += J[lane, row, col] * N[lane, col, k]
                if wp.abs(image) > wp.float64(1e-10) * wp.max(wp.float64(1.0), maximum):
                    return
        regular[lane] = True

    wp.set_module_options({"max_unroll": 1}, module=kernel.module)
    return kernel


class WarpNullspace:
    def __init__(self, rows: int, columns: int, *, batch: int, device: torch.device):
        self.kernel = make_nullspace_kernel(rows, columns)
        self.w = torch.empty((batch, columns), dtype=torch.float64, device=device)
        self.N = torch.empty((batch, columns, columns - rows), dtype=torch.float64, device=device)
        self.regular = torch.empty(batch, dtype=torch.bool, device=device)
        self.condition = torch.empty(batch, dtype=torch.float64, device=device)
        wp.load_module(module=self.kernel.module, device=str(device))
        self.outputs = [wp.from_torch(v) for v in (self.w, self.N, self.regular, self.condition)]

    def solve(self, J, goal):
        wp.launch(
            self.kernel,
            dim=J.shape[0],
            inputs=[wp.from_torch(J.contiguous()), wp.from_torch(goal.contiguous())],
            outputs=self.outputs,
            block_dim=32,
            stream=wp.stream_from_torch(torch.cuda.current_stream(J.device)),
        )
        return self.w, self.N, self.regular, self.condition
