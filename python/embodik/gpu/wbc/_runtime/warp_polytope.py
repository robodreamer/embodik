"""Experimental native SCALE LP with general hard rows, without a box lift.

Linear algebra only: no FK, pose projection, collision geometry, acceleration,
secondary objectives or robot publication. Status0 means a validated feasible
LP optimum (possibly scale0);1 invalid lane,2 infeasible,3 numerical/iteration
failure. A failed lane's zero output is diagnostic and must not be published.
"""

from functools import lru_cache

import torch
import warp as wp


@lru_cache(None)
def make_polytope_kernel(rows: int, columns: int, hard_rows: int):
    width = columns + 2
    alpha_col, slack_col = columns, columns + 1
    vector = wp.types.vector(length=width, dtype=wp.float64)
    indices = wp.types.vector(length=width, dtype=wp.int32)
    matrix = wp.types.matrix(shape=(width, width), dtype=wp.float64)
    row_vector = wp.types.vector(length=max(1, hard_rows), dtype=wp.float64)
    bounds = 2 * columns + 2 * hard_rows + 3

    @wp.func
    def bound_normal(
        bound: int,
        phase: int,
        batch: int,
        C: wp.array3d(dtype=wp.float64),
        row_scales: row_vector,
    ):
        normal = vector()
        if bound < 2 * columns:
            col = bound % columns
            normal[col] = wp.float64(1.0)
            if bound >= columns:
                normal[col] = wp.float64(-1.0)
            if phase == 0:
                normal[slack_col] = wp.float64(-1.0)
        elif bound < 2 * columns + 2 * hard_rows:
            index = bound - 2 * columns
            sign = wp.float64(1.0)
            if index >= hard_rows:
                sign = wp.float64(-1.0)
                index -= hard_rows
            for col in range(columns):
                normal[col] = sign * C[batch, index, col] * row_scales[index]
            if phase == 0:
                normal[slack_col] = wp.float64(-1.0)
        elif bound == bounds - 3:
            normal[alpha_col] = wp.float64(1.0)
        elif bound == bounds - 2:
            normal[alpha_col] = wp.float64(-1.0)
        else:
            normal[slack_col] = wp.float64(-1.0)
        return normal

    @wp.func
    def bound_limit(
        bound: int,
        batch: int,
        lower: wp.array2d(dtype=wp.float64),
        upper: wp.array2d(dtype=wp.float64),
        row_lower: wp.array2d(dtype=wp.float64),
        row_upper: wp.array2d(dtype=wp.float64),
        row_scales: row_vector,
    ):
        limit = wp.float64(0.0)
        if bound < columns:
            limit = upper[batch, bound]
        elif bound < 2 * columns:
            limit = -lower[batch, bound - columns]
        elif bound < 2 * columns + hard_rows:
            index = bound - 2 * columns
            limit = row_upper[batch, index] * row_scales[index]
        elif bound < 2 * columns + 2 * hard_rows:
            index = bound - 2 * columns - hard_rows
            limit = -row_lower[batch, index] * row_scales[index]
        elif bound == bounds - 3:
            limit = wp.float64(1.0)
        return limit

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        J: wp.array3d(dtype=wp.float64),
        goal: wp.array2d(dtype=wp.float64),
        C: wp.array3d(dtype=wp.float64),
        lower: wp.array2d(dtype=wp.float64),
        upper: wp.array2d(dtype=wp.float64),
        row_lower: wp.array2d(dtype=wp.float64),
        row_upper: wp.array2d(dtype=wp.float64),
        enabled: wp.array(dtype=wp.bool),
        velocity: wp.array2d(dtype=wp.float64),
        scales: wp.array(dtype=wp.float64),
        status: wp.array(dtype=wp.int32),
        iterations: wp.array(dtype=wp.int32),
        detail: wp.array(dtype=wp.int32),
        diagnostic: wp.array2d(dtype=wp.float64),
    ):
        batch = wp.tid()
        status[batch] = 1
        scales[batch] = wp.float64(0.0)
        iterations[batch] = 0
        detail[batch] = 0
        diagnostic[batch, 0] = wp.float64(0.0)
        diagnostic[batch, 1] = wp.float64(0.0)
        eligible = bool(True)
        slack = wp.float64(0.0)
        row_scales = row_vector()
        for col in range(columns):
            velocity[batch, col] = wp.float64(0.0)
        if not enabled[batch]:
            status[batch] = 4
            return
        for col in range(columns):
            if (
                not wp.isfinite(lower[batch, col])
                or not wp.isfinite(upper[batch, col])
                or lower[batch, col] > upper[batch, col]
            ):
                eligible = False
            slack = wp.max(slack, lower[batch, col])
            slack = wp.max(slack, -upper[batch, col])
        for row in range(rows):
            if not wp.isfinite(goal[batch, row]):
                eligible = False
            for col in range(columns):
                if not wp.isfinite(J[batch, row, col]):
                    eligible = False
        for row in range(hard_rows):
            if (
                not wp.isfinite(row_lower[batch, row])
                or not wp.isfinite(row_upper[batch, row])
                or row_lower[batch, row] > row_upper[batch, row]
            ):
                eligible = False
            norm_squared = wp.float64(0.0)
            for col in range(columns):
                if not wp.isfinite(C[batch, row, col]):
                    eligible = False
                norm_squared += C[batch, row, col] * C[batch, row, col]
            length = wp.sqrt(norm_squared)
            if not wp.isfinite(length):
                eligible = False
            if length <= wp.float64(1e-30):
                length = wp.float64(1.0)
            row_scales[row] = wp.float64(1.0) / length
            slack = wp.max(slack, row_lower[batch, row] * row_scales[row])
            slack = wp.max(slack, -row_upper[batch, row] * row_scales[row])
        if not eligible:
            return
        status[batch] = 3
        point = vector()
        valid = bool(True)
        for phase in range(2):
            if phase == 0 and slack <= wp.float64(0.0):
                continue
            if phase == 0:
                point[slack_col] = slack + wp.float64(1e-8)
            else:
                point[slack_col] = wp.float64(0.0)
            equality_rows = int(rows)
            objective_col = int(alpha_col)
            if phase == 0:
                objective_col = slack_col
            else:
                equality_rows += 1
            active = indices()
            sources = indices()
            count = int(0)
            basis, triangular = matrix(), matrix()
            equality_rank = int(0)
            done = bool(False)
            for iteration in range(256):
                iterations[batch] += 1
                rank = equality_rank
                first_row = int(0)
                if iteration > 0:
                    first_row = equality_rows
                for row in range(first_row, equality_rows + count):
                    normal = vector()
                    if row < rows:
                        for col in range(columns):
                            normal[col] = J[batch, row, col]
                        normal[alpha_col] = -goal[batch, row]
                    elif row < equality_rows:
                        normal[slack_col] = wp.float64(1.0)
                    else:
                        normal = bound_normal(
                            active[row - equality_rows], phase, batch, C, row_scales
                        )
                    length = wp.sqrt(wp.dot(normal, normal))
                    if length > wp.float64(1e-30):
                        normal /= length
                    coefficients = vector()
                    for repeat in range(2):
                        for previous in range(rank):
                            projection = wp.float64(0.0)
                            for col in range(width):
                                projection += basis[previous, col] * normal[col]
                            coefficients[previous] += projection
                            for col in range(width):
                                normal[col] -= projection * basis[previous, col]
                    length = wp.sqrt(wp.dot(normal, normal))
                    if length > wp.float64(1e-10) and rank < width:
                        for previous in range(rank):
                            triangular[previous, rank] = coefficients[previous]
                        triangular[rank, rank] = length
                        sources[rank] = row
                        for col in range(width):
                            basis[rank, col] = normal[col] / length
                        rank += 1
                    elif row >= equality_rows:
                        valid = False
                        detail[batch] = 10 * phase + 1
                    if row == equality_rows - 1:
                        equality_rank = rank
                if not valid:
                    break
                direction = vector()
                direction[objective_col] = wp.float64(1.0)
                if phase == 0:
                    direction[objective_col] = wp.float64(-1.0)
                for repeat in range(2):
                    for row in range(rank):
                        projection = wp.float64(0.0)
                        for col in range(width):
                            projection += basis[row, col] * direction[col]
                        for col in range(width):
                            direction[col] -= projection * basis[row, col]
                if wp.sqrt(wp.dot(direction, direction)) <= wp.float64(1e-10) or (
                    phase == 1 and wp.sqrt(wp.dot(direction, direction)) <= wp.float64(1e-8)
                ):
                    multipliers = vector()
                    for reverse in range(rank):
                        row = rank - reverse - 1
                        value = basis[row, objective_col]
                        if phase == 0:
                            value = -value
                        for col in range(row + 1, rank):
                            value -= triangular[row, col] * multipliers[col]
                        multipliers[row] = value / triangular[row, row]
                    remove = int(-1)
                    most_negative = wp.float64(-1e-10)
                    for index in range(count):
                        if multipliers[equality_rank + index] < most_negative:
                            most_negative = multipliers[equality_rank + index]
                            remove = index
                    if remove >= 0:
                        for index in range(remove, count - 1):
                            active[index] = active[index + 1]
                        count -= 1
                        continue
                    if phase == 0:
                        done = True
                        break
                    # A tiny projected direction alone does not prove an LP
                    # optimum. Bound the actual primal-dual gap over the finite
                    # original velocity box, keeping alpha in[0,1] and slack0.
                    dual_residual = vector()
                    dual_residual[alpha_col] = wp.float64(1.0)
                    gap = wp.float64(0.0)
                    for row in range(rank):
                        normal = vector()
                        source = sources[row]
                        limit = wp.float64(0.0)
                        if source < rows:
                            for col in range(columns):
                                normal[col] = J[batch, source, col]
                            normal[alpha_col] = -goal[batch, source]
                        elif source < equality_rows:
                            normal[slack_col] = wp.float64(1.0)
                        else:
                            bound = active[source - equality_rows]
                            normal = bound_normal(bound, phase, batch, C, row_scales)
                            limit = bound_limit(
                                bound,
                                batch,
                                lower,
                                upper,
                                row_lower,
                                row_upper,
                                row_scales,
                            )
                        length = wp.sqrt(wp.dot(normal, normal))
                        if length > wp.float64(1e-30):
                            normal /= length
                            limit /= length
                        multiplier = multipliers[row]
                        if row >= equality_rank:
                            multiplier = wp.max(wp.float64(0.0), multiplier)
                        dual_residual -= multiplier * normal
                        gap += multiplier * (limit - wp.dot(normal, point))
                    for col in range(columns):
                        limit = lower[batch, col]
                        if dual_residual[col] >= wp.float64(0.0):
                            limit = upper[batch, col]
                        gap += dual_residual[col] * (limit - point[col])
                    alpha_limit = wp.float64(0.0)
                    if dual_residual[alpha_col] >= wp.float64(0.0):
                        alpha_limit = wp.float64(1.0)
                    gap += dual_residual[alpha_col] * (alpha_limit - point[alpha_col])
                    if wp.isfinite(gap) and gap <= wp.float64(1e-8):
                        done = True
                        break
                distance = wp.float64(1e30)
                blocker = int(-1)
                for bound in range(bounds):
                    if phase == 1 and bound == bounds - 1:
                        continue
                    already = bool(False)
                    for index in range(count):
                        if active[index] == bound:
                            already = True
                    normal = bound_normal(bound, phase, batch, C, row_scales)
                    slope = wp.dot(normal, direction)
                    available = bound_limit(
                        bound, batch, lower, upper, row_lower, row_upper, row_scales
                    ) - wp.dot(normal, point)
                    if not already and slope > wp.float64(1e-10):
                        ratio = wp.max(wp.float64(0.0), available) / slope
                        if ratio < distance:
                            distance, blocker = ratio, bound
                if blocker < 0 or count >= width:
                    valid = False
                    detail[batch] = 10 * phase + 2
                    diagnostic[batch, 0] = wp.sqrt(wp.dot(direction, direction))
                    diagnostic[batch, 1] = wp.abs(direction[alpha_col])
                    break
                point += distance * direction
                active[count] = blocker
                count += 1
                if (phase == 0 and point[slack_col] <= wp.float64(1e-10)) or (
                    phase == 1 and point[alpha_col] >= wp.float64(1.0 - 1e-10)
                ):
                    done = True
                    break
            if not done:
                if valid:
                    detail[batch] = 10 * phase + 3
                valid = False
            if phase == 0 and valid and point[slack_col] > wp.float64(1e-10):
                status[batch] = 2
                return
            if not valid:
                return
        alpha = wp.clamp(point[alpha_col], wp.float64(0.0), wp.float64(1.0))
        residual, goal_norm = wp.float64(0.0), wp.float64(0.0)
        for row in range(rows):
            value = -alpha * goal[batch, row]
            for col in range(columns):
                value += J[batch, row, col] * point[col]
            residual += value * value
            goal_norm += goal[batch, row] * goal[batch, row]
        if (
            not wp.isfinite(residual)
            or not wp.isfinite(goal_norm)
            or wp.sqrt(residual) > wp.float64(1e-8) * wp.max(wp.float64(1.0), wp.sqrt(goal_norm))
        ):
            valid = False
        for col in range(columns):
            if (
                not wp.isfinite(point[col])
                or point[col] < lower[batch, col] - wp.float64(1e-8)
                or point[col] > upper[batch, col] + wp.float64(1e-8)
            ):
                valid = False
        for row in range(hard_rows):
            value = wp.float64(0.0)
            for col in range(columns):
                value += C[batch, row, col] * point[col]
            if (
                not wp.isfinite(value)
                or value < row_lower[batch, row] - wp.float64(1e-8)
                or value > row_upper[batch, row] + wp.float64(1e-8)
            ):
                valid = False
        if not valid or not wp.isfinite(alpha):
            detail[batch] = 24
        if valid and wp.isfinite(alpha):
            for col in range(columns):
                velocity[batch, col] = point[col]
            scales[batch] = alpha
            status[batch] = 0

    wp.set_module_options({"max_unroll": 1}, module=kernel.module)
    return kernel


class WarpPolytopeScale:
    """Experimental fixed-shape graph-safe float64 native primary SCALE LP."""

    def __init__(
        self,
        rows: int,
        columns: int,
        hard_rows: int,
        *,
        batch: int,
        device: str = "cuda",
    ):
        if (
            any(
                isinstance(v, bool) or not isinstance(v, int) or v < 1
                for v in (rows, columns, batch)
            )
            or isinstance(hard_rows, bool)
            or not isinstance(hard_rows, int)
            or hard_rows < 0
        ):
            raise ValueError(
                "Positive task dimensions/batch and nonnegative hard row count required"
            )
        self.shape = (batch, rows, columns, hard_rows)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CUDA device required")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.velocity = torch.empty((batch, columns), dtype=torch.float64, device=self.device)
        self.scales = torch.empty(batch, dtype=torch.float64, device=self.device)
        self.status = torch.empty(batch, dtype=torch.int32, device=self.device)
        self.iterations = torch.empty_like(self.status)
        self.detail = torch.empty_like(self.status)
        self.diagnostic = torch.empty((batch, 2), dtype=torch.float64, device=self.device)
        self.kernel = make_polytope_kernel(rows, columns, hard_rows)
        self.enabled = torch.ones(batch, dtype=torch.bool, device=self.device)
        wp.load_module(module=self.kernel.module, device=str(self.device))
        self.outputs = [
            wp.from_torch(v)
            for v in (
                self.velocity,
                self.scales,
                self.status,
                self.iterations,
                self.detail,
                self.diagnostic,
            )
        ]

    def solve(self, J, goal, C, lower, upper, row_lower, row_upper, *, enabled=None):
        batch, rows, columns, hard_rows = self.shape
        inputs = []
        for name, value, shape in zip(
            ("J", "goal", "C", "lower", "upper", "row_lower", "row_upper"),
            (J, goal, C, lower, upper, row_lower, row_upper),
            (
                (batch, rows, columns),
                (batch, rows),
                (batch, hard_rows, columns),
                (batch, columns),
                (batch, columns),
                (batch, hard_rows),
                (batch, hard_rows),
            ),
        ):
            if (
                tuple(value.shape) != shape
                or value.device != self.device
                or value.dtype != torch.float64
            ):
                raise ValueError(f"{name} requires shape{shape}, float64 on{self.device}")
            inputs.append(wp.from_torch(value.contiguous()))
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        if enabled is None:
            enabled = self.enabled
        if (
            enabled.shape != self.enabled.shape
            or enabled.device != self.device
            or enabled.dtype != torch.bool
        ):
            raise ValueError("enabled requires a bool lane mask on the solver device")
        inputs.append(wp.from_torch(enabled.contiguous()))
        wp.launch(
            self.kernel,
            dim=batch,
            inputs=inputs,
            outputs=self.outputs,
            block_dim=32,
            stream=stream,
        )
        return self.velocity, self.scales, self.status
