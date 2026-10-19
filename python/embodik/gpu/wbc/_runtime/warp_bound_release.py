"""Graph-safe SCALE bound release and primary-preserving realization repair."""

from __future__ import annotations

from functools import lru_cache

import torch
import warp as wp


@lru_cache(None)
def _make_kernel(
    rows: int,
    columns: int,
    refine_partial_scale: bool = False,
    refine_from_origin: bool = False,
    refinement_ray_tolerance: float = 1e-8,
    repair_float32_roundoff: bool = False,
):
    slack_column = columns + 1
    width = columns + 2
    vector = wp.types.vector(length=width, dtype=wp.float64)
    indices = wp.types.vector(length=width, dtype=wp.int32)
    matrix = wp.types.matrix(shape=(width, width), dtype=wp.float64)
    task_vector = wp.types.vector(length=rows, dtype=wp.float64)

    @wp.kernel(enable_backward=False, module="unique")
    def kernel(
        jacobian: wp.array3d(dtype=wp.float64),
        target: wp.array2d(dtype=wp.float64),
        lower: wp.array2d(dtype=wp.float64),
        upper: wp.array2d(dtype=wp.float64),
        original: wp.array2d(dtype=wp.float64),
        result: wp.array2d(dtype=wp.float64),
        recovered: wp.array(dtype=wp.bool),
    ):
        batch = wp.tid()
        recovered[batch] = False
        norm = wp.float64(0.0)
        eligible = bool(True)
        requires_inward = bool(False)
        original_invalid = bool(False)
        point = vector()
        for col in range(columns):
            result[batch, col] = original[batch, col]
            norm += original[batch, col] * original[batch, col]
            if original[batch, col] < lower[batch, col] - wp.float64(1e-8) or original[
                batch, col
            ] > upper[batch, col] + wp.float64(1e-8):
                original_invalid = True
            if lower[batch, col] > wp.float64(0.0) or upper[batch, col] < wp.float64(0.0):
                requires_inward = True
            if (
                not wp.isfinite(lower[batch, col])
                or not wp.isfinite(upper[batch, col])
                or not wp.isfinite(original[batch, col])
                or lower[batch, col] > upper[batch, col]
            ):
                eligible = False
            point[col] = wp.clamp(wp.float64(0.0), lower[batch, col], upper[batch, col])
        for row in range(rows):
            if not wp.isfinite(target[batch, row]):
                eligible = False
            for col in range(columns):
                if not wp.isfinite(jacobian[batch, row, col]):
                    eligible = False
        if not eligible or (
            not wp.static(refine_partial_scale)
            and (not original_invalid and not requires_inward and norm > wp.float64(1e-20))
        ):
            return
        task_goal = task_vector()
        original_task_norm = wp.float64(0.0)
        for row in range(rows):
            original_value = wp.float64(0.0)
            for col in range(columns):
                original_value += jacobian[batch, row, col] * original[batch, col]
            task_goal[row] = original_value
            original_task_norm += original_value * original_value
        repair_moving = original_invalid and original_task_norm > wp.float64(1e-20)
        refine_moving = bool(False)
        roundoff_restart = bool(False)
        original_scale = wp.float64(0.0)
        if wp.static(refine_partial_scale):
            if not original_invalid and original_task_norm > wp.float64(1e-20):
                goal_norm = wp.float64(0.0)
                image_dot_goal = wp.float64(0.0)
                for row in range(rows):
                    goal_norm += target[batch, row] * target[batch, row]
                    image_dot_goal += task_goal[row] * target[batch, row]
                if goal_norm <= wp.float64(1e-20):
                    return
                original_scale = image_dot_goal / goal_norm
                full_scale_threshold = wp.float64(1.0 - 1e-10)
                if wp.static(refine_from_origin):
                    full_scale_threshold = wp.float64(1.0) - wp.float64(refinement_ray_tolerance)
                ray_error = wp.float64(0.0)
                for row in range(rows):
                    error = task_goal[row] - original_scale * target[batch, row]
                    ray_error += error * error
                ray_norm = wp.sqrt(ray_error)
                ray_scale = wp.max(wp.float64(1.0), wp.sqrt(goal_norm))
                if wp.static(repair_float32_roundoff):
                    roundoff_restart = (
                        not requires_inward
                        and original_scale < wp.float64(1.0 - 1e-5)
                        and ray_norm > wp.float64(1e-8) * ray_scale
                        and ray_norm <= wp.float64(1e-5) * ray_scale
                    )
                if (
                    original_scale <= wp.float64(1e-10)
                    or original_scale >= full_scale_threshold
                    or (
                        ray_norm > wp.float64(refinement_ray_tolerance) * ray_scale
                        and not roundoff_restart
                    )
                ):
                    return
                refine_moving = True
        # Valid moving realizations retain their fast exit. An invalid moving
        # realization may only change in the primary nullspace: solve for its
        # achieved image, then require full scale before publishing the repair.
        if (
            not repair_moving
            and not refine_moving
            and (
                (not requires_inward and norm > wp.float64(1e-20))
                or original_task_norm > wp.float64(1e-20)
            )
        ):
            return
        if not repair_moving:
            for row in range(rows):
                task_goal[row] = target[batch, row]
        if refine_moving:
            for col in range(columns):
                point[col] = original[batch, col]
            point[columns] = original_scale
            if roundoff_restart:
                # Rounded seeds need a feasible equality start, not just
                # tangent updates that retain their original ray residual.
                point = vector()
            if wp.static(refine_from_origin):
                if not requires_inward:
                    # A rounded moving seed need not lie on Jv=alpha*t.
                    # Zero is exact and feasible when the box contains it.
                    point = vector()
        needs_seed_search = bool(False)
        if requires_inward and not refine_moving:
            seed_task_norm = wp.float64(0.0)
            for row in range(rows):
                seed_value = wp.float64(0.0)
                for col in range(columns):
                    seed_value += jacobian[batch, row, col] * point[col]
                seed_task_norm += seed_value * seed_value
            needs_seed_search = seed_task_norm > wp.float64(1e-20)
        valid = bool(True)
        # Phase I starts from v=0, alpha=0 with a common box slack. It retains
        # Jv=alpha*target, allowing feasible positive rays without a null seed.
        # Phase II fixes slack to zero and maximizes the feasible task scale.
        for phase in range(2):
            if phase == 0 and not needs_seed_search:
                continue
            if phase == 0:
                point = vector()
                slack = wp.float64(0.0)
                for col in range(columns):
                    slack = wp.max(slack, lower[batch, col])
                    slack = wp.max(slack, -upper[batch, col])
                point[slack_column] = slack + wp.float64(1e-8)
            else:
                point[slack_column] = wp.float64(0.0)
            equality_rows = int(rows)
            objective_column = int(columns)
            if phase == 0:
                objective_column = slack_column
            else:
                equality_rows += 1
            active = indices()
            count = int(0)
            # Fixed equality rows are invariant within the phase. Appended
            # active rows overwrite every basis/triangular entry they use.
            basis = matrix()
            triangular = matrix()
            equality_rank = int(0)
            for iteration in range(256):
                rank = equality_rank
                first_row = int(0)
                if iteration > 0:
                    first_row = equality_rows
                for row in range(first_row, equality_rows + count):
                    normal = vector()
                    if row < rows:
                        for col in range(columns):
                            normal[col] = jacobian[batch, row, col]
                        normal[columns] = -task_goal[row]
                    elif row < equality_rows:
                        normal[slack_column] = wp.float64(1.0)
                    else:
                        bound = active[row - equality_rows]
                        col = bound % width
                        normal[col] = wp.float64(1.0)
                        if bound >= width:
                            normal[col] = wp.float64(-1.0)
                        if phase == 0 and col < columns:
                            normal[slack_column] = wp.float64(-1.0)
                    length = wp.sqrt(wp.dot(normal, normal))
                    if length > wp.float64(1e-30):
                        normal = normal / length
                    coefficients = vector()
                    # Two-pass modified Gram-Schmidt keeps the task/bound basis
                    # orthogonal, including nearly dependent task rows.
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
                        for col in range(width):
                            basis[rank, col] = normal[col] / length
                        rank += 1
                    elif row >= equality_rows:
                        valid = False
                    if row == equality_rows - 1:
                        equality_rank = rank
                if not valid:
                    break
                direction = vector()
                direction[objective_column] = wp.float64(1.0)
                if phase == 0:
                    direction[objective_column] = wp.float64(-1.0)
                for repeat in range(2):
                    for row in range(rank):
                        projection = wp.float64(0.0)
                        for col in range(width):
                            projection += basis[row, col] * direction[col]
                        for col in range(width):
                            direction[col] -= projection * basis[row, col]
                if wp.sqrt(wp.dot(direction, direction)) <= wp.float64(1e-10):
                    multipliers = vector()
                    for reverse in range(rank):
                        row = rank - reverse - 1
                        value = basis[row, objective_column]
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
                    if remove < 0:
                        break
                    for index in range(remove, count - 1):
                        active[index] = active[index + 1]
                    count -= 1
                    continue
                distance = wp.float64(1e30)
                blocker = int(-1)
                for bound in range(2 * width):
                    if phase == 0 and bound == slack_column:
                        continue
                    col = bound % width
                    if phase == 1 and col == slack_column:
                        continue
                    sign = wp.float64(1.0)
                    limit = wp.float64(1.0)
                    if col < columns:
                        limit = upper[batch, col]
                    if bound >= width:
                        sign = wp.float64(-1.0)
                        limit = wp.float64(0.0)
                        if col < columns:
                            limit = -lower[batch, col]
                    already_active = bool(False)
                    for index in range(count):
                        if active[index] == bound:
                            already_active = True
                    slope = sign * direction[col]
                    slack = limit - sign * point[col]
                    if phase == 0 and col < columns:
                        slope -= direction[slack_column]
                        slack += point[slack_column]
                    if not already_active and slope > wp.float64(1e-10):
                        ratio = wp.max(wp.float64(0.0), slack) / slope
                        if ratio < distance:
                            distance = ratio
                            blocker = bound
                if blocker < 0 or count >= width:
                    valid = False
                    break
                point += distance * direction
                active[count] = blocker
                count += 1
                if (phase == 0 and point[slack_column] <= wp.float64(1e-10)) or (
                    phase == 1 and point[columns] >= wp.float64(1.0 - 1e-10)
                ):
                    break
            if phase == 0:
                valid = valid and point[slack_column] <= wp.float64(1e-10)
                valid = valid and wp.isfinite(point[columns])
                valid = valid and point[columns] >= wp.float64(-1e-8)
                valid = valid and point[columns] <= wp.float64(1.0 + 1e-8)
                for col in range(columns):
                    if (
                        not wp.isfinite(point[col])
                        or point[col] < lower[batch, col] - wp.float64(1e-8)
                        or point[col] > upper[batch, col] + wp.float64(1e-8)
                    ):
                        valid = False
                for row in range(rows):
                    value = -point[columns] * task_goal[row]
                    for col in range(columns):
                        value += jacobian[batch, row, col] * point[col]
                    if wp.abs(value) > wp.float64(1e-8):
                        valid = False
            if not valid:
                break
        alpha = wp.clamp(point[columns], wp.float64(0.0), wp.float64(1.0))
        # Mandatory inward motion still needs a feasible command when the
        # task ray collapses to zero. The final box/equality checks remain
        # required; this is bound recovery, not successful target tracking.
        valid = valid and (alpha > wp.float64(1e-10) or requires_inward)
        if refine_moving and alpha <= original_scale + wp.float64(1e-10):
            valid = False
        if wp.static(refine_from_origin):
            if refine_moving and alpha <= original_scale + wp.float64(refinement_ray_tolerance):
                valid = False
        if repair_moving and alpha < wp.float64(1.0) - wp.float64(1e-10):
            valid = False
        target_norm = wp.float64(0.0)
        residual_norm = wp.float64(0.0)
        for row in range(rows):
            # Moving repair preserves the original image, including any
            # least-squares residual; SCALE recovery checks the scaled request.
            required_scale = alpha
            if repair_moving:
                required_scale = wp.float64(1.0)
            residual = -required_scale * task_goal[row]
            for col in range(columns):
                residual += jacobian[batch, row, col] * point[col]
            residual_norm += residual * residual
            target_norm += task_goal[row] * task_goal[row]
        if wp.sqrt(residual_norm) > wp.float64(1e-8) * wp.max(
            wp.float64(1.0), wp.sqrt(target_norm)
        ):
            valid = False
        for col in range(columns):
            if (
                not wp.isfinite(point[col])
                or point[col] < lower[batch, col] - wp.float64(1e-8)
                or point[col] > upper[batch, col] + wp.float64(1e-8)
            ):
                valid = False
        if roundoff_restart:
            valid = valid and alpha > original_scale + wp.float64(1e-5)
            rounded_image_dot_goal = wp.float64(0.0)
            for row in range(rows):
                rounded_image = wp.float64(0.0)
                for col in range(columns):
                    rounded_velocity = wp.float64(wp.float32(point[col]))
                    rounded_image += jacobian[batch, row, col] * rounded_velocity
                rounded_image_dot_goal += rounded_image * task_goal[row]
            rounded_scale = rounded_image_dot_goal / target_norm
            valid = valid and wp.isfinite(rounded_scale)
            valid = valid and rounded_scale > original_scale + wp.float64(1e-5)
        if valid:
            for col in range(columns):
                result[batch, col] = point[col]
            recovered[batch] = True

    wp.set_module_options({"max_unroll": 1}, module=kernel.module)
    return kernel


class WarpBoundRelease:
    """A native launch with opt-in partial SCALE refinement.

    Origin refinement starts from an exact zero velocity/scale seed when the
    velocity box contains zero. The ray tolerance classifies rounded input
    seeds and skips negligible rate changes; final equality and bound checks
    retain their independent 1e-8 thresholds. Defaults preserve moving seeds.
    Explicit float32 roundoff repair conditionally restarts a slightly off-ray
    partial seed from zero only when zero belongs to its velocity box.
    """

    def __init__(
        self,
        rows: int,
        columns: int,
        *,
        batch_capacity: int,
        device: str,
        refine_partial_scale: bool = False,
        refine_from_origin: bool = False,
        refinement_ray_tolerance: float = 1e-8,
        repair_float32_roundoff: bool = False,
    ):
        import math

        if not math.isfinite(refinement_ray_tolerance) or refinement_ray_tolerance <= 0:
            raise ValueError("refinement ray tolerance must be finite and positive")
        if refine_from_origin and not refine_partial_scale:
            raise ValueError("origin refinement requires partial-scale refinement")
        if not isinstance(repair_float32_roundoff, bool):
            raise ValueError("float32 roundoff repair must be a boolean")
        self.rows = rows
        self.columns = columns
        self.batch_capacity = batch_capacity
        self.device = torch.device(device)
        if self.device.type != "cuda" or rows < 1 or columns < 1 or batch_capacity < 1:
            raise ValueError("positive dimensions, capacity and a CUDA device are required")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.result = torch.empty((batch_capacity, columns), dtype=torch.float64, device=device)
        self.recovered = torch.empty(batch_capacity, dtype=torch.bool, device=device)
        self._kernel = _make_kernel(
            rows, columns, refine_partial_scale, refine_from_origin, refinement_ray_tolerance
        )
        self._roundoff_kernel = None
        if repair_float32_roundoff and refine_partial_scale:
            self._roundoff_kernel = _make_kernel(
                rows,
                columns,
                refine_partial_scale,
                refine_from_origin,
                refinement_ray_tolerance,
                True,
            )
        for kernel in (self._kernel, self._roundoff_kernel):
            if kernel is not None:
                wp.load_module(module=kernel.module, device=device)
        self._outputs = [wp.from_torch(self.result), wp.from_torch(self.recovered)]

    def solve(self, jacobian, target, lower, upper, velocity):
        batch = velocity.shape[0]
        if not 1 <= batch <= self.batch_capacity:
            raise ValueError("batch exceeds bound-release capacity or is empty")
        tensors = []
        for name, value, shape in (
            ("jacobian", jacobian, (batch, self.rows, self.columns)),
            ("target", target, (batch, self.rows)),
            ("lower", lower, (batch, self.columns)),
            ("upper", upper, (batch, self.columns)),
            ("velocity", velocity, (batch, self.columns)),
        ):
            if tuple(value.shape) != shape or value.device != self.device:
                raise ValueError(f"{name} must have shape {shape} on {self.device}")
            tensors.append(value.to(torch.float64).contiguous())
        torch_stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        warp_stream = wp.get_stream(str(self.device))
        stream = warp_stream if warp_stream.is_capturing else torch_stream
        kernel = self._kernel
        if velocity.dtype == torch.float32 and self._roundoff_kernel is not None:
            kernel = self._roundoff_kernel
        wp.launch(
            kernel,
            dim=batch,
            inputs=[wp.from_torch(value) for value in tensors],
            outputs=self._outputs,
            block_dim=32,
            stream=stream,
        )
        return self.result[:batch].to(velocity.dtype)
