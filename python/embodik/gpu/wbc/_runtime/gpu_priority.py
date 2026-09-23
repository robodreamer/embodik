"""Pure-Torch, model-derived helpers for hierarchical GPU WBC tasks.

The posture mapping intentionally preserves EmbodiK's controlled
``PostureTask`` convention.  In particular, a controlled floating root uses
the first six raw configuration-coordinate differences as an approximation;
scalar joints are mapped from velocity to configuration coordinates through
the model-provided joint spans.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import torch


class PostureTaskRows(NamedTuple):
    """Weighted posture rows in the configured active tangent order."""

    error: torch.Tensor
    goal: torch.Tensor
    jacobian: torch.Tensor


class SecondaryPriorityResult(NamedTuple):
    """Secondary-priority velocity and per-batch solve diagnostics."""

    velocity: torch.Tensor
    primary_residual_increase: torch.Tensor
    secondary_residual_before: torch.Tensor
    secondary_residual_after: torch.Tensor
    delta_norm: torch.Tensor
    applied: torch.Tensor


def ordered_priority_velocity(
    primary_velocity: torch.Tensor,
    primary_jacobian: torch.Tensor,
    bands: Sequence[tuple[torch.Tensor, torch.Tensor]],
    lower_velocity: torch.Tensor,
    upper_velocity: torch.Tensor,
    **options,
) -> tuple[SecondaryPriorityResult, ...]:
    """Apply ordered (Jacobian, goal) bands below an already solved primary.

    Each band uses directional SRINV for its projected residual and an
    undamped projector for *all* preceding Jacobians. Even a damped or
    box-limited band retains its achieved motion when later bands run.
    Returns per-band diagnostics; an empty hierarchy returns an empty tuple.
    """
    velocity = primary_velocity
    protected = primary_jacobian
    results = []
    for band_index, (jacobian, goal) in enumerate(bands):
        result = secondary_priority_velocity(
            velocity,
            protected,
            jacobian,
            goal,
            lower_velocity,
            upper_velocity,
            **options,
        )
        results.append(result)
        velocity = result.velocity
        if band_index == len(bands) - 1:
            break
        batch = torch.broadcast_shapes(protected.shape[:-2], jacobian.shape[:-2])
        protected = torch.cat(
            (
                protected.expand(batch + protected.shape[-2:]),
                jacobian.expand(batch + jacobian.shape[-2:]),
            ),
            dim=-2,
        )
    return tuple(results)


def _integer_tuple(name: str, values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if any(value < 0 for value in result):
        raise ValueError(f"{name} must contain only nonnegative indices")
    return result


def _configuration_indices_for_velocity_rows(
    *,
    selected_source_velocity_indices: tuple[int, ...],
    floating_base: bool,
    joint_configuration_indices: tuple[int, ...],
    joint_configuration_sizes: tuple[int, ...],
    joint_velocity_indices: tuple[int, ...],
    joint_velocity_sizes: tuple[int, ...],
    configuration_dim: int,
) -> tuple[int, ...]:
    span_lengths = {
        len(joint_configuration_indices),
        len(joint_configuration_sizes),
        len(joint_velocity_indices),
        len(joint_velocity_sizes),
    }
    if len(span_lengths) != 1:
        raise ValueError("joint configuration and velocity spans must align")

    mapped: list[int] = []
    for source_velocity in selected_source_velocity_indices:
        # This is the documented controlled PostureTask approximation, not an
        # SE(3) logarithm: v offsets 0..5 read q differences 0..5 directly.
        if floating_base and source_velocity < 6:
            configuration_index = source_velocity
        else:
            matching_spans = [
                span_index
                for span_index, (velocity_start, velocity_size) in enumerate(
                    zip(joint_velocity_indices, joint_velocity_sizes, strict=True)
                )
                if velocity_start <= source_velocity < velocity_start + velocity_size
            ]
            if len(matching_spans) != 1:
                raise ValueError(
                    "each selected non-root velocity must belong to exactly one "
                    "model-derived joint span"
                )
            span_index = matching_spans[0]
            q_size = joint_configuration_sizes[span_index]
            v_size = joint_velocity_sizes[span_index]
            if q_size != v_size:
                raise ValueError(
                    "non-root posture coordinates require equal q/v span sizes; "
                    "manifold joints need an explicit posture error mapping"
                )
            configuration_index = (
                joint_configuration_indices[span_index]
                + source_velocity
                - joint_velocity_indices[span_index]
            )
        if configuration_index >= configuration_dim:
            raise ValueError("mapped posture coordinate exceeds configuration size")
        mapped.append(configuration_index)
    return tuple(mapped)


def cpu_posture_task_rows(
    current_q: torch.Tensor,
    target_q: torch.Tensor,
    *,
    floating_base: bool,
    joint_configuration_indices: Sequence[int],
    joint_configuration_sizes: Sequence[int],
    joint_velocity_indices: Sequence[int],
    joint_velocity_sizes: Sequence[int],
    active_velocity_indices: Sequence[int],
    selected_source_velocity_indices: Sequence[int],
    selected_weights: torch.Tensor | Sequence[float],
    gain: torch.Tensor | float,
) -> PostureTaskRows:
    """Build CPU-faithful controlled-posture rows for arbitrary model metadata.

    Rows follow ``selected_source_velocity_indices`` while columns follow
    ``active_velocity_indices``.  Both orders are therefore explicit and need
    not match.  Joint weights multiply both the configuration error and the
    selection Jacobian; ``gain`` multiplies only the resulting velocity goal.
    Leading dimensions are treated as batch dimensions.
    """

    if not isinstance(current_q, torch.Tensor) or not isinstance(target_q, torch.Tensor):
        raise TypeError("current_q and target_q must be Torch tensors")
    if current_q.ndim < 1 or target_q.ndim < 1:
        raise ValueError("configuration tensors must have at least one dimension")
    if current_q.shape[-1] != target_q.shape[-1]:
        raise ValueError("current_q and target_q configuration sizes must match")
    if current_q.device != target_q.device:
        raise ValueError("current_q and target_q must use the same device")
    if current_q.dtype != target_q.dtype:
        raise ValueError("current_q and target_q must use the same dtype")
    if not current_q.is_floating_point():
        raise ValueError("configuration tensors must use a floating dtype")

    active = _integer_tuple("active_velocity_indices", active_velocity_indices)
    selected = _integer_tuple("selected_source_velocity_indices", selected_source_velocity_indices)
    if len(set(active)) != len(active):
        raise ValueError("active_velocity_indices must be unique")
    if len(set(selected)) != len(selected):
        raise ValueError("selected_source_velocity_indices must be unique")
    if not active:
        raise ValueError("at least one active velocity is required")
    if any(source_velocity not in active for source_velocity in selected):
        raise ValueError("selected posture velocities must be active")

    q_starts = _integer_tuple("joint_configuration_indices", joint_configuration_indices)
    q_sizes = _integer_tuple("joint_configuration_sizes", joint_configuration_sizes)
    v_starts = _integer_tuple("joint_velocity_indices", joint_velocity_indices)
    v_sizes = _integer_tuple("joint_velocity_sizes", joint_velocity_sizes)
    q_indices = _configuration_indices_for_velocity_rows(
        selected_source_velocity_indices=selected,
        floating_base=bool(floating_base),
        joint_configuration_indices=q_starts,
        joint_configuration_sizes=q_sizes,
        joint_velocity_indices=v_starts,
        joint_velocity_sizes=v_sizes,
        configuration_dim=current_q.shape[-1],
    )

    try:
        current, target = torch.broadcast_tensors(current_q, target_q)
    except RuntimeError as error:
        raise ValueError("current_q and target_q batches are not broadcastable") from error

    row_count = len(selected)
    if row_count == 0:
        empty_shape = current.shape[:-1] + (0,)
        empty = current.new_empty(empty_shape)
        jacobian = current.new_zeros(current.shape[:-1] + (0, len(active)))
        return PostureTaskRows(empty, empty, jacobian)

    q_index_tensor = torch.as_tensor(q_indices, device=current.device)
    raw_error = torch.index_select(target - current, -1, q_index_tensor)
    weights = torch.as_tensor(selected_weights, dtype=current.dtype, device=current.device)
    if weights.ndim == 0 or weights.shape[-1] != row_count:
        raise ValueError("selected_weights must end with the selected row count")
    try:
        raw_error, weights = torch.broadcast_tensors(raw_error, weights)
    except RuntimeError as error:
        raise ValueError(
            "selected_weights are not broadcastable with configuration batches"
        ) from error
    weighted_error = raw_error * weights

    gain_tensor = torch.as_tensor(gain, dtype=current.dtype, device=current.device)
    try:
        gain_batch = torch.broadcast_to(gain_tensor, weighted_error.shape[:-1])
    except RuntimeError as error:
        raise ValueError("gain must broadcast over posture batches") from error
    goal = weighted_error * gain_batch.unsqueeze(-1)

    selected_tensor = torch.as_tensor(selected, device=current.device)
    active_tensor = torch.as_tensor(active, device=current.device)
    selection = (selected_tensor[:, None] == active_tensor[None, :]).to(dtype=current.dtype)
    jacobian = weights.unsqueeze(-1) * selection
    return PostureTaskRows(weighted_error, goal, jacobian)


def undamped_generalized_inverse(
    matrix: torch.Tensor,
    relative_rank_tolerance: float = 1.0e-6,
) -> torch.Tensor:
    """Return an SVD pseudoinverse with an Eigen-COD-style relative cutoff."""

    if matrix.ndim < 2:
        raise ValueError("matrix must have at least two dimensions")
    if not matrix.is_floating_point():
        raise ValueError("matrix must use a floating dtype")
    if not math.isfinite(relative_rank_tolerance) or relative_rank_tolerance < 0.0:
        raise ValueError("relative_rank_tolerance must be finite and nonnegative")
    left, singular_values, right_transpose = torch.linalg.svd(matrix, full_matrices=False)
    maximum = singular_values.amax(dim=-1, keepdim=True)
    cutoff = relative_rank_tolerance * maximum
    retained = singular_values > cutoff
    safe_values = torch.where(retained, singular_values, torch.ones_like(singular_values))
    inverse_spectrum = torch.where(
        retained, safe_values.reciprocal(), torch.zeros_like(singular_values)
    )
    return (right_transpose.transpose(-2, -1) * inverse_spectrum.unsqueeze(-2)) @ (
        left.transpose(-2, -1)
    )


def undamped_nullspace_projector(
    matrix: torch.Tensor,
    relative_rank_tolerance: float = 1.0e-6,
) -> torch.Tensor:
    """Return ``I - J^+ J`` using the undamped hierarchy pseudoinverse."""

    inverse = undamped_generalized_inverse(matrix, relative_rank_tolerance)
    identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device).expand(
        matrix.shape[:-2] + (matrix.shape[-1], matrix.shape[-1])
    )
    return identity - inverse @ matrix


def directional_srinv(
    matrix: torch.Tensor,
    tolerance: float,
    damping: float,
) -> torch.Tensor:
    """Apply the same two-layer extended directional SRINV spectrum as EmbodiK."""

    if matrix.ndim < 2:
        raise ValueError("matrix must have at least two dimensions")
    if not matrix.is_floating_point():
        raise ValueError("matrix must use a floating dtype")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not math.isfinite(damping) or damping < 0.0:
        raise ValueError("damping must be finite and nonnegative")

    gram = matrix @ matrix.transpose(-2, -1)
    threshold_squared = tolerance * tolerance
    determinant = torch.linalg.det(gram)
    global_regularization = torch.where(
        determinant < threshold_squared,
        (1.0 - (determinant / threshold_squared) ** 2) * threshold_squared,
        torch.zeros_like(determinant),
    )
    left, singular_values, right_transpose = torch.linalg.svd(matrix, full_matrices=False)
    normalized = torch.clamp(singular_values / tolerance, max=1.0)
    per_value_damping = damping * torch.clamp(1.0 - normalized.square(), min=0.0)
    denominator = singular_values.square() + global_regularization.unsqueeze(-1) + per_value_damping
    inverse_spectrum = singular_values / denominator
    return (right_transpose.transpose(-2, -1) * inverse_spectrum.unsqueeze(-2)) @ (
        left.transpose(-2, -1)
    )


def _broadcast_vector(value: torch.Tensor, batch: tuple[int, ...], width: int) -> torch.Tensor:
    if value.ndim < 1 or value.shape[-1] != width:
        raise ValueError(f"vector must end with width {width}")
    return torch.broadcast_to(value, batch + (width,))


def _broadcast_matrix(
    value: torch.Tensor,
    batch: tuple[int, ...],
    rows: int,
    columns: int,
) -> torch.Tensor:
    if value.ndim < 2 or value.shape[-2:] != (rows, columns):
        raise ValueError(f"matrix must end with shape ({rows}, {columns})")
    return torch.broadcast_to(value, batch + (rows, columns))


def secondary_priority_velocity(
    primary_velocity: torch.Tensor,
    primary_jacobian: torch.Tensor,
    secondary_jacobian: torch.Tensor,
    secondary_goal: torch.Tensor,
    lower_velocity: torch.Tensor,
    upper_velocity: torch.Tensor,
    *,
    srinv_tolerance: float,
    srinv_damping: float,
    relative_rank_tolerance: float = 1.0e-6,
    locked_velocity_mask: torch.Tensor | None = None,
    validate_values: bool = True,
) -> SecondaryPriorityResult:
    """Add one box-limited secondary task without rescaling primary motion.

    The hierarchy projector is deliberately undamped.  Damping is used only
    for the projected secondary task through the extended directional SRINV.
    The final scalar line search acts on the secondary delta alone.
    """

    tensors = (
        primary_velocity,
        primary_jacobian,
        secondary_jacobian,
        secondary_goal,
        lower_velocity,
        upper_velocity,
    )
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("all velocity and task inputs must be Torch tensors")
    reference = primary_velocity
    if reference.ndim < 1 or not reference.is_floating_point():
        raise ValueError("primary_velocity must be a floating vector")
    if any(value.device != reference.device for value in tensors):
        raise ValueError("all inputs must use the same device")
    if any(value.dtype != reference.dtype for value in tensors):
        raise ValueError("all inputs must use the same dtype")

    velocity_dim = reference.shape[-1]
    if primary_jacobian.ndim < 2 or primary_jacobian.shape[-1] != velocity_dim:
        raise ValueError("primary_jacobian width must match velocity dimension")
    if secondary_jacobian.ndim < 2 or secondary_jacobian.shape[-1] != velocity_dim:
        raise ValueError("secondary_jacobian width must match velocity dimension")
    primary_rows = primary_jacobian.shape[-2]
    secondary_rows = secondary_jacobian.shape[-2]
    if secondary_goal.ndim < 1 or secondary_goal.shape[-1] != secondary_rows:
        raise ValueError("secondary_goal width must match secondary rows")

    try:
        batch = torch.broadcast_shapes(
            reference.shape[:-1],
            primary_jacobian.shape[:-2],
            secondary_jacobian.shape[:-2],
            secondary_goal.shape[:-1],
            lower_velocity.shape[:-1],
            upper_velocity.shape[:-1],
        )
    except RuntimeError as error:
        raise ValueError("secondary-priority input batches are not broadcastable") from error

    primary_velocity = _broadcast_vector(reference, batch, velocity_dim)
    lower_velocity = _broadcast_vector(lower_velocity, batch, velocity_dim)
    upper_velocity = _broadcast_vector(upper_velocity, batch, velocity_dim)
    primary_jacobian = _broadcast_matrix(primary_jacobian, batch, primary_rows, velocity_dim)
    secondary_jacobian = _broadcast_matrix(secondary_jacobian, batch, secondary_rows, velocity_dim)
    secondary_goal = _broadcast_vector(secondary_goal, batch, secondary_rows)
    if validate_values:
        if torch.any(lower_velocity > upper_velocity):
            raise ValueError("lower_velocity must not exceed upper_velocity")
        if torch.any((primary_velocity < lower_velocity) | (primary_velocity > upper_velocity)):
            raise ValueError("primary_velocity must already lie inside the velocity box")

    # Locks belong in the nullspace calculation. Zeroing a locked coordinate
    # only after projection can destroy cancellation in a higher-priority row.
    mask = torch.zeros_like(primary_velocity, dtype=torch.bool)
    if locked_velocity_mask is not None:
        if not isinstance(locked_velocity_mask, torch.Tensor):
            raise TypeError("locked_velocity_mask must be a Torch tensor")
        if locked_velocity_mask.device != reference.device:
            raise ValueError("locked_velocity_mask must use the same device")
        if locked_velocity_mask.dtype is not torch.bool:
            raise ValueError("locked_velocity_mask must be boolean")
        try:
            mask = torch.broadcast_to(locked_velocity_mask, batch + (velocity_dim,))
        except RuntimeError as error:
            raise ValueError(
                "locked_velocity_mask is not broadcastable to the velocity shape"
            ) from error
    # Solve in free coordinates; embed the result back into the full tangent.
    free = (~mask).to(reference.dtype)
    projector = undamped_nullspace_projector(
        primary_jacobian * free.unsqueeze(-2), relative_rank_tolerance
    )
    projector = projector * free.unsqueeze(-1) * free.unsqueeze(-2)
    residual = secondary_goal - (secondary_jacobian @ primary_velocity.unsqueeze(-1)).squeeze(-1)
    projected_jacobian = secondary_jacobian @ projector
    projected_inverse = directional_srinv(
        projected_jacobian,
        tolerance=srinv_tolerance,
        damping=srinv_damping,
    )
    tangent_delta = (projected_inverse @ residual.unsqueeze(-1)).squeeze(-1)
    delta = (projector @ tangent_delta.unsqueeze(-1)).squeeze(-1)

    positive = delta > 0.0
    negative = delta < 0.0
    available = torch.where(
        positive,
        upper_velocity - primary_velocity,
        primary_velocity - lower_velocity,
    )
    ratios = torch.where(
        positive | negative,
        available / torch.clamp(delta.abs(), min=torch.finfo(delta.dtype).tiny),
        torch.full_like(delta, torch.inf),
    )
    scale = torch.clamp(ratios.amin(dim=-1), min=0.0, max=1.0)
    scaled_delta = scale.unsqueeze(-1) * delta
    velocity = primary_velocity + scaled_delta

    primary_change = (primary_jacobian @ scaled_delta.unsqueeze(-1)).squeeze(-1)
    secondary_after_vector = secondary_goal - (secondary_jacobian @ velocity.unsqueeze(-1)).squeeze(
        -1
    )
    delta_norm = torch.linalg.vector_norm(scaled_delta, dim=-1)
    return SecondaryPriorityResult(
        velocity=velocity,
        primary_residual_increase=torch.linalg.vector_norm(primary_change, dim=-1),
        secondary_residual_before=torch.linalg.vector_norm(residual, dim=-1),
        secondary_residual_after=torch.linalg.vector_norm(secondary_after_vector, dim=-1),
        delta_norm=delta_norm,
        applied=(scale > 0.0) & (delta_norm > 0.0),
    )
