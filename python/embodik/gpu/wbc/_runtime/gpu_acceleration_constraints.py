"""Vectorized acceleration-level constraint shaping for GPU WBC.

The state-box construction is a Torch transcription of
``cpp_core/src/acceleration_state_box.cpp``.  It deliberately keeps the C++
comparison tolerances, braking-envelope checks, continuous-path bound, stable
quadratic viability root, and final interval canonicalization.  Tensor values
are float64 so CPU and GPU assembly use the same numerical policy.

All numerical failures are reported as fixed-shape tensor flags.  A failed
world is also returned with an empty acceleration interval, which makes the
helper fail closed when consumed without inspecting the flags first.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

FINAL_INTERVAL_CANONICALIZATION_TOLERANCE = 1.0e-10
CENTROIDAL_UNBOUNDED_LIMIT = 1.0e100


class AccelerationStateBoxResult(NamedTuple):
    """Fixed-shape state-box result.

    ``lower`` and ``upper`` have shape ``[..., nv]``.  ``finite`` and
    ``feasible`` have shape ``[...]`` and therefore report one status per
    independently solved world.  ``finite`` covers inputs, every active
    derived bound, and the returned interval.  ``feasible`` additionally
    covers specification validity, current-state/braking admissibility, and a
    nonempty final interval.
    """

    lower: torch.Tensor
    upper: torch.Tensor
    finite: torch.Tensor
    feasible: torch.Tensor


def _comparison_tolerance(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return 1.0e-12 * (1.0 + torch.maximum(torch.abs(lhs), torch.abs(rhs)))


def _float_tensor(value, *, like: torch.Tensor, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float64, device=like.device)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return torch.full((), float(value), dtype=torch.float64, device=like.device)
    raise TypeError(f"{name} must be a real scalar or torch.Tensor")


def _bool_tensor(value, *, like: torch.Tensor, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.bool, device=like.device)
    if isinstance(value, bool):
        return torch.full((), value, dtype=torch.bool, device=like.device)
    raise TypeError(f"{name} must be a bool or torch.Tensor")


def _broadcast_coordinate(value, shape: torch.Size, *, name: str) -> torch.Tensor:
    try:
        return torch.broadcast_to(value, shape)
    except RuntimeError as error:
        raise ValueError(f"{name} is not broadcastable to state shape {tuple(shape)}") from error


def _broadcast_world(value, batch_shape: torch.Size, *, name: str) -> torch.Tensor:
    # A trailing singleton is convenient when callers retain a coordinate axis
    # for all state data, but dt remains one scalar per world.
    if value.ndim == len(batch_shape) + 1 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    try:
        return torch.broadcast_to(value, batch_shape)
    except RuntimeError as error:
        raise ValueError(
            f"{name} is not broadcastable to world shape {tuple(batch_shape)}"
        ) from error


def _continuous_acceleration_upper(
    margin: torch.Tensor,
    outward_rate: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    endpoint = 2.0 * (margin - outward_rate * dt) / (dt * dt)
    endpoint_branch = (outward_rate <= 0.0) | (2.0 * margin >= outward_rate * dt)
    safe_margin = torch.where(margin != 0.0, margin, torch.ones_like(margin))
    interior = -(outward_rate * outward_rate) / (2.0 * safe_margin)
    upper = torch.where(endpoint_branch, endpoint, torch.minimum(endpoint, interior))
    finite = torch.isfinite(endpoint) & (endpoint_branch | torch.isfinite(interior))
    return upper, finite


def _viability_acceleration_upper(
    margin: torch.Tensor,
    outward_rate: torch.Tensor,
    braking_acceleration: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dt_squared = dt * dt
    linear = 2.0 * outward_rate * dt + braking_acceleration * dt_squared
    constant = (
        outward_rate * outward_rate
        + 2.0 * braking_acceleration * outward_rate * dt
        - 2.0 * braking_acceleration * margin
    )
    reverse_rate_bound = -outward_rate / dt
    discriminant = linear * linear - 4.0 * dt_squared * constant
    tolerance = _comparison_tolerance(linear * linear, 4.0 * dt_squared * constant)
    valid_discriminant = torch.isfinite(discriminant) & (discriminant >= -tolerance)
    square_root = torch.sqrt(torch.clamp_min(discriminant, 0.0))

    denominator = 2.0 * dt_squared
    default_root = -linear / denominator
    stable_numerator = -0.5 * (linear + torch.copysign(square_root, linear))
    first_root = stable_numerator / dt_squared
    safe_numerator = torch.where(
        stable_numerator != 0.0,
        stable_numerator,
        torch.ones_like(stable_numerator),
    )
    second_root = torch.where(
        stable_numerator != 0.0,
        constant / safe_numerator,
        first_root,
    )
    stable_root = torch.maximum(first_root, second_root)
    use_stable_root = (square_root > 0.0) | (linear != 0.0)
    upper_root = torch.where(use_stable_root, stable_root, default_root)
    upper = torch.maximum(reverse_rate_bound, upper_root)
    finite = (
        torch.isfinite(linear)
        & torch.isfinite(constant)
        & torch.isfinite(reverse_rate_bound)
        & valid_discriminant
        & torch.isfinite(upper_root)
        & torch.isfinite(upper)
    )
    return upper, finite


def acceleration_state_box_bounds(
    value: torch.Tensor,
    rate: torch.Tensor,
    dt: torch.Tensor | float,
    *,
    state_lower: torch.Tensor | float = 0.0,
    state_upper: torch.Tensor | float = 0.0,
    rate_lower: torch.Tensor | float = 0.0,
    rate_upper: torch.Tensor | float = 0.0,
    acceleration_lower: torch.Tensor | float = 0.0,
    acceleration_upper: torch.Tensor | float = 0.0,
    lower_braking_acceleration: torch.Tensor | float = 0.0,
    upper_braking_acceleration: torch.Tensor | float = 0.0,
    state_lower_active: torch.Tensor | bool = False,
    state_upper_active: torch.Tensor | bool = False,
    rate_lower_active: torch.Tensor | bool = False,
    rate_upper_active: torch.Tensor | bool = False,
    acceleration_lower_active: torch.Tensor | bool = True,
    acceleration_upper_active: torch.Tensor | bool = True,
) -> AccelerationStateBoxResult:
    """Intersect CPU-faithful acceleration state-box bounds.

    ``value`` and ``rate`` identify the trailing generalized-coordinate axis;
    all state/bound/authority fields broadcast to ``[..., nv]``. ``dt`` is a
    scalar or broadcasts over only the leading world dimensions.  Active masks
    may be scalar, per-DoF, or per-world/per-DoF.

    As in C++, stored lower/upper values and braking authorities must be finite
    even when their corresponding bound is inactive.  Both final acceleration
    sides must become active.  Out-of-bounds state, outward boundary motion,
    a violated braking envelope, invalid specification, non-finite arithmetic,
    or a genuinely empty intersection fails the complete world closed.
    """

    if not isinstance(value, torch.Tensor) or not isinstance(rate, torch.Tensor):
        raise TypeError("value and rate must be torch.Tensor instances")
    value = value.to(dtype=torch.float64)
    rate = rate.to(dtype=torch.float64, device=value.device)
    if value.ndim < 1 or value.shape[-1] < 1:
        raise ValueError("value must have shape [..., nv] with nv >= 1")
    if rate.ndim < 1 or rate.shape[-1] != value.shape[-1]:
        raise ValueError("rate must have shape [..., nv] with the same nv as value")
    try:
        coordinate_shape = torch.broadcast_shapes(value.shape, rate.shape)
    except RuntimeError as error:
        raise ValueError("value and rate world dimensions are not broadcastable") from error
    batch_shape = coordinate_shape[:-1]
    value = torch.broadcast_to(value, coordinate_shape)
    rate = torch.broadcast_to(rate, coordinate_shape)

    float_fields = {}
    for name, field in (
        ("state_lower", state_lower),
        ("state_upper", state_upper),
        ("rate_lower", rate_lower),
        ("rate_upper", rate_upper),
        ("acceleration_lower", acceleration_lower),
        ("acceleration_upper", acceleration_upper),
        ("lower_braking_acceleration", lower_braking_acceleration),
        ("upper_braking_acceleration", upper_braking_acceleration),
    ):
        tensor = _float_tensor(field, like=value, name=name)
        float_fields[name] = _broadcast_coordinate(tensor, coordinate_shape, name=name)

    active_fields = {}
    for name, field in (
        ("state_lower_active", state_lower_active),
        ("state_upper_active", state_upper_active),
        ("rate_lower_active", rate_lower_active),
        ("rate_upper_active", rate_upper_active),
        ("acceleration_lower_active", acceleration_lower_active),
        ("acceleration_upper_active", acceleration_upper_active),
    ):
        tensor = _bool_tensor(field, like=value, name=name)
        active_fields[name] = _broadcast_coordinate(tensor, coordinate_shape, name=name)

    time_step = _broadcast_world(_float_tensor(dt, like=value, name="dt"), batch_shape, name="dt")
    dt_coordinate = time_step.unsqueeze(-1)
    safe_dt = torch.where(dt_coordinate > 0.0, dt_coordinate, torch.ones_like(dt_coordinate))

    q_lower = float_fields["state_lower"]
    q_upper = float_fields["state_upper"]
    dq_lower = float_fields["rate_lower"]
    dq_upper = float_fields["rate_upper"]
    ddq_lower = float_fields["acceleration_lower"]
    ddq_upper = float_fields["acceleration_upper"]
    brake_lower = float_fields["lower_braking_acceleration"]
    brake_upper = float_fields["upper_braking_acceleration"]
    q_lower_on = active_fields["state_lower_active"]
    q_upper_on = active_fields["state_upper_active"]
    dq_lower_on = active_fields["rate_lower_active"]
    dq_upper_on = active_fields["rate_upper_active"]
    ddq_lower_on = active_fields["acceleration_lower_active"]
    ddq_upper_on = active_fields["acceleration_upper_active"]

    all_float_inputs = torch.stack(
        (
            value,
            rate,
            q_lower,
            q_upper,
            dq_lower,
            dq_upper,
            ddq_lower,
            ddq_upper,
            brake_lower,
            brake_upper,
        ),
        dim=0,
    )
    coordinate_finite = torch.isfinite(all_float_inputs).all(dim=0)
    input_finite = coordinate_finite.all(dim=-1) & torch.isfinite(time_step)

    ordered = (
        (~(q_lower_on & q_upper_on) | (q_lower <= q_upper))
        & (~(dq_lower_on & dq_upper_on) | (dq_lower <= dq_upper))
        & (~(ddq_lower_on & ddq_upper_on) | (ddq_lower <= ddq_upper))
    )
    lower_authority_consistent = (~q_lower_on) | (
        (brake_lower > 0.0)
        & (
            (~ddq_upper_on)
            | (ddq_upper + _comparison_tolerance(ddq_upper, brake_lower) >= brake_lower)
        )
    )
    upper_authority_consistent = (~q_upper_on) | (
        (brake_upper > 0.0)
        & (
            (~ddq_lower_on)
            | (-ddq_lower + _comparison_tolerance(-ddq_lower, brake_upper) >= brake_upper)
        )
    )
    specification_valid = (
        (time_step > 0.0)
        & ordered.all(dim=-1)
        & lower_authority_consistent.all(dim=-1)
        & upper_authority_consistent.all(dim=-1)
    )

    negative_infinity = torch.full_like(value, -torch.inf)
    positive_infinity = torch.full_like(value, torch.inf)
    lower = torch.where(ddq_lower_on, ddq_lower, negative_infinity)
    upper = torch.where(ddq_upper_on, ddq_upper, positive_infinity)
    lower_active = ddq_lower_on.clone()
    upper_active = ddq_upper_on.clone()
    derived_finite = torch.ones_like(value, dtype=torch.bool)

    rate_lower_candidate = (dq_lower - rate) / safe_dt
    rate_upper_candidate = (dq_upper - rate) / safe_dt
    derived_finite &= (~dq_lower_on) | torch.isfinite(rate_lower_candidate)
    derived_finite &= (~dq_upper_on) | torch.isfinite(rate_upper_candidate)
    lower = torch.where(dq_lower_on, torch.maximum(lower, rate_lower_candidate), lower)
    upper = torch.where(dq_upper_on, torch.minimum(upper, rate_upper_candidate), upper)
    lower_active |= dq_lower_on
    upper_active |= dq_upper_on

    bounded_value = torch.where(q_lower_on, torch.maximum(value, q_lower), value)
    bounded_value = torch.where(q_upper_on, torch.minimum(bounded_value, q_upper), bounded_value)
    lower_margin = bounded_value - q_lower
    upper_margin = q_upper - bounded_value

    outside_lower = q_lower_on & (q_lower > value + _comparison_tolerance(q_lower, value))
    outside_upper = q_upper_on & (value > q_upper + _comparison_tolerance(value, q_upper))
    outward_lower = q_lower_on & (lower_margin <= 0.0) & (rate < 0.0)
    outward_upper = q_upper_on & (upper_margin <= 0.0) & (rate > 0.0)
    lower_rate_squared = rate * rate
    lower_braking_distance = 2.0 * brake_lower * lower_margin
    upper_braking_distance = 2.0 * brake_upper * upper_margin
    outside_lower_braking = (
        q_lower_on
        & (rate < 0.0)
        & (
            lower_rate_squared
            > lower_braking_distance
            + _comparison_tolerance(lower_rate_squared, lower_braking_distance)
        )
    )
    outside_upper_braking = (
        q_upper_on
        & (rate > 0.0)
        & (
            lower_rate_squared
            > upper_braking_distance
            + _comparison_tolerance(lower_rate_squared, upper_braking_distance)
        )
    )

    safe_lower_margin = torch.clamp_min(lower_margin, 0.0)
    lower_state_rate = (
        torch.maximum(
            (q_lower - bounded_value) / safe_dt,
            -torch.sqrt(2.0 * brake_lower * safe_lower_margin),
        )
        - rate
    ) / safe_dt
    lower_endpoint = (q_lower - bounded_value - rate * safe_dt) / (0.5 * safe_dt * safe_dt)
    lower_continuous_upper, lower_continuous_finite = _continuous_acceleration_upper(
        lower_margin, -rate, safe_dt
    )
    lower_viability_upper, lower_viability_finite = _viability_acceleration_upper(
        lower_margin, -rate, brake_lower, safe_dt
    )
    for candidate, candidate_finite in (
        (lower_state_rate, torch.isfinite(lower_state_rate)),
        (lower_endpoint, torch.isfinite(lower_endpoint)),
        (-lower_continuous_upper, lower_continuous_finite),
        (-lower_viability_upper, lower_viability_finite),
    ):
        derived_finite &= (~q_lower_on) | candidate_finite
        lower = torch.where(q_lower_on, torch.maximum(lower, candidate), lower)
    lower_active |= q_lower_on

    safe_upper_margin = torch.clamp_min(upper_margin, 0.0)
    upper_state_rate = (
        torch.minimum(
            (q_upper - bounded_value) / safe_dt,
            torch.sqrt(2.0 * brake_upper * safe_upper_margin),
        )
        - rate
    ) / safe_dt
    upper_endpoint = (q_upper - bounded_value - rate * safe_dt) / (0.5 * safe_dt * safe_dt)
    upper_continuous, upper_continuous_finite = _continuous_acceleration_upper(
        upper_margin, rate, safe_dt
    )
    upper_viability, upper_viability_finite = _viability_acceleration_upper(
        upper_margin, rate, brake_upper, safe_dt
    )
    for candidate, candidate_finite in (
        (upper_state_rate, torch.isfinite(upper_state_rate)),
        (upper_endpoint, torch.isfinite(upper_endpoint)),
        (upper_continuous, upper_continuous_finite),
        (upper_viability, upper_viability_finite),
    ):
        derived_finite &= (~q_upper_on) | candidate_finite
        upper = torch.where(q_upper_on, torch.minimum(upper, candidate), upper)
    upper_active |= q_upper_on

    interval_difference = lower - upper
    canonical_tolerance = torch.maximum(
        torch.full_like(lower, FINAL_INTERVAL_CANONICALIZATION_TOLERANCE),
        _comparison_tolerance(lower, upper),
    )
    canonicalize = (lower > upper) & (interval_difference <= canonical_tolerance)
    midpoint = 0.5 * (lower + upper)
    lower = torch.where(canonicalize, midpoint, lower)
    upper = torch.where(canonicalize, midpoint, upper)
    empty_interval = (lower > upper) & ~canonicalize

    admissible_state = ~(
        outside_lower
        | outside_upper
        | outward_lower
        | outward_upper
        | outside_lower_braking
        | outside_upper_braking
    )
    output_finite = torch.isfinite(lower) & torch.isfinite(upper)
    finite = input_finite & derived_finite.all(dim=-1) & output_finite.all(dim=-1)
    feasible = (
        finite
        & specification_valid
        & admissible_state.all(dim=-1)
        & lower_active.all(dim=-1)
        & upper_active.all(dim=-1)
        & (~empty_interval).all(dim=-1)
    )

    failed = ~feasible.unsqueeze(-1)
    lower = torch.where(failed, torch.full_like(lower, torch.inf), lower)
    upper = torch.where(failed, torch.full_like(upper, -torch.inf), upper)
    return AccelerationStateBoxResult(lower, upper, finite, feasible)


def acceleration_capture_point_rows(
    halfspace_normals: torch.Tensor,
    halfspace_offsets: torch.Tensor,
    com_position_xy: torch.Tensor,
    com_velocity_xy: torch.Tensor,
    com_jacobian_xy: torch.Tensor,
    com_acceleration_bias_xy: torch.Tensor,
    dt: torch.Tensor | float,
    omega: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build acceleration capture-point rows from the CPU affine equation.

    The result encodes ``lower <= rows @ ddq <= upper`` for the predicted
    capture point after one constant-acceleration step. Coordinates must
    already be expressed in the configured support frame. Numeric settings are
    expected to be host-validated before graph capture.
    """

    if not isinstance(halfspace_normals, torch.Tensor):
        raise TypeError("halfspace_normals must be a torch.Tensor")
    normals = halfspace_normals.to(dtype=torch.float64)
    device = normals.device
    offsets = _float_tensor(halfspace_offsets, like=normals, name="halfspace_offsets").to(
        device=device
    )
    position = _float_tensor(com_position_xy, like=normals, name="com_position_xy").to(
        device=device
    )
    velocity = _float_tensor(com_velocity_xy, like=normals, name="com_velocity_xy").to(
        device=device
    )
    jacobian = _float_tensor(com_jacobian_xy, like=normals, name="com_jacobian_xy").to(
        device=device
    )
    bias = _float_tensor(
        com_acceleration_bias_xy, like=normals, name="com_acceleration_bias_xy"
    ).to(device=device)
    time_step = _float_tensor(dt, like=normals, name="dt")
    frequency = _float_tensor(omega, like=normals, name="omega")
    if normals.ndim < 2 or normals.shape[-1] != 2 or normals.shape[-2] < 1:
        raise ValueError("halfspace_normals must have shape [..., m, 2] with m >= 1")
    row_count = normals.shape[-2]
    if offsets.ndim < 1 or offsets.shape[-1] != row_count:
        raise ValueError("halfspace_offsets must have shape [..., m]")
    for name, tensor in (
        ("com_position_xy", position),
        ("com_velocity_xy", velocity),
        ("com_acceleration_bias_xy", bias),
    ):
        if tensor.ndim < 1 or tensor.shape[-1] != 2:
            raise ValueError(f"{name} must have shape [..., 2]")
    if jacobian.ndim < 2 or jacobian.shape[-2] != 2 or jacobian.shape[-1] < 1:
        raise ValueError("com_jacobian_xy must have shape [..., 2, nv] with nv >= 1")
    batch_shape = torch.broadcast_shapes(
        normals.shape[:-2],
        offsets.shape[:-1],
        position.shape[:-1],
        velocity.shape[:-1],
        jacobian.shape[:-2],
        bias.shape[:-1],
        time_step.shape,
        frequency.shape,
    )
    velocity_dim = jacobian.shape[-1]
    normals = torch.broadcast_to(normals, batch_shape + (row_count, 2))
    offsets = torch.broadcast_to(offsets, batch_shape + (row_count,))
    position = torch.broadcast_to(position, batch_shape + (2,))
    velocity = torch.broadcast_to(velocity, batch_shape + (2,))
    jacobian = torch.broadcast_to(jacobian, batch_shape + (2, velocity_dim))
    bias = torch.broadcast_to(bias, batch_shape + (2,))
    time_step = torch.broadcast_to(time_step, batch_shape)
    frequency = torch.broadcast_to(frequency, batch_shape)

    scale = 0.5 * time_step * time_step + time_step / frequency
    rows = scale[..., None, None] * torch.matmul(normals, jacobian)
    constant_point = (
        position
        + time_step[..., None] * velocity
        + velocity / frequency[..., None]
        + scale[..., None] * bias
    )
    upper = offsets - torch.matmul(normals, constant_point.unsqueeze(-1)).squeeze(-1)
    lower = torch.full_like(upper, -CENTROIDAL_UNBOUNDED_LIMIT)
    return rows, lower, upper


def acceleration_zmp_constraint_rows(
    halfspace_normals: torch.Tensor,
    halfspace_offsets: torch.Tensor,
    com_position: torch.Tensor,
    centroidal_momentum_matrix: torch.Tensor,
    centroidal_momentum_bias: torch.Tensor,
    gravity: torch.Tensor,
    total_mass: torch.Tensor | float,
    fz_min: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build physical centroidal-rate ZMP acceleration rows.

    This folds the affine bias from
    ``hdot = Ag @ ddq + centroidal_momentum_bias`` into the returned interval.
    Row zero is the CPU vertical-force floor; the remaining rows are its exact
    cross-multiplied support-polygon inequalities. All vectors and matrices
    must already be expressed in the configured support frame.
    """

    if not isinstance(halfspace_normals, torch.Tensor):
        raise TypeError("halfspace_normals must be a torch.Tensor")
    normals = halfspace_normals.to(dtype=torch.float64)
    device = normals.device
    offsets = _float_tensor(halfspace_offsets, like=normals, name="halfspace_offsets").to(
        device=device
    )
    com = _float_tensor(com_position, like=normals, name="com_position").to(device=device)
    ag = _float_tensor(
        centroidal_momentum_matrix,
        like=normals,
        name="centroidal_momentum_matrix",
    ).to(device=device)
    bias = _float_tensor(
        centroidal_momentum_bias,
        like=normals,
        name="centroidal_momentum_bias",
    ).to(device=device)
    support_gravity = _float_tensor(gravity, like=normals, name="gravity").to(device=device)
    mass = _float_tensor(total_mass, like=normals, name="total_mass")
    force_floor = _float_tensor(fz_min, like=normals, name="fz_min")
    if normals.ndim < 2 or normals.shape[-1] != 2 or normals.shape[-2] < 1:
        raise ValueError("halfspace_normals must have shape [..., m, 2] with m >= 1")
    row_count = normals.shape[-2]
    if offsets.ndim < 1 or offsets.shape[-1] != row_count:
        raise ValueError("halfspace_offsets must have shape [..., m]")
    if com.ndim < 1 or com.shape[-1] != 3:
        raise ValueError("com_position must have shape [..., 3]")
    if ag.ndim < 2 or ag.shape[-2] != 6 or ag.shape[-1] < 1:
        raise ValueError("centroidal_momentum_matrix must have shape [..., 6, nv]")
    velocity_dim = ag.shape[-1]
    if bias.ndim < 1 or bias.shape[-1] != 6:
        raise ValueError("centroidal_momentum_bias must have shape [..., 6]")
    if support_gravity.ndim < 1 or support_gravity.shape[-1] != 3:
        raise ValueError("gravity must have shape [..., 3]")
    batch_shape = torch.broadcast_shapes(
        normals.shape[:-2],
        offsets.shape[:-1],
        com.shape[:-1],
        ag.shape[:-2],
        bias.shape[:-1],
        support_gravity.shape[:-1],
        mass.shape,
        force_floor.shape,
    )
    normals = torch.broadcast_to(normals, batch_shape + (row_count, 2))
    offsets = torch.broadcast_to(offsets, batch_shape + (row_count,))
    com = torch.broadcast_to(com, batch_shape + (3,))
    ag = torch.broadcast_to(ag, batch_shape + (6, velocity_dim))
    bias = torch.broadcast_to(bias, batch_shape + (6,))
    support_gravity = torch.broadcast_to(support_gravity, batch_shape + (3,))
    mass = torch.broadcast_to(mass, batch_shape)
    force_floor = torch.broadcast_to(force_floor, batch_shape)

    force_bias = bias[..., :3] - mass[..., None] * support_gravity
    fz_matrix = ag[..., 2, :]
    fz_bias = force_bias[..., 2]
    c_offset = normals[..., 0] * com[..., 0, None] + normals[..., 1] * com[..., 1, None] - offsets
    polygon_rows = (
        c_offset[..., None] * fz_matrix.unsqueeze(-2)
        - normals[..., 0, None] * ag[..., 4, :].unsqueeze(-2)
        - normals[..., 0, None] * com[..., 2, None, None] * ag[..., 0, :].unsqueeze(-2)
        + normals[..., 1, None] * ag[..., 3, :].unsqueeze(-2)
        - normals[..., 1, None] * com[..., 2, None, None] * ag[..., 1, :].unsqueeze(-2)
    )
    polygon_bias = (
        c_offset * fz_bias[..., None]
        - normals[..., 0] * bias[..., 4, None]
        - normals[..., 0] * com[..., 2, None] * force_bias[..., 0, None]
        + normals[..., 1] * bias[..., 3, None]
        - normals[..., 1] * com[..., 2, None] * force_bias[..., 1, None]
    )
    rows = torch.cat((fz_matrix.unsqueeze(-2), polygon_rows), dim=-2)
    lower = torch.full(
        batch_shape + (row_count + 1,),
        -CENTROIDAL_UNBOUNDED_LIMIT,
        dtype=torch.float64,
        device=device,
    )
    upper = torch.full_like(lower, CENTROIDAL_UNBOUNDED_LIMIT)
    lower[..., 0] = force_floor - fz_bias
    upper[..., 1:] = -polygon_bias
    return rows, lower, upper
