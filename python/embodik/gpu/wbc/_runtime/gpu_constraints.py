"""Model-agnostic, fixed-shape Torch helpers for GPU WBC constraints.

The velocity-box policy mirrors EmbodiK's C++ ``KinematicsSolver`` behavior.
All helpers operate on arbitrary leading batch dimensions and do not assume a
robot family, floating-base layout, or number of generalized velocities.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


def support_polygon_halfplanes(vertices_xy, *, margin=0.0, proximity_fraction=0.2):
    """Host-side CPU-compatible convex hull, radial margin and proximity scale.

    Returns unit outward normals A, offsets b (A x <= b), and proximity.
    Rejects degenerate/nonconvex results of excessive radial shrinking.
    """
    import numpy as np

    points = np.asarray(vertices_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("support polygon requires at least three xy vertices")
    if (
        not np.isfinite(points).all()
        or not np.isfinite([margin, proximity_fraction]).all()
    ):
        raise ValueError("support polygon settings must be finite")
    ordered = sorted(set(map(tuple, points)))

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def chain(seq):
        result = []
        for point in seq:
            while len(result) >= 2 and cross(result[-2], result[-1], point) <= 0:
                result.pop()
            result.append(point)
        return result

    hull = np.asarray(chain(ordered)[:-1] + chain(ordered[::-1])[:-1])
    if len(hull) < 3:
        raise ValueError("support polygon is degenerate")
    center = hull.mean(axis=0)
    radii = np.linalg.norm(hull - center, axis=1)
    if margin > 0:
        distance = np.clip(margin, 0, 1) * max(0.0, radii.mean() - 1e-9)
        hull = hull + (center - hull) * (distance / np.maximum(radii, 1e-12))[:, None]
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths < 1e-10):
        raise ValueError("support polygon margin collapses an edge")
    a = np.stack((edges[:, 1], -edges[:, 0]), axis=1) / lengths[:, None]
    b = np.sum(a * hull, axis=1)
    if (
        np.any(b[:, None] - a @ hull.T < -1e-10)
        or np.min(b - a @ hull.mean(axis=0)) <= 1e-10
    ):
        raise ValueError("support polygon margin must leave a convex interior")
    radius = np.min(np.abs(b - a @ hull.mean(axis=0)))
    return a, b, proximity_fraction * radius if proximity_fraction > 0 else float("inf")


def com_halfspace_velocity_bounds(
    slack, dt, vel_max, acc_max, use_acceleration_limits, proximity
):
    """Exact three-layer CPU CoM bounds, including its 0.1 mm dead zone."""
    slack = slack.to(torch.float64)
    dt = (
        dt.to(slack)
        if isinstance(dt, torch.Tensor)
        else torch.full((), dt, dtype=slack.dtype, device=slack.device)
    )
    dt = dt.clamp_min(1e-6)
    velocity = (
        vel_max.to(slack)
        if isinstance(vel_max, torch.Tensor)
        else torch.full((), vel_max, dtype=slack.dtype, device=slack.device)
    )
    positive = slack.clamp_min(0)
    upper = torch.ones_like(slack) * velocity
    enabled = (
        use_acceleration_limits.to(dtype=torch.bool, device=slack.device)
        if isinstance(use_acceleration_limits, torch.Tensor)
        else torch.full(
            (), use_acceleration_limits, dtype=torch.bool, device=slack.device
        )
    )
    upper = torch.where(
        enabled, torch.minimum(upper, torch.sqrt(2 * acc_max * positive)), upper
    )
    upper = torch.where(slack < proximity, torch.minimum(upper, positive / dt), upper)
    recovery = torch.minimum(velocity, ((-slack - 1e-4) / dt * 0.2).clamp_min(0.01))
    return -torch.ones_like(slack) * velocity, torch.where(
        slack < -1e-4, -recovery, upper
    )


# CPU policy constants from cpp_core/src/kinematics_solver.cpp and
# cpp_core/include/embodik/kinematics_solver.hpp.
CPU_MIN_BOUND_FRACTION = 0.10
CPU_MARGIN_THRESHOLD = 0.01
CPU_TORSO_BOUND_SLACK_EPS_TRANS = 1.0e-4
CPU_TORSO_BOUND_SLACK_EPS_ROT = 1.0e-3
CPU_UNBOUNDED_CONSTRAINT_LIMIT = 1.0e10
CPU_LIMIT_RECOVERY_GAIN = 0.5
CPU_LIMIT_RECOVERY_ENTER_EPSILON = 1.0e-4
CPU_LIMIT_RECOVERY_EXIT_EPSILON = 1.0e-4
CPU_LIMIT_EXIT_RELEASE_MARGIN = 0.0
CPU_MINIMUM_DT = 1.0e-9


class CyclicProjectionResult(NamedTuple):
    """Result and fixed-shape diagnostics from cyclic row projection."""

    velocity: torch.Tensor
    feasible: torch.Tensor
    max_row_violation: torch.Tensor
    max_velocity_violation: torch.Tensor
    lower_row_residual: torch.Tensor
    upper_row_residual: torch.Tensor


def _as_float64(value: torch.Tensor | float, *, like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float64, device=like.device)
    # ``as_tensor(Python scalar, device="cuda")`` performs an uncapturable
    # host-to-device copy. ``full`` emits a graph-capturable device fill.
    return torch.full((), float(value), dtype=torch.float64, device=like.device)


def _as_bool(value: torch.Tensor | bool, *, like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.bool, device=like.device)
    return torch.full((), bool(value), dtype=torch.bool, device=like.device)


def _discrete_stopping_velocity_limit(
    margin: torch.Tensor,
    acceleration: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Port of the CPU sampled-data stopping-speed limit."""

    margin_safe = torch.clamp(margin, min=0.0)
    dt_safe = torch.clamp(dt, min=CPU_MINIMUM_DT)
    valid_acceleration = torch.isfinite(acceleration) & (acceleration > 0.0)
    safe_acceleration = torch.where(
        valid_acceleration, acceleration, torch.ones_like(acceleration)
    )
    normalized_margin = margin_safe / (safe_acceleration * dt_safe.square())
    root = 0.5 * (torch.sqrt(1.0 + 8.0 * normalized_margin) - 1.0)
    interval_count = torch.clamp(torch.ceil(root - 1.0e-12), min=1.0)
    normalized_velocity = (
        normalized_margin + 0.5 * interval_count * (interval_count - 1.0)
    ) / interval_count
    sampled_limit = torch.minimum(
        margin_safe / dt_safe,
        safe_acceleration * dt_safe * normalized_velocity,
    )
    fallback = margin_safe / dt_safe
    result = torch.where(valid_acceleration, sampled_limit, fallback)
    return torch.where(margin_safe > 0.0, result, torch.zeros_like(result))


def velocity_box_bounds(
    lower_slack: torch.Tensor,
    upper_slack: torch.Tensor,
    velocity_limit: torch.Tensor,
    acceleration_limit: torch.Tensor,
    nominal_dt: torch.Tensor | float,
    acceleration_history_enabled: torch.Tensor | bool,
    *,
    min_velocity_headroom: torch.Tensor | float = -1.0,
    headroom_activation_margin: torch.Tensor | float = CPU_MARGIN_THRESHOLD,
    saturation_exit_enabled: torch.Tensor | bool = False,
    min_bound_fraction: float = CPU_MIN_BOUND_FRACTION,
    margin_threshold: float = CPU_MARGIN_THRESHOLD,
    recovery_gain: float = CPU_LIMIT_RECOVERY_GAIN,
    recovery_enter_epsilon: float = CPU_LIMIT_RECOVERY_ENTER_EPSILON,
    recovery_exit_epsilon: float = CPU_LIMIT_RECOVERY_EXIT_EPSILON,
    exit_release_margin: float = CPU_LIMIT_EXIT_RELEASE_MARGIN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build broadcastable velocity bounds from lower and upper pose slack.

    ``min_velocity_headroom >= 0`` enables explicit headroom. If it is negative,
    ``saturation_exit_enabled`` selects the CPU legacy 10%-of-limit headroom.
    Recovery uses the CPU defaults: gain 0.5, entry/exit epsilon 1e-4, and zero
    release margin. Computation and returned tensors use float64 to reproduce
    the scalar CPU policy closely on either CPU or CUDA.
    """

    raw_lower = _as_float64(lower_slack, like=lower_slack)
    raw_upper = _as_float64(upper_slack, like=raw_lower)
    velocity = torch.abs(_as_float64(velocity_limit, like=raw_lower))
    acceleration = torch.abs(_as_float64(acceleration_limit, like=raw_lower))
    dt = torch.clamp(_as_float64(nominal_dt, like=raw_lower), min=CPU_MINIMUM_DT)
    history = _as_bool(acceleration_history_enabled, like=raw_lower)

    raw_lower, raw_upper, velocity, acceleration, dt, history = torch.broadcast_tensors(
        raw_lower, raw_upper, velocity, acceleration, dt, history
    )
    outside_lower = raw_lower < -recovery_enter_epsilon
    outside_upper = raw_upper < -recovery_enter_epsilon
    outside_both = outside_lower & outside_upper
    release_margin = max(
        exit_release_margin,
        recovery_exit_epsilon - recovery_enter_epsilon,
    )

    lower_margin = torch.clamp(raw_lower, min=0.0)
    upper_margin = torch.clamp(raw_upper, min=0.0)
    position_lower = -lower_margin / dt
    position_upper = upper_margin / dt
    continuous_lower = -torch.sqrt(2.0 * acceleration * lower_margin)
    continuous_upper = torch.sqrt(2.0 * acceleration * upper_margin)
    sampled_lower = -_discrete_stopping_velocity_limit(lower_margin, acceleration, dt)
    sampled_upper = _discrete_stopping_velocity_limit(upper_margin, acceleration, dt)
    stopping_lower = torch.where(history, sampled_lower, continuous_lower)
    stopping_upper = torch.where(history, sampled_upper, continuous_upper)

    lower = torch.maximum(torch.maximum(position_lower, -velocity), stopping_lower)
    upper = torch.minimum(torch.minimum(position_upper, velocity), stopping_upper)

    explicit_headroom = _as_float64(min_velocity_headroom, like=raw_lower)
    activation = torch.clamp(
        _as_float64(headroom_activation_margin, like=raw_lower), min=0.0
    )
    legacy_headroom = _as_bool(saturation_exit_enabled, like=raw_lower)
    explicit_headroom, activation, legacy_headroom = torch.broadcast_tensors(
        explicit_headroom, activation, legacy_headroom
    )
    use_explicit = explicit_headroom >= 0.0
    use_headroom = use_explicit | legacy_headroom
    minimum_speed = torch.where(
        use_explicit,
        torch.clamp(explicit_headroom, min=0.0),
        min_bound_fraction * velocity,
    )
    effective_activation = torch.where(
        use_explicit,
        activation,
        torch.full_like(activation, margin_threshold),
    )
    inside = ~outside_lower & ~outside_upper
    lower = torch.where(
        inside
        & use_headroom
        & (minimum_speed > 0.0)
        & (lower > -minimum_speed)
        & (raw_lower > effective_activation),
        -minimum_speed,
        lower,
    )
    upper = torch.where(
        inside
        & use_headroom
        & (minimum_speed > 0.0)
        & (upper < minimum_speed)
        & (raw_upper > effective_activation),
        minimum_speed,
        upper,
    )

    lower_violation = torch.clamp(-raw_lower - release_margin, min=0.0)
    lower_recovery = torch.minimum(recovery_gain * lower_violation / dt, velocity)
    lower = torch.where(outside_lower, torch.maximum(lower, lower_recovery), lower)
    upper_violation = torch.clamp(-raw_upper - release_margin, min=0.0)
    upper_recovery = torch.maximum(-(recovery_gain * upper_violation / dt), -velocity)
    upper = torch.where(
        ~outside_lower & outside_upper,
        torch.minimum(upper, upper_recovery),
        upper,
    )

    lower = torch.where(outside_both, -velocity, lower)
    upper = torch.where(outside_both, velocity, upper)
    midpoint = 0.5 * (lower + upper)
    crossed = lower > upper
    return torch.where(crossed, midpoint, lower), torch.where(crossed, midpoint, upper)


def _quaternion_relative_log_xyzw(
    current_quaternion: torch.Tensor,
    reference_quaternion: torch.Tensor,
) -> torch.Tensor:
    """Return log(R_reference.T @ R_current) as a rotation vector."""

    current = current_quaternion / torch.clamp(
        torch.linalg.vector_norm(current_quaternion, dim=-1, keepdim=True),
        min=1.0e-15,
    )
    reference = reference_quaternion / torch.clamp(
        torch.linalg.vector_norm(reference_quaternion, dim=-1, keepdim=True),
        min=1.0e-15,
    )
    rx, ry, rz, rw = reference.unbind(dim=-1)
    cx, cy, cz, cw = current.unbind(dim=-1)
    relative_xyz = torch.stack(
        (
            rw * cx - rx * cw - ry * cz + rz * cy,
            rw * cy + rx * cz - ry * cw - rz * cx,
            rw * cz - rx * cy + ry * cx - rz * cw,
        ),
        dim=-1,
    )
    relative_w = rw * cw + rx * cx + ry * cy + rz * cz
    sign = torch.where(relative_w < 0.0, -1.0, 1.0)
    relative_xyz = relative_xyz * sign.unsqueeze(-1)
    relative_w = relative_w * sign
    vector_norm = torch.linalg.vector_norm(relative_xyz, dim=-1)
    angle = 2.0 * torch.atan2(vector_norm, torch.clamp(relative_w, min=0.0))
    scale = torch.where(
        vector_norm > 1.0e-12,
        angle / torch.clamp(vector_norm, min=1.0e-15),
        2.0 * torch.ones_like(vector_norm),
    )
    return relative_xyz * scale.unsqueeze(-1)


def relative_pose_state(
    current_pose_xyzw: torch.Tensor,
    reference_pose_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Return ``[p - p_ref, log(R_ref.T R)]`` for broadcastable poses."""

    current = current_pose_xyzw.to(dtype=torch.float64)
    reference = reference_pose_xyzw.to(dtype=torch.float64, device=current.device)
    if current.shape[-1] != 7 or reference.shape[-1] != 7:
        raise ValueError("current and reference poses must have trailing dimension 7")
    batch_shape = torch.broadcast_shapes(current.shape[:-1], reference.shape[:-1])
    current = torch.broadcast_to(current, batch_shape + (7,))
    reference = torch.broadcast_to(reference, batch_shape + (7,))
    return torch.cat(
        (
            current[..., :3] - reference[..., :3],
            _quaternion_relative_log_xyzw(current[..., 3:], reference[..., 3:]),
        ),
        dim=-1,
    )


def torso_pose_bound_rows(
    current_pose_xyzw: torch.Tensor,
    reference_pose_xyzw: torch.Tensor,
    frame_jacobian: torch.Tensor,
    axis_mask: torch.Tensor,
    lower_relative_pose_limit: torch.Tensor,
    upper_relative_pose_limit: torch.Tensor,
    velocity_limit: torch.Tensor,
    acceleration_limit: torch.Tensor,
    nominal_dt: torch.Tensor | float,
    excluded_active_columns: torch.Tensor,
    *,
    acceleration_history_enabled: torch.Tensor | bool = False,
    headroom_enabled: torch.Tensor | bool = False,
    headroom_fraction: torch.Tensor | float = CPU_MIN_BOUND_FRACTION,
    headroom_activation_margin: torch.Tensor | float = CPU_MARGIN_THRESHOLD,
    recovery_gain: float = CPU_LIMIT_RECOVERY_GAIN,
    recovery_enter_epsilon: float = CPU_LIMIT_RECOVERY_ENTER_EPSILON,
    recovery_exit_epsilon: float = CPU_LIMIT_RECOVERY_EXIT_EPSILON,
    exit_release_margin: float = CPU_LIMIT_EXIT_RELEASE_MARGIN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build six fixed-shape torso-pose rows for arbitrary ``nv``.

    Poses are ``[..., (x, y, z, qx, qy, qz, qw)]``. Translation state is
    ``current-reference`` in world coordinates; rotation state is
    ``log(R_reference.T R_current)``. ``excluded_active_columns`` is a boolean
    mask broadcastable to ``frame_jacobian[..., 0, :]``. Disabled axis rows stay
    present, are zeroed, and receive finite +/-1e10 bounds.
    """

    current = current_pose_xyzw.to(dtype=torch.float64)
    reference = reference_pose_xyzw.to(dtype=torch.float64, device=current.device)
    jacobian = frame_jacobian.to(dtype=torch.float64, device=current.device)
    if current.shape[-1] != 7 or reference.shape[-1] != 7:
        raise ValueError("current and reference poses must have trailing dimension 7")
    if jacobian.shape[-2] != 6:
        raise ValueError("frame_jacobian must have shape [..., 6, nv]")

    vector_inputs = (
        axis_mask,
        lower_relative_pose_limit,
        upper_relative_pose_limit,
        velocity_limit,
        acceleration_limit,
    )
    if any(value.shape[-1] != 6 for value in vector_inputs):
        raise ValueError("axis mask and torso limit tensors must end in dimension 6")
    batch_shape = torch.broadcast_shapes(
        current.shape[:-1],
        reference.shape[:-1],
        jacobian.shape[:-2],
        *(value.shape[:-1] for value in vector_inputs),
    )
    current = torch.broadcast_to(current, batch_shape + (7,))
    reference = torch.broadcast_to(reference, batch_shape + (7,))
    jacobian = torch.broadcast_to(jacobian, batch_shape + (6, jacobian.shape[-1]))

    relative_state = relative_pose_state(current, reference)
    lower_limit = torch.broadcast_to(
        _as_float64(lower_relative_pose_limit, like=current), batch_shape + (6,)
    )
    upper_limit = torch.broadcast_to(
        _as_float64(upper_relative_pose_limit, like=current), batch_shape + (6,)
    )
    velocity = torch.broadcast_to(
        _as_float64(velocity_limit, like=current), batch_shape + (6,)
    )
    acceleration = torch.broadcast_to(
        _as_float64(acceleration_limit, like=current), batch_shape + (6,)
    )
    lower_slack = relative_state - lower_limit
    upper_slack = upper_limit - relative_state
    # Build this constant from an existing device tensor.  Constructing a CUDA
    # tensor from a Python list performs a host-to-device copy, which is illegal
    # while the whole solver is being captured into a CUDA graph.
    slack_epsilon = torch.full_like(lower_slack, CPU_TORSO_BOUND_SLACK_EPS_TRANS)
    slack_epsilon[..., 3:] = CPU_TORSO_BOUND_SLACK_EPS_ROT
    lower_slack = torch.where(
        (lower_slack < 0.0) & (lower_slack >= -slack_epsilon),
        torch.zeros_like(lower_slack),
        lower_slack,
    )
    upper_slack = torch.where(
        (upper_slack < 0.0) & (upper_slack >= -slack_epsilon),
        torch.zeros_like(upper_slack),
        upper_slack,
    )
    enabled = _as_bool(headroom_enabled, like=current)
    fraction = _as_float64(headroom_fraction, like=current)
    minimum_headroom = torch.where(
        enabled,
        fraction * velocity,
        torch.full_like(velocity, -1.0),
    )
    row_lower, row_upper = velocity_box_bounds(
        lower_slack,
        upper_slack,
        velocity,
        acceleration,
        nominal_dt,
        acceleration_history_enabled,
        min_velocity_headroom=minimum_headroom,
        headroom_activation_margin=headroom_activation_margin,
        recovery_gain=recovery_gain,
        recovery_enter_epsilon=recovery_enter_epsilon,
        recovery_exit_epsilon=recovery_exit_epsilon,
        exit_release_margin=exit_release_margin,
    )

    active = _as_bool(axis_mask, like=current)
    active = torch.broadcast_to(active, batch_shape + (6,))
    excluded = _as_bool(excluded_active_columns, like=jacobian)
    excluded = torch.broadcast_to(excluded, batch_shape + (jacobian.shape[-1],))
    rows = torch.where(
        active.unsqueeze(-1),
        jacobian.masked_fill(excluded.unsqueeze(-2), 0.0),
        torch.zeros_like(jacobian),
    )
    unbounded = torch.full_like(row_lower, CPU_UNBOUNDED_CONSTRAINT_LIMIT)
    row_lower = torch.where(active, row_lower, -unbounded)
    row_upper = torch.where(active, row_upper, unbounded)
    return rows, row_lower, row_upper, active


def bounded_cyclic_row_projection(
    velocity: torch.Tensor,
    rows: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    velocity_lower: torch.Tensor,
    velocity_upper: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
    iterations: int = 16,
    feasibility_tolerance: float = 1.0e-8,
) -> CyclicProjectionResult:
    """Project onto row intervals and a componentwise velocity box cyclically.

    The operation is bounded by ``iterations`` and preserves arbitrary leading
    batch dimensions. Diagnostics describe the returned point; ``feasible`` is
    true only when both row and componentwise violations are within tolerance.
    """

    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    work_velocity = velocity.to(dtype=torch.float64)
    work_rows = rows.to(dtype=torch.float64, device=velocity.device)
    work_lower = lower.to(dtype=torch.float64, device=velocity.device)
    work_upper = upper.to(dtype=torch.float64, device=velocity.device)
    box_lower = velocity_lower.to(dtype=torch.float64, device=velocity.device)
    box_upper = velocity_upper.to(dtype=torch.float64, device=velocity.device)
    if work_rows.shape[-1] != work_velocity.shape[-1]:
        raise ValueError("rows and velocity must have the same trailing nv dimension")
    if (
        work_rows.shape[-2] != work_lower.shape[-1]
        or work_lower.shape != work_upper.shape
    ):
        raise ValueError("row bounds must match rows[..., m, nv]")
    if box_lower.shape != box_upper.shape:
        raise ValueError("velocity lower and upper bounds must have matching shapes")

    if active_mask is None:
        active = torch.ones_like(work_lower, dtype=torch.bool)
    else:
        active = _as_bool(active_mask, like=velocity)
        active = torch.broadcast_to(active, work_lower.shape)
    corrected = torch.maximum(torch.minimum(work_velocity, box_upper), box_lower)
    row_count = work_rows.shape[-2]
    for _ in range(iterations):
        for row_index in range(row_count):
            row = work_rows[..., row_index, :]
            value = torch.sum(row * corrected, dim=-1)
            row_lower = work_lower[..., row_index]
            row_upper = work_upper[..., row_index]
            denominator = torch.sum(row.square(), dim=-1)
            safe_denominator = torch.clamp(denominator, min=1.0e-30)
            correction = torch.where(
                value < row_lower,
                (row_lower - value) / safe_denominator,
                torch.where(
                    value > row_upper,
                    (row_upper - value) / safe_denominator,
                    torch.zeros_like(value),
                ),
            )
            correction = torch.where(
                active[..., row_index] & (denominator > 1.0e-30),
                correction,
                torch.zeros_like(correction),
            )
            corrected = corrected + correction.unsqueeze(-1) * row
            corrected = torch.maximum(torch.minimum(corrected, box_upper), box_lower)

    values = torch.sum(work_rows * corrected.unsqueeze(-2), dim=-1)
    lower_residual = torch.where(
        active, torch.clamp(work_lower - values, min=0.0), torch.zeros_like(values)
    )
    upper_residual = torch.where(
        active, torch.clamp(values - work_upper, min=0.0), torch.zeros_like(values)
    )
    row_violation = torch.maximum(lower_residual, upper_residual)
    max_row_violation = torch.amax(row_violation, dim=-1)
    lower_box_violation = torch.clamp(box_lower - corrected, min=0.0)
    upper_box_violation = torch.clamp(corrected - box_upper, min=0.0)
    max_velocity_violation = torch.amax(
        torch.maximum(lower_box_violation, upper_box_violation), dim=-1
    )
    feasible = (max_row_violation <= feasibility_tolerance) & (
        max_velocity_violation <= feasibility_tolerance
    )
    return CyclicProjectionResult(
        velocity=corrected,
        feasible=feasible,
        max_row_violation=max_row_violation,
        max_velocity_violation=max_velocity_violation,
        lower_row_residual=lower_residual,
        upper_row_residual=upper_residual,
    )
