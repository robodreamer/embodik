from __future__ import annotations

import math

import pytest
import torch

from embodik.gpu.wbc._runtime.gpu_acceleration_constraints import (
    CENTROIDAL_UNBOUNDED_LIMIT,
    acceleration_capture_point_rows,
    acceleration_state_box_bounds,
    acceleration_zmp_constraint_rows,
)

DTYPE = torch.float64


def _tolerance(lhs: float, rhs: float) -> float:
    return 1.0e-12 * (1.0 + max(abs(lhs), abs(rhs)))


def _continuous_upper(margin: float, outward_rate: float, dt: float) -> float:
    endpoint = 2.0 * (margin - outward_rate * dt) / (dt * dt)
    if outward_rate <= 0.0 or 2.0 * margin >= outward_rate * dt:
        return endpoint
    return min(endpoint, -(outward_rate * outward_rate) / (2.0 * margin))


def _viability_upper(
    margin: float,
    outward_rate: float,
    braking_acceleration: float,
    dt: float,
) -> float:
    dt_squared = dt * dt
    linear = 2.0 * outward_rate * dt + braking_acceleration * dt_squared
    constant = (
        outward_rate * outward_rate
        + 2.0 * braking_acceleration * outward_rate * dt
        - 2.0 * braking_acceleration * margin
    )
    reverse_rate_bound = -outward_rate / dt
    discriminant = linear * linear - 4.0 * dt_squared * constant
    assert discriminant >= -_tolerance(linear * linear, 4.0 * dt_squared * constant)
    square_root = math.sqrt(max(0.0, discriminant))
    upper_root = -linear / (2.0 * dt_squared)
    if square_root > 0.0 or linear != 0.0:
        stable_numerator = -0.5 * (linear + math.copysign(square_root, linear))
        first_root = stable_numerator / dt_squared
        second_root = constant / stable_numerator if stable_numerator != 0.0 else first_root
        upper_root = max(first_root, second_root)
    return max(reverse_rate_bound, upper_root)


def _scalar_cpp_reference(
    value: float,
    rate: float,
    dt: float,
    *,
    state_lower: float,
    state_upper: float,
    rate_lower: float,
    rate_upper: float,
    acceleration_lower: float,
    acceleration_upper: float,
    lower_braking_acceleration: float,
    upper_braking_acceleration: float,
) -> tuple[float, float]:
    """Direct scalar transcription used only as an independent test oracle."""

    lower = acceleration_lower
    upper = acceleration_upper
    lower = max(lower, (rate_lower - rate) / dt)
    upper = min(upper, (rate_upper - rate) / dt)
    bounded_value = min(max(value, state_lower), state_upper)

    lower_margin = bounded_value - state_lower
    assert state_lower <= value + _tolerance(state_lower, value)
    assert not (lower_margin <= 0.0 and rate < 0.0)
    assert not (
        rate < 0.0
        and rate * rate
        > 2.0 * lower_braking_acceleration * lower_margin
        + _tolerance(
            rate * rate,
            2.0 * lower_braking_acceleration * lower_margin,
        )
    )
    lower = max(
        lower,
        (
            max(
                (state_lower - bounded_value) / dt,
                -math.sqrt(2.0 * lower_braking_acceleration * lower_margin),
            )
            - rate
        )
        / dt,
        (state_lower - bounded_value - rate * dt) / (0.5 * dt * dt),
        -_continuous_upper(lower_margin, -rate, dt),
        -_viability_upper(lower_margin, -rate, lower_braking_acceleration, dt),
    )

    upper_margin = state_upper - bounded_value
    assert value <= state_upper + _tolerance(value, state_upper)
    assert not (upper_margin <= 0.0 and rate > 0.0)
    assert not (
        rate > 0.0
        and rate * rate
        > 2.0 * upper_braking_acceleration * upper_margin
        + _tolerance(
            rate * rate,
            2.0 * upper_braking_acceleration * upper_margin,
        )
    )
    upper = min(
        upper,
        (
            min(
                (state_upper - bounded_value) / dt,
                math.sqrt(2.0 * upper_braking_acceleration * upper_margin),
            )
            - rate
        )
        / dt,
        (state_upper - bounded_value - rate * dt) / (0.5 * dt * dt),
        _continuous_upper(upper_margin, rate, dt),
        _viability_upper(upper_margin, rate, upper_braking_acceleration, dt),
    )
    assert lower <= upper
    return lower, upper


def _fully_bounded(value: torch.Tensor, rate: torch.Tensor, dt) -> object:
    return acceleration_state_box_bounds(
        value,
        rate,
        dt,
        state_lower=-1.0,
        state_upper=1.0,
        rate_lower=-1.0,
        rate_upper=1.0,
        acceleration_lower=-4.0,
        acceleration_upper=4.0,
        lower_braking_acceleration=4.0,
        upper_braking_acceleration=4.0,
        state_lower_active=True,
        state_upper_active=True,
        rate_lower_active=True,
        rate_upper_active=True,
    )


def test_state_box_matches_cpp_symmetric_fixture() -> None:
    result = acceleration_state_box_bounds(
        torch.tensor([[0.04, -0.04]], dtype=DTYPE),
        torch.tensor([[0.12, -0.12]], dtype=DTYPE),
        0.1,
        state_lower=-0.05,
        state_upper=0.05,
        acceleration_lower=-1.0,
        acceleration_upper=1.0,
        lower_braking_acceleration=1.0,
        upper_braking_acceleration=1.0,
        state_lower_active=True,
        state_upper_active=True,
    )

    assert result.finite.tolist() == [True]
    assert result.feasible.tolist() == [True]
    torch.testing.assert_close(
        result.lower,
        torch.tensor([[-1.0, 0.6753049234040402]], dtype=DTYPE),
        rtol=0.0,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        result.upper,
        torch.tensor([[-0.6753049234040402, 1.0]], dtype=DTYPE),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_state_box_vectorized_arbitrary_worlds_and_dofs_match_scalar_cpp_equations() -> None:
    generator = torch.Generator().manual_seed(42)
    value = torch.rand((2, 3, 7), generator=generator, dtype=DTYPE) - 0.5
    rate = 0.4 * torch.rand((1, 3, 7), generator=generator, dtype=DTYPE) - 0.2
    dt = torch.tensor([[0.01, 0.04, 0.08], [0.02, 0.05, 0.1]], dtype=DTYPE)

    result = _fully_bounded(value, rate, dt)

    assert result.lower.shape == result.upper.shape == (2, 3, 7)
    assert result.finite.shape == result.feasible.shape == (2, 3)
    assert bool(result.finite.all())
    assert bool(result.feasible.all())
    expanded_rate = rate.expand_as(value)
    for world_0 in range(2):
        for world_1 in range(3):
            for joint in range(7):
                expected = _scalar_cpp_reference(
                    value[world_0, world_1, joint].item(),
                    expanded_rate[world_0, world_1, joint].item(),
                    dt[world_0, world_1].item(),
                    state_lower=-1.0,
                    state_upper=1.0,
                    rate_lower=-1.0,
                    rate_upper=1.0,
                    acceleration_lower=-4.0,
                    acceleration_upper=4.0,
                    lower_braking_acceleration=4.0,
                    upper_braking_acceleration=4.0,
                )
                torch.testing.assert_close(
                    result.lower[world_0, world_1, joint],
                    torch.tensor(expected[0], dtype=DTYPE),
                    rtol=1.0e-13,
                    atol=1.0e-13,
                )
                torch.testing.assert_close(
                    result.upper[world_0, world_1, joint],
                    torch.tensor(expected[1], dtype=DTYPE),
                    rtol=1.0e-13,
                    atol=1.0e-13,
                )


def test_state_box_intersects_acceleration_next_velocity_and_endpoint() -> None:
    rate_only = acceleration_state_box_bounds(
        torch.zeros(3),
        torch.tensor([0.1, -0.1, 0.0]),
        0.1,
        rate_lower=torch.tensor([-1.0, -0.2, -1.0]),
        rate_upper=torch.tensor([0.2, 1.0, 1.0]),
        acceleration_lower=-5.0,
        acceleration_upper=5.0,
        rate_lower_active=True,
        rate_upper_active=True,
    )
    torch.testing.assert_close(
        rate_only.lower,
        torch.tensor([-5.0, -1.0, -5.0], dtype=DTYPE),
    )
    torch.testing.assert_close(
        rate_only.upper,
        torch.tensor([1.0, 5.0, 5.0], dtype=DTYPE),
    )

    endpoint = acceleration_state_box_bounds(
        torch.tensor([0.9], dtype=DTYPE),
        torch.tensor([0.0], dtype=DTYPE),
        0.2,
        state_lower=-10.0,
        state_upper=1.0,
        acceleration_lower=-100.0,
        acceleration_upper=100.0,
        lower_braking_acceleration=100.0,
        upper_braking_acceleration=100.0,
        state_lower_active=True,
        state_upper_active=True,
    )
    assert bool(endpoint.feasible)
    torch.testing.assert_close(endpoint.upper, torch.tensor([2.5], dtype=DTYPE))


def test_state_box_fails_complete_world_closed_for_state_and_braking_violations() -> None:
    value = torch.tensor(
        [
            [1.01, 0.0],
            [1.0, 0.0],
            [0.9, 0.0],
            [0.2, -0.2],
        ]
    )
    rate = torch.tensor(
        [
            [0.0, 0.0],
            [0.01, 0.0],
            [2.0, 0.0],
            [0.1, -0.1],
        ]
    )

    result = _fully_bounded(value, rate, 0.01)

    assert result.finite.tolist() == [True, True, True, True]
    assert result.feasible.tolist() == [False, False, False, True]
    assert torch.isposinf(result.lower[:3]).all()
    assert torch.isneginf(result.upper[:3]).all()
    assert torch.isfinite(result.lower[3]).all()
    assert torch.isfinite(result.upper[3]).all()


def test_state_box_invalid_nonfinite_and_empty_inputs_fail_closed() -> None:
    valid_value = torch.zeros((4, 2), dtype=DTYPE)
    value = valid_value.clone()
    value[0, 0] = torch.nan
    dt = torch.tensor([0.1, 0.0, 0.1, 0.1], dtype=DTYPE)
    acceleration_lower = torch.full((4, 2), -1.0, dtype=DTYPE)
    acceleration_upper = torch.full((4, 2), 1.0, dtype=DTYPE)
    acceleration_upper[2] = 0.5
    result = acceleration_state_box_bounds(
        value,
        torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]),
        dt,
        state_lower=-1.0,
        state_upper=1.0,
        rate_upper=0.0,
        acceleration_lower=acceleration_lower,
        acceleration_upper=acceleration_upper,
        lower_braking_acceleration=1.0,
        upper_braking_acceleration=1.0,
        state_lower_active=True,
        state_upper_active=True,
        rate_upper_active=torch.tensor([[False], [False], [False], [True]]),
    )

    assert result.finite.tolist() == [False, True, True, True]
    # World 1 has invalid dt; world 2 lacks the braking authority required by
    # its active lower state bound; world 3 has an empty rate/acceleration box.
    assert result.feasible.tolist() == [False, False, False, False]
    assert torch.isposinf(result.lower).all()
    assert torch.isneginf(result.upper).all()


def test_state_box_rejects_shape_contract_errors() -> None:
    with pytest.raises(ValueError, match="same nv"):
        acceleration_state_box_bounds(torch.zeros(2, 3), torch.zeros(2, 4), 0.1)
    with pytest.raises(ValueError, match="dt"):
        acceleration_state_box_bounds(torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(3))
    with pytest.raises(ValueError, match="state_lower"):
        acceleration_state_box_bounds(
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            0.1,
            state_lower=torch.zeros(4),
        )


def test_acceleration_capture_point_rows_match_cpu_prediction_equation() -> None:
    normals = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    offsets = torch.tensor([0.4, 0.5, 0.6])
    position = torch.tensor([0.1, -0.2])
    velocity = torch.tensor([0.3, -0.1])
    jacobian = torch.tensor([[1.0, 2.0], [-3.0, 0.5]])
    bias = torch.tensor([0.2, -0.4])
    ddq = torch.tensor([0.6, -0.2], dtype=DTYPE)
    dt = 0.1
    omega = 2.5

    rows, lower, upper = acceleration_capture_point_rows(
        normals, offsets, position, velocity, jacobian, bias, dt, omega
    )

    scale = 0.5 * dt * dt + dt / omega
    predicted = (
        position.to(DTYPE)
        + dt * velocity.to(DTYPE)
        + velocity.to(DTYPE) / omega
        + scale * (jacobian.to(DTYPE) @ ddq + bias.to(DTYPE))
    )
    torch.testing.assert_close(rows @ ddq - upper, normals.to(DTYPE) @ predicted - offsets)
    assert torch.equal(lower, torch.full((3,), -CENTROIDAL_UNBOUNDED_LIMIT, dtype=DTYPE))


def test_acceleration_zmp_rows_match_physical_centroidal_rate_equations() -> None:
    normals = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    offsets = torch.tensor([0.3, 0.25, 0.4])
    com = torch.tensor([0.1, -0.05, 0.6])
    ag = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, -1.0],
            [0.5, 0.2],
            [0.3, -0.7],
            [1.2, 0.4],
            [-0.2, 0.8],
        ]
    )
    bias = torch.tensor([0.2, -0.1, 3.0, 0.5, -0.4, 0.0])
    gravity = torch.tensor([0.0, 0.0, -9.81])
    mass = 10.0
    ddq = torch.tensor([0.4, -0.2], dtype=DTYPE)

    rows, lower, upper = acceleration_zmp_constraint_rows(
        normals, offsets, com, ag, bias, gravity, mass, fz_min=20.0
    )

    hdot = ag.to(DTYPE) @ ddq + bias.to(DTYPE)
    force = hdot[:3] - mass * gravity.to(DTYPE)
    zmp = torch.stack(
        (
            com[0].to(DTYPE) - (hdot[4] + com[2] * force[0]) / force[2],
            com[1].to(DTYPE) + (hdot[3] - com[2] * force[1]) / force[2],
        )
    )
    torch.testing.assert_close(rows[0] @ ddq - lower[0], force[2] - 20.0)
    torch.testing.assert_close(
        rows[1:] @ ddq - upper[1:],
        force[2] * (normals.to(DTYPE) @ zmp - offsets.to(DTYPE)),
    )
    assert upper[0] == CENTROIDAL_UNBOUNDED_LIMIT
    assert torch.equal(lower[1:], torch.full((3,), -CENTROIDAL_UNBOUNDED_LIMIT, dtype=DTYPE))


def test_acceleration_centroidal_rows_broadcast_arbitrary_worlds_and_dofs() -> None:
    normals = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    offsets = torch.ones(4)
    position = torch.zeros(2, 1, 2)
    velocity = torch.ones(1, 3, 2)
    jacobian = torch.ones(2, 3, 2, 11)
    bias_xy = torch.zeros(2)
    dt = torch.tensor([[0.01], [0.02]])
    omega = torch.tensor([[2.0, 3.0, 4.0]])
    capture = acceleration_capture_point_rows(
        normals, offsets, position, velocity, jacobian, bias_xy, dt, omega
    )
    assert capture[0].shape == (2, 3, 4, 11)
    assert capture[1].shape == capture[2].shape == (2, 3, 4)

    zmp = acceleration_zmp_constraint_rows(
        normals,
        offsets,
        torch.ones(2, 1, 3),
        torch.ones(1, 3, 6, 11),
        torch.zeros(6),
        torch.tensor([0.0, 0.0, -9.81]),
        torch.tensor([[10.0], [20.0]]),
        torch.tensor([[1.0, 2.0, 3.0]]),
    )
    assert zmp[0].shape == (2, 3, 5, 11)
    assert zmp[1].shape == zmp[2].shape == (2, 3, 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_helpers_support_cuda_graph_replay() -> None:
    device = torch.device("cuda")
    value = torch.tensor([[0.2, -0.2, 0.1]], dtype=DTYPE, device=device)
    rate = torch.tensor([[0.1, -0.1, 0.0]], dtype=DTYPE, device=device)
    dt = torch.tensor([0.01], dtype=DTYPE, device=device)
    normals = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=DTYPE,
        device=device,
    )
    offsets = torch.ones(4, dtype=DTYPE, device=device)
    com = torch.tensor([[0.0, 0.0, 0.6]], dtype=DTYPE, device=device)
    com_velocity = torch.zeros(1, 2, dtype=DTYPE, device=device)
    com_jacobian = torch.ones(1, 2, 3, dtype=DTYPE, device=device)
    com_bias = torch.zeros(1, 2, dtype=DTYPE, device=device)
    ag = torch.ones(1, 6, 3, dtype=DTYPE, device=device)
    centroidal_bias = torch.zeros(1, 6, dtype=DTYPE, device=device)
    gravity = torch.tensor([0.0, 0.0, -9.81], dtype=DTYPE, device=device)
    omega = torch.tensor([3.0], dtype=DTYPE, device=device)
    mass = torch.tensor([10.0], dtype=DTYPE, device=device)
    force_floor = torch.tensor([1.0], dtype=DTYPE, device=device)

    def evaluate():
        state = _fully_bounded(value, rate, dt)
        capture = acceleration_capture_point_rows(
            normals,
            offsets,
            com[..., :2],
            com_velocity,
            com_jacobian,
            com_bias,
            dt,
            omega,
        )
        zmp = acceleration_zmp_constraint_rows(
            normals,
            offsets,
            com,
            ag,
            centroidal_bias,
            gravity,
            mass,
            force_floor,
        )
        return state, capture, zmp

    torch.cuda.synchronize()
    evaluate()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = evaluate()
    graph.replay()
    torch.cuda.synchronize()

    value.copy_(torch.tensor([[0.3, -0.1, 0.0]], dtype=DTYPE, device=device))
    rate.copy_(torch.tensor([[0.0, 0.05, -0.05]], dtype=DTYPE, device=device))
    graph.replay()
    torch.cuda.synchronize()
    replayed_lower = captured[0].lower.clone()
    eager = evaluate()
    torch.testing.assert_close(replayed_lower, eager[0].lower)
    assert bool(captured[0].feasible.all())
    assert all(tensor.device.type == "cuda" for family in captured[1:] for tensor in family)
