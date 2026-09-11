from __future__ import annotations

import pytest
import torch

from embodik.gpu.wbc._runtime.gpu_constraints import (
    CPU_CENTROIDAL_UNBOUNDED_LIMIT,
    capture_point_constraint_rows,
    centroidal_momentum_bound_rows,
    velocity_zmp_constraint_rows,
)


DTYPE = torch.float64


def test_capture_point_rows_encode_cpu_inequality_exactly() -> None:
    normals = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    offsets = torch.tensor([0.2, 0.3, 0.4])
    com_xy = torch.tensor([0.1, -0.2])
    jacobian = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    rows, lower, upper = capture_point_constraint_rows(
        normals, offsets, com_xy, jacobian, omega=2.0
    )

    torch.testing.assert_close(
        rows,
        torch.tensor([[0.5, 1.0], [1.5, 2.0], [-0.5, -1.0]], dtype=DTYPE),
    )
    torch.testing.assert_close(upper, torch.tensor([0.1, 0.5, 0.5], dtype=DTYPE))
    assert torch.equal(
        lower, torch.full((3,), -CPU_CENTROIDAL_UNBOUNDED_LIMIT, dtype=DTYPE)
    )

    command = torch.tensor([0.08, -0.03], dtype=DTYPE)
    predicted_capture_point = com_xy.to(DTYPE) + jacobian.to(DTYPE) @ command / 2.0
    torch.testing.assert_close(
        rows @ command,
        normals.to(DTYPE) @ predicted_capture_point
        - normals.to(DTYPE) @ com_xy.to(DTYPE),
    )
    assert bool(torch.all(rows @ command <= upper)) == bool(
        torch.all(normals.to(DTYPE) @ predicted_capture_point <= offsets.to(DTYPE))
    )


def test_capture_point_rows_broadcast_arbitrary_leading_batches() -> None:
    normals = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    ).reshape(1, 1, 4, 2)
    offsets = torch.ones(4).reshape(1, 1, 4)
    com_xy = torch.arange(4.0).reshape(2, 1, 2) / 10.0
    jacobian = torch.arange(30.0).reshape(1, 3, 2, 5) / 20.0
    omega = torch.tensor([[2.0], [4.0]]).expand(2, 3)

    rows, lower, upper = capture_point_constraint_rows(
        normals, offsets, com_xy, jacobian, omega
    )

    assert rows.shape == (2, 3, 4, 5)
    assert lower.shape == upper.shape == (2, 3, 4)
    expected_rows = torch.matmul(
        normals.to(DTYPE).expand(2, 3, 4, 2),
        jacobian.to(DTYPE).expand(2, 3, 2, 5),
    ) / omega.to(DTYPE)[..., None, None]
    torch.testing.assert_close(rows, expected_rows)


def test_velocity_zmp_rows_match_cpp_affine_equations_and_fz_floor() -> None:
    normals = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    offsets = torch.tensor([0.2, 0.3, 0.4])
    com = torch.tensor([0.1, -0.2, 0.5])
    ag = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ]
    )
    bias = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    current_dq = torch.tensor([0.2, -0.1])
    weight = torch.tensor([0.7, -0.4, 98.1])

    rows, lower, upper = velocity_zmp_constraint_rows(
        normals,
        offsets,
        com,
        ag,
        bias,
        current_dq,
        weight,
        dt=0.5,
        fz_min=20.0,
    )

    torch.testing.assert_close(
        rows,
        torch.tensor(
            [[-20.0, -23.2], [6.0, 6.0], [14.0, 16.0], [10.0, 12.0]],
            dtype=DTYPE,
        ),
    )
    torch.testing.assert_close(
        upper,
        torch.tensor(
            [9.06, 49.3, 49.5, CPU_CENTROIDAL_UNBOUNDED_LIMIT], dtype=DTYPE
        ),
    )
    torch.testing.assert_close(
        lower,
        torch.tensor(
            [
                -CPU_CENTROIDAL_UNBOUNDED_LIMIT,
                -CPU_CENTROIDAL_UNBOUNDED_LIMIT,
                -CPU_CENTROIDAL_UNBOUNDED_LIMIT,
                -77.6,
            ],
            dtype=DTYPE,
        ),
    )


def test_velocity_zmp_rows_broadcast_model_and_world_dimensions() -> None:
    normals = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    ).reshape(1, 1, 4, 2)
    offsets = torch.full((4,), 0.4)
    com = torch.tensor([0.1, -0.2, 0.7]).reshape(1, 1, 3)
    ag = torch.arange(2 * 1 * 6 * 7.0).reshape(2, 1, 6, 7) / 100.0
    bias = torch.arange(3 * 6.0).reshape(1, 3, 6) / 50.0
    current_dq = torch.arange(7.0) / 100.0
    weight = torch.tensor([0.0, 0.0, 300.0])
    dt = torch.tensor([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
    fz_min = torch.tensor([10.0, 11.0, 12.0]).reshape(1, 3)

    rows, lower, upper = velocity_zmp_constraint_rows(
        normals, offsets, com, ag, bias, current_dq, weight, dt, fz_min
    )

    assert rows.shape == (2, 3, 5, 7)
    assert lower.shape == upper.shape == (2, 3, 5)
    expanded_ag = ag.to(DTYPE).expand(2, 3, 6, 7)
    expected_gz = expanded_ag[..., 2, :] / dt.to(DTYPE)[..., None]
    torch.testing.assert_close(rows[..., -1, :], expected_gz)
    current_rate = torch.matmul(
        expanded_ag, current_dq.to(DTYPE).expand(2, 3, 7).unsqueeze(-1)
    ).squeeze(-1)
    expanded_bias = bias.to(DTYPE).expand(2, 3, 6)
    expected_kz = expanded_bias[..., 2] - current_rate[..., 2] / dt.to(DTYPE)
    torch.testing.assert_close(
        lower[..., -1], fz_min.to(DTYPE).expand(2, 3) - 300.0 - expected_kz
    )


def test_centroidal_momentum_rows_select_axes_and_broadcast() -> None:
    ag = torch.arange(2 * 1 * 6 * 5.0).reshape(2, 1, 6, 5)
    lower = -torch.arange(1.0, 4.0).reshape(1, 1, 3).expand(1, 3, 3)
    upper = torch.arange(4.0, 7.0).reshape(1, 1, 3).expand(1, 3, 3)
    mask = torch.tensor([1.0, 0.0, -2.0, 0.0, 0.0, 3.0])

    rows, actual_lower, actual_upper = centroidal_momentum_bound_rows(
        ag, lower, upper, mask
    )

    assert rows.shape == (2, 3, 3, 5)
    assert actual_lower.shape == actual_upper.shape == (2, 3, 3)
    torch.testing.assert_close(
        rows, ag.to(DTYPE).expand(2, 3, 6, 5)[..., [0, 2, 5], :]
    )
    torch.testing.assert_close(actual_lower, lower.to(DTYPE).expand(2, 3, 3))
    torch.testing.assert_close(actual_upper, upper.to(DTYPE).expand(2, 3, 3))


@pytest.mark.parametrize("bad_omega", [0.0, -1.0, float("nan"), float("inf")])
def test_capture_point_rejects_invalid_frequency(bad_omega: float) -> None:
    with pytest.raises(ValueError, match="omega"):
        capture_point_constraint_rows(
            torch.ones(2, 2),
            torch.ones(2),
            torch.zeros(2),
            torch.ones(2, 3),
            bad_omega,
        )


def test_capture_point_rejects_shape_and_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="halfspace_offsets"):
        capture_point_constraint_rows(
            torch.ones(2, 2), torch.ones(3), torch.zeros(2), torch.ones(2, 3), 1.0
        )
    with pytest.raises(ValueError, match="com_jacobian_xy"):
        capture_point_constraint_rows(
            torch.ones(2, 2),
            torch.ones(2),
            torch.zeros(2),
            torch.tensor([[1.0, float("nan")], [2.0, 3.0]]),
            1.0,
        )


@pytest.mark.parametrize(
    ("dt", "fz_min", "message"),
    [(0.0, 1.0, "dt"), (-0.1, 1.0, "dt"), (0.1, 0.0, "fz_min")],
)
def test_velocity_zmp_rejects_invalid_physical_scalars(
    dt: float, fz_min: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        velocity_zmp_constraint_rows(
            torch.ones(2, 2),
            torch.ones(2),
            torch.ones(3),
            torch.ones(6, 4),
            torch.ones(6),
            torch.ones(4),
            torch.ones(3),
            dt,
            fz_min,
        )


def test_velocity_zmp_rejects_dimension_and_batch_mismatches() -> None:
    valid = dict(
        halfspace_normals=torch.ones(2, 2),
        halfspace_offsets=torch.ones(2),
        com_position=torch.ones(3),
        centroidal_momentum_matrix=torch.ones(6, 4),
        centroidal_momentum_bias=torch.ones(6),
        current_velocity=torch.ones(4),
        support_weight_force=torch.ones(3),
        dt=0.1,
        fz_min=1.0,
    )
    with pytest.raises(ValueError, match="current_velocity"):
        velocity_zmp_constraint_rows(**{**valid, "current_velocity": torch.ones(5)})
    with pytest.raises(ValueError, match="batch dimensions"):
        velocity_zmp_constraint_rows(
            **{
                **valid,
                "com_position": torch.ones(2, 3),
                "centroidal_momentum_bias": torch.ones(3, 6),
            }
        )


def test_centroidal_momentum_rows_validate_mask_bounds_and_values() -> None:
    ag = torch.ones(6, 4)
    with pytest.raises(ValueError, match="shape"):
        centroidal_momentum_bound_rows(
            ag, torch.zeros(1), torch.ones(1), torch.ones(5)
        )
    with pytest.raises(ValueError, match="at least one"):
        centroidal_momentum_bound_rows(
            ag, torch.empty(0), torch.empty(0), torch.zeros(6)
        )
    with pytest.raises(ValueError, match="one value per selected axis"):
        centroidal_momentum_bound_rows(
            ag, torch.zeros(2), torch.ones(2), torch.tensor([1, 0, 1, 0, 0, 1])
        )
    with pytest.raises(ValueError, match="must not exceed"):
        centroidal_momentum_bound_rows(
            ag,
            torch.tensor([2.0]),
            torch.tensor([1.0]),
            torch.tensor([1, 0, 0, 0, 0, 0]),
        )
    with pytest.raises(ValueError, match="finite"):
        centroidal_momentum_bound_rows(
            torch.full((6, 4), float("nan")),
            torch.zeros(1),
            torch.ones(1),
            torch.tensor([1, 0, 0, 0, 0, 0]),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_centroidal_row_helpers_execute_entirely_on_cuda() -> None:
    device = torch.device("cuda")
    normals = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
    offsets = torch.ones(2, device=device)
    capture = capture_point_constraint_rows(
        normals,
        offsets,
        torch.zeros(3, 2, device=device),
        torch.ones(1, 2, 4, device=device),
        torch.ones(3, device=device),
    )
    zmp = velocity_zmp_constraint_rows(
        normals,
        offsets,
        torch.ones(3, 3, device=device),
        torch.ones(1, 6, 4, device=device),
        torch.zeros(6, device=device),
        torch.zeros(4, device=device),
        torch.tensor([0.0, 0.0, 100.0], device=device),
        0.01,
        10.0,
    )
    momentum = centroidal_momentum_bound_rows(
        torch.ones(3, 6, 4, device=device),
        torch.full((2,), -1.0, device=device),
        torch.full((2,), 1.0, device=device),
        torch.tensor([1, 0, 0, 1, 0, 0], device=device),
    )

    assert all(
        tensor.device.type == device.type
        for result in (capture, zmp, momentum)
        for tensor in result
    )
