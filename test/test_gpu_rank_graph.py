"""Graph-safe native rank analysis and bounded task completion."""

import numpy as np
import pytest


def test_singleton_generalized_inverse_matches_numpy():
    torch = pytest.importorskip("torch")
    from embodik.gpu.wbc._runtime.gpu_priority import undamped_generalized_inverse

    for shape in ((1, 5), (5, 1), (1, 1)):
        values = np.arange(np.prod(shape), dtype=float).reshape(shape)
        for matrix in (values, np.zeros_like(values)):
            result = undamped_generalized_inverse(torch.tensor(matrix)).numpy()
            np.testing.assert_allclose(result, np.linalg.pinv(matrix), atol=1e-12)


def test_native_completion_graph_replays_changed_targets():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("warp")
    from embodik.gpu.wbc._runtime.gpu_priority import (
        complete_consistent_redundant_velocity,
    )
    from embodik.gpu.wbc._runtime.warp_directional_srinv import WarpDirectionalSRINV

    matrix = torch.tensor([[[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [0.0, 0.0, 1.0]]], device="cuda")
    target = torch.tensor([[1.0, 2.0, 0.3]], device="cuda")
    lower = torch.full((1, 3), -2.0, device="cuda")
    upper = torch.tensor([[2.0, 0.0, 2.0]], device="cuda")
    native = WarpDirectionalSRINV(3, 3, inverse_mode="undamped_relative", device="cuda")

    def inverse(value, _tolerance):
        return native.solve(value.contiguous())

    def solve():
        unscaled = (inverse(matrix, 1e-6) @ target.unsqueeze(-1)).squeeze(-1)
        return complete_consistent_redundant_velocity(
            matrix,
            target,
            lower,
            upper,
            unscaled,
            torch.zeros_like(unscaled),
            inverse_function=inverse,
            residual_tolerance=1e-5,
        )

    for _ in range(3):
        solve()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = solve()
    graph.replay()
    torch.testing.assert_close(
        result, torch.tensor([[1.0, 0.0, 0.3]], device="cuda"), atol=1e-5, rtol=0
    )
    target.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(
        result, torch.tensor([[0.5, 0.0, 0.15]], device="cuda"), atol=1e-5, rtol=0
    )
    assert bool((native.status == 0).all())


@pytest.mark.parametrize("shape", [(3, 3), (12, 19)])
@pytest.mark.parametrize("mode", ["directional", "undamped_relative"])
def test_exact_zero_factor_clears_reused_outputs_and_preserves_nonzero_and_invalid_worlds(
    shape, mode
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("warp")
    from embodik.gpu.wbc._runtime.warp_directional_srinv import WarpDirectionalSRINV

    rows, columns = shape
    native = WarpDirectionalSRINV(rows, columns, batch_capacity=4, inverse_mode=mode, device="cuda")
    matrix = torch.eye(rows, columns, device="cuda").repeat(4, 1, 1)
    rhs = torch.ones((4, rows), device="cuda")
    # Populate every output with a nonzero factor first.
    native.solve(matrix, rhs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = native.solve(matrix, rhs)
    matrix[1].zero_()
    matrix[2].zero_()
    matrix[2, 0, 0] = 1e-12  # A small matrix is not an exact zero matrix.
    matrix[3, 0, 0] = float("nan")
    graph.replay()
    torch.cuda.synchronize()
    assert native.status[:4].cpu().tolist() == [0, 0, 0, 1]
    torch.testing.assert_close(output[0, :rows], rhs[0], atol=1e-6, rtol=0)
    assert output[2].abs().max() > 0
    for world in (1, 3):
        assert torch.count_nonzero(output[world]) == 0
        assert torch.count_nonzero(native.inverse[world]) == 0
        assert torch.count_nonzero(native.undamped_inverse[world]) == 0
    rhs[1, 0] = float("inf")
    graph.replay()
    torch.cuda.synchronize()
    assert native.status[1].item() == 1
    assert torch.count_nonzero(output[1]) == 0
    rhs[1, 0] = 1
    matrix[3].zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert native.status[:4].cpu().tolist() == [0, 0, 0, 0]
    # The inverse-only API must also clear all reused factor outputs.
    inverse = native.solve(matrix)
    for world in (1, 3):
        assert torch.count_nonzero(inverse[world]) == 0
        assert torch.count_nonzero(native.undamped_inverse[world]) == 0


def test_unused_completion_factors_are_zero_without_changing_scaled_solution():
    torch = pytest.importorskip("torch")
    from embodik.gpu.wbc._runtime.gpu_priority import complete_consistent_redundant_velocity

    matrix = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    target = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    lower = torch.full((1, 3), -4.0, dtype=torch.float64)
    upper = torch.tensor([[0.5, 4.0, 4.0]], dtype=torch.float64)
    factors = []

    def inverse(value, tolerance):
        factors.append(value.clone())
        return torch.linalg.pinv(value, rtol=tolerance)

    scaled = target * 0.5
    result = complete_consistent_redundant_velocity(
        matrix, target, lower, upper, target, scaled, inverse_function=inverse
    )
    torch.testing.assert_close(result, scaled, atol=0, rtol=0)
    torch.testing.assert_close(factors[0], matrix, atol=0, rtol=0)
    assert len(factors) == 3 and all(torch.count_nonzero(value) == 0 for value in factors[1:])


def test_unused_completion_does_not_hide_nonfinite_factor_inputs():
    torch = pytest.importorskip("torch")
    from embodik.gpu.wbc._runtime.gpu_priority import complete_consistent_redundant_velocity

    matrix = torch.eye(3).unsqueeze(0)
    matrix[0, 0, 0] = float("nan")
    factors = []

    def inverse(value, tolerance):
        factors.append(value.clone())
        return torch.zeros_like(value.transpose(-1, -2))

    zero = torch.zeros((1, 3))
    complete_consistent_redundant_velocity(
        matrix,
        zero,
        -torch.ones_like(zero),
        torch.ones_like(zero),
        zero,
        zero,
        inverse_function=inverse,
    )
    assert len(factors) == 3 and all(torch.isnan(value).any() for value in factors)


def test_accepted_violator_skips_only_its_unused_blocker_factor():
    torch = pytest.importorskip("torch")
    from embodik.gpu.wbc._runtime.gpu_priority import complete_consistent_redundant_velocity

    matrix = torch.tensor([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], dtype=torch.float64)
    matrix = matrix.repeat(3, 1, 1)
    matrix[2, 0, 0] = float("nan")
    target = torch.tensor([[1.0, 2.0]], dtype=torch.float64).repeat(3, 1)
    unscaled = torch.full((3, 3), 1.0 / 3.0, dtype=torch.float64)
    lower = -torch.ones_like(unscaled)
    upper = torch.tensor([[0.0, 1.0, 1.0], [0.0, 0.2, 0.5], [0.0, 1.0, 1.0]], dtype=torch.float64)
    factors = []
    factor_status_ok = torch.ones(3, dtype=torch.bool)

    def inverse(value, tolerance):
        factors.append(value.clone())
        factor_status_ok.logical_and_(torch.isfinite(value).all(dim=(-2, -1)))
        return torch.linalg.pinv(torch.nan_to_num(value), rtol=tolerance)

    result = complete_consistent_redundant_velocity(
        matrix,
        target,
        lower,
        upper,
        unscaled,
        torch.zeros_like(unscaled),
        inverse_function=inverse,
    )
    # Lane zero accepts the violator solution; lane one rejects it because its
    # nonzero pinned joint prevents partial scaling, then accepts the blocker.
    expected = torch.tensor(
        [[0.0, 0.5, 0.5], [0.0, 0.2, 0.2], [0.0, 0.0, 0.0]], dtype=torch.float64
    )
    torch.testing.assert_close(result, expected, atol=1e-15, rtol=0)
    assert len(factors) == 3
    assert torch.count_nonzero(factors[1][0]) > 0
    assert torch.count_nonzero(factors[2][0]) == 0
    assert torch.count_nonzero(factors[2][1]) > 0
    assert all(torch.isnan(value[2]).any() for value in factors)
    assert factor_status_ok.tolist() == [True, True, False]
