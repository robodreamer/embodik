"""Undamped factors retain both output buffers, rank policy and status clearing."""

import pytest


def _native_types():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("warp")
    from embodik.gpu.wbc._runtime.warp_directional_srinv import WarpDirectionalSRINV

    return torch, WarpDirectionalSRINV


@pytest.mark.parametrize("shape", [(3, 3), (12, 19)])
def test_undamped_buffers_and_rhs_match_original_spectrum_across_graph_replays(shape):
    torch, factor_type = _native_types()
    rows, columns = shape
    generator = torch.Generator().manual_seed(314)
    original = torch.randn((8, rows, columns), generator=generator).cuda()
    original[0] = torch.eye(rows, columns, device="cuda")
    original[1].zero_()
    original[3, -1] = original[3, 0]
    for index, diagonal in ((4, 0.9e-6), (5, 1.1e-6)):
        original[index] = torch.eye(rows, columns, device="cuda")
        original[index, -1].zero_()
        original[index, rows - 1, rows - 1] = diagonal
    original[6, 0, 0] = float("nan")
    original[7] = torch.eye(rows, columns, device="cuda") * 1e-39
    matrix = original.clone()
    rhs = torch.ones((8, rows), device="cuda")
    native = factor_type(rows, columns, batch_capacity=8, inverse_mode="undamped_relative")
    # The unchanged directional path independently constructs its undamped
    # spectrum, including the same relative rank cutoff and Jacobi ordering.
    reference = factor_type(rows, columns, batch_capacity=8)
    for _ in range(3):
        native.solve(matrix, rhs)
        reference.solve(matrix)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        solution = native.solve(matrix, rhs)
        reference.solve(matrix)
    for changed in (False, True, False):
        matrix.copy_(original)
        if changed:
            matrix[6] = torch.eye(rows, columns, device="cuda")
            matrix[7].zero_()
            rhs.mul_(-0.5)
        graph.replay()
        torch.testing.assert_close(native.inverse, native.undamped_inverse, atol=0, rtol=0)
        torch.testing.assert_close(
            native.inverse[:6], reference.undamped_inverse[:6], atol=0, rtol=0
        )
        assert native.status[:6].tolist() == [0] * 6
        assert native.status[6:].tolist() == ([0, 0] if changed else [1, 1])
        # Failed factors clear both buffers and the RHS result on reused storage.
        failed = native.status != 0
        assert torch.count_nonzero(native.inverse[failed]) == 0
        assert torch.count_nonzero(solution[failed]) == 0
        expected = (native.inverse @ rhs.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(solution, expected, atol=1e-5, rtol=5e-6)
    rhs[1, 0] = float("inf")  # Zero-matrix RHS validation remains active.
    graph.replay()
    assert native.status[1].item() == 1
    assert torch.count_nonzero(solution[1]) == 0


def test_undamped_determinant_overflow_retains_status_and_clears_outputs():
    torch, factor_type = _native_types()
    native = factor_type(12, 19, inverse_mode="undamped_relative")
    matrix = torch.eye(12, 19, device="cuda").unsqueeze(0)
    rhs = torch.ones((1, 12), device="cuda")
    native.solve(matrix, rhs)
    assert native.inverse.abs().max() == 1
    # Its pseudoinverse is representable, but the existing determinant/global
    # regularization arithmetic overflows and must continue vetoing the solve.
    matrix.mul_(1e35)
    native.solve(matrix, rhs)
    assert native.status.item() == 1
    for output in (native.inverse, native.undamped_inverse, native.solution):
        assert torch.count_nonzero(output) == 0
    matrix.zero_()
    native.solve(matrix)
    assert native.status.item() == 0
    assert torch.count_nonzero(native.inverse) == 0


def test_undamped_nonconvergence_status_matches_directional_path():
    torch, factor_type = _native_types()
    matrix = torch.randn((2, 12, 19), generator=torch.Generator().manual_seed(9)).cuda()
    factors = [
        factor_type(12, 19, batch_capacity=2, inverse_mode=mode, max_sweeps=1)
        for mode in ("directional", "undamped_relative")
    ]
    for factor in factors:
        factor.solve(matrix)
        assert factor.status.tolist() == [2, 2]
        assert torch.count_nonzero(factor.inverse) == 0
        assert torch.count_nonzero(factor.undamped_inverse) == 0


@pytest.mark.parametrize("shape", [(3, 3), (12, 19)])
def test_undamped_nonfinite_scan_preserves_world_isolation_and_rhs_vetoes(shape):
    torch, factor_type = _native_types()
    rows, columns = shape
    matrix = torch.eye(rows, columns, device="cuda").repeat(5, 1, 1)
    rhs = torch.ones((5, rows), device="cuda")
    native = factor_type(rows, columns, batch_capacity=5, inverse_mode="undamped_relative")
    native.solve(matrix, rhs)  # Reused outputs initially hold nonzero values.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        solution = native.solve(matrix, rhs)
    matrix[0, 0, 0] = float("nan")
    matrix[1, -1, -1] = float("inf")  # Validate the entire rectangular input.
    matrix[2].zero_()
    rhs[2:4, 0] = float("nan")  # Both zero and nonzero matrix RHS checks.
    graph.replay()
    assert native.status.tolist() == [1, 1, 1, 1, 0]
    for output in (solution, native.inverse, native.undamped_inverse):
        assert torch.count_nonzero(output[:4]) == 0
        assert torch.count_nonzero(output[4]) > 0
    matrix.copy_(torch.eye(rows, columns, device="cuda").repeat(5, 1, 1))
    rhs.fill_(1)
    graph.replay()
    assert native.status.tolist() == [0] * 5
    torch.testing.assert_close(native.inverse, native.undamped_inverse, atol=0, rtol=0)
    assert torch.count_nonzero(solution[:, :rows] - 1) == 0
