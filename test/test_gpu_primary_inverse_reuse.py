"""Immediate primary-factor reuse preserves independent completion fallbacks."""

from types import SimpleNamespace

import pytest


def _core(torch, *, batch=4, rows=3, columns=3, device="cpu"):
    from embodik.gpu.wbc._runtime.multi_pose_solver import DeviceResidentMultiFramePoseSolver

    core = DeviceResidentMultiFramePoseSolver.__new__(DeviceResidentMultiFramePoseSolver)
    core.torch = torch
    core.device = torch.device(device)
    core.batch_size, core.task_rows, core.velocity_dim = batch, rows, columns
    core._locked_active_columns = ()
    core._bound_release = None
    core.config = SimpleNamespace(srinv_tolerance=0.1, srinv_damping=0.1)
    settings = dict(
        rows=rows,
        columns=columns,
        batch_capacity=batch,
        relative_rank_tolerance=1e-6,
        max_sweeps=32,
        convergence=1e-12,
        device=core.device,
    )
    core._warp_srinv = SimpleNamespace(
        **settings,
        inverse_mode="directional",
        undamped_inverse=torch.eye(columns, rows, device=device).repeat(batch, 1, 1),
        status=torch.zeros(batch, dtype=torch.int32, device=device),
    )
    core._bounded_completion_inverse = SimpleNamespace(**settings, inverse_mode="undamped_relative")
    return core


@pytest.mark.parametrize(
    "mismatch",
    [
        "locks",
        "backend",
        "completion",
        "dtype",
        "stride",
        "shape",
        "primary_mode",
        "completion_mode",
        "primary_rank",
        "completion_rank",
        "sweeps",
        "convergence",
        "primary_device",
        "completion_device",
        "cache_dtype",
        "cache_shape",
        "primary_rows",
        "completion_columns",
        "primary_capacity",
    ],
)
def test_primary_reuse_requires_matching_factor_contract(mismatch):
    torch = pytest.importorskip("torch")
    core = _core(torch)
    matrix = torch.eye(3).repeat(4, 1, 1)
    assert core._matching_primary_completion_factor(matrix) is core._warp_srinv
    primary, completion = core._warp_srinv, core._bounded_completion_inverse
    if mismatch == "locks":
        core._locked_active_columns = (0,)
    elif mismatch == "backend":
        core._warp_srinv = None
    elif mismatch == "completion":
        core._bounded_completion_inverse = None
    elif mismatch == "dtype":
        matrix = matrix.double()
    elif mismatch == "stride":
        matrix = matrix.transpose(-2, -1)
    elif mismatch == "shape":
        matrix = matrix[:2]
    elif mismatch == "primary_mode":
        primary.inverse_mode = "undamped_relative"
    elif mismatch == "completion_mode":
        completion.inverse_mode = "directional"
    elif mismatch == "primary_rank":
        primary.relative_rank_tolerance = 1e-5
    elif mismatch == "completion_rank":
        completion.relative_rank_tolerance = 1e-5
    elif mismatch == "sweeps":
        primary.max_sweeps = 16
    elif mismatch == "convergence":
        primary.convergence = 1e-10
    elif mismatch == "primary_device":
        primary.device = torch.device("meta")
    elif mismatch == "completion_device":
        completion.device = torch.device("meta")
    elif mismatch == "cache_dtype":
        primary.undamped_inverse = primary.undamped_inverse.double()
    elif mismatch == "cache_shape":
        primary.undamped_inverse = primary.undamped_inverse[:2]
    elif mismatch == "primary_rows":
        primary.rows = 2
    elif mismatch == "completion_columns":
        completion.columns = 2
    elif mismatch == "primary_capacity":
        primary.batch_capacity = 8
    assert core._matching_primary_completion_factor(matrix) is None


def test_initial_cache_masks_only_good_lanes_and_preserves_fresh_statuses():
    torch = pytest.importorskip("torch")
    core = _core(torch)
    matrix = torch.eye(3).repeat(4, 1, 1)
    matrix[3, 0, 0] = float("nan")
    primary = core._warp_srinv
    primary.status[:] = torch.tensor([0, 1, 0, 1])
    primary.undamped_inverse[1].zero_()  # e.g. invalid RHS, otherwise valid matrix
    primary.undamped_inverse[2, 0, 0] = float("inf")  # unchecked undamped overflow
    primary.undamped_inverse[3].zero_()
    factors = []

    def solve(value):
        factors.append(value.clone())
        valid = torch.isfinite(value).all(dim=(-2, -1))
        core._bounded_completion_inverse.status = (~valid).to(torch.int32)
        return torch.linalg.pinv(torch.nan_to_num(value))

    core._bounded_completion_inverse.solve = solve
    target = torch.full((4, 3), 0.2)
    lower, upper = -torch.ones_like(target), torch.ones_like(target)
    baseline = core._finish_bounded_velocity(matrix, target, lower, upper, target)
    baseline_ok = core._bounded_completion_ok.clone()
    factors.clear()
    reused = core._finish_bounded_velocity(
        matrix, target, lower, upper, target, primary_factor=primary
    )
    torch.testing.assert_close(reused, baseline, atol=0, rtol=0)
    torch.testing.assert_close(core._bounded_completion_ok, baseline_ok, atol=0, rtol=0)
    assert torch.count_nonzero(factors[0][0]) == 0
    torch.testing.assert_close(factors[0][1:3], matrix[1:3], atol=0, rtol=0)
    assert torch.isnan(factors[0][3]).any()
    assert baseline_ok.tolist() == [True, True, True, False]
    # Bad RHS keeps the original independent matrix-only success. The
    # primary's existing publication veto is still false for this lane.
    assert not bool(((primary.status == 0) & core._bounded_completion_ok)[1])


@pytest.mark.parametrize("shape", [(3, 3), (12, 19)])
def test_native_primary_undamped_factor_exactly_matches_independent_valid_factor(shape):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("warp")
    from embodik.gpu.wbc._runtime.warp_directional_srinv import WarpDirectionalSRINV

    rows, columns = shape
    generator = torch.Generator().manual_seed(143)
    matrix = torch.randn((8, rows, columns), generator=generator).cuda()
    matrix[0].zero_()
    matrix[1] = torch.eye(rows, columns, device="cuda")
    matrix[2, -1] = matrix[2, 0]  # dependent rows
    matrix[3] = torch.eye(rows, columns, device="cuda")
    matrix[3, -1].zero_()
    matrix[3, rows - 1, rows - 1] = 0.9e-6  # below the common relative cutoff
    matrix[4] = matrix[3]
    matrix[4, rows - 1, rows - 1] = 1.1e-6
    matrix[5] *= 1e-4
    matrix[6] *= 1e4
    matrix[7, 0, 0] = float("nan")
    primary = WarpDirectionalSRINV(rows, columns, batch_capacity=8, device="cuda")
    fresh = WarpDirectionalSRINV(
        rows, columns, batch_capacity=8, inverse_mode="undamped_relative", device="cuda"
    )
    primary.solve(matrix, torch.ones((8, rows), device="cuda"))
    fresh.solve(matrix)
    valid = (primary.status == 0) & (fresh.status == 0)
    assert valid[:7].all() and not valid[7]
    torch.testing.assert_close(
        primary.undamped_inverse[valid], fresh.inverse[valid], atol=0, rtol=0
    )


def test_native_reused_completion_graph_replays_changed_inputs_and_clears_failures():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("warp")
    from embodik.gpu.wbc._runtime.warp_directional_srinv import WarpDirectionalSRINV

    cores = [_core(torch, device="cuda") for _ in range(2)]
    for core in cores:
        core._warp_srinv = WarpDirectionalSRINV(3, 3, batch_capacity=4, device="cuda")
        core._bounded_completion_inverse = WarpDirectionalSRINV(
            3, 3, batch_capacity=4, inverse_mode="undamped_relative", device="cuda"
        )
    original = torch.tensor([[[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [0.0, 0.0, 1.0]]], device="cuda")
    matrix = original.repeat(4, 1, 1)
    target = torch.tensor([[1.0, 2.0, 0.3]], device="cuda").repeat(4, 1)
    lower = torch.full((4, 3), -2.0, device="cuda")
    upper = torch.tensor([[2.0, 0.0, 2.0]], device="cuda").repeat(4, 1)

    def solve(core, reuse):
        if reuse:
            value = core._native_velocity(target, matrix, lower, upper)
        else:
            velocity = core._warp_srinv.solve(matrix, target)
            value = core._finish_bounded_velocity(matrix, target, lower, upper, velocity)
        ok = core._bounded_completion_ok & (core._warp_srinv.status[:4] == 0)
        return value.clone(), ok.clone()

    for _ in range(3):
        for index, core in enumerate(cores):
            solve(core, bool(index))
    torch.cuda.synchronize()
    graphs, outputs = [], []
    for index, core in enumerate(cores):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = solve(core, bool(index))
        graphs.append(graph)
        outputs.append(output)
    for changed in (True, False):
        matrix.copy_(original.repeat(4, 1, 1))
        if changed:
            matrix[1, 0, 0] = float("nan")
            matrix[2].zero_()
            matrix[3] = torch.eye(3, device="cuda") * 1e-39
        for graph in graphs:
            graph.replay()
        torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=0, rtol=0)
        torch.testing.assert_close(outputs[0][1], outputs[1][1], atol=0, rtol=0)
        if changed:
            assert not bool(outputs[1][1][1])
            assert not bool(outputs[1][1][3])  # undamped overflow, fresh status veto
            assert cores[1]._warp_srinv.status[3] == 0
            assert not bool(torch.isfinite(cores[1]._warp_srinv.undamped_inverse[3]).all())
            assert not bool(cores[1]._bounded_completion_ok[3])
        else:
            assert bool(outputs[1][1].all())
