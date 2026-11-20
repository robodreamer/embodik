"""Native general-row SCALE probe: independent LP optima and per-lane guards."""

import numpy as np
import pytest
from scipy.optimize import linprog

torch = pytest.importorskip("torch")
pytest.importorskip("warp")
from embodik.gpu.wbc._runtime.warp_polytope import WarpPolytopeScale
from embodik.gpu.wbc._runtime.warp_reduced_polytope import WarpReducedPolytopeScale


def solve_case(J, g, C, lo, hi, rl, ru, *, implementation):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    J, g, C, lo, hi, rl, ru = map(np.asarray, (J, g, C, lo, hi, rl, ru))
    solver = implementation(len(g), J.shape[1], len(C), batch=1)
    args = [
        torch.as_tensor(v, device="cuda", dtype=torch.float64)[None].contiguous()
        for v in (J, g, C, lo, hi, rl, ru)
    ]
    v, alpha, status = solver.solve(*args)
    torch.cuda.synchronize()
    return (v[0].cpu().numpy(), alpha.item(), status.item())


@pytest.mark.parametrize(
    "case", ["diamond", "redistribute", "mandatory", "prefix", "zero", "infeasible"]
)
def test_native_rows_against_independent_lp(case, implementation):
    J, g, C, lo, hi, rl, ru = (
        [[1.0, 0.0], [0.0, 1.0]],
        [1.0, 1.0],
        [[1.0, 1.0], [1.0, -1.0]],
        [-1.0, -1.0],
        [1.0, 1.0],
        [-0.5, -0.5],
        [0.5, 0.5],
    )
    if case == "redistribute":
        J, g, C, lo, hi, rl, ru = (
            [[0.0, 1.0]],
            [1.0],
            [[1.0, 0.1]],
            [-0.5, -1.0],
            [0.5, 1.0],
            [-0.5],
            [0.0],
        )
    elif case == "mandatory":
        J, g, C, lo, hi, rl, ru = ([[1.0]], [1.0], [[1.0]], [0.0], [1.0], [0.2], [0.4])
    elif case == "prefix":
        J, g, C, lo, hi, rl, ru = (
            [[0.0, 1.0]],
            [1.0],
            [[1.0, 0.0]],
            [-1.0, -1.0],
            [1.0, 1.0],
            [0.2],
            [0.2],
        )
    elif case == "zero":
        J, g = ([[0.0, 0.0], [0.0, 0.0]], [1.0, 1.0])
    elif case == "infeasible":
        C, rl, ru = ([[0.0, 0.0]], [0.2], [0.4])
    v, alpha, status = solve_case(J, g, C, lo, hi, rl, ru, implementation=implementation)
    J, g, C, lo, hi, rl, ru = map(np.asarray, (J, g, C, lo, hi, rl, ru))
    n = J.shape[1]
    oracle = linprog(
        np.r_[np.zeros(n), -1.0],
        A_eq=np.c_[J, -g],
        b_eq=np.zeros(len(g)),
        A_ub=np.r_[np.c_[C, np.zeros(len(C))], np.c_[-C, np.zeros(len(C))]],
        b_ub=np.r_[ru, -rl],
        bounds=list(zip(lo, hi)) + [(0.0, 1.0)],
        method="highs",
    )
    if not oracle.success:
        assert status == 2
        return
    assert status == 0
    assert alpha == pytest.approx(oracle.x[-1], abs=1e-08)
    np.testing.assert_allclose(J @ v, alpha * g, atol=1e-08, rtol=0)
    assert np.all(v >= lo - 1e-08) and np.all(v <= hi + 1e-08)
    assert np.all(C @ v >= rl - 1e-08) and np.all(C @ v <= ru + 1e-08)


def test_mixed_batch_redundant_rows_and_mandatory_prefix_match_lp(implementation):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(20261021)
    batch, n, m, r = (48, 7, 3, 5)
    J = rng.normal(size=(batch, m, n))
    J[:8, 1] = 2 * J[:8, 0]
    C = rng.normal(size=(batch, r, n))
    C[:, 3] = C[:, 2]
    witness = rng.uniform(-0.4, 0.4, (batch, n))
    g = np.einsum("bmn,bn->bm", J, witness) / 0.4
    image = np.einsum("brn,bn->br", C, witness)
    rl, ru = (image - 0.05, image + 0.05)
    rl[:, -1] = image[:, -1]
    ru[:, -1] = image[:, -1]
    lo, hi = (np.full((batch, n), -1.0), np.full((batch, n), 1.0))
    C[0, 0, 0] = np.nan
    args = [torch.tensor(v, dtype=torch.float64, device="cuda") for v in (J, g, C, lo, hi, rl, ru)]
    solver = implementation(m, n, r, batch=batch)
    v, alpha, status = solver.solve(*args)
    v, alpha, status = (v.cpu().numpy(), alpha.cpu().numpy(), status.cpu().numpy())
    assert status[0] == 1 and np.isfinite(v).all()
    assert np.all(status[1:] == 0), (status, solver.iterations.cpu().tolist())
    for lane in range(1, batch):
        oracle = linprog(
            np.r_[np.zeros(n), -1.0],
            A_eq=np.c_[J[lane], -g[lane]],
            b_eq=np.zeros(m),
            A_ub=np.r_[np.c_[C[lane], np.zeros(r)], np.c_[-C[lane], np.zeros(r)]],
            b_ub=np.r_[ru[lane], -rl[lane]],
            bounds=[(-1.0, 1.0)] * n + [(0.0, 1.0)],
            method="highs",
        )
        assert oracle.success
        assert alpha[lane] == pytest.approx(oracle.x[-1], abs=1e-07), lane
        np.testing.assert_allclose(J[lane] @ v[lane], alpha[lane] * g[lane], atol=1e-08, rtol=0)
        assert np.all(C[lane] @ v[lane] >= rl[lane] - 1e-08)
        assert np.all(C[lane] @ v[lane] <= ru[lane] + 1e-08)
        assert np.max(abs(v[lane])) <= 1 + 1e-08


def test_graph_native_rows_reach_boundary_then_reverse_without_reset(implementation):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    batch = 256
    kw = dict(dtype=torch.float64, device="cuda")
    q, target = (torch.zeros((batch, 2), **kw), torch.ones((batch, 2), **kw))
    J = torch.eye(2, **kw).expand(batch, 2, 2).contiguous()
    C = torch.tensor([[1.0, 1.0], [1.0, -1.0]], **kw)
    lo, hi = (torch.full((2,), -0.5, **kw), torch.full((2,), 0.5, **kw))
    dt = torch.full((batch,), 0.02, **kw)
    lower, upper = (torch.full_like(q, -1.0), torch.full_like(q, 1.0))
    solver = implementation(2, 2, 2, batch=batch)

    def step():
        goal = (10 * (target - q)).clamp(-1.0, 1.0)
        rows, rl, ru = position_rows(C, lo, hi, q, dt)
        velocity, _, _ = solver.solve(J, goal, rows, lower, upper, rl, ru)
        q.add_(dt[:, None] * velocity)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    q.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    q.zero_()
    for _ in range(50):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(q, torch.full_like(q, 0.25), atol=1e-09, rtol=0)
    assert torch.all(solver.status == 0).item()
    target.fill_(-0.1)
    for _ in range(80):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(q, target, atol=1e-06, rtol=0)
    assert torch.all(solver.status == 0).item()
    assert torch.max(abs(q @ C.T)).item() <= 0.5 + 1e-09


@pytest.mark.parametrize("factor", [1e-06, 1.0, 1000000.0])
def test_scaled_mandatory_rows_retain_same_feasible_optimum(factor, implementation):
    J, g = ([[1.0, 1.0, 1.0]], [1.0])
    C = factor * np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]])
    rl, ru = (factor * np.array([0.2, -0.1]), factor * np.array([0.2, 0.1]))
    v, alpha, status = solve_case(
        J, g, C, [0.0] * 3, [1.0] * 3, rl, ru, implementation=implementation
    )
    assert status == 0
    assert alpha == pytest.approx(1.0, abs=1e-08)
    assert np.all(C @ v >= rl - 1e-08) and np.all(C @ v <= ru + 1e-08)


def test_zero_extra_rows_and_interleaved_different_kernel_shapes(implementation):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    a, b = (implementation(1, 1, 0, batch=1), implementation(2, 2, 2, batch=1))
    kw = dict(dtype=torch.float64, device="cuda")
    args_a = [
        torch.tensor(v, **kw)
        for v in (
            [[[1.0]]],
            [[1.0]],
            np.empty((1, 0, 1)),
            [[-1.0]],
            [[0.3]],
            np.empty((1, 0)),
            np.empty((1, 0)),
        )
    ]
    args_b = [
        torch.tensor(v, **kw)
        for v in (
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[1.0, 1.0]],
            [[[1.0, 1.0], [1.0, -1.0]]],
            [[-1.0, -1.0]],
            [[1.0, 1.0]],
            [[-0.5, -0.5]],
            [[0.5, 0.5]],
        )
    ]
    for solver, args, expected in [
        (a, args_a, 0.3),
        (b, args_b, 0.25),
        (a, args_a, 0.3),
        (b, args_b, 0.25),
    ]:
        _, alpha, status = solver.solve(*args)
        assert status.item() == 0
        assert alpha.item() == pytest.approx(expected, abs=1e-08)


def test_overflow_cannot_be_reported_as_valid_ray(implementation):
    v, _, status = solve_case(
        [[1.0]], [1e308], [[1.0]], [-1.0], [1.0], [-1.0], [1.0], implementation=implementation
    )
    assert status != 0
    assert np.isfinite(v).all()


@pytest.fixture(params=[WarpPolytopeScale, WarpReducedPolytopeScale])
def implementation(request):
    return request.param


def position_rows(C, lower, upper, q, dt):
    value = q @ C.T
    return (
        C.expand(q.shape[0], *C.shape),
        (torch.minimum(value, lower) - value) / dt[:, None],
        (torch.maximum(value, upper) - value) / dt[:, None],
    )
