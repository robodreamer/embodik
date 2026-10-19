"""Native bound release agrees with an independent maximum-scale oracle."""

import numpy as np
import pytest
from scipy.optimize import linprog


def _runtime():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("warp")
    from embodik.gpu.wbc._runtime.warp_bound_release import WarpBoundRelease

    return torch, WarpBoundRelease


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_opt_in_refinement_improves_a_feasible_partial_scale(dtype: str) -> None:
    torch, native = _runtime()
    convert = lambda value: torch.tensor(value, dtype=getattr(torch, dtype), device="cuda")
    jacobian = convert([[[2.0, 1.0, -3.0], [-3.0, 1.0, 2.0]]])
    target = convert([[-2.0, -2.0]])
    lower, upper = convert([[-1.0, -1.0, 0.0]]), convert([[1.0, 1.0, 1.0]])
    original = convert([[0.5, -1.0, 0.5]])
    baseline = native(2, 3, batch_capacity=1, device="cuda")
    torch.testing.assert_close(baseline.solve(jacobian, target, lower, upper, original), original)
    solver = native(2, 3, batch_capacity=1, device="cuda", refine_partial_scale=True)
    result = solver.solve(jacobian, target, lower, upper, original)
    torch.testing.assert_close(result, convert([[1.0, -1.0, 1.0]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        (jacobian @ result.unsqueeze(-1)).squeeze(-1), target, atol=1e-6, rtol=0
    )
    assert solver.recovered.cpu().tolist() == [True]


@pytest.mark.parametrize("from_origin", [False, True])
def test_partial_scale_refinement_graph_preserves_other_worlds(from_origin: bool) -> None:
    torch, native = _runtime()
    convert = lambda value: torch.tensor(value, dtype=torch.float64, device="cuda")
    jacobian = convert([[[2.0, 1.0, -3.0], [-3.0, 1.0, 2.0]]]).repeat(4, 1, 1)
    target = convert([[-2.0, -2.0]]).repeat(4, 1)
    lower, upper = convert([[-1.0, -1.0, 0.0]]).repeat(4, 1), convert([[1.0, 1.0, 1.0]]).repeat(
        4, 1
    )
    original = convert([[0.5, -1.0, 0.5]]).repeat(4, 1)
    original[1] = convert([1.0, -1.0, 1.0])  # Already full scale.
    original[2] = convert([0.5, -0.5, 0.25])  # Feasible but not on this task ray.
    target[3, 0] = float("nan")
    solver = native(
        2,
        3,
        batch_capacity=4,
        device="cuda",
        refine_partial_scale=True,
        refine_from_origin=from_origin,
    )
    for _ in range(3):
        solver.solve(jacobian, target, lower, upper, original)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = solver.solve(jacobian, target, lower, upper, original)
    for initial_scale in (0.5, 0.75):
        original[0] = convert([2.0 * initial_scale - 1.0, -1.0, 2.0 * initial_scale - 1.0])
        graph.replay()
        torch.testing.assert_close(result[0], convert([1.0, -1.0, 1.0]), atol=1e-8, rtol=0)
        torch.testing.assert_close(result[1:], original[1:], atol=0, rtol=0)
        assert solver.recovered.cpu().tolist() == [True, False, False, False]


def test_origin_refinement_handles_a_rounded_partial_ray_without_relaxing_output():
    torch, native = _runtime()
    convert = lambda v: torch.tensor(v, dtype=torch.float32, device="cuda")
    j = convert([[[2.0, 1.0, -3.0], [-3.0, 1.0, 2.0]]])
    target = convert([[-2.0, -2.0]])
    lower, upper = convert([[-1.0, -1.0, 0.0]]), convert([[1.0, 1.0, 1.0]])
    original = convert([[0.5, -1.0, 0.5000001]])
    baseline = native(2, 3, batch_capacity=1, device="cuda", refine_partial_scale=True)
    torch.testing.assert_close(
        baseline.solve(j, target, lower, upper, original), original, atol=0, rtol=0
    )
    solver = native(
        2,
        3,
        batch_capacity=1,
        device="cuda",
        refine_partial_scale=True,
        refine_from_origin=True,
        refinement_ray_tolerance=1e-5,
    )
    result = solver.solve(j, target, lower, upper, original)
    torch.testing.assert_close(result, convert([[1.0, -1.0, 1.0]]), atol=1e-6, rtol=0)
    torch.testing.assert_close((j @ result[..., None]).squeeze(-1), target, atol=1e-6, rtol=0)
    assert ((result >= lower) & (result <= upper)).all().item()
    assert solver.recovered[0].item()


def test_origin_refinement_preserves_a_box_that_excludes_zero():
    torch, native = _runtime()
    convert = lambda v: torch.tensor(v, dtype=torch.float64, device="cuda")
    j, target = convert([[[1.0, 1.0]]]), convert([[2.0]])
    lower, upper, original = convert([[0.5, 0.0]]), convert([[1.0, 1.0]]), convert([[0.5, 0.5]])
    solver = native(
        1, 2, batch_capacity=1, device="cuda", refine_partial_scale=True, refine_from_origin=True
    )
    result = solver.solve(j, target, lower, upper, original)
    torch.testing.assert_close(result, upper, atol=1e-8, rtol=0)
    assert (result >= lower).all().item()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_native_bound_release_matches_lp(dtype: str) -> None:
    torch, native = _runtime()
    rng = np.random.default_rng(20261009)
    matrices = rng.normal(size=(64, 2, 4))
    targets = rng.normal(size=(64, 2))
    lower = -rng.uniform(0.1, 1.0, size=(64, 4))
    upper = rng.uniform(0.1, 1.0, size=(64, 4))
    for index in range(64):
        lower[index, index % 4] = 0.0
        upper[index, (index + 1) % 4] = 0.0
    convert = lambda value: torch.as_tensor(value, dtype=getattr(torch, dtype), device="cuda")
    jacobian, target, lo, hi = map(convert, (matrices, targets, lower, upper))
    solver = native(2, 4, batch_capacity=64, device="cuda")
    output = solver.solve(jacobian, target, lo, hi, torch.zeros_like(lo)).cpu().numpy()
    # Use the actual dtype-rounded problem, not a different float64 oracle.
    for j, b, l, u, velocity in zip(
        jacobian.cpu().numpy(), target.cpu().numpy(), lo.cpu().numpy(), hi.cpu().numpy(), output
    ):
        reference = linprog(
            [0.0, 0.0, 0.0, 0.0, -1.0],
            A_eq=np.column_stack([j, -b]),
            b_eq=np.zeros(2),
            bounds=list(zip(l, u)) + [(0.0, 1.0)],
            method="highs",
        )
        assert reference.success
        scale = float((j @ velocity) @ b / (b @ b))
        np.testing.assert_allclose(scale, reference.x[-1], atol=2e-6)
        np.testing.assert_allclose(j @ velocity, scale * b, atol=2e-6)
        assert np.all(velocity >= l - 2e-6)
        assert np.all(velocity <= u + 2e-6)


def test_bound_release_graph_replays_and_isolates_worlds() -> None:
    torch, native = _runtime()
    jacobian = torch.tensor(
        [[[-0.25, -0.16, -0.17, 0.0], [-0.93, -0.93, 0.0, -0.1]]], device="cuda"
    ).repeat(3, 1, 1)
    target = torch.tensor([[-0.04, -0.02]], device="cuda").repeat(3, 1)
    lower = torch.tensor([[-1.0, 0.0, -1.0, -1.0]], device="cuda").repeat(3, 1)
    upper = torch.tensor([[0.0, 1.0, 1.0, 0.0]], device="cuda").repeat(3, 1)
    original = torch.zeros_like(lower)
    original[1] = torch.tensor([-0.1, 0.1, 0.1, -0.1], device="cuda")
    target[2, 0] = float("nan")
    solver = native(2, 4, batch_capacity=3, device="cuda")
    for _ in range(3):
        solver.solve(jacobian, target, lower, upper, original)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = solver.solve(jacobian, target, lower, upper, original)
    for scale in (1.0, 0.5, 2.0):
        target[0] = torch.tensor([-0.04, -0.02], device="cuda") * scale
        graph.replay()
        torch.testing.assert_close(jacobian[0] @ output[0], target[0], atol=1e-6, rtol=0)
        torch.testing.assert_close(output[1:], original[1:], atol=0, rtol=0)
        assert solver.recovered.cpu().tolist() == [True, False, False]


def test_required_inward_seed_replays_in_cuda_graph() -> None:
    torch, native = _runtime()
    jacobian = torch.tensor([[[1.0, 1.0, 0.0]]], dtype=torch.float64, device="cuda")
    target = torch.tensor([[0.5]], dtype=torch.float64, device="cuda")
    lower, upper = -torch.ones((1, 3), device="cuda"), torch.ones((1, 3), device="cuda")
    original = torch.zeros_like(lower)
    solver = native(1, 3, batch_capacity=1, device="cuda")
    for _ in range(3):
        solver.solve(jacobian, target, lower, upper, original)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = solver.solve(jacobian, target, lower, upper, original)
    for direction in (-1.0, 1.0, -1.0):
        lower[0, 2] = 0.01 if direction > 0 else -1.0
        upper[0, 2] = 1.0 if direction > 0 else -0.01
        original[0, 2] = direction * 0.01
        graph.replay()
        torch.testing.assert_close(jacobian[0] @ output[0].double(), target[0], atol=1e-7, rtol=0)
        assert bool(solver.recovered.cpu()[0])
        assert bool(torch.all(output >= lower - 1e-7))
        assert bool(torch.all(output <= upper + 1e-7))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_moving_realization_repair_graph_replays_without_rescaling(dtype: str) -> None:
    torch, native = _runtime()
    precision = getattr(torch, dtype)
    jacobian = torch.ones((3, 1, 2), dtype=precision, device="cuda")
    target = torch.full((3, 1), 0.8, dtype=precision, device="cuda")
    lower, upper = -torch.ones((3, 2), dtype=precision, device="cuda"), torch.ones(
        (3, 2), dtype=precision, device="cuda"
    )
    upper[:, 0] = 0.0
    original = torch.tensor([[0.6, -0.1], [-0.1, 0.6], [2.0, 2.0]], dtype=precision, device="cuda")
    solver = native(1, 2, batch_capacity=3, device="cuda")
    for _ in range(3):
        solver.solve(jacobian, target, lower, upper, original)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = solver.solve(jacobian, target, lower, upper, original)
    for achieved in (0.5, 0.25, -0.25, 0.5):
        original[0, 1] = achieved - 0.6
        graph.replay()
        torch.testing.assert_close(
            jacobian[0] @ output[0], jacobian[0] @ original[0], atol=2e-6, rtol=0
        )
        assert bool(torch.all(output[0] >= lower[0] - 2e-6))
        assert bool(torch.all(output[0] <= upper[0] + 2e-6))
        torch.testing.assert_close(output[1:], original[1:], atol=0, rtol=0)
        assert solver.recovered.cpu().tolist() == [True, False, False]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_compensated_seed_graph_replay_preserves_world_isolation(dtype: str) -> None:
    torch, native = _runtime()
    precision = getattr(torch, dtype)
    jacobian = torch.tensor(
        [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], dtype=precision, device="cuda"
    ).repeat(5, 1, 1)
    target = torch.tensor(
        [[0.0, 0.0], [0.5, 1.0], [0.2, 0.4], [0.0, 0.0], [float("nan"), 0.0]],
        dtype=precision,
        device="cuda",
    )
    lower, upper = -torch.ones((5, 3), dtype=precision, device="cuda"), torch.ones(
        (5, 3), dtype=precision, device="cuda"
    )
    lower[:, 0] = 0.01
    lower[3] = 0.1
    original = torch.zeros_like(lower)
    original[2, 0] = 0.1
    solver = native(2, 3, batch_capacity=5, device="cuda")
    for _ in range(3):
        solver.solve(jacobian, target, lower, upper, original)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = solver.solve(jacobian, target, lower, upper, original)
    for direction, command in [(-1.0, [0.0, 0.0]), (1.0, [0.1, 0.5]), (-1.0, [0.25, 0.5])]:
        lower[0, 0] = 0.01 if direction > 0 else -1.0
        upper[0, 0] = 1.0 if direction > 0 else -0.01
        target[0] = torch.tensor(command, dtype=precision, device="cuda")
        graph.replay()
        assert solver.recovered.cpu().tolist() == [True, True, False, False, False]
        achieved = jacobian[:2] @ output[:2, :, None]
        expected = target[:2].clone()
        if command == [0.1, 0.5]:
            expected[0] = 0.0  # Rank-inconsistent ray has maximum scale zero.
        torch.testing.assert_close(achieved[..., 0], expected, atol=2e-6, rtol=0)
        torch.testing.assert_close(output[2:], original[2:], atol=0, rtol=0)
        assert bool(torch.all(output[:2] >= lower[:2] - 2e-6))
        assert bool(torch.all(output[:2] <= upper[:2] + 2e-6))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_positive_ray_graph_recovers_after_unreachable_target(dtype: str) -> None:
    torch, native = _runtime()
    precision = getattr(torch, dtype)
    jacobian = torch.ones((2, 1, 2), dtype=precision, device="cuda")
    target = torch.tensor([[1.0], [-1.0]], dtype=precision, device="cuda")
    lower = torch.tensor([[0.2, 0.2], [-1.0, -1.0]], dtype=precision, device="cuda")
    upper = torch.tensor([[1.0, 1.0], [-0.2, -0.2]], dtype=precision, device="cuda")
    original = torch.zeros_like(lower)
    solver = native(1, 2, batch_capacity=2, device="cuda")
    for _ in range(3):
        solver.solve(jacobian, target, lower, upper, original)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = solver.solve(jacobian, target, lower, upper, original)
    for commands in ([1.0, -1.0], [-1.0, 1.0], [3.0, -3.0], [0.0, 0.0], [1.0, -1.0]):
        target.copy_(torch.tensor(commands, dtype=precision, device="cuda")[:, None])
        graph.replay()
        flags = solver.recovered.cpu().tolist()
        assert flags == [commands[0] > 0, commands[1] < 0]
        for world, recovered in enumerate(flags):
            if not recovered:
                torch.testing.assert_close(output[world], original[world], atol=0, rtol=0)
            else:
                expected = max(-2.0, min(2.0, commands[world]))
                torch.testing.assert_close(
                    output[world].sum(),
                    torch.tensor(expected, dtype=precision, device="cuda"),
                    atol=2e-6,
                    rtol=0,
                )
                assert bool(torch.all(output[world] >= lower[world] - 2e-6))
                assert bool(torch.all(output[world] <= upper[world] + 2e-6))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("required_inward", [False, True])
def test_rectangular_near_dependent_equalities_keep_phase_and_world_isolation(
    dtype: str, required_inward: bool
) -> None:
    torch, native = _runtime()
    precision = getattr(torch, dtype)
    jacobian = torch.eye(12, 19, dtype=precision, device="cuda").repeat(5, 1, 1)
    target = torch.full((5, 12), 2.0, dtype=precision, device="cuda")
    # Exercise both sides of the normalized equality-rank cutoff. Each final
    # row is almost the first row, with a separate tiny independent component.
    for world, epsilon in enumerate((0.0, 9e-11, 1.1e-10, 1e-8)):
        jacobian[world, -1].zero_()
        jacobian[world, -1, 0] = 1.0
        jacobian[world, -1, 11] = epsilon
        target[world, -1] = 2.0 + 2.0 * epsilon
    lower = torch.full((5, 19), -0.5, dtype=precision, device="cuda")
    upper = torch.full_like(lower, 0.5)
    if required_inward:
        lower[:, 0] = 0.01  # A non-null affine seed requires phase I first.
    lower[4, 0] = 0.01
    target[4, 0] = -2.0  # This world's positive ray is infeasible.
    original = torch.zeros_like(lower)
    solver = native(
        12,
        19,
        batch_capacity=5,
        device="cuda",
        refine_partial_scale=True,
        refine_from_origin=True,
    )
    output = solver.solve(jacobian, target, lower, upper, original)
    assert solver.recovered.cpu().tolist() == [True, True, True, True, False]
    torch.testing.assert_close(
        output[:4, 0], torch.full_like(output[:4, 0], 0.5), atol=2e-6, rtol=0
    )
    achieved = (jacobian[:4] @ output[:4, :, None]).squeeze(-1)
    torch.testing.assert_close(achieved, 0.25 * target[:4], atol=2e-6, rtol=0)
    assert bool(torch.all(output[:4] >= lower[:4] - 2e-6))
    assert bool(torch.all(output[:4] <= upper[:4] + 2e-6))
    torch.testing.assert_close(output[4], original[4], atol=0, rtol=0)


def test_rank_deficient_active_set_graph_rebuilds_changed_equality_prefix() -> None:
    torch, native = _runtime()
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(2, 5))
    matrix = np.vstack((matrix, matrix.sum(axis=0)))
    target = rng.normal(size=2) * 2.0
    target = np.r_[target, target.sum()]
    lower, upper = -rng.uniform(0.05, 0.5, 5), rng.uniform(0.05, 0.5, 5)
    lower[4], upper[0] = 0.0, 0.0
    # This rank-deficient problem needs six bound additions and two removals.
    # Changed rows/goals must rebuild the fixed prefix on every graph replay.
    permutation = np.arange(4, -1, -1)
    variants = [
        (matrix, target, lower, upper),
        (-matrix[:, permutation], 0.7 * target, -upper[permutation], -lower[permutation]),
        (matrix, target, lower, upper),
    ]
    convert = lambda value: torch.tensor(value, dtype=torch.float64, device="cuda")[None]
    inputs = [convert(value) for value in variants[0]]
    original = torch.zeros((1, 5), dtype=torch.float64, device="cuda")
    solver = native(
        3, 5, batch_capacity=1, device="cuda", refine_partial_scale=True, refine_from_origin=True
    )
    for _ in range(3):
        solver.solve(*inputs, original)
    torch.cuda.synchronize()
    first = solver.result.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = solver.solve(*inputs, original)
    for index, variant in enumerate(variants):
        for destination, value in zip(inputs, variant):
            destination.copy_(convert(value))
        graph.replay()
        j, t, lo, hi = variant
        oracle = linprog(
            np.r_[np.zeros(5), -1.0],
            A_eq=np.column_stack((j, -t)),
            b_eq=np.zeros(3),
            bounds=list(zip(lo, hi)) + [(0.0, 1.0)],
            method="highs",
        )
        assert oracle.success and oracle.x[-1] > 0
        assert solver.recovered.cpu().tolist() == [True]
        torch.testing.assert_close(
            (inputs[0] @ output[:, :, None]).squeeze(-1),
            oracle.x[-1] * inputs[1],
            atol=1e-8,
            rtol=0,
        )
        assert bool(torch.all(output >= inputs[2] - 1e-8))
        assert bool(torch.all(output <= inputs[3] + 1e-8))
        if index in (0, 2):
            torch.testing.assert_close(output, first, atol=0, rtol=0)
