"""Explicit partial refinement can restart rounded float32 seeds safely."""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from embodik.gpu.wbc._runtime.multi_pose_solver import (
    DeviceResidentMultiFramePoseSolver,
    MultiFramePoseSolveConfig,
)
from embodik.gpu.wbc.solver import GpuWbcMultiFrameSolver


def test_refinement_config_defaults_and_public_signature() -> None:
    config = MultiFramePoseSolveConfig()
    defaults = dict(
        refine_partial_scale=False, refine_from_origin=False, refinement_ray_tolerance=1e-8
    )
    signature = inspect.signature(GpuWbcMultiFrameSolver.__init__)
    for name, value in defaults.items():
        assert getattr(config, name) == value
        assert signature.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default == value


@pytest.mark.parametrize("backend", ["fi_pesns", "torch_srinv"])
def test_refinement_rejects_unsupported_backend(backend: str) -> None:
    with pytest.raises(ValueError, match="requires warp_srinv or cusolver_srinv"):
        MultiFramePoseSolveConfig(velocity_solver=backend, refine_partial_scale=True)


@pytest.mark.parametrize(
    "options", [dict(refine_from_origin=True), dict(refinement_ray_tolerance=1e-5)]
)
def test_advanced_options_require_refinement(options: dict) -> None:
    with pytest.raises(ValueError, match="requires partial-scale refinement"):
        MultiFramePoseSolveConfig(velocity_solver="warp_srinv", **options)


@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_refinement_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        MultiFramePoseSolveConfig(
            velocity_solver="warp_srinv",
            refine_partial_scale=True,
            refinement_ray_tolerance=tolerance,
        )


def test_fixed_factory_forwards_refinement_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def init(self: object, *args: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(GpuWbcMultiFrameSolver, "__init__", init)
    GpuWbcMultiFrameSolver.from_robot(
        Path("robot.urdf"),
        Path("cache"),
        robot=SimpleNamespace(is_floating_base=False),
        active_joint_names=("joint",),
        refine_partial_scale=True,
        refine_from_origin=True,
        refinement_ray_tolerance=1e-5,
    )
    assert captured["refine_partial_scale"] is True
    assert captured["refine_from_origin"] is True
    assert captured["refinement_ray_tolerance"] == 1e-5


@pytest.mark.parametrize("backend", ["warp_srinv", "cusolver_srinv"])
@pytest.mark.parametrize("enabled,tolerance", [(False, 1e-8), (True, 1e-8), (True, 1e-5)])
def test_runtime_constructor_wiring(backend: str, enabled: bool, tolerance: float) -> None:
    config = MultiFramePoseSolveConfig(
        velocity_solver=backend, refine_partial_scale=enabled, refinement_ray_tolerance=tolerance
    )
    source = ast.parse(inspect.getsource(inspect.getmodule(DeviceResidentMultiFramePoseSolver)))
    call = next(
        node
        for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "WarpBoundRelease"
    )
    names = {
        "refine_partial_scale",
        "refine_from_origin",
        "refinement_ray_tolerance",
        "repair_float32_roundoff",
    }
    expression = ast.Expression(
        ast.Dict(
            keys=[ast.Constant(k.arg) for k in call.keywords if k.arg in names],
            values=[k.value for k in call.keywords if k.arg in names],
        )
    )
    options = eval(
        compile(ast.fix_missing_locations(expression), "<runtime-wiring>", "eval"),
        {"self": SimpleNamespace(config=config)},
    )
    assert options == dict(
        refine_partial_scale=enabled,
        refine_from_origin=False,
        refinement_ray_tolerance=tolerance,
        repair_float32_roundoff=enabled and tolerance == 1e-8,
    )
    public_source = ast.parse(inspect.getsource(GpuWbcMultiFrameSolver))
    config_call = next(
        node
        for node in ast.walk(public_source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "MultiFramePoseSolveConfig"
    )
    for name in names - {"repair_float32_roundoff"}:
        keyword = next(k for k in config_call.keywords if k.arg == name)
        assert isinstance(keyword.value, ast.Name) and keyword.value.id == name


def runtime():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("warp")
    from embodik.gpu.wbc._runtime.warp_bound_release import WarpBoundRelease

    return torch, WarpBoundRelease


def rounded_problem(torch, dtype: str = "float32"):
    convert = lambda value: torch.tensor(value, dtype=getattr(torch, dtype), device="cuda")
    return [
        convert(value)
        for value in (
            [[[2.0, 1.0, -3.0], [-3.0, 1.0, 2.0]]],
            [[-2.0, -2.0]],
            [[-1.0, -1.0, 0.0]],
            [[1.0, 1.0, 1.0]],
            [[0.5, -1.0, 0.5000001]],
        )
    ]


@pytest.mark.parametrize("origin", [False, True])
def test_cuda_roundoff_restart_reaches_exact_ray(origin: bool) -> None:
    torch, cls = runtime()
    j, d, lo, hi, seed = rounded_problem(torch)
    baseline = cls(
        2, 3, batch_capacity=1, device="cuda", refine_partial_scale=True, refine_from_origin=origin
    )
    torch.testing.assert_close(baseline.solve(j, d, lo, hi, seed), seed, atol=0, rtol=0)
    solver = cls(
        2,
        3,
        batch_capacity=1,
        device="cuda",
        refine_partial_scale=True,
        refine_from_origin=origin,
        repair_float32_roundoff=True,
    )
    output = solver.solve(j, d, lo, hi, seed)
    torch.testing.assert_close(
        output, torch.tensor([[1.0, -1.0, 1.0]], device="cuda"), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(j @ output[..., None], d[..., None], atol=1e-6, rtol=0)
    assert bool(((output >= lo) & (output <= hi)).all())
    assert solver.recovered.cpu().tolist() == [True]


def test_cuda_origin_excluded_retains_rounded_seed() -> None:
    torch, cls = runtime()
    convert = lambda value: torch.tensor(value, dtype=torch.float32, device="cuda")
    j = convert([[[1.0, 0.0], [0.0, 1.0]]])
    d = convert([[1.0, 1.0]])
    lo = convert([[0.5, 0.0]])
    hi = convert([[1.0, 1.0]])
    seed = convert([[0.6 + 2**-22, 0.6 - 2**-22]])
    solver = cls(
        2,
        2,
        batch_capacity=1,
        device="cuda",
        refine_partial_scale=True,
        repair_float32_roundoff=True,
    )
    torch.testing.assert_close(solver.solve(j, d, lo, hi, seed), seed, atol=0, rtol=0)
    assert solver.recovered.cpu().tolist() == [False]


@pytest.mark.parametrize("dtype,refine", [("float32", False), ("float64", True)])
def test_cuda_other_paths_preserve_seed(dtype: str, refine: bool) -> None:
    torch, cls = runtime()
    problem = rounded_problem(torch, dtype)
    solver = cls(
        2,
        3,
        batch_capacity=1,
        device="cuda",
        refine_partial_scale=refine,
        repair_float32_roundoff=True,
    )
    torch.testing.assert_close(solver.solve(*problem), problem[-1], atol=0, rtol=0)
    assert solver.recovered.cpu().tolist() == [False]


def test_cuda_no_scale_improvement_retains_seed() -> None:
    torch, cls = runtime()
    convert = lambda v: torch.tensor(v, dtype=torch.float32, device="cuda")
    problem = [
        convert(v)
        for v in (
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[1.0, 0.0]],
            [[-1.0, -1.0]],
            [[0.5, 1.0]],
            [[0.5, 2**-22]],
        )
    ]
    solver = cls(
        2,
        2,
        batch_capacity=1,
        device="cuda",
        refine_partial_scale=True,
        repair_float32_roundoff=True,
    )
    torch.testing.assert_close(solver.solve(*problem), problem[-1], atol=0, rtol=0)
    assert solver.recovered.cpu().tolist() == [False]


def test_cuda_roundoff_graph_mixed_worlds_replay() -> None:
    torch, cls = runtime()
    problem = [v.repeat(4, *([1] * (v.ndim - 1))) for v in rounded_problem(torch)]
    j, d, lo, hi, seed = problem
    lo[1, 0] = 0.25  # Rounded moving seed with a mandatory inward coordinate.
    seed[2] = torch.tensor([1.0, -1.0, 1.0], device="cuda")
    d[3, 0] = float("nan")
    solver = cls(
        2,
        3,
        batch_capacity=4,
        device="cuda",
        refine_partial_scale=True,
        repair_float32_roundoff=True,
    )
    for _ in range(3):
        solver.solve(*problem)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = solver.solve(*problem)
    for perturbation in (2**-23, 2**-22, 2**-23):
        seed[0, 2] = 0.5 + perturbation
        graph.replay()
        torch.testing.assert_close(
            output[0], torch.tensor([1.0, -1.0, 1.0], device="cuda"), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(output[1:], seed[1:], atol=0, rtol=0)
        assert solver.recovered.cpu().tolist() == [True, False, False, False]


def test_cuda_near_full_roundoff_retains_seed() -> None:
    torch, cls = runtime()
    convert = lambda value: torch.tensor(value, dtype=torch.float32, device="cuda")
    problem = [
        convert(value)
        for value in (
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[1.0, 0.0]],
            [[-1.0, -1.0]],
            [[1.0, 1.0]],
            [[1.0 - 2**-22, 2**-22]],
        )
    ]
    solver = cls(
        2,
        2,
        batch_capacity=1,
        device="cuda",
        refine_partial_scale=True,
        repair_float32_roundoff=True,
    )
    torch.testing.assert_close(solver.solve(*problem), problem[-1], atol=0, rtol=0)
    assert solver.recovered.cpu().tolist() == [False]
