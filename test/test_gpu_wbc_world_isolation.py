"""CUDA isolation tests for per-world GPU WBC reset, validity, and status.

These stay out of the default CPU suite. Run with a CUDA Pixi environment:

``pixi run -e cuda python -m pytest test/test_gpu_wbc_world_isolation.py -q``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import embodik as eik
from embodik.gpu.wbc import GpuWbcMultiFrameSolver


def _cuda():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch


def _write_chain(path: Path, dof: int = 2) -> Path:
    links = ['<link name="base"/>']
    joints = []
    for index in range(dof):
        links.append(f'<link name="link_{index}"/>')
        parent = "base" if index == 0 else f"link_{index - 1}"
        joints.append(f"""<joint name="joint_{index}" type="revolute">
  <parent link="{parent}"/><child link="link_{index}"/>
  <origin xyz="0.1 0 0"/><axis xyz="0 0 1"/>
  <limit lower="-1.2" upper="1.2" velocity="2" effort="20"/>
</joint>""")
    path.write_text(
        '<?xml version="1.0"?>\n<robot name="gpu_wbc_isolation_chain">\n'
        + "\n".join(links + joints)
        + "\n</robot>\n",
        encoding="utf-8",
    )
    return path


def _tool_frame(robot: eik.RobotModel) -> str:
    names = [str(name) for name in robot.get_frame_names()]
    for candidate in reversed(names):
        if candidate not in {"universe", "root"}:
            return candidate
    raise RuntimeError("chain model has no usable task frame")


def _identity_targets(batch_size: int, frame_count: int, torch):
    target = torch.zeros((batch_size, frame_count, 7), dtype=torch.float32, device="cuda")
    target[..., 3] = 1.0
    return target


def _solver(tmp_path: Path, batch_size: int) -> GpuWbcMultiFrameSolver:
    urdf = _write_chain(tmp_path / "chain.urdf")
    robot = eik.RobotModel(str(urdf), floating_base=False)
    frame = _tool_frame(robot)
    return GpuWbcMultiFrameSolver.from_robot(
        urdf,
        tmp_path / "cache",
        robot=robot,
        robot_name="isolation-chain",
        frames=(frame,),
        default_configuration=robot.neutral_configuration(),
        solver_backend="warp_srinv",
        batch_size=batch_size,
        iterations=1,
        dt=0.01,
    )


def test_invalid_world_does_not_reject_the_device_batch(tmp_path: Path) -> None:
    torch = _cuda()
    solver = _solver(tmp_path, batch_size=2)
    q = torch.zeros((2, solver.configuration_dim), dtype=torch.float32, device="cuda")
    target = _identity_targets(2, len(solver.frames), torch)
    q[1, 0] = float("nan")

    result = solver.solve_device_batch(q, target)

    assert result.world_status is not None
    assert int(result.world_status[0]) in {
        GpuWbcMultiFrameSolver.WORLD_STATUS_SUCCESS,
        GpuWbcMultiFrameSolver.WORLD_STATUS_HELD,
    }
    assert int(result.world_status[1]) == GpuWbcMultiFrameSolver.WORLD_STATUS_INVALID_INPUT
    assert bool(result.world_success[0]) == (
        int(result.world_status[0]) == GpuWbcMultiFrameSolver.WORLD_STATUS_SUCCESS
    )
    assert not bool(result.world_success[1])
    assert torch.isfinite(result.q_solution[0]).all()
    assert torch.equal(result.accepted_velocity[1], torch.zeros_like(result.accepted_velocity[1]))


def test_infinite_joint_position_stays_invalid_and_unmodified(tmp_path: Path) -> None:
    torch = _cuda()
    solver = _solver(tmp_path, batch_size=2)
    q = torch.zeros((2, solver.configuration_dim), dtype=torch.float32, device="cuda")
    target = _identity_targets(2, len(solver.frames), torch)
    q[1, 0] = float("inf")
    q[1, 1] = float("-inf")

    result = solver.solve_device_batch(q, target)

    assert int(result.world_status[1]) == GpuWbcMultiFrameSolver.WORLD_STATUS_INVALID_INPUT
    assert torch.equal(result.q_solution[1], q[1])


def test_inactive_out_of_limit_joint_is_not_projected(tmp_path: Path) -> None:
    torch = _cuda()
    solver = _solver(tmp_path, batch_size=2)
    q = torch.zeros((2, solver.configuration_dim), dtype=torch.float32, device="cuda")
    target = _identity_targets(2, len(solver.frames), torch)
    q[1, 0] = 5.0
    valid = torch.tensor([True, False], device="cuda")

    result = solver.solve_device_batch(q, target, valid_mask=valid)

    assert int(result.world_status[1]) == GpuWbcMultiFrameSolver.WORLD_STATUS_INACTIVE
    assert torch.equal(result.q_solution[1], q[1])


def test_valid_mask_holds_inactive_world_history(tmp_path: Path) -> None:
    torch = _cuda()
    solver = _solver(tmp_path, batch_size=2)
    q = torch.zeros((2, solver.configuration_dim), dtype=torch.float32, device="cuda")
    target = _identity_targets(2, len(solver.frames), torch)
    solver._previous_velocity.fill_(0.4)
    valid = torch.tensor([True, False], device="cuda")

    result = solver.solve_device_batch(q, target, valid_mask=valid)

    assert int(result.world_status[1]) == GpuWbcMultiFrameSolver.WORLD_STATUS_INACTIVE
    torch.testing.assert_close(
        solver._previous_velocity[1], torch.full((solver.velocity_dim,), 0.4, device="cuda")
    )
    assert torch.equal(result.q_solution[1], q[1])


def test_reset_mask_zeros_selected_history_before_the_solve(tmp_path: Path) -> None:
    torch = _cuda()
    solver = _solver(tmp_path, batch_size=2)
    q = torch.zeros((2, solver.configuration_dim), dtype=torch.float32, device="cuda")
    target = _identity_targets(2, len(solver.frames), torch)
    solver._previous_velocity.fill_(0.7)
    reset = torch.tensor([True, False], device="cuda")
    captured = {}

    original = solver._solver.solve

    def _wrapped(q_start, target_value, history, current=None, **options):
        captured["history"] = history.detach().clone()
        captured["reset"] = options.get("reset_mask")
        return original(q_start, target_value, history, current, **options)

    solver._solver.solve = _wrapped
    solver.solve_device_batch(q, target, reset_mask=reset)

    assert captured["reset"].tolist() == [True, False]
    torch.testing.assert_close(
        captured["history"][0], torch.zeros(solver.velocity_dim, device="cuda")
    )
    torch.testing.assert_close(
        captured["history"][1], torch.full((solver.velocity_dim,), 0.7, device="cuda")
    )


def test_measure_device_batch_returns_solver_path_resource_fields(tmp_path: Path) -> None:
    torch = _cuda()
    solver = _solver(tmp_path, batch_size=2)
    q = torch.zeros((2, solver.configuration_dim), dtype=torch.float32, device="cuda")
    target = _identity_targets(2, len(solver.frames), torch)

    report = solver.measure_device_batch(q, target)

    assert report.result.world_status is not None
    assert report.allocated_bytes >= 0
    assert report.reserved_bytes >= report.allocated_bytes
    assert report.host_dispatch_ms >= 0.0
    assert report.synchronization_ms >= 0.0
