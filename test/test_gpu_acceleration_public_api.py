"""Opt-in CUDA parity tests for the public acceleration-level GPU slice."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import embodik as eik

pytestmark = pytest.mark.skipif(
    os.environ.get("EMBODIK_RUN_GPU_ACCELERATION_TESTS") != "1",
    reason="set EMBODIK_RUN_GPU_ACCELERATION_TESTS=1 for CUDA tests",
)


def _torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch


def _write_chain(path: Path, dof: int, *, floating: bool = False) -> eik.RobotModel:
    links = ['<link name="base"/>']
    joints = []
    for index in range(dof):
        links.append(f'<link name="link_{index}"/>')
        parent = "base" if index == 0 else f"link_{index - 1}"
        joint_type = "prismatic" if index % 3 == 1 else "revolute"
        joints.append(
            f"""<joint name="joint_{index}" type="{joint_type}">
  <parent link="{parent}"/><child link="link_{index}"/>
  <origin xyz="0.1 0 0"/><axis xyz="0 0 1"/>
  <limit lower="-1" upper="1" velocity="2" effort="20"/>
</joint>"""
        )
    path.write_text(
        '<?xml version="1.0"?>\n<robot name="gpu_acceleration_chain">\n'
        + "\n".join(links + joints)
        + "\n</robot>\n",
        encoding="utf-8",
    )
    return eik.RobotModel(str(path), floating_base=floating)


def _cpu_solve(
    robot: eik.RobotModel,
    q: np.ndarray,
    dq: np.ndarray,
    dt: float,
    reference: np.ndarray,
    acceleration_limits: np.ndarray,
    *,
    apply_position_limits: bool,
    apply_velocity_limits: bool,
    zero_acceleration: tuple[int, ...] = (),
    zero_next_velocity: tuple[int, ...] = (),
    fixed_position: tuple[int, ...] = (),
):
    solver = eik.AccelerationSolver(robot)
    task = solver.add_posture_task("posture")
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    task_reference = eik.AccelerationTaskReference()
    task_reference.desired_acceleration = reference
    # The public GPU argument is the assembled generalized acceleration
    # reference.  This is exactly the CPU feed-forward-only task convention.
    task_reference.proportional_gain = 0.0
    task_reference.derivative_gain = 0.0
    solver.set_task_reference("posture", task_reference)
    options = eik.AccelerationSolveOptions()
    options.acceleration_limits_override = acceleration_limits
    options.apply_position_limits = apply_position_limits
    options.apply_velocity_limits = apply_velocity_limits
    options.zero_acceleration_joint_indices = list(zero_acceleration)
    options.zero_next_velocity_joint_indices = list(zero_next_velocity)
    options.fixed_current_position_joint_indices = list(fixed_position)
    return solver.solve(q, dq, dt, options)


def _mask(torch, dof: int, indices: tuple[int, ...]):
    result = torch.zeros(dof, dtype=torch.bool, device="cuda")
    if indices:
        result[list(indices)] = True
    return result


@pytest.mark.parametrize("dof", (2, 5, 9))
def test_arbitrary_scalar_models_match_cpu_joint_posture_min_error(
    tmp_path: Path, dof: int
) -> None:
    _torch()
    from embodik.gpu.wbc import GpuAccelerationSolver

    robot = _write_chain(tmp_path / f"chain_{dof}.urdf", dof)
    q = np.linspace(-0.35, 0.42, dof)
    dq = np.linspace(0.17, -0.21, dof)
    reference = np.linspace(-7.0, 6.0, dof)
    limits = np.linspace(2.0, 5.0, dof)
    dt = 0.04
    cpu = _cpu_solve(
        robot,
        q,
        dq,
        dt,
        reference,
        limits,
        apply_position_limits=True,
        apply_velocity_limits=True,
    )
    gpu = GpuAccelerationSolver(robot, limits).solve(q, dq, dt, reference)

    assert cpu.status == eik.SolverStatus.SUCCESS
    assert gpu.success
    np.testing.assert_allclose(gpu.joint_accelerations, cpu.joint_accelerations, atol=1e-9)
    np.testing.assert_allclose(gpu.joint_velocities_next, cpu.joint_velocities_next, atol=1e-9)
    np.testing.assert_allclose(gpu.q_solution, cpu.q_solution, atol=1e-9)
    assert gpu.acceleration_limits_applied
    assert gpu.velocity_limits_applied
    assert gpu.position_limits_applied


@pytest.mark.parametrize(
    ("zero_acceleration", "zero_next_velocity", "fixed_position"),
    (((0,), (), ()), ((), (1,), ()), ((), (), (0,))),
)
def test_lock_policies_match_cpu(
    tmp_path: Path,
    zero_acceleration: tuple[int, ...],
    zero_next_velocity: tuple[int, ...],
    fixed_position: tuple[int, ...],
) -> None:
    torch = _torch()
    from embodik.gpu.wbc import GpuAccelerationSolver

    robot = _write_chain(tmp_path / "locks.urdf", 2)
    q = np.array([0.1, -0.2])
    dq = np.array([0.12, -0.08])
    reference = np.array([0.7, -0.4])
    limits = np.array([20.0, 20.0])
    dt = 0.1
    cpu = _cpu_solve(
        robot,
        q,
        dq,
        dt,
        reference,
        limits,
        apply_position_limits=False,
        apply_velocity_limits=False,
        zero_acceleration=zero_acceleration,
        zero_next_velocity=zero_next_velocity,
        fixed_position=fixed_position,
    )
    gpu = GpuAccelerationSolver(
        robot,
        limits,
        apply_position_limits=False,
        apply_velocity_limits=False,
    ).solve_device_batch(
        torch.as_tensor(q, dtype=torch.float64, device="cuda").unsqueeze(0),
        torch.as_tensor(dq, dtype=torch.float64, device="cuda").unsqueeze(0),
        dt,
        torch.as_tensor(reference, dtype=torch.float64, device="cuda"),
        _mask(torch, 2, zero_acceleration),
        _mask(torch, 2, zero_next_velocity),
        _mask(torch, 2, fixed_position),
    )

    assert cpu.status == eik.SolverStatus.SUCCESS
    assert bool(gpu.success[0])
    np.testing.assert_allclose(
        gpu.joint_accelerations[0].cpu().numpy(), cpu.joint_accelerations, atol=1e-9
    )


def test_infeasible_world_fails_closed_and_other_world_remains_executable(
    tmp_path: Path,
) -> None:
    torch = _torch()
    from embodik.gpu.wbc import GpuAccelerationSolver

    robot = _write_chain(tmp_path / "failure.urdf", 2)
    solver = GpuAccelerationSolver(robot, np.full(2, 4.0))
    q = torch.tensor([[0.0, 0.0], [1.1, 0.0]], dtype=torch.float64, device="cuda")
    dq = torch.zeros_like(q)
    result = solver.solve_device_batch(q, dq, 0.05)

    assert result.success.tolist() == [True, False]
    assert torch.isfinite(result.joint_accelerations[0]).all()
    assert torch.isnan(result.joint_accelerations[1]).all()

    host = solver.solve(np.array([1.1, 0.0]), np.zeros(2), 0.05)
    assert not host.success
    assert host.joint_accelerations.size == 0
    assert host.joint_velocities_next.size == 0
    assert host.q_solution.size == 0


def test_mandatory_state_box_braking_fallback_matches_cpu(tmp_path: Path) -> None:
    _torch()
    from embodik.gpu.wbc import GpuAccelerationSolver

    robot = _write_chain(tmp_path / "braking.urdf", 2)
    q = np.array([0.9, 0.0])
    dq = np.array([0.8, 0.0])
    reference = np.array([1.0, 0.0])
    limits = np.array([4.0, 4.0])
    cpu = _cpu_solve(
        robot,
        q,
        dq,
        0.05,
        reference,
        limits,
        apply_position_limits=True,
        apply_velocity_limits=True,
    )
    gpu = GpuAccelerationSolver(robot, limits).solve(q, dq, 0.05, reference)

    assert cpu.status == eik.SolverStatus.SUCCESS
    assert gpu.success
    assert cpu.state_box_task_fallback_applied
    assert gpu.state_box_task_fallback_applied
    np.testing.assert_allclose(gpu.joint_accelerations, cpu.joint_accelerations, atol=1e-9)


def test_cuda_graph_replays_changed_1024_world_batch(tmp_path: Path) -> None:
    torch = _torch()
    from embodik.gpu.wbc import GpuAccelerationSolver

    dof = 7
    batch_size = 1024
    robot = _write_chain(tmp_path / "graph.urdf", dof)
    solver = GpuAccelerationSolver(robot, np.full(dof, 8.0))
    q = torch.zeros((batch_size, dof), dtype=torch.float64, device="cuda")
    dq = torch.zeros_like(q)
    reference = torch.full_like(q, 0.5)
    solver.solve_device_batch(q, dq, 0.01, reference)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = solver.solve_device_batch(q, dq, 0.01, reference)
    pointers = (
        captured.joint_accelerations.data_ptr(),
        captured.q_solution.data_ptr(),
    )
    reference.fill_(-0.75)
    graph.replay()
    torch.cuda.synchronize()

    assert bool(captured.success.all())
    torch.testing.assert_close(captured.joint_accelerations, reference)
    graph.replay()
    torch.cuda.synchronize()
    assert pointers == (
        captured.joint_accelerations.data_ptr(),
        captured.q_solution.data_ptr(),
    )


def test_floating_base_rejected_and_capabilities_are_explicit(tmp_path: Path) -> None:
    _torch()
    from embodik.gpu.wbc import GPU_ACCELERATION_CAPABILITIES, GpuAccelerationSolver

    capabilities = GPU_ACCELERATION_CAPABILITIES
    assert capabilities.supports_fixed_base_scalar_joints
    assert capabilities.supports_diagonal_generalized_acceleration_min_error
    assert capabilities.supports_cuda_graph_capture
    assert not capabilities.supports_frame_tasks
    assert not capabilities.supports_centroidal_tasks_and_constraints
    assert not capabilities.supports_collision_constraints
    assert not capabilities.supports_floating_base
    floating = _write_chain(tmp_path / "floating.urdf", 2, floating=True)
    with pytest.raises(NotImplementedError, match="fixed-base"):
        GpuAccelerationSolver(floating, np.ones(floating.nv))
