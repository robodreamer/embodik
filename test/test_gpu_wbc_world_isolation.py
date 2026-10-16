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


def _solver(tmp_path: Path, batch_size: int, **tracking_options) -> GpuWbcMultiFrameSolver:
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
        **tracking_options,
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


def test_public_constructor_applies_tracking_tolerances(tmp_path: Path) -> None:
    _cuda()
    solver = _solver(
        tmp_path, batch_size=2, position_tolerance_m=0.005, orientation_tolerance_rad=0.01
    )
    assert solver._solver.config.position_tolerance_m == 0.005
    assert solver._solver.config.orientation_tolerance_rad == 0.01


def test_microscopic_pose_noise_does_not_move_an_idle_world(tmp_path: Path) -> None:
    torch = _cuda()
    solver = _solver(tmp_path, batch_size=2)
    q = torch.zeros((2, solver.configuration_dim), dtype=torch.float32, device="cuda")
    target = (
        solver._solver._primary_pose(solver._solver.kinematics._evaluate_pose_trusted(q))[
            ..., [0, 1, 2, 6, 3, 4, 5]
        ]
        .clone()
        .contiguous()
    )
    # World zero has only floating-point-scale angular noise. The other
    # world has a useful rotation request and must continue progressing.
    target[0, 0, 6] = 1e-7
    target[1, 0, 3] = float(np.cos(0.01))
    target[1, 0, 6] = float(np.sin(0.01))
    result = solver.solve_device_batch(q, target)
    torch.testing.assert_close(result.q_solution[0], q[0], atol=0, rtol=0)
    assert torch.linalg.vector_norm(result.q_solution[1] - q[1]).item() > 1e-7


def test_noise_suppression_respects_a_stricter_tracking_tolerance(tmp_path: Path) -> None:
    torch = _cuda()
    solver = _solver(
        tmp_path, batch_size=2, position_tolerance_m=1e-9, orientation_tolerance_rad=1e-9
    )
    q = torch.zeros((2, solver.configuration_dim), dtype=torch.float32, device="cuda")
    pose = solver._solver._primary_pose(solver._solver.kinematics._evaluate_pose_trusted(q))
    target = pose[..., [0, 1, 2, 6, 3, 4, 5]].clone().contiguous()
    target[..., 6] = 1e-7
    _, _, twist = solver._solver._task_state_eager(pose, target)
    assert torch.linalg.vector_norm(twist, dim=-1).min().item() > 0


def test_graph_capture_defers_automatic_gc(tmp_path: Path, monkeypatch) -> None:
    import gc

    torch = _cuda()
    solver = _solver(tmp_path, batch_size=2)
    core = solver._solver
    original = core._solve_impl
    policy = gc.isenabled()
    captured_policy = []

    def observed(*args, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            captured_policy.append(gc.isenabled())
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "_solve_impl", observed)
    q = torch.zeros((2, solver.configuration_dim), dtype=torch.float32, device="cuda")
    solver.solve_device_batch(q, _identity_targets(2, len(solver.frames), torch))
    assert captured_policy and not any(captured_policy)
    assert gc.isenabled() == policy


def test_accepted_progress_is_distinct_from_terminal_convergence(tmp_path: Path) -> None:
    torch = _cuda()
    solver = _solver(tmp_path, batch_size=2)
    q = torch.zeros((2, solver.configuration_dim), dtype=torch.float32, device="cuda")
    target = (
        solver._solver._primary_pose(solver._solver.kinematics._evaluate_pose_trusted(q))[
            ..., [0, 1, 2, 6, 3, 4, 5]
        ]
        .clone()
        .contiguous()
    )
    target[0, 0, 3] = float(np.cos(0.1))
    target[0, 0, 6] = float(np.sin(0.1))
    result = solver.solve_device_batch(q, target)
    assert result.world_status.cpu().tolist() == [solver.WORLD_STATUS_SUCCESS] * 2
    assert result.converged.cpu().tolist() == [False, True]


def test_graph_results_borrow_buffers_while_cloned_snapshots_persist(tmp_path: Path) -> None:
    torch = _cuda()
    solver = _solver(tmp_path, batch_size=2)
    q = torch.zeros((2, solver.configuration_dim), dtype=torch.float32, device="cuda")
    target = (
        solver._solver._primary_pose(solver._solver.kinematics._evaluate_pose_trusted(q))[
            ..., [0, 1, 2, 6, 3, 4, 5]
        ]
        .clone()
        .contiguous()
    )
    target[..., 3] = float(np.cos(0.1))
    target[..., 6] = float(np.sin(0.1))
    first = solver.solve_device_batch(q, target)
    snapshot = first.q_solution.clone()
    expected_snapshot = snapshot.clone()
    target[..., 6] = -float(np.sin(0.1))
    second = solver.solve_device_batch(first.q_solution, target)
    assert first.q_solution.data_ptr() == second.q_solution.data_ptr()
    assert not torch.equal(first.q_solution, snapshot)
    torch.testing.assert_close(snapshot, expected_snapshot, atol=0, rtol=0)


def test_inactive_world_has_no_primary_correction_request(tmp_path: Path, monkeypatch) -> None:
    torch = _cuda()
    solver = _solver(tmp_path, batch_size=2)
    core = solver._solver
    original = core._native_velocity
    requests = []
    matrices = []

    def observed(twist, matrix, *args):
        requests.append(twist.clone())
        matrices.append(matrix.clone())
        return original(twist, matrix, *args)

    monkeypatch.setattr(core, "_native_velocity", observed)
    q = torch.zeros((2, solver.configuration_dim), dtype=torch.float32, device="cuda")
    target = (
        core._primary_pose(core.kinematics._evaluate_pose_trusted(q))[..., [0, 1, 2, 6, 3, 4, 5]]
        .clone()
        .contiguous()
    )
    target[..., 3] = float(np.cos(0.1))
    target[..., 6] = float(np.sin(0.1))
    result = solver.solve_device_batch(
        q, target, valid_mask=torch.tensor([True, False], device="cuda")
    )
    assert requests
    torch.testing.assert_close(requests[-1][1], torch.zeros_like(requests[-1][1]), atol=0, rtol=0)
    assert torch.linalg.vector_norm(requests[-1][0]).item() > 0
    torch.testing.assert_close(result.q_solution[1], q[1], atol=0, rtol=0)
    torch.testing.assert_close(matrices[-1][1], torch.zeros_like(matrices[-1][1]), atol=0, rtol=0)
    assert matrices[-1][0].abs().max().item() > 0


@pytest.mark.parametrize("feature", ["posture", "acceleration"])
def test_zero_request_keeps_physical_factor_when_other_work_requires_it(
    tmp_path: Path, monkeypatch, feature: str
) -> None:
    torch = _cuda()
    options = (
        {
            "posture_target_configuration": (0.1, -0.1),
            "posture_velocity_indices": (0, 1),
            "posture_weights": (1.0, 1.0),
        }
        if feature == "posture"
        else {"max_joint_acceleration_rad_s2": 0.1}
    )
    solver = _solver(tmp_path, batch_size=2, **options)
    core = solver._solver
    original_state = core._task_state
    original_velocity = core._native_velocity
    matrices = []

    def zero_request(*args):
        position, orientation, twist = original_state(*args)
        return position, orientation, torch.zeros_like(twist)

    def observed(twist, matrix, *args):
        matrices.append(matrix.clone())
        return original_velocity(twist, matrix, *args)

    monkeypatch.setattr(core, "_task_state", zero_request)
    monkeypatch.setattr(core, "_native_velocity", observed)
    q = torch.zeros((2, solver.configuration_dim), dtype=torch.float32, device="cuda")
    previous = torch.full_like(q, 0.5) if feature == "acceleration" else None
    solver.solve_device_batch(
        q, _identity_targets(2, len(solver.frames), torch), previous_velocity=previous
    )
    assert matrices[0].abs().max().item() > 0
