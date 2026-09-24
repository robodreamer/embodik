"""Smoke the fixed-base GPU WBC calls shown in the public guides.

This is a CUDA test, not a public example. Run it after ``pixi run setup-gpu-wbc``:

``pixi run -e cuda python -m pytest test/test_documented_gpu_wbc.py -q``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import embodik
from embodik.gpu.wbc import GpuWbcMultiFrameSolver


def _cuda_stack():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    pytest.importorskip("robot_descriptions")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch


def test_documented_fixed_base_batch_tracks_a_nearby_pose(tmp_path: Path) -> None:
    torch = _cuda_stack()
    from robot_descriptions.panda_description import URDF_PATH

    urdf = Path(URDF_PATH)
    robot = embodik.RobotModel(str(urdf), floating_base=False)
    q0 = np.asarray(robot.neutral_configuration(), dtype=np.float64)
    frame = "panda_link8"
    lower, upper = (np.asarray(bound, dtype=np.float64) for bound in robot.get_joint_limits())
    assert np.any(q0 < lower) or np.any(q0 > upper)

    batch_size = 2
    solver = GpuWbcMultiFrameSolver.from_robot(
        urdf,
        tmp_path / "gpu-wbc-cache",
        robot=robot,
        robot_name="panda",
        frames=(frame,),
        frame_task_dimensions=(6,),
        default_configuration=q0,
        batch_size=batch_size,
        dt=0.01,
    )

    q_active = q0[list(solver.active_configuration_indices)].astype(np.float32)
    q_cuda = (
        torch.as_tensor(q_active, dtype=torch.float32, device="cuda")
        .expand(batch_size, -1)
        .contiguous()
    )
    identity = torch.zeros((batch_size, 1, 7), dtype=torch.float32, device="cuda")
    identity[..., 3] = 1.0
    repaired = solver.solve_device_batch(q_cuda, identity)
    repaired_q = repaired.q_solution[0].detach().cpu().numpy()
    assert np.all(repaired_q >= lower - 1e-5)
    assert np.all(repaired_q <= upper + 1e-5)

    q_full = q0.copy()
    q_full[list(solver.active_configuration_indices)] = repaired_q
    robot.update_configuration(q_full)
    pose = robot.get_frame_pose(frame)
    position = np.asarray(pose.translation, dtype=np.float32).copy()
    position[0] += 0.04
    quaternion_wxyz = np.asarray(
        embodik.r2q(np.asarray(pose.rotation), order="sxyz"), dtype=np.float32
    )
    target = (
        torch.as_tensor(
            np.concatenate((position, quaternion_wxyz)), dtype=torch.float32, device="cuda"
        )
        .reshape(1, 1, 7)
        .expand(batch_size, 1, 7)
        .contiguous()
    )
    q_cuda = repaired.q_solution
    errors = []
    for _ in range(4):
        result = solver.solve_device_batch(q_cuda, target)
        assert result.status == "solved_or_held_needs_verification"
        assert int((result.world_status == solver.WORLD_STATUS_SUCCESS).sum()) == batch_size
        q_full = q0.copy()
        q_full[list(solver.active_configuration_indices)] = (
            result.q_solution[0].detach().cpu().numpy()
        )
        robot.update_configuration(q_full)
        current = np.asarray(robot.get_frame_pose(frame).translation, dtype=np.float64)
        errors.append(float(np.linalg.norm(current - position)))
        q_cuda = result.q_solution

    assert np.isfinite(errors).all()
    assert errors[-1] < errors[0]
