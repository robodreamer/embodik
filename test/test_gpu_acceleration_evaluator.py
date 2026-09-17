"""Opt-in CUDA tests for the Newton frame acceleration-bias evaluator.

Run with ``EMBODIK_RUN_GPU_ACCELERATION_TESTS=1 pytest -q
test/test_gpu_acceleration_evaluator.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import embodik as eik
from embodik.gpu.wbc.model_spec import robot_solve_spec_from_embodik

pytestmark = pytest.mark.skipif(
    os.environ.get("EMBODIK_RUN_GPU_ACCELERATION_TESTS") != "1",
    reason="set EMBODIK_RUN_GPU_ACCELERATION_TESTS=1 for Newton/CUDA tests",
)


def _cuda_modules():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch


def _evaluator_class():
    from embodik.gpu.wbc._runtime.newton_acceleration import (
        NewtonFrameAccelerationEvaluator,
    )

    return NewtonFrameAccelerationEvaluator


def _write_serial_chain(path: Path, dof: int) -> Path:
    links = ['<link name="base"/>']
    joints = []
    axes = ((0, 0, 1), (0, 1, 0), (1, 0, 0))
    for index in range(dof):
        links.append(f'<link name="link_{index}"/>')
        axis = axes[index % len(axes)]
        joint_type = "prismatic" if index % 3 == 1 else "revolute"
        parent = "base" if index == 0 else f"link_{index - 1}"
        joints.append(f"""<joint name="joint_{index}" type="{joint_type}">
  <parent link="{parent}"/><child link="link_{index}"/>
  <origin xyz="0.11 {0.025 * ((index % 2) * 2 - 1):.6f} 0.07"
    rpy="0.03 -0.02 0.015"/>
  <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>
  <limit lower="-1.2" upper="1.3" velocity="3" effort="20"/>
</joint>""")
    path.write_text(
        '<?xml version="1.0"?>\n<robot name="acceleration_chain">\n'
        + "\n".join(links + joints)
        + "\n</robot>\n",
        encoding="utf-8",
    )
    return path


def _spec(robot: eik.RobotModel, name: str, frames: str | tuple[str, ...]):
    task_frames = (frames,) if isinstance(frames, str) else frames
    return robot_solve_spec_from_embodik(
        robot,
        name=name,
        model_hash=f"acceleration-{name}-{robot.nq}-{robot.nv}",
        task_frames=task_frames,
        active_velocity_indices=tuple(range(robot.nv)),
    )


def _cpu_reference(
    urdf: Path,
    frame: str,
    configurations: np.ndarray,
    velocities: np.ndarray,
):
    pin = pytest.importorskip("pinocchio")
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    frame_id = model.getFrameId(frame)
    poses, jacobians, biases = [], [], []
    for q, dq in zip(configurations, velocities, strict=True):
        pin.forwardKinematics(model, data, q, dq)
        pin.computeJointJacobiansTimeVariation(model, data, q, dq)
        pin.updateFramePlacements(model, data)
        frame_pose = data.oMf[frame_id]
        quaternion = np.asarray(pin.Quaternion(frame_pose.rotation).coeffs())
        poses.append(np.r_[np.asarray(frame_pose.translation), quaternion])
        jacobian = pin.getFrameJacobian(
            model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        jacobian_rate = pin.getFrameJacobianTimeVariation(
            model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        jacobians.append(np.asarray(jacobian, dtype=float))
        biases.append(np.asarray(jacobian_rate @ dq, dtype=float))
    return np.stack(poses), np.stack(jacobians), np.stack(biases)


def _assert_pose_close(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(actual[:, :3], expected[:, :3], rtol=2e-5, atol=3e-6)
    signs = np.where(
        np.sum(actual[:, 3:] * expected[:, 3:], axis=1, keepdims=True) < 0.0,
        -1.0,
        1.0,
    )
    np.testing.assert_allclose(actual[:, 3:] * signs, expected[:, 3:], rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize("dof", (2, 5, 8))
def test_arbitrary_scalar_chain_matches_cpu_bias(tmp_path: Path, dof: int) -> None:
    torch = _cuda_modules()
    urdf = _write_serial_chain(tmp_path / f"chain_{dof}.urdf", dof)
    robot = eik.RobotModel(str(urdf), floating_base=False)
    frame = f"link_{dof - 1}"
    configurations = np.stack(
        (
            np.linspace(-0.31, 0.23, dof),
            np.linspace(0.19, -0.27, dof),
            np.linspace(-0.11, 0.17, dof),
        )
    )
    velocities = np.stack(
        (
            np.linspace(-0.63, 0.47, dof),
            np.linspace(0.51, -0.39, dof),
            np.zeros(dof),
        )
    )
    evaluator = _evaluator_class()(
        3,
        urdf,
        tmp_path / "cache",
        _spec(robot, f"chain-{dof}", frame),
        frame,
    )
    q = torch.as_tensor(configurations, dtype=torch.float32, device="cuda")
    dq = torch.as_tensor(velocities, dtype=torch.float32, device="cuda")

    pose, jacobian, bias = evaluator.evaluate(q, dq)
    torch.cuda.synchronize()
    expected_pose, expected_jacobian, expected_bias = _cpu_reference(
        urdf, frame, configurations, velocities
    )

    _assert_pose_close(pose.cpu().numpy(), expected_pose)
    np.testing.assert_allclose(jacobian.cpu().numpy(), expected_jacobian, rtol=4e-5, atol=7e-6)
    np.testing.assert_allclose(bias.cpu().numpy(), expected_bias, rtol=4e-3, atol=4e-4)
    np.testing.assert_allclose(bias[-1].cpu().numpy(), 0.0, atol=1e-7)
    assert evaluator.metadata()["bias_method"] == ("centered_directional_jacobian_difference")


def test_multiple_frames_have_cpu_row_order(tmp_path: Path) -> None:
    torch = _cuda_modules()
    urdf = _write_serial_chain(tmp_path / "multi.urdf", 5)
    robot = eik.RobotModel(str(urdf), floating_base=False)
    frames = ("link_2", "link_4")
    q_np = np.stack((np.linspace(-0.2, 0.3, 5), np.linspace(0.17, -0.24, 5)))
    dq_np = np.stack((np.linspace(0.6, -0.4, 5), np.linspace(-0.3, 0.5, 5)))
    evaluator = _evaluator_class()(
        2, urdf, tmp_path / "cache", _spec(robot, "multi", frames), frames
    )

    pose, jacobian, bias = evaluator.evaluate(
        torch.as_tensor(q_np, dtype=torch.float32, device="cuda"),
        torch.as_tensor(dq_np, dtype=torch.float32, device="cuda"),
    )
    torch.cuda.synchronize()

    expected_pose_parts, expected_jacobian_parts, expected_bias_parts = [], [], []
    for frame in frames:
        p, j, b = _cpu_reference(urdf, frame, q_np, dq_np)
        expected_pose_parts.append(p)
        expected_jacobian_parts.append(j)
        expected_bias_parts.append(b)
    expected_pose = np.stack(expected_pose_parts, axis=1)
    expected_jacobian = np.concatenate(expected_jacobian_parts, axis=1)
    expected_bias = np.concatenate(expected_bias_parts, axis=1)
    for frame_index in range(len(frames)):
        _assert_pose_close(pose[:, frame_index].cpu().numpy(), expected_pose[:, frame_index])
    np.testing.assert_allclose(jacobian.cpu().numpy(), expected_jacobian, rtol=4e-5, atol=7e-6)
    np.testing.assert_allclose(bias.cpu().numpy(), expected_bias, rtol=4e-3, atol=4e-4)


def test_cuda_graph_replays_changed_state_with_stable_outputs(tmp_path: Path) -> None:
    torch = _cuda_modules()
    urdf = _write_serial_chain(tmp_path / "captured.urdf", 6)
    robot = eik.RobotModel(str(urdf), floating_base=False)
    evaluator = _evaluator_class()(
        2,
        urdf,
        tmp_path / "cache",
        _spec(robot, "captured", "link_5"),
        "link_5",
    )
    static_q = torch.zeros((2, 6), dtype=torch.float32, device="cuda")
    static_dq = torch.zeros_like(static_q)
    evaluator.evaluate(static_q, static_dq)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_pose, captured_jacobian, captured_bias = evaluator.evaluate(static_q, static_dq)
    pointers = tuple(
        value.data_ptr() for value in (captured_pose, captured_jacobian, captured_bias)
    )

    configurations = np.stack((np.linspace(-0.27, 0.19, 6), np.linspace(0.23, -0.11, 6)))
    velocities = np.stack((np.linspace(-0.58, 0.42, 6), np.linspace(0.47, -0.34, 6)))
    static_q.copy_(torch.as_tensor(configurations, dtype=torch.float32, device="cuda"))
    static_dq.copy_(torch.as_tensor(velocities, dtype=torch.float32, device="cuda"))
    graph.replay()
    torch.cuda.synchronize()

    expected_pose, expected_jacobian, expected_bias = _cpu_reference(
        urdf, "link_5", configurations, velocities
    )
    _assert_pose_close(captured_pose.cpu().numpy(), expected_pose)
    np.testing.assert_allclose(
        captured_jacobian.cpu().numpy(), expected_jacobian, rtol=4e-5, atol=7e-6
    )
    np.testing.assert_allclose(captured_bias.cpu().numpy(), expected_bias, rtol=4e-3, atol=4e-4)
    graph.replay()
    torch.cuda.synchronize()
    assert pointers == tuple(
        value.data_ptr() for value in (captured_pose, captured_jacobian, captured_bias)
    )


def test_floating_base_is_explicitly_rejected(tmp_path: Path) -> None:
    _cuda_modules()
    urdf = _write_serial_chain(tmp_path / "floating.urdf", 2)
    robot = eik.RobotModel(str(urdf), floating_base=True)
    with pytest.raises(NotImplementedError, match="fixed-base"):
        _evaluator_class()(
            1,
            urdf,
            tmp_path / "cache",
            _spec(robot, "floating", "link_1"),
            "link_1",
        )
