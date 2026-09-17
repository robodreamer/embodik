"""Opt-in CUDA parity tests for the model-derived centroidal evaluator.

Run with ``EMBODIK_RUN_GPU_CENTROIDAL_TESTS=1 pytest -q
test/test_gpu_centroidal_evaluator.py``.  The opt-in keeps Newton compilation
and CUDA ownership out of the default CPU test suite.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import embodik as eik
from embodik.gpu.wbc import GpuWbcMultiFrameSolver
from embodik.gpu.wbc._runtime.newton_com import NewtonCoMEvaluator
from embodik.gpu.wbc.model_spec import robot_solve_spec_from_embodik

pytestmark = pytest.mark.skipif(
    os.environ.get("EMBODIK_RUN_GPU_CENTROIDAL_TESTS") != "1",
    reason="set EMBODIK_RUN_GPU_CENTROIDAL_TESTS=1 for Newton/CUDA parity tests",
)


def _cuda_modules():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch


def _spec(robot: eik.RobotModel, name: str):
    return robot_solve_spec_from_embodik(
        robot,
        name=name,
        model_hash=f"test-{name}-{robot.nq}-{robot.nv}",
        active_velocity_indices=tuple(range(robot.nv)),
    )


def _cpu_reference(
    robot: eik.RobotModel,
    configurations: np.ndarray,
    velocities: np.ndarray,
):
    positions, jacobians, centroidal_maps, biases = [], [], [], []
    for q, dq in zip(configurations, velocities, strict=True):
        robot.update_kinematics(q)
        positions.append(robot.get_com_position())
        jacobians.append(robot.get_com_jacobian())
        centroidal_maps.append(robot.compute_centroidal_momentum_matrix(q))
        biases.append(robot.compute_centroidal_momentum_matrix_bias(q, dq))
    return tuple(np.stack(values) for values in (positions, jacobians, centroidal_maps, biases))


def _assert_parity(
    evaluator: NewtonCoMEvaluator,
    robot: eik.RobotModel,
    configurations: np.ndarray,
    velocities: np.ndarray,
) -> None:
    torch = _cuda_modules()
    q = torch.as_tensor(configurations, dtype=torch.float32, device=evaluator.device)
    dq = torch.as_tensor(velocities, dtype=torch.float32, device=evaluator.device)
    position, jacobian, ag = evaluator.evaluate_centroidal(q)
    bias = evaluator.evaluate_centroidal_bias(q, dq)
    torch.cuda.synchronize(evaluator.device)
    expected_position, expected_jacobian, expected_ag, expected_bias = _cpu_reference(
        robot, configurations, velocities
    )
    np.testing.assert_allclose(position.cpu().numpy(), expected_position, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(jacobian.cpu().numpy(), expected_jacobian, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(ag.cpu().numpy(), expected_ag, rtol=4e-5, atol=5e-6)
    np.testing.assert_allclose(bias.cpu().numpy(), expected_bias, rtol=5e-5, atol=6e-6)
    torch.testing.assert_close(ag[:, :3, :], evaluator.total_mass * jacobian, rtol=2e-5, atol=3e-6)
    assert position.is_cuda and jacobian.is_cuda and ag.is_cuda and bias.is_cuda
    assert position.dtype == jacobian.dtype == ag.dtype == bias.dtype == torch.float32
    assert evaluator.supports_exact_centroidal_bias


def _write_serial_chain(path: Path, dof: int) -> Path:
    links = ["""<link name="base">
  <inertial><origin xyz="0.03 -0.02 0.04" rpy="0.01 0.02 -0.03"/>
    <mass value="2.0"/><inertia ixx="0.20" ixy="0.003" ixz="-0.002"
      iyy="0.24" iyz="0.004" izz="0.29"/></inertial>
</link>"""]
    joints = []
    axes = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
    for index in range(dof):
        mass = 0.55 + 0.17 * index
        links.append(f"""<link name="link_{index}">
  <inertial><origin xyz="{0.04 + 0.01 * index:.4f} -0.015 0.025"
    rpy="0.02 -0.01 0.03"/><mass value="{mass:.6f}"/>
    <inertia ixx="{0.025 + 0.004 * index:.6f}" ixy="0.0007" ixz="-0.0004"
      iyy="{0.031 + 0.005 * index:.6f}" iyz="0.0005"
      izz="{0.038 + 0.006 * index:.6f}"/></inertial>
</link>""")
        axis = axes[index % len(axes)]
        joint_type = "prismatic" if index % 3 == 1 else "revolute"
        joints.append(f"""<joint name="joint_{index}" type="{joint_type}">
  <parent link="{'base' if index == 0 else f'link_{index - 1}'}"/>
  <child link="link_{index}"/>
  <origin xyz="0.11 {0.02 * ((index % 2) * 2 - 1):.4f} 0.07"
    rpy="0.01 -0.02 0.015"/>
  <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>
  <limit lower="-1.2" upper="1.3" velocity="3.0" effort="20.0"/>
</joint>""")
    path.write_text(
        '<?xml version="1.0"?>\n<robot name="arbitrary_chain">\n'
        + "\n".join(links + joints)
        + "\n</robot>\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("dof", (2, 5))
def test_arbitrary_fixed_dimensions_match_embodik(tmp_path: Path, dof: int) -> None:
    _cuda_modules()
    urdf = _write_serial_chain(tmp_path / f"chain_{dof}.urdf", dof)
    robot = eik.RobotModel(str(urdf), floating_base=False)
    configurations = np.stack(
        (
            np.linspace(-0.31, 0.23, dof),
            np.linspace(0.19, -0.27, dof),
            np.zeros(dof),
        )
    )
    velocities = np.stack(
        (
            np.linspace(-0.23, 0.17, dof),
            np.linspace(0.31, -0.13, dof),
            np.zeros(dof),
        )
    )
    evaluator = NewtonCoMEvaluator(
        len(configurations), urdf, tmp_path / f"cache_{dof}", _spec(robot, "chain")
    )

    _assert_parity(evaluator, robot, configurations, velocities)


def test_panda_batch_matches_embodik(tmp_path: Path) -> None:
    _cuda_modules()
    panda = pytest.importorskip("robot_descriptions.panda_description")
    urdf = Path(panda.URDF_PATH)
    robot = eik.RobotModel(str(urdf), floating_base=False)
    neutral = np.asarray(robot.neutral_configuration())
    configurations = np.stack(
        (
            neutral,
            neutral + np.linspace(-0.08, 0.06, robot.nq),
            neutral + np.linspace(0.04, -0.05, robot.nq),
        )
    )
    velocities = np.stack(
        (
            np.linspace(-0.18, 0.21, robot.nv),
            np.linspace(0.16, -0.12, robot.nv),
            np.zeros(robot.nv),
        )
    )
    evaluator = NewtonCoMEvaluator(
        len(configurations), urdf, tmp_path / "panda_cache", _spec(robot, "panda")
    )

    _assert_parity(evaluator, robot, configurations, velocities)


def test_free_root_columns_and_bias_match_embodik(tmp_path: Path) -> None:
    _cuda_modules()
    urdf = _write_serial_chain(tmp_path / "floating_chain.urdf", 3)
    robot = eik.RobotModel(str(urdf), floating_base=True)
    neutral = np.asarray(robot.neutral_configuration())
    rotated = neutral.copy()
    rotated[:3] = (0.17, -0.09, 0.31)
    axis = np.asarray((0.3, -0.4, 0.86))
    axis /= np.linalg.norm(axis)
    angle = 0.37
    rotated[3:7] = np.r_[axis * np.sin(angle / 2.0), np.cos(angle / 2.0)]
    rotated[7:] = (0.21, -0.14, 0.33)
    configurations = np.stack((neutral, rotated))
    velocities = np.stack(
        (
            np.linspace(-0.16, 0.21, robot.nv),
            np.asarray((0.21, -0.16, 0.09, -0.13, 0.24, -0.07, -0.23, 0.31, 0.17)),
        )
    )
    evaluator = NewtonCoMEvaluator(
        len(configurations), urdf, tmp_path / "floating_cache", _spec(robot, "float")
    )

    _assert_parity(evaluator, robot, configurations, velocities)


def test_bias_cuda_graph_replays_changed_inputs(tmp_path: Path) -> None:
    torch = _cuda_modules()
    urdf = _write_serial_chain(tmp_path / "captured_chain.urdf", 4)
    robot = eik.RobotModel(str(urdf), floating_base=False)
    evaluator = NewtonCoMEvaluator(2, urdf, tmp_path / "captured_cache", _spec(robot, "captured"))
    static_q = torch.zeros((2, robot.nq), dtype=torch.float32, device="cuda")
    static_dq = torch.zeros((2, robot.nv), dtype=torch.float32, device="cuda")
    evaluator.evaluate_centroidal_bias(static_q, static_dq)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_bias = evaluator.evaluate_centroidal_bias(static_q, static_dq)

    configurations = np.stack(
        (np.linspace(-0.27, 0.19, robot.nq), np.linspace(0.23, -0.11, robot.nq))
    )
    velocities = np.stack((np.linspace(-0.18, 0.22, robot.nv), np.linspace(0.17, -0.14, robot.nv)))
    static_q.copy_(torch.as_tensor(configurations, dtype=torch.float32, device="cuda"))
    static_dq.copy_(torch.as_tensor(velocities, dtype=torch.float32, device="cuda"))
    graph.replay()
    torch.cuda.synchronize()

    expected = _cpu_reference(robot, configurations, velocities)[3]
    np.testing.assert_allclose(captured_bias.cpu().numpy(), expected, rtol=5e-5, atol=6e-6)


def test_full_model_subset_centroidal_solver_captures_and_publishes(tmp_path: Path) -> None:
    """Exercise measured-state mapping, graph capture, and compact publication."""

    torch = _cuda_modules()
    urdf = _write_serial_chain(tmp_path / "subset_chain.urdf", 5)
    robot = eik.RobotModel(str(urdf), floating_base=False)
    q_full = np.zeros(robot.nq)
    active_names = tuple(robot.get_joint_names()[:4])
    target_frame = "link_4"
    robot.update_configuration(q_full)
    pose = robot.get_frame_pose(target_frame)
    xyzw = np.asarray(eik.r2q(pose.rotation, order="xyzs"), dtype=np.float32)
    target_row = np.concatenate(
        (np.asarray(pose.translation, dtype=np.float32), xyzw[[3, 0, 1, 2]])
    )
    polygon = np.asarray(((-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)))
    solver = GpuWbcMultiFrameSolver.from_robot(
        urdf,
        tmp_path / "solver_cache",
        robot=robot,
        robot_name="unseen_subset_chain",
        frames=(target_frame,),
        active_joint_names=active_names,
        default_configuration=q_full,
        batch_size=2,
        iterations=1,
        dt=0.01,
        solver_backend="warp_srinv",
        com_support_polygon_xy=polygon,
        com_full_robot=robot,
        com_full_default_configuration=q_full,
        capture_point_support_polygon_xy=polygon,
        velocity_zmp_support_polygon_xy=polygon,
        centroidal_momentum_target=(0.0,) * 6,
        centroidal_momentum_axis_mask=(True, True, False, False, False, False),
        centroidal_momentum_weight=0.01,
    )
    q = torch.zeros((2, len(active_names)), dtype=torch.float32, device="cuda")
    target = (
        torch.as_tensor(target_row, dtype=torch.float32, device="cuda")
        .reshape(1, 1, 7)
        .expand(2, -1, -1)
        .clone()
    )
    measured = torch.zeros_like(q)

    first = solver.solve_device_batch(q, target, measured, measured)
    second = solver.solve_device_batch(q, target, measured, measured)
    torch.cuda.synchronize()

    assert first.compact_publication is not None
    assert second.compact_publication is not None
    assert second.compact_publication.dtype == torch.float32
    assert bool(torch.isfinite(second.q_solution).all())
    assert bool(second.capture_point_constraint_feasible.all())
    assert bool(second.velocity_zmp_constraint_feasible.all())
