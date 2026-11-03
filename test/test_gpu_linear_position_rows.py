"""Public fixed-base adapter qualification for opt-in native coupled rows."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import embodik as eik
from embodik.gpu.wbc import GpuWbcMultiFrameSolver

pytest.importorskip("torch")

from embodik.gpu.wbc._runtime.multi_pose_solver import (
    DeviceResidentMultiFramePoseSolver,
)


def _cuda():
    torch = pytest.importorskip("torch")
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch


def _solver(tmp_path: Path, *, batch=4, backend="warp_polytope", **options):
    path = tmp_path / "stage.urdf"
    path.write_text("""<robot name="coupled_stage"><link name="base"/>
<link name="x"/><joint name="x" type="prismatic"><parent link="base"/><child link="x"/>
<axis xyz="1 0 0"/><limit lower="-1" upper="1" velocity="2" effort="100"/></joint>
<link name="tool"/><joint name="y" type="prismatic"><parent link="x"/><child link="tool"/>
<axis xyz="0 1 0"/><limit lower="-1" upper="1" velocity="2" effort="100"/></joint></robot>""")
    robot = eik.RobotModel(str(path), floating_base=False)
    return GpuWbcMultiFrameSolver.from_robot(
        path,
        tmp_path / "cache",
        robot=robot,
        robot_name="coupled-stage",
        frames=("tool",),
        default_configuration=np.zeros(2),
        solver_backend=backend,
        batch_size=batch,
        frame_task_dimensions=(3,),
        iterations=1,
        dt=0.02,
        max_linear_speed=1.0,
        **options,
    )


def test_public_graph_coupled_limit_then_reverse_and_update_without_reset(tmp_path):
    torch = _cuda()
    solver = _solver(tmp_path)
    C = np.array([[1.0, 1.0], [1.0, -1.0]])
    solver.set_linear_position_constraints(C, [-0.5, -0.5], [0.5, 0.5])
    assert solver.get_linear_position_constraint_rows() == 2
    q = torch.zeros((4, 2), dtype=torch.float32, device="cuda")
    target = torch.zeros((4, 1, 7), dtype=torch.float32, device="cuda")
    target[:, :, 3] = 1.0
    target[:, :, :2] = 1.0
    for _ in range(50):
        previous = q.clone()
        result = solver.solve_device_batch(q, target)
        q = result.q_solution.clone()
        assert torch.isfinite(q).all()
        assert (
            torch.max(
                torch.abs(q.double() @ torch.tensor(C, dtype=torch.float64, device="cuda").T)
            ).item()
            <= 0.5 + 1e-9
        )
        assert torch.max(abs(q - previous)).item() <= 0.04 + 1e-7
        assert all(
            s in (solver.WORLD_STATUS_SUCCESS, solver.WORLD_STATUS_HELD)
            for s in result.world_status.tolist()
        )
    np.testing.assert_allclose(q.cpu(), 0.25, atol=1e-5, rtol=0)
    target[:, :, :2] = -0.1
    for _ in range(90):
        result = solver.solve_device_batch(q, target)
        q = result.q_solution.clone()
        assert (
            torch.max(
                torch.abs(q.double() @ torch.tensor(C, dtype=torch.float64, device="cuda").T)
            ).item()
            <= 0.5 + 1e-9
        )
    np.testing.assert_allclose(q.cpu(), -0.1, atol=0.005, rtol=0)
    # Setter must revoke the captured graph without moving the current state.
    previous = q.clone()
    solver.set_linear_position_constraints(C, [-0.3, -0.3], [0.3, 0.3])
    assert torch.equal(q, previous)
    target[:, :, :2] = 1.0
    for _ in range(50):
        result = solver.solve_device_batch(q, target)
        q = result.q_solution.clone()
        assert (
            torch.max(
                torch.abs(q.double() @ torch.tensor(C, dtype=torch.float64, device="cuda").T)
            ).item()
            <= 0.3 + 1e-9
        )
    np.testing.assert_allclose(q.cpu(), 0.15, atol=1e-5, rtol=0)


def test_public_rows_require_the_explicit_backend(tmp_path):
    _cuda()
    solver = _solver(tmp_path, backend="warp_srinv")
    with pytest.raises(ValueError, match="warp_polytope"):
        solver.set_linear_position_constraints([[1.0, 1.0]], [-0.5], [0.5])


@pytest.mark.parametrize(
    "C,lo,hi",
    [
        ([[1.0]], [-1.0], [1.0]),
        ([[1.0, 1.0]], [2.0], [1.0]),
        ([[float("nan"), 1.0]], [-1.0], [1.0]),
    ],
)
def test_invalid_row_updates_preserve_previous_configuration(tmp_path, C, lo, hi):
    _cuda()
    solver = _solver(tmp_path)
    solver.set_linear_position_constraints([[1.0, 1.0]], [-0.5], [0.5])
    with pytest.raises(ValueError):
        solver.set_linear_position_constraints(C, lo, hi)
    assert solver.get_linear_position_constraint_rows() == 1


@pytest.mark.parametrize(
    "family",
    [
        "collision",
        "centroidal",
        "com",
        "capture_point",
        "velocity_zmp",
        "torso",
        "secondary",
        "contact",
        "acceleration",
    ],
)
def test_unsupported_families_fail_explicitly(family):
    core = DeviceResidentMultiFramePoseSolver.__new__(DeviceResidentMultiFramePoseSolver)
    core.robot_spec = SimpleNamespace(
        floating_base=False,
        joint_velocity_indices=(0, 1),
        joint_velocity_sizes=(1, 1),
        joint_configuration_sizes=(1, 1),
        active_velocity_indices=(0, 1),
    )
    core.configuration_dim = core.velocity_dim = 2
    core.config = SimpleNamespace(collision_enabled=False, max_joint_acceleration_rad_s2=None)
    core.centroidal_enabled = core.torso_constraint_enabled = core.secondary_task_enabled = False
    core.com_constraint_enabled = False
    core.capture_point_constraint_enabled = False
    core.velocity_zmp_constraint_enabled = False
    core.frame_contact_constraints = (False,)
    if family == "collision":
        core.config.collision_enabled = True
    elif family == "acceleration":
        core.config.max_joint_acceleration_rad_s2 = 2.0
    elif family == "contact":
        core.frame_contact_constraints = (True,)
    else:
        setattr(
            core,
            {
                "centroidal": "centroidal_enabled",
                "com": "com_constraint_enabled",
                "capture_point": "capture_point_constraint_enabled",
                "velocity_zmp": "velocity_zmp_constraint_enabled",
                "torso": "torso_constraint_enabled",
                "secondary": "secondary_task_enabled",
            }[family],
            True,
        )
    with pytest.raises(ValueError, match="does not yet support"):
        core._validate_polytope_execution()
