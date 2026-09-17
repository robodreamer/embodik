"""CPU/mocked adapter regressions; no CUDA kernels or robot assets are needed."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from embodik.gpu.wbc import _runtime as prototypes
from embodik.gpu.wbc import model_spec as package
from embodik.gpu.wbc import solver as helper

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("floating", [False, True])
def test_com_constructor_and_runtime_forwarding(factory, floating):
    polygon = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    adapter = factory(
        floating,
        com_support_polygon_xy=polygon,
        com_max_constraints=6,
        com_excluded_velocity_indices=(4,),
    )
    assert adapter._solver.options["com_support_polygon_xy"] == polygon
    assert adapter._solver.options["com_max_constraints"] == 6
    assert adapter._solver.options["com_excluded_velocity_indices"] == (4,)
    adapter.configure_runtime(
        com_enabled=True,
        com_support_polygon_xy=polygon,
        com_margin=0.15,
        com_vel_max=0.3,
        com_acc_max=0.2,
        com_use_acceleration_limits=False,
        com_proximity_fraction=0.0,
    )
    adapter._solver.configure_com_constraint.assert_called_once_with(
        enabled=True,
        support_polygon_xy=polygon,
        margin=0.15,
        vel_max=0.3,
        acc_max=0.2,
        use_acceleration_limits=False,
        proximity_fraction=0.0,
    )


def test_warp_graph_capture_supports_constrained_layouts(factory):
    pose_only = factory(False, solver_backend="warp_srinv")
    assert pose_only._solver.config.standalone_cuda_graph_enabled

    constrained = factory(
        False,
        solver_backend="warp_srinv",
        posture_target_configuration=(0.0, 0.0, 0.0),
        posture_velocity_indices=(0, 1, 2),
        posture_weights=(1.0, 1.0, 1.0),
    )
    assert constrained._solver.config.standalone_cuda_graph_enabled

    floating = factory(True, solver_backend="warp_srinv")
    assert floating._solver.config.standalone_cuda_graph_enabled


def test_warp_runtime_policy_change_invalidates_captured_graph(factory):
    adapter = factory(
        False,
        solver_backend="warp_srinv",
        collision_pairs=(("a", "b"),),
    )
    assert adapter._solver._graph is not None
    adapter.configure_runtime(collision_enabled=False)
    assert adapter._solver._graph is None


def test_warp_collision_debug_layout_is_static(factory):
    adapter = factory(
        False,
        solver_backend="warp_srinv",
        collision_pairs=(("a", "b"),),
    )
    assert adapter._solver.config.collision_debug_enabled


@pytest.fixture
def factory(tmp_path, monkeypatch):
    source = tmp_path / "robot.urdf"
    source.write_text("<robot name='test'/>")
    spec = SimpleNamespace(
        model_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
        configuration_dim=11,
        velocity_dim=10,
        active_velocity_indices=(1, 4, 7),
        active_joint_names=("a", "b", "c"),
        task_frames=("old_frame",),
    )
    params = SimpleNamespace(
        robot_spec=spec,
        default_configuration=(0.0,) * 11,
        joint_lower=(-1.0,) * 3,
        joint_upper=(1.0,) * 3,
        joint_velocity_limits=(1.0,) * 3,
    )
    for name in (
        "pose_model_parameters_from_embodik",
        "floating_pose_model_parameters_from_embodik",
    ):
        monkeypatch.setattr(package, name, lambda *a, **k: params)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)

    class Core:
        def __init__(self, *args, **kwargs):
            self.batch_size = args[0]
            self.options = kwargs
            self.device = "cpu"  # Tensor allocation only; attribution is mocked below.
            self.config = kwargs["config"]
            self.collision = SimpleNamespace(decode_shape_pair=Mock(return_value=("a", "b")))
            count = len(kwargs["frames"])
            self._frame_position_gains = torch.ones(count)
            self._frame_orientation_gains = torch.ones(count)
            self.posture_task_enabled = True
            self.torso_constraint_enabled = True
            self.secondary_frame_tasks_enabled = True
            target = kwargs.get("posture_target_configuration")
            self._posture_target = torch.tensor(target if target is not None else (0.0, 0.0, 0.0))
            self._posture_weights = torch.ones(len(kwargs.get("posture_weights", ())))
            self._torso_reference_pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            self._torso_lower_relative_limits = torch.full((6,), -0.1)
            self._torso_upper_relative_limits = torch.full((6,), 0.1)
            self._torso_axis_mask = torch.ones(6, dtype=torch.bool)
            self.configure_posture = Mock()
            self.configure_com_constraint = Mock()
            self.configure_torso_constraint = Mock()
            self._secondary_frame_targets = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
            self._secondary_frame_position_gains = torch.ones(1)
            self._secondary_frame_orientation_gains = torch.ones(1)
            self._secondary_frame_weights = torch.ones(1)
            self.configure_secondary_frames = Mock()
            self.solve = Mock()
            self._graph = object()

    monkeypatch.setattr(prototypes, "DeviceResidentMultiFramePoseSolver", Core)
    robot = SimpleNamespace(
        nq=11, nv=10, get_joint_config_index=lambda n: {"a": 2, "b": 5, "c": 8}[n]
    )
    manifest = tmp_path / "manifest.json"

    def build(floating=False, contract_changes=None, **options):
        contract = {
            "robot_spec": vars(spec).copy(),
            "task_layout": {"task_dimensions": [99], "constraint_rows": 99},
        }
        contract["robot_spec"].update(contract_changes or {})
        manifest.write_text(json.dumps(contract))
        common = {
            "robot": robot,
            "robot_name": "test",
            "frames": ("tool",),
            "default_configuration": params.default_configuration,
        }
        if floating:
            cls = helper.GpuWbcFloatingMultiFrameSolver
            common.update(
                active_velocity_indices=(1, 4, 7),
                frame_task_dimensions=(3,),
                frame_position_gains=(2.0,),
                frame_orientation_gains=(0.0,),
                base_velocity_limits=(1.0,) * 6,
            )
        else:
            cls = helper.GpuWbcMultiFrameSolver
            common.update(
                active_joint_names=("a", "b", "c"),
                solver_backend="torch_srinv",
                frame_task_dimensions=(3,),
            )
        common.update(options)
        return cls(manifest, source, tmp_path, **common)

    build.spec = spec
    return build


def test_public_wrapper_forwards_parallel_world_capacity(factory):
    adapter = factory(False, solver_backend="warp_srinv", batch_size=8)

    assert adapter.batch_size == 8
    assert adapter._solver.batch_size == 8
    assert adapter._previous_velocity.shape == (8, 3)


def test_public_wrapper_rejects_batched_cusolver_with_actionable_error(factory):
    with pytest.raises(ValueError, match="use warp_srinv for parallel worlds"):
        factory(False, solver_backend="cusolver_srinv", batch_size=8)


@pytest.mark.parametrize("floating", [False, True])
def test_native_manifest_reuses_model_with_new_frame_layout_and_forwards_constraints(
    factory, floating
):
    target = (0.0,) * (11 if floating else 3)
    options = {
        "collision_pairs": (("link_a", "link_b"),),
        "collision_contacts_per_world": 17,
        "collision_triangle_pairs_per_world": 23,
        "collision_max_constraints": 2,
        "collision_graph_repair_iterations": 1,
        "collision_min_distance_m": 0.025,
        "collision_query_distance_m": 0.12,
        "adaptive_dt": True,
        "posture_target_configuration": target,
        "posture_velocity_indices": (4, 7),
        "posture_weights": (0.3, 0.7),
        "posture_gain": 2.0,
        "posture_rank_tolerance": 1e-5,
        "locked_velocity_indices": (1,),
        "secondary_frame_names": ("body",),
        "secondary_frame_task_dimensions": (6,),
        "secondary_frame_target_poses_wxyz": ((0, 0, 0, 1, 0, 0, 0),),
        "secondary_frame_position_gains": (8.0,),
        "secondary_frame_orientation_gains": (4.0,),
        "secondary_frame_weights": (1.0,),
        "torso_frame_name": "body",
        "torso_reference_pose_xyzw": (0, 0, 0, 0, 0, 0, 1),
        "torso_lower_relative_limits": (-0.1,) * 6,
        "torso_upper_relative_limits": (0.1,) * 6,
        "torso_axis_mask": (True,) * 6,
        "torso_velocity_limits": (0.7,) * 6,
        "torso_acceleration_limits": (0.3,) * 6,
        "torso_excluded_active_velocity_indices": (2,),
    }
    adapter = factory(floating, **options)
    for key, value in options.items():
        actual = (
            getattr(adapter._solver.config, key)
            if key.startswith("collision_") and key != "collision_pairs" or key == "adaptive_dt"
            else adapter._solver.options[key]
        )
        assert actual == value
    assert not adapter._solver.config.collision_debug_enabled
    assert adapter.configuration_dim == len(target)


@pytest.mark.parametrize("floating", [False, True])
@pytest.mark.parametrize(
    "change",
    [
        {"model_hash": "wrong"},
        {"configuration_dim": 12},
        {"velocity_dim": 12},
        {"active_velocity_indices": [7, 4, 1]},
    ],
)
def test_native_manifest_retains_model_attribution(factory, floating, change):
    with pytest.raises(ValueError, match="manifest does not match"):
        factory(floating, contract_changes=change)


def test_fixed_manifest_retains_joint_order(factory):
    with pytest.raises(ValueError, match="active joint order"):
        factory(contract_changes={"active_joint_names": ["c", "b", "a"]})


def test_generated_manifest_requires_exact_frame_layout(factory, monkeypatch):
    artifact = {
        "robot_spec": vars(factory.spec),
        "shape": {"task_dimensions": (6,), "velocity_dim": 3, "constraint_rows": 3},
    }
    monkeypatch.setattr(helper, "_load_fi_artifact", lambda _: (object(), object(), artifact))
    with pytest.raises(ValueError, match="ordered task frames"):
        factory(solver_backend="fi_pesns")


def test_fixed_posture_target_is_compact_and_indices_stay_in_source_order(factory):
    with pytest.raises(ValueError, match="posture target"):
        factory(posture_target_configuration=(0.0,) * 11)
    adapter = factory(
        posture_target_configuration=(0.1, 0.2, 0.3),
        posture_velocity_indices=(4,),
        posture_weights=(1.0,),
    )
    assert adapter._solver.options["posture_velocity_indices"] == (4,)
    np.testing.assert_equal(adapter.extract_active_configuration(np.arange(11)), [2, 5, 8])
    adapter.configure_runtime(posture_target_configuration=(0.3, 0.2, 0.1), posture_gain=0.0)
    adapter._solver.configure_posture.assert_called_once_with(
        target_configuration=(0.3, 0.2, 0.1), gain=0.0, weights=None
    )


@pytest.mark.parametrize("floating", [False, True])
def test_runtime_updates_shape_stable_options(factory, floating):
    adapter = factory(floating, frames=("left", "right"))
    adapter.configure_runtime(
        dt=0.02,
        iterations=4,
        adaptive_dt=True,
        acceleration_limits_enabled=True,
        max_joint_acceleration_rad_s2=12.0,
        collision_enabled=False,
        frame_position_gains=(3.0, 5.0),
        frame_orientation_gains=(0.0, 2.0),
        collision_min_distance_m=0.03,
        torso_lower_relative_limits=(-0.2,) * 6,
        torso_upper_relative_limits=(0.2,) * 6,
        secondary_frame_target_poses_wxyz=((0.01, 0, 0, 1, 0, 0, 0),),
        secondary_frame_weights=(0.5,),
    )
    assert adapter._solver.config.dt == 0.02
    assert adapter._solver.config.iterations == 4
    assert adapter._solver.config.adaptive_dt
    assert adapter._solver.config.max_joint_acceleration_rad_s2 == 12.0
    assert not adapter._solver.config.collision_enabled
    assert adapter._solver._frame_position_gains.tolist() == [3.0, 5.0]
    assert adapter._solver._frame_orientation_gains.tolist() == [0.0, 2.0]
    adapter._solver.configure_torso_constraint.assert_called_once()
    adapter._solver.configure_secondary_frames.assert_called_once()

    adapter.configure_runtime(
        dt=0.02,
        iterations=4,
        adaptive_dt=True,
        acceleration_limits_enabled=True,
        max_joint_acceleration_rad_s2=12.0,
        collision_enabled=False,
        frame_position_gains=(3.0, 5.0),
        frame_orientation_gains=(0.0, 2.0),
        collision_min_distance_m=0.03,
        torso_lower_relative_limits=(-0.2,) * 6,
        torso_upper_relative_limits=(0.2,) * 6,
        secondary_frame_target_poses_wxyz=((0.01, 0, 0, 1, 0, 0, 0),),
        secondary_frame_weights=(0.5,),
    )
    adapter._solver.configure_torso_constraint.assert_called_once()
    adapter._solver.configure_secondary_frames.assert_called_once()
    with pytest.raises(ValueError, match="rebuilding"):
        adapter.configure_runtime(collision_query_distance_m=0.3)
    with pytest.raises(ValueError, match="match frames"):
        adapter.configure_runtime(frame_position_gains=(1.0,))
    adapter.configure_runtime(acceleration_limits_enabled=False)
    assert adapter._solver.config.max_joint_acceleration_rad_s2 is None


def result_for(adapter):
    return SimpleNamespace(
        fallback_used=False,
        actual_device="cuda:0",
        q_solution=torch.ones((1, adapter.configuration_dim)),
        accepted_velocity=torch.tensor([[0.12, 0.23, 0.34]]),
        position_error_m=torch.tensor([[0.02]]),
        orientation_error_rad=torch.tensor([[0.03]]),
        converged=torch.tensor([False]),
        kernel_time_ms=0.2,
        minimum_collision_distance_m=torch.tensor([0.06]),
        collision_active=torch.tensor([True]),
        collision_step_accepted=torch.tensor([True]),
        collision_overflow=torch.tensor([False]),
        collision_constraint_applied=torch.tensor([True]),
        effective_dt=torch.tensor([0.03]),
        torso_constraint_enabled=True,
        torso_constraint_applied=torch.tensor([True]),
        torso_constraint_feasible=torch.tensor([True]),
        posture_task_enabled=True,
        posture_task_applied=torch.tensor([True]),
        posture_primary_residual_increase=torch.tensor([0.0]),
        posture_secondary_residual_before=torch.tensor([0.4]),
        posture_secondary_residual_after=torch.tensor([0.1]),
        secondary_task_enabled=True,
        secondary_task_applied=torch.tensor([True]),
        secondary_primary_residual_increase=torch.tensor([0.0]),
        secondary_residual_before=torch.tensor([0.5]),
        secondary_residual_after=torch.tensor([0.2]),
    )


@pytest.mark.parametrize("floating", [False, True])
def test_compact_publication_is_the_primary_result_readback(factory, floating):
    adapter = factory(floating)
    adapter._target_tensor = lambda _: torch.zeros((1, 1, 7))
    result = result_for(adapter)
    scalars = {name: float("nan") for name in prototypes.COMPACT_PUBLICATION_SCALARS}
    scalars.update(
        converged=0.0,
        minimum_collision_distance_m=0.061,
        collision_active=1.0,
        collision_step_accepted=1.0,
        collision_overflow=0.0,
        collision_constraint_applied=1.0,
        effective_dt=0.025,
        com_constraint_applied=1.0,
        com_constraint_feasible=1.0,
        minimum_com_slack_m=0.012,
        torso_constraint_applied=1.0,
        torso_constraint_feasible=1.0,
        posture_task_applied=1.0,
        posture_primary_residual_increase=0.0,
        posture_secondary_residual_before=0.4,
        posture_secondary_residual_after=0.1,
        secondary_task_applied=1.0,
        secondary_primary_residual_increase=0.0,
        secondary_residual_before=0.5,
        secondary_residual_after=0.2,
        spectral_solve_ok=1.0,
    )
    solved = np.linspace(0.1, 0.1 * adapter.configuration_dim, adapter.configuration_dim)
    result.compact_publication = torch.tensor(
        [[
            *solved,
            0.021,
            0.031,
            *(scalars[name] for name in prototypes.COMPACT_PUBLICATION_SCALARS),
        ]],
        dtype=torch.float32,
    )
    result.com_constraint_enabled = True
    result.q_solution.fill_(99.0)
    result.position_error_m.fill_(99.0)
    result.orientation_error_rad.fill_(99.0)
    adapter._solver.solve.return_value = result

    summary = adapter.solve_step(np.zeros(adapter.configuration_dim), ())

    np.testing.assert_allclose(summary.joints, solved, atol=1e-6)
    assert summary.position_errors == pytest.approx((0.021,))
    assert summary.rotation_errors == pytest.approx((0.031,))
    assert summary.minimum_collision_distance_m == pytest.approx(0.061)
    assert summary.minimum_com_slack_m == pytest.approx(0.012)
    assert summary.effective_dt == pytest.approx(0.025)


@pytest.mark.parametrize("floating", [False, True])
def test_accepted_velocity_history_geometry_reset_and_lazy_debug(factory, floating):
    adapter = factory(floating)
    target = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]])
    adapter._target_tensor = lambda _: target.clone()
    result = result_for(adapter)
    histories = []

    def solve(q, t, velocity, current=None, **options):
        histories.append(velocity.clone())
        return result

    adapter._solver.solve.side_effect = solve
    q = np.zeros(adapter.configuration_dim)
    summary = adapter.solve_step(q, ())
    adapter.solve_step(q, ())
    target[..., 3:] *= -1  # same geometry
    adapter.solve_step(q, ())
    target[..., 0] = 0.2
    adapter.solve_step(q, ())
    torch.testing.assert_close(histories[0], torch.zeros((1, 3)))
    torch.testing.assert_close(histories[1], result.accepted_velocity)
    torch.testing.assert_close(histories[2], result.accepted_velocity)
    torch.testing.assert_close(histories[3], torch.zeros((1, 3)))
    assert summary.posture_task_applied and summary.torso_constraint_applied
    assert summary.effective_dt == pytest.approx(0.03)
    assert summary.posture_secondary_residual_after == pytest.approx(0.1)
    assert summary.secondary_task_applied
    assert summary.secondary_residual_after == pytest.approx(0.2)
    assert summary.collision_debug is None
    adapter._solver.collision.decode_shape_pair.assert_not_called()
    # No witness fields existed above: debug-off must never read them.
    result.closest_collision_shape_pair = torch.tensor([[2, 4]])
    result.closest_collision_point0_world_m = torch.zeros((1, 3))
    result.closest_collision_point1_world_m = torch.ones((1, 3))
    summary = adapter.solve_step(q, (), include_collision_debug=True)
    assert adapter._solver.config.collision_debug_enabled
    assert summary.collision_debug.object_a == "a"
    adapter.solve_step(q, ())
    assert not adapter._solver.config.collision_debug_enabled


@pytest.mark.parametrize("floating", [False, True])
@pytest.mark.parametrize("device,fallback", [("cpu", False), ("cuda:0", True)])
def test_solve_rejects_fallback_and_wrong_device(factory, floating, device, fallback):
    adapter = factory(floating)
    adapter._target_tensor = lambda _: torch.zeros((1, 1, 7))
    adapter._solver.solve.return_value = SimpleNamespace(
        actual_device=device, fallback_used=fallback
    )
    with pytest.raises(RuntimeError, match="attribution"):
        adapter.solve_step(np.zeros(adapter.configuration_dim), ())


@pytest.mark.parametrize("floating", [False, True])
def test_cuda_required_before_construction(factory, monkeypatch, floating):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        factory(floating)


def test_runtime_invalid_task_update_is_atomic(factory):
    adapter = factory()
    old_config = adapter._solver.config
    with pytest.raises(ValueError, match="lower limits"):
        adapter.configure_runtime(dt=0.04, posture_gain=0.5, torso_lower_relative_limits=(0.2,) * 6)
    assert adapter._solver.config is old_config
    adapter._solver.configure_posture.assert_not_called()
    adapter._solver.configure_torso_constraint.assert_not_called()


def test_posture_target_changes_reset_acceleration_history(factory):
    adapter = factory(posture_target_configuration=(0.0, 0.0, 0.0))
    adapter._previous_velocity.fill_(1)
    adapter.configure_runtime(posture_target_configuration=(0.0, 0.0, 0.0))
    assert bool((adapter._previous_velocity == 1).all())
    adapter.configure_runtime(posture_target_configuration=(0.1, 0.0, 0.0))
    assert bool((adapter._previous_velocity == 0).all())
