"""G1 GPU orchestration tests without CUDA or a viewer."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from examples.example_helpers import gpu_g1_retargeting_ik as module


@pytest.fixture(params=[5, 11])
def setup(request, monkeypatch):
    # Different articulated dimensions exercise model metadata, not G1 constants.
    count = request.param
    names = ("floating", *(f"joint_{i}" for i in range(count)))
    spans = {
        name: (0, 6, 7) if i == 0 else (i + 5, 1, 1) for i, name in enumerate(names)
    }
    robot = NS(
        nq=count + 7,
        nv=count + 6,
        get_joint_names=lambda: names,
        get_joint_velocity_index=lambda n: spans[n][0],
        get_joint_velocity_size=lambda n: spans[n][1],
        get_joint_config_size=lambda n: spans[n][2],
        get_frame_names=lambda: ("pelvis",),
    )
    frames = {
        k: "model_" + k
        for k in (
            "right_palm",
            "left_palm",
            "right_ankle",
            "left_ankle",
            "imu_in_torso",
        )
    }
    args = NS(gpu_wbc_manifest=Path("manifest"), gpu_wbc_cache_dir=Path("cache"))
    instances = []

    class Solver:
        def __init__(self, *args, **options):
            self.options, self.args = options, args
            self.runtime, self.calls = [], []
            self.status = "SAFE_STEP"
            self.overflow = False
            self.reset_count = 0
            instances.append(self)

        def configure_runtime(self, **options):
            self.options.update(options)
            self.runtime.append(options)

        def warm_up(self, q, targets):
            assert len(targets) == len(self.options["frames"])

        def reset_state(self):
            self.reset_count += 1

        def solve_step(self, q, targets, **options):
            self.calls.append(options)
            return NS(
                joints=np.asarray(q) + 0.001,
                status=self.status,
                collision_overflow=self.overflow,
                collision_step_accepted=True,
                torso_constraint_feasible=True,
                collision_debug="gpu witness",
            )

    monkeypatch.setattr(
        module,
        "g1_collision_pairs_for_preset",
        lambda robot, preset: [(preset + "_a", preset + "_b")],
    )
    q = np.zeros(robot.nq)
    gpu = module.GpuG1RetargetingIK(
        args, robot, Path("robot.urdf"), frames, q, solver_factory=Solver
    )
    targets = {k: np.eye(4) for k in gpu.frames}
    return gpu, q, targets, instances


def test_model_derived_hands_feet_and_runtime_collision(setup):
    gpu, q, targets, instances = setup
    step = gpu.solve(q, targets)
    assert step.accepted
    solver = instances[0]
    assert solver.args == (Path("manifest"), Path("robot.urdf"), Path("cache"))
    assert solver.options["active_velocity_indices"] == tuple(range(len(q) - 1))
    assert solver.options["frames"][:2] == ("model_right_palm", "model_left_palm")
    assert not any(solver.options["frame_contact_constraints"])
    assert solver.options["collision_pairs"] == (("core_a", "core_b"),)
    assert not solver.options["collision_enabled"]
    assert not solver.calls[0]["include_collision_debug"]
    gpu.solve(
        q,
        targets,
        collision_enabled=True,
        collision_min_distance=0.08,
        include_collision_debug=True,
        adaptive_dt=False,
    )
    assert len(instances) == 1
    assert solver.options["collision_min_distance_m"] == 0.08
    assert not solver.options["adaptive_dt"]
    assert solver.calls[-1]["include_collision_debug"]
    assert gpu.last_result.collision_debug == "gpu witness"
    gpu.solve(q, targets, collision_preset="core_plus_legs")
    assert len(instances) == 2
    assert instances[-1].options["collision_pairs"] == (
        ("core_plus_legs_a", "core_plus_legs_b"),
    )


def test_posture_and_upright_use_ordered_third_priority(setup):
    gpu, q, targets, instances = setup
    assert gpu.solve(q, targets, posture_weight=0.02).accepted
    options = instances[-1].options
    assert options["posture_velocity_indices"] == gpu.posture_indices
    assert not set(gpu.base_indices).intersection(options["posture_velocity_indices"])
    assert len(options["posture_weights"]) == len(gpu.posture_indices)
    assert gpu.solve(q, targets, upright_target=np.eye(4)).accepted
    assert instances[-1].options["secondary_frame_names"] == (gpu.upright_frame,)
    step = gpu.solve(q, targets, upright_target=np.eye(4), posture_weight=0.02)
    assert step.accepted
    assert instances[-1].options["posture_priority"] == 2
    assert instances[-1].options["secondary_frame_names"] == (gpu.upright_frame,)
    assert instances[-1].options["posture_velocity_indices"] == gpu.posture_indices
    assert np.linalg.norm(step.joints - q) > 0
    assert gpu.last_result is step.gpu_result


@pytest.mark.parametrize(
    "unsupported",
    [
        dict(quality_recovery=True),
        dict(solve_policy="MIN_ERROR"),
    ],
)
def test_unsupported_semantics_never_allocate_or_solve(setup, unsupported):
    gpu, q, targets, instances = setup
    step = gpu.solve(q, targets, **unsupported)
    assert not step.accepted and not instances
    np.testing.assert_array_equal(step.joints, q)


def test_com_polygon_allocates_once_and_updates_runtime(setup):
    gpu, q, targets, instances = setup
    polygon = np.array([[-0.2, -0.1], [0.2, -0.1], [0.2, 0.1], [-0.2, 0.1]])
    step = gpu.solve(
        q,
        targets,
        com_enabled=True,
        com_support_polygon=polygon,
        com_margin=0.05,
        com_vel_max=0.4,
        com_acc_max=0.2,
        com_use_acceleration_limits=False,
        com_proximity_fraction=0.0,
    )
    assert step.accepted
    solver = instances[0]
    np.testing.assert_array_equal(solver.options["com_support_polygon_xy"], polygon)
    runtime = solver.runtime[-1]
    assert runtime["com_enabled"] is True
    np.testing.assert_array_equal(runtime["com_support_polygon_xy"], polygon)
    assert runtime["com_margin"] == 0.05
    assert runtime["com_vel_max"] == 0.4
    assert runtime["com_acc_max"] == 0.2
    assert runtime["com_use_acceleration_limits"] is False
    assert runtime["com_proximity_fraction"] == 0.0
    assert gpu.solve(q, targets, com_support_polygon=polygon).accepted
    assert len(instances) == 1
    held = gpu.solve(q, targets, com_enabled=True)
    assert not held.accepted and "support polygon" in held.status


def test_contacts_overflow_and_no_stale_debug(setup):
    gpu, q, targets, instances = setup
    assert gpu.solve(q, targets, contact_keys=("right_ankle", "left_ankle")).accepted
    assert instances[0].options["frame_contact_constraints"] == (
        False,
        False,
        True,
        True,
        False,
    )
    targets["right_ankle"][0, 3] += 0.01
    for _ in range(2):
        step = gpu.solve(q, targets, contact_keys=("right_ankle", "left_ankle"))
        assert not step.accepted and "fixed foot" in step.status
    gpu.reset_state(q)
    assert gpu.solve(q, targets, contact_keys=("right_ankle", "left_ankle")).accepted
    instances[0].overflow = True
    step = gpu.solve(q, targets, contact_keys=("right_ankle", "left_ankle"))
    assert not step.accepted
    np.testing.assert_array_equal(step.joints, q)


def test_constructor_failure_cached_and_empty_collision_holds(setup, monkeypatch):
    gpu, q, targets, instances = setup

    def fail(*args, **kwargs):
        raise ValueError("manifest mismatch")

    gpu.factory = fail
    for _ in range(2):
        step = gpu.solve(q, targets)
        assert not step.accepted and "manifest mismatch" in step.status
    assert not instances
    gpu.failed_key = None
    gpu.pairs["core"] = ()
    assert "no available pairs" in gpu.solve(q, targets, collision_enabled=True).status


def test_gpu_cli_routes_headless_without_cpu_harness(monkeypatch):
    path = Path(__file__).parents[1] / "07_unitree_g1_retargeting_ik.py"
    spec = importlib.util.spec_from_file_location("g1_example_gpu_test", path)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    monkeypatch.setattr(
        "sys.argv",
        [
            str(path),
            "--gpu-wbc",
            "--gpu-wbc-manifest",
            "m",
            "--gpu-wbc-cache-dir",
            "c",
            "--headless-gpu-steps",
            "3",
        ],
    )
    calls = []
    # The public script uses the examples-root import alias.
    import example_helpers.gpu_g1_retargeting_ik as public_helper

    monkeypatch.setattr(
        public_helper,
        "run_headless",
        lambda args: calls.append(args.headless_gpu_steps),
    )
    monkeypatch.setattr(
        app, "_import_g1_harnesses", lambda: pytest.fail("CPU fallback")
    )
    app.main()
    assert calls == [3]


def test_orientation_secondary_does_not_pin_translation(monkeypatch):
    torch = pytest.importorskip("torch")
    solver_module = pytest.importorskip("embodik.gpu.wbc._runtime.multi_pose_solver")
    solver_type = solver_module.DeviceResidentMultiFramePoseSolver
    core = NS(
        torch=torch,
        batch_size=1,
        velocity_dim=8,
        _kinematics_frames=("torso",),
        _secondary_frame_indices=(0,),
        _posture_rank_tolerance=1e-6,
        _secondary_frame_targets=torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
        _secondary_frame_orientation_gains=torch.tensor([2.0]),
        _secondary_frame_weights=torch.tensor([1.0]),
        _secondary_frame_position_gains=torch.tensor([1.0]),
        _secondary_frame_task_dimensions=(6,),
        secondary_task_enabled=True,
        secondary_frame_tasks_enabled=True,
        posture_task_enabled=False,
        device="cpu",
        _locked_velocity_mask=None,
        config=NS(
            max_linear_speed=2.5,
            max_angular_speed=2.5,
            srinv_tolerance=0.1,
            srinv_damping=0.1,
        ),
    )
    solver_type.configure_task_hierarchy(
        core, posture_priority=2, secondary_frame_orientation_only=(True,)
    )
    jacobian = torch.zeros((1, 6, 8))
    jacobian[0, :, :6] = torch.eye(6)
    velocity = torch.zeros((1, 8))
    velocity[0, 0] = 0.2
    # Newton poses are xyzw; stored task targets are wxyz.
    current_pose = torch.zeros_like(core._secondary_frame_targets)
    current_pose[0, 5] = np.sin(0.15)
    current_pose[0, 6] = np.cos(0.15)
    result = solver_type._apply_secondary_tasks(
        core,
        velocity,
        torch.zeros((1, 1, 8)),
        torch.zeros((1, 9)),
        current_pose,
        jacobian,
        -torch.ones_like(velocity),
        torch.ones_like(velocity),
    )
    assert result[0][0, 0].item() == pytest.approx(0.2)
    assert result[0][0, 5].item() < -0.01


@pytest.mark.skipif(
    os.environ.get("EMBODIK_G1_GPU_TESTS") != "1", reason="opt-in CUDA/robot test"
)
def test_real_g1_gpu_headless_features(tmp_path):
    import hashlib
    import json

    from examples.example_helpers.g1_ik_runtime import _apply_g1_soft_knee_seed
    from examples.example_helpers.g1_model_utils import (
        create_g1_robot_model,
        resolve_frames_for_g1_base_mode,
        resolve_g1_collision_urdf_path,
    )

    robot = create_g1_robot_model(floating_base=True)
    source = resolve_g1_collision_urdf_path()
    q = _apply_g1_soft_knee_seed(robot, robot.neutral_configuration())
    robot.update_configuration(q)
    frames = resolve_frames_for_g1_base_mode(robot.get_frame_names())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "robot_spec": {
                    "model_hash": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "configuration_dim": int(robot.nq),
                    "velocity_dim": int(robot.nv),
                    "active_velocity_indices": list(range(int(robot.nv))),
                }
            }
        )
    )
    args = NS(gpu_wbc_manifest=manifest, gpu_wbc_cache_dir=tmp_path / "cache")
    gpu = module.GpuG1RetargetingIK(args, robot, source, frames, q)
    targets = {
        key: np.asarray(robot.get_frame_pose(frame).homogeneous()).copy()
        for key, frame in gpu.frames.items()
    }
    for key in ("right_palm", "left_palm"):
        targets[key][0, 3] += 0.02
    upright = np.asarray(
        robot.get_frame_pose(frames["imu_in_torso"]).homogeneous()
    ).copy()
    com_xy = np.asarray(robot.get_com_position(), dtype=float)[:2]
    com_polygon = com_xy + np.array(
        [[-0.3, -0.3], [0.3, -0.3], [0.3, 0.3], [-0.3, 0.3]]
    )
    for options in (
        dict(upright_target=upright),
        dict(upright_target=upright, posture_weight=0.002),
        dict(upright_target=upright, posture_weight=0.002, collision_enabled=True),
        dict(posture_weight=0.002),
        dict(collision_enabled=True, include_collision_debug=True),
        dict(com_enabled=True, com_support_polygon=com_polygon),
    ):
        step = gpu.solve(q, targets, **options)
        assert step.accepted, step.status
        assert np.linalg.norm(step.joints - q) > 1e-7
        assert np.isfinite(step.joints).all()
        assert max(step.gpu_result.position_errors[:2]) < 0.02
        if options.get("collision_enabled"):
            assert gpu.solver.collision_supported
            assert step.gpu_result.minimum_collision_distance_m is not None
        if options.get("com_enabled"):
            assert step.gpu_result.com_constraint_enabled
            assert step.gpu_result.com_constraint_feasible
            assert step.gpu_result.minimum_com_slack_m > 0
        print(
            f"G1 CUDA {options.keys()}: {step.status}, {step.gpu_result.elapsed_ms:.2f} ms"
        )
