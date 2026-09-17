"""Synchronous locomanip GPU ownership, without CUDA or MuJoCo dependencies."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from examples.example_helpers import gpu_spot_locomanip as gpu


@pytest.fixture
def setup(monkeypatch):
    seed = np.arange(15, dtype=float) / 100
    pose = NS(rotation=np.eye(3), translation=np.zeros(3))
    config_indices = {"arm_a": 10, "arm_b": 12, "leg": 8, "gripper": 14}
    velocity_indices = {"arm_a": 9, "arm_b": 11, "leg": 7, "gripper": 13}
    state = NS(q=seed.copy(), fail=False, result=None)
    robot = NS(
        nq=len(seed),
        nv=len(seed) - 1,
        get_joint_config_index=lambda name: config_indices[name],
        get_joint_config_size=lambda name: 1,
        get_joint_velocity_index=lambda name: velocity_indices[name],
        get_joint_velocity_size=lambda name: 1,
        update_configuration=lambda q: setattr(state, "q", np.array(q).copy()),
        get_frame_pose=lambda name: pose,
    )

    def initialize(self, urdf, dt):
        self.enabled, self.robot = True, robot
        self.message = "ready"
        self._eik = NS(r2q=lambda *args, **kwargs: np.array([0, 0, 0, 1]))
        self._arm_joint_map = dict(
            zip(gpu.MJCF_ARM_JOINT_NAMES[:2], ("arm_a", "arm_b"))
        )
        self._leg_joint_map = {"leg": "leg"}
        self._gripper_joint_name = "gripper"
        self._arm_velocity_indices = [9, 11]
        self.tool_frame, self.body_frame = "tool", "body"
        self._collision_include_pairs = [("arm", "body")]
        self.collision_min_distance = 0.02
        self.collision_max_constraints = 3
        self.collision_tuning_mode = "balanced"
        self.dt, self.max_steps = dt, 1
        self.position_gain, self.orientation_gain = 60.0, 60.0
        self.adaptive_dt = True
        self.adaptive_dt_max_scale = 3.0
        self.adaptive_dt_reference_distance = 0.04
        self._lower = np.full(len(seed), -1.0)
        self._upper = np.full(len(seed), 1.0)
        self._sync_configuration = lambda *a, **k: seed.copy()

    monkeypatch.setattr(gpu.OptionalSpotWholeBodyIK, "__init__", initialize)
    monkeypatch.setattr(gpu, "_select_existing_frame", lambda robot, names: names[0])
    instances = []

    class Solver:
        def __init__(self, *args, **kwargs):
            self.options, self.runtime, self.debug, self.inputs, self.resets = (
                kwargs,
                [],
                [],
                [],
                0,
            )
            instances.append(self)

        def configure_runtime(self, **kwargs):
            self.runtime.append(kwargs)

        def reset_state(self):
            self.resets += 1

        def solve_step(self, q, targets, *, include_collision_debug):
            self.targets = targets
            self.debug.append(include_collision_debug)
            self.inputs.append(np.asarray(q).copy())
            if state.fail:
                raise RuntimeError("CUDA failure")
            if state.result:
                return state.result
            candidate = q.copy()
            candidate[10] += 0.01
            return NS(
                joints=candidate,
                status="SAFE_STEP",
                position_errors=(0.01,),
                rotation_errors=(0.02,),
                collision_overflow=False,
                collision_step_accepted=True,
                torso_constraint_feasible=True,
                collision_debug="debug" if include_collision_debug else None,
            )

    args = NS(
        gpu_wbc_manifest=Path("manifest.json"),
        gpu_wbc_cache_dir=Path("cache"),
        gpu_wbc_collision=True,
    )
    ik = gpu.GpuSpotLocomanipIK(Path("spot.urdf"), args=args, solver_factory=Solver)
    obs = dict(
        gpu_fixed_contacts=np.ones(len(ik.foot_frames)),
        gpu_velocity_command=np.zeros(3),
        base_lin_vel=np.zeros(3),
        base_ang_vel=np.zeros(3),
        gpu_objects_enabled=np.zeros(1),
        arm_state=np.zeros(4),
        gripper_state=np.zeros(2),
        base_pose=np.array([0, 0, 0, 1, 0, 0, 0]),
        joint_pos=np.zeros(len(ik._leg_joint_map)),
    )
    return ik, obs, pose, instances, state, seed


def solve(setup):
    ik, obs, target, *_ = setup
    return ik.solve_command(
        obs,
        np.zeros(3),
        target_pose=target,
        body_command=np.zeros(3),
        desired_pose_command=np.zeros(3),
    )


def test_generalized_layout_and_stationary_actuation(setup):
    ik, _, _, instances, state, seed = setup
    result = solve(setup)
    assert result.success
    assert result.body_command is None and result.desired_pose_command is None
    assert result.arm_command[0] == pytest.approx(seed[10] + 0.01)
    options = instances[0].options
    assert options["active_velocity_indices"] == tuple(range(14 - 1))
    assert options["locked_velocity_indices"] == tuple(
        i for i in range(13) if i not in (9, 11)
    )
    assert options["posture_velocity_indices"] == (9, 11)
    assert len(options["posture_weights"]) == 2
    assert options["frame_contact_constraints"] == (False, True, True, True, True)
    assert options["secondary_frame_names"] == ("body",)
    assert options["torso_frame_name"] == "body"
    assert options["collision_pairs"] == (("arm", "body"),)
    assert options["adaptive_dt"]
    np.testing.assert_array_equal(state.q, seed)


def test_runtime_constraints_and_lazy_debug(setup):
    ik, _, _, instances, *_ = setup
    assert solve(setup).success
    assert ik.collision_debug_rows() == []
    ik.enable_collision = False
    ik.collision_min_distance = 0.035
    ik.adaptive_dt = False
    ik.include_collision_debug = True
    assert solve(setup).success
    assert instances[0].runtime[-1]["collision_enabled"] is False
    assert instances[0].runtime[-1]["collision_min_distance_m"] == 0.035
    assert instances[0].runtime[-1]["adaptive_dt"] is False
    assert instances[0].debug == [False, True]
    ik.reset_reference()
    assert ik.collision_debug_rows() == []
    assert instances[0].resets == 1


@pytest.mark.parametrize(
    "key,value",
    [
        ("gpu_fixed_contacts", [1, float("nan"), 1, 1]),
        ("gpu_fixed_contacts", []),
        ("gpu_fixed_contacts", [1, 0.5, 1, 1]),
        ("gpu_velocity_command", [float("inf"), 0, 0]),
        ("base_ang_vel", [float("nan"), 0, 0]),
        ("gpu_objects_enabled", [1]),
        ("gpu_objects_enabled", []),
        ("joint_pos", []),
        ("base_pose", []),
    ],
)
def test_unsupported_dynamics_hold_before_gpu_allocation(setup, key, value):
    setup[1][key] = np.array(value)
    result = solve(setup)
    assert not result.success and result.arm_command is None
    assert not setup[3]


def test_finite_policy_balance_velocity_does_not_block_stationary_arm_ik(setup):
    setup[1]["base_lin_vel"] = np.array([0.12, 0.0, 0.0])
    setup[1]["base_ang_vel"] = np.array([0.0, 0.08, 0.0])
    assert solve(setup).success


def test_cuda_failure_never_uses_cpu(setup):
    setup[4].fail = True
    result = solve(setup)
    assert not result.success and "CUDA failure" in result.message
    assert result.arm_command is None


def test_seeds_collision_from_measured_arm(setup):
    ik, obs, *_ = setup
    obs["arm_state"] = np.array([0.2, 0.3, 0, 0])
    captured = []
    ik._sync_configuration = lambda observation, arm, **kw: (
        captured.append(arm.copy()) or setup[5].copy()
    )
    assert solve(setup).success
    np.testing.assert_allclose(captured[0], [0.2, 0.3, 0])


def test_arm_seed_is_clamped_but_physics_owned_state_is_restored(setup):
    ik, _, _, instances, state, seed = setup
    measured = seed.copy()
    measured[10] = 1.2
    ik._sync_configuration = lambda *a, **k: measured.copy()

    assert solve(setup).success
    assert instances[0].inputs[0][10] == pytest.approx(1.0)
    np.testing.assert_array_equal(state.q, measured)


@pytest.mark.parametrize(
    "flag",
    ["collision_overflow", "collision_step_accepted", "torso_constraint_feasible"],
)
def test_constraint_failure_never_emits_command(setup, flag):
    result = NS(
        joints=setup[5].copy(),
        status="SAFE_STEP",
        collision_overflow=False,
        collision_step_accepted=True,
        torso_constraint_feasible=True,
    )
    setattr(result, flag, flag == "collision_overflow")
    setup[4].result = result
    assert not solve(setup).success


def test_rejects_virtual_base_displacement(setup):
    candidate = setup[5].copy()
    candidate[0] += 0.01
    setup[4].result = NS(
        joints=candidate,
        status="SAFE_STEP",
        collision_overflow=False,
        collision_step_accepted=True,
        torso_constraint_feasible=True,
    )
    result = solve(setup)
    assert not result.success and "physics-owned" in result.message


def test_runtime_capacity_change_is_explicit_hold(setup):
    assert solve(setup).success
    setup[0].collision_max_constraints += 1
    assert "restart required" in solve(setup).message


def test_torso_command_change_remains_policy_owned(setup):
    assert solve(setup).success
    ik, obs, pose, *_ = setup
    result = ik.solve_command(
        obs,
        np.zeros(3),
        target_pose=pose,
        body_command=np.ones(3),
        desired_pose_command=np.zeros(3),
    )
    assert result.success
    assert result.body_command is None and result.desired_pose_command is None


def test_walking_contact_transitions_reanchor_measured_state(setup):
    ik, obs, _, instances, state, seed = setup
    assert solve(setup).success
    initial_posture = instances[0].options["posture_target_configuration"]
    for contacts in ([1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]):
        measured = seed.copy()
        measured[0] += 0.2
        measured[8] -= 0.1
        seed = measured
        ik._sync_configuration = lambda *a, **k: measured.copy()
        pose = NS(rotation=np.eye(3), translation=measured[:3].copy())
        ik.robot.get_frame_pose = lambda name: pose
        obs["gpu_fixed_contacts"] = np.array(contacts)
        obs["gpu_velocity_command"] = np.array([0.5, 0.1, 0.3])
        obs["base_lin_vel"] = np.array([0.4, 0.1, 0])
        obs["base_ang_vel"] = np.array([0, 0, 0.3])
        result = solve(setup)
        assert result.success
        assert result.body_command is None and result.desired_pose_command is None
        solver = instances[0]
        np.testing.assert_array_equal(solver.inputs[-1], measured)
        for foot in solver.targets[1:]:
            np.testing.assert_array_equal(foot.translation, measured[:3])
        runtime = solver.runtime[-1]
        np.testing.assert_array_equal(
            runtime["torso_reference_pose_xyzw"][:3], measured[:3]
        )
        np.testing.assert_array_equal(
            runtime["secondary_frame_target_poses_wxyz"][0][:3], measured[:3]
        )
        assert runtime["posture_target_configuration"] == initial_posture
        assert runtime["collision_enabled"]
        np.testing.assert_array_equal(state.q, measured)


@pytest.mark.parametrize("index", [0, 8, 14])
def test_rejects_non_arm_motion_during_walking(setup, index):
    setup[1]["gpu_velocity_command"] = np.array([0.5, 0, 0])
    candidate = setup[5].copy()
    candidate[index] += 1e-8
    setup[4].result = NS(
        joints=candidate,
        status="SAFE_STEP",
        collision_overflow=False,
        collision_step_accepted=True,
        torso_constraint_feasible=True,
    )
    result = solve(setup)
    assert not result.success and "physics-owned" in result.message


def test_com_and_unsupported_policy_hold(setup):
    setup[0].enable_com = True
    assert "CoM" in solve(setup).message
    setup[0].enable_com = False
    setup[0].target_solve_mode = "MIN_ERROR"
    assert "policy" in solve(setup).message


def test_accepts_only_fp32_roundtrip_for_locked_coordinates(setup):
    candidate = setup[5].astype(np.float32).astype(float)
    setup[4].result = NS(
        joints=candidate,
        status="SAFE_STEP",
        collision_overflow=False,
        collision_step_accepted=True,
        torso_constraint_feasible=True,
        position_errors=(0.01,),
        rotation_errors=(0.02,),
        collision_debug=None,
    )
    assert solve(setup).success
    candidate[8] = float(np.nextafter(np.float32(candidate[8]), np.float32(np.inf)))
    assert "physics-owned" in solve(setup).message


@pytest.mark.parametrize(
    "key",
    [
        "base_pose",
        "joint_pos",
        "arm_state",
        "gripper_state",
        "base_lin_vel",
        "gpu_fixed_contacts",
        "gpu_objects_enabled",
    ],
)
def test_missing_observation_holds(setup, key):
    del setup[1][key]
    assert not solve(setup).success
    assert not setup[3]


def test_contact_attestation_requires_ground_contact():
    model = NS(site_bodyid=np.arange(1, 5), geom_bodyid=np.arange(5))
    adapter = NS(
        model=model,
        _mujoco=NS(
            mjtObj=NS(mjOBJ_SITE=1),
            mj_name2id=lambda m, kind, name: ("FL", "FR", "HL", "HR").index(name),
        ),
    )
    data = NS(ncon=4, contact=[NS(dist=0, geom1=i, geom2=0) for i in range(1, 5)])
    assert np.all(
        gpu.stationary_contact_state(adapter, data, np.zeros(3))["gpu_fixed_contacts"]
    )
    data.contact[0].geom2 = 2
    assert not gpu.stationary_contact_state(adapter, data, np.zeros(3))[
        "gpu_fixed_contacts"
    ][0]
    data.contact[0].dist = float("nan")
    assert np.isnan(
        gpu.stationary_contact_state(adapter, data, np.zeros(3))["gpu_fixed_contacts"]
    ).all()


def test_contact_attestation_accepts_menagerie_named_foot_geometries():
    model = NS(site_bodyid=np.array([], dtype=int), geom_bodyid=np.arange(5))
    names = ("FL", "FR", "HL", "HR")

    def name_to_id(model, kind, name):
        return -1 if kind == 1 else names.index(name) + 1

    adapter = NS(
        model=model,
        _mujoco=NS(
            mjtObj=NS(mjOBJ_SITE=1, mjOBJ_GEOM=2),
            mj_name2id=name_to_id,
        ),
    )
    data = NS(ncon=4, contact=[NS(dist=0, geom1=0, geom2=i) for i in range(1, 5)])
    assert np.all(
        gpu.stationary_contact_state(adapter, data, np.zeros(3))["gpu_fixed_contacts"]
    )


def test_headless_requests_gpu_ik(monkeypatch, capsys):
    path = Path(__file__).parents[1] / "09_spot_locomanip_mjviser.py"
    spec = importlib.util.spec_from_file_location("spot09_gpu_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    steps = []
    pose = NS(translation=np.zeros(3), rotation=np.eye(3))
    adapter = NS(
        set_manipulation_objects_enabled=lambda *a: None,
        read_policy_observation=lambda data: {},
    )
    controller = NS(
        adapter=adapter,
        commands=NS(arm=np.zeros(7), body=np.zeros(3), desired_pose=np.zeros(3)),
        policy=NS(value="test"),
        checkpoint=Path("test.onnx"),
        last_ik_status="GPU SAFE_STEP: test",
        ik=NS(_eik=NS(r2q=lambda *a, **kw: np.array([0, 0, 0, 1]))),
        reset=lambda *a: None,
        command_tool_pose=lambda *a, **kw: pose,
        step=lambda *a, **kw: steps.append(kw),
    )

    def factory(model, **kwargs):
        assert kwargs["gpu_args"].gpu_wbc
        return controller

    monkeypatch.setattr(module, "SpotLocomanipController", factory)
    monkeypatch.setattr(module, "load_spot_mujoco_model", lambda *a, **kw: object())
    monkeypatch.setattr(
        module, "_resolve_optional_spot_urdf", lambda *a: Path("spot.urdf")
    )
    monkeypatch.setattr(module, "resolve_spot_scene_arm_xml", lambda: Path("spot.xml"))
    monkeypatch.setattr(module, "target_pose_from_wxyz", lambda p, q: p)
    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        NS(
            MjData=lambda model: NS(
                qpos=np.zeros(1), qvel=np.zeros(1), ctrl=np.zeros(1)
            ),
            mj_step=lambda *a: None,
        ),
    )
    args = module.build_parser().parse_args(["--headless", "--gpu-wbc", "--steps", "2"])
    module.run_headless(args)
    assert len(steps) == 2 and all(step["ik_enabled"] for step in steps)
    np.testing.assert_allclose(steps[0]["ik_target_pose"], [0.02, 0, 0])
    assert "gpu_accepted_steps=2" in capsys.readouterr().out
