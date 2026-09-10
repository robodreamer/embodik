"""Orchestration regressions independent of CUDA and robot assets."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from examples.example_helpers.gpu_spot_full_body_ik import GpuSpotFullBodyIK
from examples.example_helpers.spot_whole_body_ik import (
    _ARM_TORSO_POSE_AXIS_MASK,
    SpotFullBodyIKConfig,
)
from examples.example_helpers.spot_whole_body_ik import (
    SpotFullBodyIKMode as Mode,
)


@pytest.fixture
def setup():
    seed = np.arange(31, dtype=float) / 100
    state = NS(q=seed.copy())
    indices = {"arm_b": 22, "arm_a": 17, "gripper": 30}
    robot = NS(
        nq=len(seed),
        nv=len(seed) - 1,
        get_joint_config_index=lambda name: indices[name],
        get_joint_config_size=lambda name: 1,
        update_configuration=lambda q: setattr(state, "q", np.array(q).copy()),
    )

    def rt(*, R, t):
        return NS(rotation=np.array(R), translation=np.array(t))

    def homogeneous(pose):
        out = np.eye(4)
        out[:3, :3], out[:3, 3] = pose.rotation, pose.translation
        return out

    bias = np.eye(4)
    bias[:3, 3] = (0.1, 0.2, 0.3)
    bias[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    b = NS(
        q=seed.copy(),
        q0=seed.copy(),
        robot=robot,
        eik=NS(Rt=rt, r2q=lambda R, order: np.array([0, 0, 0, 1])),
        config=SpotFullBodyIKConfig(torso_pose_half_range=np.full(6, 0.1)),
        tool_frame="custom_tool",
        body_frame="custom_body",
        foot_frames=("foot_a", "foot_b", "foot_c", "foot_d"),
        _foot_anchor_poses=tuple(np.eye(4) for _ in range(4)),
        _torso_bias_pose=bias,
        _torso_bounds_reference_pose=np.eye(4),
        _posture_bias_q=seed.copy(),
        _arm_joint_map={"b": "arm_b", "a": "arm_a"},
        _gripper_joint_name="gripper",
        _gripper_velocity_indices=[29],
        _arm_velocity_indices=[21, 16],
        _base_velocity_indices=list(range(6)),
        _leg_velocity_indices=[7, 10, 12, 14],
        _stage1_posture_velocity_indices=[0, 1, 2, 21, 16],
        _stage1_posture_weights=[1, 1, 1, 2, 2],
        _posture_velocity_indices=[0, 1, 2, 7, 10, 12, 14, 21, 16],
        _full_posture_weights=[1] * 9,
        _collision_include_pairs=(("a", "b"),),
        _as_homogeneous=homogeneous,
    )
    tool = np.eye(4)
    tool[:3, 3] = (0.4, 0.5, 0.6)
    b._pose_homogeneous = lambda frame: tool.copy()
    instances = []

    class Solver:
        def __init__(self, *args, **kw):
            for name in (
                "secondary_frame_task_dimensions",
                "secondary_frame_target_poses_wxyz",
                "secondary_frame_position_gains",
                "secondary_frame_orientation_gains",
                "secondary_frame_weights",
            ):
                assert len(kw.get(name, ())) == len(kw["secondary_frame_names"])
            self.options, self.calls, self.warms, self.resets = kw, [], [], 0
            self.collision_clear_state_fast_path_enabled = bool(
                kw.get("collision_clear_state_fast_path_enabled", False)
            )
            self._previous_velocity = np.zeros((1, 1), dtype=float)
            self._last_target = None
            self._last_target_host = None
            instances.append(self)

        def warm_up(self, q, targets):
            self.warms.append((np.array(q).copy(), targets))

        def configure_runtime(self, **options):
            assert "position_gain" not in options
            assert "orientation_gain" not in options
            self.options.update(options)

        def reset_state(self):
            self.resets += 1
            self._previous_velocity.fill(0.0)
            self._last_target = None
            self._last_target_host = None

        def solve_step(self, q, targets, **options):
            self.calls.append((np.array(q).copy(), targets, options))
            # Deliberately corrupt every coordinate in stage 1 to catch wholesale copying.
            out = np.array(q).copy() + (
                1 if self.options["frames"] == (b.tool_frame,) else 0
            )
            return NS(
                joints=out,
                status="SAFE_STEP",
                collision_step_accepted=True,
                collision_overflow=False,
            )

    args = NS(
        gpu_wbc_collision=True, gpu_wbc_manifest="manifest", gpu_wbc_cache_dir="cache"
    )
    return (
        GpuSpotFullBodyIK(args, b, "robot.urdf", solver_factory=Solver),
        b,
        instances,
        rt,
        tool,
    )


def test_two_stage_copy_residual_orientation_and_locking(setup):
    gpu, b, solvers, rt, tool = setup
    assert not gpu.device_two_stage_enabled
    assert gpu._arm_configuration_indices == (22, 17)
    seed = b.q.copy()
    target = rt(R=np.eye(3), t=[0.8, 0.7, 0.9])
    result = gpu.solve(Mode.TWO_STAGE, target, include_collision_debug=True)
    assert isinstance(result, tuple) and len(result) == 2
    arm, torso = solvers
    np.testing.assert_array_equal(arm.calls[0][0], seed)
    expected = seed.copy()
    expected[[22, 17]] += 1
    np.testing.assert_array_equal(torso.calls[0][0], expected)
    np.testing.assert_array_equal(b.q, expected)
    target2 = torso.calls[0][1][0]
    np.testing.assert_allclose(
        target2.translation,
        b._torso_bias_pose[:3, 3] + target.translation - tool[:3, 3],
    )
    np.testing.assert_array_equal(target2.rotation, b._torso_bias_pose[:3, :3])
    assert set(arm.options["locked_velocity_indices"]) == set(
        b._base_velocity_indices + b._leg_velocity_indices
    )
    assert set(torso.options["locked_velocity_indices"]) == set(b._arm_velocity_indices)
    assert arm.options["frame_contact_constraints"] == (False,)
    assert torso.options["frame_contact_constraints"] == (False, True, True, True, True)
    assert arm.options["posture_velocity_indices"] == tuple(
        b._stage1_posture_velocity_indices
    )
    assert torso.options["posture_velocity_indices"] == tuple(
        b._posture_velocity_indices
    )
    for solver in solvers:
        assert solver.options["collision_pairs"] == b._collision_include_pairs
        assert solver.options["active_velocity_indices"] == tuple(range(b.robot.nv - 1))
        assert len(solver.warms) == 1
        assert solver.calls[0][2]["include_collision_debug"]


@pytest.mark.parametrize("mode", [Mode.ARM_TORSO, Mode.TORSO_ONLY, Mode.FULL_BODY])
def test_layouts_runtime_updates_and_lazy_cache(setup, mode):
    gpu, b, solvers, rt, _ = setup
    tool_target = rt(R=np.eye(3), t=[1, 2, 3])
    torso_target = rt(R=np.eye(3), t=[4, 5, 6])
    gpu.solve(mode, tool_target, torso_target)
    assert len(solvers) == 1
    s = solvers[0]
    arm = mode == Mode.ARM_TORSO
    torso = mode == Mode.TORSO_ONLY
    assert s.options["frames"] == (
        b.body_frame if torso else b.tool_frame,
        *(() if arm else b.foot_frames),
    )
    assert s.options["secondary_frame_names"] == (() if torso else (b.body_frame,))
    assert set(s.options["locked_velocity_indices"]) == set(
        b._leg_velocity_indices if arm else b._arm_velocity_indices if torso else ()
    )
    assert s.options["torso_axis_mask"] == tuple(
        map(bool, _ARM_TORSO_POSE_AXIS_MASK if arm else np.ones(6))
    )
    assert s.calls[0][1][0] is (torso_target if torso else tool_target)
    b.config.dt = 0.023
    b.config.nullspace_gain = 0.37
    b.config.collision_min_distance = 0.08
    b.config.torso_pose_half_range = np.full(6, 0.12)
    gpu.solve(mode, tool_target, torso_target)
    assert len(solvers) == len(s.warms) == 1
    assert s.options["dt"] == 0.023
    assert s.options["posture_gain"] == 0.37
    assert s.options["collision_min_distance_m"] == 0.08
    assert s.options["torso_upper_relative_limits"] == (0.12,) * 6


def test_stage_failure_restores_robot_seed(setup):
    gpu, b, _, rt, _ = setup
    seed = b.q.copy()
    original = gpu._stage

    def fail_second(key, *args):
        if key == Mode.TORSO_ONLY:
            raise RuntimeError("query failed")
        return original(key, *args)

    gpu._stage = fail_second
    with pytest.raises(RuntimeError, match="query failed"):
        gpu.solve(Mode.TWO_STAGE, rt(R=np.eye(3), t=[1, 2, 3]))
    np.testing.assert_array_equal(b.q, seed)


def test_no_collision_allocation_and_switch_resets(setup):
    gpu, _b, solvers, rt, _ = setup
    gpu.pairs = ()
    gpu.collision_supported = False
    target = rt(R=np.eye(3), t=[1, 2, 3])
    gpu.solve(Mode.FULL_BODY, target)
    gpu.solve(Mode.TORSO_ONLY, target)
    gpu.solve(Mode.FULL_BODY, target)
    assert len(solvers) == 2
    assert solvers[0].resets == 2
    assert solvers[0].options["collision_pairs"] == ()
    assert "collision_min_distance_m" not in solvers[0].options


def test_certified_nominal_uses_persistent_recovery_only_when_needed(setup):
    gpu, b, solvers, rt, _ = setup
    gpu.solver_backend = "cusolver_srinv"
    gpu.certified_nominal_enabled = True
    target = rt(R=np.eye(3), t=[1, 2, 3])
    targets = (target,)
    nominal = gpu._get(Mode.ARM_TORSO, b.q, targets, repair_iterations=0)
    recovery = gpu._get(Mode.ARM_TORSO, b.q, targets, repair_iterations=2)
    assert len(solvers) == 2
    assert nominal.options["collision_clear_state_fast_path_enabled"]
    assert nominal.options["collision_candidate_convex_certificate_enabled"]
    assert nominal.options["collision_current_convex_certificate_enabled"]
    assert nominal.options["cusolver_reuse_locked_primary_inverse_enabled"]
    assert nominal.options["cusolver_specialized_outputs_enabled"]
    assert nominal.options["torso_projection_noop_fast_path_enabled"]
    assert not recovery.options["collision_clear_state_fast_path_enabled"]
    assert not recovery.options["collision_candidate_convex_certificate_enabled"]
    assert not recovery.options["collision_current_convex_certificate_enabled"]
    assert not recovery.options["cusolver_reuse_locked_primary_inverse_enabled"]
    assert recovery.options["cusolver_specialized_outputs_enabled"]
    assert not recovery.options["torso_projection_noop_fast_path_enabled"]
    nominal_result = NS(
        joints=b.q.copy() + 0.1,
        status="SAFE_STEP",
        collision_clear_state_certified=True,
        collision_step_accepted=True,
        collision_overflow=False,
        collision_active=False,
    )
    nominal.solve_step = lambda *args, **kwargs: nominal_result
    recovered_result = NS(
        joints=b.q.copy() + 0.2,
        status="SAFE_STEP",
        collision_step_accepted=True,
        collision_overflow=False,
    )
    recovery_calls = []
    recovery.solve_step = lambda *args, **kwargs: (
        recovery_calls.append(args),
        recovered_result,
    )[1]

    assert gpu._stage(Mode.ARM_TORSO, b.q, target, False) is nominal_result
    assert not recovery_calls

    nominal_result.collision_active = True
    assert gpu._stage(Mode.ARM_TORSO, b.q, target, True) is recovered_result
    assert len(recovery_calls) == 1
    nominal_result.collision_active = False
    nominal_result.status = "SAFE_HOLD_COLLISION"
    nominal_result.collision_step_accepted = False
    assert gpu._stage(Mode.ARM_TORSO, b.q, target, False) is recovered_result
    assert len(recovery_calls) == 2


def test_certified_clear_stage_keeps_exact_recovery_lazy(setup):
    gpu, b, solvers, rt, _ = setup
    gpu.solver_backend = "cusolver_srinv"
    gpu.certified_nominal_enabled = True
    target = rt(R=np.eye(3), t=[1, 2, 3])
    nominal_result = NS(
        joints=b.q.copy() + 0.1,
        status="SAFE_STEP",
        collision_clear_state_certified=True,
        collision_step_accepted=True,
        collision_overflow=False,
        collision_active=False,
    )
    nominal = gpu._get(Mode.ARM_TORSO, b.q, (target,), repair_iterations=0)
    nominal.solve_step = lambda *args, **kwargs: nominal_result

    assert gpu._stage(Mode.ARM_TORSO, b.q, target, False) is nominal_result
    assert len(solvers) == 1
    assert (Mode.ARM_TORSO, gpu.collision_graph_repair_iterations) not in gpu.cache


def test_lazy_exact_recovery_restores_pre_speculation_history(setup):
    gpu, b, solvers, rt, _ = setup
    gpu.solver_backend = "cusolver_srinv"
    gpu.certified_nominal_enabled = True
    target = rt(R=np.eye(3), t=[1, 2, 3])
    nominal = gpu._get(Mode.ARM_TORSO, b.q, (target,), repair_iterations=0)
    nominal._previous_velocity.fill(0.42)
    nominal._last_target = np.array([1.0, 2.0])
    nominal._last_target_host = np.array([3.0, 4.0])
    rejected = NS(
        joints=b.q.copy(),
        status="SAFE_STEP",
        collision_clear_state_certified=False,
        collision_step_accepted=False,
        collision_overflow=True,
        collision_active=False,
    )

    def reject(*args, **kwargs):
        nominal._previous_velocity.fill(9.0)
        nominal._last_target = np.array([9.0, 9.0])
        nominal._last_target_host = np.array([9.0, 9.0])
        return rejected

    nominal.solve_step = reject
    gpu._stage(Mode.ARM_TORSO, b.q, target, False)
    recovery = gpu.cache[(Mode.ARM_TORSO, gpu.collision_graph_repair_iterations)]

    assert len(solvers) == 2
    np.testing.assert_allclose(recovery._previous_velocity, 0.42)
    np.testing.assert_array_equal(recovery._last_target, [1.0, 2.0])
    np.testing.assert_array_equal(recovery._last_target_host, [3.0, 4.0])
