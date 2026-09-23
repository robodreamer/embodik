"""Headless regressions for Example 05's GPU wiring (no CUDA or viewer required)."""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def example():
    # Load the pure control helpers without importing Viser or downloading robot assets.
    path = Path(__file__).parents[1] / "05_dual_arm_ects.py"
    tree = ast.parse(path.read_text())
    names = {
        "_dual_iiwa_collision_exclusions",
        "_gpu_feature_options",
        "_gpu_feature_step",
        "_gpu_configure_inverse",
        "_gpu_semantic_holds",
        "_collision_debug_for_backend",
    }
    namespace = dict(
        np=np,
        COLLISION_MIN_DISTANCE=0.05,
        DEFAULT_NULLSPACE_GAIN_EXP=-2.0,
        derive_frames_active_joint_names=lambda robot, frames: robot.active_names,
    )
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("count", [3, 9, 17])
def test_model_derived_pairs_and_posture_indices(example, count):
    adjacent = ("iiwa_left_iiwa_link_2_0", "iiwa_left_iiwa_link_4_0")
    bases = ("iiwa_left_iiwa_link_0_0", "iiwa_right_iiwa_link_1_0")
    cross = ("iiwa_left_iiwa_link_4_0", "iiwa_right_iiwa_link_4_0")
    distant = ("iiwa_right_iiwa_link_1_0", "iiwa_right_iiwa_link_6_0")
    names = tuple(f"joint_{i}" for i in reversed(range(count)))
    config_indices = dict(zip(names, range(1, count + 1)))
    velocity_indices = dict(zip(names, range(2, 2 * count + 1, 2)))
    robot = SimpleNamespace(
        active_names=names,
        get_collision_pair_names=lambda: [adjacent, bases, cross, distant],
        get_joint_config_index=config_indices.__getitem__,
        get_joint_velocity_index=velocity_indices.__getitem__,
    )
    q = np.arange(count + 2, dtype=float) / 10
    options = example._gpu_feature_options(robot, ("left", "right"), q)
    assert options["collision_pairs"] == (cross, distant)
    assert options["active_joint_names"] == names
    assert options["posture_velocity_indices"] == tuple(velocity_indices.values())
    np.testing.assert_array_equal(
        options["posture_target_configuration"], q[1 : count + 1]
    )
    assert len(options["posture_weights"]) == count
    assert options["solver_backend"] == "torch_srinv"


@pytest.mark.parametrize(
    "enabled,show", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("objective", ["None", "Bias Pose"])
def test_runtime_features_and_lazy_debug(example, enabled, show, objective):
    gpu = Mock()
    gpu._solver.config = InverseConfig()
    gpu.extract_active_configuration.side_effect = lambda q: np.asarray(q)[[4, 1, 3]]
    targets = (object(), object())
    q = np.arange(6, dtype=float)
    result = example._gpu_feature_step(
        gpu,
        q,
        targets,
        bias_configuration=q + 1,
        secondary_objective=objective,
        secondary_weight=0.3,
        collision_enabled=enabled,
        collision_min_distance_m=0.12,
        adaptive_dt=True,
        dt=0.02,
        iterations=3,
        position_gain=7.0,
        orientation_gain=5.0,
        show_collision_debug=show,
        damping=0.25,
        tolerance=0.4,
    )
    options = gpu.configure_runtime.call_args.kwargs
    assert gpu._solver.config.srinv_damping == 0.25
    assert gpu._solver.config.srinv_tolerance == 0.4
    assert options["collision_enabled"] is enabled
    assert options["collision_min_distance_m"] == 0.12
    assert "collision_query_distance_m" not in options
    assert options["adaptive_dt"] is True
    assert options["dt"] == 0.02 and options["iterations"] == 3
    assert options["frame_position_gains"] == (7.0, 7.0)
    assert options["frame_orientation_gains"] == (5.0, 5.0)
    assert options["posture_target_configuration"] == (5.0, 2.0, 4.0)
    assert options["posture_weights"] == (
        (0.3,) * 3 if objective == "Bias Pose" else (0.0,) * 3
    )
    np.testing.assert_array_equal(gpu.solve_step.call_args.args[0], q[[4, 1, 3]])
    assert gpu.solve_step.call_args.args[1] == targets
    assert gpu.solve_step.call_args.kwargs["include_collision_debug"] is (
        enabled and show
    )
    assert result is gpu.solve_step.return_value


def test_debug_never_queries_cpu_in_gpu_mode(example):
    cpu = Mock()
    debug = object()
    gpu_result = SimpleNamespace(collision_debug=debug)
    for enabled, show in [(False, True), (True, False), (False, False)]:
        assert (
            example._collision_debug_for_backend(
                cpu,
                gpu_result,
                gpu_active=True,
                enabled=enabled,
                show=show,
            )
            is None
        )
    assert (
        example._collision_debug_for_backend(
            cpu,
            gpu_result,
            gpu_active=True,
            enabled=True,
            show=True,
        )
        is debug
    )
    assert (
        example._collision_debug_for_backend(
            cpu,
            None,
            gpu_active=True,
            enabled=True,
            show=True,
        )
        is None
    )
    cpu.get_last_collision_debug.assert_not_called()
    assert (
        example._collision_debug_for_backend(
            cpu,
            gpu_result,
            gpu_active=False,
            enabled=True,
            show=True,
        )
        is cpu.get_last_collision_debug.return_value
    )


def test_unsupported_secondary_objective_does_not_solve(example):
    gpu = Mock()
    with pytest.raises(ValueError, match="None or Bias Pose"):
        example._gpu_feature_step(
            gpu,
            [0],
            (),
            bias_configuration=[0],
            secondary_objective="Metric Optimization",
            secondary_weight=1,
            collision_enabled=True,
            collision_min_distance_m=0.05,
            adaptive_dt=False,
            dt=0.01,
            iterations=1,
            position_gain=1,
            orientation_gain=1,
            show_collision_debug=False,
        )
    gpu.configure_runtime.assert_not_called()
    gpu.solve_step.assert_not_called()


@dataclass(frozen=True)
class InverseConfig:
    velocity_solver: str = "torch_srinv"
    srinv_damping: float = 0.1
    srinv_tolerance: float = 0.1
    dt: float = 0.02


@pytest.mark.parametrize(
    "damping,tolerance", [(0, 0.1), (-1, 0.1), (0.1, np.nan), (np.inf, 1)]
)
def test_invalid_inverse_settings_leave_core_unchanged(example, damping, tolerance):
    config = InverseConfig()
    gpu = SimpleNamespace(_solver=SimpleNamespace(config=config))
    with pytest.raises(ValueError, match="positive and finite"):
        example._gpu_configure_inverse(gpu, damping, tolerance)
    assert gpu._solver.config is config


def test_inverse_settings_preserve_other_config_and_reject_other_backend(example):
    gpu = SimpleNamespace(_solver=SimpleNamespace(config=InverseConfig()))
    example._gpu_configure_inverse(gpu, 0.3, 0.7)
    config = gpu._solver.config
    assert config.dt == 0.02
    example._gpu_configure_inverse(gpu, 0.3, 0.7)
    assert gpu._solver.config is config
    gpu._solver.config = InverseConfig(velocity_solver="fi_pesns")
    with pytest.raises(ValueError, match="require torch_srinv"):
        example._gpu_configure_inverse(gpu, 0.3, 0.7)


@pytest.mark.parametrize(
    "change,missing",
    [
        ({"orthogonal": False}, "absolute and relative task Jacobians"),
        ({"constraint_mode": "relative_bounds"}, "inequality rows"),
        ({"objective": "Metric Optimization"}, "manipulability-gradient"),
        ({"solve_mode": "SCALE_ELASTIC"}, "elastic task-scaling"),
        ({"solve_mode": "MIN_ERROR"}, "constrained weighted residual"),
        ({"fallback": True}, "retry and candidate selection"),
        ({"axes": (False,) + (True,) * 11}, "Jacobian row masks"),
    ],
)
def test_semantic_holds_identify_missing_primitive(example, change, missing):
    options = dict(
        orthogonal=True,
        constraint_mode="disable",
        objective="Bias Pose",
        solve_mode="SCALE",
        fallback=False,
        axes=(True,) * 12,
    )
    assert example._gpu_semantic_holds(**options) == []
    holds = example._gpu_semantic_holds(**(options | change))
    assert len(holds) == 1 and missing in holds[0]
