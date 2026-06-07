from __future__ import annotations

import ast
from pathlib import Path

from examples.example_helpers.common_bimanual_teleop_app import (
    DEFAULT_ADAPTIVE_DT_MAX_SCALE,
    DEFAULT_MAX_ANGULAR_SPEED,
    DEFAULT_MAX_LINEAR_SPEED,
    MIN_ERROR_FALLBACK_MAX_POSITION_ERROR_M,
    _near_enough_for_min_error_fallback,
)
from examples.example_helpers.ik_common import configure_solver_runtime_policy

REPO_ROOT = Path(__file__).resolve().parents[1]

POLICY_IMPORT_MODULES = {
    "embodik.interactive_ik",
}

PYTHON_POLICY_NAMES = {
    "ConstraintBoundary",
    "ConstraintDecision",
    "ConstrainedStepGuard",
    "clear_all_target_velocities_if_available",
    "clip_configuration",
    "configure_primary_solve_mode",
    "is_collision_boundary_stall",
    "is_com_boundary_stall",
    "joint_velocity_norm",
    "robust_solve_position_step",
    "solver_status_name",
}

ALLOWED_POLICY_USERS: set[Path] = set()


def _example_python_files() -> list[Path]:
    return sorted((REPO_ROOT / "examples").rglob("*.py"))


def _relative(path: Path) -> Path:
    return path.relative_to(REPO_ROOT)


def _imports_python_constraint_policy(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in POLICY_IMPORT_MODULES:
            imported_names = {alias.name for alias in node.names}
            if imported_names & PYTHON_POLICY_NAMES:
                return True
    return False


def _calls_python_constraint_policy(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in PYTHON_POLICY_NAMES:
            return True
        if isinstance(func, ast.Attribute) and func.attr in PYTHON_POLICY_NAMES:
            return True
    return False


def test_constraint_policy_helper_allowlist_is_current() -> None:
    missing = [path for path in ALLOWED_POLICY_USERS if not (REPO_ROOT / path).exists()]
    assert missing == []

    current_users = {
        _relative(path)
        for path in _example_python_files()
        if _imports_python_constraint_policy(path) or _calls_python_constraint_policy(path)
    }

    assert current_users == ALLOWED_POLICY_USERS


def test_new_examples_do_not_add_python_side_constraint_policy() -> None:
    unexpected_users = [
        _relative(path)
        for path in _example_python_files()
        if _relative(path) not in ALLOWED_POLICY_USERS
        and (_imports_python_constraint_policy(path) or _calls_python_constraint_policy(path))
    ]

    assert unexpected_users == []


def test_example_runtime_policy_enables_auto_switch_and_weighted_fallback() -> None:
    class _Solver:
        def __init__(self) -> None:
            import embodik

            self.config = embodik.SolverRuntimeConfig()
            self.applied = None

        def runtime_config(self):
            return self.config

        def configure_runtime(self, config) -> None:
            self.applied = config

    solver = _Solver()
    solver.config.enable_auto_task_layout = False
    solver.config.weighted_fallback_enabled = False

    configure_solver_runtime_policy(solver)

    assert solver.applied is solver.config
    assert solver.config.enable_auto_task_layout is True
    assert solver.config.weighted_fallback_enabled is True
    assert solver.config.health_sampling.enabled is True
    assert solver.config.health_sampling.sample_count == 8
    assert solver.config.health_sampling.gain == 0.025
    assert solver.config.health_sampling.best_config_cache_enabled is True
    assert solver.config.health_sampling.min_score_improvement == 1e-4
    assert solver.config.health_sampling.activation_joint_limit_cost == 50.0
    assert solver.config.health_sampling.activation_singularity_threshold == -1.0
    assert solver.config.health_sampling.singularity_normalization_scale == 1e-6


def test_common_bimanual_min_error_recovery_is_near_target_and_bounded() -> None:
    assert 1.0 <= DEFAULT_ADAPTIVE_DT_MAX_SCALE <= 5.0
    assert 0.0 < DEFAULT_MAX_LINEAR_SPEED <= 1.0
    assert 0.0 < DEFAULT_MAX_ANGULAR_SPEED <= 2.0
    assert _near_enough_for_min_error_fallback(MIN_ERROR_FALLBACK_MAX_POSITION_ERROR_M)
    assert not _near_enough_for_min_error_fallback(MIN_ERROR_FALLBACK_MAX_POSITION_ERROR_M + 1e-3)
    assert not _near_enough_for_min_error_fallback(float("inf"))

    source = (REPO_ROOT / "examples/example_helpers/common_bimanual_teleop_app.py").read_text(
        encoding="utf-8"
    )
    assert "_apply_position_step_speed_caps(" in source
    assert "max_linear_speed=float(max_linear_speed.value)" in source
    assert "max_angular_speed=float(max_angular_speed.value)" in source
    assert "and near_enough_for_min_error" in source
    assert "and active_mode != embodik.TaskSolveMode.MIN_ERROR" in source


def test_examples_that_construct_solvers_apply_runtime_policy() -> None:
    checked_roots = [REPO_ROOT / "examples", REPO_ROOT / "examples/example_helpers"]
    users = sorted(
        path.relative_to(REPO_ROOT)
        for root in checked_roots
        for path in root.glob("*.py")
        if "KinematicsSolver(" in path.read_text(encoding="utf-8")
    )

    missing = [
        path
        for path in users
        if "configure_solver_runtime_policy(" not in (REPO_ROOT / path).read_text(encoding="utf-8")
    ]

    assert missing == []


def test_solver_owned_backtrack_handles_all_no_motion_failure_statuses() -> None:
    source = (REPO_ROOT / "cpp_core/src/kinematics_solver.cpp").read_text(encoding="utf-8")
    predicate_start = source.index("const bool allow_backtrack =")
    predicate_end = source.index("if (allow_backtrack)", predicate_start)
    predicate = source[predicate_start:predicate_end]

    assert "SolverStatus::kInfeasible" in predicate
    assert "SolverStatus::kNumericalError" in predicate
    assert "SolverStatus::kNoProgress" in predicate
    assert "SolverStatus::kInvalidInput" not in predicate
    assert "vel_out.joint_velocities.norm() <= stall_config_.dq_stall_eps" in predicate
