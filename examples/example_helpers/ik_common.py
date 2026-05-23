"""Shared defaults and small helpers for interactive IK examples."""

from __future__ import annotations

import logging

import embodik

DEFAULT_SOLVER_DT = 0.01
DEFAULT_POS_GAIN = 10.0
DEFAULT_ROT_GAIN = 10.0
DEFAULT_NULLSPACE_GAIN = 1e-2
DEFAULT_NULLSPACE_ENABLED = True
DEFAULT_ADAPTIVE_DT = True
DEFAULT_ADAPTIVE_DT_MAX_SCALE = 10.0
DEFAULT_ADAPTIVE_DT_REFERENCE_DISTANCE = 0.02
DEFAULT_VISER_PORT = 8080

DEFAULT_COLLISION_TUNING_MODE = "balanced"
COLLISION_TUNING_OPTIONS = ("speed", "balanced", "precise")
COLLISION_DEBUG_LOG_PERIOD_S = 5.0


def quiet_websocket_handshake_logs() -> None:
    """Hide benign Viser websocket disconnect traces from example output."""

    for logger_name in ("websockets.server", "websockets.asyncio.server"):
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)


def configure_solver_runtime_policy(solver: embodik.KinematicsSolver) -> None:
    """Use the solver-owned robust layout/fallback policy in public examples."""

    if not (
        hasattr(embodik, "SolverRuntimeConfig")
        and hasattr(solver, "runtime_config")
        and hasattr(solver, "configure_runtime")
    ):
        return

    cfg = solver.runtime_config()
    cfg.enable_auto_task_layout = True
    cfg.weighted_fallback_enabled = True
    solver.configure_runtime(cfg)


def apply_collision_tuning_mode(
    solver: embodik.KinematicsSolver,
    mode_label: str,
) -> None:
    """Apply the collision tuning mode used by the collision/teleop examples."""

    label = mode_label.lower()
    if hasattr(solver, "set_collision_tuning_mode") and hasattr(embodik, "CollisionTuningMode"):
        enum_map = {
            "precise": embodik.CollisionTuningMode.PRECISE,
            "balanced": embodik.CollisionTuningMode.BALANCED,
            "speed": embodik.CollisionTuningMode.SPEED,
        }
        solver.set_collision_tuning_mode(enum_map.get(label, embodik.CollisionTuningMode.BALANCED))
        if label != "precise" and hasattr(solver, "enable_sphere_broadphase"):
            solver.enable_sphere_broadphase(True)
        return

    # Backward-compatible fallback for older bindings.
    if label == "precise":
        if hasattr(solver, "enable_collision_pair_cache"):
            solver.enable_collision_pair_cache(False, 1, 0.0, 128)
        if hasattr(solver, "set_collision_refinement_time_budget_us"):
            solver.set_collision_refinement_time_budget_us(0)
    elif label == "balanced":
        if hasattr(solver, "enable_sphere_broadphase"):
            solver.enable_sphere_broadphase(True)
        if hasattr(solver, "enable_collision_pair_cache"):
            solver.enable_collision_pair_cache(True, 20, 0.05, 256)
        if hasattr(solver, "set_collision_refinement_time_budget_us"):
            solver.set_collision_refinement_time_budget_us(0)
    else:
        if hasattr(solver, "enable_sphere_broadphase"):
            solver.enable_sphere_broadphase(True)
        if hasattr(solver, "enable_collision_pair_cache"):
            solver.enable_collision_pair_cache(True, 100, 0.03, 128)
        if hasattr(solver, "set_collision_refinement_time_budget_us"):
            solver.set_collision_refinement_time_budget_us(300)
