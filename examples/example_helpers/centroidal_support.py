"""Shared centroidal controls and diagnostics for interactive examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

LINEAR_XY_MOMENTUM_MASK = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])


@dataclass(frozen=True)
class CentroidalDiagnostics:
    """Accepted command diagnostics in the configured support frame."""

    capture_point: np.ndarray | None
    zmp: np.ndarray | None
    capture_point_min_slack: float | None
    zmp_min_slack: float | None
    force_z: float | None


def configure_horizontal_momentum_damping(
    task: Any,
    *,
    enabled: bool,
    weight: float,
    priority: int,
) -> None:
    """Configure an opt-in zero-horizontal-momentum regularizer."""
    task.priority = int(priority)
    task.weight = max(float(weight), 0.0) if enabled else 0.0
    task.set_target_momentum(np.zeros(6, dtype=float))
    task.set_axis_mask(LINEAR_XY_MOMENTUM_MASK)
    if hasattr(task, "active"):
        task.active = bool(enabled)


def configure_capture_point_constraint(
    solver: Any,
    *,
    enabled: bool,
    support_polygon: np.ndarray,
    margin: float,
    frame_name: str,
) -> None:
    """Enable or clear the position-step-compatible capture-point constraint."""
    if enabled:
        solver.configure_capture_point_constraint(
            np.asarray(support_polygon, dtype=float),
            margin=float(margin),
            frame_name=str(frame_name),
        )
    else:
        solver.clear_capture_point_constraint()


def configure_velocity_zmp_constraint(
    solver: Any,
    *,
    enabled: bool,
    support_polygon: np.ndarray,
    margin: float,
    frame_name: str,
    fz_min: float = 1.0,
) -> None:
    """Enable or clear explicit-state velocity-ZMP enforcement."""
    if enabled:
        solver.configure_velocity_zmp_constraint(
            np.asarray(support_polygon, dtype=float),
            margin=float(margin),
            frame_name=str(frame_name),
            fz_min=float(fz_min),
        )
    else:
        solver.clear_velocity_zmp_constraint()


def configure_centroidal_diagnostic_solver(
    solver: Any,
    *,
    support_polygon: np.ndarray,
    margin: float,
    frame_name: str,
    dt: float,
    fz_min: float = 1.0,
) -> None:
    """Configure a read-only observer for explicit-state CP and ZMP evidence."""
    polygon = np.asarray(support_polygon, dtype=float)
    solver.dt = float(dt)
    solver.configure_capture_point_constraint(
        polygon,
        margin=float(margin),
        frame_name=str(frame_name),
    )
    solver.configure_velocity_zmp_constraint(
        polygon,
        margin=float(margin),
        frame_name=str(frame_name),
        fz_min=float(fz_min),
    )


def _successful_debug(debug: dict[str, Any]) -> bool:
    status = debug.get("status")
    return getattr(status, "name", str(status)) == "SUCCESS"


def evaluate_centroidal_diagnostics(
    solver: Any,
    q: np.ndarray,
    current_dq: np.ndarray,
    dq_command: np.ndarray,
) -> CentroidalDiagnostics:
    """Evaluate CP and finite-difference ZMP from caller-owned velocity state."""
    q_array = np.asarray(q, dtype=float)
    current = np.asarray(current_dq, dtype=float)
    command = np.asarray(dq_command, dtype=float)
    capture_point = solver.evaluate_capture_point_constraint(q_array, command)
    zmp = solver.evaluate_velocity_zmp_constraint(q_array, current, command)

    cp_ok = _successful_debug(capture_point)
    zmp_ok = _successful_debug(zmp)
    cp_slacks = np.asarray(capture_point.get("slacks", []), dtype=float)
    zmp_slacks = np.asarray(zmp.get("slacks", []), dtype=float)
    return CentroidalDiagnostics(
        capture_point=(np.asarray(capture_point["point"], dtype=float).copy() if cp_ok else None),
        zmp=np.asarray(zmp["point"], dtype=float).copy() if zmp_ok else None,
        capture_point_min_slack=(float(np.min(cp_slacks)) if cp_ok and cp_slacks.size else None),
        zmp_min_slack=(float(np.min(zmp_slacks)) if zmp_ok and zmp_slacks.size else None),
        force_z=float(zmp["force_z"]) if zmp_ok else None,
    )
