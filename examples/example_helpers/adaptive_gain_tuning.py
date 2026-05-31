"""Error-based scaling for teleop position/orientation IK gains.

Pure Python — no Viser or solver imports. Used by headless harnesses and the
shared bimanual teleop app.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdaptiveGainTuningConfig:
    """Reference errors and clamp/smoothing for gain scaling."""

    position_reference_m: float = 0.05
    orientation_reference_rad: float = 0.10
    min_scale: float = 1.0
    max_scale: float = 3.0
    smoothing: float = 0.2


@dataclass(slots=True)
class AdaptiveGainTuningState:
    """Per-session smoothed scale factors (starts at unity)."""

    position_scale: float = 1.0
    orientation_scale: float = 1.0


def clamp_scale(raw_scale: float, config: AdaptiveGainTuningConfig) -> float:
    return float(max(config.min_scale, min(config.max_scale, raw_scale)))


def raw_error_scales(
    position_error_m: float,
    orientation_error_rad: float,
    config: AdaptiveGainTuningConfig,
) -> tuple[float, float]:
    """Map tracking errors to unsmoothed scale factors."""
    pos_ref = max(config.position_reference_m, 1e-9)
    rot_ref = max(config.orientation_reference_rad, 1e-9)
    pos_scale = clamp_scale(float(position_error_m) / pos_ref, config)
    rot_scale = clamp_scale(float(orientation_error_rad) / rot_ref, config)
    return pos_scale, rot_scale


def reset_adaptive_gain_state(state: AdaptiveGainTuningState) -> None:
    state.position_scale = 1.0
    state.orientation_scale = 1.0


def compute_effective_gains(
    *,
    base_position_gain: float,
    base_orientation_gain: float,
    position_error_m: float,
    orientation_error_rad: float,
    config: AdaptiveGainTuningConfig,
    state: AdaptiveGainTuningState,
    enabled: bool,
) -> tuple[float, float, float, float]:
    """Return (effective_pos_gain, effective_rot_gain, pos_scale, rot_scale).

    When ``enabled`` is False, returns base gains and unity scales without
    mutating ``state``.
    """
    if not enabled:
        return (
            float(base_position_gain),
            float(base_orientation_gain),
            1.0,
            1.0,
        )

    pos_raw, rot_raw = raw_error_scales(position_error_m, orientation_error_rad, config)
    alpha = float(max(0.0, min(1.0, config.smoothing)))
    state.position_scale = (1.0 - alpha) * state.position_scale + alpha * pos_raw
    state.orientation_scale = (1.0 - alpha) * state.orientation_scale + alpha * rot_raw

    return (
        float(base_position_gain) * state.position_scale,
        float(base_orientation_gain) * state.orientation_scale,
        state.position_scale,
        state.orientation_scale,
    )
