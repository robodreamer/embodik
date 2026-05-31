"""Unit tests for error-based adaptive IK gain scaling."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import pytest
from example_helpers.adaptive_gain_tuning import (
    AdaptiveGainTuningConfig,
    AdaptiveGainTuningState,
    clamp_scale,
    compute_effective_gains,
    raw_error_scales,
    reset_adaptive_gain_state,
)


def test_clamp_scale_respects_bounds() -> None:
    cfg = AdaptiveGainTuningConfig(min_scale=1.0, max_scale=3.0)
    assert clamp_scale(0.01, cfg) == 1.0
    assert clamp_scale(1.0, cfg) == 1.0
    assert clamp_scale(10.0, cfg) == 3.0


def test_raw_error_scales_reference_errors() -> None:
    cfg = AdaptiveGainTuningConfig(position_reference_m=0.05, orientation_reference_rad=0.10)
    pos, rot = raw_error_scales(0.10, 0.20, cfg)
    assert pos == pytest.approx(2.0)
    assert rot == pytest.approx(2.0)


def test_disabled_returns_base_gains_without_state_mutation() -> None:
    cfg = AdaptiveGainTuningConfig()
    state = AdaptiveGainTuningState(position_scale=2.0, orientation_scale=2.0)
    pos, rot, ps, rs = compute_effective_gains(
        base_position_gain=10.0,
        base_orientation_gain=8.0,
        position_error_m=0.2,
        orientation_error_rad=0.3,
        config=cfg,
        state=state,
        enabled=False,
    )
    assert pos == 10.0
    assert rot == 8.0
    assert ps == 1.0
    assert rs == 1.0
    assert state.position_scale == 2.0


def test_enabled_scales_up_for_large_error() -> None:
    cfg = AdaptiveGainTuningConfig(smoothing=1.0)
    state = AdaptiveGainTuningState()
    pos, rot, _, _ = compute_effective_gains(
        base_position_gain=10.0,
        base_orientation_gain=10.0,
        position_error_m=0.10,
        orientation_error_rad=0.0,
        config=cfg,
        state=state,
        enabled=True,
    )
    assert pos == pytest.approx(20.0)
    assert rot == pytest.approx(10.0)


def test_smoothing_reduces_step_changes() -> None:
    cfg = AdaptiveGainTuningConfig(smoothing=0.5, min_scale=0.5, max_scale=3.0)
    state = AdaptiveGainTuningState()
    compute_effective_gains(
        base_position_gain=10.0,
        base_orientation_gain=10.0,
        position_error_m=0.0,
        orientation_error_rad=0.0,
        config=cfg,
        state=state,
        enabled=True,
    )
    pos_after_jump, _, _, _ = compute_effective_gains(
        base_position_gain=10.0,
        base_orientation_gain=10.0,
        position_error_m=0.15,
        orientation_error_rad=0.0,
        config=cfg,
        state=state,
        enabled=True,
    )
    assert 10.0 < pos_after_jump < 30.0


def test_reset_adaptive_gain_state() -> None:
    state = AdaptiveGainTuningState(position_scale=2.5, orientation_scale=1.8)
    reset_adaptive_gain_state(state)
    assert state.position_scale == 1.0
    assert state.orientation_scale == 1.0
