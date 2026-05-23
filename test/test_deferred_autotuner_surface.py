from __future__ import annotations

import embodik as eik

DEFERRED_RUNTIME_FIELDS = {
    "enable_auto_floor",
    "auto_floor_min",
    "auto_floor_max",
    "enable_auto_floor_per_channel",
    "auto_floor_rot_min",
    "auto_floor_rot_max",
    "enable_auto_gain_adapt",
    "auto_gain_climb_rate",
    "auto_gain_backoff_rate",
    "auto_gain_cascade_window_ticks",
    "auto_gain_cascade_threshold",
    "auto_gain_task_scale_backoff_threshold",
    "auto_gain_task_scale_backoff_window_ticks",
    "auto_gain_pos_error_band_m",
    "auto_gain_ori_error_band_rad",
    "auto_gain_jerk_ceiling_rad_s3",
    "auto_gain_climb_band_override_multiplier",
    "auto_gain_min",
    "auto_gain_max",
}


def test_high_risk_stateful_autotuners_are_not_public_runtime_surface() -> None:
    """Auto-floor and auto-gain stay deferred until their math gate is reopened."""
    cfg = eik.SolverRuntimeConfig()

    leaked = [name for name in DEFERRED_RUNTIME_FIELDS if hasattr(cfg, name)]

    assert leaked == []


def test_advisor_scale_pi_experiment_is_default_off_and_resettable() -> None:
    cfg = eik.SolverRuntimeConfig()

    assert cfg.enable_advisor_scale_adapt is False
    assert cfg.advisor_scale_adapt_ki > 0.0
    assert cfg.advisor_scale_min < cfg.advisor_scale_max
    assert cfg.advisor_scale_epoch_s > 0.0
    assert hasattr(eik.KinematicsSolver, "reset_adaptive_state")
