#!/usr/bin/env python3
"""Regression tests for weighted-advisor row intent on pose objectives."""

from __future__ import annotations

import math

import numpy as np
import pytest

import embodik as eik

pytest.importorskip("robot_descriptions")

_PANDA_DEFAULT_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.05, 0.05])
_PANDA_EE_FRAME = "panda_hand"


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _make_pose_conflict(*, grouped: bool) -> tuple[eik.KinematicsSolver, np.ndarray]:
    from robot_descriptions.panda_description import URDF_PATH

    robot = eik.RobotModel(URDF_PATH, floating_base=False)
    q = _PANDA_DEFAULT_Q.copy()
    robot.update_configuration(q)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01

    pose = robot.get_frame_pose(_PANDA_EE_FRAME)
    pose_task = solver.add_frame_task("ee_pose", _PANDA_EE_FRAME, eik.TaskType.FRAME_POSE)
    pose_task.priority = 0
    pose_task.weight = 1.0
    pose_task.set_target_position(
        np.asarray(pose.translation, dtype=float) + np.array([0.05, 0.03, 0.02])
    )
    pose_task.set_target_orientation(np.asarray(pose.rotation, dtype=float) @ _rot_z(0.6))

    posture_task = solver.add_posture_task("posture")
    posture_task.priority = 0 if grouped else 1
    posture_task.weight = 50.0
    posture_task.set_target_configuration(q)
    return solver, q


def _solve_pose_advisor(
    *, pos_scale: float, ori_scale: float, grouped: bool
) -> tuple[float, float]:
    solver, q = _make_pose_conflict(grouped=grouped)
    cfg = eik.SolverRuntimeConfig()
    cfg.weighted_advisor_enabled = True
    cfg.advisor_position_weight_scale = pos_scale
    cfg.advisor_orientation_weight_scale = ori_scale
    solver.configure_runtime(cfg)

    result = solver.solve_velocity(q, apply_limits=True)

    assert result.weighted_advisory_available is True
    pos_error = float(result.weighted_advisory_pos_task_error_norm)
    ori_error = float(result.weighted_advisory_ori_task_error_norm)
    assert math.isfinite(pos_error) and pos_error > 0.0
    assert math.isfinite(ori_error) and ori_error > 0.0
    return pos_error, ori_error


@pytest.mark.parametrize("grouped", [False, True])
def test_pose_orientation_rows_respond_to_orientation_scale(grouped: bool):
    _pos_base, ori_base = _solve_pose_advisor(pos_scale=1.0, ori_scale=1.0, grouped=grouped)
    _pos_scaled, ori_scaled = _solve_pose_advisor(pos_scale=1.0, ori_scale=10.0, grouped=grouped)

    assert (ori_base - ori_scaled) / ori_base > 0.05


@pytest.mark.parametrize("grouped", [False, True])
def test_pose_position_rows_respond_to_position_scale(grouped: bool):
    pos_base, _ori_base = _solve_pose_advisor(pos_scale=1.0, ori_scale=1.0, grouped=grouped)
    pos_scaled, _ori_scaled = _solve_pose_advisor(pos_scale=100.0, ori_scale=1.0, grouped=grouped)

    assert (pos_base - pos_scaled) / pos_base > 0.05


def test_advisor_orientation_weight_scale_defaults_to_one():
    cfg = eik.SolverRuntimeConfig()
    assert cfg.advisor_orientation_weight_scale == pytest.approx(1.0)


def test_advisor_scale_pi_increases_position_scale_when_position_ratio_is_high():
    solver, q = _make_pose_conflict(grouped=True)
    cfg = eik.SolverRuntimeConfig()
    cfg.weighted_advisor_enabled = True
    cfg.enable_advisor_scale_adapt = True
    cfg.advisor_position_weight_scale = 1.0
    cfg.advisor_scale_adapt_target_ratio = 0.0
    cfg.advisor_scale_adapt_ki = 0.5
    cfg.advisor_scale_epoch_s = 0.01
    solver.configure_runtime(cfg)

    result = solver.solve_velocity(q, apply_limits=True)

    assert result.weighted_advisory_available is True
    assert result.advisor_scale_adapt_active is True
    assert result.advisor_position_weight_scale_current > 1.0


def test_advisor_scale_pi_reset_restores_configured_scale():
    solver, q = _make_pose_conflict(grouped=True)
    cfg = eik.SolverRuntimeConfig()
    cfg.weighted_advisor_enabled = True
    cfg.enable_advisor_scale_adapt = True
    cfg.advisor_position_weight_scale = 1.5
    cfg.advisor_scale_adapt_target_ratio = 0.0
    cfg.advisor_scale_adapt_ki = 0.5
    cfg.advisor_scale_epoch_s = 0.01
    solver.configure_runtime(cfg)

    adapted = solver.solve_velocity(q, apply_limits=True)
    adapted_again = solver.solve_velocity(q, apply_limits=True)
    assert adapted.advisor_position_weight_scale_current > 1.5
    assert (
        adapted_again.advisor_position_weight_scale_current
        > adapted.advisor_position_weight_scale_current
    )

    solver.reset_adaptive_state()
    reset = solver.solve_velocity(q, apply_limits=True)

    assert reset.advisor_position_weight_scale_current > 1.5
    assert reset.advisor_position_weight_scale_current == pytest.approx(
        adapted.advisor_position_weight_scale_current
    )
