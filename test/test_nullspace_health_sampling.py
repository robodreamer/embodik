#!/usr/bin/env python3
"""Tests for solver-owned nullspace health sampling."""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np

import embodik as eik


def _create_lower_limited_two_joint_urdf() -> str:
    urdf_content = """<?xml version="1.0"?>
<robot name="health_sampling_robot">
  <link name="base_link"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="0.0" upper="3.14" effort="100" velocity="100"/>
  </joint>
  <link name="link1"/>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="ee"/>
    <origin xyz="0.2 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="100"/>
  </joint>
  <link name="ee"/>
</robot>
"""
    fd, path = tempfile.mkstemp(suffix=".urdf")
    with os.fdopen(fd, "w") as f:
        f.write(urdf_content)
    return path


def _make_hold_joint2_solver():
    urdf_path = _create_lower_limited_two_joint_urdf()
    robot = eik.RobotModel(urdf_path, floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    task = solver.add_joint_task("hold_joint2", "joint2", target_value=0.0)
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = eik.TaskSolveMode.SCALE
    return urdf_path, robot, solver


def _enable_health_sampling(solver: eik.KinematicsSolver, *, seed: int = 42) -> None:
    cfg = eik.SolverRuntimeConfig()
    cfg.weighted_fallback_enabled = False
    cfg.health_sampling.enabled = True
    cfg.health_sampling.seed = seed
    cfg.health_sampling.sample_count = 6
    cfg.health_sampling.sample_radius = 0.04
    cfg.health_sampling.gain = 0.5
    cfg.health_sampling.joint_limit_weight = 1.0
    cfg.health_sampling.singularity_weight = 0.0
    solver.configure_runtime(cfg)


def test_health_sampling_default_off_preserves_velocity_solution():
    urdf_a, robot_a, solver_a = _make_hold_joint2_solver()
    urdf_b, robot_b, solver_b = _make_hold_joint2_solver()
    try:
        q = np.array([0.001, 0.0], dtype=float)
        baseline = solver_a.solve_velocity(q, apply_limits=True)
        solver_b.configure_runtime(eik.SolverRuntimeConfig())
        candidate = solver_b.solve_velocity(q, apply_limits=True)

        assert baseline.health_sampling_available is False
        assert candidate.health_sampling_available is False
        np.testing.assert_allclose(
            np.asarray(candidate.joint_velocities, dtype=float),
            np.asarray(baseline.joint_velocities, dtype=float),
            atol=1e-12,
            rtol=0.0,
        )
    finally:
        os.unlink(urdf_a)
        os.unlink(urdf_b)


def test_health_sampling_adds_nullspace_bias_away_from_joint_limit():
    urdf_path, _robot, solver = _make_hold_joint2_solver()
    try:
        _enable_health_sampling(solver)
        q = np.array([0.001, 0.0], dtype=float)
        result = solver.solve_velocity(q, apply_limits=True)
        dq = np.asarray(result.joint_velocities, dtype=float)

        assert result.status == eik.SolverStatus.SUCCESS
        assert result.health_sampling_available is True
        assert result.health_sampling_sampled == 6
        assert result.health_sampling_accepted >= 1
        assert result.health_sampling_applied is True
        assert math.isfinite(result.health_sampling_score_delta)
        assert result.health_sampling_joint_limit_delta > 0.0
        assert result.health_sampling_bias_norm > 0.0
        assert dq[0] > 0.0
        assert abs(dq[1]) < 1e-9
    finally:
        os.unlink(urdf_path)


def test_health_sampling_is_deterministic_for_same_seed_and_call_order():
    urdf_a, _robot_a, solver_a = _make_hold_joint2_solver()
    urdf_b, _robot_b, solver_b = _make_hold_joint2_solver()
    try:
        _enable_health_sampling(solver_a, seed=7)
        _enable_health_sampling(solver_b, seed=7)
        q = np.array([0.001, 0.0], dtype=float)

        a = solver_a.solve_velocity(q, apply_limits=True)
        b = solver_b.solve_velocity(q, apply_limits=True)

        np.testing.assert_allclose(
            np.asarray(a.joint_velocities, dtype=float),
            np.asarray(b.joint_velocities, dtype=float),
            atol=1e-12,
            rtol=0.0,
        )
        assert a.health_sampling_score_delta == b.health_sampling_score_delta
    finally:
        os.unlink(urdf_a)
        os.unlink(urdf_b)


def test_health_sampling_cache_reuses_best_observed_posture_without_random_samples():
    urdf_path, _robot, solver = _make_hold_joint2_solver()
    try:
        cfg = eik.SolverRuntimeConfig()
        cfg.weighted_fallback_enabled = False
        cfg.health_sampling.enabled = True
        cfg.health_sampling.sample_count = 0
        cfg.health_sampling.best_config_cache_enabled = True
        cfg.health_sampling.sample_radius = 0.04
        cfg.health_sampling.gain = 0.5
        cfg.health_sampling.joint_limit_weight = 1.0
        cfg.health_sampling.singularity_weight = 0.0
        solver.configure_runtime(cfg)

        healthy = solver.solve_velocity(np.array([1.0, 0.0], dtype=float), apply_limits=True)
        assert healthy.health_sampling_available is True
        assert healthy.health_sampling_cache_available is False
        assert healthy.health_sampling_applied is False
        assert healthy.health_sampling_sampled == 0

        near_limit = solver.solve_velocity(np.array([0.001, 0.0], dtype=float), apply_limits=True)
        dq = np.asarray(near_limit.joint_velocities, dtype=float)

        assert near_limit.status == eik.SolverStatus.SUCCESS
        assert near_limit.health_sampling_cache_available is True
        assert near_limit.health_sampling_cache_used is True
        assert near_limit.health_sampling_applied is True
        assert near_limit.health_sampling_sampled == 1
        assert near_limit.health_sampling_accepted == 1
        assert near_limit.health_sampling_joint_limit_delta > 0.0
        assert dq[0] > 0.0
        assert abs(dq[1]) < 1e-9

        solver.configure_runtime(cfg)
        reset = solver.solve_velocity(np.array([0.001, 0.0], dtype=float), apply_limits=True)
        assert reset.health_sampling_cache_available is False
        assert reset.health_sampling_applied is False
    finally:
        os.unlink(urdf_path)
