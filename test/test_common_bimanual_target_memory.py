#!/usr/bin/env python3
"""Target-return memory helper coverage for the shared bimanual app."""

from __future__ import annotations

import numpy as np

from examples.example_helpers.common_bimanual_teleop_app import _TargetReturnMemory


def _pose(x: float = 0.0, *, yaw: float = 0.0) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    T[:3, 3] = np.array([x, 0.0, 0.0], dtype=float)
    return T


def test_target_memory_near_target_lookup_and_far_miss() -> None:
    memory = _TargetReturnMemory(position_tolerance_m=0.04, rotation_tolerance_rad=0.2)
    q_lo = -np.ones(3)
    q_hi = np.ones(3)
    assert memory.observe(
        context_key=("SCALE_ELASTIC", True),
        active_sides=("left",),
        target_poses={"left": _pose(0.50)},
        q_solution=np.array([0.1, 0.2, 0.3]),
        q_lower=q_lo,
        q_upper=q_hi,
        health_score=1.0,
        max_position_error_m=0.002,
        max_rotation_error_rad=0.01,
    )

    hit = memory.lookup(
        context_key=("SCALE_ELASTIC", True),
        active_sides=("left",),
        target_poses={"left": _pose(0.525, yaw=0.05)},
        q_current=np.zeros(3),
        q_lower=q_lo,
        q_upper=q_hi,
    )
    assert hit is not None
    assert hit.hits == 1

    miss = memory.lookup(
        context_key=("SCALE_ELASTIC", True),
        active_sides=("left",),
        target_poses={"left": _pose(0.60)},
        q_current=np.zeros(3),
        q_lower=q_lo,
        q_upper=q_hi,
    )
    assert miss is None


def test_target_memory_context_and_active_set_are_part_of_lookup() -> None:
    memory = _TargetReturnMemory()
    q_lo = -np.ones(2)
    q_hi = np.ones(2)
    target = {"left": _pose(0.2)}
    assert memory.observe(
        context_key=("SCALE", False),
        active_sides=("left",),
        target_poses=target,
        q_solution=np.array([0.1, 0.2]),
        q_lower=q_lo,
        q_upper=q_hi,
        health_score=0.5,
        max_position_error_m=0.0,
        max_rotation_error_rad=0.0,
    )

    assert (
        memory.lookup(
            context_key=("MIN_ERROR", False),
            active_sides=("left",),
            target_poses=target,
            q_current=np.zeros(2),
            q_lower=q_lo,
            q_upper=q_hi,
        )
        is None
    )
    assert (
        memory.lookup(
            context_key=("SCALE", False),
            active_sides=("right",),
            target_poses={"right": _pose(0.2)},
            q_current=np.zeros(2),
            q_lower=q_lo,
            q_upper=q_hi,
        )
        is None
    )


def test_target_memory_replaces_only_with_better_health_or_error() -> None:
    memory = _TargetReturnMemory(min_score_improvement=0.1)
    q_lo = -np.ones(2)
    q_hi = np.ones(2)
    target = {"left": _pose(0.2)}
    assert memory.observe(
        context_key=("SCALE",),
        active_sides=("left",),
        target_poses=target,
        q_solution=np.array([0.1, 0.2]),
        q_lower=q_lo,
        q_upper=q_hi,
        health_score=1.0,
        max_position_error_m=0.010,
        max_rotation_error_rad=0.0,
    )
    assert not memory.observe(
        context_key=("SCALE",),
        active_sides=("left",),
        target_poses=target,
        q_solution=np.array([0.3, 0.4]),
        q_lower=q_lo,
        q_upper=q_hi,
        health_score=1.05,
        max_position_error_m=0.010,
        max_rotation_error_rad=0.0,
    )
    assert memory.observe(
        context_key=("SCALE",),
        active_sides=("left",),
        target_poses=target,
        q_solution=np.array([0.3, 0.4]),
        q_lower=q_lo,
        q_upper=q_hi,
        health_score=1.2,
        max_position_error_m=0.010,
        max_rotation_error_rad=0.0,
    )
    hit = memory.lookup(
        context_key=("SCALE",),
        active_sides=("left",),
        target_poses=target,
        q_current=np.zeros(2),
        q_lower=q_lo,
        q_upper=q_hi,
    )
    assert hit is not None
    assert np.allclose(hit.q_solution, [0.3, 0.4])


def test_target_memory_rejects_large_tracking_error_and_limit_violation() -> None:
    memory = _TargetReturnMemory(store_position_error_m=0.02)
    q_lo = -np.ones(2)
    q_hi = np.ones(2)
    assert not memory.observe(
        context_key=("SCALE",),
        active_sides=("left",),
        target_poses={"left": _pose()},
        q_solution=np.zeros(2),
        q_lower=q_lo,
        q_upper=q_hi,
        health_score=1.0,
        max_position_error_m=0.03,
        max_rotation_error_rad=0.0,
    )
    assert not memory.observe(
        context_key=("SCALE",),
        active_sides=("left",),
        target_poses={"left": _pose()},
        q_solution=np.array([0.0, 2.0]),
        q_lower=q_lo,
        q_upper=q_hi,
        health_score=1.0,
        max_position_error_m=0.0,
        max_rotation_error_rad=0.0,
    )


def test_target_memory_bias_readout_is_ramped_not_direct_jump() -> None:
    memory = _TargetReturnMemory(max_bias_step_norm=0.2, bias_gain=1.0)
    q_lo = -np.ones(3) * 5.0
    q_hi = np.ones(3) * 5.0
    target = {"left": _pose()}
    assert memory.observe(
        context_key=("SCALE",),
        active_sides=("left",),
        target_poses=target,
        q_solution=np.array([1.0, 0.0, 0.0]),
        q_lower=q_lo,
        q_upper=q_hi,
        health_score=1.0,
        max_position_error_m=0.0,
        max_rotation_error_rad=0.0,
    )
    hit = memory.lookup(
        context_key=("SCALE",),
        active_sides=("left",),
        target_poses=target,
        q_current=np.zeros(3),
        q_lower=q_lo,
        q_upper=q_hi,
    )
    assert hit is not None
    q_bias = memory.bias_configuration(hit, np.zeros(3))
    assert np.linalg.norm(q_bias) <= 0.2 + 1e-12
    assert not np.allclose(q_bias, hit.q_solution)


def test_target_memory_sparse_store_reuses_nearby_entry() -> None:
    memory = _TargetReturnMemory(
        min_store_position_separation_m=0.05,
        min_store_rotation_separation_rad=0.2,
        min_score_improvement=0.1,
    )
    q_lo = -np.ones(2)
    q_hi = np.ones(2)
    assert memory.observe(
        context_key=("SCALE",),
        active_sides=("left",),
        target_poses={"left": _pose(0.20)},
        q_solution=np.array([0.1, 0.2]),
        q_lower=q_lo,
        q_upper=q_hi,
        health_score=1.0,
        max_position_error_m=0.0,
        max_rotation_error_rad=0.0,
    )
    assert memory.entry_count == 1

    assert not memory.observe(
        context_key=("SCALE",),
        active_sides=("left",),
        target_poses={"left": _pose(0.23)},
        q_solution=np.array([0.3, 0.4]),
        q_lower=q_lo,
        q_upper=q_hi,
        health_score=1.05,
        max_position_error_m=0.0,
        max_rotation_error_rad=0.0,
    )
    assert memory.entry_count == 1

    assert memory.observe(
        context_key=("SCALE",),
        active_sides=("left",),
        target_poses={"left": _pose(0.23)},
        q_solution=np.array([0.3, 0.4]),
        q_lower=q_lo,
        q_upper=q_hi,
        health_score=1.2,
        max_position_error_m=0.0,
        max_rotation_error_rad=0.0,
    )
    assert memory.entry_count == 1
    hit = memory.lookup(
        context_key=("SCALE",),
        active_sides=("left",),
        target_poses={"left": _pose(0.23)},
        q_current=np.zeros(2),
        q_lower=q_lo,
        q_upper=q_hi,
    )
    assert hit is not None
    assert np.allclose(hit.q_solution, [0.3, 0.4])
