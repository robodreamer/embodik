"""Shared box-scale cases for the CPU ESNS solver and its GPU counterparts.

Each case locks a scale and, when the direction is part of the contract, the
velocity. ``optimal`` marks cases whose scale is the largest feasible value.
A false flag is a first feasible uniform scale that is intentionally smaller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScaleCase:
    name: str
    matrix: np.ndarray
    target: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    scale: float
    velocity: np.ndarray
    optimal: bool = True
    fi: bool = True
    pph: bool = False
    torch: bool = True
    float32: bool = False
    velocity_atol: float = 1e-5
    scale_atol: float = 1e-6


def _arrays(
    matrix,
    target,
    lower,
    upper,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
    target = np.asarray(target, dtype=float).ravel()
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()
    if matrix.shape[0] != target.size:
        raise ValueError(f"target width {target.size} != rows {matrix.shape[0]}")
    if matrix.shape[1] != lower.size or matrix.shape[1] != upper.size:
        raise ValueError("bound width must match the jacobian columns")
    return matrix, target, lower, upper


def _case(
    name: str,
    matrix,
    target,
    lower,
    upper,
    scale: float,
    velocity,
    **kwargs,
) -> ScaleCase:
    matrix, target, lower, upper = _arrays(matrix, target, lower, upper)
    velocity = np.asarray(velocity, dtype=float).ravel()
    if velocity.size != matrix.shape[1]:
        raise ValueError(f"{name}: velocity width {velocity.size} != columns {matrix.shape[1]}")
    return ScaleCase(name, matrix, target, lower, upper, scale, velocity, **kwargs)


def _last_free(degree: int, upper_last: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.zeros(degree)
    upper = np.zeros(degree)
    upper[-1] = upper_last
    velocity = np.zeros(degree)
    velocity[-1] = 1.0
    return lower, upper, velocity


def _endpoint_span(degree: int) -> tuple[np.ndarray, np.ndarray]:
    ones = np.ones(degree)
    slope = np.linspace(-1.0, 1.0, degree)
    return ones, np.vstack((ones, slope, ones + slope))


def scale_cases() -> tuple[ScaleCase, ...]:
    """Return the numerical task-scale contract, in a stable order."""

    cases: list[ScaleCase] = []

    def add(*args, **kwargs) -> None:
        cases.append(_case(*args, **kwargs))

    for degree in (4, 21):
        lower, upper, velocity = _last_free(degree)
        add(
            f"redundant_zero_n{degree}",
            np.vstack((np.ones(degree), np.zeros(degree))),
            [1.0, 0.0],
            lower,
            upper,
            1.0,
            velocity,
            pph=degree == 4,
            float32=degree == 21,
            fi=True,
        )
        add(
            f"redundant_dependent_n{degree}",
            np.vstack((np.ones(degree), 2.0 * np.ones(degree))),
            [1.0, 2.0],
            lower,
            upper,
            1.0,
            velocity,
        )

    for degree in (6, 21):
        lower, upper, velocity = _last_free(degree)
        cpu_only = degree >= 21
        add(
            f"redundant_negative_n{degree}",
            np.vstack((np.ones(degree), -2.0 * np.ones(degree))),
            [1.0, -2.0],
            lower,
            upper,
            1.0,
            velocity,
            fi=not cpu_only,
        )
        add(
            f"redundant_triple_n{degree}",
            np.vstack((np.ones(degree), 2.0 * np.ones(degree), -0.5 * np.ones(degree))),
            [1.0, 2.0, -0.5],
            lower,
            upper,
            1.0,
            velocity,
            fi=not cpu_only,
        )
        duplicate = np.zeros((3, degree))
        duplicate[0, :] = 1.0
        duplicate[1, :] = 2.0
        duplicate[2, 0] = 1.0
        ends = np.zeros(degree)
        ends[0] = 2.0
        ends[-1] = 2.0
        expected = np.zeros(degree)
        expected[0] = 1.0
        add(
            f"duplicate_plus_independent_n{degree}",
            duplicate,
            [1.0, 2.0, 1.0],
            np.zeros(degree),
            ends,
            1.0,
            expected,
            fi=not cpu_only,
        )
        ones, combined = _endpoint_span(degree)
        half = np.zeros(degree)
        half[0] = 0.5
        half[-1] = 0.5
        add(
            f"linear_combination_tight_n{degree}",
            combined,
            [1.0, 0.0, 1.0],
            np.zeros(degree),
            ends,
            1.0,
            half,
            fi=not cpu_only,
        )
        axis = np.zeros(degree)
        axis[0] = 1.0
        axis_rows = np.vstack((ones, axis, 2.0 * ones - axis))
        add(
            f"linear_combination_axis_n{degree}",
            axis_rows,
            [1.0, 1.0, 1.0],
            np.zeros(degree),
            ends,
            1.0,
            expected,
            fi=not cpu_only,
        )

    add(
        "repeated_row_partial_cap",
        np.vstack((np.ones(4), np.zeros(4))),
        [1.0, 0.0],
        np.zeros(4),
        [0.0, 0.0, 0.0, 0.5],
        0.5,
        [0.0, 0.0, 0.0, 0.5],
        pph=True,
    )
    loose = np.full(6, 5.0)
    _, loose_rows = _endpoint_span(6)
    add(
        "linear_combination_loose",
        loose_rows,
        [1.0, 0.0, 1.0],
        -loose,
        loose,
        1.0,
        np.full(6, 1.0 / 6.0),
        # GPU SRINV tolerance 0.1 damps the dependent direction. CPU does not.
        fi=False,
    )
    add(
        "linear_combination_inconsistent",
        loose_rows,
        [1.0, 0.0, 1.5],
        np.zeros(6),
        np.array([2.0, 0.0, 0.0, 0.0, 0.0, 2.0]),
        0.0,
        np.zeros(6),
    )
    angle = np.deg2rad(2.0)
    skew = np.vstack((np.array([1.0, 0.0]), np.array([np.cos(angle), np.sin(angle)])))
    skew_target = np.array([1.0, np.cos(angle)])
    add(
        "skew_two_degrees_tight",
        skew,
        skew_target,
        [-2.0, -2.0],
        [0.0, 2.0],
        0.0,
        np.zeros(2),
    )
    add(
        "skew_two_degrees_loose",
        skew,
        skew_target,
        [-2.0, -2.0],
        [2.0, 2.0],
        1.0,
        [1.0, 0.0],
        # sin(2 deg) is below the GPU SRINV tolerance, so FI damps that direction.
        fi=False,
    )

    add(
        "full_rank_blocked_joint",
        [[1.0, 1.0]],
        [1.0],
        [0.0, 0.0],
        [0.0, 2.0],
        1.0,
        [0.0, 1.0],
        pph=True,
        float32=True,
    )
    add(
        "full_rank_negative_blocked_joint",
        [[1.0, 1.0]],
        [-1.0],
        [-2.0, 0.0],
        [0.0, 0.0],
        1.0,
        [-1.0, 0.0],
    )
    add(
        "full_rank_partial_cap",
        [[1.0, 1.0]],
        [1.0],
        [0.0, 0.0],
        [0.0, 0.4],
        0.4,
        [0.0, 0.4],
    )
    add(
        "full_rank_share_two_free_joints",
        [[1.0, 1.0, 1.0]],
        [1.0],
        np.zeros(3),
        [0.0, 2.0, 2.0],
        1.0,
        [0.0, 0.5, 0.5],
    )
    add(
        "signed_row_keeps_the_open_bound",
        [[1.0, -1.0]],
        [1.0],
        [-2.0, 0.0],
        [0.2, 2.0],
        0.2,
        [0.2, 0.0],
        pph=True,
        float32=True,
    )
    add(
        "signed_row_negative_target",
        [[1.0, -1.0]],
        [-1.0],
        [0.0, -2.0],
        [0.0, 0.2],
        0.2,
        [0.0, 0.2],
    )
    add(
        "two_free_caps",
        np.ones((1, 3)),
        [1.0],
        np.zeros(3),
        [0.0, 0.4, 0.4],
        0.8,
        [0.0, 0.4, 0.4],
    )
    add(
        "midcap_uses_the_open_joint",
        np.ones((1, 3)),
        [1.0],
        np.zeros(3),
        [0.0, 0.3, 2.0],
        1.0,
        [0.0, 0.3, 0.7],
    )
    add(
        "split_caps_stop_at_the_first_feasible_scale",
        np.ones((1, 4)),
        [1.0],
        np.zeros(4),
        [0.0, 0.0, 1.0, 0.25],
        0.5,
        [0.0, 0.0, 0.25, 0.25],
        optimal=False,
    )
    for degree in (21, 40):
        lower, upper, velocity = _last_free(degree)
        add(
            f"full_rank_past_iteration_limit_n{degree}",
            np.ones((1, degree)),
            [1.0],
            lower,
            upper,
            1.0,
            velocity,
            fi=degree == 21,
            torch=True,
        )
    two_column = np.zeros((1, 21))
    two_column[0, 0] = 1.0
    two_column[0, -1] = 1.0
    _lower, upper, velocity = _last_free(21)
    add(
        "two_column_past_iteration_limit",
        two_column,
        [1.0],
        _lower,
        upper,
        1.0,
        velocity,
        fi=True,
    )
    add(
        "both_tight_keeps_uniform_scale",
        [[1.0, 1.0]],
        [1.0],
        [0.0, 0.0],
        [0.2, 0.6],
        0.4,
        [0.2, 0.2],
        optimal=False,
        float32=True,
    )
    add(
        "nonzero_lower_keeps_uniform_scale",
        [[1.0, 1.0]],
        [1.0],
        [-0.2, 0.0],
        [0.1, 2.0],
        0.2,
        [0.1, 0.1],
        optimal=False,
    )
    add(
        "wide_box_keeps_uniform_scale",
        np.ones((1, 3)),
        [1.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.2, 0.2],
        0.6,
        [0.2, 0.2, 0.2],
        optimal=False,
    )
    add(
        "offset_bound_scales_the_open_joint",
        [[1.0, 1.0]],
        [1.0],
        [0.0, -1.0],
        [0.0, 0.3],
        0.3,
        [0.0, 0.3],
    )
    add(
        "mixed_sign_blockers_n4",
        [[1.0, 1.0, -1.0, -1.0]],
        [1.0],
        np.zeros(4),
        [0.01, 0.01, 2.0, 2.0],
        0.02,
        [0.01, 0.01, 0.0, 0.0],
        pph=True,
    )
    mixed = np.ones((1, 24))
    mixed[0, 2:] = -1.0
    mixed_upper = np.full(24, 2.0)
    mixed_upper[:2] = 0.01
    mixed_velocity = np.zeros(24)
    mixed_velocity[:2] = 0.01
    add(
        "mixed_sign_blockers_n24",
        mixed,
        [1.0],
        np.zeros(24),
        mixed_upper,
        0.02,
        mixed_velocity,
        fi=False,
        velocity_atol=1e-5,
    )
    free = np.zeros(25)
    free[-3:] = [0.2, 0.3, 0.5]
    free_velocity = np.zeros(25)
    free_velocity[-3:] = 0.2
    add(
        "several_free_caps_keep_first_feasible_scale",
        np.ones((1, 25)),
        [1.0],
        np.zeros(25),
        free,
        0.6,
        free_velocity,
        optimal=False,
    )
    add(
        "identity_block_cannot_share_one_scale",
        np.eye(2),
        [1.0, 1.0],
        [0.0, -1.0],
        [0.0, 2.0],
        0.0,
        np.zeros(2),
    )
    add(
        "column_singularity",
        [[1.0, 0.0], [0.5, 0.0]],
        [1.0, 0.5],
        [0.0, 0.0],
        [0.0, 2.0],
        0.0,
        np.zeros(2),
    )
    add(
        "locked_joint_stays_fixed",
        np.ones((1, 4)),
        [1.0],
        np.zeros(4),
        [0.0, 2.0, 2.0, 2.0],
        1.0,
        [0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    )
    add(
        "both_joints_locked",
        [[1.0, 1.0]],
        [1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        0.0,
        np.zeros(2),
        pph=True,
    )
    add(
        "value_already_on_the_bound",
        [[1.0, 0.0]],
        [0.5],
        [0.0, 0.0],
        [0.5, 1.0],
        1.0,
        [0.5, 0.0],
    )
    rank2 = np.zeros((2, 4))
    rank2[0, :] = 1.0
    rank2[1, 0] = 1.0
    add(
        "rank2_blocked_column",
        rank2,
        [1.0, 0.0],
        np.zeros(4),
        [0.0, 2.0, 2.0, 2.0],
        1.0,
        [0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    )
    rank2_wide = np.zeros((2, 21))
    rank2_wide[0, :] = 1.0
    rank2_wide[1, 0] = 1.0
    shared = np.full(21, 0.05)
    shared[0] = 0.0
    add(
        "rank2_blocked_column_n21",
        rank2_wide,
        [1.0, 0.0],
        np.zeros(21),
        np.r_[0.0, np.full(20, 2.0)],
        1.0,
        shared,
        fi=True,
    )
    add(
        "zero_jacobian_zero_target",
        np.zeros((2, 4)),
        [0.0, 0.0],
        np.full(4, -1.0),
        np.full(4, 1.0),
        1.0,
        np.zeros(4),
    )
    add(
        "zero_jacobian_nonzero_target",
        np.zeros((2, 4)),
        [0.0, 1.0],
        np.full(4, -1.0),
        np.full(4, 1.0),
        0.0,
        np.zeros(4),
        pph=True,
    )
    add(
        "inconsistent_duplicate_tight",
        np.vstack((np.ones(4), 2.0 * np.ones(4))),
        [1.0, 2.01],
        np.zeros(4),
        [0.0, 0.0, 0.0, 2.0],
        0.0,
        np.zeros(4),
        pph=True,
    )
    parallel = np.array([[1.0, 1.0], [2.0, 2.0]])
    add(
        "target_within_rank_tolerance",
        parallel,
        [1.0, 2.0 + 1.0e-7],
        [-2.0, -2.0],
        [2.0, 0.0],
        1.0,
        [1.0, 0.0],
        # The exact linear program has scale 0. CPU accepts the residual inside
        # its rank tolerance. The GPU rank estimate still rejects it.
        optimal=False,
        fi=False,
        torch=False,
    )
    add(
        "target_outside_rank_tolerance",
        parallel,
        [1.0, 2.0 + 1.0e-5],
        [-2.0, -2.0],
        [2.0, 0.0],
        0.0,
        np.zeros(2),
        torch=False,
    )

    names = [case.name for case in cases]
    if len(names) != len(set(names)):
        raise RuntimeError("scale case names must be unique")
    return tuple(cases)


def cases_for(backend: str) -> tuple[ScaleCase, ...]:
    """Return the cases one backend must satisfy."""

    if backend == "cpu":
        selected = scale_cases()
    elif backend == "fi":
        selected = tuple(case for case in scale_cases() if case.fi)
    elif backend == "pph":
        selected = tuple(case for case in scale_cases() if case.pph)
    elif backend == "torch":
        selected = tuple(case for case in scale_cases() if case.torch)
    else:
        raise ValueError(backend)
    return selected
