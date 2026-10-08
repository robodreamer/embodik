"""Direction-preserving scaling and explicit recovery at exact joint bounds."""

from pathlib import Path

import numpy as np
import pytest

import embodik as eik
from examples.harnesses.joint_limit_recovery_harness import (
    BOUND_TOLERANCE,
    DT,
    ROBOT_URDF,
    TARGET_SPEED,
)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("allow_fallback", [False, True])
def test_exact_joint_bound_respects_requested_recovery(
    tmp_path: Path, direction: float, allow_fallback: bool
) -> None:
    urdf = tmp_path / "two_sliders.urdf"
    urdf.write_text(ROBOT_URDF)
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = DT
    runtime = eik.SolverRuntimeConfig()
    runtime.weighted_fallback_enabled = False
    solver.configure_runtime(runtime)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    lower, upper = robot.get_joint_limits()
    configuration = np.zeros(robot.nq)
    configuration[1] = upper[1] if direction > 0 else lower[1]
    robot.update_configuration(configuration)
    task = solver.add_frame_task("tool_motion", "tool", eik.TaskType.FRAME_POSITION)
    task.solve_mode = eik.TaskSolveMode.SCALE
    task.allow_min_error_fallback = allow_fallback
    desired_twist = np.array([TARGET_SPEED, direction * TARGET_SPEED, 0.0])
    task.set_target_velocity(desired_twist)

    result = solver.solve_velocity(configuration, apply_limits=True)
    velocity = np.asarray(result.joint_velocities)
    candidate = configuration + DT * velocity
    assert np.all(np.isfinite(velocity))
    assert np.all(candidate >= np.asarray(lower) - BOUND_TOLERANCE)
    assert np.all(candidate <= np.asarray(upper) + BOUND_TOLERANCE)
    assert not result.weighted_fallback_used
    if allow_fallback:
        assert result.status == eik.SolverStatus.SUCCESS
        assert result.task_used_fallback == [True]
        assert result.task_modes_effective == [eik.TaskSolveMode.MIN_ERROR]
        assert velocity[0] > 0
        achieved = robot.get_frame_jacobian("tool")[:3] @ velocity
        assert np.linalg.norm(desired_twist - achieved) < np.linalg.norm(desired_twist)
    else:
        np.testing.assert_allclose(velocity, np.zeros_like(velocity), atol=BOUND_TOLERANCE)
        assert result.task_used_fallback == [False]
        assert result.task_modes_effective == [eik.TaskSolveMode.SCALE]


def test_terminal_fallback_preserves_higher_priority_motion(tmp_path: Path) -> None:
    urdf = tmp_path / "two_sliders.urdf"
    urdf.write_text(ROBOT_URDF)
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = DT
    runtime = eik.SolverRuntimeConfig()
    runtime.weighted_fallback_enabled = False
    solver.configure_runtime(runtime)
    configuration = np.zeros(robot.nq)
    robot.update_configuration(configuration)
    primary = solver.add_joint_task("free_motion", "free_slider", TARGET_SPEED)
    primary.priority = 0
    primary.allow_min_error_fallback = False
    baseline = solver.solve_velocity(configuration, apply_limits=True)
    assert baseline.status == eik.SolverStatus.SUCCESS
    protected_velocity = np.asarray(baseline.joint_velocities)[0]
    assert protected_velocity > 0
    secondary = solver.add_frame_task("tool_motion", "tool", eik.TaskType.FRAME_POSITION)
    secondary.priority = 1
    secondary.allow_min_error_fallback = True
    secondary.set_target_velocity(np.array([TARGET_SPEED, TARGET_SPEED, 0.0]))

    result = solver.solve_velocity(configuration, apply_limits=True)
    assert result.status == eik.SolverStatus.SUCCESS
    assert result.task_used_fallback == [False, True]
    assert result.task_modes_effective == [
        eik.TaskSolveMode.SCALE,
        eik.TaskSolveMode.MIN_ERROR,
    ]
    np.testing.assert_allclose(
        np.asarray(result.joint_velocities)[0], protected_velocity, atol=BOUND_TOLERANCE
    )
    assert np.asarray(result.joint_velocities)[1] <= BOUND_TOLERANCE


@pytest.mark.parametrize("rows", ["zero", "dependent", "inconsistent"])
def test_saturated_task_rank_uses_target_reachability(rows: str) -> None:
    from scipy.optimize import linprog

    matrix = (
        np.array([[1.0, 1.0], [0.0, 0.0]]) if rows == "zero" else np.array([[1.0, 1.0], [2.0, 2.0]])
    )
    target = np.array([1.0, 0.0]) if rows == "zero" else np.array([1.0, 2.0])
    if rows == "inconsistent":
        target[1] = 2.01
    lower, upper = np.array([-2.0, -2.0]), np.array([2.0, 0.0])
    reference = linprog(
        [0.0, 0.0, -1.0],
        A_eq=np.column_stack((matrix, -target)),
        b_eq=np.zeros(2),
        bounds=[(-2.0, 2.0), (-2.0, 0.0), (0.0, 1.0)],
        method="highs",
    )
    assert reference.success
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [target],
        [matrix],
        np.eye(2),
        lower,
        upper,
    )
    velocity = np.asarray(result.solution)
    assert result.status == eik.SolverStatus.SUCCESS
    np.testing.assert_allclose(result.task_scales[0], reference.x[-1], atol=1e-8)
    np.testing.assert_allclose(matrix @ velocity, result.task_scales[0] * target, atol=1e-8)
    assert np.all(velocity >= lower - 1e-8)
    assert np.all(velocity <= upper + 1e-8)


@pytest.mark.parametrize("redundant", ["zero", "dependent"])
def test_consistent_redundant_row_terminates_past_default_iteration_limit(
    redundant: str,
) -> None:
    """A duplicate row is not a rank failure, even past the 20-iteration cap."""

    degrees_of_freedom = 21
    matrix = np.zeros((2, degrees_of_freedom))
    matrix[0, :] = 1.0
    matrix[1, :] = 0.0 if redundant == "zero" else 2.0
    target = np.array([1.0, 0.0 if redundant == "zero" else 2.0])
    lower = np.zeros(degrees_of_freedom)
    upper = np.zeros(degrees_of_freedom)
    upper[-1] = 2.0
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [target],
        [matrix],
        np.eye(degrees_of_freedom),
        lower,
        upper,
    )
    velocity = np.asarray(result.solution)
    assert result.status == eik.SolverStatus.SUCCESS
    np.testing.assert_allclose(result.task_scales[0], 1.0, atol=1e-8)
    np.testing.assert_allclose(matrix @ velocity, target, atol=1e-8)
    np.testing.assert_allclose(velocity[:-1], 0.0, atol=1e-8)
    np.testing.assert_allclose(velocity[-1], 1.0, atol=1e-8)
    assert np.all(velocity >= lower - 1e-8)
    assert np.all(velocity <= upper + 1e-8)


def test_inconsistent_redundant_row_scales_instead_of_rank_error() -> None:
    matrix = np.array([[1.0, 1.0], [2.0, 2.0]])
    target = np.array([1.0, 2.01])
    lower, upper = np.array([-2.0, -2.0]), np.array([2.0, 0.0])
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [target],
        [matrix],
        np.eye(2),
        lower,
        upper,
    )
    velocity = np.asarray(result.solution)
    assert result.status == eik.SolverStatus.SUCCESS
    np.testing.assert_allclose(result.task_scales[0], 0.0, atol=1e-8)
    np.testing.assert_allclose(matrix @ velocity, 0.0, atol=1e-8)
    assert np.all(velocity >= lower - 1e-8)
    assert np.all(velocity <= upper + 1e-8)
