"""Legacy SCALE must distinguish impossible task rays from supported damping."""

from pathlib import Path

import numpy as np
import pytest

import embodik as eik


def _solve(matrix: np.ndarray, goal: np.ndarray, **kwargs):
    width = matrix.shape[1]
    return eik.computeMultiObjectiveVelocitySolutionEigen(
        [goal],
        [np.asfortranarray(matrix)],
        np.eye(width),
        -10.0 * np.ones(width),
        10.0 * np.ones(width),
        **kwargs,
    )


@pytest.mark.parametrize("row_order", [[0, 1, 2], [2, 0, 1]])
@pytest.mark.parametrize("row_scales", [[1.0, 1.0, 1.0], [0.01, 10.0, 0.5]])
def test_inconsistent_dependent_ray_cannot_report_success(
    row_order: list[int], row_scales: list[float]
) -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    goal = np.array([1.0, 1.0, 3.0])
    # Row 3 equals rows 1+2, while its demand differs. Every positive
    # original-ray scale is impossible, independently of velocity bounds.
    scaling = np.asarray(row_scales)
    matrix = (scaling[:, None] * matrix)[row_order]
    goal = (scaling * goal)[row_order]
    result = _solve(matrix, goal, sr_damping=0.0)
    assert result.task_modes_effective == [eik.TaskSolveMode.SCALE]
    assert result.task_used_fallback == [False]
    if result.status == eik.SolverStatus.SUCCESS:
        assert result.task_scales[0] <= 1e-6
        np.testing.assert_allclose(matrix @ result.solution, 0.0, atol=1e-6)
    else:
        assert result.status == eik.SolverStatus.NO_PROGRESS
    assert np.max(np.abs(result.solution)) <= 10.0 + 1e-6


@pytest.mark.parametrize("extra_row", ["zero", "repeated", "dependent"])
@pytest.mark.parametrize("sr_tolerance", [1e-6, 0.1])
def test_consistent_redundant_ray_preserves_regularized_solution(
    extra_row: str, sr_tolerance: float
) -> None:
    base = np.eye(2)
    row = {
        "zero": np.array([0.0, 0.0]),
        "repeated": np.array([1.0, 0.0]),
        "dependent": np.array([1.0, 1.0]),
    }[extra_row]
    matrix = np.vstack((base, row))
    goal = matrix @ np.array([0.2, -0.1])
    result = _solve(matrix, goal, sr_tolerance=sr_tolerance)
    assert result.status == eik.SolverStatus.SUCCESS
    assert result.task_scales == pytest.approx([1.0])
    assert result.task_used_fallback == [False]
    np.testing.assert_allclose(matrix @ result.solution, goal, atol=0.005)


@pytest.mark.parametrize("weak_axis", [1e-5, 1e-8])
def test_supported_near_singular_ray_retains_regularized_behavior(
    weak_axis: float,
) -> None:
    matrix = np.diag([1.0, weak_axis])
    goal = np.array([0.1, 0.1])
    result = _solve(matrix, goal, sr_tolerance=0.1)
    assert result.status == eik.SolverStatus.SUCCESS
    assert result.task_scales == pytest.approx([1.0])
    assert result.task_used_fallback == [False]
    assert np.all(np.isfinite(result.solution))


@pytest.mark.parametrize("tiny_axis", [1e-12, 1e-11])
def test_numerically_zero_primary_row_retains_legacy_rejection(
    tiny_axis: float,
) -> None:
    matrix = np.diag([1.0, tiny_axis])
    goal = np.array([0.1, 0.1])
    result = _solve(matrix, goal, sr_tolerance=0.1)
    assert result.status == eik.SolverStatus.NO_PROGRESS
    assert result.task_used_fallback == [False]
    assert result.task_modes_effective == [eik.TaskSolveMode.SCALE]


def _solver(tmp_path: Path):
    urdf = tmp_path / "oblique_sliders.urdf"
    urdf.write_text(
        '<robot name="oblique_sliders"><link name="base"/>'
        '<link name="middle"/><link name="tool"/>'
        '<joint name="first" type="prismatic"><parent link="base"/>'
        '<child link="middle"/><axis xyz="1 0 1"/>'
        '<limit lower="-100" upper="100" effort="1" velocity="10"/></joint>'
        '<joint name="second" type="prismatic"><parent link="middle"/>'
        '<child link="tool"/><axis xyz="0 1 1"/>'
        '<limit lower="-100" upper="100" effort="1" velocity="10"/></joint></robot>'
    )
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    runtime = solver.runtime_config()
    runtime.weighted_fallback_enabled = False
    solver.configure_runtime(runtime)
    return robot, solver


@pytest.mark.parametrize("allow_fallback", [False, True])
@pytest.mark.parametrize("mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.SCALE_ELASTIC])
def test_structurally_inconsistent_primary_honors_explicit_fallback(
    tmp_path: Path, allow_fallback: bool, mode: eik.TaskSolveMode
) -> None:
    robot, solver = _solver(tmp_path)
    task = solver.add_frame_task("motion", "tool", eik.TaskType.FRAME_POSITION)
    task.solve_mode = mode
    task.allow_min_error_fallback = allow_fallback
    task.set_target_velocity(np.array([1.0, 1.0, 3.0]))
    result = solver.solve_velocity(np.zeros(robot.nq), apply_limits=True)
    assert not result.weighted_fallback_used
    if allow_fallback:
        assert result.status == eik.SolverStatus.SUCCESS
        assert result.task_used_fallback == [True]
        assert result.task_modes_effective == [eik.TaskSolveMode.MIN_ERROR]
        assert result.task_scales == [-1.0]
    else:
        assert result.status == eik.SolverStatus.NO_PROGRESS
        assert result.task_used_fallback == [False]
        assert result.task_modes_effective == [eik.TaskSolveMode.SCALE]
    assert np.max(np.abs(result.joint_velocities)) <= 10.0 + 1e-6


def test_scale_elastic_retains_supported_task_motion(tmp_path: Path) -> None:
    robot, solver = _solver(tmp_path)
    q = np.zeros(robot.nq)
    robot.update_configuration(q)
    matrix = np.asarray(robot.get_frame_jacobian("tool"))[:3].copy()
    goal = matrix @ np.array([0.2, -0.1])
    task = solver.add_frame_task("motion", "tool", eik.TaskType.FRAME_POSITION)
    task.solve_mode = eik.TaskSolveMode.SCALE_ELASTIC
    task.allow_min_error_fallback = False
    task.set_target_velocity(goal)
    result = solver.solve_velocity(q, apply_limits=True)
    assert result.status == eik.SolverStatus.SUCCESS
    assert result.task_used_fallback == [False]
    assert result.task_modes_effective == [eik.TaskSolveMode.SCALE]
    assert result.task_scales == pytest.approx([1.0])
    np.testing.assert_allclose(matrix @ result.joint_velocities, goal, atol=0.005)


@pytest.mark.parametrize("allow_fallback", [False, True])
def test_legacy_secondary_projection_retains_primary_and_requested_policy(
    tmp_path: Path, allow_fallback: bool
) -> None:
    robot, solver = _solver(tmp_path)
    primary = solver.add_joint_task("primary", "first")
    primary.priority = 0
    primary.allow_min_error_fallback = False
    primary.set_target_velocity(np.array([0.2]))
    q = np.zeros(robot.nq)
    baseline = solver.solve_velocity(q, apply_limits=True)
    assert baseline.status == eik.SolverStatus.SUCCESS
    protected_velocity = np.asarray(baseline.joint_velocities)[0]
    secondary = solver.add_frame_task("secondary", "tool", eik.TaskType.FRAME_POSITION)
    secondary.priority = 1
    secondary.allow_min_error_fallback = allow_fallback
    secondary.set_target_velocity(np.array([1.0, 1.0, 3.0]))
    result = solver.solve_velocity(q, apply_limits=True)
    np.testing.assert_allclose(
        np.asarray(result.joint_velocities)[0], protected_velocity, atol=1e-8
    )
    # Lower legacy objectives retain their projected preference behavior. The
    # primary structural check must not reinterpret that policy as a failed
    # primary ray or invoke an otherwise unnecessary MIN_ERROR retry.
    assert result.status == eik.SolverStatus.SUCCESS
    assert result.task_used_fallback == [False, False]
    assert result.task_modes_effective == [eik.TaskSolveMode.SCALE] * 2


def test_inconsistent_ray_does_not_replace_feasible_nonzero_velocity_with_zero() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    goal = np.array([1.0, 1.0, 3.0])
    lower, upper = np.array([0.1, 0.1]), np.array([10.0, 10.0])
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [goal], [matrix], np.eye(2), lower, upper, sr_damping=0.0
    )
    assert result.status == eik.SolverStatus.NO_PROGRESS
    velocity = np.asarray(result.solution)
    assert np.all(velocity >= lower - 1e-6)
    assert np.all(velocity <= upper + 1e-6)


@pytest.mark.parametrize("multiple_targets", [False, True])
def test_position_step_projection_scope_restores_direct_velocity_validation(
    tmp_path: Path, multiple_targets: bool
) -> None:
    robot, solver = _solver(tmp_path)
    task = solver.add_frame_task("motion", "tool", eik.TaskType.FRAME_POSITION)
    task.allow_min_error_fallback = False
    q = np.zeros(robot.nq)
    robot.update_configuration(q)
    target = np.asarray(robot.get_frame_pose("tool").homogeneous()).copy()
    target[:3, 3] += np.array([0.01, 0.01, 0.03])
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.position_gain = 10.0
    options.orientation_gain = 0.0
    options.primary_allow_min_error_fallback = False

    def step():
        if multiple_targets:
            return solver.solve_position_step(
                q, [eik.TaskTarget("motion", target, 10.0, 0.0)], options
            )
        return solver.solve_position_step(q, target, "motion", options)

    # The position adapter retains its established projected pose-error policy
    # for an overconstrained model; direct velocity requests validate their ray.
    result = step()
    assert result.status == eik.SolverStatus.SUCCESS
    assert not result.weighted_fallback_used
    assert np.linalg.norm(np.asarray(result.q_solution) - q) > 1e-8
    robot.update_configuration(result.q_solution)
    achieved = np.asarray(robot.get_frame_pose("tool").translation)
    assert np.linalg.norm(target[:3, 3] - achieved) < np.linalg.norm(target[:3, 3])

    def assert_direct_velocity_validated() -> None:
        task.set_target_velocity(np.array([1.0, 1.0, 3.0]))
        result = solver.solve_velocity(q, apply_limits=True)
        assert result.status == eik.SolverStatus.NO_PROGRESS
        assert result.task_used_fallback == [False]

    assert_direct_velocity_validated()
    # This validation return occurs after the position-depth RAII guard is
    # constructed. Its early exit must restore direct-velocity policy too.
    options.continuity_reference_frame = "missing_scope_reference"
    assert step().status == eik.SolverStatus.INVALID_INPUT
    assert_direct_velocity_validated()


def test_iterative_position_retains_projected_progress_without_changing_direct_policy(
    tmp_path: Path,
) -> None:
    robot, solver = _solver(tmp_path)
    q = np.zeros(robot.nq)
    robot.update_configuration(q)
    target = np.asarray(robot.get_frame_pose("tool").homogeneous()).copy()
    target[:3, 3] += np.array([0.01, 0.01, 0.03])
    options = eik.PositionIKOptions()
    options.max_iterations = 1
    options.position_gain = 10.0
    options.orientation_gain = 0.0
    options.primary_allow_min_error_fallback = False
    result = solver.solve_position(q, target, "tool", options)
    assert result.status in {eik.SolverStatus.SUCCESS, eik.SolverStatus.INFEASIBLE}
    assert np.linalg.norm(np.asarray(result.q_solution) - q) > 1e-8
    robot.update_configuration(result.q_solution)
    achieved = np.asarray(robot.get_frame_pose("tool").translation)
    assert np.linalg.norm(target[:3, 3] - achieved) < np.linalg.norm(target[:3, 3])
    task = solver.add_frame_task("motion", "tool", eik.TaskType.FRAME_POSITION)
    task.allow_min_error_fallback = False
    task.set_target_velocity(np.array([1.0, 1.0, 3.0]))
    assert solver.solve_velocity(q, apply_limits=True).status == eik.SolverStatus.NO_PROGRESS
