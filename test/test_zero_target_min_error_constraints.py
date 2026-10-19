"""Zero MIN_ERROR requests still obey hard rows that require motion."""

from pathlib import Path

import numpy as np
import pytest

import embodik as eik


def _solver(tmp_path: Path, dofs: int = 1) -> eik.KinematicsSolver:
    joints = "".join(
        f'<link name="link_{index}"/>'
        f'<joint name="slide_{index}" type="prismatic">'
        f'<parent link="{("base" if index == 0 else f"link_{index - 1}")}"/>'
        f'<child link="link_{index}"/><axis xyz="1 0 0"/>'
        '<limit lower="-1" upper="1" velocity="1" effort="1"/></joint>'
        for index in range(dofs)
    )
    urdf = tmp_path / "slides.urdf"
    urdf.write_text(f'<robot name="slides"><link name="base"/>{joints}</robot>')
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    runtime = solver.runtime_config()
    runtime.enable_auto_task_layout = False
    runtime.weighted_fallback_enabled = False
    solver.configure_runtime(runtime)
    return solver


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("target", [0.0, 1e-12, 0.2, 0.9])
@pytest.mark.parametrize("task_kind", ["posture", "frame"])
def test_min_error_origin_excluding_rows(
    tmp_path: Path, direction: float, target: float, task_kind: str
) -> None:
    solver = _solver(tmp_path)
    if task_kind == "posture":
        task = solver.add_posture_task("command")
        task.set_target_velocity(np.array([direction * target]))
    else:
        # Two zero Jacobian rows make this objective rank deficient.
        task = solver.add_frame_task("command", "link_0", eik.TaskType.FRAME_POSITION)
        task.set_target_velocity(np.array([direction * target, 0.0, 0.0]))
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    task.allow_min_error_fallback = False
    # Rank-deficient frame tasks use the configured damped inverse. Preserve
    # its unconstrained reference rather than assuming an undamped 0.9 result.
    reference = solver.solve_velocity(np.zeros(1), apply_limits=True)
    assert reference.status == eik.SolverStatus.SUCCESS
    reference_speed = direction * np.asarray(reference.joint_velocities)[0]
    lower, upper = sorted(direction * np.array([0.85, 1.15]))
    solver.set_linear_velocity_constraints(np.eye(1), np.array([lower]), np.array([upper]))

    result = solver.solve_velocity(np.zeros(1), apply_limits=True)

    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    np.testing.assert_allclose(
        result.joint_velocities, [direction * max(0.85, reference_speed)], atol=1e-8, rtol=0.0
    )
    assert result.task_modes_effective[0] == eik.TaskSolveMode.MIN_ERROR
    assert result.task_scales[0] == -1.0
    assert not result.task_used_fallback[0]


def test_zero_min_error_with_zero_containing_rows_stays_stationary(tmp_path: Path) -> None:
    solver = _solver(tmp_path)
    task = solver.add_posture_task("hold")
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    task.set_target_velocity(np.zeros(1))
    solver.set_linear_velocity_constraints(np.eye(1), np.array([-0.2]), np.array([0.2]))

    result = solver.solve_velocity(np.zeros(1), apply_limits=True)

    assert result.status == eik.SolverStatus.SUCCESS
    np.testing.assert_array_equal(result.joint_velocities, np.zeros(1))


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_zero_jacobian_min_error_still_finds_hard_row_feasibility(
    tmp_path: Path, direction: float
) -> None:
    solver = _solver(tmp_path)
    # A prismatic joint cannot change tool orientation. All three objective
    # rows are zero, so every feasible velocity has the same task error.
    task = solver.add_frame_task("hold", "link_0", eik.TaskType.FRAME_ORIENTATION)
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    task.set_target_velocity(np.zeros(3))
    lower, upper = sorted(direction * np.array([0.85, 1.15]))
    solver.set_linear_velocity_constraints(np.eye(1), np.array([lower]), np.array([upper]))
    result = solver.solve_velocity(np.zeros(1), apply_limits=True)
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    np.testing.assert_allclose(result.joint_velocities, [direction * 0.85], atol=1e-8)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_zero_min_error_infeasible_rows_report_infeasible(tmp_path: Path, direction: float) -> None:
    solver = _solver(tmp_path)
    task = solver.add_posture_task("hold")
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    task.set_target_velocity(np.zeros(1))
    lower, upper = sorted(direction * np.array([1.1, 1.2]))
    solver.set_linear_velocity_constraints(np.eye(1), np.array([lower]), np.array([upper]))

    result = solver.solve_velocity(np.zeros(1), apply_limits=True)

    assert result.status == eik.SolverStatus.INFEASIBLE
    assert np.all(np.isfinite(result.joint_velocities))


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_zero_min_error_preserves_earlier_task(tmp_path: Path, direction: float) -> None:
    solver = _solver(tmp_path, dofs=2)
    primary = solver.add_joint_task("primary", "slide_0")
    primary.priority = 0
    primary.solve_mode = eik.TaskSolveMode.SCALE
    primary.allow_min_error_fallback = False
    primary.set_target_velocity(np.array([0.2]))
    hold = solver.add_posture_task("hold")
    hold.priority = 1
    hold.solve_mode = eik.TaskSolveMode.MIN_ERROR
    hold.set_target_velocity(np.zeros(2))
    lower, upper = sorted(direction * np.array([0.85, 1.15]))
    solver.set_linear_velocity_constraints(
        np.array([[0.0, 1.0]]), np.array([lower]), np.array([upper])
    )

    result = solver.solve_velocity(np.zeros(2), apply_limits=True)

    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    np.testing.assert_allclose(result.joint_velocities, [0.2, direction * 0.85], atol=1e-8)
    assert result.task_modes_effective[0] == eik.TaskSolveMode.SCALE
    assert result.task_scales[0] == pytest.approx(1.0)


@pytest.mark.parametrize("mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.MIN_ERROR])
@pytest.mark.parametrize("dofs", [1, 2])
def test_satisfied_zero_primary_protects_its_joint_and_leaves_nullspace_free(tmp_path, mode, dofs):
    solver = _solver(tmp_path, dofs=dofs)
    primary = solver.add_joint_task("hold", "slide_0")
    primary.priority = 0
    primary.solve_mode = mode
    primary.allow_min_error_fallback = False
    primary.set_target_velocity(np.zeros(1))
    secondary = solver.add_posture_task("move")
    secondary.priority = 1
    secondary.solve_mode = eik.TaskSolveMode.MIN_ERROR
    secondary.set_target_velocity(np.full(dofs, 0.5))
    result = solver.solve_velocity(np.zeros(dofs), apply_limits=True)
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    assert result.joint_velocities[0] == pytest.approx(0, abs=1e-8)
    if dofs == 2:
        # Projection makes the posture objective rank deficient; retain its
        # configured damping while requiring productive nullspace motion.
        assert 0.49 < result.joint_velocities[1] <= 0.5


@pytest.mark.parametrize("mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.MIN_ERROR])
def test_zero_secondary_cancels_residual_inside_higher_task_nullspace(tmp_path, mode):
    solver = _solver(tmp_path, dofs=2)
    primary = solver.add_joint_task("move_first", "slide_0")
    primary.priority = 0
    primary.solve_mode = eik.TaskSolveMode.SCALE
    primary.allow_min_error_fallback = False
    primary.set_target_velocity(np.array([0.2]))
    secondary = solver.add_frame_task("hold_sum", "link_1", eik.TaskType.FRAME_POSITION)
    secondary.priority = 1
    secondary.solve_mode = mode
    secondary.allow_min_error_fallback = False
    secondary.set_target_velocity(np.zeros(3))
    result = solver.solve_velocity(np.zeros(2), apply_limits=True)
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    # Frame damping permits a small residual; primary motion stays exact.
    assert result.joint_velocities[0] == pytest.approx(0.2, abs=1e-8)
    assert result.joint_velocities[1] < -0.19
    assert abs(sum(result.joint_velocities)) < 0.003
    if mode == eik.TaskSolveMode.SCALE:
        assert abs(sum(result.joint_velocities)) < 1e-8
        assert result.task_scales[1] == pytest.approx(1.0)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_zero_scale_cancellation_does_not_cross_hard_rows(tmp_path, direction):
    solver = _solver(tmp_path, dofs=2)
    primary = solver.add_joint_task("move_first", "slide_0")
    primary.priority = 0
    primary.solve_mode = eik.TaskSolveMode.SCALE
    primary.allow_min_error_fallback = False
    primary.set_target_velocity(np.array([0.2]))
    secondary = solver.add_frame_task("hold_sum", "link_1", eik.TaskType.FRAME_POSITION)
    secondary.priority = 1
    secondary.solve_mode = eik.TaskSolveMode.SCALE
    secondary.allow_min_error_fallback = False
    secondary.set_target_velocity(np.zeros(3))
    lower, upper = sorted(direction * np.array([0.85, 1.15]))
    solver.set_linear_velocity_constraints(
        np.array([[0.0, 1.0]]), np.array([lower]), np.array([upper])
    )
    result = solver.solve_velocity(np.zeros(2), apply_limits=True)
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    assert result.joint_velocities[0] == pytest.approx(0.2, abs=1e-8)
    assert lower - 1e-8 <= result.joint_velocities[1] <= min(1.0, upper) + 1e-8
    assert abs(result.joint_velocities[1]) <= 1 + 1e-8
    # No exact cancellation is feasible under these rows. Keep the higher
    # objective and bounds rather than advertising a fulfilled zero task.
    assert result.task_scales[1] == pytest.approx(0.0)


@pytest.mark.parametrize("explicit_hold", [False, True])
def test_unused_frame_is_not_a_hold_but_explicit_zero_velocity_is(tmp_path, explicit_hold):
    solver = _solver(tmp_path)
    frame = solver.add_frame_task("frame", "link_0", eik.TaskType.FRAME_POSITION)
    frame.priority = 0
    frame.solve_mode = eik.TaskSolveMode.SCALE
    frame.allow_min_error_fallback = False
    if explicit_hold:
        frame.set_target_velocity(np.zeros(3))
    posture = solver.add_posture_task("move")
    posture.priority = 1
    posture.solve_mode = eik.TaskSolveMode.MIN_ERROR
    posture.set_target_velocity(np.array([0.5]))
    result = solver.solve_velocity(np.zeros(1), apply_limits=True)
    assert result.status == eik.SolverStatus.SUCCESS
    np.testing.assert_allclose(result.joint_velocities, [0.0 if explicit_hold else 0.5], atol=1e-8)
    # Velocity assembly ignores an unconfigured reference; the public
    # kinematic Jacobian remains available and independent of the command.
    np.testing.assert_allclose(frame.get_jacobian(), [[1.0], [0.0], [0.0]], atol=1e-8)
