"""Explicit quadratic acceleration route preserves bounds and scaling policy."""

import numpy as np
import pytest

import embodik as eik


@pytest.fixture
def planar_solver(tmp_path):
    path = tmp_path / "planar.xml"
    path.write_text(
        '<robot name="planar"><link name="base"/><link name="x"/><link name="hand"/>'
        '<joint name="x_joint" type="prismatic"><parent link="base"/><child link="x"/>'
        '<axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="100" velocity="10"/></joint>'
        '<joint name="y_joint" type="prismatic"><parent link="x"/><child link="hand"/>'
        '<axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="100" velocity="10"/></joint>'
        "</robot>"
    )
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.AccelerationSolver(robot)
    task = solver.add_frame_task("hand", "hand", eik.TaskType.FRAME_POSITION)
    task.set_target_pose(np.array([0.3, 0.3, 0.0]), np.eye(3))
    reference = eik.AccelerationTaskReference()
    reference.proportional_gain, reference.derivative_gain = 100.0, 20.0
    solver.set_task_reference("hand", reference)
    options = eik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.ones(2)
    coupled = eik.AffineAccelerationConstraint()
    coupled.source_id = "coupled-rate"
    coupled.coefficient_matrix = np.array([[1.0, 1.0]])
    coupled.affine_bias = np.zeros(1)
    coupled.lower_bounds, coupled.upper_bounds = np.array([-0.5]), np.array([0.5])
    options.affine_constraints = [coupled]
    return robot, solver, task, options


def test_direct_min_error_solves_coupled_bound_and_roundtrips(planar_solver):
    robot, solver, task, options = planar_solver
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    options.prefer_direct_min_error_qp = True
    result = solver.solve(np.zeros(2), np.zeros(2), 0.01, options)
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    assert "direct hard-constrained minimum-error QP" in result.status_message
    np.testing.assert_allclose(result.joint_accelerations, [0.25, 0.25], atol=1e-8)
    np.testing.assert_allclose(result.joint_velocities_next, [0.0025, 0.0025], atol=1e-10)
    assert result.task_modes_effective == [eik.TaskSolveMode.MIN_ERROR]
    assert not any(result.task_used_fallback)
    options.prefer_direct_min_error_qp = False
    restored = solver.solve(np.zeros(2), np.zeros(2), 0.01, options)
    assert restored.status == eik.SolverStatus.SUCCESS
    assert "direct hard-constrained minimum-error QP" not in restored.status_message


def test_direct_preference_preserves_strict_scale(planar_solver):
    robot, solver, task, options = planar_solver
    task.solve_mode = eik.TaskSolveMode.SCALE
    task.allow_min_error_fallback = False
    options.allow_state_box_task_fallback = False
    baseline = solver.solve(np.zeros(2), np.zeros(2), 0.01, options)
    options.prefer_direct_min_error_qp = True
    result = solver.solve(np.zeros(2), np.zeros(2), 0.01, options)
    assert baseline.status == result.status == eik.SolverStatus.SUCCESS
    np.testing.assert_array_equal(result.joint_accelerations, baseline.joint_accelerations)
    np.testing.assert_array_equal(result.task_scales, baseline.task_scales)
    assert result.task_modes_effective == [eik.TaskSolveMode.SCALE]
    assert not any(result.task_used_fallback)


def test_direct_preference_rejects_infeasible_hard_rows(planar_solver):
    robot, solver, task, options = planar_solver
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    options.prefer_direct_min_error_qp = True
    coupled = options.affine_constraints[0]
    coupled.lower_bounds, coupled.upper_bounds = np.array([3.0]), np.array([4.0])
    options.affine_constraints = [coupled]
    result = solver.solve(np.zeros(2), np.zeros(2), 0.01, options)
    assert result.status != eik.SolverStatus.SUCCESS
