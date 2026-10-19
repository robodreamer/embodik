"""Opt-in SCALE improvement preserves physical bounds and earlier tasks."""

import numpy as np
import pytest
from scipy.optimize import linprog

import embodik as eik


def _solve(jacobians, goals, lower, upper, refine):
    return eik.computeMultiObjectiveVelocitySolutionEigen(
        goals,
        jacobians,
        np.eye(len(lower)),
        lower,
        upper,
        sr_tolerance=1e-9,
        sr_damping=0.0,
        refine_partial_scale=refine,
    )


def test_partial_scale_refinement_releases_a_suboptimal_bound_face():
    jacobian = np.array([[2.0, 1.0, -3.0], [-3.0, 1.0, 2.0]])
    target = np.array([-2.0, -2.0])
    lower, upper = np.array([-1.0, -1.0, 0.0]), np.ones(3)
    baseline = _solve([jacobian], [target], lower, upper, False)
    improved = _solve([jacobian], [target], lower, upper, True)
    np.testing.assert_allclose(jacobian @ baseline.solution, 0.75 * target, atol=1e-9)
    np.testing.assert_allclose(jacobian @ improved.solution, target, atol=1e-9)
    np.testing.assert_allclose(improved.task_scales, [1.0], atol=1e-9)
    np.testing.assert_allclose(improved.solution, [1.0, -1.0, 1.0], atol=1e-9)
    assert np.all(improved.solution >= lower - 1e-9)
    assert np.all(improved.solution <= upper + 1e-9)


@pytest.mark.parametrize("protected_value", [-1.0, -0.5])
def test_refinement_preserves_an_earlier_task(protected_value):
    protected = np.array([[0.0, 1.0, 0.0]])
    jacobian = np.array([[2.0, 1.0, -3.0], [-3.0, 1.0, 2.0]])
    target = np.array([-2.0, -2.0])
    lower, upper = np.array([-1.0, -1.0, 0.0]), np.ones(3)
    result = _solve(
        [protected, jacobian], [np.array([protected_value]), target], lower, upper, True
    )
    reference = linprog(
        [0.0, 0.0, 0.0, -1.0],
        A_eq=np.vstack([np.column_stack([jacobian, -target]), np.column_stack([protected, [0.0]])]),
        b_eq=np.array([0.0, 0.0, protected_value]),
        bounds=list(zip(lower, upper)) + [(0.0, 1.0)],
        method="highs",
    )
    assert reference.success
    np.testing.assert_allclose(protected @ result.solution, [protected_value], atol=1e-9)
    np.testing.assert_allclose(jacobian @ result.solution, reference.x[-1] * target, atol=1e-9)
    assert np.all(result.solution >= lower - 1e-9)
    assert np.all(result.solution <= upper + 1e-9)


def test_refinement_never_regresses_random_feasible_ray_or_bounds():
    rng = np.random.default_rng(20261013)
    for _ in range(32):
        jacobian = rng.normal(size=(2, 4))
        target = rng.normal(size=2)
        lower, upper = -rng.uniform(0.1, 1.0, 4), rng.uniform(0.1, 1.0, 4)
        lower[0] = 0.0
        baseline = _solve([jacobian], [target], lower, upper, False)
        refined = _solve([jacobian], [target], lower, upper, True)
        assert refined.task_scales[0] >= baseline.task_scales[0] - 1e-8
        np.testing.assert_allclose(
            jacobian @ refined.solution, refined.task_scales[0] * target, atol=1e-8
        )
        assert np.all(refined.solution >= lower - 1e-8)
        assert np.all(refined.solution <= upper + 1e-8)


def _linear_rate_robot(tmp_path):
    axes = np.array([[2.0, -3.0, 0.0], [1.0, 1.0, 0.0], [-3.0, 2.0, 0.0]])
    speeds = np.linalg.norm(axes, axis=1)
    axes /= speeds[:, None]
    pieces = ['<robot name="linear_rate_test"><link name="base"/>']
    parent = "base"
    for index, (axis, speed) in enumerate(zip(axes, speeds)):
        child = f"link{index}"
        pieces.append(
            f'<link name="{child}"/><joint name="joint{index}" type="prismatic">'
            f'<parent link="{parent}"/><child link="{child}"/>'
            f'<axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>'
            f'<limit lower="{0 if index == 2 else -10}" upper="10" effort="100" velocity="{speed}"/>'
            "</joint>"
        )
        parent = child
    pieces.append("</robot>")
    path = tmp_path / "linear_rate.urdf"
    path.write_text("".join(pieces))
    robot = eik.RobotModel(str(path), floating_base=False)
    return robot, axes, speeds, parent


def test_high_level_runtime_refinement_reaches_lp_rate(tmp_path):
    robot, axes, speeds, parent = _linear_rate_robot(tmp_path)
    solver = eik.KinematicsSolver(robot)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.set_regularization_epsilon(1e-9)
    task = solver.add_frame_task("position", parent, eik.TaskType.FRAME_POSITION)
    target = np.array([-2.0, -2.0, 0.0])
    task.set_target_position_velocity(target)
    q = np.zeros(3)
    cfg = solver.runtime_config()
    assert cfg.refine_partial_scale is False
    cfg.damping = 0.0
    solver.configure_runtime(cfg)
    baseline = solver.solve_velocity(q, True, False)
    cfg.refine_partial_scale = True
    solver.configure_runtime(cfg)
    refined = solver.solve_velocity(q, True, False)
    assert solver.runtime_config().refine_partial_scale is True
    j = axes.T
    lower, upper = -speeds, speeds.copy()
    lower[2] = 0.0
    lp = linprog(
        np.r_[np.zeros(3), -1.0],
        A_eq=np.column_stack((j, -target)),
        b_eq=np.zeros(3),
        bounds=list(zip(lower, upper)) + [(0.0, 1.0)],
        method="highs",
    )
    assert lp.success
    velocity = np.asarray(refined.solution)
    np.testing.assert_allclose(j @ velocity, lp.x[-1] * target, atol=1e-6)
    assert refined.task_scales[0] > baseline.task_scales[0] + 0.1
    assert np.all(velocity >= lower - 1e-6) and np.all(velocity <= upper + 1e-6)


def test_acceleration_refinement_preserves_optimal_rate_and_bounds(tmp_path):
    robot, axes, speeds, tip = _linear_rate_robot(tmp_path)
    solver = eik.AccelerationSolver(robot)
    task = solver.add_frame_task("position", tip, eik.TaskType.FRAME_POSITION)
    task.solve_mode = eik.TaskSolveMode.SCALE
    target = np.array([-2.0, -2.0, 0.0])
    reference = eik.AccelerationTaskReference()
    reference.proportional_gain = 0.0
    reference.derivative_gain = 0.0
    reference.desired_acceleration = target
    solver.set_task_reference("position", reference)
    options = eik.AccelerationSolveOptions()
    options.acceleration_limits_override = speeds
    options.allow_state_box_task_fallback = False
    q = np.zeros(3)
    baseline = solver.solve(q, q, 0.01, options)
    options.refine_partial_scale = True
    refined = solver.solve(q, q, 0.01, options)
    assert baseline.status == eik.SolverStatus.SUCCESS, baseline.status_message
    assert refined.status == eik.SolverStatus.SUCCESS, refined.status_message
    assert refined.task_scales[0] >= baseline.task_scales[0] - 1e-10
    np.testing.assert_allclose(refined.task_scales, [1.0], atol=1e-9)
    acceleration = np.asarray(refined.joint_accelerations)
    np.testing.assert_allclose(axes.T @ acceleration, target, atol=1e-9)
    assert np.all(acceleration >= -speeds - 1e-9)
    assert np.all(acceleration <= speeds + 1e-9)
    assert acceleration[2] >= -1e-9
    options.refine_partial_scale = False
    restored = solver.solve(q, q, 0.01, options)
    assert restored.status == eik.SolverStatus.SUCCESS, restored.status_message
    np.testing.assert_array_equal(restored.joint_accelerations, baseline.joint_accelerations)
