"""Residual-factor QP steps retain task accuracy at redundant active faces."""

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("shape", ["planar", "spatial"])
@pytest.mark.parametrize("active_face", [False, True])
@pytest.mark.parametrize("rows", ["original", "scaled", "permuted", "duplicated"])
def test_redundant_min_error_qp_with_equivalent_hard_rows(tmp_path, shape, active_face, rows):
    axes = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    if shape == "spatial":
        axes = np.random.default_rng(412).normal(size=(6, 3))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    parts = ['<robot name="redundant"><link name="base"/>']
    for index, axis in enumerate(axes):
        parent = "base" if index == 0 else f"link{index-1}"
        axis_text = " ".join(map(str, axis))
        parts.append(
            f'<link name="link{index}"/><joint name="j{index}" type="prismatic">'
            f'<parent link="{parent}"/><child link="link{index}"/>'
            f'<axis xyz="{axis_text}"/>'
            '<limit lower="-10" upper="10" effort="100" velocity="100"/></joint>'
        )
    parts.append("</robot>")
    path = tmp_path / "redundant.urdf"
    path.write_text("".join(parts))
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.AccelerationSolver(robot)
    task = solver.add_frame_task("move", f"link{len(axes)-1}", eik.TaskType.FRAME_POSITION)
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    target = np.array([2.0, 0.7, 0.1])
    reference = eik.AccelerationTaskReference()
    reference.proportional_gain = reference.derivative_gain = 0.0
    reference.desired_acceleration = target
    solver.set_task_reference("move", reference)

    # Independent SVD solution of the same ridge objective. A bound along its
    # first right singular vector has an analytic constrained optimum as well.
    J = axes.T
    U, singular, Vt = np.linalg.svd(J, full_matrices=False)
    ridge = 1e-12 * np.max(np.sum(J * J, axis=0))
    coefficients = np.where(singular > 1e-10, singular / (singular**2 + ridge), 0)
    expected = Vt.T @ (coefficients * (U.T @ target))
    face = Vt[0] * np.sign(Vt[0] @ expected)
    face_upper = 100.0
    if active_face:
        face_upper = 0.5 * (face @ expected)
        expected += (face_upper - face @ expected) * face
    C = np.vstack([face, np.ones(len(axes))])
    lower, upper = np.full(2, -100.0), np.array([face_upper, 100.0])
    if rows == "scaled":
        scale = np.array([1e-3, 1e3])
        C, lower, upper = C * scale[:, None], lower * scale, upper * scale
    elif rows == "permuted":
        C, lower, upper = C[::-1].copy(), lower[::-1].copy(), upper[::-1].copy()
    elif rows == "duplicated":
        C = np.vstack([C, C, 2 * C[:1]])
        lower, upper = np.r_[lower, lower, 2 * lower[:1]], np.r_[upper, upper, 2 * upper[:1]]

    constraint = eik.AffineAccelerationConstraint()
    constraint.source_id = "equivalent-faces"
    constraint.coefficient_matrix = C
    constraint.affine_bias = np.zeros(len(lower))
    constraint.lower_bounds, constraint.upper_bounds = lower, upper
    options = eik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.full(len(axes), 15.0)
    options.affine_constraints = [constraint]
    options.prefer_direct_min_error_qp = options.record_linear_problems = True
    result = solver.solve(np.zeros(len(axes)), np.zeros(len(axes)), 0.03, options)
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    trace = result.linear_problems[0]
    assert trace.diagnostics.direct_min_error_succeeded, trace.diagnostics.direct_min_error_message
    assert trace.backend_result.iterations < 50
    # Task accuracy and objective accuracy are the contract. A 1e-12 ridge
    # resolves null ties numerically; demanding 1e-7 coordinate agreement in
    # task-null directions would be much stricter than its KKT tolerance.
    actual = result.joint_accelerations
    np.testing.assert_allclose(J @ actual, J @ expected, atol=1e-9, rtol=0)
    objective = (
        lambda x: 0.5 * np.linalg.norm(J @ x - target) ** 2 + 0.5 * ridge * np.linalg.norm(x) ** 2
    )
    assert abs(objective(actual) - objective(expected)) <= 1e-13
    normalized_values = (C @ result.joint_accelerations) / np.linalg.norm(C, axis=1)
    assert np.all(normalized_values >= lower / np.linalg.norm(C, axis=1) - 1e-10)
    assert np.all(normalized_values <= upper / np.linalg.norm(C, axis=1) + 1e-10)
    replay = eik.replay_acceleration_linear_problem(trace)
    assert replay.status == result.status
    np.testing.assert_array_equal(replay.solution, trace.backend_result.solution)
