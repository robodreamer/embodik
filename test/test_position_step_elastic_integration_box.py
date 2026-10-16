"""Elastic position commands solve inside the true execution interval."""

from pathlib import Path

import numpy as np
import pytest

import embodik as eik

URDF = """<robot name="elastic_execution_box"><link name="base"/>
<link name="first"/><joint name="joint0" type="revolute"><parent link="base"/>
<child link="first"/><axis xyz="0 0 1"/><limit lower="0" upper="1" velocity="1" effort="10"/></joint>
<link name="tool"/><joint name="joint1" type="revolute"><parent link="first"/>
<child link="tool"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="1" effort="10"/></joint></robot>"""


@pytest.mark.parametrize("multi_target", [False, True], ids=["single", "multi"])
def test_elastic_nominal_command_preserves_true_joint_box_and_task_image(
    tmp_path: Path,
    multi_target: bool,
) -> None:
    path = tmp_path / "elastic_execution_box.urdf"
    path.write_text(URDF)
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.set_acceleration_limits(np.full(2, 15.0))
    solver.enable_acceleration_limits(True)
    incoming = np.array([0.0, 0.10])
    solver.set_previous_joint_velocities(incoming)
    runtime = solver.runtime_config()
    runtime.weighted_fallback_enabled = False
    solver.configure_runtime(runtime)
    q = np.array([0.0001, 0.0096])
    robot.update_configuration(q)
    primary = solver.add_frame_task("primary", "tool", eik.TaskType.FRAME_POSE)
    primary.priority = 0
    primary.solve_mode = eik.TaskSolveMode.SCALE_ELASTIC
    primary.allow_min_error_fallback = False
    primary.set_target_pose(np.zeros(3), np.eye(3))
    secondary = solver.add_frame_task("lower", "base", eik.TaskType.FRAME_POSE)
    secondary.priority = 1
    secondary.set_target_pose(np.zeros(3), np.eye(3))
    options = eik.PositionStepOptions()
    options.dt = 0.01
    options.max_steps = 1
    options.adaptive_dt = False
    options.limit_change_from_seed = True
    options.position_gain = 0.0
    options.orientation_gain = 10.0
    options.primary_solve_mode = eik.TaskSolveMode.SCALE_ELASTIC
    options.primary_allow_min_error_fallback = False
    options.record_velocity_candidates = True
    if multi_target:
        target = eik.TaskTarget("primary", np.eye(4), 0.0, 10.0)
        target.priority_orientation_tolerance = 0.01
        result = solver.solve_position_step(
            q, [target, eik.TaskTarget("lower", np.eye(4), 0.0, 10.0)], options
        )
    else:
        result = solver.solve_position_step(q, np.eye(4), "primary", options)

    # This nonzero command follows the requested negative yaw ray while
    # satisfying history, true limits, and the protected yaw zone. Elastic
    # joint limits do not enable an off-ray MIN_ERROR fallback here.
    witness = np.array([0.0, -0.05])
    assert np.max(np.abs(witness - incoming)) <= 0.15 + 1e-15
    assert abs(np.sum(q + 0.01 * witness)) <= 0.01 + 1e-15
    assert np.sum(witness) / (-10.0 * np.sum(q)) > 0.0
    assert result.velocity_candidates
    lower, upper = map(np.asarray, robot.get_joint_limits())
    for candidate in result.velocity_candidates:
        nominal = np.asarray(candidate.solver_velocity)
        reference = np.asarray(candidate.reference_q)
        nominal_q = reference + candidate.integration_dt * nominal
        assert np.all(nominal_q >= lower - 1e-10), (
            "nominal command crosses a true joint limit before integration",
            nominal_q,
            lower,
            nominal,
            candidate.integrated_velocity,
        )
        assert np.all(nominal_q <= upper + 1e-10)
        # Both joints rotate about the same axis: yaw velocity is v0+v1.
        assert np.sum(candidate.integrated_velocity) == pytest.approx(np.sum(nominal), abs=1e-10)
    actual = (np.asarray(result.q_solution) - q) / 0.01
    assert result.status in (eik.SolverStatus.SUCCESS, eik.SolverStatus.NO_PROGRESS)
    assert np.linalg.norm(actual) > 1e-8
    assert np.sum(actual) < 0.0
    assert np.max(np.abs(actual - incoming)) <= 0.15 + 1e-8
    assert np.all(result.q_solution >= lower - 1e-10)
    assert np.all(result.q_solution <= upper + 1e-10)
    if multi_target:
        assert abs(np.sum(result.q_solution)) <= 0.01 + 1e-9
    np.testing.assert_allclose(result.joint_velocities, actual, rtol=0, atol=1e-9)
    np.testing.assert_allclose(solver.get_previous_joint_velocities(), actual, rtol=0, atol=1e-9)
