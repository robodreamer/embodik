"""Configured acceleration limits bound physical inter-tick velocity changes."""

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize(
    "api,max_steps", [("velocity", 1), ("single", 1), ("single", 2), ("multi", 1), ("multi", 2)]
)
@pytest.mark.parametrize(
    "previous,acceleration,direction",
    [
        (0.0, 15.0, 1),
        (0.0, 15.0, -1),
        (0.4, 15.0, 1),
        (0.4, 15.0, -1),
        (0.0, 0.0, 1),
        (0.4, 0.0, 1),
    ],
)
def test_scalar_acceleration_maximum(tmp_path, api, max_steps, previous, acceleration, direction):
    urdf = tmp_path / "slide.urdf"
    urdf.write_text("""<robot name="slide"><link name="base"/><link name="tool"/>
<joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/>
<axis xyz="1 0 0"/><limit lower="-10" upper="10" velocity="10" effort="100"/>
</joint></robot>""")
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.set_acceleration_limits(np.array([acceleration]))
    solver.enable_acceleration_limits(True)
    solver.set_previous_joint_velocities(np.array([previous]))
    task = solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSITION)
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    initial = np.zeros(1)
    target = np.eye(4)
    target[0, 3] = direction
    task.set_target_pose(target[:3, 3], target[:3, :3])
    options = eik.PositionStepOptions()
    options.dt = solver.dt
    options.max_steps = max_steps
    options.position_gain = 10.0
    options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
    options.adaptive_dt = False
    options.limit_change_from_seed = True
    if api == "velocity":
        task.set_target_velocity(np.array([float(direction), 0.0, 0.0]))
        result = solver.solve_velocity(initial, apply_limits=True)
        applied = np.asarray(result.joint_velocities)
    else:
        if api == "single":
            result = solver.solve_position_step(initial, target, "move", options)
        else:
            result = solver.solve_position_step(
                initial, [eik.TaskTarget("move", target, 10.0, 0.0)], options
            )
        applied = (np.asarray(result.q_solution) - initial) / solver.dt

    assert result.status == eik.SolverStatus.SUCCESS
    exact_step = acceleration * solver.dt
    assert abs(applied[0] - previous) <= exact_step + 1e-10
    # Require the useful saturated solution, rather than permitting a held zero.
    expected = previous + direction * exact_step
    assert applied[0] == pytest.approx(expected, abs=1e-10)
    np.testing.assert_allclose(result.joint_velocities, applied, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        solver.get_previous_joint_velocities(), applied, rtol=0.0, atol=1e-12
    )


def _braking_solver(tmp_path, floating_base, coordinate):
    urdf = tmp_path / "roundtrip.urdf"
    urdf.write_text("""<robot name="roundtrip"><link name="base"/><link name="tool"/>
<joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/>
<axis xyz="1 0 0"/><limit lower="-1e18" upper="1e18" velocity="10" effort="100"/>
</joint></robot>""")
    robot = eik.RobotModel(str(urdf), floating_base=floating_base)
    solver = eik.KinematicsSolver(robot)
    solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSITION)
    q = np.asarray(robot.neutral_configuration())
    q[-1] = coordinate
    if floating_base:
        q[:3] = [2.0, -1.0, 0.5]
    robot.update_configuration(q)
    target = robot.get_frame_pose("tool").homogeneous()
    options = eik.PositionStepOptions()
    options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
    options.max_steps = 1
    if floating_base:
        options.excluded_joint_indices = list(range(6))
    return robot, solver, q, target, options


def _prime_stationary_monitor(solver, q, target, options, multiple_targets):
    for _ in range(25):
        result = (
            solver.solve_position_step(q, [eik.TaskTarget("move", target)], options)
            if multiple_targets
            else solver.solve_position_step(q, target, "move", options)
        )
        np.testing.assert_array_equal(result.q_solution, q)


@pytest.mark.parametrize("multiple_targets", [False, True])
@pytest.mark.parametrize("floating_base", [False, True])
@pytest.mark.parametrize(
    "coordinate,dt,acceleration", [(3.14, 0.001, 150.0), (10.0, 0.01, 15.0), (3.14, 0.001, 0.0)]
)
def test_continuity_brake_accepts_representable_roundtrip(
    tmp_path, multiple_targets, floating_base, coordinate, dt, acceleration
):
    robot, solver, q, target, options = _braking_solver(tmp_path, floating_base, coordinate)
    solver.dt = options.dt = dt
    solver.set_acceleration_limits(np.full(robot.nv, acceleration))
    solver.enable_acceleration_limits(True)
    _prime_stationary_monitor(solver, q, target, options, multiple_targets)
    previous = np.zeros(robot.nv)
    previous[-1] = 0.4
    solver.set_previous_joint_velocities(previous)
    result = (
        solver.solve_position_step(q, [eik.TaskTarget("move", target)], options)
        if multiple_targets
        else solver.solve_position_step(q, target, "move", options)
    )
    applied = np.asarray(robot.difference(q, result.q_solution)) / dt
    assert result.status == eik.SolverStatus.SUCCESS
    assert not result.position_step_hold_active
    assert applied[-1] == pytest.approx(0.4 - acceleration * dt, abs=1e-10)
    assert np.max(np.abs(applied - previous)) <= acceleration * dt + 1e-10
    np.testing.assert_allclose(result.joint_velocities, applied, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        solver.get_previous_joint_velocities(), applied, rtol=0.0, atol=1e-12
    )


@pytest.mark.parametrize("multiple_targets", [False, True])
@pytest.mark.parametrize("floating_base", [False, True])
def test_continuity_brake_does_not_certify_erased_motion(tmp_path, multiple_targets, floating_base):
    robot, solver, q, target, options = _braking_solver(tmp_path, floating_base, 1e15)
    solver.dt = options.dt = 0.001
    solver.set_acceleration_limits(np.full(robot.nv, 150.0))
    solver.enable_acceleration_limits(True)
    _prime_stationary_monitor(solver, q, target, options, multiple_targets)
    previous = np.zeros(robot.nv)
    previous[-1] = 0.4
    solver.set_previous_joint_velocities(previous)
    result = (
        solver.solve_position_step(q, [eik.TaskTarget("move", target)], options)
        if multiple_targets
        else solver.solve_position_step(q, target, "move", options)
    )
    # The configuration cannot represent the requested sub-ulp displacement.
    # Holding is explicit; it is not certified acceleration-limited braking.
    np.testing.assert_array_equal(result.q_solution, q)
    assert result.position_step_hold_active
    assert "decelerating" not in result.status_message
