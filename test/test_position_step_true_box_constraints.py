"""True scalar endpoint bounds constrain acceleration-limited position solves."""

from pathlib import Path

import numpy as np
import pytest

import embodik as eik


def _solver(path: Path, urdf: str) -> tuple[eik.RobotModel, eik.KinematicsSolver]:
    path.write_text(urdf)
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.set_acceleration_limits(np.full(robot.nv, 15.0))
    solver.enable_acceleration_limits(True)
    solver.set_previous_joint_velocities(np.zeros(robot.nv))
    runtime = solver.runtime_config()
    runtime.weighted_fallback_enabled = False
    solver.configure_runtime(runtime)
    return robot, solver


def _options() -> eik.PositionStepOptions:
    options = eik.PositionStepOptions()
    options.max_steps, options.adaptive_dt, options.dt = 1, False, 0.01
    options.position_gain = options.orientation_gain = 10.0
    options.max_linear_speed = options.max_angular_speed = 1.0
    options.primary_solve_mode = eik.TaskSolveMode.SCALE_ELASTIC
    options.primary_allow_min_error_fallback = False
    options.record_velocity_candidates = True
    return options


def _task(solver: eik.KinematicsSolver, name: str, kind: eik.TaskType) -> None:
    task = solver.add_frame_task(name, name, kind)
    task.priority = 0
    task.solve_mode = eik.TaskSolveMode.SCALE_ELASTIC
    task.allow_min_error_fallback = False


@pytest.mark.parametrize("multi_target", [False, True], ids=["single", "multi"])
def test_contradictory_acceleration_and_true_endpoint_box_is_rejected(
    tmp_path: Path, multi_target: bool
) -> None:
    robot, solver = _solver(
        tmp_path / "one_slider.urdf",
        """<robot name="one_slider"><link name="base"/><link name="tool"/>
<joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/>
<axis xyz="1 0 0"/><limit lower="0" upper="1" velocity="1" effort="1"/></joint></robot>""",
    )
    q = np.array([0.0001])
    previous = np.array([-0.4])
    solver.set_previous_joint_velocities(previous)
    _task(solver, "tool", eik.TaskType.FRAME_POSITION)
    target = np.eye(4)
    target[0, 3] = -0.1
    # True endpoint demands v>=-.01, while acceleration requires v<=-.25.
    assert (0.0 - q[0]) / solver.dt > previous[0] + 15.0 * solver.dt
    if multi_target:
        result = solver.solve_position_step(
            q, [eik.TaskTarget("tool", target, 10.0, 10.0)], _options()
        )
    else:
        result = solver.solve_position_step(q, target, "tool", _options())
    assert result.status == eik.SolverStatus.INFEASIBLE, result.status_message
    assert result.status_message
    assert np.isfinite(result.q_solution).all()
    lower, upper = map(np.asarray, robot.get_joint_limits())
    assert np.all(result.q_solution >= lower - 1e-12)
    assert np.all(result.q_solution <= upper + 1e-12)
    np.testing.assert_allclose(solver.get_previous_joint_velocities(), previous, rtol=0, atol=0)
    # A held failure vector is diagnostic; it is not a feasible braking command
    # and must not be used as evidence that this origin-excluding box is safe.


@pytest.mark.parametrize("multi_target", [False, True], ids=["single", "multi"])
def test_true_box_uses_scalar_mapping_without_bounding_continuous_coordinates(
    tmp_path: Path, multi_target: bool
) -> None:
    robot, solver = _solver(
        tmp_path / "continuous_and_scalar.urdf",
        """<robot name="continuous_and_scalar"><link name="base"/><link name="spin"/><link name="slide"/>
<joint name="turn" type="continuous"><parent link="base"/><child link="spin"/>
<axis xyz="0 0 1"/><limit velocity="1" effort="1"/></joint>
<joint name="translate" type="prismatic"><parent link="base"/><child link="slide"/>
<axis xyz="1 0 0"/><limit lower="0" upper="1" velocity="1" effort="1"/></joint></robot>""",
    )
    assert robot.nq == 3 and robot.nv == 2
    q = np.asarray(robot.neutral_configuration()).copy()
    scalar_q = robot.get_joint_config_index("translate")
    continuous_q = robot.get_joint_config_index("turn")
    q[scalar_q] = 0.0001
    _task(solver, "spin", eik.TaskType.FRAME_ORIENTATION)
    _task(solver, "slide", eik.TaskType.FRAME_POSITION)
    rotation = np.eye(3)
    angle = 0.1
    rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    spin_target = np.eye(4)
    spin_target[:3, :3] = rotation
    slide_target = np.eye(4)
    slide_target[0, 3] = 0.1
    solver.get_task("slide").set_target_pose(slide_target[:3, 3], np.eye(3))
    if multi_target:
        result = solver.solve_position_step(
            q,
            [
                eik.TaskTarget("spin", spin_target, 10.0, 10.0),
                eik.TaskTarget("slide", slide_target, 10.0, 10.0),
            ],
            _options(),
        )
    else:
        solver.get_task("slide").set_target_velocity(np.array([1.0, 0.0, 0.0]))
        result = solver.solve_position_step(q, spin_target, "spin", _options())
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    next_q = np.asarray(result.q_solution)
    assert np.linalg.norm(next_q[continuous_q : continuous_q + 2]) == pytest.approx(1.0, abs=1e-12)
    assert np.arctan2(next_q[continuous_q + 1], next_q[continuous_q]) > 1e-8
    assert q[scalar_q] < next_q[scalar_q] <= 1.0
    assert result.joint_velocities.shape == (2,)
    assert np.max(np.abs(result.joint_velocities)) <= 0.15 + 1e-9
