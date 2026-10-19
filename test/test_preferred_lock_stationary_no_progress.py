"""A stalled preferred lock must allow the existing free fallback to run."""

from pathlib import Path

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("multiple_targets", [False, True])
def test_stationary_no_progress_inside_preference_tolerance_runs_fallback(
    tmp_path: Path, multiple_targets: bool
) -> None:
    urdf = tmp_path / "slider.urdf"
    urdf.write_text(
        '<robot name="slider"><link name="base"/><link name="tool"/>'
        '<joint name="slide" type="prismatic"><parent link="base"/>'
        '<child link="tool"/><axis xyz="1 0 0"/>'
        '<limit lower="-1" upper="1" velocity="1" effort="1"/>'
        "</joint></robot>"
    )
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.set_acceleration_limits(np.array([15.0]))
    solver.enable_acceleration_limits(True)
    task = solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSITION)
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    target = np.eye(4)
    target[0, 3] = 0.01
    task.set_target_pose(target[:3, 3], target[:3, :3])
    options = eik.PositionStepOptions()
    options.dt = solver.dt
    options.max_steps = 1
    options.position_gain = 10.0
    options.orientation_gain = 0.0
    options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
    options.adaptive_dt = False
    options.preferred_locked_joint_indices = [0]
    options.preferred_lock_tracking_tolerance = 0.035
    options.preferred_lock_orientation_tolerance = 0.25

    q = np.zeros(robot.nq)
    incoming = np.zeros(robot.nv)
    stalled_candidate_seen = False
    fallback = None
    # The fixed target is unreachable with the sole coordinate locked. Once
    # continuity reports that stationary attempt as NO_PROGRESS, the normal
    # free solve can move, even though the error is inside the coarse lock gate.
    for _ in range(32):
        if multiple_targets:
            result = solver.solve_position_step(
                q, [eik.TaskTarget("move", target, 10.0, 0.0)], options
            )
        else:
            result = solver.solve_position_step(q, target, "move", options)
        assert result.preferred_lock_attempted
        if result.preferred_lock_candidate_status == eik.SolverStatus.NO_PROGRESS:
            stalled_candidate_seen = True
            assert result.preferred_lock_candidate_step_norm <= 1e-12
            assert result.preferred_lock_candidate_position_error < 0.035
        applied = (np.asarray(result.q_solution) - q) / options.dt
        assert np.max(np.abs(applied - incoming)) <= 15.0 * options.dt + 1e-10
        np.testing.assert_allclose(result.joint_velocities, applied, atol=1e-12)
        np.testing.assert_allclose(solver.get_previous_joint_velocities(), applied, atol=1e-12)
        q = np.asarray(result.q_solution).copy()
        incoming = applied
        if result.preferred_lock_fallback_used:
            fallback = result
            break

    assert stalled_candidate_seen
    assert fallback is not None, "Stationary NO_PROGRESS suppressed the free solve"
    assert not fallback.preferred_lock_used
    assert fallback.status == eik.SolverStatus.SUCCESS
    assert q[0] > 1e-5
    assert abs(q[0] - target[0, 3]) < 0.01
