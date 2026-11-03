"""Explicit posture recovery cannot change hard locks or persistent task modes."""

import numpy as np
import pytest
import embodik as eik


def test_posture_retry_is_explicit_and_default_off():
    options = eik.PositionStepOptions()
    assert not options.allow_best_effort_posture_retry
    options.allow_best_effort_posture_retry = True
    assert options.allow_best_effort_posture_retry


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("mixed_band", [False, True])
@pytest.mark.parametrize("target_position", [0.0, 0.5])
@pytest.mark.parametrize("posture_priority", [0, 2])
def test_failed_posture_retry_keeps_lock_and_requested_modes(
    tmp_path, enabled, posture_priority, mixed_band, target_position
):
    path = tmp_path / "slider.urdf"
    path.write_text("""<robot name="slider"><link name="world"/><link name="tool"/>
<joint name="slide" type="prismatic"><parent link="world"/><child link="tool"/>
<axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="100" velocity="1"/></joint>
</robot>""")
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    runtime = solver.runtime_config()
    runtime.weighted_fallback_enabled = False
    solver.configure_runtime(runtime)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    frame = solver.add_frame_task("reach", "tool", eik.TaskType.FRAME_POSITION)
    posture = solver.add_posture_task("posture")
    posture.priority, posture.solve_mode = posture_priority, eik.TaskSolveMode.SCALE
    posture.set_target_configuration(np.array([0.5]))
    if mixed_band:
        other = solver.add_frame_task("same_band", "tool", eik.TaskType.FRAME_POSITION)
        other.priority = posture_priority
        other.set_target_pose(np.zeros(3), np.eye(3))
    target = np.eye(4)
    target[0, 3] = target_position
    options = eik.PositionStepOptions()
    options.allow_best_effort_posture_retry = enabled
    options.locked_joint_indices = [0]
    options.stall_recovery = False
    options.max_steps = 1
    solver.enable_problem_tracing(True)
    options.allow_best_effort_posture_retry = False
    solver.solve_position_step(np.zeros(1), [eik.TaskTarget("reach", target, 1.0, 0.0)], options)
    baseline_problems = len(solver.get_last_problem_trace().velocity_problems)
    options.allow_best_effort_posture_retry = enabled
    result = solver.solve_position_step(
        np.zeros(1), [eik.TaskTarget("reach", target, 1.0, 0.0)], options
    )
    problems = len(solver.get_last_problem_trace().velocity_problems)
    if enabled and posture_priority > 0 and not mixed_band:
        assert problems > baseline_problems  # Failed retry is recorded, then rolled back.
    else:
        assert problems == baseline_problems  # No trial in protected or mixed bands.
    np.testing.assert_array_equal(result.q_solution, np.zeros(1))
    assert frame.solve_mode == eik.TaskSolveMode.SCALE
    assert posture.solve_mode == eik.TaskSolveMode.SCALE
    assert not posture.used_min_error_fallback
