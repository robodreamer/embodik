"""A safe step entering pose tolerance must not become a continuity hold."""

from pathlib import Path

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("initially_satisfied", [False, True])
@pytest.mark.parametrize("multiple_targets", [False, True])
def test_newly_satisfied_pose_is_published_before_stationary_hold(
    tmp_path: Path, multiple_targets: bool, initially_satisfied: bool
) -> None:
    urdf = tmp_path / "completion_slider.urdf"
    urdf.write_text(
        '<robot name="completion_slider"><link name="base"/><link name="tool"/>'
        '<joint name="slide" type="prismatic"><parent link="base"/>'
        '<child link="tool"/><axis xyz="1 0 0"/>'
        '<limit lower="-1" upper="1" effort="10" velocity="1"/></joint></robot>'
    )
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    cfg = solver.runtime_config()
    cfg.enable_auto_task_layout = False
    cfg.weighted_fallback_enabled = False
    cfg.normalized_pose_recovery_enabled = True
    solver.configure_runtime(cfg)
    task = solver.add_frame_task("goal", "tool", eik.TaskType.FRAME_POSE)
    task.solve_mode = eik.TaskSolveMode.SCALE
    task.allow_min_error_fallback = True
    target = np.eye(4)
    target[0, 3] = 0.0 if initially_satisfied else 0.04
    task.set_target_pose(target[:3, 3], target[:3, :3])
    opts = eik.PositionStepOptions()
    opts.dt = 0.01
    opts.max_steps = 1
    opts.adaptive_dt = False
    opts.limit_change_from_seed = True
    opts.position_gain = opts.orientation_gain = 10.0
    opts.primary_solve_mode = eik.TaskSolveMode.SCALE
    opts.primary_allow_min_error_fallback = True
    q = np.zeros(robot.nq)
    crossed = False
    results = []
    for _ in range(60):
        before = float(target[0, 3] - q[0])
        if multiple_targets:
            result = solver.solve_position_step(
                q, [eik.TaskTarget("goal", target, 10.0, 10.0)], opts
            )
        else:
            result = solver.solve_position_step(q, target, "goal", opts)
        after_q = np.asarray(result.q_solution).copy()
        after = float(target[0, 3] - after_q[0])
        # The unconstrained scalar command is a monotone contraction. At call
        # 20 it enters 5 mm, coinciding with the stationary monitoring window.
        # Holding that candidate leaves an unsatisfied seed despite safe progress.
        if before > 0.005:
            assert result.status == eik.SolverStatus.SUCCESS, (
                len(results) + 1,
                before,
                after,
                result.status_message,
            )
            assert not result.position_step_hold_active
            assert after < before
        if before > 0.005 and after <= 0.005:
            crossed = True
            assert np.linalg.norm(after_q - q) > 0.0
        assert np.isfinite(after_q).all()
        assert np.max(np.abs(after_q - q)) <= 0.01 + 1e-12
        q = after_q
        results.append(result)
    assert crossed == (not initially_satisfied)
    if initially_satisfied:
        np.testing.assert_array_equal(q, np.zeros(robot.nq))
    assert results[-1].status == eik.SolverStatus.SUCCESS
    assert results[-1].position_step_hold_active
    assert "held satisfied stationary target" in results[-1].status_message
