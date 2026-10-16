"""Roundoff at a braking face must not create false physical infeasibility."""

from pathlib import Path

import numpy as np
import pytest

import embodik as eik

_URDF = """<robot name="braking_face_sliders"><link name="base"/>
<link name="bounded"/><link name="moving"/>
<joint name="bounded_joint" type="prismatic"><parent link="base"/><child link="bounded"/>
<axis xyz="1 0 0"/><limit lower="0" upper="1" velocity="1" effort="10"/></joint>
<joint name="moving_joint" type="prismatic"><parent link="base"/><child link="moving"/>
<axis xyz="1 0 0"/><limit lower="0" upper="1" velocity="1" effort="10"/></joint>
</robot>"""


@pytest.mark.parametrize("multi_target", [False, True], ids=["single", "multi"])
@pytest.mark.parametrize("upper_face", [False, True], ids=["lower", "upper"])
@pytest.mark.parametrize("real_gap", [False, True], ids=["roundoff", "real_conflict"])
def test_braking_face_roundoff_retains_feasible_cartesian_motion(
    tmp_path: Path, multi_target: bool, upper_face: bool, real_gap: bool
) -> None:
    path = tmp_path / "braking_face_sliders.urdf"
    path.write_text(_URDF)
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    dt = solver.dt = 0.01
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.set_acceleration_limits(np.full(2, 15.0))
    solver.enable_acceleration_limits(True)
    direction = 1.0 if upper_face else -1.0
    speed = 0.15 + 1e-9 if real_gap else 0.15000000000000013
    incoming = np.array([direction * speed, 0.0])
    solver.set_previous_joint_velocities(incoming)
    q = np.array([0.9999 if upper_face else 0.0001, 0.5])
    robot.update_configuration(q)
    targets = []
    for name, position in [("bounded", q[0]), ("moving", 0.6)]:
        task = solver.add_frame_task(name, name, eik.TaskType.FRAME_POSITION)
        task.priority = 0
        task.solve_mode = eik.TaskSolveMode.SCALE
        task.allow_min_error_fallback = False
        pose = np.eye(4)
        pose[0, 3] = position
        task.set_target_pose(pose[:3, 3], np.eye(3))
        targets.append(eik.TaskTarget(name, pose, 10.0, 0.0))
    options = eik.PositionStepOptions()
    options.dt, options.max_steps, options.adaptive_dt = dt, 1, False
    options.primary_solve_mode = eik.TaskSolveMode.SCALE
    options.primary_allow_min_error_fallback = False
    options.position_gain, options.orientation_gain = 10.0, 0.0
    options.limit_change_from_seed = True
    runtime = solver.runtime_config()
    runtime.weighted_fallback_enabled = False
    solver.configure_runtime(runtime)

    # A previous applied velocity can differ from one braking increment by a
    # few ulps after manifold integration. Zero is on the physical inset face.
    step = 15.0 * dt
    assert abs(incoming[0]) > step
    witness = np.array([0.0, step])
    roundoff = 64 * np.finfo(float).eps
    if real_gap:
        assert abs(incoming[0]) - step > roundoff
    else:
        assert np.all(np.abs(witness - incoming) <= step + roundoff)
    if multi_target:
        result = solver.solve_position_step(q, targets, options)
    else:
        # The registered boundary task remains active with a zero demand.
        result = solver.solve_position_step(q, targets[1].target_pose, "moving", options)

    if real_gap:
        # The inset stopping boundary and incoming acceleration interval have
        # a real gap. Rejected diagnostics must not advance physical history.
        assert result.status == eik.SolverStatus.INFEASIBLE, result.status_message
        # A failed result can retain a moving diagnostic configuration; its
        # status forbids execution, and it must not advance accepted history.
        assert np.isfinite(result.q_solution).all()
        assert np.all(np.asarray(result.q_solution) >= 0.0)
        assert np.all(np.asarray(result.q_solution) <= 1.0)
        np.testing.assert_array_equal(solver.get_previous_joint_velocities(), incoming)
        return
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    applied = (np.asarray(result.q_solution) - q) / dt
    assert applied[0] == pytest.approx(0.0, abs=1e-13)
    assert applied[1] > 0.01
    assert np.all(np.abs(applied - incoming) <= step + roundoff)
    assert np.all(np.asarray(result.q_solution) >= 0.0)
    assert np.all(np.asarray(result.q_solution) <= 1.0)
    np.testing.assert_allclose(result.joint_velocities, applied, rtol=0, atol=1e-13)
    np.testing.assert_allclose(solver.get_previous_joint_velocities(), applied, rtol=0, atol=1e-13)
