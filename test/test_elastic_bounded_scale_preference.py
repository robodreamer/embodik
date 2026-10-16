"""Prefer an executable Cartesian ray to clipping an expanded-band realization."""

import numpy as np
import pytest

import embodik as eik


def make_solver(tmp_path, direction, *, posture=False, duplicate=False):
    lower, upper = (-2, 0) if direction > 0 else (0, 2)
    urdf = tmp_path / "coupled_slides.urdf"
    urdf.write_text(f"""<robot name="coupled_slides"><link name="base"/>
<link name="diagonal"/><link name="horizontal"/><link name="tool"/>
<joint name="first" type="prismatic"><parent link="base"/><child link="diagonal"/>
<axis xyz="1 1 0"/><limit lower="{lower}" upper="{upper}" velocity="1" effort="1"/></joint>
<joint name="second" type="prismatic"><parent link="diagonal"/><child link="horizontal"/>
<axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="0.1" effort="1"/></joint>
<joint name="third" type="prismatic"><parent link="horizontal"/><child link="tool"/>
<axis xyz="0 1 0"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
</robot>""")
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    runtime = solver.runtime_config()
    runtime.refine_partial_scale = True
    runtime.weighted_fallback_enabled = False
    solver.configure_runtime(runtime)
    tasks = [solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSE)]
    if duplicate:
        tasks.append(solver.add_frame_task("duplicate", "tool", eik.TaskType.FRAME_POSE))
    if posture:
        bias = solver.add_posture_task("bias")
        bias.priority = 2
        bias.set_target_configuration(direction * np.array([0.5, 0.3, -0.3]))
    return robot, solver, tasks


def solve(solver, tasks, target, *, multiple, mode, locked=()):
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.position_gain = options.orientation_gain = 10.0
    options.primary_solve_mode = mode
    options.primary_allow_min_error_fallback = False
    options.record_velocity_candidates = True
    options.locked_joint_indices = list(locked)
    if multiple:
        return solver.solve_position_step(
            np.zeros(3),
            [eik.TaskTarget(task.name, target, 10.0, 10.0) for task in tasks],
            options,
        )
    return solver.solve_position_step(np.zeros(3), target, tasks[0].name, options)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("multiple,duplicate", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("posture", [False, True])
def test_elastic_prefers_positive_true_box_ray(tmp_path, direction, multiple, duplicate, posture):
    robot, solver, tasks = make_solver(tmp_path, direction, posture=posture, duplicate=duplicate)
    target = np.eye(4)
    target[0, 3] = direction * 0.1
    result = solve(solver, tasks, target, multiple=multiple, mode=eik.TaskSolveMode.SCALE_ELASTIC)
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    candidate = result.velocity_candidates[0]
    # x_dot = first/sqrt(2) + second. The boundary forbids outward first,
    # so the maximum positive SCALE is 0.1, realized by [0, +/-0.1, 0].
    assert result.task_scales[0] == pytest.approx(0.1, abs=1e-8)
    assert result.task_modes_effective[0] == eik.TaskSolveMode.SCALE
    assert not result.task_used_fallback[0]
    if posture:
        assert result.task_scales[1] == pytest.approx(0, abs=1e-8)
        assert result.task_modes_effective[1] == eik.TaskSolveMode.SCALE
    np.testing.assert_allclose(candidate.integrated_velocity, [0, direction * 0.1, 0], atol=1e-8)
    np.testing.assert_allclose(candidate.solver_velocity, candidate.integrated_velocity, atol=1e-8)
    robot.update_configuration(np.asarray(result.q_solution))
    pose = robot.get_frame_pose("tool")
    np.testing.assert_allclose(pose.translation, [direction * 0.001, 0, 0], atol=1e-10)
    assert np.linalg.norm(target[:3, 3] - pose.translation) < 0.1


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.SCALE_ELASTIC])
def test_zero_true_box_ray_retains_elastic_closest_motion(tmp_path, direction, multiple, mode):
    robot, solver, tasks = make_solver(tmp_path, direction)
    target = np.eye(4)
    target[:2, 3] = direction * np.array([0.1, -0.1])
    result = solve(solver, tasks, target, multiple=multiple, mode=mode, locked=[1])
    q = np.asarray(result.q_solution)
    # Locking second makes any positive x ray impossible. Moving only third
    # toward the requested y velocity is the closest attainable Cartesian step.
    if mode == eik.TaskSolveMode.SCALE:
        assert result.task_scales[0] == pytest.approx(0, abs=1e-10)
        np.testing.assert_allclose(q, 0, atol=1e-10)
    else:
        assert result.status == eik.SolverStatus.SUCCESS, result.status_message
        np.testing.assert_allclose(q, [0, 0, -direction * 0.01], atol=1e-10)
        robot.update_configuration(q)
        assert np.linalg.norm(target[:3, 3] - robot.get_frame_pose("tool").translation) < np.sqrt(
            0.02
        )


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("multiple", [False, True])
def test_preference_keeps_original_velocity_override(tmp_path, direction, multiple):
    robot, solver, tasks = make_solver(tmp_path, direction, posture=True)
    # The added true execution box uses model limits. The original hard rows
    # must also survive the preference solve to enforce this smaller override.
    solver.set_joint_velocity_limit(1, 0.04)
    target = np.eye(4)
    target[0, 3] = direction * 0.1
    result = solve(solver, tasks, target, multiple=multiple, mode=eik.TaskSolveMode.SCALE_ELASTIC)
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    assert result.task_scales[0] == pytest.approx(0.04, abs=1e-8)
    candidate = result.velocity_candidates[0]
    np.testing.assert_allclose(candidate.integrated_velocity, [0, direction * 0.04, 0], atol=1e-8)
    np.testing.assert_allclose(candidate.integrated_velocity, candidate.solver_velocity, atol=1e-8)
