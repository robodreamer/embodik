"""Stationary minimum-error compromises must not sacrifice the worst target."""

import numpy as np

import embodik as eik


def test_min_error_worst_target_hold_reopens_when_commands_return(tmp_path):
    urdf = tmp_path / "shared_slide.urdf"
    urdf.write_text("""<robot name="shared_slide"><link name="base"/><link name="tool"/>
<joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/>
<axis xyz="1 0 0"/><limit lower="-3" upper="3" velocity="1" effort="1"/></joint>
</robot>""")
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    runtime = solver.runtime_config()
    runtime.enable_auto_task_layout = False
    runtime.weighted_fallback_enabled = False
    solver.configure_runtime(runtime)
    for name in ("worse", "easier"):
        task = solver.add_frame_task(name, "tool", eik.TaskType.FRAME_POSITION)
        task.priority = 0
        task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
    worse, easier = np.eye(4), np.eye(4)
    worse[0, 3], easier[0, 3] = 2.0, -1.0
    targets = [
        eik.TaskTarget("worse", worse, 0.1, 0.0),
        eik.TaskTarget("easier", easier, 10.0, 0.0),
    ]
    q = np.zeros(1)
    configurations = []
    for _ in range(5):
        result = solver.solve_position_step(q, targets, options)
        q = np.asarray(result.q_solution).copy()
        configurations.append(q)
    # The first compromise may move; subsequent stationary proposals that
    # worsen the already worst target must hold before the 20-call window.
    np.testing.assert_allclose(configurations[-1], configurations[1], atol=1e-12)
    assert abs(q[0]) < 0.03

    returned = [eik.TaskTarget(name, np.eye(4), 10.0, 0.0) for name in ("worse", "easier")]
    initial_error = abs(q[0])
    assert initial_error > 1e-6
    result = solver.solve_position_step(q, returned, options)
    q = np.asarray(result.q_solution).copy()
    assert abs(q[0]) < initial_error
    for _ in range(79):
        result = solver.solve_position_step(q, returned, options)
        q = np.asarray(result.q_solution).copy()
    assert abs(q[0]) < min(5e-4, 0.05 * initial_error)
