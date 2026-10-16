"""Coupled CoM velocity rows survive acceleration-limited command realization."""

from pathlib import Path

import numpy as np
import pytest

import embodik as eik

_URDF = """<robot name="parallel_mass_sliders">
<link name="base"/>
<link name="tool0"><inertial><origin xyz="0 0 0"/><mass value="1"/>
<inertia ixx=".001" ixy="0" ixz="0" iyy=".001" iyz="0" izz=".001"/></inertial></link>
<link name="tool1"><inertial><origin xyz="0 0 0"/><mass value="1"/>
<inertia ixx=".001" ixy="0" ixz="0" iyy=".001" iyz="0" izz=".001"/></inertial></link>
<joint name="joint0" type="prismatic"><parent link="base"/><child link="tool0"/>
<axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="1" effort="10"/></joint>
<joint name="joint1" type="prismatic"><parent link="base"/><child link="tool1"/>
<axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="1" effort="10"/></joint>
</robot>"""


@pytest.mark.parametrize("weighted_fallback", [False, True])
@pytest.mark.parametrize(
    "api", ["velocity", "single", "multi", "single_near_pin", "multi_near_pin"]
)
def test_com_rows_survive_origin_excluding_acceleration_box(
    tmp_path: Path, weighted_fallback: bool, api: str
) -> None:
    """A feasible coupled command must not become independent joint clipping.

    Equal branch masses give CoMx=(q0+q1)/2. The current and previous
    configurations have the same CoMx inside the margin-shrunk edge .95.
    Acceleration requires v0 in[.99,1], v1 in[-1,-.85]. A command such as
    [.99,-.99] satisfies that box and the original CoM rows. Independently
    clipping the requested[-1,+1] gives[.99,-.85], violating the CoM bound.
    Near-pin variants place that coupled violation at 1e-6, which must not
    disappear into the projection's numerical acceptance tolerance.
    """
    path = tmp_path / "parallel_sliders.urdf"
    path.write_text(_URDF)
    robot = eik.RobotModel(
        str(path), floating_base=False, actuated_joint_names=["joint0", "joint1"]
    )
    solver = eik.KinematicsSolver(robot)
    dt = solver.dt = 0.01
    q = np.array([1.1, 0.798])
    if api.endswith("_near_pin"):
        q[1] = 2 * (0.95 - (0.07 - 1e-6) ** 2 / 0.2) - q[0]
    previous = np.array([1.0, -1.0])
    acceleration = np.array([1.0, 15.0])
    config = solver.runtime_config()
    config.weighted_fallback_enabled = weighted_fallback
    solver.configure_runtime(config)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.set_acceleration_limits(acceleration)
    solver.enable_acceleration_limits(True)
    solver.set_previous_joint_velocities(previous)
    solver.configure_com_constraint(
        np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]),
        margin=0.05,
        frame_name="base",
        com_vel_max=0.4,
        com_acc_max=0.1,
        use_acceleration_limits=True,
        proximity_fraction=0.05,
    )
    robot.update_configuration(q - dt * previous)
    np.testing.assert_allclose(robot.get_com_position(), [q.mean(), 0, 0], atol=1e-14)
    robot.update_configuration(q)
    com = np.asarray(robot.get_com_position()).copy()
    jcom = np.asarray(robot.get_com_jacobian()).copy()
    np.testing.assert_allclose(jcom, [[0.5, 0.5], [0, 0], [0, 0]], atol=1e-14)
    # Native radial vertex shrink makes this square's edge exactly .95.
    normals = np.array([[1.0, 0], [-1.0, 0], [0, 1.0], [0, -1.0]])
    slack = 0.95 - normals @ com[:2]
    upper = np.minimum(0.4, np.sqrt(2 * 0.1 * np.maximum(slack, 0)))
    upper = np.where(slack < 0.05 * 0.95, np.minimum(upper, slack / dt), upper)
    com_rows = normals @ jcom[:2]
    witness = np.array([0.99, -0.99])
    assert np.all(np.abs(witness - previous) <= acceleration * dt + 1e-12)
    assert np.all(com_rows @ witness <= upper + 1e-12)
    assert np.all(com_rows @ witness >= -0.4 - 1e-12)

    tasks, matrices = [], []
    for index, displacement in enumerate([-0.1, 0.1]):
        task = solver.add_frame_task(f"move{index}", f"tool{index}", eik.TaskType.FRAME_POSITION)
        task.priority, task.weight = 0, 10.0
        task.solve_mode = eik.TaskSolveMode.MIN_ERROR
        task.allow_min_error_fallback = False
        matrix = np.eye(4)
        matrix[0, 3] = q[index] + displacement
        task.set_target_pose(matrix[:3, 3], matrix[:3, :3])
        tasks.append(task)
        matrices.append(matrix)

    if api == "velocity":
        tasks[0].set_target_velocity(np.array([-1.0, 0, 0]))
        tasks[1].set_target_velocity(np.array([1.0, 0, 0]))
        result = solver.solve_velocity(q, True, False)
        next_q = q + dt * np.asarray(result.joint_velocities)
    else:
        options = eik.PositionStepOptions()
        options.max_steps, options.adaptive_dt, options.dt = 1, False, dt
        options.position_gain = options.orientation_gain = 10.0
        options.max_linear_speed = options.max_angular_speed = 1.0
        options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
        options.primary_allow_min_error_fallback = False
        options.preferred_locked_joint_indices = []
        if api in ("single", "single_near_pin"):
            # The named task is assigned through the single-target API; the
            # other registered task retains the same +1 m/s Cartesian demand.
            tasks[1].set_target_velocity(np.array([1.0, 0, 0]))
            result = solver.solve_position_step(q, matrices[0], "move0", options)
        else:
            targets = [
                eik.TaskTarget(f"move{i}", matrix, 10.0, 10.0) for i, matrix in enumerate(matrices)
            ]
            result = solver.solve_position_step(q, targets, options)
        next_q = np.asarray(result.q_solution)

    applied = (next_q - q) / dt
    values = com_rows @ applied
    diagnostics = dict(
        api=api,
        weighted_fallback=weighted_fallback,
        status=result.status.name,
        message=result.status_message,
        task_scales=list(result.task_scales),
        weighted_fallback_used=bool(result.weighted_fallback_used),
        applied=applied.tolist(),
        com_rows=values.tolist(),
        com_upper=upper.tolist(),
    )
    assert result.status == eik.SolverStatus.SUCCESS, diagnostics
    assert np.isfinite(next_q).all() and np.isfinite(applied).all(), diagnostics
    assert np.all(values <= upper + 1e-9), diagnostics
    assert np.all(values >= -0.4 - 1e-9), diagnostics
    assert np.all(np.abs(applied) <= 1.0 + 1e-9), diagnostics
    assert np.all(np.abs(applied - previous) <= acceleration * dt + 1e-9), diagnostics
    assert np.all(next_q >= -2.0 - 1e-12) and np.all(next_q <= 2.0 + 1e-12)
    np.testing.assert_allclose(result.joint_velocities, applied, rtol=0, atol=1e-12)
    np.testing.assert_allclose(solver.get_previous_joint_velocities(), applied, rtol=0, atol=1e-12)


@pytest.mark.parametrize("api", ["single", "multi"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_com_proximity_realization_uses_declared_outer_interval(
    tmp_path: Path, api: str, adaptive: bool
) -> None:
    """The outer .02 s command must not use solver.dt=.01 for its CoM row."""
    path = tmp_path / "proximity_sliders.urdf"
    path.write_text(_URDF)
    robot = eik.RobotModel(
        str(path), floating_base=False, actuated_joint_names=["joint0", "joint1"]
    )
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    outer_dt = 0.02
    q = np.array([1.1, 0.799998])
    previous = np.array([1.0, -1.0])
    acceleration = np.array([1.0, 15.0])
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.set_acceleration_limits(acceleration)
    solver.enable_acceleration_limits(True)
    solver.set_previous_joint_velocities(previous)
    solver.configure_com_constraint(
        np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]),
        margin=0.05,
        frame_name="base",
        com_vel_max=0.4,
        com_acc_max=0.1,
        use_acceleration_limits=True,
        proximity_fraction=0.05,
    )
    robot.update_configuration(q)
    com = np.asarray(robot.get_com_position()).copy()
    jcom = np.asarray(robot.get_com_jacobian()).copy()
    slack = 0.95 - com[0]
    assert 0 < slack < 1.1e-6
    expected_upper = slack / outer_dt
    assert expected_upper < np.sqrt(2 * 0.1 * slack)
    matrices, tasks = [], []
    for index, displacement in enumerate([-0.1, 0.1]):
        task = solver.add_frame_task(f"move{index}", f"tool{index}", eik.TaskType.FRAME_POSITION)
        task.priority, task.weight = 0, 10.0
        task.solve_mode = eik.TaskSolveMode.MIN_ERROR
        task.allow_min_error_fallback = False
        matrix = np.eye(4)
        matrix[0, 3] = q[index] + displacement
        task.set_target_pose(matrix[:3, 3], matrix[:3, :3])
        matrices.append(matrix)
        tasks.append(task)
    options = eik.PositionStepOptions()
    options.max_steps, options.dt = 1, outer_dt
    options.adaptive_dt = adaptive
    options.adaptive_dt_max_scale = 5.0
    options.adaptive_dt_reference_distance = 0.02
    options.record_velocity_candidates = True
    options.position_gain = options.orientation_gain = 10.0
    options.max_linear_speed = options.max_angular_speed = 1.0
    options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
    options.primary_allow_min_error_fallback = False
    options.preferred_locked_joint_indices = []
    if api == "single":
        tasks[1].set_target_velocity(np.array([1.0, 0, 0]))
        result = solver.solve_position_step(q, matrices[0], "move0", options)
    else:
        result = solver.solve_position_step(
            q, [eik.TaskTarget(f"move{i}", m, 10.0, 10.0) for i, m in enumerate(matrices)], options
        )
    next_q = np.asarray(result.q_solution)
    applied = (next_q - q) / outer_dt
    com_velocity = jcom @ applied
    diagnostics = dict(
        api=api,
        adaptive=adaptive,
        status=result.status.name,
        message=result.status_message,
        applied=applied.tolist(),
        com_velocity=com_velocity.tolist(),
        expected_upper=expected_upper,
        configured_solver_dt=solver.dt,
        declared_outer_dt=outer_dt,
        inner_integration_dts=[c.integration_dt for c in result.velocity_candidates],
    )
    assert result.status == eik.SolverStatus.SUCCESS, diagnostics
    assert np.isfinite(next_q).all() and np.isfinite(applied).all(), diagnostics
    assert com_velocity[0] <= expected_upper + 1e-9, diagnostics
    assert np.all(np.abs(applied) <= 1.0 + 1e-9), diagnostics
    assert np.all(np.abs(applied - previous) <= acceleration * outer_dt + 1e-9), diagnostics
    assert np.all(next_q >= -2.0 - 1e-12) and np.all(next_q <= 2.0 + 1e-12)
    np.testing.assert_allclose(result.joint_velocities, applied, rtol=0, atol=1e-12)
    np.testing.assert_allclose(solver.get_previous_joint_velocities(), applied, rtol=0, atol=1e-12)
    assert solver.dt == 0.01, "Temporary outer-row dt must restore the configured dt"
