"""The strongest sampled cut must be selected within the fixed refinement budget."""

from pathlib import Path

import numpy as np

import embodik as eik


def test_refinement_selects_strongest_later_sample_with_fixed_budget(tmp_path: Path) -> None:
    path = tmp_path / "vertical_slider.urdf"
    path.write_text('''<robot name="vertical_slider">
<link name="base"><collision><origin xyz="0 0 -.01"/>
<geometry><box size="2 2 .02"/></geometry></collision></link>
<link name="slider"><inertial><mass value="1"/>
<inertia ixx=".1" ixy="0" ixz="0" iyy=".1" iyz="0" izz=".1"/></inertial>
<collision><geometry><sphere radius=".1"/></geometry></collision></link>
<joint name="height" type="prismatic"><parent link="base"/><child link="slider"/>
<origin xyz="0 0 .1"/><axis xyz="0 0 1"/>
<limit lower="-1" upper="1" velocity="1" effort="100"/></joint></robot>''')
    robot = eik.RobotModel(str(path), floating_base=False)
    q, v, dt = np.array([.035023]), np.array([-.012]), .01
    solver = eik.AccelerationSolver(robot)
    task = solver.add_posture_task("acceleration")
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    task.allow_min_error_fallback = False
    task.weight = 1.
    task.set_target_configuration(q)
    reference = eik.AccelerationTaskReference()
    reference.proportional_gain = reference.derivative_gain = 0.
    reference.desired_acceleration = np.array([2.08])
    solver.set_task_reference("acceleration", reference)
    provider = eik.KinematicsSolver(robot)
    provider.dt = dt
    provider.configure_collision_constraint(.035, [], [], True, 1)
    provider.set_proximity_gated_collision_activation_enabled(False)
    provider.set_collision_tuning_mode(eik.CollisionTuningMode.PRECISE)
    provider.set_acceleration_limits(np.array([15.]))
    provider.enable_acceleration_limits(True)
    provider.set_previous_joint_velocities(v)
    options = eik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.array([15.])
    options.prefer_direct_min_error_qp = True
    lift = eik.VelocityCollisionLiftOptions()
    lift.validation_substeps = 4
    lift.max_refinement_iterations = 2
    # At the feasible unrefined 2.08 reference, analytic floor gaps are
    # -0.5, -11, -8.5 and +7 micrometres. The first rejected time needs
    # +0.48 acceleration, while the strongest later time needs +0.96.
    # Accumulated local trust constraints exhaust the same two-attempt
    # budget after the weaker first cut. A single stronger cut must recover
    # within the existing trust-only retry and full physical validation.
    assert lift.refinement_trust_radius == .5
    assert lift.refinement_clearance_reserve == 1e-6
    result = solver.solve_with_velocity_collision(provider, q, v, dt, options, lift)
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    assert result.collision_refinement_iterations <= 2
    acceleration = np.asarray(result.joint_accelerations)
    assert abs(acceleration[0]) <= 15. + 1e-8
    for fraction in (0., .25, .5, .75, 1.):
        time = fraction * dt
        # Sphere bottom minus plane top: exact analytic geometry.
        clearance = q + time * v + .5 * time**2 * acceleration
        assert clearance[0] >= .035 - 1e-8
    np.testing.assert_allclose(result.q_solution, q + dt*v + .5*dt**2*acceleration, atol=1e-14)
    np.testing.assert_allclose(result.joint_velocities_next, v + dt*acceleration, atol=1e-14)
