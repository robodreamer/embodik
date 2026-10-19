"""A feasible CoM-constrained command must survive changing path geometry."""

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.MIN_ERROR])
def test_com_curvature_retains_productive_feasible_acceleration(tmp_path, mode):
    urdf = tmp_path / "com_pendulum.urdf"
    urdf.write_text("""<robot name="com_pendulum"><link name="base"/>
<link name="body"><inertial><origin xyz="0 1 0"/><mass value="1"/>
<inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
</inertial></link><joint name="pivot" type="revolute">
<parent link="base"/><child link="body"/><axis xyz="0 0 1"/>
<limit lower="-2" upper="2" velocity="10" effort="100"/>
</joint></robot>""")
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.AccelerationSolver(robot)
    task = solver.add_joint_task("move", "pivot", 0.0)
    task.solve_mode = mode
    task.allow_min_error_fallback = False
    reference = eik.AccelerationTaskReference()
    reference.proportional_gain = reference.derivative_gain = 0.0
    reference.desired_acceleration = np.array([2.0])
    solver.set_task_reference("move", reference)
    definition = eik.ComSupportPolygonConstraintDefinition()
    definition.support_polygon = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])
    definition.frame_name = "base"
    definition.margin = 0.0
    policy = eik.ComSupportPolygonAccelerationPolicy()
    policy.rate_limit = 10.0
    policy.acceleration_limit = policy.braking_acceleration = 1.0
    constraint = eik.ComSupportPolygonAccelerationConstraint()
    constraint.source_id = "pendulum-support"
    constraint.definition, constraint.policy = definition, policy
    options = eik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.array([10.0])
    options.com_support_polygon_constraints = [constraint]
    q, dq, dt = np.array([-0.5]), np.zeros(1), 0.01

    def accelerations(acceleration):
        times = np.linspace(0.0, dt, 101)
        angles = q[0] + 0.5 * times**2 * acceleration
        rates = times * acceleration
        return np.array(
            [
                -np.cos(angles) * acceleration + np.sin(angles) * rates**2,
                -np.sin(angles) * acceleration - np.cos(angles) * rates**2,
            ]
        )

    # An analytic, productive constant-acceleration witness exists. The exact
    # initial linear face instead becomes unsafe as the pendulum moves.
    assert np.max(np.abs(accelerations(1.12))) < 1.0
    assert np.max(np.abs(accelerations(1.0 / np.cos(q[0])))) > 1.0 + 1e-8
    result = solver.solve(q, dq, dt, options)
    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    acceleration = float(result.joint_accelerations[0])
    assert 1.12 <= acceleration <= 2.0
    assert np.max(np.abs(accelerations(acceleration))) <= 1.0 + 1e-8
