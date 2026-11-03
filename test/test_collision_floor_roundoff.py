"""A structural floor must not demand recovery for distance-query roundoff."""

import numpy as np
import pytest

import embodik as eik


@pytest.fixture(params=[0.2, 0.09], ids=["separated", "overlapping"])
def floor_system(tmp_path, request):
    urdf = tmp_path / "floor.urdf"
    urdf.write_text(f"""<robot name="floor"><link name="world"/>
<link name="obstacle"><collision><geometry><sphere radius="0.05"/></geometry></collision></link>
<joint name="fixed" type="fixed"><parent link="world"/><child link="obstacle"/></joint>
<link name="tool"><inertial><mass value="1"/><inertia ixx=".1" ixy="0" ixz="0" iyy=".1" iyz="0" izz=".1"/></inertial>
<collision><geometry><sphere radius="0.05"/></geometry></collision></link>
<joint name="slide" type="prismatic"><parent link="world"/><child link="tool"/>
<origin xyz="{request.param} 0 0"/><axis xyz="1 0 0"/>
<limit lower="-1" upper="1" velocity="1" effort="10"/></joint></robot>""")
    robot = eik.RobotModel(str(urdf))
    provider = eik.KinematicsSolver(robot)
    provider.dt = 0.01
    provider.configure_collision_constraint(min_distance=0.15, max_constraints=1)
    provider.set_non_worsening_collision_floor_enabled(True)
    provider.initialize_collision_recovery_floors(np.zeros(1))
    provider.enable_acceleration_limits(True)
    provider.set_acceleration_limits(np.array([15.0]))
    provider.set_previous_joint_velocities(np.zeros(1))
    options = eik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.array([15.0])
    lift = eik.VelocityCollisionLiftOptions()
    lift.max_refinement_iterations = 0
    lift.validation_substeps = 4
    return robot, provider, options, lift


@pytest.mark.parametrize(
    "offset,accepted", [(0.0, True), (-3e-16, True), (-1e-12, False), (-1e-6, False), (1e-6, True)]
)
def test_floor_roundoff_preserves_rest_but_material_violation_requires_recovery(
    floor_system, offset, accepted
):
    robot, provider, options, lift = floor_system
    initial = provider.evaluate_collision_pair_distances(np.zeros(1))
    assert len(initial) == 1
    floor = initial[0].persistent_floor_m
    q = np.array([offset])
    v = np.zeros(1)
    before = provider.evaluate_collision_pair_distances(q)[0]
    if offset < 0:
        assert before.distance < floor
    result = eik.AccelerationSolver(robot).validate_candidate_with_velocity_collision(
        provider, q, v, 0.01, np.zeros(1), options, lift
    )
    assert (result.status == eik.SolverStatus.SUCCESS) == accepted, result.status_message
    assert provider.evaluate_collision_pair_distances(q)[0].persistent_floor_m == floor
    if accepted:
        np.testing.assert_array_equal(result.q_solution, q)
        np.testing.assert_array_equal(result.joint_velocities_next, v)
        np.testing.assert_array_equal(result.joint_accelerations, np.zeros(1))
    else:
        assert np.asarray(result.q_solution).size == 0


def test_roundoff_rule_does_not_hide_an_ordinary_nominal_clearance_violation(floor_system):
    robot, provider, options, lift = floor_system
    provider.set_non_worsening_collision_floor_enabled(False)
    result = eik.AccelerationSolver(robot).validate_candidate_with_velocity_collision(
        provider, np.zeros(1), np.zeros(1), 0.01, np.zeros(1), options, lift
    )
    assert result.status != eik.SolverStatus.SUCCESS
    assert np.asarray(result.q_solution).size == 0
