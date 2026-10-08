"""Fresh collision-pair queries preserve solver state and pair-specific margins."""

import numpy as np
import pytest

import embodik

SPHERE_RADIUS = 0.05
MOVING_ORIGIN = 0.2
FAR_ORIGIN = 0.6
GLOBAL_MINIMUM = 0.02
PAIR_MINIMUM = 0.15
QUERY_DISPLACEMENT = 0.03
INITIAL_VELOCITY = 0.1


@pytest.fixture
def pair_solver(tmp_path):
    urdf = tmp_path / "pair_query.urdf"
    urdf.write_text(f"""<robot name="pair_query"><link name="world"/>
<link name="near"><collision><geometry><sphere radius="{SPHERE_RADIUS}"/></geometry></collision></link>
<joint name="near_fixed" type="fixed"><parent link="world"/><child link="near"/></joint>
<link name="far"><collision><geometry><sphere radius="{SPHERE_RADIUS}"/></geometry></collision></link>
<joint name="far_fixed" type="fixed"><parent link="world"/><child link="far"/><origin xyz="{FAR_ORIGIN} 0 0"/></joint>
<link name="moving"><collision><geometry><sphere radius="{SPHERE_RADIUS}"/></geometry></collision></link>
<joint name="slide" type="prismatic"><parent link="world"/><child link="moving"/>
<origin xyz="{MOVING_ORIGIN} 0 0"/><axis xyz="1 0 0"/>
<limit lower="0" upper="1" effort="1" velocity="1"/></joint></robot>""")
    robot = embodik.RobotModel(str(urdf))
    solver = embodik.KinematicsSolver(robot)
    return robot, solver


def test_query_returns_all_pairs_with_their_own_threshold_and_restores_configuration(pair_solver):
    robot, solver = pair_solver
    solver.configure_collision_constraint(min_distance=GLOBAL_MINIMUM, max_constraints=1)
    solver.set_collision_pair_min_distance("far", "moving", PAIR_MINIMUM, activate_when_clear=False)
    robot.update_kinematics(robot.get_current_configuration(), np.array([INITIAL_VELOCITY]))
    before = robot.get_current_configuration().copy()
    velocity_before = robot.get_current_velocity().copy()
    overrides_before = solver.get_collision_pair_min_distance_overrides()
    records = solver.evaluate_collision_pair_distances(np.array([QUERY_DISPLACEMENT]))
    assert len(records) == len(solver.get_active_collision_pairs())
    assert len(records) > 1  # The query is independent of the one-row solver budget.
    for record in records:
        pair = {record.geometry_a, record.geometry_b}
        expected_minimum = (
            PAIR_MINIMUM
            if any("far" in name for name in pair) and any("moving" in name for name in pair)
            else GLOBAL_MINIMUM
        )
        assert record.minimum_distance == pytest.approx(expected_minimum)
        assert np.isfinite(record.distance)
    moving_near = next(
        record
        for record in records
        if "near" in record.geometry_a + record.geometry_b
        and "moving" in record.geometry_a + record.geometry_b
    )
    assert moving_near.distance == pytest.approx(
        MOVING_ORIGIN + QUERY_DISPLACEMENT - 2 * SPHERE_RADIUS
    )
    np.testing.assert_array_equal(robot.get_current_configuration(), before)
    np.testing.assert_array_equal(robot.get_current_velocity(), velocity_before)
    assert solver.get_collision_pair_min_distance_overrides() == overrides_before
    second = solver.evaluate_collision_pair_distances(before)
    second_near = next(
        record
        for record in second
        if {record.geometry_a, record.geometry_b}
        == {moving_near.geometry_a, moving_near.geometry_b}
    )
    assert moving_near.distance - second_near.distance == pytest.approx(QUERY_DISPLACEMENT)


def test_query_obeys_include_exclude_and_unavailable_policy(pair_solver):
    robot, solver = pair_solver
    configuration = robot.get_current_configuration()
    assert solver.evaluate_collision_pair_distances(configuration) is None
    solver.configure_collision_constraint(min_distance=GLOBAL_MINIMUM)
    pair = solver.get_active_collision_pairs()[0]
    solver.configure_collision_constraint(min_distance=GLOBAL_MINIMUM, include_pairs=[pair])
    records = solver.evaluate_collision_pair_distances(configuration)
    assert len(records) == 1
    assert {records[0].geometry_a, records[0].geometry_b} == set(pair)
    solver.configure_collision_constraint(
        min_distance=GLOBAL_MINIMUM, include_pairs=[pair], exclude_pairs=[pair]
    )
    assert solver.evaluate_collision_pair_distances(configuration) is None
    solver.clear_collision_constraint()
    assert solver.evaluate_collision_pair_distances(configuration) is None


@pytest.mark.parametrize(
    "configuration", [np.array([]), np.zeros(2), np.array([np.nan]), np.array([np.inf])]
)
def test_query_rejects_invalid_configuration_without_mutation(pair_solver, configuration):
    robot, solver = pair_solver
    before = robot.get_current_configuration().copy()
    solver.configure_collision_constraint(min_distance=GLOBAL_MINIMUM)
    with pytest.raises((ValueError, RuntimeError), match="configuration|Configuration"):
        solver.evaluate_collision_pair_distances(configuration)
    np.testing.assert_array_equal(robot.get_current_configuration(), before)


def test_fresh_query_does_not_activate_deferred_override(pair_solver):
    robot, solver = pair_solver
    solver.configure_collision_constraint(min_distance=GLOBAL_MINIMUM)
    solver.set_collision_pair_min_distance("far", "moving", PAIR_MINIMUM, activate_when_clear=True)
    records = solver.evaluate_collision_pair_distances(robot.get_current_configuration())
    far_moving = next(
        record
        for record in records
        if "far" in record.geometry_a + record.geometry_b
        and "moving" in record.geometry_a + record.geometry_b
    )
    assert far_moving.distance > PAIR_MINIMUM
    assert far_moving.minimum_distance == GLOBAL_MINIMUM
    again = solver.evaluate_collision_pair_distances(np.array([QUERY_DISPLACEMENT]))
    assert all(record.minimum_distance == GLOBAL_MINIMUM for record in again)


def test_fresh_query_preserves_last_solve_collision_diagnostics(pair_solver):
    robot, solver = pair_solver
    solver.configure_collision_constraint(min_distance=GLOBAL_MINIMUM, max_constraints=1)
    task = solver.add_frame_task("moving_task", "moving")
    task.weight = 1.0
    configuration = robot.get_current_configuration().copy()
    pose = robot.get_frame_pose("moving")
    target = np.eye(4)
    target[:3, :3] = pose.rotation
    target[:3, 3] = pose.translation
    options = embodik.PositionStepOptions()
    options.max_steps = 1
    solver.solve_position_step(configuration, target, "moving_task", options)
    before = [
        (record.object_a, record.object_b, record.distance)
        for record in solver.get_last_collision_debug_list()
    ]
    queries_before = solver.get_last_post_step_collision_exact_distance_queries()
    solver.evaluate_collision_pair_distances(np.array([QUERY_DISPLACEMENT]))
    after = [
        (record.object_a, record.object_b, record.distance)
        for record in solver.get_last_collision_debug_list()
    ]
    assert after == before
    assert solver.get_last_post_step_collision_exact_distance_queries() == queries_before
