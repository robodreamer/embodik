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


def test_query_reports_initialized_persistent_floor_without_seeding_it(pair_solver):
    robot, solver = pair_solver
    solver.configure_collision_constraint(min_distance=0.15)
    solver.set_non_worsening_collision_floor_enabled(True)
    solver.set_collision_structural_floor(0.01)
    q = robot.get_current_configuration().copy()
    for _ in range(2):
        records = solver.evaluate_collision_pair_distances(q)
        assert all(record.persistent_floor_m is None for record in records)
    task = solver.add_frame_task("prime", "world", embodik.TaskType.FRAME_POSITION)
    task.set_target_velocity(np.zeros(3))
    result = solver.solve_velocity(q)
    assert result.status == embodik.SolverStatus.SUCCESS
    records = solver.evaluate_collision_pair_distances(q)
    near = next(
        record
        for record in records
        if {record.geometry_a, record.geometry_b} == {"near_0", "moving_0"}
    )
    assert near.persistent_floor_m == pytest.approx(0.01)
    assert near.minimum_distance == pytest.approx(0.15)
    # Observing a recovered state must not ratchet the initialized floor.
    recovered = solver.evaluate_collision_pair_distances(np.array([0.1]))
    near_recovered = next(
        record
        for record in recovered
        if {record.geometry_a, record.geometry_b} == {"near_0", "moving_0"}
    )
    assert near_recovered.persistent_floor_m == pytest.approx(0.01)
    np.testing.assert_array_equal(robot.get_current_configuration(), q)
    solver.set_collision_pair_min_distance("near", "moving", 0.005, activate_when_clear=False)
    lowered = solver.evaluate_collision_pair_distances(q)
    near_lowered = next(
        record
        for record in lowered
        if {record.geometry_a, record.geometry_b} == {"near_0", "moving_0"}
    )
    assert near_lowered.persistent_floor_m == pytest.approx(0.005)
    solver.set_non_worsening_collision_floor_enabled(False)
    assert all(
        record.persistent_floor_m is None for record in solver.evaluate_collision_pair_distances(q)
    )


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


def test_startup_floor_initialization_preserves_pair_margins_and_state(pair_solver):
    robot, solver = pair_solver
    solver.configure_collision_constraint(min_distance=0.15, max_constraints=1)
    solver.set_non_worsening_collision_floor_enabled(True)
    solver.set_collision_structural_floor(0.01)
    solver.set_collision_pair_min_distance("far", "moving", 0.2, activate_when_clear=False)
    actual_q = robot.get_current_configuration().copy()
    actual_dq = np.array([INITIAL_VELOCITY])
    robot.update_kinematics(actual_q, actual_dq)
    startup = np.array([QUERY_DISPLACEMENT])
    before = solver.evaluate_collision_pair_distances(startup)
    assert len(before) > 1  # Initialize the full catalog despite a one-row budget.
    assert all(pair.persistent_floor_m is None for pair in before)
    with pytest.raises(ValueError):
        solver.initialize_collision_recovery_floors(np.array([np.nan]))
    assert all(
        pair.persistent_floor_m is None
        for pair in solver.evaluate_collision_pair_distances(startup)
    )
    solver.initialize_collision_recovery_floors(startup)
    after = solver.evaluate_collision_pair_distances(startup)
    for initial, initialized in zip(before, after, strict=True):
        assert initialized.minimum_distance == initial.minimum_distance
        assert initialized.persistent_floor_m == min(initial.distance, initial.minimum_distance)
    np.testing.assert_array_equal(robot.get_current_configuration(), actual_q)
    np.testing.assert_array_equal(robot.get_current_velocity(), actual_dq)
    with pytest.raises(ValueError, match="already initialized"):
        solver.initialize_collision_recovery_floors(actual_q)


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


@pytest.mark.parametrize("query", ["distance", "floor"])
def test_fresh_query_preserves_last_solve_collision_diagnostics(pair_solver, query):
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
    if query == "distance":
        solver.evaluate_collision_pair_distances(np.array([QUERY_DISPLACEMENT]))
    else:
        solver.get_collision_pair_floors()
    after = [
        (record.object_a, record.object_b, record.distance)
        for record in solver.get_last_collision_debug_list()
    ]
    assert after == before
    assert solver.get_last_post_step_collision_exact_distance_queries() == queries_before


def _floor_metadata(records):
    if records is None:
        return None
    return [(p.geometry_a, p.geometry_b, p.minimum_distance, p.persistent_floor_m) for p in records]


def test_floor_catalog_matches_live_policy_without_mutating_state(pair_solver):
    robot, solver = pair_solver
    assert solver.get_collision_pair_floors() is None
    solver.configure_collision_constraint(min_distance=0.15, max_constraints=1)
    solver.set_non_worsening_collision_floor_enabled(True)
    solver.set_collision_structural_floor(0.01)
    q = np.array([QUERY_DISPLACEMENT])
    velocity = np.array([INITIAL_VELOCITY])
    robot.update_kinematics(q, velocity)

    def compare():
        records = solver.get_collision_pair_floors()
        assert _floor_metadata(records) == _floor_metadata(
            solver.evaluate_collision_pair_distances(q)
        )
        np.testing.assert_array_equal(robot.get_current_configuration(), q)
        np.testing.assert_array_equal(robot.get_current_velocity(), velocity)
        return records

    assert all(p.persistent_floor_m is None for p in compare())
    solver.initialize_collision_recovery_floors(q)
    assert all(p.persistent_floor_m is not None for p in compare())
    solver.set_collision_pair_min_distance("near", "moving", 0.005, activate_when_clear=False)
    assert any(p.persistent_floor_m == 0.005 for p in compare())
    solver.clear_collision_pair_min_distance("near", "moving")
    compare()
    solver.set_collision_min_distance(0.2)
    compare()
    solver.set_non_worsening_collision_floor_enabled(False)
    assert all(p.persistent_floor_m is None for p in compare())
    solver.clear_collision_constraint()
    assert solver.get_collision_pair_floors() is None


def test_floor_catalog_does_not_latch_pending_overrides_or_seed_floors(pair_solver):
    robot, solver = pair_solver
    solver.configure_collision_constraint(min_distance=GLOBAL_MINIMUM)
    solver.set_non_worsening_collision_floor_enabled(True)
    solver.set_collision_pair_min_distance("far", "moving", PAIR_MINIMUM, activate_when_clear=True)
    before = solver.get_collision_pair_min_distance_overrides()
    for _ in range(2):
        records = solver.get_collision_pair_floors()
        assert all(
            p.minimum_distance == GLOBAL_MINIMUM and p.persistent_floor_m is None for p in records
        )
        assert _floor_metadata(records) == _floor_metadata(
            solver.evaluate_collision_pair_distances(robot.get_current_configuration())
        )
    assert solver.get_collision_pair_min_distance_overrides() == before


def test_floor_catalog_respects_pair_selection(pair_solver):
    robot, solver = pair_solver
    solver.configure_collision_constraint(min_distance=GLOBAL_MINIMUM)
    pair = solver.get_active_collision_pairs()[0]
    solver.configure_collision_constraint(min_distance=GLOBAL_MINIMUM, include_pairs=[pair])
    records = solver.get_collision_pair_floors()
    assert len(records) == 1
    assert {records[0].geometry_a, records[0].geometry_b} == set(pair)
    assert _floor_metadata(records) == _floor_metadata(
        solver.evaluate_collision_pair_distances(robot.get_current_configuration())
    )
    solver.configure_collision_constraint(
        min_distance=GLOBAL_MINIMUM, include_pairs=[pair], exclude_pairs=[pair]
    )
    assert solver.get_collision_pair_floors() is None


def test_floor_catalog_preserves_trajectory_anchor_precedence(pair_solver):
    robot, solver = pair_solver
    q = robot.get_current_configuration().copy()
    solver.configure_collision_constraint(min_distance=0.15)
    solver.set_collision_trajectory_floor_anchor(q)
    before = _floor_metadata(solver.get_collision_pair_floors())
    assert before == _floor_metadata(solver.evaluate_collision_pair_distances(q))
    solver.set_collision_min_distance(0.001)
    after = solver.get_collision_pair_floors()
    assert _floor_metadata(after) == _floor_metadata(solver.evaluate_collision_pair_distances(q))
    assert [p[3] for p in before] == [p.persistent_floor_m for p in after]
    assert any(p.persistent_floor_m > p.minimum_distance for p in after)
    solver.clear_collision_trajectory_floor_anchor()
    assert all(p.persistent_floor_m is None for p in solver.get_collision_pair_floors())
