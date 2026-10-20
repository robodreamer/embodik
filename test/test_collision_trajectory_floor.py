"""Caller-owned original pair floors survive commands and margin adaptation."""

import numpy as np
import pytest

import embodik as eik

URDF = """<robot name="anchored"><link name="world"/>
<link name="near"><collision><geometry><sphere radius="0.05"/></geometry></collision></link>
<joint name="near_fixed" type="fixed"><parent link="world"/><child link="near"/></joint>
<link name="far"><collision><geometry><sphere radius="0.05"/></geometry></collision></link>
<joint name="far_fixed" type="fixed"><parent link="world"/><child link="far"/><origin xyz="1 0 0"/></joint>
<link name="first"><collision><geometry><sphere radius="0.05"/></geometry></collision></link>
<joint name="first_slide" type="prismatic"><parent link="world"/><child link="first"/>
<origin xyz="0.1065 0 0"/><axis xyz="1 0 0"/>
<limit lower="-0.1" upper="0.2" velocity="1" effort="10"/></joint>
<link name="second"><collision><geometry><sphere radius="0.05"/></geometry></collision></link>
<joint name="second_slide" type="prismatic"><parent link="world"/><child link="second"/>
<origin xyz="1.15 0 0"/><axis xyz="1 0 0"/>
<limit lower="-0.1" upper="0.2" velocity="1" effort="10"/></joint></robot>"""
MARGIN = 0.025
TOLERANCE = 1e-9
PAIRS = [("near_0", "first_0"), ("far_0", "second_0")]


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / "anchored.urdf"
    path.write_text(URDF)
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.configure_collision_constraint(
        min_distance=MARGIN, max_constraints=1, include_pairs=PAIRS
    )
    for name in ("first", "second"):
        solver.add_frame_task(name, name, eik.TaskType.FRAME_POSITION)
    return robot, solver


def distances(solver, q):
    return {
        tuple(sorted((p.geometry_a, p.geometry_b))): p
        for p in solver.evaluate_collision_pair_distances(q)
    }


def targets(first, second):
    result = []
    for name, position in [("first", first), ("second", second)]:
        pose = np.eye(4)
        pose[0, 3] = position
        result.append(eik.TaskTarget(name, pose, 10.0, 0.0))
    return result


@pytest.mark.parametrize(
    "mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.MIN_ERROR, eik.TaskSolveMode.SCALE_ELASTIC]
)
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("inner_steps", [1, 3])
def test_original_floors_do_not_ratchet_on_departure_or_new_target(
    setup, mode, fallback, inner_steps
):
    robot, solver = setup
    q = np.zeros(robot.nq)
    original = distances(solver, q)
    floors = {key: min(p.distance, p.minimum_distance) for key, p in original.items()}
    solver.set_collision_trajectory_floor_anchor(q, TOLERANCE)
    options = eik.PositionStepOptions()
    options.primary_solve_mode = mode
    options.primary_allow_min_error_fallback = fallback
    options.max_steps = inner_steps
    for target_set, count in [(targets(0.15, 1.2), 40), (targets(0.101, 1.1), 80)]:
        start = q.copy()
        for _ in range(count):
            result = solver.solve_position_step(q, target_set, options)
            q = np.asarray(result.q_solution).copy()
            assert q.shape == (robot.nq,) and np.isfinite(q).all()
            for key, pair in distances(solver, q).items():
                assert pair.distance >= floors[key] - TOLERANCE
                assert pair.persistent_floor_m == floors[key]
        assert np.linalg.norm(q - start) > 1e-3
    # Re-anchoring at the outward command would incorrectly prevent this return.
    assert q[0] <= 1e-3


def test_anchor_query_is_pure_and_nominal_margin_changes_do_not_rebase_it(setup):
    robot, solver = setup
    robot.update_configuration(np.array([0.02, 0.01]))
    current = robot.get_current_configuration().copy()
    solver.set_collision_trajectory_floor_anchor(np.zeros(robot.nq))
    np.testing.assert_array_equal(robot.get_current_configuration(), current)
    first = distances(solver, np.zeros(robot.nq))
    solver.set_collision_min_distance(0.001)
    second = distances(solver, current)
    assert {k: p.persistent_floor_m for k, p in first.items()} == {
        k: p.persistent_floor_m for k, p in second.items()
    }
    solver.clear_collision_trajectory_floor_anchor()
    assert not solver.has_collision_trajectory_floor_anchor()
    assert all(p.persistent_floor_m is None for p in distances(solver, current).values())


def test_seed_admission_has_no_extra_hidden_floor_tolerance(setup):
    robot, solver = setup
    q = np.zeros(robot.nq)
    solver.set_collision_trajectory_floor_anchor(q, TOLERANCE)
    near = q.copy()
    near[0] = -0.5 * TOLERANCE
    admitted = solver.solve_position_step(near, targets(0.15, 1.2), eik.PositionStepOptions())
    assert admitted.status == eik.SolverStatus.SUCCESS
    bad = q.copy()
    bad[0] = -10 * TOLERANCE
    rejected = solver.solve_position_step(bad, targets(0.15, 1.2), eik.PositionStepOptions())
    assert rejected.status == eik.SolverStatus.COLLISION_VIOLATED
    assert "measured seed" in rejected.status_message
    # Small allowed losses cannot accumulate by rebasing to the previous seed.
    bad[0] = -1.5 * TOLERANCE
    rejected = solver.solve_position_step(bad, targets(0.15, 1.2), eik.PositionStepOptions())
    assert rejected.status == eik.SolverStatus.COLLISION_VIOLATED


def test_catalog_change_fails_closed_until_explicit_clear(setup):
    robot, solver = setup
    q = np.zeros(robot.nq)
    solver.set_collision_trajectory_floor_anchor(q)
    solver.configure_collision_constraint(min_distance=MARGIN, include_pairs=PAIRS[:1])
    result = solver.solve_position_step(q, targets(0.15, 1.2), eik.PositionStepOptions())
    assert result.status == eik.SolverStatus.INVALID_INPUT
    np.testing.assert_array_equal(result.q_solution, q)
    assert "catalog changed" in result.status_message
    solver.clear_collision_trajectory_floor_anchor()
    result = solver.solve_position_step(q, targets(0.15, 1.2), eik.PositionStepOptions())
    assert result.status == eik.SolverStatus.SUCCESS


@pytest.mark.parametrize("invalid", [-1.0, np.nan, np.inf])
def test_invalid_tolerance_does_not_replace_existing_anchor(setup, invalid):
    robot, solver = setup
    solver.set_collision_trajectory_floor_anchor(np.zeros(robot.nq))
    with pytest.raises(ValueError):
        solver.set_collision_trajectory_floor_anchor(np.ones(robot.nq), invalid)
    assert solver.has_collision_trajectory_floor_anchor()
    assert solver.get_collision_trajectory_floor_tolerance() == TOLERANCE


def test_acceleration_policy_is_not_silently_certified(setup):
    robot, solver = setup
    q = np.zeros(robot.nq)
    solver.set_collision_trajectory_floor_anchor(q)
    solver.enable_acceleration_limits(True)
    result = solver.solve_position_step(q, targets(0.15, 1.2), eik.PositionStepOptions())
    assert result.status == eik.SolverStatus.INVALID_INPUT
    assert "acceleration" in result.status_message
    np.testing.assert_array_equal(result.q_solution, q)


def test_both_anchored_faces_are_mandatory_despite_one_row_budget(setup):
    robot, solver = setup
    solver.set_collision_trajectory_floor_anchor(np.zeros(robot.nq))
    q = np.array([0.0, -0.025])
    for name in ("first", "second"):
        solver.get_task(name).set_target_velocity(np.zeros(3))
    result = solver.solve_velocity(q)
    assert result.status == eik.SolverStatus.SUCCESS
    assert len(solver.get_last_collision_debug_list()) == len(PAIRS)
