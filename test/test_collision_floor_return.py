"""A declared structural clearance remains valid after an outward excursion."""

import numpy as np
import pytest

import embodik as eik


@pytest.fixture
def structural_solver(tmp_path):
    urdf = tmp_path / "structural_return.urdf"
    urdf.write_text(
        """<robot name="structural_return"><link name="world"/>
<link name="obstacle"><collision><geometry><sphere radius="0.05"/></geometry></collision></link>
<joint name="fixed" type="fixed"><parent link="world"/><child link="obstacle"/></joint>
<link name="tool"><collision><geometry><sphere radius="0.05"/></geometry></collision></link>
<joint name="slide" type="prismatic"><parent link="world"/><child link="tool"/>
<origin xyz="0.2 0 0"/><axis xyz="1 0 0"/><limit lower="-0.09" upper="1" velocity="1" effort="1"/></joint></robot>"""
    )
    robot = eik.RobotModel(str(urdf))
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.set_non_worsening_collision_floor_enabled(True)
    solver.configure_collision_constraint(min_distance=0.15, max_constraints=1)
    task = solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSITION)
    task.allow_min_error_fallback = False
    return robot, solver


def test_structural_clearance_does_not_become_nominal_on_return(structural_solver):
    robot, solver = structural_solver
    q = np.zeros(1)
    options = eik.PositionStepOptions()
    options.dt = 0.02
    for target_x in (0.3, 0.2, 0.12):
        target = np.eye(4)
        target[0, 3] = target_x
        for _ in range(240):
            result = solver.solve_position_step(
                q, [eik.TaskTarget("move", target, 10.0, 10.0)], options
            )
            q = np.asarray(result.q_solution)
            assert np.isfinite(q).all()
            assert -0.09 <= q[0] <= 1.0
            records = solver.evaluate_collision_pair_distances(q)
            assert records and min(r.distance for r in records) >= 0.005 - 1e-4
        np.testing.assert_allclose(q[0], target_x - 0.2, atol=0.003)


def test_iterative_position_solve_accepts_clear_outward_motion(structural_solver):
    robot, solver = structural_solver
    target = np.eye(4)
    target[0, 3] = 0.31
    options = eik.PositionIKOptions()
    options.max_iterations = 100
    result = solver.solve_position(np.array([0.1]), target, "tool", options)
    assert result.collision_rejection_count == 0
    assert result.position_error < 1e-3
    assert result.q_solution[0] > 0.102
    assert (
        min(pair.distance for pair in solver.evaluate_collision_pair_distances(result.q_solution))
        >= 0.15
    )


@pytest.mark.parametrize("multi_target", [False, True])
@pytest.mark.parametrize("acceleration_limits", [False, True])
def test_global_nominal_clearance_does_not_override_structural_floor(
    structural_solver, multi_target, acceleration_limits
):
    """Nominal-clearance crossing must not silently replace an effective floor."""
    robot, solver = structural_solver
    options = eik.PositionStepOptions()
    options.dt = 0.02
    options.record_collision_candidates = True
    if acceleration_limits:
        solver.set_acceleration_limits(np.full(robot.nv, 15.0))
        solver.enable_acceleration_limits(True)

    def solve(q, x):
        target = np.eye(4)
        target[0, 3] = x
        incoming = np.asarray(solver.get_previous_joint_velocities()).copy()
        if multi_target:
            result = solver.solve_position_step(
                q, [eik.TaskTarget("move", target, 10.0, 10.0)], options
            )
        else:
            result = solver.solve_position_step(q, target, "move", options)
        if acceleration_limits:
            applied = (np.asarray(result.q_solution) - q) / options.dt
            assert np.max(np.abs(applied - incoming)) <= 15.0 * options.dt + 1e-10
            np.testing.assert_allclose(result.joint_velocities, applied, rtol=0.0, atol=1e-12)
            np.testing.assert_allclose(
                solver.get_previous_joint_velocities(), applied, rtol=0.0, atol=1e-12
            )
        return result

    # Seed the declared structural floor from the initial below-nominal pair.
    seeded = solve(np.zeros(1), 0.21)
    assert seeded.q_solution[0] > 0.0
    # Begin the return while well clear of nominal; preserve that command's
    # identity so its entry clearance is not replaced by a near-boundary anchor.
    solve(np.array([0.1]), 0.12)
    # Above nominal clearance by 30 micrometres: all old backoff fractions
    # cross nominal, while staying far above the declared 5mm floor.
    q = np.array([0.05003])
    returned = solve(q, 0.12)
    assert returned.q_solution[0] < q[0] - 1e-3
    assert returned.collision_rejection_count == 0
    assert all(c.floor_acceptable for c in returned.collision_candidates)
    assert (
        min(pair.distance for pair in solver.evaluate_collision_pair_distances(returned.q_solution))
        >= 0.005
    )


@pytest.mark.parametrize("multi_target", [False, True])
def test_collision_candidate_trace_is_opt_in(structural_solver, multi_target):
    robot, solver = structural_solver
    options = eik.PositionStepOptions()
    assert not options.record_collision_candidates
    target = np.eye(4)
    target[0, 3] = 0.3

    def solve():
        if multi_target:
            return solver.solve_position_step(
                np.zeros(1), [eik.TaskTarget("move", target, 10.0, 10.0)], options
            )
        return solver.solve_position_step(np.zeros(1), target, "move", options)

    baseline = solve()
    assert not baseline.collision_candidates
    options.record_collision_candidates = True
    traced = solve()
    assert traced.collision_candidates
    np.testing.assert_allclose(traced.q_solution, baseline.q_solution, atol=1e-12)
    assert traced.status == baseline.status
    for candidate in traced.collision_candidates:
        assert candidate.stage in {"collision_nominal", "collision_backoff"}
        assert candidate.reference_q.shape == (robot.nq,)
        assert np.isfinite(candidate.candidate_q).all()
        assert len(candidate.reference_margins) == len(candidate.candidate_margins)
        assert np.isfinite(candidate.candidate_min_distance)


@pytest.mark.parametrize("multi_target", [False, True])
@pytest.mark.parametrize("acceleration_limits", [False, True])
def test_nominal_clearance_remains_hard_without_structural_policy(
    structural_solver, multi_target, acceleration_limits
):
    robot, solver = structural_solver
    solver.set_non_worsening_collision_floor_enabled(False)
    if acceleration_limits:
        solver.set_acceleration_limits(np.full(robot.nv, 15.0))
        solver.enable_acceleration_limits(True)
    options = eik.PositionStepOptions()
    options.dt = 0.02
    target = np.eye(4)
    target[0, 3] = 0.12
    q = np.array([0.05003])
    if multi_target:
        result = solver.solve_position_step(
            q, [eik.TaskTarget("move", target, 10.0, 10.0)], options
        )
    else:
        result = solver.solve_position_step(q, target, "move", options)
    assert (
        min(pair.distance for pair in solver.evaluate_collision_pair_distances(result.q_solution))
        >= 0.15
    )


@pytest.mark.parametrize("multi_target", [False, True])
def test_constrained_stationary_hold_reopens_for_productive_motion(structural_solver, multi_target):
    """A held command must recover when measured state again permits progress."""
    robot, solver = structural_solver
    options = eik.PositionStepOptions()
    options.dt = 0.02
    options.continuity_command_revision = 1
    target = np.eye(4)
    target[0, 3] = 2.0
    targets = [eik.TaskTarget("move", target, 10.0, 10.0)]

    def solve(q):
        if multi_target:
            return solver.solve_position_step(q, targets, options)
        return solver.solve_position_step(q, target, "move", options)

    q = np.ones(1)
    for _ in range(30):
        result = solve(q)
        q = np.asarray(result.q_solution)
    # A valid boundary hold does not force an unrequested inset restoration.
    np.testing.assert_allclose(q, [1.0], atol=1e-8)
    # Simulate measured backdrive, keeping the same target and solver history.
    q = np.array([0.8])
    recovered = solve(q)
    assert recovered.q_solution[0] > q[0] + 1e-3
    assert recovered.q_solution[0] <= 1.0
    assert not recovered.position_step_hold_active
    assert (
        min(
            pair.distance for pair in solver.evaluate_collision_pair_distances(recovered.q_solution)
        )
        >= 0.15
    )


@pytest.mark.parametrize("step_dt, should_reopen", [(0.001, True), (0.2, False)])
def test_constrained_hold_keeps_satisfied_block_within_tolerance(tmp_path, step_dt, should_reopen):
    """A satisfied hold must admit small residuals, but reject excessive ones."""
    urdf = tmp_path / "coupled_rotation.urdf"
    urdf.write_text(
        """<robot name="coupled_rotation"><link name="world"/>
<link name="obstacle"><collision><origin xyz="0 0 3"/><geometry><sphere radius="0.05"/></geometry></collision></link>
<joint name="fixed" type="fixed"><parent link="world"/><child link="obstacle"/></joint>
<link name="arm"><collision><origin xyz="1 0 0"/><geometry><sphere radius="0.05"/></geometry></collision></link>
<joint name="rotate" type="revolute"><parent link="world"/><child link="arm"/>
<axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
<link name="tool"/><joint name="tcp" type="fixed"><parent link="arm"/><child link="tool"/><origin xyz="1 0 0"/></joint>
<link name="held_tool"/><joint name="held_tcp" type="fixed"><parent link="arm"/><child link="held_tool"/><origin xyz="0.02 0 0"/></joint></robot>"""
    )
    robot = eik.RobotModel(str(urdf))
    solver = eik.KinematicsSolver(robot)
    solver.dt = step_dt
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.configure_collision_constraint(min_distance=0.05, max_constraints=1)
    for name, frame in (("moving", "tool"), ("held", "held_tool")):
        task = solver.add_frame_task(name, frame, eik.TaskType.FRAME_POSITION)
        task.priority = 0
    q = np.array([-0.1])
    robot.update_configuration(q)
    target = np.eye(4)
    target[:3, 3] = robot.get_frame_pose("held_tool").translation
    moving = np.eye(4)
    moving[:3, 3] = robot.get_frame_pose("tool").translation
    moving[1, 3] = 2.0
    targets = [
        eik.TaskTarget("moving", moving, 10.0, 0.0),
        eik.TaskTarget("held", target, 10.0, 0.0),
    ]
    options = eik.PositionStepOptions()
    options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
    options.continuity_command_revision = 1
    options.record_velocity_candidates = True
    q = np.ones(1)
    for _ in range(40):
        result = solver.solve_position_step(q, targets, options)
        q = np.asarray(result.q_solution)
    assert result.position_step_hold_active
    # The short lever is exact at the backdriven state; the long lever must move.
    q = np.array([-0.1])
    recovered = solver.solve_position_step(q, targets, options)
    if not should_reopen:
        robot.update_configuration(recovered.q_solution)
        residual = np.linalg.norm(robot.get_frame_pose("held_tool").translation - target[:3, 3])
        assert recovered.position_step_hold_active, (recovered.q_solution, residual)
        np.testing.assert_allclose(recovered.q_solution, q, atol=1e-12)
        assert recovered.velocity_candidates
        robot.update_configuration(recovered.velocity_candidates[-1].candidate_q)
        candidate_error = np.linalg.norm(
            robot.get_frame_pose("held_tool").translation - target[:3, 3]
        )
        assert candidate_error > 2e-3
        return
    assert recovered.q_solution[0] > q[0] + 1e-4
    assert not recovered.position_step_hold_active
    robot.update_configuration(recovered.q_solution)
    held_error = np.linalg.norm(robot.get_frame_pose("held_tool").translation - target[:3, 3])
    assert 0.0 < held_error <= 2e-3
    assert (
        min(p.distance for p in solver.evaluate_collision_pair_distances(recovered.q_solution))
        >= 0.05
    )
