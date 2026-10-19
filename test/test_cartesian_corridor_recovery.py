"""Native free-space recovery lifecycle and physical endpoint gates."""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import embodik as eik


def _build(
    tmp_path,
    *,
    caps=0.2,
    support=False,
    endpoint_safety_steering=False,
    collision=False,
    feedback=False,
    reentry=False,
    warm_backoff=False,
):
    path = tmp_path / "corridor.urdf"
    path.write_text("""<robot name="corridor">
<link name="world"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
<link name="moving"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
<link name="tip"/>
<joint name="slide" type="prismatic"><parent link="world"/><child link="moving"/>
<axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="100" velocity="0.2"/></joint>
<joint name="tip_fixed" type="fixed"><parent link="moving"/><child link="tip"/><origin xyz="0 0.1 0"/></joint>
</robot>""")
    if collision:
        path.write_text(
            path.read_text()
            .replace(
                '<link name="world">',
                '<link name="world"><collision><origin xyz="0.3 0 0"/><geometry><sphere radius="0.05"/></geometry></collision>',
            )
            .replace(
                '<link name="moving">',
                '<link name="moving"><collision><geometry><sphere radius="0.05"/></geometry></collision>',
            )
        )
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.add_posture_task("guide")
    options = eik.CartesianCorridorOptions()
    options.endpoint_safety_steering = endpoint_safety_steering
    options.allow_measured_state_updates = feedback
    options.allow_pose_zone_reentry = reentry
    options.warm_start_guide_backoff = warm_backoff
    options.budget_velocity_constraints = reentry
    options.pose_caps = np.array([[caps, 0.1], [caps, 0.1]])
    options.terminal_tolerances = np.array([[0.005, 0.01], [0.005, 0.01]])
    if collision:
        solver.configure_collision_constraint(
            min_distance=0.06, include_pairs=[("world_0", "moving_0")]
        )
        pair = eik.CollisionGeometryPair()
        pair.geometry_a, pair.geometry_b = "world_0", "moving_0"
        options.collision_floor_pairs = [pair]
        options.collision_floors = np.array([0.06])
    if support:
        options.support_frame = "world"
        options.support_halfplanes = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        options.support_upper = np.array([0.04, 1.0, 1.0, 1.0])
    controller = eik.CartesianCorridorRecovery(solver, ["moving", "tip"], options)
    targets = [np.eye(4), np.eye(4)]
    for target in targets:
        target[0, 3] = 0.15
    targets[1][1, 3] = 0.1
    return robot, solver, controller, targets


def test_backoff_memory_is_opt_in_and_resets_on_activation(tmp_path):
    assert not eik.CartesianCorridorOptions().warm_start_guide_backoff
    _, _, controller, targets = _build(tmp_path, caps=0.1501, feedback=True, warm_backoff=True)
    guide = np.array([-0.15])
    controller.start(np.zeros(1), guide, targets, True)
    first = controller.step(np.zeros(1), targets, True)
    assert first.accepted
    assert first.accepted_backoff == 7
    assert first.attempts == 8
    np.testing.assert_allclose(first.q_next, 0.02 * first.velocity, atol=1e-12)

    # A physically valid measured change makes every gain admissible. Memory
    # tries one level above the last accepted gain before the larger gains.
    measured = np.array([0.01])
    second = controller.step(measured, targets, True)
    assert second.accepted and second.attempts == 1
    assert second.accepted_backoff == 6
    np.testing.assert_allclose(second.q_next, measured + 0.02 * second.velocity, atol=1e-12)
    assert np.all(second.pose_errors <= np.array([[0.1501, 0.1], [0.1501, 0.1]]))
    assert abs(second.velocity[0]) <= 0.2 + 1e-8

    controller.start(measured, guide, targets, True)
    restarted = controller.step(measured, targets, True)
    assert restarted.accepted and restarted.attempts == 1
    assert restarted.accepted_backoff == 0


def test_remembered_backoff_retains_all_candidates_when_blocked(tmp_path):
    _, _, controller, targets = _build(tmp_path, caps=0.1501, warm_backoff=True)
    controller.start(np.zeros(1), np.array([-0.15]), targets, True)
    first = controller.step(np.zeros(1), targets, True)
    assert first.accepted and first.accepted_backoff == 7
    blocked = controller.step(first.q_next, targets, True)
    assert not blocked.accepted
    assert blocked.phase == eik.CartesianRecoveryPhase.BLOCKED
    assert blocked.attempts == 16  # Eight original gains for each of two modes.
    assert blocked.accepted_backoff == -1


@pytest.mark.parametrize("existing_rows", [False, True])
def test_safety_steered_search_preserves_live_solver_and_configuration(tmp_path, existing_rows):
    robot, solver, controller, targets = _build(
        tmp_path, support=True, endpoint_safety_steering=True
    )
    for target in targets:
        target[0, 3] = 0.03
    robot.update_configuration(np.array([-0.1]))
    if existing_rows:
        solver.set_linear_velocity_constraints(np.ones((1, 1)), np.array([-0.05]), np.array([0.05]))
    before_dt = solver.dt
    result = controller.find_guide([np.zeros(1)], targets, max_iterations=30)
    assert result.found
    assert not result.corridor_verified
    assert result.q_guide[0] <= 0.04 + 1e-9
    np.testing.assert_array_equal(robot.get_current_configuration(), [-0.1])
    assert solver.dt == before_dt
    assert solver.get_linear_velocity_constraint_rows() == int(existing_rows)
    velocity = solver.solve_velocity(np.array([-0.1])).joint_velocities
    assert abs(velocity[0]) <= (0.05 if existing_rows else 0.2) + 1e-8


def test_safety_steered_search_does_not_admit_unsupported_endpoint(tmp_path):
    robot, _, controller, targets = _build(tmp_path, support=True, endpoint_safety_steering=True)
    robot.update_configuration(np.array([-0.1]))
    result = controller.find_guide([np.zeros(1)], targets, max_iterations=30)
    assert not result.found  # 0.15 exceeds the declared CoM support envelope.
    np.testing.assert_array_equal(robot.get_current_configuration(), [-0.1])


def test_measured_feedback_is_revalidated_and_integrated_without_reset(tmp_path):
    _, _, controller, targets = _build(tmp_path, feedback=True)
    q = np.zeros(1)
    controller.start(q, np.array([0.15]), targets, True)
    for tick in range(200):
        measured = q + (0.00005 if tick % 2 else -0.00005)
        result = controller.step(measured, targets, True)
        if result.accepted:
            np.testing.assert_allclose(result.q_next, measured + 0.02 * result.velocity, atol=1e-12)
            assert abs(result.velocity[0]) <= 0.2 + 1e-8
            q = np.asarray(result.q_next).copy()
        if result.phase == eik.CartesianRecoveryPhase.SETTLED:
            break
        assert result.accepted
    assert controller.phase == eik.CartesianRecoveryPhase.SETTLED


def test_pose_zone_reentry_preserves_original_bounds_after_entry(tmp_path):
    robot, _, controller, targets = _build(tmp_path, feedback=True, reentry=True)
    q = np.array([-0.08])
    controller.start(q, np.array([0.15]), targets, True)
    original_cap = 0.2
    previous = abs(q[0] - 0.15)
    assert previous > original_cap
    for _ in range(250):
        result = controller.step(q, targets, True)
        if result.accepted:
            current = abs(result.q_next[0] - 0.15)
            assert current <= max(original_cap, previous) + 1e-12
            assert abs(result.velocity[0]) <= 0.2 + 1e-8
            np.testing.assert_allclose(result.q_next, q + 0.02 * result.velocity)
            q, previous = np.asarray(result.q_next).copy(), current
        if result.phase == eik.CartesianRecoveryPhase.SETTLED:
            break
        assert result.accepted
    assert controller.phase == eik.CartesianRecoveryPhase.SETTLED
    robot.update_configuration(q)
    assert abs(robot.get_frame_pose("moving").translation[0] - 0.15) <= 0.005


def test_pose_zone_reentry_never_relaxes_physical_support(tmp_path):
    _, _, controller, targets = _build(tmp_path, support=True, reentry=True)
    for target in targets:
        target[0, 3] = 0.03
    with pytest.raises(ValueError, match="physical safety budgets"):
        controller.start(np.array([0.05]), np.array([0.03]), targets, True)


@pytest.mark.parametrize("collision", [False, True])
def test_measured_feedback_cannot_reuse_a_safe_state_certificate_for_unsafe_q(tmp_path, collision):
    _, _, controller, targets = _build(
        tmp_path, feedback=True, support=not collision, collision=collision
    )
    for target in targets:
        target[0, 3] = 0.03 if not collision else 0.1
    controller.start(np.zeros(1), np.array([targets[0][0, 3]]), targets, True)
    measured = np.array([0.05 if not collision else 0.15])
    result = controller.step(measured, targets, True)
    assert not result.accepted
    assert result.phase == eik.CartesianRecoveryPhase.INTERRUPTED
    assert result.reason == "current_safety_budget_violated"


@pytest.mark.parametrize(
    "target,seed,expected", [(0.1, 0.0, True), (0.15, 0.0, False), (0.1, 0.25, False)]
)
def test_safety_steered_collision_endpoint_and_penetration_guard(tmp_path, target, seed, expected):
    robot, solver, controller, targets = _build(
        tmp_path, collision=True, endpoint_safety_steering=True
    )
    for pose in targets:
        pose[0, 3] = target
    robot.update_configuration(np.array([-0.1]))
    before = solver.evaluate_collision_pair_distances(np.zeros(1))[0].distance
    result = controller.find_guide([np.array([seed])], targets, max_iterations=30)
    assert result.found == expected
    if result.found:
        assert solver.evaluate_collision_pair_distances(result.q_guide)[0].distance >= 0.06 - 1e-9
    np.testing.assert_array_equal(robot.get_current_configuration(), [-0.1])
    assert solver.evaluate_collision_pair_distances(np.zeros(1))[0].distance == before


def test_native_corridor_integrates_bounded_motion_to_original_targets(tmp_path):
    robot, _, controller, targets = _build(tmp_path)
    q = np.zeros(1)
    controller.start(q, np.array([0.15]), targets, both_hands_free=True)
    history = []
    for _ in range(100):
        step = controller.step(q, targets, both_hands_free=True)
        if step.accepted:
            np.testing.assert_allclose(step.q_next, q + 0.02 * step.velocity, atol=1e-12)
            assert abs(step.velocity[0]) <= 0.2 + 1e-8
            assert not step.solver_result.weighted_fallback_used
            assert step.computation_time_ms >= 0.0 and 1 <= step.attempts <= 16
            q = np.asarray(step.q_next).copy()
            history.append(q)
        if step.phase == eik.CartesianRecoveryPhase.SETTLED:
            break
    assert controller.phase == eik.CartesianRecoveryPhase.SETTLED
    assert controller.accepted_steps == len(history) > 1
    robot.update_configuration(q)
    for frame, target in zip(["moving", "tip"], targets):
        assert np.linalg.norm(robot.get_frame_pose(frame).translation - target[:3, 3]) <= 0.005


@pytest.mark.parametrize(
    "change,reason",
    [
        ("target", "target_changed"),
        ("hand", "hand_protection_changed"),
        ("state", "input_state_changed"),
        ("acceleration", "solver_policy_changed"),
        ("task", "solver_policy_changed"),
    ],
)
def test_native_corridor_cancels_stale_guide_without_executable_output(tmp_path, change, reason):
    _, solver, controller, targets = _build(tmp_path)
    q = np.zeros(1)
    controller.start(q, np.array([0.15]), targets, both_hands_free=True)
    if change == "target":
        targets[0][0, 3] += 0.01
    if change == "state":
        q += 0.001
    if change == "acceleration":
        solver.enable_acceleration_limits(True)
    if change == "task":
        solver.add_posture_task("unexpected")
    step = controller.step(q, targets, both_hands_free=change != "hand")
    assert not step.accepted and step.q_next.size == step.velocity.size == 0
    assert step.phase == eik.CartesianRecoveryPhase.INTERRUPTED and step.reason == reason
    assert controller.accepted_steps == 0


def test_native_corridor_pose_budget_rejects_every_wrong_direction_step(tmp_path):
    _, _, controller, targets = _build(tmp_path, caps=0.15)
    q = np.zeros(1)
    controller.start(q, np.array([-0.15]), targets, both_hands_free=True)
    step = controller.step(q, targets, both_hands_free=True)
    assert not step.accepted and step.phase == eik.CartesianRecoveryPhase.BLOCKED
    assert step.q_next.size == step.velocity.size == 0
    assert step.attempts == 16


def test_retargeting_rearms_from_current_state_without_joint_reset(tmp_path):
    _, _, controller, targets = _build(tmp_path)
    q = np.zeros(1)
    controller.start(q, np.array([0.15]), targets, both_hands_free=True)
    first = controller.step(q, targets, both_hands_free=True)
    assert first.accepted
    q = np.asarray(first.q_next).copy()
    for target in targets:
        target[0, 3] = 0.16
    canceled = controller.step(q, targets, both_hands_free=True)
    assert not canceled.accepted and canceled.reason == "target_changed"
    controller.start(q, np.array([0.16]), targets, both_hands_free=True)
    next_step = controller.step(q, targets, both_hands_free=True)
    assert next_step.accepted
    np.testing.assert_allclose(next_step.q_next, q + 0.02 * next_step.velocity, atol=1e-12)
    assert 0 < next_step.q_next[0] - q[0] <= 0.004 + 1e-9


def test_native_corridor_support_envelope_prevents_endpoint_violation(tmp_path):
    robot, _, controller, targets = _build(tmp_path, support=True)
    q = np.zeros(1)
    controller.start(q, np.array([0.15]), targets, both_hands_free=True)
    for _ in range(100):
        step = controller.step(q, targets, both_hands_free=True)
        if not step.accepted:
            break
        q = np.asarray(step.q_next).copy()
        robot.update_configuration(q)
        assert robot.get_com_position()[0] <= 0.04 + 1e-9
    assert controller.phase == eik.CartesianRecoveryPhase.BLOCKED


@pytest.mark.parametrize("invalid", ["hand", "guide", "transform"])
def test_invalid_activation_invalidates_previous_guide(tmp_path, invalid):
    _, _, controller, targets = _build(tmp_path)
    q, guide = np.zeros(1), np.array([0.15])
    controller.start(q, guide, targets, both_hands_free=True)
    if invalid == "guide":
        guide[0] = 2.0
    if invalid == "transform":
        targets[0][0, 0] = 2.0
    with pytest.raises(ValueError):
        controller.start(q, guide, targets, both_hands_free=invalid != "hand")
    assert controller.phase == eik.CartesianRecoveryPhase.INTERRUPTED


def test_missing_collision_floor_geometry_fails_closed(tmp_path):
    _, solver, _, targets = _build(tmp_path)
    options = eik.CartesianCorridorOptions()
    options.pose_caps = np.array([[0.2, 0.1], [0.2, 0.1]])
    options.terminal_tolerances = np.array([[0.005, 0.01], [0.005, 0.01]])
    pair = eik.CollisionGeometryPair()
    pair.geometry_a, pair.geometry_b = "missing_a", "missing_b"
    options.collision_floor_pairs = [pair]
    options.collision_floors = np.array([0.01])
    controller = eik.CartesianCorridorRecovery(solver, ["moving", "tip"], options)
    with pytest.raises(ValueError, match="safety budgets"):
        controller.start(np.zeros(1), np.array([0.15]), targets, both_hands_free=True)
    result = controller.step(np.zeros(1), targets, both_hands_free=True)
    assert not result.accepted and result.q_next.size == result.velocity.size == 0


def test_native_guide_search_returns_only_a_hypothetical_endpoint(tmp_path):
    robot, _, controller, targets = _build(tmp_path)
    measured = np.asarray(robot.get_current_configuration()).copy()
    guide = controller.find_guide([np.zeros(1)], targets, max_iterations=100)
    assert guide.found and guide.q_guide.shape == (1,)
    np.testing.assert_array_equal(robot.get_current_configuration(), measured)
    assert controller.phase == eik.CartesianRecoveryPhase.IDLE
    assert guide.computation_time_ms >= 0 and 1 <= guide.total_iterations <= 100
    assert np.all(np.asarray(guide.pose_errors) <= [[0.005, 0.01], [0.005, 0.01]])
    controller.start(measured, guide.q_guide, targets, both_hands_free=True)
    step = controller.step(measured, targets, both_hands_free=True)
    assert step.accepted
    assert step.q_next[0] < guide.q_guide[0]
    np.testing.assert_allclose(step.q_next, measured + 0.02 * step.velocity, atol=1e-12)


def test_native_guide_search_fails_without_executable_output_for_unreachable_target(tmp_path):
    robot, _, controller, targets = _build(tmp_path)
    measured = np.asarray(robot.get_current_configuration()).copy()
    for target in targets:
        target[0, 3] = 2.0
    guide = controller.find_guide([np.zeros(1)], targets, max_iterations=50)
    assert not guide.found and guide.q_guide.size == 0
    assert guide.total_iterations <= 50
    np.testing.assert_array_equal(robot.get_current_configuration(), measured)


def test_native_guide_search_requires_interruption_before_retargeting(tmp_path):
    _, _, controller, targets = _build(tmp_path)
    controller.start(np.zeros(1), np.array([0.15]), targets, both_hands_free=True)
    with pytest.raises(ValueError, match="Interrupt"):
        controller.find_guide([np.zeros(1)], targets)


def test_native_guide_search_tracks_coupled_noncommuting_rotations(tmp_path):
    path = tmp_path / "rotating_corridor.urdf"
    path.write_text("""<robot name="rotating_corridor">
<link name="world"/><link name="first"/><link name="tip"/>
<joint name="yaw" type="revolute"><parent link="world"/><child link="first"/>
<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="100" velocity="1"/></joint>
<joint name="pitch" type="revolute"><parent link="first"/><child link="tip"/>
<origin xyz="0.2 0.1 0"/><axis xyz="0 1 0"/>
<limit lower="-1" upper="1" effort="100" velocity="1"/></joint></robot>""")
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.add_posture_task("guide")
    options = eik.CartesianCorridorOptions()
    options.pose_caps = np.array([[1.0, 2.0], [1.0, 2.0]])
    options.terminal_tolerances = np.array([[0.005, 0.01], [0.005, 0.01]])
    controller = eik.CartesianCorridorRecovery(solver, ["first", "tip"], options)
    robot.update_configuration(np.array([0.5, -0.4]))
    targets = []
    for frame in ["first", "tip"]:
        pose = robot.get_frame_pose(frame)
        target = np.eye(4)
        target[:3, :3], target[:3, 3] = pose.rotation, pose.translation
        targets.append(target)
    robot.update_configuration(np.zeros(2))
    result = controller.find_guide([np.zeros(2)], targets, max_iterations=50)
    assert result.found
    np.testing.assert_array_equal(robot.get_current_configuration(), np.zeros(2))
    robot.update_configuration(result.q_guide)
    for frame, target in zip(["first", "tip"], targets):
        actual = robot.get_frame_pose(frame)
        assert np.linalg.norm(actual.translation - target[:3, 3]) <= 0.005
        relative = target[:3, :3] @ actual.rotation.T
        assert np.arccos(np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0)) <= 0.01


def _circular_recovery(tmp_path):
    path = tmp_path / "circular_corridor.urdf"
    path.write_text(
        """<robot name="circle"><link name="world"/><link name="rotating"/><link name="tip"/>
<joint name="yaw" type="revolute"><parent link="world"/><child link="rotating"/>
<axis xyz="0 0 1"/><limit lower="0" upper="7" effort="100" velocity="0.5"/></joint>
<joint name="offset" type="fixed"><parent link="rotating"/><child link="tip"/><origin xyz="1 0 0"/></joint>
</robot>"""
    )
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.add_posture_task("guide")
    options = eik.CartesianCorridorOptions()
    options.pose_caps = np.full((2, 2), 0.101)
    options.terminal_tolerances = np.tile([0.005, 0.01], (2, 1))
    controller = eik.CartesianCorridorRecovery(solver, ["rotating", "tip"], options)
    targets = [np.eye(4), np.eye(4)]
    targets[1][0, 3] = 1.0
    robot.update_configuration(np.array([0.1]))
    return robot, controller, targets


def test_surrogate_rollout_rejects_wrong_branch_before_selecting_executable_guide(tmp_path):
    robot, controller, targets = _circular_recovery(tmp_path)
    q = np.array([0.1])
    # Both endpoints track the target, but the long branch initially moves
    # away from it and cannot execute within the same declared pose caps.
    endpoint = controller.find_guide([np.array([2 * np.pi])], targets)
    assert endpoint.found and not endpoint.corridor_verified
    result = controller.find_corridor_guide(q, [np.array([2 * np.pi]), np.zeros(1)], targets)
    assert result.found and result.corridor_verified and result.seed_index == 1
    assert result.rollout_steps > 0
    np.testing.assert_array_equal(robot.get_current_configuration(), q)
    controller.start(q, result.q_guide, targets, both_hands_free=True)
    actual = controller.step(q, targets, both_hands_free=True)
    assert actual.accepted and actual.q_next[0] < q[0]
    np.testing.assert_allclose(actual.q_next, q + 0.02 * actual.velocity, atol=1e-12)


def test_background_rollout_can_be_cancelled_without_executing_a_pose(tmp_path):
    robot, controller, targets = _circular_recovery(tmp_path)
    q = np.array([0.1])
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            controller.find_corridor_guide,
            q,
            [np.array([2 * np.pi]) for _ in range(500)],
            targets,
            20,
            100,
        )
        deadline = time.monotonic() + 10
        while not controller.search_running and not future.done() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert controller.search_running
        controller.cancel_search()
        result = future.result(timeout=10)
    assert result.cancelled and not result.found and not result.corridor_verified
    assert result.q_guide.size == 0 and not controller.search_running
    np.testing.assert_array_equal(robot.get_current_configuration(), q)


def test_budget_aware_recovery_finds_inward_tangent_at_position_boundary(tmp_path):
    path = tmp_path / "planar_corridor.urdf"
    path.write_text("""<robot name="planar">
<link name="world"/><link name="x"/><link name="hand"/><link name="other"/>
<joint name="x_slide" type="prismatic"><parent link="world"/><child link="x"/>
<axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="100" velocity="1"/></joint>
<joint name="y_slide" type="prismatic"><parent link="x"/><child link="hand"/>
<axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="100" velocity="1"/></joint>
<joint name="fixed" type="fixed"><parent link="hand"/><child link="other"/></joint>
</robot>""")
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.add_posture_task("guide")
    options = eik.CartesianCorridorOptions()
    options.pose_caps = np.tile([0.1, 0.1], (2, 1))
    options.terminal_tolerances = np.tile([0.005, 0.01], (2, 1))
    # Endpoint rejection alone cannot turn an outward request into a feasible
    # inward/tangential step. The guide is deliberately beyond the pose zone;
    # this checks one local step, not a claim that this guide can be reached.
    options.budget_velocity_constraints = True
    controller = eik.CartesianCorridorRecovery(solver, ["hand", "other"], options)
    q = np.array([0.1, 0.0])
    targets = [np.eye(4), np.eye(4)]
    controller.start(q, np.array([0.12, 0.15]), targets, both_hands_free=True)
    step = controller.step(q, targets, both_hands_free=True)
    assert step.accepted
    assert step.mode_requested == eik.TaskSolveMode.MIN_ERROR
    assert step.q_next[0] < q[0] and step.q_next[1] > q[1]
    assert np.linalg.norm(step.q_next) <= 0.1 + 1e-12
    np.testing.assert_allclose(step.q_next, q + options.dt * step.velocity)
    assert solver.get_linear_velocity_constraint_rows() == 0


def test_budget_aware_recovery_preserves_external_velocity_rows(tmp_path):
    _, solver, _, _ = _build(tmp_path)
    solver.set_linear_velocity_constraints(np.ones((1, 1)), np.array([-0.01]), np.array([0.01]))
    options = eik.CartesianCorridorOptions()
    options.budget_velocity_constraints = True
    with pytest.raises(ValueError, match="exclusive ownership"):
        eik.CartesianCorridorRecovery(solver, ["moving", "tip"], options)
    assert solver.get_linear_velocity_constraint_rows() == 1


def test_budget_aware_recovery_finds_tangent_at_orientation_boundary(tmp_path):
    path = tmp_path / "angular_zone.urdf"
    path.write_text("""<robot name="angular_zone">
<link name="world"/><link name="yaw"/><link name="hand"/><link name="other"/>
<joint name="yaw" type="revolute"><parent link="world"/><child link="yaw"/>
<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="100" velocity="1"/></joint>
<joint name="pitch" type="revolute"><parent link="yaw"/><child link="hand"/>
<axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="100" velocity="1"/></joint>
<joint name="fixed" type="fixed"><parent link="hand"/><child link="other"/></joint>
</robot>""")
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.add_posture_task("guide")
    options = eik.CartesianCorridorOptions()
    options.pose_caps = np.tile([0.1, 0.1], (2, 1))
    options.terminal_tolerances = np.tile([0.005, 0.01], (2, 1))
    options.budget_velocity_constraints = True
    controller = eik.CartesianCorridorRecovery(solver, ["hand", "other"], options)
    q = np.array([0.1, 0.0])
    targets = [np.eye(4), np.eye(4)]
    controller.start(q, np.array([0.12, 0.15]), targets, both_hands_free=True)
    step = controller.step(q, targets, both_hands_free=True)
    assert step.accepted and step.mode_requested == eik.TaskSolveMode.MIN_ERROR
    assert step.q_next[0] < q[0] and step.q_next[1] > q[1]
    assert np.all(np.asarray(step.pose_errors) <= options.pose_caps + 1e-12)
    assert solver.get_linear_velocity_constraint_rows() == 0


@pytest.mark.parametrize("gain", [0.0, -1.0, np.nan, np.inf])
def test_invalid_guide_gain_is_rejected(tmp_path, gain):
    _, solver, _, _ = _build(tmp_path)
    options = eik.CartesianCorridorOptions()
    options.pose_caps = np.tile([0.2, 0.1], (2, 1))
    options.terminal_tolerances = np.tile([0.005, 0.01], (2, 1))
    options.guide_gain = gain
    with pytest.raises(ValueError, match="guide gain"):
        eik.CartesianCorridorRecovery(solver, ["moving", "tip"], options)


def test_higher_guide_gain_shortens_convergence_without_widening_limits(tmp_path):
    robot, solver, baseline, targets = _build(tmp_path)
    options = eik.CartesianCorridorOptions()
    options.pose_caps = np.tile([0.2, 0.1], (2, 1))
    options.terminal_tolerances = np.tile([0.005, 0.01], (2, 1))
    options.guide_gain = 30.0
    fast = eik.CartesianCorridorRecovery(solver, ["moving", "tip"], options)
    counts = []
    for controller in (baseline, fast):
        q = np.zeros(1)
        controller.start(q, np.array([0.15]), targets, both_hands_free=True)
        for n in range(100):
            result = controller.step(q, targets, both_hands_free=True)
            assert result.accepted
            assert abs(result.velocity[0]) <= 0.2 + 1e-8
            assert 0 <= result.q_next[0] <= 0.15 + 1e-12
            np.testing.assert_allclose(result.q_next, q + 0.02 * result.velocity)
            q = np.asarray(result.q_next).copy()
            if result.phase == eik.CartesianRecoveryPhase.SETTLED:
                break
        assert controller.phase == eik.CartesianRecoveryPhase.SETTLED
        counts.append(n + 1)
    assert counts[1] < counts[0]


def test_explicit_best_effort_guide_reports_mode_and_retains_endpoint_gates(tmp_path):
    _, solver, _, targets = _build(tmp_path)
    options = eik.CartesianCorridorOptions()
    options.pose_caps = np.tile([0.2, 0.1], (2, 1))
    options.terminal_tolerances = np.tile([0.005, 0.01], (2, 1))
    options.guide_mode = eik.TaskSolveMode.MIN_ERROR
    controller = eik.CartesianCorridorRecovery(solver, ["moving", "tip"], options)
    q = np.zeros(1)
    controller.start(q, np.array([0.15]), targets, both_hands_free=True)
    result = controller.step(q, targets, both_hands_free=True)
    assert result.accepted and result.mode_requested == eik.TaskSolveMode.MIN_ERROR
    assert abs(result.velocity[0]) <= 0.2 + 1e-8
    np.testing.assert_allclose(result.q_next, q + options.dt * result.velocity)
    changed = [p.copy() for p in targets]
    changed[0][0, 3] += 0.01
    cancelled = controller.step(result.q_next, changed, both_hands_free=True)
    assert not cancelled.accepted and cancelled.reason == "target_changed"


def test_unsupported_guide_mode_is_rejected(tmp_path):
    _, solver, _, _ = _build(tmp_path)
    options = eik.CartesianCorridorOptions()
    options.guide_mode = eik.TaskSolveMode.SCALE_ELASTIC
    with pytest.raises(ValueError, match="Guide mode"):
        eik.CartesianCorridorRecovery(solver, ["moving", "tip"], options)
