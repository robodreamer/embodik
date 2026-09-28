"""Focused tests for the explicit registered-task hierarchy."""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

import embodik as eik


def _make_two_joint_robot(tmp_path) -> eik.RobotModel:
    urdf_path = tmp_path / "task_stack_two_joint.urdf"
    urdf_path.write_text(textwrap.dedent("""
            <robot name="task_stack_two_joint">
              <link name="base"/>
              <link name="link1">
                <inertial>
                  <origin xyz="0.5 0 0"/>
                  <mass value="1"/>
                  <inertia ixx="0.01" ixy="0" ixz="0"
                           iyy="0.01" iyz="0" izz="0.01"/>
                </inertial>
              </link>
              <joint name="joint1" type="revolute">
                <parent link="base"/>
                <child link="link1"/>
                <axis xyz="0 0 1"/>
                <limit lower="-2" upper="2" velocity="2" effort="10"/>
              </joint>
              <link name="link2">
                <inertial>
                  <origin xyz="0.5 0 0"/>
                  <mass value="1"/>
                  <inertia ixx="0.01" ixy="0" ixz="0"
                           iyy="0.01" iyz="0" izz="0.01"/>
                </inertial>
              </link>
              <joint name="joint2" type="revolute">
                <parent link="link1"/>
                <child link="link2"/>
                <origin xyz="1 0 0"/>
                <axis xyz="0 0 1"/>
                <limit lower="-2" upper="2" velocity="2" effort="10"/>
              </joint>
              <link name="tool"/>
              <joint name="tool_fixed" type="fixed">
                <parent link="link2"/>
                <child link="tool"/>
                <origin xyz="1 0 0"/>
              </joint>
            </robot>
            """))
    return eik.RobotModel(str(urdf_path), floating_base=False)


def _level(
    name: str,
    task_names: list[str],
    solve_mode: eik.TaskSolveMode = eik.TaskSolveMode.SCALE,
    allow_min_error_fallback: bool = False,
) -> eik.TaskLevelSpec:
    return eik.TaskLevelSpec(name, task_names, solve_mode, allow_min_error_fallback)


def _lex_stack(levels: list[eik.TaskLevelSpec]) -> eik.TaskStackConfig:
    return eik.TaskStackConfig(levels, eik.TaskStackBackend.LEXICOGRAPHIC_LEAST_SQUARES)


def test_explicit_two_level_frame_and_posture_stack(tmp_path):
    """An EE level is preserved while posture uses its remaining null space."""
    robot = _make_two_joint_robot(tmp_path)
    solver = eik.KinematicsSolver(robot)
    q = np.array([0.4, -0.7], dtype=float)
    robot.update_configuration(q)

    ee = solver.add_frame_task("ee", "tool", eik.TaskType.FRAME_POSITION)
    ee.priority = 9
    ee.weight = 1.0
    current_position = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)
    ee.set_target_position(current_position + np.array([0.0, 0.04, 0.0]))

    posture = solver.add_posture_task("posture")
    posture.priority = 0
    posture.weight = 0.2
    posture.set_target_configuration(np.zeros(robot.nq, dtype=float))

    solver.configure_task_stack(
        eik.TaskStackConfig(
            [
                _level("tracking", ["ee"]),
                _level("regularization", ["posture"], eik.TaskSolveMode.MIN_ERROR),
            ]
        )
    )

    result = solver.solve_velocity(q, apply_limits=False)
    dq = np.asarray(result.joint_velocities, dtype=float)

    assert result.status == eik.SolverStatus.SUCCESS
    assert result.prioritized_status == eik.SolverStatus.SUCCESS
    assert result.hierarchy_solve_path == eik.HierarchySolvePath.EXPLICIT_SNS
    assert result.hierarchy_backend == eik.TaskStackBackend.SNS
    assert result.higher_level_preservation_active is True
    assert len(result.task_scales) == 2
    assert [d.name for d in result.task_level_diagnostics] == [
        "tracking",
        "regularization",
    ]
    assert result.task_level_diagnostics[0].task_names == ["ee"]
    assert result.task_level_diagnostics[1].effective_solve_mode == eik.TaskSolveMode.MIN_ERROR
    assert np.isfinite(result.task_level_diagnostics[0].residual_norm)
    assert result.task_level_diagnostics[0].target_norm > 0.0
    assert result.task_level_diagnostics[0].normalized_residual >= 0.0

    achieved = np.asarray(ee.get_jacobian(), dtype=float) @ dq
    scaled_target = result.task_scales[0] * np.asarray(ee.get_velocity(), dtype=float)
    assert np.linalg.norm(achieved - scaled_target) < 5e-3
    assert ee.priority == 9
    assert posture.priority == 0


def test_explicit_order_overrides_priority_and_clear_restores_legacy(tmp_path):
    """Level order is authoritative without changing legacy priority fields."""
    robot = _make_two_joint_robot(tmp_path)
    solver = eik.KinematicsSolver(robot)
    q = np.zeros(robot.nq, dtype=float)

    positive = solver.add_joint_task("positive", "joint1", target_value=1.0)
    positive.priority = 10
    negative = solver.add_joint_task("negative", "joint1", target_value=-1.0)
    negative.priority = 0

    solver.configure_task_stack(
        eik.TaskStackConfig([_level("first", ["positive"]), _level("second", ["negative"])])
    )
    explicit_result = solver.solve_velocity(q, apply_limits=False)

    assert explicit_result.status == eik.SolverStatus.SUCCESS
    assert explicit_result.joint_velocities[0] > 0.5
    assert positive.priority == 10
    assert negative.priority == 0

    solver.clear_task_stack()
    legacy_result = solver.solve_velocity(q, apply_limits=False)

    assert solver.has_explicit_task_stack() is False
    assert solver.task_stack_config is None
    assert legacy_result.status == eik.SolverStatus.SUCCESS
    assert legacy_result.prioritized_status == eik.SolverStatus.SUCCESS
    assert legacy_result.hierarchy_solve_path == eik.HierarchySolvePath.LEGACY_PRIORITY
    assert legacy_result.higher_level_preservation_active is True
    assert legacy_result.joint_velocities[0] < -0.5
    assert legacy_result.task_level_diagnostics == []


def test_explicit_stack_omits_unlisted_active_tasks_until_cleared(tmp_path):
    """Only named members participate; clearing restores every active task."""
    robot = _make_two_joint_robot(tmp_path)
    solver = eik.KinematicsSolver(robot)
    q = np.zeros(robot.nq, dtype=float)

    included = solver.add_joint_task("included", "joint1", target_value=1.0)
    included.priority = 1
    omitted = solver.add_joint_task("omitted", "joint1", target_value=-1.0)
    omitted.priority = 0

    solver.configure_task_stack(eik.TaskStackConfig([_level("only", ["included"])]))
    explicit_result = solver.solve_velocity(q, apply_limits=False)

    assert included.active is True
    assert omitted.active is True
    assert explicit_result.status == eik.SolverStatus.SUCCESS
    assert explicit_result.joint_velocities[0] > 0.5
    assert explicit_result.task_level_diagnostics[0].task_names == ["included"]

    solver.clear_task_stack()
    legacy_result = solver.solve_velocity(q, apply_limits=False)

    assert legacy_result.status == eik.SolverStatus.SUCCESS
    assert legacy_result.joint_velocities[0] < -0.5
    assert legacy_result.task_level_diagnostics == []


def test_same_level_peers_are_canonical_and_one_backend_objective(tmp_path):
    """Peer permutation cannot become an implicit hierarchy."""
    robot = _make_two_joint_robot(tmp_path)
    solver = eik.KinematicsSolver(robot)
    q = np.zeros(robot.nq, dtype=float)

    z_task = solver.add_joint_task("z_task", "joint1", target_value=0.3)
    z_task.solve_mode = eik.TaskSolveMode.SCALE
    z_task.allow_min_error_fallback = True
    a_task = solver.add_joint_task("a_task", "joint2", target_value=-0.4)
    a_task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    a_task.allow_min_error_fallback = False

    first_config = eik.TaskStackConfig(
        [_level("peers", ["z_task", "a_task"], eik.TaskSolveMode.MIN_ERROR)]
    )
    solver.configure_task_stack(first_config)
    first_result = solver.solve_velocity(q, apply_limits=False)

    assert solver.task_stack_config.levels[0].task_names == ["a_task", "z_task"]
    assert first_result.hierarchy_solve_path == eik.HierarchySolvePath.EXPLICIT_SNS
    assert first_result.higher_level_preservation_active is True
    assert len(first_result.task_scales) == 1
    assert len(first_result.task_level_diagnostics) == 1
    assert (
        first_result.task_level_diagnostics[0].effective_solve_mode == eik.TaskSolveMode.MIN_ERROR
    )
    assert first_result.task_level_diagnostics[0].target_norm > 0.0
    assert first_result.task_level_diagnostics[0].normalized_residual >= 0.0
    assert z_task.solve_mode == eik.TaskSolveMode.SCALE
    assert z_task.allow_min_error_fallback is True
    assert a_task.solve_mode == eik.TaskSolveMode.MIN_ERROR

    solver.configure_task_stack(
        eik.TaskStackConfig([_level("peers", ["a_task", "z_task"], eik.TaskSolveMode.MIN_ERROR)])
    )
    second_result = solver.solve_velocity(q, apply_limits=False)

    assert np.array_equal(
        np.asarray(first_result.joint_velocities), np.asarray(second_result.joint_velocities)
    )
    assert second_result.task_level_diagnostics[0].task_names == ["a_task", "z_task"]
    assert second_result.task_level_diagnostics[0].normalized_residual == pytest.approx(
        first_result.task_level_diagnostics[0].normalized_residual
    )


def test_primary_infeasibility_and_lower_level_degradation_are_distinct(tmp_path):
    robot = _make_two_joint_robot(tmp_path)
    solver = eik.KinematicsSolver(robot)
    runtime = eik.SolverRuntimeConfig()
    runtime.weighted_fallback_enabled = False
    solver.configure_runtime(runtime)
    q = np.array([2.0, 0.0], dtype=float)

    blocked_primary = solver.add_joint_task("blocked_primary", "joint1", target_value=3.0)
    helpful_secondary = solver.add_joint_task("helpful_secondary", "joint2", target_value=0.8)
    solver.configure_task_stack(
        eik.TaskStackConfig(
            [_level("primary", ["blocked_primary"]), _level("secondary", ["helpful_secondary"])]
        )
    )

    infeasible = solver.solve_velocity(q, apply_limits=True)

    assert infeasible.status == eik.SolverStatus.INFEASIBLE
    assert infeasible.prioritized_status == eik.SolverStatus.INFEASIBLE
    assert infeasible.hierarchy_solve_path == eik.HierarchySolvePath.EXPLICIT_SNS
    assert infeasible.higher_level_preservation_active is True
    assert infeasible.task_level_diagnostics[0].normalized_residual > 0.9
    assert infeasible.task_level_diagnostics[1].normalized_residual < 1e-9
    assert infeasible.joint_velocities[1] > 0.1

    solver.clear_tasks()
    q = np.array([2.0, 0.0], dtype=float)
    helpful_primary = solver.add_joint_task("helpful_primary", "joint2", target_value=0.8)
    blocked_secondary = solver.add_joint_task("blocked_secondary", "joint1", target_value=3.0)
    solver.configure_task_stack(
        eik.TaskStackConfig(
            [_level("primary", ["helpful_primary"]), _level("secondary", ["blocked_secondary"])]
        )
    )

    degraded = solver.solve_velocity(q, apply_limits=True)

    assert degraded.status == eik.SolverStatus.SUCCESS
    assert degraded.prioritized_status == eik.SolverStatus.SUCCESS
    assert degraded.hierarchy_solve_path == eik.HierarchySolvePath.EXPLICIT_SNS
    assert degraded.higher_level_preservation_active is True
    assert degraded.task_level_diagnostics[0].normalized_residual < 1e-9
    assert degraded.task_level_diagnostics[1].normalized_residual > 0.9
    assert degraded.joint_velocities[1] > 0.1


def test_lower_level_addition_preserves_higher_level_achieved_output(tmp_path):
    robot = _make_two_joint_robot(tmp_path)
    solver = eik.KinematicsSolver(robot)
    q = np.array([0.4, -0.7], dtype=float)
    robot.update_configuration(q)

    ee = solver.add_frame_task("ee", "tool", eik.TaskType.FRAME_POSITION)
    current_position = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)
    ee.set_target_position(current_position + np.array([0.0, 0.04, 0.0]))

    posture = solver.add_posture_task("posture")
    posture.weight = 0.2
    posture.set_target_configuration(np.zeros(robot.nq, dtype=float))

    solver.configure_task_stack(eik.TaskStackConfig([_level("tracking", ["ee"])]))
    tracking_only = solver.solve_velocity(q, apply_limits=False)

    solver.configure_task_stack(
        eik.TaskStackConfig(
            [
                _level("tracking", ["ee"]),
                _level("posture", ["posture"], eik.TaskSolveMode.MIN_ERROR),
            ]
        )
    )
    stacked = solver.solve_velocity(q, apply_limits=False)

    primary_jacobian = np.asarray(ee.get_jacobian(), dtype=float)
    achieved_tracking_only = primary_jacobian @ np.asarray(
        tracking_only.joint_velocities, dtype=float
    )
    achieved_stacked = primary_jacobian @ np.asarray(stacked.joint_velocities, dtype=float)

    assert tracking_only.status == eik.SolverStatus.SUCCESS
    assert stacked.status == eik.SolverStatus.SUCCESS
    assert stacked.higher_level_preservation_active is True
    assert np.allclose(achieved_tracking_only, achieved_stacked, atol=5e-4)
    assert stacked.task_level_diagnostics[0].normalized_residual == pytest.approx(
        tracking_only.task_level_diagnostics[0].normalized_residual, abs=5e-4
    )


@pytest.mark.parametrize(
    ("levels", "message"),
    [
        ([_level("level", ["missing"])], "not registered"),
        ([_level("level", ["one", "one"])], "more than once"),
        ([_level("first", ["one"]), _level("second", ["one"])], "more than once"),
    ],
)
def test_invalid_membership_is_rejected(tmp_path, levels, message):
    robot = _make_two_joint_robot(tmp_path)
    solver = eik.KinematicsSolver(robot)
    solver.add_joint_task("one", "joint1", target_value=0.1)

    with pytest.raises(ValueError, match=message):
        solver.configure_task_stack(eik.TaskStackConfig(levels))


def test_inactive_and_removed_members_invalidate_configured_stack(tmp_path):
    robot = _make_two_joint_robot(tmp_path)
    solver = eik.KinematicsSolver(robot)
    task = solver.add_joint_task("one", "joint1", target_value=0.1)
    config = eik.TaskStackConfig([_level("primary", ["one"])])

    task.active = False
    with pytest.raises(ValueError, match="inactive"):
        solver.configure_task_stack(config)

    task.active = True
    solver.configure_task_stack(config)
    task.active = False
    inactive_result = solver.solve_velocity(np.zeros(robot.nq), apply_limits=False)
    assert inactive_result.status == eik.SolverStatus.INVALID_INPUT
    assert "configured task stack is invalid" in inactive_result.status_message
    assert "inactive" in inactive_result.status_message

    task.active = True
    solver.remove_task("one")
    removed_result = solver.solve_velocity(np.zeros(robot.nq), apply_limits=False)
    assert removed_result.status == eik.SolverStatus.INVALID_INPUT
    assert "not registered" in removed_result.status_message
    assert solver.has_explicit_task_stack() is True

    solver.clear_task_stack()
    cleared_result = solver.solve_velocity(np.zeros(robot.nq), apply_limits=False)
    assert cleared_result.status == eik.SolverStatus.SUCCESS


def test_incompatible_level_fallback_policy_is_rejected(tmp_path):
    robot = _make_two_joint_robot(tmp_path)
    solver = eik.KinematicsSolver(robot)
    solver.add_joint_task("one", "joint1", target_value=0.1)

    with pytest.raises(ValueError, match="already MIN_ERROR"):
        solver.configure_task_stack(
            eik.TaskStackConfig([_level("primary", ["one"], eik.TaskSolveMode.MIN_ERROR, True)])
        )


def test_weighted_fallback_stays_separate_from_level_diagnostics(tmp_path):
    """A weighted recovery result is not attributed to explicit SNS levels."""
    robot = _make_two_joint_robot(tmp_path)
    solver = eik.KinematicsSolver(robot)
    q = np.array([2.0, 0.0], dtype=float)

    blocked = solver.add_joint_task("blocked", "joint1", target_value=3.0)
    blocked.weight = 1.0
    solver.configure_task_stack(eik.TaskStackConfig([_level("primary", ["blocked"])]))

    runtime = eik.SolverRuntimeConfig()
    runtime.weighted_fallback_enabled = True
    solver.configure_runtime(runtime)

    result = solver.solve_velocity(q, apply_limits=True)

    assert result.status == eik.SolverStatus.SUCCESS
    assert result.prioritized_status == eik.SolverStatus.INFEASIBLE
    assert "primary task scale collapsed" in result.prioritized_status_message
    assert result.weighted_fallback_used is True
    assert result.hierarchy_solve_path == eik.HierarchySolvePath.WEIGHTED_FALLBACK
    assert result.hierarchy_backend == eik.TaskStackBackend.SNS
    assert result.higher_level_preservation_active is False
    assert result.task_level_diagnostics == []
    assert result.joint_velocities[0] <= 1e-9


def test_registered_position_step_reports_explicit_level_diagnostics(tmp_path):
    """Position stepping preserves the final inner stack diagnostics."""
    robot = _make_two_joint_robot(tmp_path)
    solver = eik.KinematicsSolver(robot)
    q = np.array([0.2, -0.4], dtype=float)
    robot.update_configuration(q)

    tracking = solver.add_frame_task("tracking", "tool", eik.TaskType.FRAME_POSITION)
    current_position = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)

    posture = solver.add_posture_task("posture")
    posture.set_target_configuration(np.zeros(robot.nq, dtype=float))

    solver.configure_task_stack(
        eik.TaskStackConfig(
            [
                _level("tracking_level", ["tracking"]),
                _level("posture_level", ["posture"], eik.TaskSolveMode.MIN_ERROR),
            ]
        )
    )

    target = np.eye(4, dtype=float)
    target[:3, 3] = current_position + np.array([0.0, 0.02, 0.0])
    options = eik.PositionStepOptions()
    options.position_gain = 2.0
    options.max_steps = 1
    options.dt = 0.01

    result = solver.solve_position_step(q, target, "tracking", options)

    assert result.status == eik.SolverStatus.SUCCESS
    assert result.iterations_used == 1
    assert not np.array_equal(np.asarray(result.q_solution), q)
    assert [diagnostic.name for diagnostic in result.task_level_diagnostics] == [
        "tracking_level",
        "posture_level",
    ]
    assert [diagnostic.task_names for diagnostic in result.task_level_diagnostics] == [
        ["tracking"],
        ["posture"],
    ]
    for index, diagnostic in enumerate(result.task_level_diagnostics):
        assert diagnostic.scale == pytest.approx(result.task_scales[index])
        assert diagnostic.residual_norm == pytest.approx(result.task_errors[index])
    assert result.task_level_diagnostics[1].effective_solve_mode == eik.TaskSolveMode.MIN_ERROR


def test_lexicographic_backend_uses_reachable_primary_components(tmp_path):
    """Minimum residual uses free rows where direction-preserving SNS collapses."""
    robot = _make_two_joint_robot(tmp_path)
    solver = eik.KinematicsSolver(robot)
    q = np.zeros(robot.nq, dtype=float)

    primary = solver.add_posture_task("primary")
    primary.set_target_velocity(np.array([1.0, 1.0], dtype=float))
    runtime = solver.runtime_config()
    runtime.weighted_fallback_enabled = False
    solver.configure_runtime(runtime)
    solver.set_linear_velocity_constraints(
        np.array([[1.0, 0.0]], dtype=float),
        np.array([0.0], dtype=float),
        np.array([0.0], dtype=float),
    )

    default_stack = eik.TaskStackConfig([_level("primary", ["primary"])])
    assert default_stack.backend == eik.TaskStackBackend.SNS
    solver.configure_task_stack(default_stack)
    sns = solver.solve_velocity(q, apply_limits=False)

    solver.configure_task_stack(_lex_stack([_level("primary", ["primary"])]))
    lex = solver.solve_velocity(q, apply_limits=False)

    assert sns.status == eik.SolverStatus.INFEASIBLE
    assert sns.hierarchy_solve_path == eik.HierarchySolvePath.EXPLICIT_SNS
    assert abs(sns.joint_velocities[1]) < 1e-9
    assert lex.status == eik.SolverStatus.SUCCESS
    assert lex.prioritized_status == eik.SolverStatus.SUCCESS
    assert lex.hierarchy_solve_path == eik.HierarchySolvePath.EXPLICIT_LEXICOGRAPHIC_LEAST_SQUARES
    assert lex.hierarchy_backend == eik.TaskStackBackend.LEXICOGRAPHIC_LEAST_SQUARES
    assert lex.higher_level_preservation_active is True
    np.testing.assert_allclose(lex.joint_velocities, np.array([0.0, 1.0]), atol=1e-8)
    assert lex.task_level_diagnostics[0].configured_solve_mode == eik.TaskSolveMode.SCALE
    assert lex.task_level_diagnostics[0].effective_solve_mode == eik.TaskSolveMode.MIN_ERROR
    assert lex.task_level_diagnostics[0].scale == pytest.approx(-1.0)
    assert lex.task_level_diagnostics[0].residual_norm == pytest.approx(1.0, abs=1e-8)


def test_lexicographic_primary_is_preserved_under_conflicting_secondary(tmp_path):
    robot = _make_two_joint_robot(tmp_path)
    solver = eik.KinematicsSolver(robot)
    q = np.zeros(robot.nq, dtype=float)

    primary = solver.add_joint_task("primary", "joint1", target_value=0.5)
    secondary = solver.add_joint_task("secondary", "joint1", target_value=-1.0)
    solver.configure_task_stack(
        _lex_stack([_level("primary", ["primary"]), _level("secondary", ["secondary"])])
    )

    result = solver.solve_velocity(q, apply_limits=False)

    assert result.status == eik.SolverStatus.SUCCESS
    assert result.joint_velocities[0] == pytest.approx(0.5, abs=1e-8)
    assert result.task_level_diagnostics[0].residual_norm == pytest.approx(0.0, abs=1e-8)
    assert result.task_level_diagnostics[1].residual_norm > 1.4
    assert primary.solve_mode == eik.TaskSolveMode.SCALE
    assert secondary.solve_mode == eik.TaskSolveMode.SCALE


def test_lexicographic_same_level_permutation_is_invariant(tmp_path):
    robot = _make_two_joint_robot(tmp_path)
    solver = eik.KinematicsSolver(robot)
    q = np.zeros(robot.nq, dtype=float)

    solver.add_joint_task("z_positive", "joint1", target_value=0.5)
    solver.add_joint_task("a_negative", "joint1", target_value=-1.0)
    solver.add_joint_task("middle", "joint2", target_value=0.3)
    solver.configure_task_stack(
        _lex_stack([_level("peers", ["z_positive", "middle", "a_negative"])])
    )
    forward = solver.solve_velocity(q, apply_limits=False)

    solver.configure_task_stack(
        _lex_stack([_level("peers", ["a_negative", "z_positive", "middle"])])
    )
    reverse = solver.solve_velocity(q, apply_limits=False)

    assert forward.status == eik.SolverStatus.SUCCESS
    assert reverse.status == eik.SolverStatus.SUCCESS
    assert solver.task_stack_config.levels[0].task_names == [
        "a_negative",
        "middle",
        "z_positive",
    ]
    np.testing.assert_array_equal(forward.joint_velocities, reverse.joint_velocities)
    assert forward.joint_velocities[0] == pytest.approx(-0.25, abs=1e-8)
    assert forward.joint_velocities[1] == pytest.approx(0.3, abs=1e-8)


def test_lexicographic_rank_deficient_primary_leaves_secondary_freedom(tmp_path):
    robot = _make_two_joint_robot(tmp_path)
    solver = eik.KinematicsSolver(robot)
    q = np.zeros(robot.nq, dtype=float)

    solver.add_joint_task("primary_a", "joint1", target_value=0.3)
    solver.add_joint_task("primary_b", "joint1", target_value=0.3)
    solver.add_joint_task("secondary", "joint2", target_value=0.4)
    solver.set_joint_metric_weights(np.array([4.0, 1.0], dtype=float))
    solver.configure_task_stack(
        _lex_stack(
            [
                _level("rank_deficient", ["primary_b", "primary_a"]),
                _level("secondary", ["secondary"]),
            ]
        )
    )

    result = solver.solve_velocity(q, apply_limits=False)

    assert result.status == eik.SolverStatus.SUCCESS
    np.testing.assert_allclose(result.joint_velocities, np.array([0.3, 0.4]), atol=1e-8)
    assert [level.residual_norm for level in result.task_level_diagnostics] == pytest.approx(
        [0.0, 0.0], abs=1e-8
    )


def test_lexicographic_global_infeasibility_is_not_task_degradation(tmp_path):
    robot = _make_two_joint_robot(tmp_path)
    solver = eik.KinematicsSolver(robot)
    q = np.zeros(robot.nq, dtype=float)
    solver.add_joint_task("task", "joint2", target_value=0.4)
    solver.configure_task_stack(_lex_stack([_level("task", ["task"])]))
    solver.set_linear_velocity_constraints(
        np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float),
        np.array([1.0, -1e10], dtype=float),
        np.array([1e10, 0.0], dtype=float),
    )

    result = solver.solve_velocity(q, apply_limits=False)

    assert result.status == eik.SolverStatus.INFEASIBLE
    assert result.prioritized_status == eik.SolverStatus.INFEASIBLE
    assert "global linear constraints are infeasible" in result.status_message
    assert result.hierarchy_backend == eik.TaskStackBackend.LEXICOGRAPHIC_LEAST_SQUARES
    assert (
        result.hierarchy_solve_path == eik.HierarchySolvePath.EXPLICIT_LEXICOGRAPHIC_LEAST_SQUARES
    )
    assert result.higher_level_preservation_active is False
    assert result.weighted_fallback_used is False
    assert result.task_level_diagnostics == []


def test_registered_position_step_reports_lexicographic_backend(tmp_path):
    robot = _make_two_joint_robot(tmp_path)
    solver = eik.KinematicsSolver(robot)
    q = np.array([0.2, -0.4], dtype=float)
    robot.update_configuration(q)

    tracking = solver.add_frame_task("tracking", "tool", eik.TaskType.FRAME_POSITION)
    current_position = np.asarray(robot.get_frame_pose("tool").translation, dtype=float)
    posture = solver.add_posture_task("posture")
    posture.set_target_configuration(np.zeros(robot.nq, dtype=float))
    solver.configure_task_stack(
        _lex_stack([_level("tracking", ["tracking"]), _level("posture", ["posture"])])
    )

    target = np.eye(4, dtype=float)
    target[:3, 3] = current_position + np.array([0.0, 0.02, 0.0])
    options = eik.PositionStepOptions()
    options.position_gain = 2.0
    options.max_steps = 1
    options.dt = 0.01

    result = solver.solve_position_step(q, target, "tracking", options)

    assert result.status == eik.SolverStatus.SUCCESS
    assert result.hierarchy_backend == eik.TaskStackBackend.LEXICOGRAPHIC_LEAST_SQUARES
    assert result.diagnostics.hierarchy_backend == eik.TaskStackBackend.LEXICOGRAPHIC_LEAST_SQUARES
    assert (
        result.hierarchy_solve_path == eik.HierarchySolvePath.EXPLICIT_LEXICOGRAPHIC_LEAST_SQUARES
    )
    assert result.higher_level_preservation_active is True
    assert len(result.task_level_diagnostics) == 2


def test_lexicographic_rejects_sns_only_level_recovery_policies(tmp_path):
    robot = _make_two_joint_robot(tmp_path)
    solver = eik.KinematicsSolver(robot)
    solver.add_joint_task("task", "joint1", target_value=0.1)

    with pytest.raises(ValueError, match="cannot use SCALE_ELASTIC"):
        solver.configure_task_stack(
            _lex_stack([_level("task", ["task"], eik.TaskSolveMode.SCALE_ELASTIC)])
        )
    with pytest.raises(ValueError, match="cannot enable an SNS MIN_ERROR fallback"):
        solver.configure_task_stack(
            _lex_stack([_level("task", ["task"], allow_min_error_fallback=True)])
        )
