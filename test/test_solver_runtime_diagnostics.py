from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

import embodik as eik


def _create_two_joint_urdf() -> str:
    urdf = """<?xml version="1.0"?>
<robot name="runtime_diagnostics_test_robot">
  <link name="base_link"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="100"/>
  </joint>
  <link name="link1"/>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="ee"/>
    <origin xyz="0.2 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="100"/>
  </joint>
  <link name="ee"/>
</robot>
"""
    fd, path = tempfile.mkstemp(suffix=".urdf")
    with os.fdopen(fd, "w") as f:
        f.write(urdf)
    return path


def _solve_one_step():
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01
        task = solver.add_frame_task("ee_task", "ee")
        task.priority = 0

        q = np.zeros(robot.nq, dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[0, 3] += 0.01

        opts = eik.PositionStepOptions()
        opts.max_steps = 1
        return solver.solve_position_step(q, target, "ee_task", opts)
    finally:
        os.remove(urdf_path)


def _solve_one_step_with_options(options_factory):
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01
        task = solver.add_frame_task("ee_task", "ee")
        task.priority = 0

        q = np.zeros(robot.nq, dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[0, 3] += 0.01

        return solver.solve_position_step(q, target, "ee_task", options_factory(solver))
    finally:
        os.remove(urdf_path)


def test_solve_diagnostics_mirrors_existing_result_fields() -> None:
    result = _solve_one_step()

    _assert_diagnostics_mirror_result(result)


def _assert_diagnostics_mirror_result(result) -> None:
    diagnostics = result.diagnostics

    assert diagnostics.collision_rejection_count == result.collision_rejection_count
    assert diagnostics.stall_escape_count == result.stall_escape_count
    assert diagnostics.condition_number == result.condition_number
    assert list(diagnostics.task_scales) == list(result.task_scales)
    assert list(diagnostics.task_used_fallback) == list(result.task_used_fallback)
    assert list(diagnostics.task_modes_effective) == list(result.task_modes_effective)
    expected_any = result.collision_rejection_count > 0 or result.stall_escape_count > 0
    assert diagnostics.any_intervention is expected_any


def test_solve_diagnostics_mirror_invalid_input_result() -> None:
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        task = solver.add_frame_task("ee_task", "ee")
        task.priority = 0

        q = np.zeros(robot.nq, dtype=float)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)

        opts = eik.PositionStepOptions()
        opts.locked_joint_indices = [robot.nv]
        result = solver.solve_position_step(q, target, "ee_task", opts)

        assert result.status == eik.SolverStatus.INVALID_INPUT
        _assert_diagnostics_mirror_result(result)
    finally:
        os.remove(urdf_path)


def test_solve_diagnostics_mirror_infeasible_position_ik_result() -> None:
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        q = np.zeros(robot.nq, dtype=float)
        robot.update_configuration(q)
        pose = robot.get_frame_pose("ee")
        target = np.eye(4, dtype=float)
        target[:3, :3] = np.array(pose.rotation, dtype=float)
        target[:3, 3] = np.array(pose.translation, dtype=float)
        target[0, 3] += 0.05

        opts = eik.PositionIKOptions()
        opts.max_iterations = 6
        opts.stagnation_iterations = 2
        opts.position_gain = 30.0
        opts.orientation_gain = 30.0
        opts.classify_stagnation_as_no_progress = False
        opts.excluded_joint_indices = [0, 1]
        result = solver.solve_position(q, target, "ee", opts)

        assert result.status == eik.SolverStatus.INFEASIBLE
        _assert_diagnostics_mirror_result(result)
    finally:
        os.remove(urdf_path)


def test_solve_diagnostics_mirror_collision_limited_panda_result() -> None:
    pytest.importorskip("robot_descriptions.panda_description")
    pytest.importorskip("viser")

    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    collision_example = importlib.import_module("02_collision_aware_IK")

    cfg = collision_example.resolve_robot_configuration("panda")

    robot = eik.RobotModel(str(cfg.urdf_path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    try:
        solver.configure_collision_constraint(
            min_distance=0.02,
            include_pairs=[],
            exclude_pairs=cfg.collision_exclusions,
            max_constraints=3,
        )
    except RuntimeError as exc:
        pytest.skip(f"Panda collision geometry unavailable: {exc}")
    task = solver.add_frame_task("ee_task", "panda_hand")
    task.priority = 0

    q = np.zeros(robot.nq, dtype=float)
    default_q = np.asarray(cfg.default_configuration, dtype=float)
    q[: default_q.size] = default_q
    if robot.nq > default_q.size:
        q[default_q.size :] = 0.02
    robot.update_configuration(q)
    if solver.evaluate_min_collision_distance(q) < 0.02:
        pytest.skip("Panda seed is already inside the collision threshold")

    pose = robot.get_frame_pose("panda_hand")
    target = np.eye(4, dtype=float)
    target[:3, :3] = np.array(pose.rotation, dtype=float)
    target[:3, 3] = np.array(pose.translation, dtype=float) + np.array([0.5, 0.0, 0.0])

    opts = eik.PositionStepOptions()
    opts.max_steps = 1
    opts.position_gain = 10.0
    result = solver.solve_position_step(q, target, "ee_task", opts)

    assert result.status in (
        eik.SolverStatus.SUCCESS,
        eik.SolverStatus.INFEASIBLE,
        eik.SolverStatus.NO_PROGRESS,
        eik.SolverStatus.COLLISION_VIOLATED,
    )
    assert result.task_scales
    assert min(result.task_scales) < 1.0
    _assert_diagnostics_mirror_result(result)


def test_solver_runtime_config_materializes_position_step_defaults() -> None:
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)

        cfg = eik.SolverRuntimeConfig()
        cfg.damping = 0.123
        assert cfg.refine_partial_scale is False
        cfg.refine_partial_scale = True
        cfg.position_step_max_steps = 4
        cfg.adaptive_dt = True
        cfg.adaptive_dt_max_scale = 9.0
        cfg.adaptive_dt_reference_distance = 0.07

        solver.configure_runtime(cfg)
        solver.set_damping(0.321)
        stored = solver.runtime_config()
        opts = solver.make_position_step_options()

        assert stored.damping == 0.321
        assert stored.refine_partial_scale is True
        solver.reset_adaptive_state()
        assert solver.runtime_config().refine_partial_scale is True
        assert stored.position_step_max_steps == cfg.position_step_max_steps
        assert stored.adaptive_dt is True
        assert stored.adaptive_dt_max_scale == cfg.adaptive_dt_max_scale
        assert stored.adaptive_dt_reference_distance == cfg.adaptive_dt_reference_distance
        assert opts.max_steps == cfg.position_step_max_steps
        assert opts.adaptive_dt is True
        assert opts.adaptive_dt_max_scale == cfg.adaptive_dt_max_scale
        assert opts.adaptive_dt_reference_distance == cfg.adaptive_dt_reference_distance

        opts.max_steps = 1
        assert solver.make_position_step_options().max_steps == cfg.position_step_max_steps
    finally:
        os.remove(urdf_path)


def test_default_runtime_config_matches_position_step_defaults() -> None:
    cfg = eik.SolverRuntimeConfig()
    opts = eik.PositionStepOptions()

    assert cfg.position_step_max_steps == opts.max_steps
    assert cfg.adaptive_dt == opts.adaptive_dt
    assert cfg.adaptive_dt_max_scale == opts.adaptive_dt_max_scale
    assert cfg.adaptive_dt_reference_distance == opts.adaptive_dt_reference_distance


def test_default_runtime_config_preserves_default_position_step_result() -> None:
    baseline = _solve_one_step_with_options(lambda _solver: eik.PositionStepOptions())

    def configured_default_options(solver):
        solver.configure_runtime(eik.SolverRuntimeConfig())
        return solver.make_position_step_options()

    configured = _solve_one_step_with_options(configured_default_options)

    assert configured.status == baseline.status
    assert configured.iterations_used == baseline.iterations_used
    assert np.allclose(configured.q_solution, baseline.q_solution)
    assert configured.position_error == baseline.position_error
    assert configured.orientation_error == baseline.orientation_error


def test_normalized_recovery_tolerances_can_be_updated_online() -> None:
    urdf_path = _create_two_joint_urdf()
    try:
        solver = eik.KinematicsSolver(eik.RobotModel(urdf_path, floating_base=False))
        cfg = solver.runtime_config()
        assert not cfg.normalized_pose_recovery_enabled
        assert cfg.normalized_pose_position_tolerance == pytest.approx(0.005)
        assert cfg.normalized_pose_orientation_tolerance == pytest.approx(0.01)
        cfg.normalized_pose_recovery_enabled = True
        for position, orientation in ((0.02, 0.1), (0.005, 0.01)):
            cfg.normalized_pose_position_tolerance = position
            cfg.normalized_pose_orientation_tolerance = orientation
            solver.configure_runtime(cfg)
            stored = solver.runtime_config()
            assert stored.normalized_pose_position_tolerance == pytest.approx(position)
            assert stored.normalized_pose_orientation_tolerance == pytest.approx(orientation)
    finally:
        os.remove(urdf_path)


@pytest.mark.parametrize(
    "field", ["normalized_pose_position_tolerance", "normalized_pose_orientation_tolerance"]
)
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_normalized_recovery_rejects_invalid_tolerances(field: str, value: float) -> None:
    urdf_path = _create_two_joint_urdf()
    try:
        solver = eik.KinematicsSolver(eik.RobotModel(urdf_path, floating_base=False))
        cfg = solver.runtime_config()
        cfg.normalized_pose_recovery_enabled = True
        setattr(cfg, field, value)
        with pytest.raises(ValueError, match="normalized pose tolerances"):
            solver.configure_runtime(cfg)
        assert not solver.runtime_config().normalized_pose_recovery_enabled
    finally:
        os.remove(urdf_path)


@pytest.mark.parametrize("kind", ["wrong_size", "nonfinite"])
def test_normalized_recovery_reference_validation(kind: str) -> None:
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        solver = eik.KinematicsSolver(robot)
        cfg = solver.runtime_config()
        reference = np.zeros(robot.nq + (kind == "wrong_size"))
        if kind == "nonfinite":
            reference[0] = np.nan
        cfg.normalized_pose_recovery_reference = reference
        with pytest.raises(ValueError, match="normalized recovery reference"):
            solver.configure_runtime(cfg)
        assert solver.runtime_config().normalized_pose_recovery_reference.size == 0
    finally:
        os.remove(urdf_path)


def test_normalized_recovery_guide_does_not_reset_robot_configuration() -> None:
    urdf_path = _create_two_joint_urdf()
    try:
        robot = eik.RobotModel(urdf_path, floating_base=False)
        current = np.zeros(robot.nq)
        reference = np.array([0.4, 0.2])
        robot.update_configuration(reference)
        target_pose = robot.get_frame_pose("ee")
        target = np.eye(4)
        target[:3, :3] = target_pose.rotation
        target[:3, 3] = target_pose.translation
        robot.update_configuration(current)
        solver = eik.KinematicsSolver(robot)
        solver.enable_position_limits(True)
        solver.enable_velocity_limits(True)
        task = solver.add_frame_task("goal", "ee", eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.solve_mode = eik.TaskSolveMode.SCALE
        task.allow_min_error_fallback = True
        task.set_target_pose(target[:3, 3], target[:3, :3])
        posture = solver.add_posture_task("posture")
        posture.priority = 2
        posture.set_target_configuration(current)
        cfg = solver.runtime_config()
        cfg.normalized_pose_recovery_enabled = True
        cfg.normalized_pose_recovery_reference = reference
        solver.configure_runtime(cfg)
        np.testing.assert_array_equal(robot.get_current_configuration(), current)
        options = eik.PositionStepOptions()
        options.dt = 0.01
        options.max_steps = 1
        options.primary_allow_min_error_fallback = True
        result = solver.solve_position_step(current, target, "goal", options)
        applied = np.asarray(result.q_solution)
        assert 0 < np.linalg.norm(applied - current) < 0.1 * np.linalg.norm(reference - current)
        assert "pose-solution guide accepted" in result.status_message
        np.testing.assert_array_equal(current, np.zeros(2))
    finally:
        os.remove(urdf_path)


@pytest.mark.parametrize(
    "case", ["changed_target", "outside_limits", "satisfied_target", "locally_regressive"]
)
def test_inapplicable_recovery_guide_matches_unguided_result(case: str) -> None:
    urdf_path = _create_two_joint_urdf()
    try:
        results = []
        current = np.zeros(2)
        target_q = current if case == "satisfied_target" else np.array([0.1, -0.3])
        reference = {
            "changed_target": np.array([0.4, 0.2]),
            "outside_limits": np.array([4.0, 0.0]),
            "satisfied_target": current,
            "locally_regressive": np.array([-3.0, 0.0]),
        }[case]
        if case == "locally_regressive":
            # The endpoint has the desired pose, but its direct joint route
            # rotates away from the short Cartesian arc at the current state.
            current = np.array([2.7, 0.0])
            target_q = reference
        for use_guide in (False, True):
            robot = eik.RobotModel(urdf_path, floating_base=False)
            robot.update_configuration(target_q)
            pose = robot.get_frame_pose("ee")
            target = np.eye(4)
            target[:3, :3] = pose.rotation
            target[:3, 3] = pose.translation
            robot.update_configuration(current)
            solver = eik.KinematicsSolver(robot)
            task = solver.add_frame_task("goal", "ee", eik.TaskType.FRAME_POSE)
            task.priority = 0
            task.solve_mode = eik.TaskSolveMode.SCALE
            task.allow_min_error_fallback = True
            task.set_target_pose(target[:3, 3], target[:3, :3])
            posture = solver.add_posture_task("posture")
            posture.priority = 2
            posture.set_target_configuration(current)
            cfg = solver.runtime_config()
            cfg.normalized_pose_recovery_enabled = True
            if use_guide:
                cfg.normalized_pose_recovery_reference = reference
            solver.configure_runtime(cfg)
            options = eik.PositionStepOptions()
            options.dt = 0.01
            options.max_steps = 1
            options.primary_allow_min_error_fallback = True
            results.append(solver.solve_position_step(current, target, "goal", options))
        baseline, guided = results
        assert guided.status == baseline.status
        assert guided.status_message == baseline.status_message
        np.testing.assert_allclose(guided.q_solution, baseline.q_solution, atol=1e-12, rtol=0)
        assert "pose-solution guide accepted" not in guided.status_message
        if case == "satisfied_target":
            np.testing.assert_array_equal(guided.q_solution, current)
        if case == "locally_regressive":
            assert guided.q_solution[0] > current[0]
    finally:
        os.remove(urdf_path)


def test_pose_guide_accepts_consistent_zero_pose_rows(tmp_path: Path) -> None:
    links = ['<link name="base"/>']
    parent = "base"
    for index, axis in enumerate(("1 0 0", "0 1 0", "0 0 1")):
        child = f"link{index}"
        links.append(f"""<link name="{child}"/>
<joint name="joint{index}" type="prismatic"><parent link="{parent}"/>
<child link="{child}"/><axis xyz="{axis}"/>
<limit lower="-1" upper="1" effort="100" velocity="1"/></joint>""")
        parent = child
    urdf = tmp_path / "protected_rows.urdf"
    urdf.write_text('<robot name="protected_rows">' + "".join(links) + "</robot>")
    robot = eik.RobotModel(str(urdf), floating_base=False)
    current = np.zeros(3)
    reference = np.array([0.04, 0.02, 0.01])
    target = np.eye(4)
    target[:3, 3] = reference
    solver = eik.KinematicsSolver(robot)
    targets = []
    for name, gain in (("first", 10.0), ("second", 5.0)):
        task = solver.add_frame_task(name, parent, eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.solve_mode = eik.TaskSolveMode.SCALE
        task.allow_min_error_fallback = True
        task.set_target_pose(reference, np.eye(3))
        targets.append(eik.TaskTarget(name, target, gain, gain))
    posture = solver.add_posture_task("posture")
    posture.priority = 2
    posture.set_target_configuration(current)
    cfg = solver.runtime_config()
    cfg.normalized_pose_recovery_enabled = True
    cfg.normalized_pose_recovery_reference = reference
    solver.configure_runtime(cfg)
    options = eik.PositionStepOptions()
    options.dt = 0.01
    options.max_steps = 1
    options.primary_allow_min_error_fallback = True
    result = solver.solve_position_step(current, targets, options)
    assert result.status == eik.SolverStatus.SUCCESS
    assert "pose-solution guide accepted" in result.status_message
    assert 0 < np.linalg.norm(result.q_solution) < np.linalg.norm(reference)
    assert result.task_used_fallback[0]
    np.testing.assert_array_equal(current, np.zeros(3))
