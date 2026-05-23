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
        cfg.position_step_max_steps = 4
        cfg.adaptive_dt = True
        cfg.adaptive_dt_max_scale = 9.0
        cfg.adaptive_dt_reference_distance = 0.07

        solver.configure_runtime(cfg)
        solver.set_damping(0.321)
        stored = solver.runtime_config()
        opts = solver.make_position_step_options()

        assert stored.damping == 0.321
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
