"""Tests for acceleration-level constraint cascading in velocity IK.

Experiment 2: Add acceleration limits that tighten velocity bounds based on
the previous tick's joint velocities, inspired by RB-Y1 SDK's cascaded
constraint integration: position → velocity → acceleration bounds.

This produces smoother joint velocity profiles (lower jerk) without
degrading tracking accuracy, especially during fast direction changes.
"""

import pathlib

import numpy as np
import pytest

import embodik as eik

PANDA_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])


def _write_prismatic_limit_urdf(tmp_path: pathlib.Path) -> pathlib.Path:
    urdf_path = tmp_path / "prismatic_limit.urdf"
    urdf_path.write_text(
        """<?xml version="1.0"?>
<robot name="prismatic_limit">
  <link name="world"/>
  <link name="moving"/>
  <joint name="moving_slide" type="prismatic">
    <parent link="world"/>
    <child link="moving"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="0.2" effort="100" velocity="1"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    return urdf_path


@pytest.fixture
def panda_setup():
    pytest.importorskip("robot_descriptions.panda_description")
    from robot_descriptions.panda_description import URDF_PATH

    robot = eik.RobotModel(URDF_PATH, floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    return robot, solver


class TestAccelerationConstraints:

    def test_acceleration_limits_api_exists(self, panda_setup):
        """KinematicsSolver should expose acceleration limit methods."""
        _, solver = panda_setup
        assert hasattr(
            solver, "enable_acceleration_limits"
        ), "KinematicsSolver should have enable_acceleration_limits method"
        assert hasattr(
            solver, "set_acceleration_limits"
        ), "KinematicsSolver should have set_acceleration_limits method"

    def test_acceleration_limits_validate_shape_and_values(self, panda_setup):
        """Acceleration limits should reject invalid vectors before solving."""
        robot, solver = panda_setup

        with pytest.raises(ValueError, match="size nv"):
            solver.set_acceleration_limits(np.ones(robot.nv - 1))

        bad_limits = np.full(robot.nv, 1.0)
        bad_limits[0] = np.nan
        with pytest.raises(ValueError, match="finite and non-negative"):
            solver.set_acceleration_limits(bad_limits)

        bad_limits = np.full(robot.nv, 1.0)
        bad_limits[0] = -1.0
        with pytest.raises(ValueError, match="finite and non-negative"):
            solver.set_acceleration_limits(bad_limits)

    def test_sampled_stopping_bound_only_applies_with_acceleration_history(self, tmp_path):
        """Acceleration mode must not tighten the established position-only bound."""
        robot = eik.RobotModel(str(_write_prismatic_limit_urdf(tmp_path)))
        solver = eik.KinematicsSolver(robot)
        margin = 0.04
        acceleration_limit = 10.0
        dt = 0.01

        _, legacy_upper = solver.calculate_velocity_box_constraint(
            margin, margin, 100.0, acceleration_limit, dt
        )
        np.testing.assert_allclose(
            legacy_upper,
            np.sqrt(2.0 * acceleration_limit * margin),
            rtol=0.0,
            atol=1e-12,
        )

        solver.set_acceleration_limits(np.array([acceleration_limit], dtype=float))
        solver.enable_acceleration_limits(True)
        _, sampled_upper = solver.calculate_velocity_box_constraint(
            margin, margin, 100.0, acceleration_limit, dt
        )

        normalized_margin = margin / (acceleration_limit * dt * dt)
        braking_intervals = np.ceil(0.5 * (np.sqrt(1.0 + 8.0 * normalized_margin) - 1.0) - 1e-12)
        expected_sampled_upper = (
            acceleration_limit
            * dt
            * (normalized_margin + 0.5 * braking_intervals * (braking_intervals - 1.0))
            / braking_intervals
        )
        np.testing.assert_allclose(sampled_upper, expected_sampled_upper, rtol=0.0, atol=1e-12)
        assert sampled_upper < legacy_upper

    def test_acceleration_limits_bound_first_enabled_tick_from_rest(self, panda_setup):
        """Enabling acceleration limits should constrain the first solve from zero velocity."""
        robot, solver = panda_setup
        task = solver.add_frame_task("ee", "panda_hand", eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 1.0
        solver.enable_position_limits(True)

        q = PANDA_HOME.copy()
        robot.update_configuration(q)
        ee_pose = robot.get_frame_pose("panda_hand")
        target_pos = ee_pose.translation.copy()
        target_pos[0] += 0.3
        T = ee_pose.homogeneous()
        task.set_target_pose(target_pos, T[:3, :3])

        a_max = 2.0
        solver.set_acceleration_limits(np.full(robot.nv, a_max))
        solver.enable_acceleration_limits(True)
        result = solver.solve_velocity(q, apply_limits=True)

        assert result.status == eik.SolverStatus.SUCCESS
        assert np.max(np.abs(np.asarray(result.joint_velocities))) <= a_max * solver.dt * 1.02

    def test_multistep_position_velocity_matches_applied_outer_step(self, panda_setup):
        """A multi-step result must report the velocity that produced q_solution."""
        robot, solver = panda_setup
        task = solver.add_frame_task("ee", "panda_hand", eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 1.0
        solver.enable_position_limits(True)
        solver.enable_velocity_limits(True)

        q = PANDA_HOME.copy()
        robot.update_configuration(q)
        ee_pose = robot.get_frame_pose("panda_hand")
        target = ee_pose.homogeneous()
        target[0, 3] += 0.2

        options = eik.PositionStepOptions()
        options.dt = solver.dt
        options.max_steps = 3
        options.position_gain = 10.0
        options.orientation_gain = 10.0

        result = solver.solve_position_step(q, [eik.TaskTarget("ee", target)], options)

        assert result.status == eik.SolverStatus.SUCCESS
        applied_velocity = (np.asarray(result.q_solution) - q) / solver.dt
        np.testing.assert_allclose(
            np.asarray(result.joint_velocities),
            applied_velocity,
            rtol=1e-7,
            atol=1e-9,
        )

    def test_multistep_position_respects_outer_tick_acceleration_limit(self, panda_setup):
        """Inner refinements must not multiply the configured outer-tick acceleration."""
        robot, solver = panda_setup
        task = solver.add_frame_task("ee", "panda_hand", eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 1.0
        solver.enable_position_limits(True)
        solver.enable_velocity_limits(True)

        acceleration_limit = 10.0
        solver.set_acceleration_limits(np.full(robot.nv, acceleration_limit))
        solver.enable_acceleration_limits(True)

        q = PANDA_HOME.copy()
        robot.update_configuration(q)
        ee_pose = robot.get_frame_pose("panda_hand")
        target = ee_pose.homogeneous()
        target[0, 3] += 0.3

        options = eik.PositionStepOptions()
        options.dt = solver.dt
        options.max_steps = 3
        options.position_gain = 10.0
        options.orientation_gain = 10.0

        previous_applied_velocity = np.zeros(robot.nv)
        for _ in range(5):
            result = solver.solve_position_step(q, [eik.TaskTarget("ee", target)], options)
            assert result.status == eik.SolverStatus.SUCCESS
            q_next = np.asarray(result.q_solution)
            applied_velocity = (q_next - q) / solver.dt
            max_velocity_change = acceleration_limit * solver.dt * 1.02
            assert (
                np.max(np.abs(applied_velocity - previous_applied_velocity)) <= max_velocity_change
            )
            np.testing.assert_allclose(
                np.asarray(result.joint_velocities),
                applied_velocity,
                rtol=1e-7,
                atol=1e-9,
            )
            previous_applied_velocity = applied_velocity
            q = q_next

    def test_multistep_position_applies_physical_acceleration_once(self):
        """Speculative refinements must not consume extra acceleration intervals."""
        pytest.importorskip("robot_descriptions.panda_description")
        from robot_descriptions.panda_description import URDF_PATH

        def build_solver():
            robot = eik.RobotModel(URDF_PATH, floating_base=False)
            solver = eik.KinematicsSolver(robot)
            solver.dt = 0.02
            solver.set_damping(0.05)
            solver.enable_position_limits(True)
            solver.enable_velocity_limits(True)
            task = solver.add_frame_task("ee", "panda_hand", eik.TaskType.FRAME_POSE)
            task.priority = 0
            task.weight = 1.0
            solver.set_acceleration_limits(np.full(robot.nv, 2.0))
            solver.enable_acceleration_limits(True)
            return robot, solver

        q = PANDA_HOME.copy()
        one_step_robot, one_step_solver = build_solver()
        one_step_robot.update_configuration(q)
        target = one_step_robot.get_frame_pose("panda_hand").homogeneous()
        target[:3, 3] += np.array([0.20, 0.12, 0.08])

        one_step_options = eik.PositionStepOptions()
        one_step_options.dt = one_step_solver.dt
        one_step_options.max_steps = 1
        one_step_options.position_gain = 10.0
        one_step_options.orientation_gain = 10.0
        one_step = one_step_solver.solve_position_step(
            q, [eik.TaskTarget("ee", target)], one_step_options
        )

        multistep_robot, multistep_solver = build_solver()
        multistep_options = eik.PositionStepOptions()
        multistep_options.dt = multistep_solver.dt
        multistep_options.max_steps = 3
        multistep_options.position_gain = 10.0
        multistep_options.orientation_gain = 10.0
        multistep = multistep_solver.solve_position_step(
            q, [eik.TaskTarget("ee", target)], multistep_options
        )

        assert one_step.status == eik.SolverStatus.SUCCESS
        assert multistep.status == eik.SolverStatus.SUCCESS
        one_step_velocity = (np.asarray(one_step.q_solution) - q) / one_step_options.dt
        multistep_velocity = (np.asarray(multistep.q_solution) - q) / multistep_options.dt
        assert np.max(np.abs(one_step_velocity)) <= 2.0 * one_step_options.dt * 1.02
        assert np.max(np.abs(multistep_velocity)) <= 2.0 * multistep_options.dt * 1.02

    @pytest.mark.parametrize(
        ("translation_offset", "rotation_angle"),
        (
            (np.array([0.20, 0.12, 0.08]), 0.35),
            (np.zeros(3), 0.35),
            (np.array([0.20, 0.12, 0.08]), 0.0),
        ),
        ids=("full-pose", "orientation-only", "translation-only"),
    )
    def test_multistep_position_masks_uncommanded_predictive_blocks(
        self, translation_offset, rotation_angle
    ):
        """Prediction stays feasible without activating an uncommanded pose block."""
        pytest.importorskip("robot_descriptions.panda_description")
        from robot_descriptions.panda_description import URDF_PATH

        def build_solver(*, acceleration_limited: bool):
            robot = eik.RobotModel(URDF_PATH, floating_base=False)
            solver = eik.KinematicsSolver(robot)
            solver.dt = 0.02
            solver.set_damping(0.05)
            solver.enable_position_limits(True)
            solver.enable_velocity_limits(True)
            if acceleration_limited:
                solver.set_acceleration_limits(np.full(robot.nv, 100.0))
                solver.enable_acceleration_limits(True)
            task = solver.add_frame_task("ee", "panda_hand", eik.TaskType.FRAME_POSE)
            task.priority = 0
            task.weight = 1.0
            return robot, solver, task

        q = PANDA_HOME.copy()
        q[-2:] = 0.02
        target_robot, nominal_solver, _ = build_solver(acceleration_limited=False)
        target_robot.update_configuration(q)
        target = target_robot.get_frame_pose("panda_hand").homogeneous()
        target[:3, 3] += translation_offset
        rotation_delta = np.array(
            [
                [np.cos(rotation_angle), 0.0, np.sin(rotation_angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(rotation_angle), 0.0, np.cos(rotation_angle)],
            ]
        )
        target[:3, :3] = rotation_delta @ target[:3, :3]

        position_gain = 1.0
        orientation_gain = 1.0
        options = eik.PositionStepOptions()
        options.dt = 0.02
        options.max_steps = 3
        options.position_gain = position_gain
        options.orientation_gain = orientation_gain
        options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR

        predictor_steps = 1 if np.linalg.norm(translation_offset) <= 1e-12 else 3
        options.max_steps = predictor_steps
        nominal_result = nominal_solver.solve_position_step(
            q, [eik.TaskTarget("ee", target)], options
        )
        assert nominal_result.status == eik.SolverStatus.SUCCESS
        nominal_velocity = (np.asarray(nominal_result.q_solution) - q) / options.dt
        options.max_steps = 3

        direct_robot, direct_solver, direct_task = build_solver(acceleration_limited=True)
        direct_task.solve_mode = eik.TaskSolveMode.MIN_ERROR
        direct_robot.update_configuration(q)
        direct_task.set_target_pose(target[:3, 3], target[:3, :3])
        direct_task.update(direct_robot)
        current_error = np.asarray(direct_task.get_error(), dtype=float)
        current_task_velocity = np.concatenate(
            (
                position_gain * current_error[:3],
                orientation_gain * current_error[3:],
            )
        )
        predictive_task_velocity = (
            np.asarray(direct_task.get_jacobian(), dtype=float) @ nominal_velocity
        )

        projected_task_velocity = np.zeros(6, dtype=float)
        linear_commanded = float(current_task_velocity[:3] @ current_task_velocity[:3]) > 1e-12
        angular_commanded = float(current_task_velocity[3:] @ current_task_velocity[3:]) > 1e-12
        if linear_commanded:
            projected_task_velocity[:3] = predictive_task_velocity[:3]
        if linear_commanded or angular_commanded:
            projected_task_velocity[3:] = predictive_task_velocity[3:]

        direct_task.set_target_velocity(projected_task_velocity)
        direct_result = direct_solver.solve_velocity(q, apply_limits=True)
        direct_task.clear_target_velocity()
        assert direct_result.status == eik.SolverStatus.SUCCESS

        step_robot, step_solver, _ = build_solver(acceleration_limited=True)
        options.dt = step_solver.dt
        step_result = step_solver.solve_position_step(q, [eik.TaskTarget("ee", target)], options)
        assert step_result.status == eik.SolverStatus.SUCCESS

        applied_velocity = (np.asarray(step_result.q_solution) - q) / options.dt
        np.testing.assert_allclose(
            applied_velocity,
            np.asarray(direct_result.joint_velocities),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_multistep_position_selects_lower_merit_feasible_outer_candidate(self):
        """Outer acceleration projection should keep the better safe task step."""
        pytest.importorskip("robot_descriptions.panda_description")
        from robot_descriptions.panda_description import URDF_PATH

        def build_solver(*, acceleration_limited: bool):
            robot = eik.RobotModel(URDF_PATH, floating_base=False)
            solver = eik.KinematicsSolver(robot)
            solver.dt = 0.02
            solver.set_damping(0.05)
            solver.enable_position_limits(True)
            solver.enable_velocity_limits(True)
            if acceleration_limited:
                solver.set_acceleration_limits(np.full(robot.nv, 2.0))
                solver.enable_acceleration_limits(True)
            task = solver.add_frame_task("ee", "panda_hand", eik.TaskType.FRAME_POSE)
            task.solve_mode = eik.TaskSolveMode.MIN_ERROR
            return robot, solver, task

        q = np.array(
            [
                -0.17136382,
                -0.78512805,
                -0.20969206,
                -2.17748321,
                0.05088892,
                1.60011931,
                0.95083846,
                0.02,
                0.02,
            ]
        )
        translation_offset = np.array([-0.05516292, 0.0813279, -0.04271275])
        axis = np.array([-0.75289164, 0.03292882, 0.51645381])
        axis /= np.linalg.norm(axis)
        angle = 0.7206624117090492
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        one_minus_c = 1.0 - c
        rotation_delta = np.array(
            [
                [
                    c + x * x * one_minus_c,
                    x * y * one_minus_c - z * s,
                    x * z * one_minus_c + y * s,
                ],
                [
                    y * x * one_minus_c + z * s,
                    c + y * y * one_minus_c,
                    y * z * one_minus_c - x * s,
                ],
                [
                    z * x * one_minus_c - y * s,
                    z * y * one_minus_c + x * s,
                    c + z * z * one_minus_c,
                ],
            ]
        )

        nominal_robot, nominal_solver, nominal_task = build_solver(acceleration_limited=False)
        nominal_robot.update_configuration(q)
        target = nominal_robot.get_frame_pose("panda_hand").homogeneous()
        target[:3, 3] += translation_offset
        target[:3, :3] = rotation_delta @ target[:3, :3]

        options = eik.PositionStepOptions()
        options.dt = nominal_solver.dt
        options.max_steps = 3
        options.position_gain = 1.0
        options.orientation_gain = 1.0
        options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
        nominal_result = nominal_solver.solve_position_step(
            q, [eik.TaskTarget("ee", target)], options
        )
        assert nominal_result.status == eik.SolverStatus.SUCCESS
        nominal_velocity = (np.asarray(nominal_result.q_solution) - q) / options.dt

        acceleration_step = 2.0 * options.dt * 1.01
        componentwise_velocity = np.clip(nominal_velocity, -acceleration_step, acceleration_step)
        componentwise_q = q + options.dt * componentwise_velocity

        metric_robot, metric_solver, metric_task = build_solver(acceleration_limited=True)
        metric_robot.update_configuration(q)
        metric_task.set_target_pose(target[:3, 3], target[:3, :3])
        metric_task.update(metric_robot)
        metric_task.set_target_velocity(np.asarray(metric_task.get_jacobian()) @ nominal_velocity)
        metric_result = metric_solver.solve_velocity(q, apply_limits=True)
        metric_task.clear_target_velocity()
        assert metric_result.status == eik.SolverStatus.SUCCESS
        metric_q = q + options.dt * np.asarray(metric_result.joint_velocities)

        def task_block_merits(robot, task, candidate_q):
            robot.update_configuration(candidate_q)
            task.set_target_pose(target[:3, 3], target[:3, :3])
            task.update(robot)
            error = np.asarray(task.get_error())
            return float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))

        componentwise_merits = task_block_merits(nominal_robot, nominal_task, componentwise_q)
        metric_merits = task_block_merits(metric_robot, metric_task, metric_q)
        assert componentwise_merits[0] + 1e-7 < metric_merits[0]
        assert componentwise_merits[1] + 1e-7 < metric_merits[1]

        step_robot, step_solver, _ = build_solver(acceleration_limited=True)
        step_result = step_solver.solve_position_step(q, [eik.TaskTarget("ee", target)], options)
        assert step_result.status == eik.SolverStatus.SUCCESS
        np.testing.assert_allclose(
            np.asarray(step_result.q_solution), componentwise_q, rtol=0.0, atol=1e-9
        )

    def test_streaming_rotation_target_keeps_first_tick_projection(self):
        """Incidental position error must not reclassify a rotation command."""
        pytest.importorskip("robot_descriptions.panda_description")
        from robot_descriptions.panda_description import URDF_PATH

        def build_solver(*, acceleration_limited: bool):
            robot = eik.RobotModel(URDF_PATH, floating_base=False)
            solver = eik.KinematicsSolver(robot)
            solver.dt = 0.02
            solver.set_damping(0.05)
            solver.enable_position_limits(True)
            solver.enable_velocity_limits(True)
            if acceleration_limited:
                solver.set_acceleration_limits(np.full(robot.nv, 100.0))
                solver.enable_acceleration_limits(True)
            task = solver.add_frame_task("ee", "panda_hand", eik.TaskType.FRAME_POSE)
            task.solve_mode = eik.TaskSolveMode.MIN_ERROR
            return robot, solver, task

        q0 = PANDA_HOME.copy()
        q0[-2:] = 0.02
        target_robot, _, _ = build_solver(acceleration_limited=False)
        target_robot.update_configuration(q0)
        base_target = target_robot.get_frame_pose("panda_hand").homogeneous()

        def rotated_target(angle):
            target = base_target.copy()
            rotation_delta = np.array(
                [
                    [np.cos(angle), 0.0, np.sin(angle)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(angle), 0.0, np.cos(angle)],
                ]
            )
            target[:3, :3] = rotation_delta @ target[:3, :3]
            return target

        options = eik.PositionStepOptions()
        options.dt = 0.02
        options.max_steps = 3
        options.position_gain = 1.0
        options.orientation_gain = 1.0
        options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR

        step_robot, step_solver, step_task = build_solver(acceleration_limited=True)
        first_target = rotated_target(0.20)
        first_result = step_solver.solve_position_step(
            q0, [eik.TaskTarget("ee", first_target)], options
        )
        assert first_result.status == eik.SolverStatus.SUCCESS
        q1 = np.asarray(first_result.q_solution)
        previous_velocity = (q1 - q0) / options.dt

        second_target = rotated_target(0.35)
        step_robot.update_configuration(q1)
        step_task.set_target_pose(second_target[:3, 3], second_target[:3, :3])
        step_task.update(step_robot)
        assert np.linalg.norm(np.asarray(step_task.get_error())[:3]) > 1e-8

        nominal_robot, nominal_solver, _ = build_solver(acceleration_limited=False)
        options.max_steps = 1
        nominal_result = nominal_solver.solve_position_step(
            q1, [eik.TaskTarget("ee", second_target)], options
        )
        assert nominal_result.status == eik.SolverStatus.SUCCESS
        first_tick_velocity = (np.asarray(nominal_result.q_solution) - q1) / options.dt
        options.max_steps = 3

        direct_robot, direct_solver, direct_task = build_solver(acceleration_limited=True)
        direct_solver.set_previous_joint_velocities(previous_velocity)
        direct_robot.update_configuration(q1)
        direct_task.set_target_pose(second_target[:3, 3], second_target[:3, :3])
        direct_task.update(direct_robot)
        direct_task.set_target_velocity(
            np.asarray(direct_task.get_jacobian()) @ first_tick_velocity
        )
        direct_result = direct_solver.solve_velocity(q1, apply_limits=True)
        direct_task.clear_target_velocity()
        assert direct_result.status == eik.SolverStatus.SUCCESS

        second_result = step_solver.solve_position_step(
            q1, [eik.TaskTarget("ee", second_target)], options
        )
        assert second_result.status == eik.SolverStatus.SUCCESS
        applied_velocity = (np.asarray(second_result.q_solution) - q1) / options.dt
        np.testing.assert_allclose(
            applied_velocity,
            np.asarray(direct_result.joint_velocities),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_position_step_accepts_caller_applied_velocity_reference(self, panda_setup):
        """A caller can synchronize acceleration state to the velocity it applied."""
        robot, solver = panda_setup
        task = solver.add_frame_task("ee", "panda_hand", eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 1.0
        solver.enable_position_limits(True)
        solver.enable_velocity_limits(True)

        acceleration_limit = 10.0
        solver.set_acceleration_limits(np.full(robot.nv, acceleration_limit))
        solver.enable_acceleration_limits(True)
        caller_velocity = np.full(robot.nv, 0.5)
        caller_velocity[-2:] = 0.0
        solver.set_previous_joint_velocities(caller_velocity)

        q = PANDA_HOME.copy()
        robot.update_configuration(q)
        ee_pose = robot.get_frame_pose("panda_hand")
        target = ee_pose.homogeneous()
        target[0, 3] -= 0.3

        options = eik.PositionStepOptions()
        options.dt = solver.dt
        options.max_steps = 3
        options.position_gain = 10.0
        options.orientation_gain = 10.0

        result = solver.solve_position_step(q, [eik.TaskTarget("ee", target)], options)
        applied_velocity = (np.asarray(result.q_solution) - q) / solver.dt
        np.testing.assert_array_less(
            np.abs(applied_velocity - caller_velocity),
            np.full(robot.nv, acceleration_limit * solver.dt * 1.02 + 1e-12),
        )
        np.testing.assert_allclose(
            solver.get_previous_joint_velocities(),
            applied_velocity,
            rtol=1e-7,
            atol=1e-9,
        )

    def test_position_limit_overrides_infeasible_acceleration_reference(self, panda_setup):
        """A hard position limit stops an outward caller velocity immediately."""
        robot, solver = panda_setup
        task = solver.add_frame_task("ee", "panda_hand", eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 1.0
        solver.enable_position_limits(True)
        solver.enable_velocity_limits(True)
        solver.set_acceleration_limits(np.full(robot.nv, 10.0))
        solver.enable_acceleration_limits(True)

        q = PANDA_HOME.copy()
        robot.update_configuration(q)
        ee_pose = robot.get_frame_pose("panda_hand")
        target = ee_pose.homogeneous()
        target[0, 3] -= 0.3

        infeasible_reference = np.zeros(robot.nv)
        infeasible_reference[-2:] = 0.5
        solver.set_previous_joint_velocities(infeasible_reference)
        options = eik.PositionStepOptions()
        options.dt = solver.dt
        options.max_steps = 3

        result = solver.solve_position_step(q, [eik.TaskTarget("ee", target)], options)
        applied_velocity = (np.asarray(result.q_solution) - q) / solver.dt

        assert np.all(np.asarray(result.q_solution)[-2:] <= q[-2:] + 1e-12)
        assert np.all(applied_velocity[-2:] <= 1e-12)

    def test_far_stationary_target_decelerates_before_hold(self, panda_setup):
        """A continuity hold must brake a feasible moving command before stopping."""
        robot, solver = panda_setup
        solver.dt = 0.02
        task = solver.add_frame_task("ee", "panda_hand", eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 1.0
        solver.enable_position_limits(True)
        solver.enable_velocity_limits(True)

        acceleration_limit = 10.0
        solver.set_acceleration_limits(np.full(robot.nv, acceleration_limit))
        solver.enable_acceleration_limits(True)

        q = PANDA_HOME.copy()
        robot.update_configuration(q)
        target = robot.get_frame_pose("panda_hand").homogeneous()
        target[0, 3] += 0.8
        target[2, 3] += 0.8

        options = eik.PositionStepOptions()
        options.dt = solver.dt
        options.max_steps = 3
        options.position_gain = 10.0
        options.orientation_gain = 10.0
        options.max_configuration_step_norm = 0.12

        previous_velocity = np.zeros(robot.nv)
        hold_seen = False
        for _ in range(80):
            result = solver.solve_position_step(q, [eik.TaskTarget("ee", target)], options)
            q_next = np.asarray(result.q_solution)
            applied_velocity = (q_next - q) / solver.dt
            assert (
                np.max(np.abs(applied_velocity - previous_velocity))
                <= acceleration_limit * solver.dt * 1.02
            )
            if result.position_step_hold_active:
                np.testing.assert_allclose(applied_velocity, np.zeros(robot.nv), atol=1e-12)
                hold_seen = True
            previous_velocity = applied_velocity
            q = q_next

        assert hold_seen
        np.testing.assert_allclose(previous_velocity, np.zeros(robot.nv), atol=1e-12)

    def test_multistep_position_brakes_before_joint_limit(self, tmp_path):
        robot = eik.RobotModel(str(_write_prismatic_limit_urdf(tmp_path)))
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.02
        solver.set_damping(0.01)
        solver.enable_position_limits(True)
        solver.enable_velocity_limits(True)
        acceleration_limit = 10.0
        solver.set_acceleration_limits(np.array([acceleration_limit], dtype=float))
        solver.enable_acceleration_limits(True)

        task = solver.add_frame_task("moving_task", "moving")
        task.priority = 0
        task.weight = 1.0
        q = np.array([0.02], dtype=float)
        robot.update_configuration(q)
        target = robot.get_frame_pose("moving").homogeneous()
        target[0, 3] += 0.5

        options = eik.PositionStepOptions()
        options.dt = solver.dt
        options.max_steps = 3
        options.position_gain = 10.0
        options.orientation_gain = 1.0

        previous_velocity = np.zeros(robot.nv)
        for _ in range(100):
            result = solver.solve_position_step(q, target, "moving_task", options)
            q_next = np.asarray(result.q_solution)
            applied_velocity = (q_next - q) / solver.dt
            assert q_next[0] <= 0.2 + 1e-12
            assert (
                np.max(np.abs(applied_velocity - previous_velocity))
                <= acceleration_limit * solver.dt * 1.02
            )
            previous_velocity = applied_velocity
            q = q_next

    def test_acceleration_limits_reduce_jerk(self, panda_setup):
        """With acceleration limits, max joint velocity jump (jerk proxy)
        should be smaller than without.
        """
        robot, solver = panda_setup
        task = solver.add_frame_task("ee", "panda_hand", eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 1.0
        solver.enable_position_limits(True)

        # Target that requires a fast direction change
        q_start = PANDA_HOME.copy()
        robot.update_configuration(q_start)
        ee_pose = robot.get_frame_pose("panda_hand")
        base_pos = ee_pose.translation.copy()
        T = ee_pose.homogeneous()
        base_rot = T[:3, :3]

        n_steps = 150

        def run_trajectory(accel_enabled):
            q = q_start.copy()
            velocities = []
            # Phase 1: reach right
            target_pos = base_pos.copy()
            target_pos[1] += 0.15
            task.set_target_pose(target_pos, base_rot)
            for _ in range(n_steps // 2):
                robot.update_configuration(q)
                result = solver.solve_velocity(q, apply_limits=True)
                if result.status != eik.SolverStatus.SUCCESS:
                    break
                dq = np.array(result.joint_velocities)
                velocities.append(dq.copy())
                q = q + dq * solver.dt

            # Phase 2: sharp reversal — reach left
            target_pos = base_pos.copy()
            target_pos[1] -= 0.15
            task.set_target_pose(target_pos, base_rot)
            for _ in range(n_steps // 2):
                robot.update_configuration(q)
                result = solver.solve_velocity(q, apply_limits=True)
                if result.status != eik.SolverStatus.SUCCESS:
                    break
                dq = np.array(result.joint_velocities)
                velocities.append(dq.copy())
                q = q + dq * solver.dt

            return velocities

        # Without acceleration limits
        solver.enable_acceleration_limits(False)
        vel_no_accel = run_trajectory(False)

        # With acceleration limits (conservative: 10 rad/s^2 per joint)
        solver.enable_acceleration_limits(True)
        nv = robot.nv
        accel_limits = np.full(nv, 10.0)
        solver.set_acceleration_limits(accel_limits)
        vel_with_accel = run_trajectory(True)

        def max_jerk(velocities):
            if len(velocities) < 2:
                return 0.0
            jumps = [
                np.max(np.abs(velocities[i + 1] - velocities[i]))
                for i in range(len(velocities) - 1)
            ]
            return max(jumps)

        jerk_no_accel = max_jerk(vel_no_accel)
        jerk_with_accel = max_jerk(vel_with_accel)

        # Acceleration limits should reduce the max velocity jump
        assert jerk_with_accel < jerk_no_accel, (
            f"Jerk with accel limits ({jerk_with_accel:.4f}) should be less "
            f"than without ({jerk_no_accel:.4f})"
        )

    def test_acceleration_limits_bound_velocity_change(self, panda_setup):
        """Each tick's velocity change per joint should not exceed a_max * dt."""
        robot, solver = panda_setup
        task = solver.add_frame_task("ee", "panda_hand", eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 1.0
        solver.enable_position_limits(True)

        nv = robot.nv
        a_max = 5.0  # rad/s^2
        solver.enable_acceleration_limits(True)
        solver.set_acceleration_limits(np.full(nv, a_max))

        q = PANDA_HOME.copy()
        robot.update_configuration(q)
        ee_pose = robot.get_frame_pose("panda_hand")
        target_pos = ee_pose.translation.copy()
        target_pos[0] += 0.2  # large reach
        T = ee_pose.homogeneous()
        task.set_target_pose(target_pos, T[:3, :3])

        prev_dq = np.zeros(nv)
        dt = solver.dt
        max_accel_violation = 0.0

        for step in range(100):
            robot.update_configuration(q)
            result = solver.solve_velocity(q, apply_limits=True)
            if result.status != eik.SolverStatus.SUCCESS:
                break
            dq = np.array(result.joint_velocities)

            if step > 0:  # skip first step (no previous velocity)
                accel = np.abs(dq - prev_dq) / dt
                max_accel_violation = max(max_accel_violation, np.max(accel) - a_max)

            prev_dq = dq.copy()
            q = q + dq * dt

        # Allow small numerical tolerance
        assert (
            max_accel_violation < 0.1
        ), f"Acceleration violation {max_accel_violation:.4f} exceeds tolerance"

    def test_tracking_not_catastrophically_degraded(self, panda_setup):
        """Acceleration limits should not degrade EE tracking by more than 15%."""
        robot, solver = panda_setup
        task = solver.add_frame_task("ee", "panda_hand", eik.TaskType.FRAME_POSE)
        task.priority = 0
        task.weight = 1.0
        solver.enable_position_limits(True)

        q_start = PANDA_HOME.copy()
        robot.update_configuration(q_start)
        ee_pose = robot.get_frame_pose("panda_hand")
        target_pos = ee_pose.translation.copy()
        target_pos[0] += 0.1
        T = ee_pose.homogeneous()
        target_rot = T[:3, :3]
        task.set_target_pose(target_pos, target_rot)

        n_steps = 100

        def run_and_measure_error():
            q = q_start.copy()
            for _ in range(n_steps):
                robot.update_configuration(q)
                result = solver.solve_velocity(q, apply_limits=True)
                if result.status != eik.SolverStatus.SUCCESS:
                    break
                q = q + np.array(result.joint_velocities) * solver.dt
            robot.update_configuration(q)
            final_pos = robot.get_frame_pose("panda_hand").translation
            return np.linalg.norm(final_pos - target_pos)

        # Without acceleration limits
        solver.enable_acceleration_limits(False)
        error_baseline = run_and_measure_error()

        # With acceleration limits
        solver.enable_acceleration_limits(True)
        solver.set_acceleration_limits(np.full(robot.nv, 10.0))
        error_with_accel = run_and_measure_error()

        # Should not be more than 15% worse
        assert error_with_accel < error_baseline * 1.15 + 1e-6, (
            f"Error with accel limits ({error_with_accel:.6f}) is more than "
            f"15% worse than baseline ({error_baseline:.6f})"
        )
