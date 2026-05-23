#!/usr/bin/env python3
"""Headless regression harnesses for the G1 retargeting Viser example.

Kept separate from ``07_unitree_g1_retargeting_ik.py`` so the example
script stays focused on the interactive Viser application.
"""

from __future__ import annotations

import time

import embodik
import numpy as np

try:
    from example_helpers.g1_ik_runtime import (
        _apply_g1_soft_knee_seed,
        _apply_zero_based_task_hierarchy,
        _clear_interactive_solver_transients,
        _clip_q,
        _configure_g1_posture_task,
        _configure_interactive_elastic_band,
        _limit_tangent_step,
        _percentile,
        _solve_quality_step,
        _target_position_error,
        _target_solve_mode,
    )
    from example_helpers.g1_model_utils import (
        build_retargeting_delta_target_poses,
        build_retargeting_target_poses,
        com_min_slack,
        compute_support_polygon_from_foot_poses,
        create_g1_robot_model,
        get_retargeting_clip_duration,
        get_retargeting_clips,
        get_retargeting_presets,
        ground_floating_base_from_feet_center,
        resolve_frames_for_g1_base_mode,
        sample_retargeting_clip,
    )
except ModuleNotFoundError as exc:
    if exc.name != "example_helpers" and not str(exc.name).startswith("example_helpers."):
        raise
    from examples.example_helpers.g1_ik_runtime import (
        _apply_g1_soft_knee_seed,
        _apply_zero_based_task_hierarchy,
        _clear_interactive_solver_transients,
        _clip_q,
        _configure_g1_posture_task,
        _configure_interactive_elastic_band,
        _limit_tangent_step,
        _percentile,
        _solve_quality_step,
        _target_position_error,
        _target_solve_mode,
    )
    from examples.example_helpers.g1_model_utils import (
        build_retargeting_delta_target_poses,
        build_retargeting_target_poses,
        com_min_slack,
        compute_support_polygon_from_foot_poses,
        create_g1_robot_model,
        get_retargeting_clip_duration,
        get_retargeting_clips,
        get_retargeting_presets,
        ground_floating_base_from_feet_center,
        resolve_frames_for_g1_base_mode,
        sample_retargeting_clip,
    )


def run_headless_smoke(steps: int) -> None:
    """Exercise retargeting helpers without starting Viser."""
    frame_map = {
        "right_palm": "right_palm",
        "left_palm": "left_palm",
        "imu_in_torso": "imu_in_torso",
        "right_ankle": "right_ankle",
        "left_ankle": "left_ankle",
    }
    resolved = resolve_frames_for_g1_base_mode(frame_map.values())
    required = set(frame_map)
    if not required.issubset(resolved):
        raise RuntimeError(f"unexpected frame resolution: {resolved}")

    anchor = np.eye(4, dtype=float)
    for name in get_retargeting_clips():
        duration = get_retargeting_clip_duration(name)
        for t in np.linspace(0.0, duration, max(2, int(steps))):
            local_offsets = sample_retargeting_clip(name, float(t))
            poses = build_retargeting_target_poses(anchor, local_offsets)
            missing = required - set(poses)
            if missing:
                raise RuntimeError(f"{name} missing retargeting frames: {sorted(missing)}")
            for key, pose in poses.items():
                if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
                    raise RuntimeError(f"{name}:{key} produced invalid pose")
    print(f"[13][headless-smoke] ok clips={len(get_retargeting_clips())} steps={steps}")


def run_headless_com_smoke() -> None:
    """Validate G1 CoM support-polygon setup without starting Viser."""
    robot = create_g1_robot_model(floating_base=True, reduced_ik=True)
    q = _apply_g1_soft_knee_seed(robot, robot.neutral_configuration())
    q_lo, q_hi = robot.get_joint_limits()
    robot.update_configuration(q)
    frame_map = resolve_frames_for_g1_base_mode(robot.get_frame_names())

    r0 = np.asarray(robot.get_frame_pose(frame_map["right_ankle"]).translation, dtype=float)
    l0 = np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).translation, dtype=float)
    q = ground_floating_base_from_feet_center(q, r0, l0)
    q = _clip_q(robot, q, q_lo, q_hi)
    robot.update_configuration(q)

    right_foot_pose = np.asarray(
        robot.get_frame_pose(frame_map["right_ankle"]).homogeneous(), dtype=float
    )
    left_foot_pose = np.asarray(
        robot.get_frame_pose(frame_map["left_ankle"]).homogeneous(), dtype=float
    )
    support_polygon = compute_support_polygon_from_foot_poses(
        right_foot_pose,
        left_foot_pose,
        foot_length=0.22,
        foot_width=0.10,
        toe_pad=0.06,
        side_pad=0.07,
    )
    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.configure_com_constraint(
        support_polygon=support_polygon,
        margin=0.05,
        frame_name="world",
        com_vel_max=0.3,
        com_acc_max=0.3,
        use_acceleration_limits=True,
        proximity_fraction=0.05,
    )
    proximity_threshold = float(solver.get_com_proximity_threshold())
    slack = com_min_slack(
        support_polygon,
        np.asarray(robot.get_com_position(), dtype=float)[:2],
        margin_fraction=0.05,
    )
    if support_polygon.shape[0] < 3:
        raise RuntimeError(f"invalid support polygon: {support_polygon}")
    if not np.isfinite(slack):
        raise RuntimeError(f"invalid CoM slack: {slack}")
    if proximity_threshold <= 0.0:
        raise RuntimeError(f"invalid CoM proximity threshold: {proximity_threshold}")
    print(
        "[13][headless-com-smoke] ok "
        f"vertices={support_polygon.shape[0]} "
        f"slack={slack:.4f} "
        f"proximity={proximity_threshold:.4f}"
    )


def run_headless_ik_smoke(steps: int, reset_interval: int = 0) -> None:
    """Run sampled retargeting targets through the real G1 IK stack."""
    robot = create_g1_robot_model(floating_base=True, reduced_ik=True)
    q = _apply_g1_soft_knee_seed(robot, robot.neutral_configuration())
    q_lo, q_hi = robot.get_joint_limits()
    robot.update_configuration(q)
    frame_map = resolve_frames_for_g1_base_mode(robot.get_frame_names())

    r0 = np.asarray(robot.get_frame_pose(frame_map["right_ankle"]).translation, dtype=float)
    l0 = np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).translation, dtype=float)
    q = ground_floating_base_from_feet_center(q, r0, l0)
    q = _clip_q(robot, q, q_lo, q_hi)
    robot.update_configuration(q)
    q_initial = q.copy()

    torso_frame = frame_map["imu_in_torso"]
    retarget_anchor = np.asarray(robot.get_frame_pose(torso_frame).homogeneous(), dtype=float)
    retarget_reference_poses = {
        "right_palm": np.asarray(
            robot.get_frame_pose(frame_map["right_palm"]).homogeneous(), dtype=float
        ),
        "left_palm": np.asarray(
            robot.get_frame_pose(frame_map["left_palm"]).homogeneous(), dtype=float
        ),
        "right_ankle": np.asarray(
            robot.get_frame_pose(frame_map["right_ankle"]).homogeneous(), dtype=float
        ),
        "left_ankle": np.asarray(
            robot.get_frame_pose(frame_map["left_ankle"]).homogeneous(), dtype=float
        ),
        "imu_in_torso": retarget_anchor.copy(),
    }
    retarget_neutral_offsets = get_retargeting_presets()["neutral"]
    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    _configure_interactive_elastic_band(solver)

    right_task = solver.add_frame_task(
        "right_palm_pose", frame_map["right_palm"], embodik.TaskType.FRAME_POSE
    )
    left_task = solver.add_frame_task(
        "left_palm_pose", frame_map["left_palm"], embodik.TaskType.FRAME_POSE
    )
    right_foot_task = solver.add_frame_task(
        "right_ankle_pose", frame_map["right_ankle"], embodik.TaskType.FRAME_POSE
    )
    left_foot_task = solver.add_frame_task(
        "left_ankle_pose", frame_map["left_ankle"], embodik.TaskType.FRAME_POSE
    )
    posture = solver.add_posture_task("posture")
    torso_ori = solver.add_frame_task(
        "torso_upright_ori", frame_map["imu_in_torso"], embodik.TaskType.FRAME_ORIENTATION
    )
    _apply_zero_based_task_hierarchy(
        [right_task, left_task, right_foot_task, left_foot_task],
        [torso_ori],
        [posture],
    )
    posture.weight = 0.01
    _configure_g1_posture_task(posture, robot, q)
    target_mode, target_mode_label, _ = _target_solve_mode("SCALE_ELASTIC")
    right_task.solve_mode = target_mode
    left_task.solve_mode = target_mode
    right_task.allow_min_error_fallback = True
    left_task.allow_min_error_fallback = True
    right_foot_task.solve_mode = target_mode
    left_foot_task.solve_mode = target_mode
    right_foot_task.allow_min_error_fallback = True
    left_foot_task.allow_min_error_fallback = True
    torso_ori.active = True
    torso_ori.solve_mode = embodik.TaskSolveMode.MIN_ERROR
    torso_ori.allow_min_error_fallback = True
    torso_target = np.asarray(
        robot.get_frame_pose(frame_map["imu_in_torso"]).homogeneous(), dtype=float
    )

    opts = embodik.PositionStepOptions()
    opts.max_steps = 4
    opts.max_linear_speed = 1.8
    opts.max_angular_speed = 2.5
    opts.stall_recovery = True
    opts.adaptive_dt = True
    opts.adaptive_dt_reference_distance = 0.04
    opts.adaptive_dt_max_scale = 3.0

    statuses: list[str] = []
    accepted = 0
    recoveries = 0
    stability_ticks = 0
    max_solve_ms = 0.0
    max_all_target_error = 0.0
    max_target_errors = {"right_palm": 0.0, "left_palm": 0.0, "right_ankle": 0.0, "left_ankle": 0.0}
    max_q_step = 0.0
    reset_count = 0
    reset_max_recapture_error = 0.0
    reset_first_move_min_dq = float("inf")
    pending_reset_move = False
    bad_statuses: list[str] = []
    samples = max(2, int(steps))

    def _reset_headless_state() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        nonlocal q, stability_ticks
        _clear_interactive_solver_transients(solver)
        stability_ticks = 12
        q = q_initial.copy()
        robot.update_configuration(q)
        _configure_g1_posture_task(posture, robot, q)
        torso_target[:, :] = np.asarray(
            robot.get_frame_pose(frame_map["imu_in_torso"]).homogeneous(), dtype=float
        )
        recaptured = {
            key: np.asarray(robot.get_frame_pose(frame_map[key]).homogeneous(), dtype=float)
            for key in ("right_palm", "left_palm", "right_ankle", "left_ankle")
        }
        recapture_error = max(
            _target_position_error(robot, frame_map[key], pose) for key, pose in recaptured.items()
        )
        return (
            recaptured["right_palm"],
            recaptured["left_palm"],
            recaptured["right_ankle"],
            recaptured["left_ankle"],
            recapture_error,
        )

    for name in get_retargeting_clips():
        duration = get_retargeting_clip_duration(name)
        for t in np.linspace(0.0, duration, samples):
            local_offsets = sample_retargeting_clip(name, float(t))
            poses = build_retargeting_delta_target_poses(
                retarget_anchor,
                retarget_reference_poses,
                local_offsets,
                retarget_neutral_offsets,
            )
            targets = [
                embodik.TaskTarget("right_palm_pose", poses["right_palm"], 12.0, 8.0),
                embodik.TaskTarget("left_palm_pose", poses["left_palm"], 12.0, 8.0),
                embodik.TaskTarget("right_ankle_pose", poses["right_ankle"], 40.0, 14.0),
                embodik.TaskTarget("left_ankle_pose", poses["left_ankle"], 40.0, 14.0),
                embodik.TaskTarget("torso_upright_ori", torso_target, 0.0, 2.0),
            ]
            frame_targets = [
                (frame_map["right_palm"], poses["right_palm"]),
                (frame_map["left_palm"], poses["left_palm"]),
                (frame_map["right_ankle"], poses["right_ankle"]),
                (frame_map["left_ankle"], poses["left_ankle"]),
            ]
            t0 = time.perf_counter()
            result, recovered = _solve_quality_step(
                solver,
                robot,
                q,
                targets,
                opts,
                [right_task, left_task, right_foot_task, left_foot_task],
                frame_targets,
                prefer_min_error=stability_ticks > 0,
            )
            max_solve_ms = max(max_solve_ms, (time.perf_counter() - t0) * 1e3)
            recoveries += int(recovered)
            if recovered:
                stability_ticks = 12
            else:
                stability_ticks = max(0, stability_ticks - 1)
            statuses.append(result.status.name)
            if hasattr(result, "q_solution") and result.status.name != "COLLISION_VIOLATED":
                q = np.asarray(result.q_solution, dtype=float).copy()
                robot.update_configuration(q)
                accepted += 1

    # Exercise a gizmo-like drag path with warm-started IK and fixed soft feet.
    robot.update_configuration(q)
    rh0 = np.asarray(robot.get_frame_pose(frame_map["right_palm"]).homogeneous(), dtype=float)
    lh0 = np.asarray(robot.get_frame_pose(frame_map["left_palm"]).homogeneous(), dtype=float)
    rf0 = np.asarray(robot.get_frame_pose(frame_map["right_ankle"]).homogeneous(), dtype=float)
    lf0 = np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).homogeneous(), dtype=float)
    stability_ticks = max(stability_ticks, 4)
    for i, phase in enumerate(np.linspace(0.0, 2.0 * np.pi, max(8, samples * 4), endpoint=False)):
        rh = rh0.copy()
        lh = lh0.copy()
        rh[:3, 3] += np.array(
            [0.08 * np.sin(phase), 0.04 * np.cos(phase), 0.03 * np.sin(2.0 * phase)]
        )
        lh[:3, 3] += np.array(
            [0.06 * np.sin(phase + np.pi), 0.04 * np.cos(phase), 0.02 * np.cos(2.0 * phase)]
        )
        targets = [
            embodik.TaskTarget("right_palm_pose", rh, 12.0, 8.0),
            embodik.TaskTarget("left_palm_pose", lh, 12.0, 8.0),
            embodik.TaskTarget("right_ankle_pose", rf0, 40.0, 14.0),
            embodik.TaskTarget("left_ankle_pose", lf0, 40.0, 14.0),
            embodik.TaskTarget("torso_upright_ori", torso_target, 0.0, 2.0),
        ]
        frame_targets = [
            (frame_map["right_palm"], rh),
            (frame_map["left_palm"], lh),
            (frame_map["right_ankle"], rf0),
            (frame_map["left_ankle"], lf0),
        ]
        t0 = time.perf_counter()
        result, recovered = _solve_quality_step(
            solver,
            robot,
            q,
            targets,
            opts,
            [right_task, left_task, right_foot_task, left_foot_task],
            frame_targets,
            prefer_min_error=stability_ticks > 0,
        )
        max_solve_ms = max(max_solve_ms, (time.perf_counter() - t0) * 1e3)
        recoveries += int(recovered)
        if recovered:
            stability_ticks = 12
        else:
            stability_ticks = max(0, stability_ticks - 1)
        statuses.append(result.status.name)
        if not hasattr(result, "q_solution") or result.status.name == "COLLISION_VIOLATED":
            continue
        q_next = np.asarray(result.q_solution, dtype=float).copy()
        if np.all(np.isfinite(q_next)):
            q = q_next
            robot.update_configuration(q)
            accepted += 1

    rf0 = np.asarray(robot.get_frame_pose(frame_map["right_ankle"]).homogeneous(), dtype=float)
    lf0 = np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).homogeneous(), dtype=float)
    rh0 = np.asarray(robot.get_frame_pose(frame_map["right_palm"]).homogeneous(), dtype=float)
    lh0 = np.asarray(robot.get_frame_pose(frame_map["left_palm"]).homogeneous(), dtype=float)
    max_foot_error = 0.0
    stability_ticks = max(stability_ticks, 4)
    for phase in np.linspace(0.0, 2.0 * np.pi, max(8, samples * 4), endpoint=False):
        rf = rf0.copy()
        lf = lf0.copy()
        rf[:3, 3] += np.array(
            [0.04 * np.sin(phase), 0.03 * np.cos(phase), 0.025 * np.sin(2.0 * phase)]
        )
        lf[:3, 3] += np.array(
            [0.035 * np.sin(phase + np.pi), 0.025 * np.cos(phase), 0.020 * np.cos(2.0 * phase)]
        )
        targets = [
            embodik.TaskTarget("right_palm_pose", rh0, 12.0, 8.0),
            embodik.TaskTarget("left_palm_pose", lh0, 12.0, 8.0),
            embodik.TaskTarget("right_ankle_pose", rf, 40.0, 14.0),
            embodik.TaskTarget("left_ankle_pose", lf, 40.0, 14.0),
            embodik.TaskTarget("torso_upright_ori", torso_target, 0.0, 2.0),
        ]
        frame_targets = [
            (frame_map["right_palm"], rh0),
            (frame_map["left_palm"], lh0),
            (frame_map["right_ankle"], rf),
            (frame_map["left_ankle"], lf),
        ]
        t0 = time.perf_counter()
        result, recovered = _solve_quality_step(
            solver,
            robot,
            q,
            targets,
            opts,
            [right_task, left_task, right_foot_task, left_foot_task],
            frame_targets,
            prefer_min_error=stability_ticks > 0,
        )
        max_solve_ms = max(max_solve_ms, (time.perf_counter() - t0) * 1e3)
        recoveries += int(recovered)
        if recovered:
            stability_ticks = 12
        else:
            stability_ticks = max(0, stability_ticks - 1)
        statuses.append(result.status.name)
        if not hasattr(result, "q_solution") or result.status.name == "COLLISION_VIOLATED":
            continue
        q_next = np.asarray(result.q_solution, dtype=float).copy()
        if np.all(np.isfinite(q_next)):
            q = q_next
            robot.update_configuration(q)
            r_now = np.asarray(
                robot.get_frame_pose(frame_map["right_ankle"]).homogeneous(), dtype=float
            )
            l_now = np.asarray(
                robot.get_frame_pose(frame_map["left_ankle"]).homogeneous(), dtype=float
            )
            max_foot_error = max(
                max_foot_error,
                float(np.linalg.norm(r_now[:3, 3] - rf[:3, 3])),
                float(np.linalg.norm(l_now[:3, 3] - lf[:3, 3])),
            )
            accepted += 1

    robot.update_configuration(q)
    rh0 = np.asarray(robot.get_frame_pose(frame_map["right_palm"]).homogeneous(), dtype=float)
    lh0 = np.asarray(robot.get_frame_pose(frame_map["left_palm"]).homogeneous(), dtype=float)
    rf0 = np.asarray(robot.get_frame_pose(frame_map["right_ankle"]).homogeneous(), dtype=float)
    lf0 = np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).homogeneous(), dtype=float)
    simultaneous_steps = max(16, samples * 8)
    stability_ticks = max(stability_ticks, 4)
    segment_phase_origin = 0.0
    for step_idx, phase in enumerate(
        np.linspace(0.0, 2.0 * np.pi, simultaneous_steps, endpoint=False)
    ):
        if reset_interval > 0 and step_idx > 0 and step_idx % int(reset_interval) == 0:
            rh0, lh0, rf0, lf0, recapture_error = _reset_headless_state()
            segment_phase_origin = float(phase)
            reset_count += 1
            pending_reset_move = True
            reset_max_recapture_error = max(reset_max_recapture_error, recapture_error)
            continue
        phase = float(phase) - segment_phase_origin
        rh = rh0.copy()
        lh = lh0.copy()
        rf = rf0.copy()
        lf = lf0.copy()
        rh[:3, 3] += np.array(
            [0.055 * np.sin(phase), 0.030 * np.sin(1.3 * phase), 0.025 * np.sin(2.0 * phase)]
        )
        lh[:3, 3] += np.array(
            [
                -0.050 * np.sin(phase),
                -0.030 * np.sin(1.1 * phase),
                0.022 * np.sin(1.7 * phase),
            ]
        )
        rf[:3, 3] += np.array(
            [0.025 * np.sin(0.8 * phase), 0.020 * np.sin(phase), 0.012 * np.sin(1.6 * phase)]
        )
        lf[:3, 3] += np.array(
            [
                0.022 * np.sin(0.9 * phase + np.pi),
                -0.018 * np.sin(phase),
                0.010 * np.sin(1.4 * phase),
            ]
        )
        targets = [
            embodik.TaskTarget("right_palm_pose", rh, 12.0, 8.0),
            embodik.TaskTarget("left_palm_pose", lh, 12.0, 8.0),
            embodik.TaskTarget("right_ankle_pose", rf, 40.0, 14.0),
            embodik.TaskTarget("left_ankle_pose", lf, 40.0, 14.0),
            embodik.TaskTarget("torso_upright_ori", torso_target, 0.0, 2.0),
        ]
        frame_targets = [
            (frame_map["right_palm"], rh),
            (frame_map["left_palm"], lh),
            (frame_map["right_ankle"], rf),
            (frame_map["left_ankle"], lf),
        ]
        q_before = q.copy()
        t0 = time.perf_counter()
        result, recovered = _solve_quality_step(
            solver,
            robot,
            q,
            targets,
            opts,
            [right_task, left_task, right_foot_task, left_foot_task],
            frame_targets,
            prefer_min_error=stability_ticks > 0,
        )
        max_solve_ms = max(max_solve_ms, (time.perf_counter() - t0) * 1e3)
        recoveries += int(recovered)
        if recovered:
            stability_ticks = 12
        else:
            stability_ticks = max(0, stability_ticks - 1)
        statuses.append(result.status.name)
        if not hasattr(result, "q_solution") or result.status.name == "COLLISION_VIOLATED":
            current_errors = {
                "right_palm": _target_position_error(robot, frame_map["right_palm"], rh),
                "left_palm": _target_position_error(robot, frame_map["left_palm"], lh),
                "right_ankle": _target_position_error(robot, frame_map["right_ankle"], rf),
                "left_ankle": _target_position_error(robot, frame_map["left_ankle"], lf),
            }
            for key, value in current_errors.items():
                max_target_errors[key] = max(max_target_errors[key], value)
            current_target_error = max(current_errors.values())
            max_all_target_error = max(max_all_target_error, current_target_error)
            if current_target_error > 0.03:
                worst = max(current_errors, key=current_errors.get)
                bad_statuses.append(f"{result.status.name}@{worst}={current_target_error:.4f}")
            continue
        q_next = np.asarray(result.q_solution, dtype=float).copy()
        if np.all(np.isfinite(q_next)):
            q_next, q_step_component = _limit_tangent_step(robot, q_before, q_next, 0.8)
            q = q_next
            robot.update_configuration(q)
            max_q_step = max(max_q_step, q_step_component)
            current_errors = {
                "right_palm": _target_position_error(robot, frame_map["right_palm"], rh),
                "left_palm": _target_position_error(robot, frame_map["left_palm"], lh),
                "right_ankle": _target_position_error(robot, frame_map["right_ankle"], rf),
                "left_ankle": _target_position_error(robot, frame_map["left_ankle"], lf),
            }
            for key, value in current_errors.items():
                max_target_errors[key] = max(max_target_errors[key], value)
            current_target_error = max(current_errors.values())
            max_all_target_error = max(max_all_target_error, current_target_error)
            if result.status == embodik.SolverStatus.NO_PROGRESS and current_target_error > 0.03:
                worst = max(current_errors, key=current_errors.get)
                bad_statuses.append(f"{result.status.name}@{worst}={current_target_error:.4f}")
            accepted += 1
            if pending_reset_move:
                reset_first_move_min_dq = min(
                    reset_first_move_min_dq,
                    float(np.linalg.norm(np.asarray(result.joint_velocities, dtype=float))),
                )
                pending_reset_move = False

    rh_current, lh_current, rf_current, lf_current, recapture_error = _reset_headless_state()
    reset_count += 1
    reset_max_recapture_error = max(reset_max_recapture_error, recapture_error)
    rh_reset = np.asarray(robot.get_frame_pose(frame_map["right_palm"]).homogeneous(), dtype=float)
    rh_reset[:3, 3] += np.array([0.03, 0.0, 0.0])
    reset_targets = [
        embodik.TaskTarget("right_palm_pose", rh_reset, 12.0, 8.0),
        embodik.TaskTarget(
            "left_palm_pose",
            np.asarray(robot.get_frame_pose(frame_map["left_palm"]).homogeneous(), dtype=float),
            12.0,
            8.0,
        ),
        embodik.TaskTarget(
            "right_ankle_pose",
            np.asarray(robot.get_frame_pose(frame_map["right_ankle"]).homogeneous(), dtype=float),
            40.0,
            14.0,
        ),
        embodik.TaskTarget(
            "left_ankle_pose",
            np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).homogeneous(), dtype=float),
            40.0,
            14.0,
        ),
        embodik.TaskTarget("torso_upright_ori", torso_target, 0.0, 2.0),
    ]
    reset_frame_targets = [
        (frame_map["right_palm"], rh_reset),
        (
            frame_map["left_palm"],
            np.asarray(robot.get_frame_pose(frame_map["left_palm"]).homogeneous(), dtype=float),
        ),
        (
            frame_map["right_ankle"],
            np.asarray(robot.get_frame_pose(frame_map["right_ankle"]).homogeneous(), dtype=float),
        ),
        (
            frame_map["left_ankle"],
            np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).homogeneous(), dtype=float),
        ),
    ]
    t0 = time.perf_counter()
    reset_result, reset_recovered = _solve_quality_step(
        solver,
        robot,
        q,
        reset_targets,
        opts,
        [right_task, left_task, right_foot_task, left_foot_task],
        reset_frame_targets,
        prefer_min_error=stability_ticks > 0,
    )
    max_solve_ms = max(max_solve_ms, (time.perf_counter() - t0) * 1e3)
    recoveries += int(reset_recovered)
    reset_dq_norm = float(np.linalg.norm(np.asarray(reset_result.joint_velocities, dtype=float)))
    reset_first_move_min_dq = min(reset_first_move_min_dq, reset_dq_norm)
    statuses.append(reset_result.status.name)
    if reset_dq_norm <= 1e-10:
        raise RuntimeError(
            "post-reset first hand move produced no motion: "
            f"status={reset_result.status.name}, msg={getattr(reset_result, 'status_message', '')}"
        )
    accepted += int(hasattr(reset_result, "q_solution"))

    if not statuses:
        raise RuntimeError("headless IK smoke produced no solver samples")
    if not any(s in {"SUCCESS", "NO_PROGRESS", "INFEASIBLE"} for s in statuses):
        raise RuntimeError(f"unexpected IK smoke statuses: {statuses}")
    if accepted == 0:
        raise RuntimeError(f"headless IK smoke accepted no solver samples: {statuses}")
    if reset_count <= 0:
        raise RuntimeError("headless IK smoke did not execute any reset events")
    if reset_max_recapture_error > 1e-9:
        raise RuntimeError(
            f"headless reset recapture error too high: {reset_max_recapture_error:.3e} m"
        )
    if not np.isfinite(reset_first_move_min_dq) or reset_first_move_min_dq <= 1e-10:
        raise RuntimeError(
            f"headless reset first move produced no motion: dq={reset_first_move_min_dq:.3e}"
        )
    if bad_statuses:
        raise RuntimeError(
            "four-gizmo stress produced bad statuses: "
            f"{bad_statuses}; max_target_errors={max_target_errors}"
        )
    if max_all_target_error > 0.06:
        raise RuntimeError(f"four-gizmo stress target error too high: {max_all_target_error:.4f} m")
    if max_q_step > 0.8:
        raise RuntimeError(f"four-gizmo stress q jump too high: {max_q_step:.4f}")
    print(
        f"[13][headless-ik-smoke] ok clips={len(get_retargeting_clips())} "
        f"samples={len(statuses)} accepted={accepted} mode={target_mode_label} "
        f"recoveries={recoveries} reset_dq={reset_dq_norm:.3e} "
        f"reset_count={reset_count} reset_recapture={reset_max_recapture_error:.3e} "
        f"max_solve_ms={max_solve_ms:.2f} max_foot_error={max_foot_error:.4f} "
        f"max_all_target_error={max_all_target_error:.4f} max_q_step={max_q_step:.4f} "
        f"max_target_errors={max_target_errors} "
        f"statuses={sorted(set(statuses))}"
    )


def run_headless_reset_smoke() -> None:
    """Verify reset clears state and the first post-reset target move responds."""
    robot = create_g1_robot_model(floating_base=True, reduced_ik=True)
    q = _apply_g1_soft_knee_seed(robot, robot.neutral_configuration())
    q_lo, q_hi = robot.get_joint_limits()
    robot.update_configuration(q)
    frame_map = resolve_frames_for_g1_base_mode(robot.get_frame_names())

    r0 = np.asarray(robot.get_frame_pose(frame_map["right_ankle"]).translation, dtype=float)
    l0 = np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).translation, dtype=float)
    q = ground_floating_base_from_feet_center(q, r0, l0)
    q = _clip_q(robot, q, q_lo, q_hi)
    robot.update_configuration(q)
    q_initial = q.copy()

    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    _configure_interactive_elastic_band(solver)

    right_task = solver.add_frame_task(
        "right_palm_pose", frame_map["right_palm"], embodik.TaskType.FRAME_POSE
    )
    left_task = solver.add_frame_task(
        "left_palm_pose", frame_map["left_palm"], embodik.TaskType.FRAME_POSE
    )
    right_foot_task = solver.add_frame_task(
        "right_ankle_pose", frame_map["right_ankle"], embodik.TaskType.FRAME_POSE
    )
    left_foot_task = solver.add_frame_task(
        "left_ankle_pose", frame_map["left_ankle"], embodik.TaskType.FRAME_POSE
    )
    posture = solver.add_posture_task("posture")
    torso_ori = solver.add_frame_task(
        "torso_upright_ori", frame_map["imu_in_torso"], embodik.TaskType.FRAME_ORIENTATION
    )
    _apply_zero_based_task_hierarchy(
        [right_task, left_task, right_foot_task, left_foot_task],
        [torso_ori],
        [posture],
    )
    posture.weight = 0.002
    target_mode, _, _ = _target_solve_mode("SCALE_ELASTIC")
    for task in (right_task, left_task, right_foot_task, left_foot_task):
        task.solve_mode = target_mode
        task.allow_min_error_fallback = True
    torso_ori.solve_mode = embodik.TaskSolveMode.MIN_ERROR
    torso_ori.allow_min_error_fallback = True

    opts = embodik.PositionStepOptions()
    opts.max_steps = 4
    opts.max_linear_speed = 1.8
    opts.max_angular_speed = 2.5
    opts.stall_recovery = True
    opts.adaptive_dt = True
    opts.adaptive_dt_reference_distance = 0.04
    opts.adaptive_dt_max_scale = 3.0

    # Simulate a reset from stale state: recovery handlers enabled, changed
    # posture/upright targets, and a non-initial robot configuration.
    solver.enable_stall_handler(0.0)
    q_drifted = q_initial.copy()
    q_drifted[0] += 0.05
    q_drifted[2] += 0.04
    q = _clip_q(robot, q_drifted, q_lo, q_hi)
    robot.update_configuration(q)
    _configure_g1_posture_task(posture, robot, q)
    stale_torso_target = np.asarray(
        robot.get_frame_pose(frame_map["imu_in_torso"]).homogeneous(), dtype=float
    )

    _clear_interactive_solver_transients(solver)
    q = q_initial.copy()
    robot.update_configuration(q)
    _configure_g1_posture_task(posture, robot, q)
    torso_target = np.asarray(
        robot.get_frame_pose(frame_map["imu_in_torso"]).homogeneous(), dtype=float
    )
    if np.allclose(stale_torso_target, torso_target):
        raise RuntimeError("reset smoke did not exercise stale upright target state")

    reset_targets_by_key = {
        key: np.asarray(robot.get_frame_pose(frame_map[key]).homogeneous(), dtype=float)
        for key in ("right_palm", "left_palm", "right_ankle", "left_ankle")
    }
    reset_error = max(
        _target_position_error(robot, frame_map[key], pose)
        for key, pose in reset_targets_by_key.items()
    )
    if reset_error > 1e-9:
        raise RuntimeError(f"reset recapture target error too high: {reset_error:.3e}")

    rh_move = reset_targets_by_key["right_palm"].copy()
    rh_move[:3, 3] += np.array([0.03, 0.0, 0.0])
    targets = [
        embodik.TaskTarget("right_palm_pose", rh_move, 12.0, 8.0),
        embodik.TaskTarget("left_palm_pose", reset_targets_by_key["left_palm"], 12.0, 8.0),
        embodik.TaskTarget("right_ankle_pose", reset_targets_by_key["right_ankle"], 40.0, 14.0),
        embodik.TaskTarget("left_ankle_pose", reset_targets_by_key["left_ankle"], 40.0, 14.0),
        embodik.TaskTarget("torso_upright_ori", torso_target, 0.0, 2.0),
    ]
    frame_targets = [
        (frame_map["right_palm"], rh_move),
        (frame_map["left_palm"], reset_targets_by_key["left_palm"]),
        (frame_map["right_ankle"], reset_targets_by_key["right_ankle"]),
        (frame_map["left_ankle"], reset_targets_by_key["left_ankle"]),
    ]
    result, recovered = _solve_quality_step(
        solver,
        robot,
        q,
        targets,
        opts,
        [right_task, left_task, right_foot_task, left_foot_task],
        frame_targets,
        prefer_min_error=True,
    )
    dq_norm = float(np.linalg.norm(np.asarray(result.joint_velocities, dtype=float)))
    if dq_norm <= 1e-10:
        raise RuntimeError(
            "reset smoke first post-reset move produced no motion: "
            f"status={result.status.name}, msg={getattr(result, 'status_message', '')}"
        )
    print(
        "[13][headless-reset-smoke] ok "
        f"status={result.status.name} recovered={recovered} dq={dq_norm:.3e}"
    )


_SINGLE_TARGET_KEYS = ("right_palm", "left_palm", "right_ankle", "left_ankle", "pelvis")


def _single_target_task_gains(moving_key: str) -> dict[str, tuple[float, float]]:
    gains = {
        "right_palm": (2.0, 1.5),
        "left_palm": (2.0, 1.5),
        "right_ankle": (15.0, 6.0),
        "left_ankle": (15.0, 6.0),
        "pelvis": (8.0, 4.0),
    }
    if moving_key == "right_palm":
        gains["right_palm"] = (24.0, 8.0)
    elif moving_key == "left_palm":
        gains["left_palm"] = (24.0, 8.0)
    elif moving_key == "right_ankle":
        gains["right_ankle"] = (120.0, 4.0)
        gains["left_ankle"] = (0.5, 0.2)
        gains["right_palm"] = (0.1, 0.1)
        gains["left_palm"] = (0.1, 0.1)
        gains["pelvis"] = (0.0, 0.0)
    elif moving_key == "left_ankle":
        gains["left_ankle"] = (120.0, 4.0)
        gains["right_ankle"] = (0.5, 0.2)
        gains["right_palm"] = (0.1, 0.1)
        gains["left_palm"] = (0.1, 0.1)
        gains["pelvis"] = (0.0, 0.0)
    elif moving_key == "pelvis":
        gains["pelvis"] = (55.0, 16.0)
        gains["right_palm"] = (1.0, 0.8)
        gains["left_palm"] = (1.0, 0.8)
        gains["right_ankle"] = (4.0, 2.0)
        gains["left_ankle"] = (4.0, 2.0)
    else:
        raise ValueError(f"unsupported single target: {moving_key}")
    return gains


def _single_target_offset(key: str, phase: float) -> np.ndarray:
    if key in {"right_palm", "left_palm"}:
        return np.array(
            [
                0.075 * np.sin(phase),
                0.025 * np.sin(0.5 * phase),
                0.020 * np.sin(1.3 * phase),
            ],
            dtype=float,
        )
    if key == "pelvis":
        return np.array(
            [
                0.040 * np.sin(phase),
                0.022 * np.sin(0.8 * phase),
                0.018 * np.sin(1.1 * phase),
            ],
            dtype=float,
        )
    return np.array(
        [
            0.028 * np.sin(phase),
            0.016 * np.sin(0.7 * phase),
            0.010 * np.sin(1.1 * phase),
        ],
        dtype=float,
    )


def run_headless_single_target_oscillation(target_key: str, steps: int) -> None:
    """Stress one target gizmo moving repeatedly back and forth."""
    if target_key not in _SINGLE_TARGET_KEYS:
        raise ValueError(f"unsupported single target: {target_key}")
    robot = create_g1_robot_model(floating_base=True, reduced_ik=True)
    q = _apply_g1_soft_knee_seed(robot, robot.neutral_configuration())
    q_lo, q_hi = robot.get_joint_limits()
    robot.update_configuration(q)
    frame_map = resolve_frames_for_g1_base_mode(robot.get_frame_names())
    pelvis_upright_frame = (
        "pelvis" if "pelvis" in robot.get_frame_names() else frame_map["imu_in_torso"]
    )

    r0 = np.asarray(robot.get_frame_pose(frame_map["right_ankle"]).translation, dtype=float)
    l0 = np.asarray(robot.get_frame_pose(frame_map["left_ankle"]).translation, dtype=float)
    q = ground_floating_base_from_feet_center(q, r0, l0)
    q = _clip_q(robot, q, q_lo, q_hi)
    robot.update_configuration(q)

    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    _configure_interactive_elastic_band(solver)

    right_task = solver.add_frame_task(
        "right_palm_pose", frame_map["right_palm"], embodik.TaskType.FRAME_POSE
    )
    left_task = solver.add_frame_task(
        "left_palm_pose", frame_map["left_palm"], embodik.TaskType.FRAME_POSE
    )
    right_foot_task = solver.add_frame_task(
        "right_ankle_pose", frame_map["right_ankle"], embodik.TaskType.FRAME_POSE
    )
    left_foot_task = solver.add_frame_task(
        "left_ankle_pose", frame_map["left_ankle"], embodik.TaskType.FRAME_POSE
    )
    pelvis_task = solver.add_frame_task(
        "pelvis_pose", pelvis_upright_frame, embodik.TaskType.FRAME_POSE
    )
    posture = solver.add_posture_task("posture")
    torso_ori = solver.add_frame_task(
        "torso_upright_ori", frame_map["imu_in_torso"], embodik.TaskType.FRAME_ORIENTATION
    )
    _apply_zero_based_task_hierarchy(
        [right_task, left_task, right_foot_task, left_foot_task, pelvis_task],
        [torso_ori],
        [posture],
    )
    target_mode, _, _ = _target_solve_mode("SCALE_ELASTIC")
    for task in (right_task, left_task, right_foot_task, left_foot_task, pelvis_task):
        task.solve_mode = target_mode
        task.allow_min_error_fallback = True
    torso_ori.solve_mode = embodik.TaskSolveMode.MIN_ERROR
    torso_ori.allow_min_error_fallback = True
    posture.weight = 0.01
    _configure_g1_posture_task(posture, robot, q)

    torso_target = np.asarray(
        robot.get_frame_pose(frame_map["imu_in_torso"]).homogeneous(), dtype=float
    )
    target_frame_by_key = {
        "right_palm": frame_map["right_palm"],
        "left_palm": frame_map["left_palm"],
        "right_ankle": frame_map["right_ankle"],
        "left_ankle": frame_map["left_ankle"],
        "pelvis": pelvis_upright_frame,
    }
    home_poses = {
        key: np.asarray(robot.get_frame_pose(frame_name).homogeneous(), dtype=float)
        for key, frame_name in target_frame_by_key.items()
    }

    opts = embodik.PositionStepOptions()
    opts.max_steps = (
        10
        if target_key == "pelvis"
        else 6 if target_key in {"right_palm", "left_palm", "right_ankle", "left_ankle"} else 4
    )
    if target_key in {"right_ankle", "left_ankle"}:
        opts.max_steps = 8
    opts.max_linear_speed = 1.8
    opts.max_angular_speed = 2.5
    opts.stall_recovery = True
    opts.adaptive_dt = True
    opts.adaptive_dt_reference_distance = 0.04
    opts.adaptive_dt_max_scale = 3.0

    statuses: list[str] = []
    solve_ms: list[float] = []
    max_target_error = 0.0
    max_q_step = 0.0
    max_consecutive_nonproductive = 0
    consecutive_nonproductive = 0
    zero_motion_events = 0
    recovered_steps = 0
    accepted = 0
    samples = max(16, int(steps))
    previous_target = home_poses[target_key].copy()
    gains = _single_target_task_gains(target_key)
    for phase in np.linspace(0.0, 8.0 * np.pi, samples, endpoint=False):
        q_before = q.copy()
        target_poses = {key: pose.copy() for key, pose in home_poses.items()}
        moving_target = target_poses[target_key]
        moving_target[:3, 3] += _single_target_offset(target_key, float(phase))
        target_delta = float(np.linalg.norm(moving_target[:3, 3] - previous_target[:3, 3]))
        previous_target = moving_target.copy()
        targets = [
            embodik.TaskTarget("right_palm_pose", target_poses["right_palm"], *gains["right_palm"]),
            embodik.TaskTarget("left_palm_pose", target_poses["left_palm"], *gains["left_palm"]),
            embodik.TaskTarget(
                "right_ankle_pose", target_poses["right_ankle"], *gains["right_ankle"]
            ),
            embodik.TaskTarget("left_ankle_pose", target_poses["left_ankle"], *gains["left_ankle"]),
            embodik.TaskTarget("pelvis_pose", target_poses["pelvis"], *gains["pelvis"]),
            embodik.TaskTarget("torso_upright_ori", torso_target, 0.0, 2.0),
        ]
        frame_targets = [
            (target_frame_by_key[key], target_poses[key]) for key in _SINGLE_TARGET_KEYS
        ]
        quality_frame_targets = [(target_frame_by_key[target_key], moving_target)]
        t0 = time.perf_counter()
        result, recovered = _solve_quality_step(
            solver,
            robot,
            q,
            targets,
            opts,
            [right_task, left_task, right_foot_task, left_foot_task, pelvis_task],
            quality_frame_targets,
            prefer_min_error=False,
        )
        if recovered:
            recovered_steps += 1
        solve_ms.append((time.perf_counter() - t0) * 1e3)
        if not hasattr(result, "q_solution") or result.status.name == "COLLISION_VIOLATED":
            statuses.append(result.status.name)
            consecutive_nonproductive += 1
            max_consecutive_nonproductive = max(
                max_consecutive_nonproductive, consecutive_nonproductive
            )
            continue
        q_candidate = np.asarray(result.q_solution, dtype=float).copy()
        if not np.all(np.isfinite(q_candidate)):
            statuses.append(result.status.name)
            consecutive_nonproductive += 1
            max_consecutive_nonproductive = max(
                max_consecutive_nonproductive, consecutive_nonproductive
            )
            continue
        q_limited, q_step_component = _limit_tangent_step(robot, q_before, q_candidate, 0.8)
        q = q_limited
        robot.update_configuration(q)
        accepted += 1
        max_q_step = max(max_q_step, q_step_component)
        dq_norm = float(np.linalg.norm(np.asarray(robot.difference(q_before, q), dtype=float)))
        if target_delta > 1e-4 and dq_norm <= 1e-8:
            zero_motion_events += 1
        target_error = _target_position_error(robot, target_frame_by_key[target_key], moving_target)
        max_target_error = max(max_target_error, target_error)
        productive = (
            result.status == embodik.SolverStatus.SUCCESS
            or (target_delta <= 1e-4 and target_error <= 0.02)
            or (dq_norm > 1e-8 and target_error <= 0.03)
        )
        statuses.append("SUCCESS" if productive else result.status.name)
        if productive:
            consecutive_nonproductive = 0
        else:
            consecutive_nonproductive += 1
            max_consecutive_nonproductive = max(
                max_consecutive_nonproductive, consecutive_nonproductive
            )

    p95_ms = _percentile(solve_ms, 95.0)
    status_counts = {name: statuses.count(name) for name in sorted(set(statuses))}
    if accepted < samples:
        raise RuntimeError(
            f"{target_key} oscillation accepted {accepted}/{samples} samples "
            f"status_counts={status_counts}"
        )
    if any(status != "SUCCESS" for status in statuses):
        raise RuntimeError(
            f"{target_key} oscillation nonproductive statuses: {status_counts} "
            f"max_target_error={max_target_error:.4f} "
            f"max_consecutive_nonproductive={max_consecutive_nonproductive}"
        )
    if zero_motion_events:
        raise RuntimeError(f"{target_key} oscillation had zero-motion events: {zero_motion_events}")
    if max_consecutive_nonproductive:
        raise RuntimeError(
            f"{target_key} oscillation had consecutive nonproductive ticks: "
            f"{max_consecutive_nonproductive}"
        )
    if max_target_error > 0.025:
        raise RuntimeError(
            f"{target_key} oscillation target error too high: {max_target_error:.4f} m"
        )
    if p95_ms > 140.0:
        raise RuntimeError(
            f"{target_key} oscillation p95 solve time too high: {p95_ms:.2f} ms "
            f"max_target_error={max_target_error:.4f} recovered_steps={recovered_steps} "
            f"status_counts={status_counts}"
        )
    print(
        "[13][headless-single-target-oscillation] ok "
        f"target={target_key} samples={samples} accepted={accepted} p95_ms={p95_ms:.2f} "
        f"max_target_error={max_target_error:.4f} max_q_step={max_q_step:.4f} "
        f"recovered_steps={recovered_steps} status_counts={status_counts}"
    )


def run_headless_left_ee_oscillation(steps: int) -> None:
    """Backward-compatible left-hand-only stress entrypoint."""
    run_headless_single_target_oscillation("left_palm", steps)
