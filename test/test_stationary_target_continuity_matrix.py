#!/usr/bin/env python3
"""Long-horizon continuity gates for stationary position-step targets."""

from __future__ import annotations

import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import embodik as eik

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from utils.robot_models import resolve_robot_configuration  # noqa: E402


@dataclass(frozen=True)
class SolveModeCase:
    case_id: str
    solve_mode: eik.TaskSolveMode


SOLVE_MODE_CASES = (
    SolveModeCase("scale", eik.TaskSolveMode.SCALE),
    SolveModeCase("scale-elastic", eik.TaskSolveMode.SCALE_ELASTIC),
    SolveModeCase("min-error", eik.TaskSolveMode.MIN_ERROR),
)


def _write_decoupled_xy_stage_urdf(tmp_path: Path) -> Path:
    urdf = textwrap.dedent("""
        <robot name="decoupled_xy_stage">
          <link name="base"/>
          <link name="x_stage"/>
          <link name="tool"/>
          <joint name="slide_x" type="prismatic">
            <parent link="base"/><child link="x_stage"/>
            <axis xyz="1 0 0"/>
            <limit lower="-0.5" upper="0.5" effort="10" velocity="2"/>
          </joint>
          <joint name="slide_y" type="prismatic">
            <parent link="x_stage"/><child link="tool"/>
            <axis xyz="0 1 0"/>
            <limit lower="-0.5" upper="0.5" effort="10" velocity="2"/>
          </joint>
        </robot>
        """).strip()
    path = tmp_path / "decoupled_xy_stage.urdf"
    path.write_text(urdf, encoding="utf-8")
    return path


PUBLIC_LIMIT_JOINTS = {
    "panda": "panda_joint4",
    "iiwa": "iiwa_joint_2",
}


def _pose_matrix(robot: eik.RobotModel, frame_name: str) -> np.ndarray:
    pose = robot.get_frame_pose(frame_name)
    result = np.eye(4, dtype=float)
    result[:3, :3] = np.asarray(pose.rotation, dtype=float)
    result[:3, 3] = np.asarray(pose.translation, dtype=float)
    return result


def _compute_stationary_metrics(
    *,
    robot: eik.RobotModel,
    q_trace: list[np.ndarray],
    errors: list[float],
    progress: list[float],
    statuses: list[str],
    dt: float,
    settle_steps: int,
) -> dict[str, float | int | list[str]]:
    assert 3 <= settle_steps < len(errors)
    q_tail = q_trace[-(settle_steps + 1) :]
    q_deltas = np.vstack(
        [
            np.asarray(robot.difference(before, after), dtype=float)
            for before, after in zip(q_tail, q_tail[1:])
        ]
    )
    joint_velocities = q_deltas / dt
    joint_speeds = np.linalg.norm(joint_velocities, axis=1)
    accelerations = np.diff(joint_velocities, axis=0) / dt
    jerks = np.diff(accelerations, axis=0) / dt
    error_tail = np.asarray(errors[-settle_steps:], dtype=float)
    progress_tail = np.asarray(progress[-settle_steps:], dtype=float)
    step_products = np.einsum("ij,ij->i", joint_velocities[:-1], joint_velocities[1:])
    active_steps = (joint_speeds[:-1] > 1e-3) & (joint_speeds[1:] > 1e-3)

    q_lower, q_upper = robot.get_joint_limits()
    q_tail_array = np.vstack(q_tail)
    finite_lower = np.isfinite(q_lower)
    finite_upper = np.isfinite(q_upper)
    limit_slacks: list[np.ndarray] = []
    if np.any(finite_lower):
        limit_slacks.append(q_tail_array[:, finite_lower] - q_lower[finite_lower])
    if np.any(finite_upper):
        limit_slacks.append(q_upper[finite_upper] - q_tail_array[:, finite_upper])

    return {
        "statuses": sorted(set(statuses)),
        "no_progress_steps": int(statuses.count("NO_PROGRESS")),
        "final_error": float(errors[-1]),
        "rms_joint_velocity": float(np.sqrt(np.mean(joint_speeds**2))),
        "peak_joint_velocity": float(joint_speeds.max(initial=0.0)),
        "velocity_total_variation": float(
            np.sum(np.linalg.norm(np.diff(joint_velocities, axis=0), axis=1))
        ),
        "configuration_drift": float(np.linalg.norm(robot.difference(q_tail[0], q_tail[-1]))),
        "alternating_steps": int(np.sum((step_products < 0.0) & active_steps)),
        "error_increases": int(np.sum(np.diff(error_tail) > 1e-4)),
        "backsteps": int(np.sum(np.diff(progress_tail) < -1e-4)),
        "peak_acceleration": float(np.linalg.norm(accelerations, axis=1).max(initial=0.0)),
        "peak_jerk": float(np.linalg.norm(jerks, axis=1).max(initial=0.0)),
        "settle_error_change": float(abs(error_tail[-1] - error_tail[0])),
        "minimum_limit_slack": float(min(np.min(slack) for slack in limit_slacks)),
    }


def _run_stationary_unreachable_position_target(
    *,
    solve_mode: eik.TaskSolveMode,
    robot_key: str = "panda",
    multi_target: bool = False,
    target_offset: np.ndarray | None = None,
    steps: int = 480,
    settle_steps: int = 120,
) -> dict[str, float | int | list[str]]:
    config = resolve_robot_configuration(robot_key)
    robot = config["robot"]
    frame_name = str(config["target_link"])
    q = np.asarray(config["default_configuration"], dtype=float)
    robot.update_configuration(q)

    initial_pose = _pose_matrix(robot, frame_name)
    target_pose = initial_pose.copy()
    if target_offset is None:
        target_offset = np.array([1.2, 0.4, 0.6], dtype=float)
    target_offset = np.asarray(target_offset, dtype=float)
    target_pose[:3, 3] += target_offset
    progress_axis = target_offset / np.linalg.norm(target_offset)

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    runtime = eik.SolverRuntimeConfig()
    runtime.weighted_fallback_enabled = False
    runtime.enable_auto_task_layout = False
    solver.configure_runtime(runtime)

    task = solver.add_frame_task("stationary_target", frame_name, eik.TaskType.FRAME_POSITION)
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = solve_mode
    options = eik.PositionStepOptions()
    options.max_steps = 2
    options.dt = 0.02
    options.position_gain = 60.0
    options.orientation_gain = 0.0
    options.primary_solve_mode = solve_mode
    options.primary_allow_min_error_fallback = False
    options.max_configuration_step_norm = 0.08

    q_trace = [q.copy()]
    errors: list[float] = []
    progress: list[float] = []
    statuses: list[str] = []
    for _ in range(steps):
        if multi_target:
            result = solver.solve_position_step(
                q,
                [
                    eik.TaskTarget(
                        "stationary_target",
                        target_pose,
                        options.position_gain,
                        options.orientation_gain,
                    )
                ],
                options,
            )
        else:
            result = solver.solve_position_step(q, target_pose, "stationary_target", options)
        q = np.asarray(result.q_solution, dtype=float)
        assert q.shape == q_trace[-1].shape
        assert np.all(np.isfinite(q))
        robot.update_configuration(q)
        current_pose = _pose_matrix(robot, frame_name)
        q_trace.append(q.copy())
        errors.append(float(np.linalg.norm(target_pose[:3, 3] - current_pose[:3, 3])))
        progress.append(float(np.dot(current_pose[:3, 3] - initial_pose[:3, 3], progress_axis)))
        statuses.append(result.status.name)

    return _compute_stationary_metrics(
        robot=robot,
        q_trace=q_trace,
        errors=errors,
        progress=progress,
        statuses=statuses,
        dt=options.dt,
        settle_steps=settle_steps,
    )


def _assert_stationary_continuity(
    metrics: dict[str, float | int | list[str]],
) -> None:
    assert metrics["settle_error_change"] <= 2e-3, metrics
    assert metrics["rms_joint_velocity"] <= 0.005, metrics
    assert metrics["peak_joint_velocity"] <= 0.02, metrics
    assert metrics["velocity_total_variation"] <= 0.05, metrics
    assert metrics["configuration_drift"] <= 0.005, metrics
    assert metrics["alternating_steps"] <= 2, metrics
    assert metrics["error_increases"] <= 1, metrics
    assert metrics["backsteps"] <= 1, metrics
    assert metrics["peak_acceleration"] <= 2.5, metrics
    assert metrics["peak_jerk"] <= 250.0, metrics
    assert metrics["minimum_limit_slack"] >= -1e-9, metrics


def _run_stationary_joint_limit_target(
    *,
    robot_key: str,
    solve_mode: eik.TaskSolveMode,
    allow_weighted_fallback: bool = False,
    steps: int = 480,
    settle_steps: int = 120,
) -> dict[str, float | int | list[str]]:
    config = resolve_robot_configuration(robot_key)
    robot = config["robot"]
    frame_name = str(config["target_link"])
    q = np.asarray(config["default_configuration"], dtype=float)
    q_lower, q_upper = robot.get_joint_limits()
    finite_lower = np.isfinite(q_lower)
    finite_upper = np.isfinite(q_upper)
    q[finite_lower] = np.maximum(q[finite_lower], q_lower[finite_lower] + 1e-4)
    q[finite_upper] = np.minimum(q[finite_upper], q_upper[finite_upper] - 1e-4)
    limit_joint = PUBLIC_LIMIT_JOINTS[robot_key]
    limit_q = int(robot.get_joint_config_index(limit_joint))
    q[limit_q] = float(q_upper[limit_q] - 1e-4)
    robot.update_configuration(q)

    initial_pose = _pose_matrix(robot, frame_name)
    q_inside = q.copy()
    q_inside[limit_q] -= 1e-4
    robot.update_configuration(q_inside)
    inside_pose = _pose_matrix(robot, frame_name)
    direction = initial_pose[:3, 3] - inside_pose[:3, 3]
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    target_pose = initial_pose.copy()
    target_pose[:3, 3] += 0.35 * direction

    robot.update_configuration(q)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    runtime = eik.SolverRuntimeConfig()
    runtime.weighted_advisor_enabled = allow_weighted_fallback
    runtime.weighted_fallback_enabled = allow_weighted_fallback
    runtime.enable_auto_task_layout = False
    solver.configure_runtime(runtime)
    task = solver.add_frame_task("joint_limit_target", frame_name, eik.TaskType.FRAME_POSITION)
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = solve_mode
    task.allow_min_error_fallback = allow_weighted_fallback

    options = eik.PositionStepOptions()
    options.max_steps = 2
    options.dt = 0.02
    options.position_gain = 60.0
    options.orientation_gain = 0.0
    options.primary_solve_mode = solve_mode
    options.primary_allow_min_error_fallback = allow_weighted_fallback
    options.max_configuration_step_norm = 0.08

    q_trace = [q.copy()]
    errors: list[float] = []
    progress: list[float] = []
    statuses: list[str] = []
    weighted_fallback_steps = 0
    for _ in range(steps):
        result = solver.solve_position_step(q, target_pose, "joint_limit_target", options)
        q = np.asarray(result.q_solution, dtype=float)
        assert q.shape == q_trace[-1].shape
        assert np.all(np.isfinite(q))
        robot.update_configuration(q)
        current_pose = _pose_matrix(robot, frame_name)
        q_trace.append(q.copy())
        errors.append(float(np.linalg.norm(target_pose[:3, 3] - current_pose[:3, 3])))
        progress.append(float(np.dot(current_pose[:3, 3] - initial_pose[:3, 3], direction)))
        statuses.append(result.status.name)
        weighted_fallback_steps += int(result.weighted_fallback_used)

    metrics = _compute_stationary_metrics(
        robot=robot,
        q_trace=q_trace,
        errors=errors,
        progress=progress,
        statuses=statuses,
        dt=options.dt,
        settle_steps=settle_steps,
    )
    metrics["weighted_fallback_steps"] = int(weighted_fallback_steps)
    return metrics


def _run_far_then_reachable_target(
    *,
    robot_key: str,
    solve_mode: eik.TaskSolveMode,
    far_steps: int = 240,
    return_steps: int = 240,
    settle_steps: int = 100,
) -> dict[str, float | int | list[str] | dict[str, float | int | list[str]]]:
    config = resolve_robot_configuration(robot_key)
    robot = config["robot"]
    frame_name = str(config["target_link"])
    q = np.asarray(config["default_configuration"], dtype=float)
    q_lower, q_upper = robot.get_joint_limits()
    finite_lower = np.isfinite(q_lower)
    finite_upper = np.isfinite(q_upper)
    q[finite_lower] = np.maximum(q[finite_lower], q_lower[finite_lower] + 1e-4)
    q[finite_upper] = np.minimum(q[finite_upper], q_upper[finite_upper] - 1e-4)
    robot.update_configuration(q)

    initial_pose = _pose_matrix(robot, frame_name)
    far_offset = np.array([1.2, 0.4, 0.6], dtype=float)
    far_target = initial_pose.copy()
    far_target[:3, 3] += far_offset

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    runtime = eik.SolverRuntimeConfig()
    runtime.weighted_advisor_enabled = True
    runtime.weighted_fallback_enabled = True
    runtime.enable_auto_task_layout = False
    solver.configure_runtime(runtime)
    task = solver.add_frame_task("return_target", frame_name, eik.TaskType.FRAME_POSITION)
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = solve_mode
    task.allow_min_error_fallback = solve_mode != eik.TaskSolveMode.MIN_ERROR
    conditioning_indices = [
        int(robot.get_joint_velocity_index(name))
        for name in robot.get_joint_names()
        if "finger" not in name and "gripper" not in name
    ]
    conditioning = solver.add_manipulability_task(
        "return_conditioning", frame_name, eik.TaskType.FRAME_POSITION
    )
    conditioning.priority = 1
    conditioning.weight = 10.0
    conditioning.solve_mode = eik.TaskSolveMode.MIN_ERROR
    conditioning.allow_min_error_fallback = False
    conditioning.set_controlled_joint_indices(conditioning_indices)
    conditioning.set_regularization(0.03)

    options = eik.PositionStepOptions()
    options.max_steps = 2
    options.dt = 0.02
    options.position_gain = 60.0
    options.orientation_gain = 0.0
    options.primary_solve_mode = solve_mode
    options.primary_allow_min_error_fallback = solve_mode != eik.TaskSolveMode.MIN_ERROR
    options.max_configuration_step_norm = 0.08

    q_trace = [q.copy()]
    errors: list[float] = []
    progress: list[float] = []
    statuses: list[str] = []

    def step(target_pose: np.ndarray, progress_origin: np.ndarray, axis: np.ndarray) -> None:
        nonlocal q
        result = solver.solve_position_step(q, target_pose, "return_target", options)
        q = np.asarray(result.q_solution, dtype=float)
        assert q.shape == q_trace[-1].shape
        assert np.all(np.isfinite(q))
        robot.update_configuration(q)
        current_pose = _pose_matrix(robot, frame_name)
        q_trace.append(q.copy())
        errors.append(float(np.linalg.norm(target_pose[:3, 3] - current_pose[:3, 3])))
        progress.append(float(np.dot(current_pose[:3, 3] - progress_origin, axis)))
        statuses.append(result.status.name)

    far_axis = far_offset / np.linalg.norm(far_offset)
    for _ in range(far_steps):
        step(far_target, initial_pose[:3, 3], far_axis)

    held_q = q.copy()
    held_pose = _pose_matrix(robot, frame_name)
    q_lower, q_upper = robot.get_joint_limits()
    best_displacement = -1.0
    reachable_target = held_pose.copy()
    for joint_name in robot.get_joint_names():
        if int(robot.get_joint_config_size(joint_name)) != 1:
            continue
        q_index = int(robot.get_joint_config_index(joint_name))
        if not np.isfinite(q_lower[q_index]) or not np.isfinite(q_upper[q_index]):
            continue
        midpoint = 0.5 * (q_lower[q_index] + q_upper[q_index])
        delta = float(np.clip(midpoint - held_q[q_index], -0.08, 0.08))
        if abs(delta) <= 1e-6:
            continue
        q_reachable = held_q.copy()
        q_reachable[q_index] += delta
        robot.update_configuration(q_reachable)
        candidate_pose = _pose_matrix(robot, frame_name)
        displacement = float(np.linalg.norm(candidate_pose[:3, 3] - held_pose[:3, 3]))
        if displacement > best_displacement:
            best_displacement = displacement
            reachable_target = candidate_pose
    assert best_displacement >= 1e-3
    robot.update_configuration(held_q)
    return_direction = reachable_target[:3, 3] - held_pose[:3, 3]
    return_direction /= max(float(np.linalg.norm(return_direction)), 1e-12)
    return_entry_error = float(np.linalg.norm(reachable_target[:3, 3] - held_pose[:3, 3]))
    for _ in range(return_steps):
        step(reachable_target, held_pose[:3, 3], return_direction)

    far_metrics = _compute_stationary_metrics(
        robot=robot,
        q_trace=q_trace[: far_steps + 1],
        errors=errors[:far_steps],
        progress=progress[:far_steps],
        statuses=statuses[:far_steps],
        dt=options.dt,
        settle_steps=min(settle_steps, far_steps - 1),
    )
    return_metrics = _compute_stationary_metrics(
        robot=robot,
        q_trace=q_trace[far_steps:],
        errors=errors[far_steps:],
        progress=progress[far_steps:],
        statuses=statuses[far_steps:],
        dt=options.dt,
        settle_steps=min(settle_steps, return_steps - 1),
    )

    recovery_lag = return_steps
    for index in range(far_steps, far_steps + return_steps):
        step_norm = float(np.linalg.norm(robot.difference(q_trace[index], q_trace[index + 1])))
        previous_error = return_entry_error if index == far_steps else errors[index - 1]
        if step_norm > 1e-5 and errors[index] < previous_error - 1e-6:
            recovery_lag = index - far_steps
            break

    transition_q = np.vstack(q_trace[far_steps : far_steps + 12])
    transition_steps = np.linalg.norm(np.diff(transition_q, axis=0), axis=1)
    transition_accel = np.linalg.norm(np.diff(transition_q, n=2, axis=0), axis=1)
    transition_jerk = np.linalg.norm(np.diff(transition_q, n=3, axis=0), axis=1)
    return {
        "far": far_metrics,
        "returned": return_metrics,
        "recovery_lag_steps": int(recovery_lag),
        "max_transition_step": float(transition_steps.max(initial=0.0)),
        "max_transition_accel": float(transition_accel.max(initial=0.0)),
        "max_transition_jerk": float(transition_jerk.max(initial=0.0)),
    }


def _run_stationary_weighted_fallback_limit_conflict(
    *, steps: int = 720, settle_steps: int = 120
) -> dict[str, float | int | list[str]]:
    config = resolve_robot_configuration("panda")
    robot = config["robot"]
    q = np.asarray(config["default_configuration"], dtype=float)
    q_lower, q_upper = robot.get_joint_limits()
    q = np.clip(q, q_lower + 0.02, q_upper - 0.02)
    blocked_q = int(robot.get_joint_config_index("panda_joint1"))
    useful_q = int(robot.get_joint_config_index("panda_joint2"))
    useful_v = int(robot.get_joint_velocity_index("panda_joint2"))
    q[blocked_q] = float(q_upper[blocked_q])
    robot.update_configuration(q)
    q_start = q.copy()
    useful_target = float(q[useful_q] + 0.3)

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.0)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    runtime = eik.SolverRuntimeConfig()
    runtime.weighted_advisor_enabled = True
    runtime.weighted_fallback_enabled = True
    runtime.enable_auto_task_layout = False
    solver.configure_runtime(runtime)

    blocked = solver.add_joint_task(
        "blocked_joint1",
        "panda_joint1",
        target_value=float(q_upper[blocked_q] + 1.0),
    )
    blocked.priority = 0
    blocked.weight = 1.0
    blocked.allow_min_error_fallback = False

    useful = solver.add_posture_task("useful_joint2", [useful_v])
    useful.priority = 1
    useful.weight = 1.0
    useful.allow_min_error_fallback = False
    useful.set_controlled_joint_targets(np.array([useful_target]))

    q_trace = [q.copy()]
    errors: list[float] = []
    progress: list[float] = []
    statuses: list[str] = []
    weighted_fallback_steps = 0
    for _ in range(steps):
        result = solver.solve_velocity(q, apply_limits=True)
        dq = np.asarray(result.solution, dtype=float)
        assert dq.shape == (robot.nv,)
        assert np.all(np.isfinite(dq))
        q = np.asarray(robot.integrate(q, dq, solver.dt), dtype=float)
        robot.update_configuration(q)
        q_trace.append(q.copy())
        errors.append(abs(useful_target - float(q[useful_q])))
        progress.append(float(q[useful_q] - q_start[useful_q]))
        statuses.append(result.status.name)
        weighted_fallback_steps += int(result.weighted_fallback_used)

    metrics = _compute_stationary_metrics(
        robot=robot,
        q_trace=q_trace,
        errors=errors,
        progress=progress,
        statuses=statuses,
        dt=solver.dt,
        settle_steps=settle_steps,
    )
    metrics["weighted_fallback_steps"] = int(weighted_fallback_steps)
    metrics["useful_progress"] = float(q[useful_q] - q_start[useful_q])
    return metrics


def test_min_error_stationary_unreachable_target_settles_without_self_motion() -> None:
    metrics = _run_stationary_unreachable_position_target(solve_mode=eik.TaskSolveMode.MIN_ERROR)
    _assert_stationary_continuity(metrics)


def test_multi_target_min_error_stationary_target_settles_without_self_motion() -> None:
    metrics = _run_stationary_unreachable_position_target(
        solve_mode=eik.TaskSolveMode.MIN_ERROR,
        multi_target=True,
    )
    _assert_stationary_continuity(metrics)


@pytest.mark.benchmark
@pytest.mark.parametrize("robot_key", ("panda", "iiwa"))
@pytest.mark.parametrize(
    "mode_case", SOLVE_MODE_CASES, ids=[case.case_id for case in SOLVE_MODE_CASES]
)
@pytest.mark.parametrize("multi_target", (False, True), ids=("single", "vector"))
def test_stationary_unreachable_target_continuity_across_public_modes(
    robot_key: str,
    mode_case: SolveModeCase,
    multi_target: bool,
) -> None:
    metrics = _run_stationary_unreachable_position_target(
        solve_mode=mode_case.solve_mode,
        robot_key=robot_key,
        multi_target=multi_target,
    )

    _assert_stationary_continuity(metrics)
    assert int(metrics["no_progress_steps"]) > 0, metrics


@pytest.mark.benchmark
@pytest.mark.parametrize("robot_key", ("panda", "iiwa"))
@pytest.mark.parametrize(
    "mode_case", SOLVE_MODE_CASES, ids=[case.case_id for case in SOLVE_MODE_CASES]
)
def test_stationary_reachable_target_continuity_across_public_modes(
    robot_key: str,
    mode_case: SolveModeCase,
) -> None:
    metrics = _run_stationary_unreachable_position_target(
        solve_mode=mode_case.solve_mode,
        robot_key=robot_key,
        target_offset=np.array([0.04, 0.02, 0.01], dtype=float),
        steps=300,
        settle_steps=100,
    )

    _assert_stationary_continuity(metrics)
    assert float(metrics["final_error"]) <= 1e-3, metrics


@pytest.mark.benchmark
@pytest.mark.parametrize("robot_key", ("panda", "iiwa"))
@pytest.mark.parametrize(
    "mode_case", SOLVE_MODE_CASES, ids=[case.case_id for case in SOLVE_MODE_CASES]
)
def test_stationary_joint_limit_target_continuity_across_public_modes(
    robot_key: str,
    mode_case: SolveModeCase,
) -> None:
    metrics = _run_stationary_joint_limit_target(
        robot_key=robot_key,
        solve_mode=mode_case.solve_mode,
    )

    _assert_stationary_continuity(metrics)


@pytest.mark.benchmark
def test_stationary_weighted_fallback_converges_without_tail_self_motion() -> None:
    metrics = _run_stationary_weighted_fallback_limit_conflict()

    _assert_stationary_continuity(metrics)
    assert int(metrics["weighted_fallback_steps"]) > 0, metrics
    assert float(metrics["useful_progress"]) >= 0.29, metrics
    assert float(metrics["final_error"]) <= 1e-3, metrics


@pytest.mark.benchmark
@pytest.mark.parametrize("robot_key", ("panda", "iiwa"))
@pytest.mark.parametrize(
    "mode_case", SOLVE_MODE_CASES, ids=[case.case_id for case in SOLVE_MODE_CASES]
)
def test_stationary_unreachable_hold_resumes_when_target_returns(
    robot_key: str,
    mode_case: SolveModeCase,
) -> None:
    metrics = _run_far_then_reachable_target(
        robot_key=robot_key,
        solve_mode=mode_case.solve_mode,
    )

    _assert_stationary_continuity(metrics["far"])
    _assert_stationary_continuity(metrics["returned"])
    assert float(metrics["returned"]["final_error"]) <= 1e-3, metrics
    assert int(metrics["recovery_lag_steps"]) <= 1, metrics
    assert float(metrics["max_transition_step"]) <= 0.08 + 1e-9, metrics
    assert float(metrics["max_transition_accel"]) <= 0.16 + 1e-9, metrics
    assert float(metrics["max_transition_jerk"]) <= 0.24 + 1e-9, metrics


def test_stationary_guard_does_not_reopen_after_weak_reversal_window() -> None:
    """Once exhausted motion is held, unchanged targets must stay held."""
    config = resolve_robot_configuration("panda")
    robot = config["robot"]
    frame_name = str(config["target_link"])
    q = np.asarray(config["default_configuration"], dtype=float).copy()
    robot.update_configuration(q)

    target_pose = _pose_matrix(robot, frame_name)
    target_pose[:3, 3] += np.array(
        [0.3649481129330421, 0.49391242483850917, 0.9058548242738668],
        dtype=float,
    )

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    runtime = eik.SolverRuntimeConfig()
    runtime.weighted_fallback_enabled = False
    runtime.enable_auto_task_layout = False
    solver.configure_runtime(runtime)

    task = solver.add_frame_task("stationary_target", frame_name, eik.TaskType.FRAME_POSITION)
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    posture = solver.add_posture_task("posture")
    posture.priority = 1
    posture.weight = 1.0
    posture.set_target_configuration(q.copy())

    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.dt = 0.02
    options.position_gain = 10.0
    options.orientation_gain = 0.0
    options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
    options.max_configuration_step_norm = 0.08

    held_q: np.ndarray | None = None
    post_hold_results: list[object] = []
    for _ in range(140):
        result = solver.solve_position_step(q, target_pose, "stationary_target", options)
        q = np.asarray(result.q_solution, dtype=float)
        robot.update_configuration(q)
        if held_q is None and result.position_step_hold_active:
            held_q = q.copy()
        elif held_q is not None:
            post_hold_results.append(result)

    assert held_q is not None
    assert post_hold_results
    assert all(result.position_step_hold_active for result in post_hold_results)
    for result in post_hold_results:
        np.testing.assert_allclose(result.q_solution, held_q, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(result.joint_velocities, 0.0, rtol=0.0, atol=1e-12)

    returned_target = _pose_matrix(robot, frame_name)
    returned_target[:3, 3] += np.array([0.01, -0.005, 0.005], dtype=float)
    resumed = solver.solve_position_step(q, returned_target, "stationary_target", options)
    assert resumed.position_step_hold_active is False
    assert np.linalg.norm(np.asarray(resumed.q_solution, dtype=float) - q) > 1e-6


def test_continuity_reference_frame_ignores_base_motion_but_reopens_for_local_target_change(
    tmp_path: Path,
) -> None:
    robot = eik.RobotModel(str(_write_decoupled_xy_stage_urdf(tmp_path)), floating_base=False)
    q = np.zeros(robot.nq, dtype=float)
    robot.update_configuration(q)

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    task = solver.add_frame_task("stationary_target", "tool", eik.TaskType.FRAME_POSITION)
    task.priority = 0
    task.weight = 1.0
    task.solve_mode = eik.TaskSolveMode.MIN_ERROR
    task.set_position_mask(np.array([1.0, 0.0, 0.0], dtype=float))
    posture = solver.add_posture_task("secondary_y", [1])
    posture.priority = 1
    posture.weight = 1.0
    posture.solve_mode = eik.TaskSolveMode.MIN_ERROR
    posture.set_controlled_joint_targets(np.array([0.5], dtype=float))

    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.dt = 0.02
    options.position_gain = 10.0
    options.orientation_gain = 0.0
    options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
    options.continuity_reference_frame = "x_stage"

    def target_in_reference(x_position: float) -> np.ndarray:
        robot.update_configuration(q)
        reference_pose = _pose_matrix(robot, "x_stage")
        target_pose = reference_pose.copy()
        target_pose[0, 3] += x_position
        return target_pose

    for _ in range(30):
        result = solver.solve_position_step(
            q,
            target_in_reference(0.0),
            "stationary_target",
            options,
        )
        q = np.asarray(result.q_solution, dtype=float)
    assert result.position_step_hold_active is True

    q = q.copy()
    q[0] += 0.05
    same_local_target = target_in_reference(0.0)
    moved_reference_result = solver.solve_position_step(
        q,
        same_local_target,
        "stationary_target",
        options,
    )
    assert moved_reference_result.position_step_hold_active is True
    np.testing.assert_allclose(moved_reference_result.q_solution, q, rtol=0.0, atol=1e-12)

    changed_local_target = target_in_reference(0.1)
    resumed = solver.solve_position_step(
        q,
        changed_local_target,
        "stationary_target",
        options,
    )
    assert resumed.position_step_hold_active is False
    assert np.linalg.norm(np.asarray(resumed.q_solution, dtype=float) - q) > 1e-6


def test_changed_secondary_posture_target_reopens_satisfied_primary_hold(tmp_path: Path) -> None:
    robot = eik.RobotModel(str(_write_decoupled_xy_stage_urdf(tmp_path)), floating_base=False)
    q = np.zeros(robot.nq, dtype=float)
    robot.update_configuration(q)

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    primary = solver.add_frame_task("held_x", "tool", eik.TaskType.FRAME_POSITION)
    primary.priority = 0
    primary.weight = 1.0
    primary.solve_mode = eik.TaskSolveMode.MIN_ERROR
    primary.set_position_mask(np.array([1.0, 0.0, 0.0], dtype=float))

    posture = solver.add_posture_task("secondary_y", [1])
    posture.priority = 1
    posture.weight = 1.0
    posture.solve_mode = eik.TaskSolveMode.MIN_ERROR
    posture.set_controlled_joint_targets(np.array([0.5], dtype=float))

    target_pose = _pose_matrix(robot, "tool")
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.dt = 0.02
    options.position_gain = 10.0
    options.orientation_gain = 0.0
    options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR

    for _ in range(30):
        result = solver.solve_position_step(q, target_pose, "held_x", options)
        q = np.asarray(result.q_solution, dtype=float)
    assert "held satisfied stationary target" in result.status_message
    assert result.position_step_hold_active is True

    posture.set_controlled_joint_targets(np.array([-0.5], dtype=float))
    result = solver.solve_position_step(q, target_pose, "held_x", options)
    q_next = np.asarray(result.q_solution, dtype=float)

    assert q_next[1] <= q[1] - 1e-3
    assert q_next[0] == pytest.approx(q[0], abs=1e-9)
    assert "held satisfied stationary target" not in result.status_message
    assert result.position_step_hold_active is False


@pytest.mark.parametrize("secondary_kind", ("posture", "frame"))
def test_secondary_direct_velocity_reopens_satisfied_primary_hold(
    tmp_path: Path,
    secondary_kind: str,
) -> None:
    robot = eik.RobotModel(str(_write_decoupled_xy_stage_urdf(tmp_path)), floating_base=False)
    q = np.zeros(robot.nq, dtype=float)
    robot.update_configuration(q)

    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.02
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    primary = solver.add_frame_task("held_x", "tool", eik.TaskType.FRAME_POSITION)
    primary.priority = 0
    primary.weight = 1.0
    primary.solve_mode = eik.TaskSolveMode.MIN_ERROR
    primary.set_position_mask(np.array([1.0, 0.0, 0.0], dtype=float))

    if secondary_kind == "posture":
        secondary = solver.add_posture_task("secondary_y", [1])
        secondary.set_controlled_joint_targets(np.array([0.5], dtype=float))
    else:
        secondary = solver.add_frame_task("secondary_y", "tool", eik.TaskType.FRAME_POSITION)
        secondary.set_position_mask(np.array([0.0, 1.0, 0.0], dtype=float))
        secondary_position = _pose_matrix(robot, "tool")[:3, 3]
        secondary_position[1] += 0.5
        secondary.set_target_position(secondary_position)
    secondary.priority = 1
    secondary.weight = 1.0
    secondary.solve_mode = eik.TaskSolveMode.MIN_ERROR

    target_pose = _pose_matrix(robot, "tool")
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.dt = 0.02
    options.position_gain = 10.0
    options.orientation_gain = 0.0
    options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR

    for _ in range(30):
        result = solver.solve_position_step(q, target_pose, "held_x", options)
        q = np.asarray(result.q_solution, dtype=float)
    assert "held satisfied stationary target" in result.status_message

    direct_velocity = (
        np.array([-0.05], dtype=float)
        if secondary_kind == "posture"
        else np.array([0.0, -0.05, 0.0], dtype=float)
    )
    secondary.set_target_velocity(direct_velocity)
    result = solver.solve_position_step(q, target_pose, "held_x", options)
    q_next = np.asarray(result.q_solution, dtype=float)

    assert q_next[1] <= q[1] - 5e-4
    assert q_next[0] == pytest.approx(q[0], abs=1e-9)
    assert "held satisfied stationary target" not in result.status_message
