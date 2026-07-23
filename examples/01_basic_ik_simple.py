"""Minimal interactive IK example.

This is the shortest path for trying embodiK on a fixed-base robot:
load a robot preset, add one end-effector task plus a posture bias, drag the
target transform, and step IK in a loop.

To try a new robot model, add a preset in ``examples/robot_models`` or extend
``utils.robot_models`` with the URDF path, target link, and default posture.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from example_helpers.ik_common import (
    DEFAULT_ADAPTIVE_DT,
    DEFAULT_ADAPTIVE_DT_MAX_SCALE,
    DEFAULT_ADAPTIVE_DT_REFERENCE_DISTANCE,
    DEFAULT_NULLSPACE_ENABLED,
    DEFAULT_NULLSPACE_GAIN,
    DEFAULT_POS_GAIN,
    DEFAULT_ROT_GAIN,
    DEFAULT_SOLVER_DT,
    DEFAULT_VISER_PORT,
    SOLVER_LEVEL_ACCELERATION,
    SOLVER_LEVEL_OPTIONS,
    SOLVER_LEVEL_VELOCITY,
    configure_solver_runtime_policy,
    quiet_websocket_handshake_logs,
)
from utils.robot_models import load_robot_presets, resolve_robot_configuration

import embodik
from embodik import Rt, create_robot_visualizer, q2r, r2q

quiet_websocket_handshake_logs()

BASIC_SOLVER_LEVEL_OPTIONS = SOLVER_LEVEL_OPTIONS
DEFAULT_BASIC_ACCELERATION_LIMIT = 15.0


@dataclass
class BasicAccelerationRuntime:
    solver: Any
    target_link: str
    position_task: Any
    orientation_task: Any
    posture_task: Any
    dq: np.ndarray
    unavailable_reason: str = ""


@dataclass
class BasicAccelerationStep:
    status: Any
    q_solution: np.ndarray
    dq_solution: np.ndarray
    position_error: float
    rotation_error: float
    unavailable_reason: str = ""


def reset_basic_acceleration_state(
    runtime: BasicAccelerationRuntime | None, velocity_dimension: int
) -> None:
    if runtime is not None:
        runtime.dq = np.zeros(velocity_dimension, dtype=float)


def parse_args() -> argparse.Namespace:
    presets = load_robot_presets()
    default_robot = "panda" if "panda" in presets else sorted(presets)[0]

    parser = argparse.ArgumentParser(description="Minimal embodiK IK demo.")
    parser.add_argument("--robot", choices=sorted(presets), default=default_robot)
    parser.add_argument(
        "--visualizer",
        choices=("pinocchio", "viserurdf"),
        default="pinocchio",
        help="Use Pinocchio's ViserVisualizer or ViserUrdf.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_VISER_PORT)
    return parser.parse_args()


def make_step_options() -> embodik.PositionStepOptions:
    opts = embodik.PositionStepOptions()
    opts.position_gain = DEFAULT_POS_GAIN
    opts.orientation_gain = DEFAULT_ROT_GAIN
    opts.max_steps = 1
    opts.adaptive_dt = DEFAULT_ADAPTIVE_DT
    opts.adaptive_dt_max_scale = DEFAULT_ADAPTIVE_DT_MAX_SCALE
    opts.adaptive_dt_reference_distance = DEFAULT_ADAPTIVE_DT_REFERENCE_DISTANCE
    return opts


def acceleration_api_unavailable_reason() -> str:
    required = (
        "AccelerationSolver",
        "AccelerationTaskReference",
        "AccelerationSolveOptions",
    )
    missing = [name for name in required if not hasattr(embodik, name)]
    if missing:
        return "missing Python acceleration API: " + ", ".join(missing)
    if not hasattr(embodik, "TaskType"):
        return "missing Python TaskType API"
    return ""


def _acceleration_reference(dimension: int, gain: float) -> Any:
    reference = embodik.AccelerationTaskReference()
    reference.desired_velocity = np.zeros(dimension)
    reference.desired_acceleration = np.zeros(dimension)
    reference.proportional_gain = max(float(gain), 0.0)
    reference.derivative_gain = 2.0 * math.sqrt(reference.proportional_gain)
    return reference


def _rotation_error_rad(current: np.ndarray, target: np.ndarray) -> float:
    relative = np.asarray(target, dtype=float).T @ np.asarray(current, dtype=float)
    cos_theta = (float(np.trace(relative)) - 1.0) * 0.5
    return float(math.acos(min(1.0, max(-1.0, cos_theta))))


def make_basic_acceleration_options(
    robot: embodik.RobotModel,
    acceleration_limit: float = DEFAULT_BASIC_ACCELERATION_LIMIT,
) -> Any:
    options = embodik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.full(robot.nv, float(acceleration_limit), dtype=float)
    options.apply_velocity_limits = True
    options.apply_position_limits = True
    return options


def _try_configure_basic_acceleration_runtime(
    robot: embodik.RobotModel,
    target_link: str,
    q_default: np.ndarray,
) -> tuple[BasicAccelerationRuntime | None, str]:
    reason = acceleration_api_unavailable_reason()
    if reason:
        return None, reason

    try:
        solver = embodik.AccelerationSolver(robot)
        position_task = solver.add_frame_task(
            "ee_position", target_link, embodik.TaskType.FRAME_POSITION
        )
        orientation_task = solver.add_frame_task(
            "ee_orientation", target_link, embodik.TaskType.FRAME_ORIENTATION
        )
        posture_task = solver.add_posture_task("posture_bias")
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    for task in (position_task, orientation_task):
        task.priority = 0
        task.weight = 1.0
        task.solve_mode = embodik.TaskSolveMode.SCALE
        task.allow_min_error_fallback = False

    posture_task.priority = 1
    posture_task.weight = DEFAULT_NULLSPACE_GAIN if DEFAULT_NULLSPACE_ENABLED else 0.0
    posture_task.solve_mode = embodik.TaskSolveMode.MIN_ERROR
    posture_task.allow_min_error_fallback = False
    posture_task.set_target_configuration(np.asarray(q_default, dtype=float))

    return (
        BasicAccelerationRuntime(
            solver=solver,
            target_link=target_link,
            position_task=position_task,
            orientation_task=orientation_task,
            posture_task=posture_task,
            dq=np.zeros(robot.nv, dtype=float),
        ),
        "",
    )


def configure_basic_acceleration_runtime(
    robot: embodik.RobotModel,
    target_link: str,
    q_default: np.ndarray,
) -> BasicAccelerationRuntime | None:
    runtime, _ = _try_configure_basic_acceleration_runtime(robot, target_link, q_default)
    return runtime


def solve_basic_acceleration_step(
    runtime: BasicAccelerationRuntime,
    robot: embodik.RobotModel,
    q: np.ndarray,
    target_pose: Any,
    q_default: np.ndarray,
    *,
    pos_gain: float,
    rot_gain: float,
    posture_gain: float,
    dt: float,
    acceleration_limit: float = DEFAULT_BASIC_ACCELERATION_LIMIT,
) -> BasicAccelerationStep:
    q_current = np.asarray(q, dtype=float)
    runtime.position_task.set_target_position(np.asarray(target_pose.translation, dtype=float))
    runtime.orientation_task.set_target_orientation(np.asarray(target_pose.rotation, dtype=float))
    runtime.posture_task.set_target_configuration(np.asarray(q_default, dtype=float))
    runtime.posture_task.weight = max(float(posture_gain), 0.0)

    runtime.solver.set_task_reference("ee_position", _acceleration_reference(3, pos_gain))
    runtime.solver.set_task_reference("ee_orientation", _acceleration_reference(3, rot_gain))
    runtime.solver.set_task_reference(
        "posture_bias",
        _acceleration_reference(robot.nv, max(float(posture_gain), 0.0)),
    )

    options = make_basic_acceleration_options(robot, acceleration_limit)
    result = runtime.solver.solve(q_current, runtime.dq, float(dt), options)

    q_solution = q_current.copy()
    dq_solution = np.zeros(robot.nv, dtype=float)
    if result.status is embodik.SolverStatus.SUCCESS:
        q_solution = np.asarray(result.q_solution, dtype=float)
        dq_solution = np.asarray(result.joint_velocities_next, dtype=float)

    runtime.dq = dq_solution
    robot.update_configuration(q_solution)
    current_pose = robot.get_frame_pose(runtime.target_link)

    return BasicAccelerationStep(
        status=result.status,
        q_solution=q_solution,
        dq_solution=dq_solution,
        position_error=float(
            np.linalg.norm(
                np.asarray(current_pose.translation, dtype=float)
                - np.asarray(target_pose.translation, dtype=float)
            )
        ),
        rotation_error=_rotation_error_rad(
            np.asarray(current_pose.rotation, dtype=float),
            np.asarray(target_pose.rotation, dtype=float),
        ),
    )


def basic_acceleration_runtime_status(
    robot: embodik.RobotModel,
    target_link: str,
    q_default: np.ndarray,
) -> tuple[BasicAccelerationRuntime | None, str]:
    runtime, reason = _try_configure_basic_acceleration_runtime(robot, target_link, q_default)
    if runtime is None:
        return None, reason or "failed to construct AccelerationSolver runtime"
    robot.update_configuration(np.asarray(q_default, dtype=float))
    target_pose = robot.get_frame_pose(target_link)
    step = solve_basic_acceleration_step(
        runtime,
        robot,
        np.asarray(q_default, dtype=float),
        target_pose,
        np.asarray(q_default, dtype=float),
        pos_gain=DEFAULT_POS_GAIN,
        rot_gain=DEFAULT_ROT_GAIN,
        posture_gain=DEFAULT_NULLSPACE_GAIN if DEFAULT_NULLSPACE_ENABLED else 0.0,
        dt=DEFAULT_SOLVER_DT,
    )
    reset_basic_acceleration_state(runtime, robot.nv)
    robot.update_configuration(np.asarray(q_default, dtype=float))
    if step.status is not embodik.SolverStatus.SUCCESS:
        return None, f"stationary acceleration smoke solve returned {step.status.name}"
    return runtime, ""


def main(args: argparse.Namespace) -> None:
    config: dict[str, Any] = resolve_robot_configuration(args.robot)
    robot: embodik.RobotModel = config["robot"]
    target_link: str = config["target_link"]
    q_default = np.asarray(config["default_configuration"], dtype=float)

    solver = embodik.KinematicsSolver(robot)
    solver.dt = DEFAULT_SOLVER_DT
    configure_solver_runtime_policy(solver)

    ee_task = solver.add_frame_task("ee_task", target_link)
    ee_task.priority = 0
    ee_task.weight = 1.0
    ee_task.solve_mode = embodik.TaskSolveMode.SCALE_ELASTIC
    ee_task.allow_min_error_fallback = False

    posture_task = solver.add_posture_task("posture_bias")
    posture_task.priority = 1
    posture_task.weight = DEFAULT_NULLSPACE_GAIN if DEFAULT_NULLSPACE_ENABLED else 0.0
    posture_task.solve_mode = embodik.TaskSolveMode.MIN_ERROR
    posture_task.allow_min_error_fallback = False
    posture_task.set_target_configuration(q_default)

    q = q_default.copy()
    robot.update_configuration(q)
    acceleration_runtime, acceleration_unavailable_reason = basic_acceleration_runtime_status(
        robot, target_link, q_default
    )

    preset = load_robot_presets()[args.robot.lower()]
    viz = create_robot_visualizer(
        robot_model=robot,
        backend=args.visualizer,
        description_name=str(preset.get("description_name", "")),
        port=args.port,
        open_browser=True,
    )
    viz.add_grid("/ground", width=2, height=2)
    viz.display(q)

    initial_pose = robot.get_frame_pose(target_link)
    target = viz.scene.add_transform_controls(
        "/ik_target",
        scale=0.2,
        position=tuple(initial_pose.translation),
        wxyz=tuple(r2q(initial_pose.rotation)),
    )

    with viz.gui.add_folder("IK"):
        solver_level_options = (
            BASIC_SOLVER_LEVEL_OPTIONS
            if acceleration_runtime is not None
            else (SOLVER_LEVEL_VELOCITY,)
        )
        solver_level = viz.gui.add_dropdown(
            "Solver Level",
            options=solver_level_options,
            initial_value=SOLVER_LEVEL_VELOCITY,
        )
        status = viz.gui.add_text("Status", initial_value="Ready")
        reset_button = viz.gui.add_button("Reset Robot & Target")

    def reset() -> None:
        nonlocal q
        q = q_default.copy()
        reset_basic_acceleration_state(acceleration_runtime, robot.nv)
        robot.update_configuration(q)
        viz.display(q)
        pose = robot.get_frame_pose(target_link)
        target.position = tuple(pose.translation)
        target.wxyz = tuple(r2q(pose.rotation))
        status.value = "Reset"

    @reset_button.on_click
    def _(_) -> None:
        reset()

    @solver_level.on_update
    def _(_) -> None:
        reset_basic_acceleration_state(acceleration_runtime, robot.nv)
        if solver_level.value == SOLVER_LEVEL_ACCELERATION and acceleration_runtime is None:
            solver_level.value = SOLVER_LEVEL_VELOCITY
            status.value = f"Acceleration unavailable: {acceleration_unavailable_reason}"

    step_opts = make_step_options()
    accepted_statuses = {
        embodik.SolverStatus.SUCCESS,
        embodik.SolverStatus.INFEASIBLE,
        embodik.SolverStatus.NUMERICAL_ERROR,
        embodik.SolverStatus.NO_PROGRESS,
    }

    while True:
        target_pose = Rt(
            R=q2r(np.asarray(target.wxyz, dtype=float)),
            t=np.asarray(target.position, dtype=float),
        )
        if solver_level.value == SOLVER_LEVEL_ACCELERATION and acceleration_runtime is not None:
            accel_step = solve_basic_acceleration_step(
                acceleration_runtime,
                robot,
                q,
                target_pose,
                q_default,
                pos_gain=DEFAULT_POS_GAIN,
                rot_gain=DEFAULT_ROT_GAIN,
                posture_gain=DEFAULT_NULLSPACE_GAIN if DEFAULT_NULLSPACE_ENABLED else 0.0,
                dt=DEFAULT_SOLVER_DT,
            )
            if accel_step.status is embodik.SolverStatus.SUCCESS:
                q = accel_step.q_solution
                viz.display(q)
            else:
                reset_basic_acceleration_state(acceleration_runtime, robot.nv)

            status.value = (
                f"{SOLVER_LEVEL_ACCELERATION} {accel_step.status.name}: "
                f"pos={accel_step.position_error * 1e3:.1f} mm, "
                f"rot={accel_step.rotation_error:.3f} rad"
            )
            time.sleep(1e-3)
            continue

        result = solver.solve_position_step(q, target_pose, "ee_task", step_opts)

        if result.status in accepted_statuses:
            q = np.asarray(result.q_solution, dtype=float)
            robot.update_configuration(q)
            viz.display(q)
        elif acceleration_runtime is not None:
            reset_basic_acceleration_state(acceleration_runtime, robot.nv)

        status.value = (
            f"{SOLVER_LEVEL_VELOCITY} {result.status.name}: "
            f"pos={result.position_error * 1e3:.1f} mm, "
            f"rot={result.orientation_error:.3f} rad"
        )
        time.sleep(1e-3)


if __name__ == "__main__":
    main(parse_args())
