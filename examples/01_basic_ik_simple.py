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
from pathlib import Path
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
from embodik.gpu.wbc import (
    GpuWbcMultiFrameSolver,
    derive_frame_active_joint_names,
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


def create_gpu_solver(
    args,
    robot,
    urdf_path,
    robot_name,
    target_link,
    q_default,
    exclusions=(),
    collision_enabled=False,
    min_distance=0.03,
):
    """Build a fixed-base GPU solver using source-model joint and geometry indices."""
    names = derive_frame_active_joint_names(robot, target_link)
    gpu_robot = embodik.RobotModel(str(urdf_path), actuated_joint_names=names, floating_base=False)
    excluded = {frozenset(pair) for pair in exclusions}
    pairs = tuple(
        tuple(pair)
        for pair in gpu_robot.get_collision_pair_names()
        if frozenset(pair) not in excluded
    )
    source_indices = tuple(int(robot.get_joint_config_index(name)) for name in names)
    posture = tuple(np.asarray(q_default)[list(source_indices)])
    gpu_solver = GpuWbcMultiFrameSolver(
        args.gpu_wbc_manifest,
        Path(urdf_path),
        args.gpu_wbc_cache_dir,
        robot=gpu_robot,
        robot_name=robot_name,
        frames=(target_link,),
        frame_task_dimensions=(6,),
        active_joint_names=names,
        default_configuration=posture,
        solver_backend="torch_srinv",
        iterations=2,
        dt=DEFAULT_SOLVER_DT,
        position_gain=DEFAULT_POS_GAIN,
        orientation_gain=DEFAULT_ROT_GAIN,
        adaptive_dt=DEFAULT_ADAPTIVE_DT,
        adaptive_dt_max_scale=DEFAULT_ADAPTIVE_DT_MAX_SCALE,
        adaptive_dt_reference_distance=DEFAULT_ADAPTIVE_DT_REFERENCE_DISTANCE,
        collision_pairs=pairs,
        collision_min_distance_m=min_distance,
        collision_query_distance_m=0.15,
        posture_target_configuration=posture,
        posture_velocity_indices=tuple(
            int(gpu_robot.get_joint_velocity_index(name)) for name in names
        ),
        posture_weights=tuple(1.0 for _ in names),
        posture_gain=DEFAULT_NULLSPACE_GAIN if DEFAULT_NULLSPACE_ENABLED else 0.0,
    )
    gpu_solver.configure_runtime(collision_enabled=bool(pairs) and collision_enabled)
    return gpu_solver, source_indices, bool(pairs)


class GpuControls:
    """GPU-only controls shared by the two introductory examples."""

    def __init__(self, gui, scene, collision_supported, collision_enabled, min_distance):
        self.scene = scene
        self.collision_supported = collision_supported
        self.line = None
        with gui.add_folder("GPU settings (CPU unchanged)"):
            self.collision = gui.add_checkbox(
                "Self Collision",
                initial_value=collision_enabled and collision_supported,
                disabled=not collision_supported,
            )
            self.distance = gui.add_slider(
                "Minimum Distance (m)", min=0.001, max=0.10, step=0.001, initial_value=min_distance
            )
            self.debug = gui.add_checkbox(
                "Show Collision Debug", initial_value=False, disabled=not collision_supported
            )
            self.posture = gui.add_slider(
                "Posture Gain",
                min=0.0,
                max=10.0,
                step=0.001,
                initial_value=DEFAULT_NULLSPACE_GAIN if DEFAULT_NULLSPACE_ENABLED else 0.0,
            )
            self.adaptive = gui.add_checkbox("Adaptive dt", initial_value=DEFAULT_ADAPTIVE_DT)
            self.pos = gui.add_slider(
                "Position Gain", min=0.1, max=100.0, step=0.1, initial_value=DEFAULT_POS_GAIN
            )
            self.rot = gui.add_slider(
                "Orientation Gain", min=0.1, max=100.0, step=0.1, initial_value=DEFAULT_ROT_GAIN
            )

    def solve(self, solver, q, target):
        solver.configure_runtime(
            collision_enabled=bool(self.collision.value and self.collision_supported),
            collision_min_distance_m=(
                float(self.distance.value) if self.collision_supported else None
            ),
            posture_gain=float(self.posture.value),
            adaptive_dt=bool(self.adaptive.value),
            frame_position_gains=(float(self.pos.value),),
            frame_orientation_gains=(float(self.rot.value),),
        )
        result = solver.solve_step(
            q, (target,), include_collision_debug=bool(self.debug.value and self.collision.value)
        )
        self.show_debug(result.collision_debug)
        return result

    def show_debug(self, debug=None):
        if self.line is not None:
            self.line.visible = False
        if debug is not None and self.debug.value and self.collision.value:
            points = np.asarray([[debug.point_a_world, debug.point_b_world]], dtype=float)
            if self.line is None:
                self.line = self.scene.add_line_segments(
                    "/gpu_collision_debug",
                    points=points,
                    colors=np.asarray([[[255, 50, 50], [50, 255, 50]]], dtype=np.uint8),
                    line_width=3.0,
                )
            else:
                self.line.points = points
            self.line.visible = True


def parse_args() -> argparse.Namespace:
    presets = load_robot_presets()
    default_robot = "panda" if "panda" in presets else sorted(presets)[0]

    parser = argparse.ArgumentParser(description="Minimal embodiK IK demo.")
    parser.add_argument("--robot", choices=sorted(presets), default=default_robot)
    parser.add_argument(
        "--visualizer",
        choices=("pinocchio", "viserurdf"),
        default="viserurdf",
        help="Use ViserUrdf (default) or Pinocchio's ViserVisualizer.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_VISER_PORT)
    parser.add_argument("--gpu-wbc", action="store_true")
    parser.add_argument("--gpu-wbc-manifest", type=Path)
    parser.add_argument(
        "--gpu-wbc-cache-dir",
        type=Path,
        default=Path("build/gpu-wbc-newton-cache"),
    )
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
    urdf_path = Path(config["urdf_path"])

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

    gpu_solver: GpuWbcMultiFrameSolver | None = None
    gpu_target_offset = None
    gpu_fault: str | None = None
    if args.gpu_wbc:
        import importlib

        collision_example = importlib.import_module("02_collision_aware_IK")
        collision_cfg = collision_example.resolve_robot_configuration(args.robot)
        gpu_solver, gpu_indices, gpu_collision_supported = create_gpu_solver(
            args,
            robot,
            urdf_path,
            str(config["key"]),
            target_link,
            q_default,
            exclusions=collision_cfg.collision_exclusions,
        )
        visible_pose = robot.get_frame_pose(target_link)
        solver_pose = robot.get_frame_pose(gpu_solver.frames[0])
        gpu_target_offset = visible_pose.inverse() * solver_pose
        gpu_solver.warm_up(
            q[list(gpu_indices)],
            (visible_pose * gpu_target_offset,),
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
        backend_select = viz.gui.add_dropdown(
            "Solver Backend",
            options=("CPU EmbodiK",) + (("GPU Newton/Warp",) if gpu_solver is not None else ()),
            initial_value="CPU EmbodiK",
        )
        status = viz.gui.add_text("Status", initial_value="Ready")
        reset_button = viz.gui.add_button("Reset Robot & Target")

    gpu_controls = (
        GpuControls(viz.gui, viz.scene, gpu_collision_supported, False, 0.03)
        if gpu_solver is not None
        else None
    )

    def reset() -> None:
        nonlocal q, gpu_fault
        gpu_fault = None
        if gpu_solver is not None:
            gpu_solver.reset_state()
            gpu_controls.show_debug()
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

    @backend_select.on_update
    def _(_) -> None:
        nonlocal gpu_fault
        gpu_fault = None
        reset_basic_acceleration_state(acceleration_runtime, robot.nv)
        if gpu_solver is not None:
            gpu_solver.reset_state()
            gpu_controls.show_debug()

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
        if backend_select.value == "GPU Newton/Warp":
            assert gpu_solver is not None
            assert gpu_target_offset is not None
            if gpu_fault is None:
                try:
                    gpu_result = gpu_controls.solve(
                        gpu_solver,
                        q[list(gpu_indices)],
                        target_pose * gpu_target_offset,
                    )
                except Exception as exc:
                    gpu_fault = f"{type(exc).__name__}: {exc}"
                    gpu_controls.show_debug()
                    status.value = f"GPU FAULT — SAFE HOLD: {gpu_fault}"
                else:
                    q[list(gpu_indices)] = gpu_result.joints
                    robot.update_configuration(q)
                    viz.display(q)
                    status.value = (
                        f"GPU {gpu_result.status}: "
                        f"pos={max(gpu_result.position_errors) * 1e3:.1f} mm, "
                        f"rot={max(gpu_result.rotation_errors):.3f} rad, "
                        f"wall={gpu_result.elapsed_ms:.2f} ms"
                    )
        elif solver_level.value == SOLVER_LEVEL_ACCELERATION and acceleration_runtime is not None:
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
        else:
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
