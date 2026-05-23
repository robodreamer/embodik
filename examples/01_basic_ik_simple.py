"""Minimal interactive IK example.

This is the shortest path for trying embodiK on a fixed-base robot:
load a robot preset, add one end-effector task plus a posture bias, drag the
target transform, and step IK in a loop.

To try a new robot model, add a preset in ``examples/robot_models`` or extend
``utils.robot_models`` with the URDF path, target link, and default posture.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import embodik
import numpy as np
from embodik import Rt, create_robot_visualizer, q2r, r2q
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
    configure_solver_runtime_policy,
    quiet_websocket_handshake_logs,
)
from utils.robot_models import load_robot_presets, resolve_robot_configuration

quiet_websocket_handshake_logs()


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
        status = viz.gui.add_text("Status", initial_value="Ready")
        reset_button = viz.gui.add_button("Reset Robot & Target")

    def reset() -> None:
        nonlocal q
        q = q_default.copy()
        robot.update_configuration(q)
        viz.display(q)
        pose = robot.get_frame_pose(target_link)
        target.position = tuple(pose.translation)
        target.wxyz = tuple(r2q(pose.rotation))
        status.value = "Reset"

    @reset_button.on_click
    def _(_) -> None:
        reset()

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
        result = solver.solve_position_step(q, target_pose, "ee_task", step_opts)

        if result.status in accepted_statuses:
            q = np.asarray(result.q_solution, dtype=float)
            robot.update_configuration(q)
            viz.display(q)

        status.value = (
            f"{result.status.name}: "
            f"pos={result.position_error * 1e3:.1f} mm, "
            f"rot={result.orientation_error:.3f} rad"
        )
        time.sleep(1e-3)


if __name__ == "__main__":
    main(parse_args())
