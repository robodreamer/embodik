#!/usr/bin/env python3
"""Developer-only advanced basic IK surface.

This keeps the diagnostic/tuning controls that were intentionally removed from
the public ``examples/01_basic_ik_simple.py``:

- manual joint sliders
- joint-limit scaling
- diagnostics for joint-limit distance and manipulability
- debug logging controls
- explicit nullspace joint selection and bias controls
- adaptive-dt tuning controls
- EE solve-mode/fallback controls
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import embodik
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
    quiet_websocket_handshake_logs,
)
from utils.robot_models import load_robot_presets, resolve_robot_configuration


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
quiet_websocket_handshake_logs()

DEFAULT_TASK_WEIGHT = 1.0


def parse_args() -> argparse.Namespace:
    presets = load_robot_presets()
    default_robot = "panda" if "panda" in presets else sorted(presets)[0]

    parser = argparse.ArgumentParser(description="Developer advanced basic IK surface.")
    parser.add_argument("--robot", choices=sorted(presets), default=default_robot)
    parser.add_argument(
        "--visualizer",
        choices=("pinocchio", "viserurdf"),
        default="pinocchio",
        help="Use Pinocchio's ViserVisualizer or ViserUrdf.",
    )
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    config: dict[str, Any] = resolve_robot_configuration(args.robot)
    robot: embodik.RobotModel = config["robot"]
    target_link = str(config["target_link"])
    q_default = np.asarray(config["default_configuration"], dtype=float)

    solver = embodik.KinematicsSolver(robot)
    solver.dt = DEFAULT_SOLVER_DT
    solver.set_damping(0.1)
    solver.set_tolerance(0.1)

    q_current = q_default.copy()
    robot.update_configuration(q_current)

    q_lower_orig, q_upper_orig = robot.get_joint_limits()
    q_lower = q_lower_orig.copy()
    q_upper = q_upper_orig.copy()

    initial_pose = robot.get_frame_pose(target_link)
    preset = load_robot_presets()[args.robot.lower()]

    viz = create_robot_visualizer(
        robot_model=robot,
        backend=args.visualizer,
        description_name=str(preset.get("description_name", "")),
        port=args.port,
        open_browser=True,
    )
    viz.add_grid("/ground", width=2, height=2)
    viz.display(q_current)

    target = viz.scene.add_transform_controls(
        "/ik_target",
        scale=0.2,
        position=tuple(initial_pose.translation),
        wxyz=tuple(r2q(initial_pose.rotation)),
    )

    ee_task = solver.add_frame_task("ee_task", target_link)
    ee_task.priority = 0
    ee_task.weight = DEFAULT_TASK_WEIGHT
    ee_task.solve_mode = embodik.TaskSolveMode.SCALE
    ee_task.allow_min_error_fallback = False

    posture_task = solver.add_posture_task("posture_bias")
    posture_task.priority = 1
    posture_task.weight = 0.0
    posture_task.solve_mode = embodik.TaskSolveMode.MIN_ERROR
    posture_task.allow_min_error_fallback = False
    posture_task.set_target_configuration(q_default)

    with viz.gui.add_folder("IK Controls"):
        timing_handle = viz.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
        pos_gain = viz.gui.add_slider(
            "Position Gain", min=0.1, max=200.0, initial_value=DEFAULT_POS_GAIN, step=0.1
        )
        rot_gain = viz.gui.add_slider(
            "Orientation Gain", min=0.1, max=200.0, initial_value=DEFAULT_ROT_GAIN, step=0.1
        )
        iterations = viz.gui.add_slider("IK Iterations", min=1, max=20, initial_value=1, step=1)
        ee_mode = viz.gui.add_dropdown(
            "EE Solve Mode",
            options=("SCALE", "SCALE_ELASTIC", "MIN_ERROR"),
            initial_value="SCALE_ELASTIC",
        )
        ee_fallback = viz.gui.add_checkbox("Allow SCALE fallback to MIN_ERROR", initial_value=False)
        damping = viz.gui.add_slider("Solver Damping", min=0.01, max=1.0, initial_value=0.1, step=0.01)
        adaptive_dt = viz.gui.add_checkbox("Adaptive dt", initial_value=DEFAULT_ADAPTIVE_DT)
        adaptive_dt_scale = viz.gui.add_slider(
            "Adaptive dt Max Scale",
            min=1.0,
            max=10.0,
            step=0.5,
            initial_value=DEFAULT_ADAPTIVE_DT_MAX_SCALE,
        )
        adaptive_dt_ref = viz.gui.add_slider(
            "Adaptive dt Ref Dist (m)",
            min=0.01,
            max=0.20,
            step=0.01,
            initial_value=DEFAULT_ADAPTIVE_DT_REFERENCE_DISTANCE,
        )
        snap_target = viz.gui.add_button("Snap Target to Current EE")
        reset_robot = viz.gui.add_button("Reset Robot & Target")

    with viz.gui.add_folder("Joint Limit Scaling"):
        limit_scale = viz.gui.add_slider("Limit Scale", min=0.1, max=1.0, initial_value=1.0, step=0.01)
        limit_scale_info = viz.gui.add_text(
            "Info", initial_value="1.0 = original limits, smaller = narrower"
        )
        apply_limit_scale = viz.gui.add_button("Apply Limit Scale")
        reset_limits = viz.gui.add_button("Reset to Original Limits")

    with viz.gui.add_folder("Diagnostics", expand_by_default=False):
        jl_dist_text = viz.gui.add_text("JL Distance", initial_value="--")
        manip_text = viz.gui.add_text("Manipulability", initial_value="--")
        combined_text = viz.gui.add_text("Combined Metric", initial_value="--")

    joint_names = robot.get_joint_names()
    joint_sliders: dict[int, Any] = {}
    with viz.gui.add_folder("Joint Configuration", expand_by_default=False):
        for idx in range(min(7, len(q_current))):
            joint_name = joint_names[idx] if idx < len(joint_names) else f"joint{idx + 1}"
            joint_sliders[idx] = viz.gui.add_slider(
                f"{joint_name} (joint{idx + 1})",
                min=float(q_lower[idx]),
                max=float(q_upper[idx]),
                step=0.01,
                initial_value=float(q_current[idx]),
            )
        manual_control = viz.gui.add_checkbox("Enable Manual Control", initial_value=False)

    with viz.gui.add_folder("Debug Options"):
        enable_debug = viz.gui.add_checkbox("Enable Debug Logging", initial_value=False)
        debug_rate = viz.gui.add_slider("Debug Rate (Hz)", min=0.1, max=10.0, initial_value=2.0, step=0.1)

    nullspace_joint_checkboxes: dict[int, Any] = {}
    with viz.gui.add_folder("Nullspace Control"):
        enable_nullspace = viz.gui.add_checkbox(
            "Enable Nullspace Bias", initial_value=DEFAULT_NULLSPACE_ENABLED
        )
        nullspace_gain = viz.gui.add_slider(
            "Nullspace Gain",
            min=0.0,
            max=2.0,
            initial_value=DEFAULT_NULLSPACE_GAIN,
            step=0.05,
        )
        bias_to_initial = viz.gui.add_button("Bias to Initial Config")
        bias_to_zero = viz.gui.add_button("Bias to Zero Config")
        with viz.gui.add_folder("Joint Selection"):
            for idx in range(robot.nq):
                joint_name = joint_names[idx] if idx < len(joint_names) else f"joint{idx + 1}"
                nullspace_joint_checkboxes[idx] = viz.gui.add_checkbox(
                    f"{joint_name} (joint{idx + 1})",
                    initial_value=idx < min(7, robot.nq),
                )

    nullspace_bias = q_default.copy()

    def update_visualization(q: np.ndarray) -> None:
        viz.display(q)

    def sync_joint_sliders() -> None:
        for idx, slider in joint_sliders.items():
            if idx < len(q_current):
                slider.value = float(q_current[idx])

    def sync_target_to_current_ee() -> None:
        pose = robot.get_frame_pose(target_link)
        target.position = tuple(pose.translation)
        target.wxyz = tuple(r2q(pose.rotation))

    def apply_limit_scale_value(scale: float) -> None:
        nonlocal q_lower, q_upper, q_current
        center = 0.5 * (q_lower_orig + q_upper_orig)
        half_range = 0.5 * (q_upper_orig - q_lower_orig) * scale
        q_lower = center - half_range
        q_upper = center + half_range
        robot.set_joint_limits(q_lower, q_upper)
        q_current = np.clip(q_current, q_lower, q_upper)
        robot.update_configuration(q_current)
        update_visualization(q_current)
        for idx, slider in joint_sliders.items():
            if idx < len(q_lower):
                slider.min = float(q_lower[idx])
                slider.max = float(q_upper[idx])
                slider.value = float(q_current[idx])
        limit_scale_info.value = f"Scale={scale:.2f}  range={float(np.mean(q_upper - q_lower)):.3f} rad"
        logger.info("Joint limits scaled to %.0f%% of original range", scale * 100.0)

    @apply_limit_scale.on_click
    def _(_) -> None:
        apply_limit_scale_value(float(limit_scale.value))

    @reset_limits.on_click
    def _(_) -> None:
        limit_scale.value = 1.0
        apply_limit_scale_value(1.0)

    @bias_to_initial.on_click
    def _(_) -> None:
        nonlocal nullspace_bias
        nullspace_bias = q_default.copy()
        logger.info("Nullspace bias set to initial configuration")

    @bias_to_zero.on_click
    def _(_) -> None:
        nonlocal nullspace_bias
        nullspace_bias = np.zeros(robot.nq)
        logger.info("Nullspace bias set to zero configuration")

    @damping.on_update
    def _(_) -> None:
        solver.set_damping(float(damping.value))

    @snap_target.on_click
    def _(_) -> None:
        sync_target_to_current_ee()
        logger.info("Target snapped to current end-effector pose")

    @reset_robot.on_click
    def _(_) -> None:
        nonlocal q_current
        q_current = q_default.copy()
        robot.update_configuration(q_current)
        update_visualization(q_current)
        sync_joint_sliders()
        sync_target_to_current_ee()
        logger.info("Robot reset to default configuration")

    logger.info("Advanced basic IK dev surface - %s", config["display_name"])
    logger.info("Robot: %d DoF | target link: %s", robot.nq, target_link)

    step_opts = embodik.PositionStepOptions()
    last_debug_time = 0.0

    while True:
        if manual_control.value:
            for idx, slider in joint_sliders.items():
                if idx < len(q_current):
                    q_current[idx] = float(slider.value)
            robot.update_configuration(q_current)
            update_visualization(q_current)
            ee_task.weight = 0.0
            posture_task.weight = 0.0
            posture_task.set_controlled_joint_indices([])
            sync_target_to_current_ee()
            time.sleep(solver.dt)
            continue

        target_pose = Rt(
            R=q2r(np.asarray(target.wxyz, dtype=float)),
            t=np.asarray(target.position, dtype=float),
        )

        current_time = time.time()
        should_log_debug = enable_debug.value and (
            current_time - last_debug_time >= 1.0 / float(debug_rate.value)
        )
        start_time = time.time()

        ee_task.weight = DEFAULT_TASK_WEIGHT
        ee_task.solve_mode = getattr(embodik.TaskSolveMode, ee_mode.value, embodik.TaskSolveMode.SCALE)
        ee_task.allow_min_error_fallback = bool(ee_fallback.value)

        if enable_nullspace.value:
            active_indices = [
                idx for idx, checkbox in nullspace_joint_checkboxes.items() if checkbox.value
            ]
            if active_indices:
                posture_task.set_controlled_joint_indices(active_indices)
                posture_task.set_target_configuration(nullspace_bias)
                posture_task.weight = float(nullspace_gain.value)
            else:
                posture_task.weight = 0.0
                posture_task.set_controlled_joint_indices([])
        else:
            posture_task.weight = 0.0
            posture_task.set_controlled_joint_indices([])

        step_opts.position_gain = float(pos_gain.value)
        step_opts.orientation_gain = float(rot_gain.value)
        step_opts.max_steps = int(iterations.value)
        step_opts.adaptive_dt = bool(adaptive_dt.value)
        step_opts.adaptive_dt_max_scale = float(adaptive_dt_scale.value)
        step_opts.adaptive_dt_reference_distance = float(adaptive_dt_ref.value)
        result = solver.solve_position_step(q_current, target_pose, "ee_task", step_opts)

        if should_log_debug:
            logger.info(
                "status=%s elapsed=%.2fms pos=%.4fm rot=%.4frad adt=%s",
                result.status.name,
                result.computation_time_ms,
                result.position_error,
                result.orientation_error,
                step_opts.adaptive_dt,
            )
            last_debug_time = current_time

        if result.status in (
            embodik.SolverStatus.SUCCESS,
            embodik.SolverStatus.INFEASIBLE,
            embodik.SolverStatus.NUMERICAL_ERROR,
            embodik.SolverStatus.NO_PROGRESS,
        ):
            q_current = np.clip(np.asarray(result.q_solution, dtype=float), q_lower, q_upper)
            robot.update_configuration(q_current)
            update_visualization(q_current)
            sync_joint_sliders()

        if should_log_debug or current_time - last_debug_time >= 0.1:
            _, jl_agg = embodik.joint_limit_distance(q_current, q_lower, q_upper)
            jacobian = robot.get_frame_jacobian(target_link)
            manip = embodik.velocity_manipulability(jacobian)
            combined = embodik.singularity_joint_limit_metric(q_current, jacobian, q_lower, q_upper)
            jl_dist_text.value = f"{jl_agg:.4f}"
            manip_text.value = f"{manip:.6f}"
            combined_text.value = f"{combined:.6f}"

        timing_handle.value = 0.9 * timing_handle.value + 0.1 * ((time.time() - start_time) * 1000.0)
        time.sleep(1e-3)


if __name__ == "__main__":
    main(parse_args())
