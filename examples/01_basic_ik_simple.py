"""Basic IK Example with embodiK

Simple inverse kinematics example using velocity-based control.
"""

import argparse
import time
import logging
from typing import Any, Dict
from pathlib import Path
import numpy as np
import pinocchio as pin

import embodik
from embodik import r2q, q2r, Rt
from embodik import RobotVisualizer, create_robot_visualizer

# Import robot model utilities
from utils.robot_models import load_robot_presets, resolve_robot_configuration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Get examples directory (parent of this file)
_EXAMPLES_DIR = Path(__file__).parent



def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    # Load available robots from presets
    presets = load_robot_presets()

    parser = argparse.ArgumentParser(description="embodiK basic IK demo.")
    parser.add_argument(
        "--robot",
        choices=sorted(presets.keys()),
        default="panda" if "panda" in presets else sorted(presets.keys())[0] if presets else None,
        help="Select which robot model to load.",
    )
    parser.add_argument(
        "--visualizer",
        choices=["pinocchio", "viserurdf"],
        default="pinocchio",
        help="Visualization backend to use. 'pinocchio' (default) uses Pinocchio ViserVisualizer, "
             "'viserurdf' uses ViserUrdf for color-preserving visualization.",
    )
    return parser.parse_args()




def main(args: argparse.Namespace):
    """Main function for basic IK."""
    config = resolve_robot_configuration(args.robot)
    robot = config["robot"]
    target_link_name = config["target_link"]

    # Create solver
    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01  # Integration timestep
    solver.set_damping(0.1)  # Higher damping can help avoid numerical issues
    solver.set_tolerance(0.1)  # Tolerance for solver convergence

    # Set default configuration
    q_default = config["default_configuration"]

    # Set initial configuration
    q_current = q_default.copy()
    robot.update_configuration(q_current)

    # Original joint limits from URDF
    q_lower_orig, q_upper_orig = robot.get_joint_limits()
    q_lower = q_lower_orig.copy()
    q_upper = q_upper_orig.copy()

    # Get initial end-effector pose
    initial_pose = robot.get_frame_pose(target_link_name)

    # Set up robot visualizer with selected backend
    preset = load_robot_presets()[args.robot.lower()]
    description_name = preset.get("description_name", "")

    viz = create_robot_visualizer(
        robot_model=robot,
        backend=args.visualizer,
        description_name=description_name,
        port=8080,
        open_browser=True,
    )

    # Add grid and display initial configuration
    viz.add_grid("/ground", width=2, height=2)
    viz.display(q_current)

    # Get server, scene, and gui from visualizer
    server = viz.server
    scene = viz.scene
    gui = viz.gui

    # Helper function to update visualization (uses unified API)
    def update_visualization(q: np.ndarray):
        """Update robot visualization using robot visualizer."""
        viz.display(q)

    # Convert initial rotation matrix to quaternion for target
    initial_wxyz = tuple(r2q(initial_pose.rotation))

    # Create interactive controller at initial end-effector pose
    ik_target = server.scene.add_transform_controls(
        "/ik_target",
        scale=0.2,
        position=tuple(initial_pose.translation),
        wxyz=initial_wxyz,  # Use actual end-effector orientation
    )

    frame_task = solver.add_frame_task("ee_task", target_link_name)
    frame_task.priority = 0
    frame_task.weight = 1.0

    nullspace_task = solver.add_posture_task("nullspace_bias_task")
    nullspace_task.priority = 1
    nullspace_task.weight = 0.0
    nullspace_task.set_target_configuration(q_default)

    # GUI elements
    with server.gui.add_folder("IK Controls"):
        timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
        task_weight = server.gui.add_slider("Task Weight", min=0.1, max=100, initial_value=1.0, step=0.1)
        pos_gain_slider = server.gui.add_slider("Position Gain", min=0.1, max=200, initial_value=10.0, step=0.1)
        rot_gain_slider = server.gui.add_slider("Orientation Gain", min=0.1, max=200, initial_value=10.0, step=0.1)
        iterations_slider = server.gui.add_slider("IK Iterations", min=1, max=20, initial_value=1, step=1)
        ee_mode_dropdown = server.gui.add_dropdown(
            "EE Solve Mode",
            options=("SCALE", "SCALE_ELASTIC", "MIN_ERROR"),
            initial_value="SCALE_ELASTIC",
        )
        ee_fallback_checkbox = server.gui.add_checkbox(
            "Allow SCALE fallback to MIN_ERROR",
            initial_value=False,
        )
        damping_slider = server.gui.add_slider("Solver Damping", min=0.01, max=1.0, initial_value=0.1, step=0.01)

        # Target control buttons
        snap_target_button = server.gui.add_button("Snap Target to Current EE")
        reset_arm_button = server.gui.add_button("Reset Arm & Target")

    # Joint limit scaling for interactive saturation testing
    with server.gui.add_folder("Joint Limit Scaling"):
        limit_scale_slider = server.gui.add_slider(
            "Limit Scale",
            min=0.1, max=1.0, initial_value=1.0, step=0.01,
        )
        limit_scale_info = server.gui.add_text(
            "Info", initial_value="1.0 = original limits, smaller = narrower",
        )
        apply_limit_scale_button = server.gui.add_button("Apply Limit Scale")
        reset_limits_button = server.gui.add_button("Reset to Original Limits")

    def _apply_limit_scale(scale: float):
        """Narrow joint limits symmetrically around the center of each range."""
        nonlocal q_lower, q_upper, q_current
        center = 0.5 * (q_lower_orig + q_upper_orig)
        half_range = 0.5 * (q_upper_orig - q_lower_orig) * scale
        q_lower = center - half_range
        q_upper = center + half_range
        robot.set_joint_limits(q_lower, q_upper)
        # Clip current configuration to stay inside new limits
        q_current = np.clip(q_current, q_lower, q_upper)
        robot.update_configuration(q_current)
        update_visualization(q_current)
        # Update joint slider ranges
        for i, slider in joint_sliders.items():
            if i < len(q_lower):
                slider.min = float(q_lower[i])
                slider.max = float(q_upper[i])
                slider.value = float(q_current[i])
        limit_scale_info.value = f"Scale={scale:.2f}  range={float(np.mean(q_upper - q_lower)):.3f} rad"
        logger.info(f"Joint limits scaled to {scale:.0%} of original range")

    @apply_limit_scale_button.on_click
    def _(_):
        _apply_limit_scale(limit_scale_slider.value)

    @reset_limits_button.on_click
    def _(_):
        limit_scale_slider.value = 1.0
        _apply_limit_scale(1.0)

    # Metrics display
    with server.gui.add_folder("Diagnostics", expand_by_default=False):
        jl_dist_text = server.gui.add_text("JL Distance", initial_value="--")
        manip_text = server.gui.add_text("Manipulability", initial_value="--")
        combined_text = server.gui.add_text("Combined Metric", initial_value="--")

    # Get joint names from robot model (parsed from URDF) - used throughout
    joint_names = robot.get_joint_names()

    # Joint configuration display (arm joints only)
    joint_sliders = {}
    with server.gui.add_folder("🦾 Joint Configuration", expand_by_default=False):
        # Arm joints only (first 7)
        for i in range(min(7, len(q_current))):  # Only first 7 joints
            # Use joint name from URDF
            joint_name = joint_names[i] if i < len(joint_names) else f"joint{i+1}"
            display_name = f"{joint_name} (joint{i+1})"
            slider = server.gui.add_slider(
                display_name,
                min=float(q_lower[i]),
                max=float(q_upper[i]),
                step=0.01,
                initial_value=float(q_current[i])
            )
            joint_sliders[i] = slider

        # Manual control toggle
        manual_control = server.gui.add_checkbox("Enable Manual Control", initial_value=False)

    # Debug options
    with server.gui.add_folder("Debug Options"):
        enable_debug = server.gui.add_checkbox("Enable Debug Logging", initial_value=False)
        debug_rate = server.gui.add_slider("Debug Rate (Hz)", min=0.1, max=10.0, initial_value=2.0, step=0.1)

    # Nullspace control
    # Type hint for GUI handles (accessed through Pinocchio's viewer)
    nullspace_joint_checkboxes: dict[int, Any] = {}
    with server.gui.add_folder("Nullspace Control"):
        enable_nullspace = server.gui.add_checkbox("Enable Nullspace Bias", initial_value=False)
        nullspace_gain = server.gui.add_slider("Nullspace Gain", min=0.0, max=2.0, initial_value=1e-2, step=0.1)
        bias_to_initial = server.gui.add_button("Bias to Initial Config")
        bias_to_zero = server.gui.add_button("Bias to Zero Config")
        with server.gui.add_folder("Joint Selection"):
            for idx in range(robot.nq):
                joint_name = joint_names[idx] if idx < len(joint_names) else f"joint{idx + 1}"
                checkbox = server.gui.add_checkbox(
                    f"{joint_name} (joint{idx + 1})",
                    initial_value=True if idx < min(7, robot.nq) else False,
                )
                nullspace_joint_checkboxes[idx] = checkbox

    # Nullspace bias configuration (default to initial configuration)
    nullspace_bias = q_default.copy()

    @bias_to_initial.on_click
    def _(_):
        nonlocal nullspace_bias
        nullspace_bias = q_default.copy()
        logger.info("Nullspace bias set to initial configuration")

    @bias_to_zero.on_click
    def _(_):
        nonlocal nullspace_bias
        nullspace_bias = np.zeros(robot.nq)
        logger.info("Nullspace bias set to zero configuration")

    @damping_slider.on_update
    def _(_):
        solver.set_damping(damping_slider.value)
        if enable_debug.value:
            logger.info(f"Solver damping updated to: {damping_slider.value:.3f}")

    @snap_target_button.on_click
    def _(_):
        # Get current end-effector pose
        current_ee_pose = robot.get_frame_pose(target_link_name)

        # Update target position
        ik_target.position = tuple(current_ee_pose.translation)
        ik_target.wxyz = tuple(r2q(current_ee_pose.rotation))

        logger.info("Target snapped to current end-effector pose")

    @reset_arm_button.on_click
    def _(_):
        nonlocal q_current

        # Reset arm to default configuration
        q_current = q_default.copy()
        robot.update_configuration(q_current)

        # Update visualization
        update_visualization(q_current)

        # Update joint sliders
        for i, slider in joint_sliders.items():
            if i < len(q_current):
                slider.value = float(q_current[i])

        # Get new end-effector pose after reset
        new_ee_pose = robot.get_frame_pose(target_link_name)

        # Update target to match new pose
        ik_target.position = tuple(new_ee_pose.translation)
        ik_target.wxyz = tuple(r2q(new_ee_pose.rotation))

        logger.info("Arm reset to default configuration and target updated")

    logger.info("="*60)
    logger.info(f"embodiK Basic IK Example - {config['display_name']}")
    logger.info(f"Robot: {robot.nq} DOF (joints)")
    logger.info(f"Target Link: {target_link_name}")
    logger.info("Debug logging disabled by default - enable in Debug Options (rate-based logging)")
    logger.info("="*60)

    last_debug_time = 0.0
    step_opts = embodik.PositionStepOptions()

    while True:
        # Check if manual control is enabled
        if manual_control.value:
            # Read joint values from sliders
            for i, slider in joint_sliders.items():
                if i < len(q_current):
                    q_current[i] = slider.value

            # Update robot configuration
            robot.update_configuration(q_current)

            # Update visualization
            update_visualization(q_current)

            frame_task.weight = 0.0
            nullspace_task.weight = 0.0
            nullspace_task.set_controlled_joint_indices([])

            # Update target to current EE position
            current_ee_pose = robot.get_frame_pose(target_link_name)
            ik_target.position = tuple(current_ee_pose.translation)
            ik_target.wxyz = tuple(r2q(current_ee_pose.rotation))

            # Small delay and continue
            time.sleep(solver.dt)
            continue

        # Build target pose from interactive control
        target_position = np.array(ik_target.position)
        target_wxyz = np.array(ik_target.wxyz)
        target_rotation = q2r(target_wxyz)
        target_pose = Rt(R=target_rotation, t=target_position)

        # Check if it's time to log debug info
        current_time = time.time()
        debug_interval = 1.0 / debug_rate.value
        should_log_debug = enable_debug.value and (current_time - last_debug_time) >= debug_interval

        start_time = time.time()

        frame_task.weight = task_weight.value
        frame_task.solve_mode = getattr(
            embodik.TaskSolveMode, ee_mode_dropdown.value, embodik.TaskSolveMode.SCALE
        )
        frame_task.allow_min_error_fallback = bool(ee_fallback_checkbox.value)

        # Update nullspace task
        if enable_nullspace.value:
            selected_joint_indices = [
                idx for idx, checkbox in nullspace_joint_checkboxes.items() if checkbox.value
            ]
            if selected_joint_indices:
                nullspace_task.set_controlled_joint_indices(selected_joint_indices)
                nullspace_task.set_target_configuration(nullspace_bias)
                nullspace_task.weight = nullspace_gain.value
            else:
                nullspace_task.weight = 0.0
                nullspace_task.set_controlled_joint_indices([])
        else:
            nullspace_task.weight = 0.0
            nullspace_task.set_controlled_joint_indices([])

        step_opts.position_gain = pos_gain_slider.value
        step_opts.orientation_gain = rot_gain_slider.value
        step_opts.max_steps = int(iterations_slider.value)
        result = solver.solve_position_step(q_current, target_pose, "ee_task", step_opts)

        # Log solver results (debug)
        if should_log_debug:
            logger.info(f"Solver status: {result.status}")
            logger.info(f"Solver elapsed time: {result.computation_time_ms:.2f} ms")
            if result.status == embodik.SolverStatus.SUCCESS:
                logger.info(f"Joint velocities norm: {np.linalg.norm(result.joint_velocities):.4f}")
                logger.info(f"Joint velocities: [{', '.join(f'{v:.4f}' for v in result.joint_velocities)}]")
                if result.task_scales:
                    logger.info(f"Task scales: [{', '.join(f'{s:.4f}' for s in result.task_scales)}]")
                # Check for limit application
                if hasattr(result, 'limits_applied'):
                    logger.info(f"Limits applied: {result.limits_applied}")

                # Check for joints near position limits (exclude gripper joints)
                joints_near_limits = []

                # For Panda, only check arm joints (0-6), not gripper joints (7-8)
                num_arm_joints = 7 if robot.nq > 7 else robot.nq

                for i in range(num_arm_joints):
                    margin = 0.1  # Consider "near" if within 0.1 rad of limit
                    if q_current[i] <= q_lower[i] + margin or q_current[i] >= q_upper[i] - margin:
                        joints_near_limits.append(i)

                if joints_near_limits:
                    logger.info(f"Arm joints near position limits: {joints_near_limits}")
                    for joint_idx in joints_near_limits:
                        logger.info(f"  Joint {joint_idx}: q={q_current[joint_idx]:.3f} "
                                  f"(position limits: [{q_lower[joint_idx]:.3f}, {q_upper[joint_idx]:.3f}])")

                if hasattr(result, 'task_errors') and result.task_errors:
                    logger.info(f"Task errors: [{', '.join(f'{e:.4f}' for e in result.task_errors)}]")

        if result.status in (
            embodik.SolverStatus.SUCCESS,
            embodik.SolverStatus.INFEASIBLE,
            embodik.SolverStatus.NUMERICAL_ERROR,
        ):
            q_current = np.clip(result.q_solution.copy(), q_lower, q_upper)
            robot.update_configuration(q_current)

            # Update visualization
            update_visualization(q_current)

            # Update joint sliders
            for i, slider in joint_sliders.items():
                if i < len(q_current):
                    slider.value = float(q_current[i])

            # Update diagnostics (throttled to ~10 Hz)
            if should_log_debug or (current_time - last_debug_time) >= 0.1:
                _, jl_agg = embodik.joint_limit_distance(q_current, q_lower, q_upper)
                J = robot.get_frame_jacobian(target_link_name)
                manip = embodik.velocity_manipulability(J)
                combined = embodik.singularity_joint_limit_metric(q_current, J, q_lower, q_upper)
                jl_dist_text.value = f"{jl_agg:.4f}"
                manip_text.value = f"{manip:.6f}"
                combined_text.value = f"{combined:.6f}"
        else:
            if enable_debug.value or result.status == embodik.SolverStatus.NUMERICAL_ERROR:
                logger.warning(f"Solver failed with status: {result.status}")
                if result.status == embodik.SolverStatus.NUMERICAL_ERROR:
                    logger.warning(f"  Position error: {result.position_error:.4f} m")
                    logger.warning(f"  Orientation error: {result.orientation_error:.4f} rad")
                    logger.warning(f"  Task weight: {task_weight.value}")

                    try:
                        J = robot.get_frame_jacobian(target_link_name)
                        J_rank = np.linalg.matrix_rank(J)
                        logger.warning(f"  Jacobian rank: {J_rank} (full rank = 6)")
                        if J_rank < 6:
                            logger.warning("  Robot may be near a singularity!")
                    except Exception:
                        pass

        # Update timing
        elapsed_time = (time.time() - start_time) * 1000
        timing_handle.value = 0.9 * timing_handle.value + 0.1 * elapsed_time

        # Update last debug time if we logged this iteration
        if should_log_debug:
            last_debug_time = current_time

        # Small delay
        time.sleep(1e-3)


if __name__ == "__main__":
    main(parse_args())
