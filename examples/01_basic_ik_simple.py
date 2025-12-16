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
from embodik.utils import (
    compute_pose_error,
    limit_task_velocity,
    apply_joint_limit_barrier_to_velocities,
)
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
        wxyz=initial_wxyz  # Use actual end-effector orientation
    )

    zero_velocity = np.zeros(6, dtype=float)
    frame_task = solver.add_frame_task("ee_task", target_link_name)
    frame_task.priority = 0
    frame_task.weight = 0.0
    frame_task.set_target_velocity(zero_velocity)

    nullspace_task = solver.add_posture_task("nullspace_bias_task")
    nullspace_task.priority = 1
    nullspace_task.weight = 0.0
    nullspace_task.set_target_configuration(q_default)

    # GUI elements
    with server.gui.add_folder("IK Controls"):
        timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
        pos_gain = server.gui.add_slider("Position Gain", min=10, max=100, initial_value=60, step=5)
        rot_gain = server.gui.add_slider("Rotation Gain", min=10, max=100, initial_value=60, step=5)
        damping_slider = server.gui.add_slider("Solver Damping", min=0.01, max=1.0, initial_value=0.1, step=0.01)

        # Step size limits
        max_linear_step = server.gui.add_slider("Max Linear Step (m/s)", min=0.1, max=1.0, initial_value=0.5, step=0.01)
        max_angular_step = server.gui.add_slider("Max Angular Step (rad/s)", min=0.1, max=1.0, initial_value=0.5, step=0.01)

        # Barrier function controls
        enable_barrier = server.gui.add_checkbox("Enable Barrier Function", initial_value=False)
        barrier_margin_slider = server.gui.add_slider("Barrier Margin", min=0.01, max=0.5, initial_value=0.15, step=0.01)
        barrier_gain_slider = server.gui.add_slider("Barrier Gain", min=0.1, max=5.0, initial_value=1.6, step=0.1)

        # Target control buttons
        snap_target_button = server.gui.add_button("Snap Target to Current EE")
        reset_arm_button = server.gui.add_button("Reset Arm & Target")

    # Get joint names from robot model (parsed from URDF) - used throughout
    joint_names = robot.get_joint_names()

    # Joint configuration display (arm joints only)
    joint_sliders = {}
    with server.gui.add_folder("🦾 Joint Configuration", expand_by_default=False):
        # Get joint limits
        q_lower, q_upper = robot.get_joint_limits()

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
        nullspace_joint_checkboxes: dict[int, Any] = {}
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

    # Initialize debug timing
    last_debug_time = 0.0

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

        # Get target pose as SE3
        target_position = np.array(ik_target.position)
        target_wxyz = np.array(ik_target.wxyz)
        target_rotation = q2r(target_wxyz)
        target_pose = Rt(R=target_rotation, t=target_position)

        # Get current end-effector pose (already a Pinocchio SE3)
        current_ee_pose = robot.get_frame_pose(target_link_name)
        current_pose = current_ee_pose  # Already SE3, no need to wrap

        # Compute pose error
        # position error = goal - current, rotation error = log(R_goal @ R_current^T)
        pose_error = compute_pose_error(current_pose, target_pose)
        frame_task.weight = 0.0

        # Calculate error magnitudes
        position_error = np.linalg.norm(pose_error[:3])
        rotation_error = np.linalg.norm(pose_error[3:])

        # Skip if already at target
        if np.linalg.norm(pose_error) < 0.0005:  # Within 0.5mm
            continue

        # Check if it's time to log debug info
        current_time = time.time()
        debug_interval = 1.0 / debug_rate.value  # Convert Hz to seconds
        should_log_debug = enable_debug.value and (current_time - last_debug_time) >= debug_interval

        if should_log_debug:
            logger.info(f"\n--- Debug Info (t={current_time:.2f}s) ---")
            logger.info(f"Target position: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
            logger.info(f"Current position: [{current_ee_pose.translation[0]:.3f}, {current_ee_pose.translation[1]:.3f}, {current_ee_pose.translation[2]:.3f}]")
            logger.info(f"Position error: {position_error:.4f} m")
            logger.info(f"Rotation error: {rotation_error:.4f} rad")

        # Setup velocity IK
        start_time = time.time()
        frame_task.weight = 0.0

        # Set target velocity as scaled error (velocity IK)
        # Apply gains directly to the pose error
        target_velocity = np.concatenate([
            pos_gain.value * pose_error[:3],  # Position error with gain
            rot_gain.value * pose_error[3:]   # Rotation error with gain
        ])

        # Limit the velocity to prevent large jumps
        target_velocity = limit_task_velocity(
            target_velocity,
            max_linear_step=max_linear_step.value,
            max_angular_step=max_angular_step.value,
            enable_debug=enable_debug.value and should_log_debug,
            debug_logger=logger
        )

        frame_task.weight = 1.0
        frame_task.set_target_velocity(target_velocity)

        if should_log_debug:
            logger.info(f"Target velocity: linear=[{target_velocity[0]:.3f}, {target_velocity[1]:.3f}, {target_velocity[2]:.3f}], "
                       f"angular=[{target_velocity[3]:.3f}, {target_velocity[4]:.3f}, {target_velocity[5]:.3f}]")

        # Add nullspace/posture task if enabled
        if enable_nullspace.value:
            selected_joint_indices = [
                idx for idx, checkbox in nullspace_joint_checkboxes.items() if checkbox.value
            ]
            if selected_joint_indices:
                nullspace_task.set_controlled_joint_indices(selected_joint_indices)
                nullspace_task.set_target_configuration(nullspace_bias)
                nullspace_task.weight = nullspace_gain.value

                if should_log_debug:
                    ns_error_selected = np.array(
                        [nullspace_bias[i] - q_current[i] for i in selected_joint_indices]
                    )
                    logger.info(
                        f"Nullspace task: weight={nullspace_gain.value:.2f}, "
                        f"joints={selected_joint_indices}, "
                        f"error_norm={np.linalg.norm(ns_error_selected):.4f}"
                    )
            else:
                nullspace_task.weight = 0.0
                nullspace_task.set_controlled_joint_indices([])
        else:
            nullspace_task.weight = 0.0
            nullspace_task.set_controlled_joint_indices([])

        # Solve for joint velocities
        result = solver.solve_velocity(q_current, apply_limits=True)

        # Log solver results
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
                q_lower, q_upper = robot.get_joint_limits()
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

        if result.status == embodik.SolverStatus.SUCCESS:
            # Get joint velocities
            joint_velocities = result.joint_velocities.copy()

            # Apply barrier function to velocities if enabled
            if enable_barrier.value:
                joint_velocities = apply_joint_limit_barrier_to_velocities(
                    q_current,
                    q_lower,
                    q_upper,
                    joint_velocities,
                    solver.dt,
                    barrier_margin=barrier_margin_slider.value,
                    barrier_gain=barrier_gain_slider.value,
                    num_arm_joints=7,
                    enable_debug=enable_debug.value and should_log_debug,
                    debug_logger=logger,
                )

            # Integrate velocities
            dq = joint_velocities * solver.dt
            q_current = q_current + dq

            # Ensure we stay within limits
            q_current = np.clip(q_current, q_lower, q_upper)

            # Update robot
            robot.update_configuration(q_current)

            # Update visualization
            update_visualization(q_current)

            # Update joint sliders
            for i, slider in joint_sliders.items():
                if i < len(q_current):
                    slider.value = float(q_current[i])
        else:
            if enable_debug.value or result.status == embodik.SolverStatus.NUMERICAL_ERROR:
                logger.warning(f"Solver failed with status: {result.status}")
                if result.status == embodik.SolverStatus.NUMERICAL_ERROR:
                    logger.warning("  NUMERICAL_ERROR can occur when:")
                    logger.warning("  1. Task contribution magnitude exceeds 1e10 (extreme velocities)")
                    logger.warning("  2. Maximum iterations (20) reached without convergence")
                    logger.warning("  3. Final solution violates constraints despite scaling")
                    logger.warning(f"  Current position error: {position_error:.4f} m")
                    logger.warning(f"  Current rotation error: {rotation_error:.4f} rad")
                    logger.warning(f"  Gains: pos={pos_gain.value}, rot={rot_gain.value}")

                    # Show requested velocities
                    if 'target_velocity' in locals():
                        logger.warning(f"  Requested velocity norm: {np.linalg.norm(target_velocity):.3f}")
                        logger.warning(f"  Linear velocity: {np.linalg.norm(target_velocity[:3]):.3f} m/s")
                        logger.warning(f"  Angular velocity: {np.linalg.norm(target_velocity[3:]):.3f} rad/s")

                    # Check if we might be in a singularity
                    if hasattr(robot, 'compute_jacobian'):
                        try:
                            J = robot.compute_jacobian(target_link_name)
                            J_rank = np.linalg.matrix_rank(J)
                            logger.warning(f"  Jacobian rank: {J_rank} (full rank = 6)")
                            if J_rank < 6:
                                logger.warning("  ⚠️  Robot may be near a singularity!")
                        except:
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
