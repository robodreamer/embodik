"""Helper class for dual-arm IK control functionality.

This module provides reusable utilities for managing dual-arm IK control,
interactive markers, and pose conversions shared across examples.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import viser
import pinocchio as pin

import embodik
from embodik.utils import (
    compute_pose_error,
    limit_task_velocity,
)
from embodik import r2q, q2r, Rt

__all__ = ["DualArmIkHelper", "transform_handle_to_pose", "pose_to_handle", "extract_pose_from_backend"]


def transform_handle_to_pose(handle: viser.TransformControlsHandle) -> pin.SE3:
    """Convert viser transform control handle to SE3 pose.

    Args:
        handle: Viser transform control handle

    Returns:
        SE3 pose object
    """
    wxyz = getattr(handle, "wxyz", None)
    position = np.array(getattr(handle, "position", (0.0, 0.0, 0.0)), dtype=float)
    if wxyz is None:
        return Rt(R=np.eye(3), t=position)
    try:
        rotation = q2r(np.array(wxyz))
    except ValueError:
        rotation = np.eye(3)
    return Rt(R=rotation, t=position)


def pose_to_handle(handle: viser.TransformControlsHandle, pose: pin.SE3) -> None:
    """Update viser transform control handle with SE3 pose.

    Args:
        handle: Viser transform control handle
        pose: SE3 pose object
    """
    handle.position = tuple(float(x) for x in pose.translation)
    handle.wxyz = tuple(r2q(pose.rotation))


def extract_pose_from_backend(pose_obj) -> pin.SE3:
    """Extract SE3 pose from either embodiK or Placo pose objects.

    embodiK returns an object with .rotation and .translation attributes.
    Placo returns an SE3 object with .R and .t attributes.

    Args:
        pose_obj: Pose object from embodiK or Placo backend

    Returns:
        SE3 pose object
    """
    if hasattr(pose_obj, 'rotation') and hasattr(pose_obj, 'translation'):
        # embodiK format
        return Rt(R=pose_obj.rotation, t=pose_obj.translation)
    elif hasattr(pose_obj, 'R') and hasattr(pose_obj, 't'):
        # SE3 format (Placo)
        return Rt(R=pose_obj.R, t=pose_obj.t)
    else:
        # Try to convert directly if it's already an SE3
        return pose_obj


class DualArmIkHelper:
    """Helper class for dual-arm IK control functionality.

    This class encapsulates common patterns for:
    - Interactive marker management
    - Snap handlers for targets
    - Target visibility updates
    - Dual-arm IK solving logic
    - Error display updates
    """

    def __init__(
        self,
        robot,
        solver,
        ik_arm_states: List[Dict[str, object]],
        left_handle: Optional[viser.TransformControlsHandle],
        right_handle: Optional[viser.TransformControlsHandle],
        left_task,
        right_task,
        nullspace_task,
        zero_velocity: np.ndarray,
        *,
        get_q_current: Callable[[], np.ndarray],
        update_visual_configuration: Callable[[], None],
        update_metric_display: Optional[Callable[[], None]] = None,
        use_placo: bool = False,
        placo_backend: Optional[object] = None,
        get_frame_pose: Optional[Callable[[str], object]] = None,
    ):
        """Initialize the dual-arm IK helper.

        Args:
            robot: Robot model instance (embodiK RobotModel)
            solver: IK solver instance (embodiK KinematicsSolver)
            ik_arm_states: List of arm state dictionaries with 'name', 'link', 'handle', 'task' keys
            left_handle: Left arm transform control handle (optional)
            right_handle: Right arm transform control handle (optional)
            left_task: Left arm frame task
            right_task: Right arm frame task
            nullspace_task: Nullspace posture task
            zero_velocity: Zero velocity vector (6D)
            get_q_current: Callable that returns current joint configuration
            update_visual_configuration: Callable to update robot visualization
            update_metric_display: Optional callable to update metric display
            use_placo: Whether using Placo backend
            placo_backend: Optional Placo backend instance
            get_frame_pose: Optional callable to get frame pose (for Placo compatibility)
        """
        self.robot = robot
        self.solver = solver
        self.ik_arm_states = ik_arm_states
        self.left_handle = left_handle
        self.right_handle = right_handle
        self.left_task = left_task
        self.right_task = right_task
        self.nullspace_task = nullspace_task
        self.zero_velocity = zero_velocity
        self.get_q_current = get_q_current
        self.update_visual_configuration = update_visual_configuration
        self.update_metric_display = update_metric_display
        self.use_placo = use_placo
        self.placo_backend = placo_backend
        self.get_frame_pose = get_frame_pose

    def snap_all_targets_to_current_pose(self) -> None:
        """Snap all interactive markers to current end-effector poses."""
        q_current = self.get_q_current()
        if self.use_placo and self.placo_backend is not None:
            self.placo_backend.set_q(q_current)
        else:
            self.robot.update_configuration(q_current)

        for arm_state in self.ik_arm_states:
            if self.use_placo and self.placo_backend is not None:
                pose_obj = self.placo_backend.get_frame_pose(arm_state["link"])
            else:
                pose_obj = self.robot.get_frame_pose(arm_state["link"])
            pose = extract_pose_from_backend(pose_obj)
            pose_to_handle(arm_state["handle"], pose)

    def snap_target_to_current_pose(self, arm_name: str) -> None:
        """Snap a specific arm's target marker to current end-effector pose.

        Args:
            arm_name: Name of the arm ('left' or 'right')
        """
        q_current = self.get_q_current()
        if self.use_placo and self.placo_backend is not None:
            self.placo_backend.set_q(q_current)
        else:
            self.robot.update_configuration(q_current)

        for arm_state in self.ik_arm_states:
            if arm_state["name"] == arm_name:
                if self.use_placo and self.placo_backend is not None:
                    pose_obj = self.placo_backend.get_frame_pose(arm_state["link"])
                else:
                    pose_obj = self.robot.get_frame_pose(arm_state["link"])
                pose = extract_pose_from_backend(pose_obj)
                pose_to_handle(arm_state["handle"], pose)
                break

    def update_target_visibility(self, visible: bool) -> None:
        """Update visibility of all target markers.

        Args:
            visible: Whether markers should be visible
        """
        for arm_state in self.ik_arm_states:
            arm_state["handle"].visible = visible
        if visible:
            self.snap_all_targets_to_current_pose()

    def solve_dual_arm_ik_step(
        self,
        ik_enabled: bool,
        enable_left: Optional[bool],
        enable_right: Optional[bool],
        pos_gain: float,
        rot_gain: float,
        max_linear_step: float,
        max_angular_step: float,
        nullspace_enabled: bool,
        nullspace_bias: np.ndarray,
        nullspace_gain: float,
        left_error_display: Optional[viser.GuiNumberHandle],
        right_error_display: Optional[viser.GuiNumberHandle],
        left_rot_error: Optional[viser.GuiNumberHandle],
        right_rot_error: Optional[viser.GuiNumberHandle],
        manual_mode: bool = False,
    ) -> Tuple[int, float]:
        """Solve one step of dual-arm IK.

        Args:
            ik_enabled: Whether IK is enabled
            enable_left: Whether left arm is enabled (None means always enabled)
            enable_right: Whether right arm is enabled (None means always enabled)
            pos_gain: Position gain for IK
            rot_gain: Rotation gain for IK
            max_linear_step: Maximum linear step size
            max_angular_step: Maximum angular step size
            nullspace_enabled: Whether nullspace bias is enabled
            nullspace_bias: Nullspace bias configuration
            nullspace_gain: Nullspace task weight
            left_error_display: Optional GUI handle for left position error display
            right_error_display: Optional GUI handle for right position error display
            left_rot_error: Optional GUI handle for left rotation error display
            right_rot_error: Optional GUI handle for right rotation error display
            manual_mode: Whether manual joint control mode is active

        Returns:
            Tuple of (active_tasks_count, placeholder_time_ms)
            Note: solver_time_ms is always 0.0 as the caller handles solve_velocity
        """
        # Reset error displays
        if left_error_display is not None:
            left_error_display.value = 0.0
        if right_error_display is not None:
            right_error_display.value = 0.0
        if left_rot_error is not None:
            left_rot_error.value = 0.0
        if right_rot_error is not None:
            right_rot_error.value = 0.0

        q_current = self.get_q_current()
        if self.use_placo and self.placo_backend is not None:
            self.placo_backend.set_q(q_current)
        else:
            self.robot.update_configuration(q_current)

        active_tasks = 0

        if not ik_enabled or manual_mode:
            # Disable all tasks
            if not self.use_placo:
                self.left_task.weight = 0.0
                self.left_task.set_target_velocity(self.zero_velocity)
                self.right_task.weight = 0.0
                self.right_task.set_target_velocity(self.zero_velocity)
                self.nullspace_task.weight = 0.0
            return (0, 0.0)

        # Handle left arm IK
        left_arm_state = next((s for s in self.ik_arm_states if s["name"] == "left"), None)
        if left_arm_state and (enable_left is None or enable_left):
            target_pose = transform_handle_to_pose(left_arm_state["handle"])

            if self.use_placo and self.placo_backend is not None:
                pose_obj = self.placo_backend.get_frame_pose(left_arm_state["link"])
            else:
                pose_obj = self.robot.get_frame_pose(left_arm_state["link"])
            current_pose = extract_pose_from_backend(pose_obj)

            error_vec = compute_pose_error(current_pose, target_pose)
            position_error = float(np.linalg.norm(error_vec[:3]))
            rotation_error = float(np.linalg.norm(error_vec[3:]))

            if left_error_display is not None:
                left_error_display.value = position_error * 1000.0
            if left_rot_error is not None:
                left_rot_error.value = np.degrees(rotation_error)

            if position_error > 1e-4 or rotation_error > 1e-3:
                if not self.use_placo:
                    desired_velocity = np.concatenate(
                        [
                            pos_gain * error_vec[:3],
                            rot_gain * error_vec[3:],
                        ]
                    )
                    limited_velocity = limit_task_velocity(
                        desired_velocity,
                        max_linear_step=max_linear_step,
                        max_angular_step=max_angular_step,
                    )
                    self.left_task.weight = 1.0
                    self.left_task.set_target_velocity(limited_velocity)
                    active_tasks += 1
                else:
                    # Placo will be handled separately
                    active_tasks += 1
            else:
                if not self.use_placo:
                    self.left_task.weight = 0.0
                    self.left_task.set_target_velocity(self.zero_velocity)
        elif left_arm_state:
            if not self.use_placo:
                self.left_task.weight = 0.0
                self.left_task.set_target_velocity(self.zero_velocity)

        # Handle right arm IK
        right_arm_state = next((s for s in self.ik_arm_states if s["name"] == "right"), None)
        if right_arm_state and (enable_right is None or enable_right):
            target_pose = transform_handle_to_pose(right_arm_state["handle"])

            if self.use_placo and self.placo_backend is not None:
                pose_obj = self.placo_backend.get_frame_pose(right_arm_state["link"])
            else:
                pose_obj = self.robot.get_frame_pose(right_arm_state["link"])
            current_pose = extract_pose_from_backend(pose_obj)

            error_vec = compute_pose_error(current_pose, target_pose)
            position_error = float(np.linalg.norm(error_vec[:3]))
            rotation_error = float(np.linalg.norm(error_vec[3:]))

            if right_error_display is not None:
                right_error_display.value = position_error * 1000.0
            if right_rot_error is not None:
                right_rot_error.value = np.degrees(rotation_error)

            if position_error > 1e-4 or rotation_error > 1e-3:
                if not self.use_placo:
                    desired_velocity = np.concatenate(
                        [
                            pos_gain * error_vec[:3],
                            rot_gain * error_vec[3:],
                        ]
                    )
                    limited_velocity = limit_task_velocity(
                        desired_velocity,
                        max_linear_step=max_linear_step,
                        max_angular_step=max_angular_step,
                    )
                    self.right_task.weight = 1.0
                    self.right_task.set_target_velocity(limited_velocity)
                    active_tasks += 1
                else:
                    # Placo will be handled separately
                    active_tasks += 1
            else:
                if not self.use_placo:
                    self.right_task.weight = 0.0
                    self.right_task.set_target_velocity(self.zero_velocity)
        elif right_arm_state:
            if not self.use_placo:
                self.right_task.weight = 0.0
                self.right_task.set_target_velocity(self.zero_velocity)

        # Handle nullspace
        if nullspace_enabled and active_tasks > 0:
            if not self.use_placo:
                self.nullspace_task.set_target_configuration(nullspace_bias)
                self.nullspace_task.weight = max(nullspace_gain, 0.0)
                if self.nullspace_task.weight > 1e-9:
                    active_tasks += 1
            else:
                active_tasks += 1
        else:
            if not self.use_placo:
                self.nullspace_task.weight = 0.0

        # Note: This helper sets task velocities but does NOT call solve_velocity
        # The caller should call solve_velocity after this method returns
        # This allows the caller to apply joint limit barriers and other post-processing

        # Update metric display if callback provided
        if self.update_metric_display is not None:
            self.update_metric_display()

        # Return active_tasks count (solver_time_ms is computed by caller)
        return (active_tasks, 0.0)

