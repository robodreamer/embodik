"""IK backend used by the teleop example."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pinocchio as pin

import embodik
from example_helpers.ik_common import (
    DEFAULT_ADAPTIVE_DT,
    DEFAULT_ADAPTIVE_DT_MAX_SCALE,
    DEFAULT_ADAPTIVE_DT_REFERENCE_DISTANCE,
    DEFAULT_COLLISION_TUNING_MODE,
    DEFAULT_NULLSPACE_GAIN,
    DEFAULT_POS_GAIN,
    DEFAULT_ROT_GAIN,
    DEFAULT_SOLVER_DT,
    apply_collision_tuning_mode,
)
from utils.robot_models import ensure_ros_package_path

DEFAULT_COLLISION_MIN_DISTANCE = 0.05


@dataclass
class IKResult:
    joints: np.ndarray
    status: str
    position_error: float
    rotation_error: float
    elapsed_ms: float
    collision_time_ms: float = 0.0
    collision_sphere_culled: int = 0
    collision_exact_queries: int = 0


class TeleopIKBackend:
    """Fixed-base stepping IK backend for teleop."""

    def __init__(
        self,
        cfg: Any,
        enable_collision: bool = True,
        nullspace_joint_weights: Optional[np.ndarray] = None,
    ):
        self.cfg = cfg
        ensure_ros_package_path(cfg.urdf_path)
        self.robot = embodik.RobotModel(str(cfg.urdf_path), floating_base=False)
        self.solver = embodik.KinematicsSolver(self.robot)
        self.solver.dt = DEFAULT_SOLVER_DT
        self.solver.set_damping(0.1)
        self.solver.set_tolerance(0.1)
        self._collision_tuning_mode = DEFAULT_COLLISION_TUNING_MODE
        apply_collision_tuning_mode(self.solver, self._collision_tuning_mode)
        try:
            self.solver.enable_timing_breakdown(True)
        except Exception:
            pass

        self.arm_dofs = len(cfg.joint_names)
        self.full_dofs = self.robot.nq
        self._collision_exclusions = list(cfg.collision_exclusions)
        if enable_collision and self._collision_exclusions:
            try:
                self.robot.apply_collision_exclusions(self._collision_exclusions)
            except Exception as exc:
                print(f"Warning: failed to apply collision exclusions: {exc}")
        self._collision_enabled = False
        self.nullspace_joint_weights = (
            nullspace_joint_weights.copy()
            if nullspace_joint_weights is not None
            else None
        )

        self.default_arm = cfg.default_configuration.copy()
        if len(self.default_arm) < self.arm_dofs:
            padding = np.zeros(self.arm_dofs - len(self.default_arm), dtype=float)
            self.default_arm = np.concatenate([self.default_arm, padding])
        elif len(self.default_arm) > self.arm_dofs:
            self.default_arm = self.default_arm[:self.arm_dofs]

        self.default_full = np.zeros(self.full_dofs, dtype=float)
        self.default_full[:self.arm_dofs] = self.default_arm
        self.q = self.default_full.copy()
        self.robot.update_configuration(self.q)

        lower, upper = self.robot.get_joint_limits()
        self.lower = lower.astype(float)
        self.upper = upper.astype(float)
        self.initial_pose = self.get_pose()

        self.frame_task = self.solver.add_frame_task("ee_task", self.cfg.target_link)
        self.frame_task.priority = 0
        self.frame_task.weight = 1.0
        self.frame_task.solve_mode = embodik.TaskSolveMode.SCALE
        self.frame_task.allow_min_error_fallback = False

        self.nullspace_task = self.solver.add_posture_task("posture_task")
        self.nullspace_task.priority = 1
        self.nullspace_task.weight = 0.0
        self.nullspace_task.solve_mode = embodik.TaskSolveMode.MIN_ERROR
        self.nullspace_task.allow_min_error_fallback = False
        self.nullspace_task.set_target_configuration(self.q.copy())
        self.nullspace_task.set_controlled_joint_indices([])

        self._step_opts = embodik.PositionStepOptions()

        if enable_collision:
            self.enable_self_collision(True)

    def get_pose(self) -> pin.SE3:
        return self.robot.get_frame_pose(self.cfg.target_link)

    def get_q(self) -> np.ndarray:
        return self.q[:self.arm_dofs].copy()

    def set_q(self, q_arm: np.ndarray) -> None:
        self.q[:self.arm_dofs] = np.clip(q_arm, self.lower[:self.arm_dofs], self.upper[:self.arm_dofs])
        self.robot.update_configuration(self.q)

    def solve_step(
        self,
        target: pin.SE3,
        pos_gain: float = DEFAULT_POS_GAIN,
        rot_gain: float = DEFAULT_ROT_GAIN,
        nullspace_gain: float = DEFAULT_NULLSPACE_GAIN,
        ee_mode: str = "SCALE_ELASTIC",
        ee_fallback: bool = False,
        max_steps: int = 1,
        limit_change_from_seed: bool = False,
        adaptive_dt: bool = DEFAULT_ADAPTIVE_DT,
        adaptive_dt_max_scale: float = DEFAULT_ADAPTIVE_DT_MAX_SCALE,
        adaptive_dt_reference_distance: float = DEFAULT_ADAPTIVE_DT_REFERENCE_DISTANCE,
    ) -> IKResult:
        self.frame_task.solve_mode = getattr(
            embodik.TaskSolveMode, ee_mode, embodik.TaskSolveMode.SCALE_ELASTIC
        )
        self.frame_task.allow_min_error_fallback = bool(ee_fallback)

        if nullspace_gain > 0.0:
            self.nullspace_task.set_target_configuration(self.default_full.copy())
            self.nullspace_task.set_controlled_joint_indices(list(range(self.arm_dofs)))
            self.nullspace_task.weight = float(nullspace_gain)
            if self.nullspace_joint_weights is not None:
                self.nullspace_task.set_controlled_joint_weights(
                    np.asarray(self.nullspace_joint_weights, dtype=float)
                )
        else:
            self.nullspace_task.weight = 0.0
            self.nullspace_task.set_controlled_joint_indices([])

        self._step_opts.position_gain = float(pos_gain)
        self._step_opts.orientation_gain = float(rot_gain)
        self._step_opts.max_steps = int(max_steps)
        self._step_opts.limit_change_from_seed = bool(limit_change_from_seed)
        self._step_opts.adaptive_dt = bool(adaptive_dt)
        self._step_opts.adaptive_dt_max_scale = float(adaptive_dt_max_scale)
        self._step_opts.adaptive_dt_reference_distance = float(adaptive_dt_reference_distance)

        ik_start = time.perf_counter()
        result = self.solver.solve_position_step(
            self.q, target, "ee_task", self._step_opts
        )
        elapsed_ms = (time.perf_counter() - ik_start) * 1000.0

        if result.status in (
            embodik.SolverStatus.SUCCESS,
            embodik.SolverStatus.INFEASIBLE,
            embodik.SolverStatus.NUMERICAL_ERROR,
            embodik.SolverStatus.NO_PROGRESS,
        ):
            self.q = np.clip(np.array(result.q_solution), self.lower, self.upper)
            self.robot.update_configuration(self.q)

        return IKResult(
            joints=self.get_q(),
            status=result.status.name,
            position_error=float(result.position_error),
            rotation_error=float(result.orientation_error),
            elapsed_ms=elapsed_ms,
            collision_time_ms=float(getattr(result, "collision_constraint_time_ms", 0.0)),
            collision_sphere_culled=int(getattr(result, "collision_sphere_culled_pairs", 0)),
            collision_exact_queries=int(getattr(result, "collision_exact_distance_queries", 0)),
        )

    def reset(self) -> pin.SE3:
        self.q = self.default_full.copy()
        self.robot.update_configuration(self.q)
        return self.get_pose()

    def enable_self_collision(
        self, enable: bool, min_distance: float = DEFAULT_COLLISION_MIN_DISTANCE
    ) -> None:
        if enable and not self._collision_enabled:
            try:
                apply_collision_tuning_mode(
                    self.solver, getattr(self, "_collision_tuning_mode", DEFAULT_COLLISION_TUNING_MODE)
                )
                self.solver.configure_collision_constraint(
                    min_distance=float(min_distance),
                    include_pairs=[],
                    exclude_pairs=list(self._collision_exclusions),
                )
                self._collision_enabled = True
            except RuntimeError as exc:
                print(f"[embodiK] Collision configuration failed: {exc}")
                self._collision_enabled = False
        elif enable and self._collision_enabled:
            apply_collision_tuning_mode(
                self.solver, getattr(self, "_collision_tuning_mode", DEFAULT_COLLISION_TUNING_MODE)
            )
            if hasattr(self.solver, "set_collision_min_distance"):
                self.solver.set_collision_min_distance(float(min_distance))
        elif not enable and self._collision_enabled:
            self.solver.clear_collision_constraint()
            self._collision_enabled = False

    def set_collision_tuning_mode(self, mode_label: str) -> None:
        self._collision_tuning_mode = mode_label.lower()
        apply_collision_tuning_mode(self.solver, self._collision_tuning_mode)
