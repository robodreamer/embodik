#!/usr/bin/env python3
"""Headless solver fixture for the public ROBOTIS AI Worker example."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

import embodik

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from example_helpers.ai_worker_model_utils import (
        default_ai_worker_ik_joint_names,
        resolve_ai_worker_frames,
    )
    from example_helpers.public_ai_worker_paths import resolve_public_ai_worker_urdf_paths
    from example_helpers.ai_worker_constraint_teleop_app import (
        DEFAULT_ARM_NULLSPACE_WEIGHT,
        DEFAULT_POSTURE_WEIGHT,
        DEFAULT_SOLVER_DT,
        DEFAULT_WORKER_SEED,
        WORKER_SUPPORT_CONTACT_FRAMES,
        _apply_named_joint_seed,
        _apply_soft_lift_margin,
        _compute_support_polygon_from_contacts,
        _configure_collision_constraint,
        _generate_consecutive_collision_exclusions,
        _generate_worker_collision_include_pairs,
        _shrink_polygon_2d,
    )
except ModuleNotFoundError as exc:
    if exc.name != "example_helpers" and not str(exc.name).startswith("example_helpers."):
        raise
    from examples.example_helpers.ai_worker_model_utils import (
        default_ai_worker_ik_joint_names,
        resolve_ai_worker_frames,
    )
    from examples.example_helpers.public_ai_worker_paths import resolve_public_ai_worker_urdf_paths
    from examples.example_helpers.ai_worker_constraint_teleop_app import (
        DEFAULT_ARM_NULLSPACE_WEIGHT,
        DEFAULT_POSTURE_WEIGHT,
        DEFAULT_SOLVER_DT,
        DEFAULT_WORKER_SEED,
        WORKER_SUPPORT_CONTACT_FRAMES,
        _apply_named_joint_seed,
        _apply_soft_lift_margin,
        _compute_support_polygon_from_contacts,
        _configure_collision_constraint,
        _generate_consecutive_collision_exclusions,
        _generate_worker_collision_include_pairs,
        _shrink_polygon_2d,
    )


@dataclass
class SolverFixture:
    """Bundle of objects a caller needs to drive a headless worker solve loop."""

    variant: str
    urdf_path: Path
    collision_urdf_path: Path
    robot: object
    solver: object
    right_task: object
    left_task: object
    posture_task: object
    arm_nullspace_task: object
    frame_map: dict[str, str]
    q0: np.ndarray
    q_lo: np.ndarray
    q_hi: np.ndarray
    locked_velocity_indices: list[int]
    left_arm_velocity_indices: list[int]
    right_arm_velocity_indices: list[int]
    lift_velocity_indices: list[int]
    arm_controlled_indices: list[int]
    support_polygon: np.ndarray
    collision_include_pairs: list[tuple[str, str]]
    collision_exclude_pairs: list[tuple[str, str]]
    collision_available: bool
    joint_name_to_cfg: dict[str, int] = field(default_factory=dict)


def _collect_joint_indices(robot, joint_names: Iterable[str], ik_joint_names: Iterable[str]):
    allowed = set(ik_joint_names)
    locked: list[int] = []
    left_arm: list[int] = []
    right_arm: list[int] = []
    lift: list[int] = []
    arm_controlled: list[int] = []
    for joint_name in joint_names:
        if not hasattr(robot, "get_joint_velocity_index"):
            continue
        idx_v = int(robot.get_joint_velocity_index(joint_name))
        if hasattr(robot, "get_joint_velocity_size"):
            nv_joint = int(robot.get_joint_velocity_size(joint_name))
        else:
            nv_joint = 1
        target = None
        if joint_name.startswith(("arm_l_", "gripper_l_")):
            target = left_arm
        elif joint_name.startswith(("arm_r_", "gripper_r_")):
            target = right_arm
        elif joint_name.startswith("lift_"):
            target = lift
        for offset in range(max(nv_joint, 1)):
            expanded_idx = idx_v + offset
            if target is not None:
                target.append(expanded_idx)
            if joint_name.startswith("arm_"):
                arm_controlled.append(expanded_idx)
            if joint_name in allowed:
                continue
            locked.append(expanded_idx)
    return (
        sorted(set(locked)),
        sorted(set(left_arm)),
        sorted(set(right_arm)),
        sorted(set(lift)),
        sorted(set(arm_controlled)),
    )


def build_worker_solver_fixture(
    variant: str,
    *,
    collision_enabled: bool = True,
    collision_min_distance_m: float = 0.035,
    max_collision_constraints: int = 2,
    collision_tuning_mode: str = "balanced",
    configure_com: bool = False,
    com_margin_frac: float = 0.10,
    com_vel_max: float = 0.4,
    com_acc_max: float = 0.1,
    com_use_acc_limits: bool = False,
    com_proximity_fraction: float = 0.05,
) -> SolverFixture:
    """Build a headless solver fixture identical to the GUI example."""
    urdf_path, collision_urdf_path_or_none = resolve_public_ai_worker_urdf_paths(variant=variant)
    collision_urdf_path = collision_urdf_path_or_none or urdf_path

    full_robot = embodik.RobotModel(str(collision_urdf_path), floating_base=False)
    ik_joint_names = default_ai_worker_ik_joint_names(full_robot.get_joint_names())
    robot = embodik.RobotModel(
        str(collision_urdf_path),
        actuated_joint_names=ik_joint_names,
        floating_base=False,
    )

    q_lo, q_hi = robot.get_joint_limits()
    q = robot.neutral_configuration()
    joint_names = list(robot.get_joint_names())
    joint_name_to_cfg: dict[str, int] = {}
    if hasattr(robot, "get_joint_config_index"):
        for name in joint_names:
            try:
                joint_name_to_cfg[name] = int(robot.get_joint_config_index(name))
            except Exception:
                pass
    posture_controlled_indices = [
        idx
        for name in ("lift_joint",)
        if (idx := joint_name_to_cfg.get(name)) is not None
    ]
    q = _apply_named_joint_seed(q, joint_name_to_cfg, q_lo, q_hi, DEFAULT_WORKER_SEED)
    q = _apply_soft_lift_margin(q, joint_name_to_cfg=joint_name_to_cfg, q_lo=q_lo, q_hi=q_hi)
    nullspace_bias_q = np.asarray(q, dtype=float).copy()
    robot.update_configuration(q)
    support_polygon = _compute_support_polygon_from_contacts(robot, WORKER_SUPPORT_CONTACT_FRAMES)

    frame_map = resolve_ai_worker_frames(robot.get_frame_names())

    solver = embodik.KinematicsSolver(robot)
    solver.dt = DEFAULT_SOLVER_DT
    solver.set_damping(0.1)
    solver.set_tolerance(0.1)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)

    right_task = solver.add_frame_task("right_tool_pose", frame_map["right_tool"], embodik.TaskType.FRAME_POSE)
    left_task = solver.add_frame_task("left_tool_pose", frame_map["left_tool"], embodik.TaskType.FRAME_POSE)
    right_task.priority = 0
    left_task.priority = 0
    right_task.weight = 1.0
    left_task.weight = 1.0

    posture_task = solver.add_posture_task("worker_posture")
    posture_task.priority = 1
    posture_task.weight = DEFAULT_POSTURE_WEIGHT
    posture_task.set_target_configuration(np.asarray(q, dtype=float).copy())
    if hasattr(posture_task, "set_controlled_joint_indices"):
        posture_task.set_controlled_joint_indices(list(posture_controlled_indices))

    arm_nullspace_task = solver.add_posture_task("arm_nullspace")
    arm_nullspace_task.priority = 1
    arm_nullspace_task.weight = 0.0
    arm_nullspace_task.set_target_configuration(nullspace_bias_q)

    (
        locked_velocity_indices,
        left_arm_velocity_indices,
        right_arm_velocity_indices,
        lift_velocity_indices,
        arm_controlled_indices,
    ) = _collect_joint_indices(robot, joint_names, ik_joint_names)
    if hasattr(arm_nullspace_task, "set_controlled_joint_indices"):
        arm_nullspace_task.set_controlled_joint_indices(list(arm_controlled_indices))
    arm_nullspace_task.weight = float(DEFAULT_ARM_NULLSPACE_WEIGHT)

    collision_available = False
    if hasattr(robot, "has_collision_geometry"):
        try:
            collision_available = bool(robot.has_collision_geometry())
        except Exception:
            collision_available = False

    collision_exclude_pairs = _generate_consecutive_collision_exclusions(robot, urdf_path)
    collision_include_pairs = _generate_worker_collision_include_pairs(
        robot, urdf_path, collision_exclude_pairs
    )

    if collision_enabled and collision_available:
        _configure_collision_constraint(
            solver,
            enabled=True,
            min_distance_m=float(collision_min_distance_m),
            max_constraints=int(max_collision_constraints),
            tuning_mode=str(collision_tuning_mode),
            include_pairs=collision_include_pairs,
            exclude_pairs=collision_exclude_pairs,
        )

    if configure_com and hasattr(solver, "configure_com_constraint"):
        try:
            solver.configure_com_constraint(
                support_polygon=support_polygon,
                margin=float(com_margin_frac),
                frame_name="base_link",
                com_vel_max=float(com_vel_max),
                com_acc_max=float(com_acc_max),
                use_acceleration_limits=bool(com_use_acc_limits),
                proximity_fraction=float(com_proximity_fraction),
            )
        except Exception:
            if hasattr(solver, "clear_com_constraint"):
                try:
                    solver.clear_com_constraint()
                except Exception:
                    pass

    return SolverFixture(
        variant=variant,
        urdf_path=urdf_path,
        collision_urdf_path=collision_urdf_path,
        robot=robot,
        solver=solver,
        right_task=right_task,
        left_task=left_task,
        posture_task=posture_task,
        arm_nullspace_task=arm_nullspace_task,
        frame_map=frame_map,
        q0=np.asarray(q, dtype=float).copy(),
        q_lo=np.asarray(q_lo, dtype=float).copy(),
        q_hi=np.asarray(q_hi, dtype=float).copy(),
        locked_velocity_indices=list(locked_velocity_indices),
        left_arm_velocity_indices=list(left_arm_velocity_indices),
        right_arm_velocity_indices=list(right_arm_velocity_indices),
        lift_velocity_indices=list(lift_velocity_indices),
        arm_controlled_indices=list(arm_controlled_indices),
        support_polygon=np.asarray(support_polygon, dtype=float).copy(),
        collision_include_pairs=list(collision_include_pairs),
        collision_exclude_pairs=list(collision_exclude_pairs),
        collision_available=collision_available,
        joint_name_to_cfg=joint_name_to_cfg,
    )


def compute_inner_polygon(support_polygon: np.ndarray, margin_frac: float) -> np.ndarray:
    """Shrink the support polygon by ``margin_frac * char_size`` toward centroid."""
    return _shrink_polygon_2d(np.asarray(support_polygon, dtype=float), float(margin_frac))
