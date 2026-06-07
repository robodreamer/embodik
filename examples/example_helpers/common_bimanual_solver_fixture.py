#!/usr/bin/env python3
"""Headless solver fixture for the common bimanual example."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

import embodik

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from example_helpers.common_bimanual_model_utils import (
        default_common_bimanual_ik_joint_names,
        resolve_common_bimanual_frames,
    )
    from example_helpers.common_bimanual_teleop_app import (
        COMMON_BIMANUAL_SUPPORT_CONTACT_FRAMES,
        DEFAULT_ARM_NULLSPACE_WEIGHT,
        DEFAULT_POSTURE_WEIGHT,
        DEFAULT_SOLVER_DT,
        _apply_named_joint_seed,
        _apply_soft_lift_margin,
        _build_link_adjacency_graph,
        _collision_object_to_link_name,
        _compute_support_polygon_from_contacts,
        _configure_collision_constraint,
        _default_seed_for_joint_names,
        _generate_common_bimanual_collision_include_pairs,
        _generate_consecutive_collision_exclusions,
        _is_arm_joint,
        _is_left_arm_joint,
        _is_right_arm_joint,
        _posture_control_joint_names,
        _shrink_polygon_2d,
    )
    from example_helpers.ik_common import configure_solver_runtime_policy
    from example_helpers.public_ai_worker_paths import resolve_public_ai_worker_urdf_paths
except ModuleNotFoundError as exc:
    if exc.name != "example_helpers" and not str(exc.name).startswith("example_helpers."):
        raise
    from examples.example_helpers.common_bimanual_model_utils import (
        default_common_bimanual_ik_joint_names,
        resolve_common_bimanual_frames,
    )
    from examples.example_helpers.common_bimanual_teleop_app import (
        COMMON_BIMANUAL_SUPPORT_CONTACT_FRAMES,
        DEFAULT_ARM_NULLSPACE_WEIGHT,
        DEFAULT_POSTURE_WEIGHT,
        DEFAULT_SOLVER_DT,
        _apply_named_joint_seed,
        _apply_soft_lift_margin,
        _build_link_adjacency_graph,
        _collision_object_to_link_name,
        _compute_support_polygon_from_contacts,
        _configure_collision_constraint,
        _default_seed_for_joint_names,
        _generate_common_bimanual_collision_include_pairs,
        _generate_consecutive_collision_exclusions,
        _is_arm_joint,
        _is_left_arm_joint,
        _is_right_arm_joint,
        _posture_control_joint_names,
        _shrink_polygon_2d,
    )
    from examples.example_helpers.ik_common import configure_solver_runtime_policy
    from examples.example_helpers.public_ai_worker_paths import resolve_public_ai_worker_urdf_paths


@dataclass
class SolverFixture:
    """Bundle of objects a caller needs to drive a headless bimanual solve loop."""

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


def _collect_joint_indices(
    robot,
    joint_names: Iterable[str],
    ik_joint_names: Iterable[str],
    *,
    lock_joint_names: Iterable[str] | None = None,
):
    allowed = set(ik_joint_names)
    lock_names = set(lock_joint_names or ())
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
        if _is_left_arm_joint(joint_name):
            target = left_arm
        elif _is_right_arm_joint(joint_name):
            target = right_arm
        elif joint_name.startswith("lift_") or joint_name in lock_names:
            target = lift
        for offset in range(max(nv_joint, 1)):
            expanded_idx = idx_v + offset
            if target is not None:
                target.append(expanded_idx)
            if _is_arm_joint(joint_name):
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


def _link_names_from_urdf(urdf_path: Path) -> set[str]:
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return set(_build_link_adjacency_graph(urdf_path).keys())
    return {str(link.get("name", "")).strip() for link in root.findall("link") if link.get("name")}


def _map_collision_link_pairs_to_geometry_pairs(
    robot,
    urdf_path: Path,
    link_pairs: Iterable[tuple[str, str]],
    exclude_pairs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Map explicit link-pair whitelist entries onto available collision geometry pairs."""
    if not hasattr(robot, "get_collision_pair_names"):
        return []
    try:
        pair_names = list(robot.get_collision_pair_names())
    except Exception:
        return []
    curated = {tuple(sorted((str(a), str(b)))) for a, b in link_pairs}
    if not curated:
        return []
    exclude_set = {tuple(p) for p in exclude_pairs}
    link_names = _link_names_from_urdf(urdf_path)
    include_pairs: list[tuple[str, str]] = []
    for a, b in pair_names:
        pair = (str(a), str(b))
        if pair in exclude_set or (pair[1], pair[0]) in exclude_set:
            continue
        link_a = _collision_object_to_link_name(pair[0], link_names)
        link_b = _collision_object_to_link_name(pair[1], link_names)
        if not link_a or not link_b:
            continue
        if tuple(sorted((link_a, link_b))) in curated:
            include_pairs.append(pair)
    return include_pairs


def _fallback_support_polygon(robot, frame_names: Iterable[str]) -> np.ndarray:
    names = set(frame_names)
    base_frame = next(
        (candidate for candidate in ("base_link", "base", "base_body") if candidate in names),
        None,
    )
    center = np.zeros(2, dtype=float)
    if base_frame is not None:
        try:
            center = np.asarray(robot.get_frame_pose(base_frame).translation, dtype=float)[:2]
        except Exception:
            center = np.zeros(2, dtype=float)
    half_x = 0.35
    half_y = 0.25
    return np.asarray(
        [
            [center[0] - half_x, center[1] - half_y],
            [center[0] + half_x, center[1] - half_y],
            [center[0] + half_x, center[1] + half_y],
            [center[0] - half_x, center[1] + half_y],
        ],
        dtype=float,
    )


def build_common_bimanual_solver_fixture(
    variant: str,
    *,
    urdf_path: str | Path | None = None,
    collision_urdf_path: str | Path | None = None,
    ik_joint_names: Iterable[str] | None = None,
    support_contact_frames: Iterable[str] | None = None,
    seed: dict[str, float] | None = None,
    posture_joint_names: Iterable[str] | None = None,
    lock_joint_names: Iterable[str] | None = None,
    collision_link_pairs: Iterable[tuple[str, str]] | None = None,
    com_frame_name: str = "base_link",
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
    if urdf_path is None:
        resolved_urdf, resolved_collision_urdf = resolve_public_ai_worker_urdf_paths(
            variant=variant
        )
        urdf_path = resolved_urdf
        collision_urdf_path = collision_urdf_path or resolved_collision_urdf or resolved_urdf
    else:
        urdf_path = Path(urdf_path).expanduser().resolve()
        collision_urdf_path = Path(collision_urdf_path or urdf_path).expanduser().resolve()
    urdf_path = Path(urdf_path)
    collision_urdf_path = Path(collision_urdf_path)

    full_robot = embodik.RobotModel(str(collision_urdf_path), floating_base=False)
    if ik_joint_names is None:
        ik_joint_names = default_common_bimanual_ik_joint_names(full_robot.get_joint_names())
    else:
        available_joint_names = set(full_robot.get_joint_names())
        ik_joint_names = [name for name in ik_joint_names if name in available_joint_names]
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
    posture_names = (
        list(posture_joint_names)
        if posture_joint_names is not None
        else _posture_control_joint_names(joint_names)
    )
    posture_controlled_indices = [
        idx for name in posture_names if (idx := joint_name_to_cfg.get(name)) is not None
    ]
    default_seed = dict(seed) if seed is not None else _default_seed_for_joint_names(joint_names)
    q = _apply_named_joint_seed(q, joint_name_to_cfg, q_lo, q_hi, default_seed)
    q = _apply_soft_lift_margin(q, joint_name_to_cfg=joint_name_to_cfg, q_lo=q_lo, q_hi=q_hi)
    nullspace_bias_q = np.asarray(q, dtype=float).copy()
    robot.update_configuration(q)
    contact_frames = (
        tuple(support_contact_frames)
        if support_contact_frames is not None
        else COMMON_BIMANUAL_SUPPORT_CONTACT_FRAMES
    )
    try:
        support_polygon = _compute_support_polygon_from_contacts(robot, contact_frames)
        if np.asarray(support_polygon).reshape(-1, 2).shape[0] < 3:
            raise RuntimeError("support polygon has fewer than three vertices")
    except Exception:
        support_polygon = _fallback_support_polygon(robot, robot.get_frame_names())

    frame_map = resolve_common_bimanual_frames(robot.get_frame_names())

    solver = embodik.KinematicsSolver(robot)
    solver.dt = DEFAULT_SOLVER_DT
    configure_solver_runtime_policy(solver)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    if hasattr(solver, "set_non_worsening_collision_floor_enabled"):
        solver.set_non_worsening_collision_floor_enabled(True)

    right_task = solver.add_frame_task(
        "right_tool_pose", frame_map["right_tool"], embodik.TaskType.FRAME_POSE
    )
    left_task = solver.add_frame_task(
        "left_tool_pose", frame_map["left_tool"], embodik.TaskType.FRAME_POSE
    )
    right_task.priority = 0
    left_task.priority = 0
    right_task.weight = 1.0
    left_task.weight = 1.0

    posture_task = solver.add_posture_task("bimanual_posture")
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
    ) = _collect_joint_indices(
        robot,
        joint_names,
        ik_joint_names,
        lock_joint_names=lock_joint_names,
    )
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
    if collision_link_pairs is None:
        collision_include_pairs = _generate_common_bimanual_collision_include_pairs(
            robot, urdf_path, collision_exclude_pairs
        )
    else:
        collision_include_pairs = _map_collision_link_pairs_to_geometry_pairs(
            robot, urdf_path, collision_link_pairs, collision_exclude_pairs
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
                frame_name=str(com_frame_name),
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
