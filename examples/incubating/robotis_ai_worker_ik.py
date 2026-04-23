#!/usr/bin/env python3
"""ROBOTIS AI worker dual-arm IK demo using local FFW URDF assets.

This example targets the locally available FFW SG2/BG2 URDFs and mirrors the
interactive dual-tool workflow used in the G1 notebooks, adapted to an
embodiK + Viser loop.
"""

from __future__ import annotations

import argparse
import collections
import sys
import tempfile
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

import embodik
from embodik.utils import q2r, r2q

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.incubating.example_helpers.robotis_ai_worker_utils import (
    default_worker_ik_joint_names,
    resolve_generated_ffw_collision_urdf_path,
    resolve_ai_worker_frames,
    resolve_ffw_urdf_path,
)
from examples.incubating.g1_port_phase1.g1_viser_utils import make_visual_config_mapper
from examples.incubating.g1_port_phase1.robust_ik_runtime import (
    clear_all_target_velocities_if_available,
    clip_configuration,
    configure_primary_solve_mode,
    robust_solve_position_step,
)

DEFAULT_SOLVER_DT = 0.01
DEFAULT_POS_GAIN = 10.0
DEFAULT_ROT_GAIN = 10.0
DEFAULT_POSTURE_WEIGHT = 1e-2
DEFAULT_ARM_NULLSPACE_WEIGHT = 1.0
COLLISION_TUNING_OPTIONS = ("speed", "balanced", "precise")
GEOMETRY_VIEW_OPTIONS = ("Visual", "Collision", "Both")
POSTURE_SLIDER_DEADBAND = 1e-3
EE_POSITION_DEADBAND = 1e-4
EE_ROTATION_DEADBAND = 1e-3
LIFT_LIMIT_MARGIN = 1e-3
DEFAULT_MAX_COLLISION_CONSTRAINTS = 3
WORKER_SUPPORT_CONTACT_FRAMES = (
    "left_wheel_drive_link",
    "right_wheel_drive_link",
    "rear_wheel_drive_link",
)
COLOR_OUTER_POLY = np.array([[[0.18, 0.80, 0.30], [0.18, 0.80, 0.30]]], dtype=float)
COLOR_INNER_POLY = np.array([[[1.00, 0.65, 0.12], [1.00, 0.65, 0.12]]], dtype=float)
COLOR_DROP_LINE = np.array([[[0.25, 0.55, 1.00], [0.25, 0.55, 1.00]]], dtype=float)
COLOR_CONTACT_POINT = (0.98, 0.90, 0.18)
COLOR_COM_INSIDE = (0.12, 0.78, 0.32)
COLOR_COM_NEAR = (1.00, 0.68, 0.10)
COLOR_COM_OUTSIDE = (0.92, 0.20, 0.24)
DEFAULT_WORKER_SEED = {
    "lift_joint": -0.1,
    "head_joint1": 0.0,
    "head_joint2": 0.0,
    "arm_l_joint1": 0.50,
    "arm_l_joint2": 0.35,
    "arm_l_joint3": 0.10,
    "arm_l_joint4": -1.90,
    "arm_l_joint5": 0.35,
    "arm_l_joint6": -0.15,
    "arm_l_joint7": 0.0,
    "arm_r_joint1": 0.50,
    "arm_r_joint2": -0.35,
    "arm_r_joint3": -0.10,
    "arm_r_joint4": -1.90,
    "arm_r_joint5": -0.35,
    "arm_r_joint6": -0.15,
    "arm_r_joint7": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("sg2", "bg2"), default="sg2")
    parser.add_argument("--port", type=int, default=8092)
    return parser.parse_args()


def _pose_from_ctrl(ctrl) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = np.asarray(ctrl.position, dtype=float)
    wxyz = np.asarray(ctrl.wxyz, dtype=float)
    n = float(np.linalg.norm(wxyz))
    if not np.isfinite(n) or n < 1e-12:
        wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        wxyz = wxyz / n
    pose[:3, :3] = q2r(np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=float), order="xyzs")
    return pose


def _ctrl_from_pose(ctrl, pose) -> None:
    pose = np.asarray(pose, dtype=float)
    q_xyzw = r2q(pose[:3, :3], order="xyzs")
    ctrl.position = tuple(np.asarray(pose[:3, 3], dtype=float))
    ctrl.wxyz = (float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))


def _rotation_error_rad(R_target: np.ndarray, R_current: np.ndarray) -> float:
    R_rel = np.asarray(R_target, dtype=float) @ np.asarray(R_current, dtype=float).T
    cos_theta = float((np.trace(R_rel) - 1.0) * 0.5)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.arccos(cos_theta))


def _convex_hull_2d(points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    if pts.shape[0] <= 1:
        return pts.copy()
    pts = np.unique(np.round(pts, decimals=9), axis=0)
    if pts.shape[0] <= 2:
        return pts.copy()
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull


def _polygon_edge_pts(poly_xy: np.ndarray, z: float) -> np.ndarray:
    poly = np.asarray(poly_xy, dtype=float).reshape(-1, 2)
    if poly.shape[0] < 2:
        return np.zeros((0, 2, 3), dtype=float)
    pts3 = np.column_stack([poly, np.full(poly.shape[0], float(z))])
    return np.array([[pts3[i], pts3[(i + 1) % pts3.shape[0]]] for i in range(pts3.shape[0])], dtype=float)


def _shrink_polygon_2d(poly: np.ndarray, margin_frac: float) -> np.ndarray:
    if margin_frac <= 0.0:
        return np.asarray(poly, dtype=float).copy()
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    centroid = poly.mean(axis=0)
    char_size = float(np.mean(np.linalg.norm(poly - centroid, axis=1)))
    shrink = float(np.clip(margin_frac, 0.0, 1.0)) * max(0.0, char_size - 1e-9)
    result = []
    for vertex in poly:
        delta = centroid - vertex
        norm = float(np.linalg.norm(delta))
        result.append(vertex if norm < 1e-12 else vertex + (shrink / norm) * delta)
    return np.asarray(result, dtype=float)


def _polygon_slack(poly_xy: np.ndarray, point_xy: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly_xy, dtype=float).reshape(-1, 2)
    point = np.asarray(point_xy, dtype=float).reshape(2)
    if poly.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    slacks = np.empty(poly.shape[0], dtype=float)
    for i in range(poly.shape[0]):
        v0 = poly[i]
        v1 = poly[(i + 1) % poly.shape[0]]
        edge = v1 - v0
        normal = np.array([edge[1], -edge[0]], dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm > 1e-12:
            normal /= norm
        slacks[i] = float(np.dot(normal, v0 - point))
    return slacks


def _com_color(min_slack: float, near_boundary_threshold: float = 0.02) -> tuple[float, float, float]:
    if min_slack < 0.0:
        return COLOR_COM_OUTSIDE
    if min_slack < float(near_boundary_threshold):
        return COLOR_COM_NEAR
    return COLOR_COM_INSIDE


def _compute_support_polygon_from_contacts(robot, frame_names: tuple[str, ...]) -> np.ndarray:
    points_xy: list[np.ndarray] = []
    for frame_name in frame_names:
        pose = robot.get_frame_pose(frame_name)
        translation = np.asarray(pose.translation, dtype=float)
        points_xy.append(translation[:2].copy())
    return _convex_hull_2d(np.asarray(points_xy, dtype=float))


def _support_contact_points(robot, frame_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    points: dict[str, np.ndarray] = {}
    for frame_name in frame_names:
        pose = robot.get_frame_pose(frame_name)
        points[frame_name] = np.asarray(pose.translation, dtype=float).copy()
    return points


def _apply_named_joint_seed(
    q_seed: np.ndarray,
    joint_name_to_cfg: dict[str, int],
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    joint_values: dict[str, float],
) -> np.ndarray:
    q_out = np.asarray(q_seed, dtype=float).copy()
    for joint_name, value in joint_values.items():
        idx = joint_name_to_cfg.get(joint_name)
        if idx is None or idx >= q_out.size:
            continue
        q_out[idx] = float(value)
    return np.clip(q_out, np.asarray(q_lo, dtype=float), np.asarray(q_hi, dtype=float))


def _apply_soft_lift_margin(
    q_in: np.ndarray,
    *,
    joint_name_to_cfg: dict[str, int],
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    margin: float = LIFT_LIMIT_MARGIN,
) -> np.ndarray:
    q_out = np.asarray(q_in, dtype=float).copy()
    lift_idx = joint_name_to_cfg.get("lift_joint")
    if lift_idx is None or lift_idx >= q_out.size or lift_idx >= q_lo.size or lift_idx >= q_hi.size:
        return q_out
    lo = float(q_lo[lift_idx])
    hi = float(q_hi[lift_idx])
    eff_lo = lo + float(margin)
    eff_hi = hi - float(margin)
    if eff_lo > eff_hi:
        eff_lo = lo
        eff_hi = hi
    q_out[lift_idx] = float(np.clip(q_out[lift_idx], eff_lo, eff_hi))
    return q_out


def _joint_velocity_norm(result: object) -> float:
    dq = getattr(result, "joint_velocities", None)
    if dq is None:
        return 0.0
    try:
        return float(np.linalg.norm(np.asarray(dq, dtype=float)))
    except Exception:
        return 0.0


def _is_collision_boundary_stall(
    *,
    result: object,
    collision_enabled: bool,
    current_collision_min: float | None,
    collision_min_distance_m: float,
    distance_slack_m: float = 5e-3,
    dq_stall_eps: float = 1e-8,
) -> bool:
    """Classify zero-motion near-margin exits as collision-limited holds.

    In the worker teleop flow, repeatedly commanding farther into the torso can
    leave the solver parked at the last safe configuration with statuses like
    ``INFEASIBLE`` or ``NUMERICAL_ERROR`` instead of ``COLLISION_VIOLATED``.
    Treat that plateau the same way as a safe collision hold so the UI snaps the
    target back to the achievable boundary rather than re-requesting the same
    impossible step every tick.
    """
    if not collision_enabled or current_collision_min is None:
        return False

    status_name = getattr(getattr(result, "status", None), "name", str(getattr(result, "status", "")))
    if status_name not in {"NO_PROGRESS", "INFEASIBLE", "NUMERICAL_ERROR"}:
        return False

    if float(current_collision_min) > float(collision_min_distance_m) + float(distance_slack_m):
        return False

    if _joint_velocity_norm(result) > float(dq_stall_eps):
        return False

    return True


def _attempt_deep_penetration_escape_burst(
    *,
    robot,
    solver,
    q_current: np.ndarray,
    targets: list[object],
    options,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    zero_velocity_indices: list[int],
    min_distance_m: float,
    max_constraints: int,
    tuning_mode: str,
    include_pairs: list[tuple[str, str]],
    exclude_pairs: list[tuple[str, str]],
    trial_steps: int = 3,
    improvement_epsilon: float = 1e-4,
) -> np.ndarray | None:
    """Try a short unconstrained pull-away burst when already in penetration.

    The worker can enter a deadlocked state when the current configuration is
    already in contact and the collision-constrained solve returns zero-motion
    ``INFEASIBLE`` repeatedly. In that case, briefly dropping the collision
    constraint can let the commanded pull-away motion unwind the configuration.
    Only accept the burst if the full collision debug distance improves.
    """
    if not hasattr(solver, "evaluate_collision_debug") or not hasattr(solver, "clear_collision_constraint"):
        return None

    try:
        dbg_before = solver.evaluate_collision_debug(np.asarray(q_current, dtype=float))
    except Exception:
        return None
    if dbg_before is None or not np.isfinite(dbg_before.distance):
        return None

    before_distance = float(dbg_before.distance)
    if before_distance > 0.0:
        return None

    q_seed = np.asarray(q_current, dtype=float).copy()
    q_trial = q_seed.copy()
    had_stall_recovery = bool(getattr(options, "stall_recovery", False))

    try:
        solver.clear_collision_constraint()
    except Exception:
        return None
    if hasattr(solver, "disable_stall_handler"):
        try:
            solver.disable_stall_handler()
        except Exception:
            pass

    if hasattr(options, "stall_recovery"):
        options.stall_recovery = False

    try:
        for _ in range(max(int(trial_steps), 1)):
            step = robust_solve_position_step(
                robot=robot,
                solver=solver,
                q_current=q_trial,
                targets=targets,
                options=options,
                q_lo=q_lo,
                q_hi=q_hi,
                zero_velocity_indices=zero_velocity_indices,
                fallback_status_names=("INVALID_INPUT", "NUMERICAL_ERROR"),
            )
            q_trial = np.asarray(step.q_next, dtype=float).copy()
            robot.update_configuration(q_trial)

        dbg_after = solver.evaluate_collision_debug(q_trial)
        if dbg_after is None or not np.isfinite(dbg_after.distance):
            return None
        after_distance = float(dbg_after.distance)
        if after_distance > before_distance + float(improvement_epsilon):
            return q_trial
        return None
    finally:
        if hasattr(options, "stall_recovery"):
            options.stall_recovery = had_stall_recovery
        _configure_collision_constraint(
            solver,
            enabled=True,
            min_distance_m=min_distance_m,
            max_constraints=max_constraints,
            tuning_mode=tuning_mode,
            include_pairs=include_pairs,
            exclude_pairs=exclude_pairs,
        )
        robot.update_configuration(q_seed)


def _prepare_viewer_urdf_path(urdf_path: Path) -> Path:
    """Rewrite broken absolute mesh URIs for Viser-only loading when needed."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    changed = False

    def _find_local_mesh(name: str) -> Path | None:
        candidates = [
            urdf_path.parent / "meshes" / name,
            urdf_path.parent / name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        try:
            return next(urdf_path.parent.rglob(name))
        except StopIteration:
            return None

    for geometry in root.findall(".//geometry"):
        mesh = geometry.find("mesh")
        if mesh is None:
            continue
        filename = str(mesh.get("filename", "")).strip()
        if not filename.startswith("file://"):
            continue
        mesh_path = Path(filename[len("file://") :])
        try:
            if mesh_path.is_file():
                mesh.set("filename", str(mesh_path))
                changed = True
                continue
        except (OSError, PermissionError):
            pass
        replacement = _find_local_mesh(mesh_path.name)
        if replacement is not None:
            mesh.set("filename", str(replacement))
        else:
            geometry.remove(mesh)
            ET.SubElement(geometry, "box", size="0.001 0.001 0.001")
        changed = True

    if not changed:
        return urdf_path

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{urdf_path.stem}_viser.urdf",
        prefix="embodik_",
        delete=False,
    )
    tree.write(tmp, encoding="unicode")
    tmp.close()
    return Path(tmp.name)


def _build_link_adjacency_graph(urdf_path: Path) -> dict[str, set[str]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    graph: dict[str, set[str]] = collections.defaultdict(set)
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_link = str(parent.get("link", "")).strip()
        child_link = str(child.get("link", "")).strip()
        if not parent_link or not child_link:
            continue
        graph[parent_link].add(child_link)
        graph[child_link].add(parent_link)
    return graph


def _collision_object_to_link_name(collision_name: str, link_names: set[str]) -> str | None:
    name = str(collision_name)
    if name in link_names:
        return name
    for suffix in ("_0", "_1", "_2", "_3"):
        if name.endswith(suffix):
            candidate = name[: -len(suffix)]
            if candidate in link_names:
                return candidate
    if name.rsplit("_", 1)[0] in link_names:
        return name.rsplit("_", 1)[0]
    return None


def _shortest_link_distance(
    graph: dict[str, set[str]], start: str, goal: str, max_hops: int
) -> int | None:
    if start == goal:
        return 0
    queue = collections.deque([(start, 0)])
    visited = {start}
    while queue:
        node, dist = queue.popleft()
        if dist >= max_hops:
            continue
        for nxt in graph.get(node, ()):
            if nxt == goal:
                return dist + 1
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, dist + 1))
    return None


def _worker_collision_group(name: str) -> str:
    lower = str(name).lower()
    if lower.startswith("head_") or lower.startswith("camera_"):
        return "head"
    if "_l_" in lower or lower.startswith("arm_l_") or lower.startswith("gripper_l_") or "camera_l_" in lower:
        return "left"
    if "_r_" in lower or lower.startswith("arm_r_") or lower.startswith("gripper_r_") or "camera_r_" in lower:
        return "right"
    return "core"


def _worker_manual_curated_link_pairs() -> list[tuple[str, str]]:
    """Explicit worker collision whitelist tuned for teleop responsiveness.

    This replaces the broader heuristic filter with a smaller, intentional set:
    - torso core vs proximal/mid arm links plus gripper base / fingertips
    - head vs distal forearm / gripper
    - cross-arm distal forearm / gripper
    """
    core_links = ("base_link", "lift_link", "arm_base_link")
    side_torso_template = (
        "arm_{side}_link2",
        "arm_{side}_link3",
        "arm_{side}_link4",
        "arm_{side}_link5",
        "arm_{side}_link6",
        "gripper_{side}_rh_p12_rn_base",
        "gripper_{side}_rh_p12_rn_l2",
        "gripper_{side}_rh_p12_rn_r2",
    )
    side_head_template = (
        "arm_{side}_link6",
        "arm_{side}_link7",
        "gripper_{side}_rh_p12_rn_base",
        "gripper_{side}_rh_p12_rn_l2",
        "gripper_{side}_rh_p12_rn_r2",
    )
    side_cross_template = (
        "arm_{side}_link5",
        "arm_{side}_link6",
        "arm_{side}_link7",
        "gripper_{side}_rh_p12_rn_base",
        "gripper_{side}_rh_p12_rn_l2",
        "gripper_{side}_rh_p12_rn_r2",
    )

    left_torso = tuple(name.format(side="l") for name in side_torso_template)
    right_torso = tuple(name.format(side="r") for name in side_torso_template)
    left_head = tuple(name.format(side="l") for name in side_head_template)
    right_head = tuple(name.format(side="r") for name in side_head_template)
    left_cross = tuple(name.format(side="l") for name in side_cross_template)
    right_cross = tuple(name.format(side="r") for name in side_cross_template)

    pairs: set[tuple[str, str]] = set()
    for core in core_links:
        for link in left_torso + right_torso:
            pairs.add(tuple(sorted((core, link))))
    for link in left_head + right_head:
        pairs.add(tuple(sorted(("head_link2", link))))
    for left in left_cross:
        for right in right_cross:
            pairs.add(tuple(sorted((left, right))))
    return sorted(pairs)


def _generate_consecutive_collision_exclusions(robot, urdf_path: Path) -> list[tuple[str, str]]:
    """Exclude structurally adjacent link pairs using the URDF link graph."""
    if not hasattr(robot, "get_collision_pair_names") or not hasattr(robot, "get_collision_geometries"):
        return []
    try:
        pair_names = list(robot.get_collision_pair_names())
        geoms = list(robot.get_collision_geometries())
    except Exception:
        return []

    link_graph = _build_link_adjacency_graph(urdf_path)
    link_names = set(link_graph.keys())
    parent_frame_by_geom: dict[str, str] = {}
    for geom in geoms:
        name = str(geom.get("name", ""))
        if not name:
            continue
        parent_frame_by_geom[name] = str(geom.get("parent_frame", ""))

    exclusions: list[tuple[str, str]] = []
    for a, b in pair_names:
        a = str(a)
        b = str(b)
        fa = parent_frame_by_geom.get(a, "")
        fb = parent_frame_by_geom.get(b, "")
        link_a = _collision_object_to_link_name(a, link_names)
        link_b = _collision_object_to_link_name(b, link_names)
        distance = None
        max_allowed_distance = 1
        if link_a and link_b:
            group_a = _worker_collision_group(link_a)
            group_b = _worker_collision_group(link_b)
            if {group_a, group_b}.issubset({"core", "head"}):
                max_allowed_distance = 4
            elif "core" in {group_a, group_b} and ({group_a, group_b} & {"left", "right"}):
                max_allowed_distance = 2
            elif group_a == group_b:
                max_allowed_distance = 2
            else:
                max_allowed_distance = 1
            distance = _shortest_link_distance(link_graph, link_a, link_b, max_hops=max_allowed_distance + 1)
        if (fa and fb and fa == fb) or (distance is not None and distance <= max_allowed_distance):
            exclusions.append((a, b))
    return exclusions


def _generate_worker_collision_include_pairs(
    robot, urdf_path: Path, exclude_pairs: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Map the curated worker link-pair whitelist onto geometry-pair names."""
    if not hasattr(robot, "get_collision_pair_names"):
        return []
    try:
        pair_names = list(robot.get_collision_pair_names())
    except Exception:
        return []

    exclude_set = {tuple(p) for p in exclude_pairs}
    link_graph = _build_link_adjacency_graph(urdf_path)
    link_names = set(link_graph.keys())
    curated_link_pairs = set(_worker_manual_curated_link_pairs())
    include_pairs: list[tuple[str, str]] = []
    for a, b in pair_names:
        pair = (str(a), str(b))
        if pair in exclude_set or (pair[1], pair[0]) in exclude_set:
            continue
        link_a = _collision_object_to_link_name(pair[0], link_names)
        link_b = _collision_object_to_link_name(pair[1], link_names)
        if not link_a or not link_b:
            continue
        if tuple(sorted((link_a, link_b))) in curated_link_pairs:
            include_pairs.append(pair)
    return include_pairs


def _apply_collision_tuning_mode(solver, mode_label: str) -> None:
    label = str(mode_label).lower()
    if hasattr(solver, "set_collision_tuning_mode") and hasattr(embodik, "CollisionTuningMode"):
        mode_map = {
            "precise": embodik.CollisionTuningMode.PRECISE,
            "balanced": embodik.CollisionTuningMode.BALANCED,
            # Worker-specific note: the generic SPEED preset performs worse
            # than BALANCED on this curated convex-pair workload, so keep the
            # fast UI mode on the lower-latency backend path.
            "speed": embodik.CollisionTuningMode.BALANCED,
        }
        solver.set_collision_tuning_mode(mode_map.get(label, embodik.CollisionTuningMode.BALANCED))
    if hasattr(solver, "set_proximity_gated_collision_activation_enabled"):
        solver.set_proximity_gated_collision_activation_enabled(label != "precise")
    if hasattr(solver, "set_collision_constraint_activation_multiplier"):
        if label == "precise":
            solver.set_collision_constraint_activation_multiplier(0.0)
        elif label == "balanced":
            solver.set_collision_constraint_activation_multiplier(5.0)
        else:
            solver.set_collision_constraint_activation_multiplier(3.0)
    # Worker-specific override: with the manually curated ~100-pair set, the
    # generic SPEED refinement budget bookkeeping can cost more than it saves.
    # Keep broadphase/caching enabled but disable the tiny refinement budget.
    if label == "speed":
        if hasattr(solver, "enable_collision_pair_cache"):
            solver.enable_collision_pair_cache(True, 20, 0.05, 128)
        if hasattr(solver, "set_collision_refinement_time_budget_us"):
            solver.set_collision_refinement_time_budget_us(0)
        if hasattr(solver, "enable_sphere_broadphase"):
            solver.enable_sphere_broadphase(True)


def _configure_collision_constraint(
    solver,
    *,
    enabled: bool,
    min_distance_m: float,
    max_constraints: int,
    tuning_mode: str,
    include_pairs: list[tuple[str, str]],
    exclude_pairs: list[tuple[str, str]],
) -> None:
    if not hasattr(solver, "configure_collision_constraint"):
        return
    if enabled:
        _apply_collision_tuning_mode(solver, tuning_mode)
        try:
            solver.configure_collision_constraint(
                min_distance=float(min_distance_m),
                max_constraints=int(max_constraints),
                include_pairs=list(include_pairs),
                exclude_pairs=list(exclude_pairs),
            )
            if hasattr(solver, "enable_stall_handler"):
                solver.enable_stall_handler(float(min_distance_m))
                if hasattr(solver, "configure_stall_handler"):
                    solver.configure_stall_handler(
                        stall_threshold=3,
                        restore_rate=0.2,
                        floor_fraction=0.0,
                    )
        except Exception:
            if hasattr(solver, "clear_collision_constraint"):
                try:
                    solver.clear_collision_constraint()
                except Exception:
                    pass
            if hasattr(solver, "disable_stall_handler"):
                try:
                    solver.disable_stall_handler()
                except Exception:
                    pass
    elif hasattr(solver, "clear_collision_constraint"):
        solver.clear_collision_constraint()
        if hasattr(solver, "disable_stall_handler"):
            try:
                solver.disable_stall_handler()
            except Exception:
                pass


def main() -> None:
    args = parse_args()
    urdf_path = resolve_ffw_urdf_path(args.variant)
    collision_urdf_path = resolve_generated_ffw_collision_urdf_path(args.variant) or urdf_path

    import viser
    import yourdfpy
    from viser.extras import ViserUrdf

    full_robot = embodik.RobotModel(str(collision_urdf_path), floating_base=False)
    ik_joint_names = default_worker_ik_joint_names(full_robot.get_joint_names())
    robot = embodik.RobotModel(str(collision_urdf_path), actuated_joint_names=ik_joint_names, floating_base=False)
    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=4, height=4)

    viewer_urdf_path = _prepare_viewer_urdf_path(urdf_path)
    collision_viewer_urdf_path = _prepare_viewer_urdf_path(collision_urdf_path)
    urdf_vis = ViserUrdf(
        server,
        yourdfpy.URDF.load(str(viewer_urdf_path), mesh_dir=urdf_path.parent),
        root_node_name="/robot_visual",
    )
    urdf_collision_vis = ViserUrdf(
        server,
        yourdfpy.URDF.load(
            str(collision_viewer_urdf_path),
            mesh_dir=collision_urdf_path.parent,
            build_scene_graph=False,
            build_collision_scene_graph=True,
            load_meshes=False,
            load_collision_meshes=True,
        ),
        root_node_name="/robot_collision",
        load_meshes=False,
        load_collision_meshes=True,
    )
    map_q_visual = make_visual_config_mapper(robot, urdf_vis)
    map_q_collision = make_visual_config_mapper(robot, urdf_collision_vis)

    def _set_geometry_view(mode: str) -> None:
        show_visual = str(mode) in {"Visual", "Both"}
        show_collision = str(mode) in {"Collision", "Both"}
        urdf_vis.show_visual = show_visual
        urdf_vis.show_collision = False
        urdf_collision_vis.show_visual = False
        urdf_collision_vis.show_collision = show_collision

    def _update_robot_visuals(q_now: np.ndarray) -> None:
        urdf_vis.update_cfg(map_q_visual(q_now))
        urdf_collision_vis.update_cfg(map_q_collision(q_now))

    q_lo, q_hi = robot.get_joint_limits()
    q = robot.neutral_configuration()
    joint_names = list(robot.get_joint_names())
    joint_name_to_cfg = {}
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
    _set_geometry_view("Visual")
    _update_robot_visuals(q)

    frame_map = resolve_ai_worker_frames(robot.get_frame_names())
    print(f"[worker] variant={args.variant} urdf={urdf_path}")
    print(f"[worker] frames={frame_map}")

    def _build_solver(q_posture_seed: np.ndarray):
        solver_local = embodik.KinematicsSolver(robot)
        solver_local.dt = DEFAULT_SOLVER_DT
        solver_local.set_damping(0.1)
        solver_local.set_tolerance(0.1)
        solver_local.enable_position_limits(True)
        solver_local.enable_velocity_limits(True)

        right_task_local = solver_local.add_frame_task(
            "right_tool_pose", frame_map["right_tool"], embodik.TaskType.FRAME_POSE
        )
        left_task_local = solver_local.add_frame_task(
            "left_tool_pose", frame_map["left_tool"], embodik.TaskType.FRAME_POSE
        )
        right_task_local.priority = 0
        left_task_local.priority = 0
        right_task_local.weight = 1.0
        left_task_local.weight = 1.0

        posture_local = solver_local.add_posture_task("worker_posture")
        posture_local.priority = 1
        posture_local.weight = DEFAULT_POSTURE_WEIGHT
        posture_local.set_target_configuration(np.asarray(q_posture_seed, dtype=float).copy())
        if hasattr(posture_local, "set_controlled_joint_indices"):
            posture_local.set_controlled_joint_indices(list(posture_controlled_indices))
        arm_nullspace_local = solver_local.add_posture_task("arm_nullspace")
        arm_nullspace_local.priority = 1
        arm_nullspace_local.weight = 0.0
        arm_nullspace_local.set_target_configuration(np.asarray(nullspace_bias_q, dtype=float).copy())
        return solver_local, right_task_local, left_task_local, posture_local, arm_nullspace_local

    solver, right_task, left_task, posture, arm_nullspace = _build_solver(q)

    allowed_joint_names = set(ik_joint_names)
    locked_velocity_indices: list[int] = []
    left_arm_velocity_indices: list[int] = []
    right_arm_velocity_indices: list[int] = []
    lift_velocity_indices: list[int] = []
    arm_controlled_indices: list[int] = []
    for joint_name in joint_names:
        if not hasattr(robot, "get_joint_velocity_index"):
            continue
        idx_v = int(robot.get_joint_velocity_index(joint_name))
        if hasattr(robot, "get_joint_velocity_size"):
            nv_joint = int(robot.get_joint_velocity_size(joint_name))
        else:
            nv_joint = 1
        target_index_list = None
        if joint_name.startswith(("arm_l_", "gripper_l_")):
            target_index_list = left_arm_velocity_indices
        elif joint_name.startswith(("arm_r_", "gripper_r_")):
            target_index_list = right_arm_velocity_indices
        elif joint_name.startswith("lift_"):
            target_index_list = lift_velocity_indices
        for offset in range(max(nv_joint, 1)):
            expanded_idx = idx_v + offset
            if target_index_list is not None:
                target_index_list.append(expanded_idx)
            if joint_name.startswith("arm_"):
                arm_controlled_indices.append(expanded_idx)
            if joint_name in allowed_joint_names:
                continue
            locked_velocity_indices.append(expanded_idx)
    locked_velocity_indices = sorted(set(locked_velocity_indices))
    left_arm_velocity_indices = sorted(set(left_arm_velocity_indices))
    right_arm_velocity_indices = sorted(set(right_arm_velocity_indices))
    lift_velocity_indices = sorted(set(lift_velocity_indices))
    arm_controlled_indices = sorted(set(arm_controlled_indices))
    if hasattr(arm_nullspace, "set_controlled_joint_indices"):
        arm_nullspace.set_controlled_joint_indices(list(arm_controlled_indices))
    collision_exclusions = _generate_consecutive_collision_exclusions(robot, urdf_path)
    collision_include_pairs = _generate_worker_collision_include_pairs(robot, urdf_path, collision_exclusions)
    collision_cfg = None
    collision_available = False
    if hasattr(robot, "has_collision_geometry"):
        try:
            collision_available = bool(robot.has_collision_geometry())
        except Exception:
            collision_available = False

    def frame_pose(frame_name: str) -> np.ndarray:
        pose = robot.get_frame_pose(frame_name)
        T = np.eye(4, dtype=float)
        T[:3, :3] = np.asarray(pose.rotation, dtype=float)
        T[:3, 3] = np.asarray(pose.translation, dtype=float)
        return T

    def _current_task_errors() -> tuple[float, float, float, float]:
        right_pose_now = frame_pose(frame_map["right_tool"])
        left_pose_now = frame_pose(frame_map["left_tool"])
        right_target_pose = _pose_from_ctrl(right_ctrl)
        left_target_pose = _pose_from_ctrl(left_ctrl)
        right_pos_err = float(np.linalg.norm(right_target_pose[:3, 3] - right_pose_now[:3, 3]))
        left_pos_err = float(np.linalg.norm(left_target_pose[:3, 3] - left_pose_now[:3, 3]))
        right_rot_err = _rotation_error_rad(right_target_pose[:3, :3], right_pose_now[:3, :3])
        left_rot_err = _rotation_error_rad(left_target_pose[:3, :3], left_pose_now[:3, :3])
        return right_pos_err, left_pos_err, right_rot_err, left_rot_err

    right_pose0 = frame_pose(frame_map["right_tool"])
    left_pose0 = frame_pose(frame_map["left_tool"])
    right_wxyz0_xyzw = r2q(right_pose0[:3, :3], order="xyzs")
    left_wxyz0_xyzw = r2q(left_pose0[:3, :3], order="xyzs")
    right_ctrl = server.scene.add_transform_controls(
        "/target/right_tool",
        scale=0.2,
        position=tuple(np.asarray(right_pose0[:3, 3], dtype=float)),
        wxyz=(
            float(right_wxyz0_xyzw[3]),
            float(right_wxyz0_xyzw[0]),
            float(right_wxyz0_xyzw[1]),
            float(right_wxyz0_xyzw[2]),
        ),
    )
    left_ctrl = server.scene.add_transform_controls(
        "/target/left_tool",
        scale=0.2,
        position=tuple(np.asarray(left_pose0[:3, 3], dtype=float)),
        wxyz=(
            float(left_wxyz0_xyzw[3]),
            float(left_wxyz0_xyzw[0]),
            float(left_wxyz0_xyzw[1]),
            float(left_wxyz0_xyzw[2]),
        ),
    )

    posture_target = q.copy()

    def _joint_value(name: str, default: float = 0.0) -> float:
        idx = joint_name_to_cfg.get(name)
        if idx is None or idx >= q.size:
            return default
        return float(q[idx])

    def _joint_limits(name: str, default_lo: float, default_hi: float) -> tuple[float, float]:
        idx = joint_name_to_cfg.get(name)
        if idx is None or idx >= q_lo.size or idx >= q_hi.size:
            return float(default_lo), float(default_hi)
        return float(q_lo[idx]), float(q_hi[idx])

    lift_lo_raw, lift_hi_raw = _joint_limits("lift_joint", -0.5, 0.0)
    lift_lo = min(lift_lo_raw + LIFT_LIMIT_MARGIN, lift_hi_raw)
    lift_hi = max(lift_lo_raw, lift_hi_raw - LIFT_LIMIT_MARGIN)
    with server.gui.add_folder("IK Controls"):
        timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
        auto_ik_solve = server.gui.add_checkbox("Auto IK Solve", initial_value=True)
        enable_left_ee = server.gui.add_checkbox("Enable Left EE", initial_value=True)
        enable_right_ee = server.gui.add_checkbox("Enable Right EE", initial_value=True)
        pos_gain = server.gui.add_slider("Position Gain", min=0.1, max=200.0, initial_value=DEFAULT_POS_GAIN, step=0.1)
        ori_gain = server.gui.add_slider(
            "Orientation Gain", min=0.1, max=200.0, initial_value=DEFAULT_ROT_GAIN, step=0.1
        )
        ik_steps = server.gui.add_slider("IK Iterations", min=1, max=20, initial_value=1, step=1)
        adaptive_dt = server.gui.add_checkbox("Adaptive dt", initial_value=False)
        adaptive_dt_max_scale = server.gui.add_slider(
            "Adaptive dt Max Scale", min=1.0, max=10.0, step=0.5, initial_value=10.0
        )
        adaptive_dt_ref_dist = server.gui.add_slider(
            "Adaptive dt Ref Dist (m)", min=0.01, max=0.20, step=0.01, initial_value=0.02
        )
        solve_mode = server.gui.add_dropdown(
            "EE Solve Mode",
            options=("SCALE", "SCALE_ELASTIC", "MIN_ERROR"),
            initial_value="SCALE_ELASTIC",
        )
        allow_fallback = server.gui.add_checkbox("Allow SCALE fallback to MIN_ERROR", initial_value=False)
        lock_passive = server.gui.add_checkbox("Lock passive joints", initial_value=True)
        arm_nullspace_enable = server.gui.add_checkbox("Enable Arm Nullspace Bias", initial_value=True)
        arm_nullspace_weight = server.gui.add_slider(
            "Arm Nullspace Gain", min=0.0, max=2.0, initial_value=DEFAULT_ARM_NULLSPACE_WEIGHT, step=0.01
        )
        posture_weight = server.gui.add_slider(
            "Lift/Head Bias Weight", min=0.0, max=2.0, initial_value=DEFAULT_POSTURE_WEIGHT, step=0.01
        )
        lock_lift_joint = server.gui.add_checkbox("Lock Lift Joint During IK", initial_value=False)
        manual_control = server.gui.add_checkbox("Manual Joint Control", initial_value=False)

    with server.gui.add_folder("Collision"):
        enable_collision = server.gui.add_checkbox(
            "Enable self-collision constraint",
            initial_value=hasattr(solver, "configure_collision_constraint") and collision_available,
            disabled=not hasattr(solver, "configure_collision_constraint") or not collision_available,
        )
        collision_min_dist_mm = server.gui.add_slider(
            "Collision min distance (mm)", 0.0, 120.0, 1.0, 35.0
        )
        collision_max_constraints = server.gui.add_slider(
            "Max Collision Constraints",
            min=1,
            max=64,
            step=1,
            initial_value=DEFAULT_MAX_COLLISION_CONSTRAINTS,
        )
        collision_tuning = server.gui.add_dropdown(
            "Collision Tuning",
            options=COLLISION_TUNING_OPTIONS,
            initial_value="balanced",
            disabled=not hasattr(solver, "set_collision_tuning_mode"),
        )
        exclude_consecutive = server.gui.add_checkbox(
            "Exclude consecutive links", initial_value=True, disabled=not collision_exclusions
        )
        show_collision_debug = server.gui.add_checkbox(
            "Show collision debug",
            initial_value=True,
            disabled=not hasattr(solver, "get_last_collision_debug"),
        )
        collision_debug_text = server.gui.add_text("Collision Debug", initial_value="Collision: --")
        collision_pairs_stats = server.gui.add_text(
            "Collision Pairs",
            initial_value=(
                f"available={collision_available}, "
                f"total={len(list(robot.get_collision_pair_names())) if hasattr(robot, 'get_collision_pair_names') else 0}, "
                f"included={len(collision_include_pairs)}, "
                f"excluded={len(collision_exclusions)}"
            ),
        )

    with server.gui.add_folder("CoM Constraint"):
        enable_com_constraint = server.gui.add_checkbox(
            "Enable CoM Constraint",
            initial_value=hasattr(solver, "configure_com_constraint"),
            disabled=not hasattr(solver, "configure_com_constraint"),
        )
        com_margin_pct = server.gui.add_slider(
            "Safety margin (%)", min=0.0, max=40.0, initial_value=5.0, step=1.0
        )
        com_use_proximity = server.gui.add_checkbox("Use proximity activation", initial_value=True)
        com_prox_display = server.gui.add_number(
            "Proximity threshold (m)", initial_value=0.0, disabled=True
        )
        com_vel_max = server.gui.add_slider(
            "com_vel_max (m/s)", min=0.05, max=1.0, initial_value=0.4, step=0.05
        )
        com_acc_max = server.gui.add_slider(
            "com_acc_max (m/s²)", min=0.01, max=0.5, initial_value=0.1, step=0.01
        )
        com_use_acc_limits = server.gui.add_checkbox("Acceleration limits", initial_value=True)

    with server.gui.add_folder("Visualization"):
        geometry_view = server.gui.add_dropdown(
            "Robot Geometry",
            options=GEOMETRY_VIEW_OPTIONS,
            initial_value="Visual",
        )
        show_support_polygon = server.gui.add_checkbox("Show support polygon", initial_value=True)
        show_support_contacts = server.gui.add_checkbox("Show support contacts", initial_value=True)
        show_com_viz = server.gui.add_checkbox("Show CoM", initial_value=True)
        show_drop_line = server.gui.add_checkbox("Show CoM drop line", initial_value=True)

    with server.gui.add_folder("Worker Posture"):
        lift_slider = server.gui.add_slider("Lift Joint", lift_lo, lift_hi, 0.001, _joint_value("lift_joint"))
        snap_targets = server.gui.add_button("Snap Targets to Current Tools")
        reset_pose = server.gui.add_button("Reset Robot + Targets")

    manual_joint_names = [
        name
        for name in joint_names
        if name in allowed_joint_names and not name.startswith("gripper_")
    ]
    joint_sliders = []
    with server.gui.add_folder("Joint Configuration", expand_by_default=False):
        for joint_name in manual_joint_names:
            lo, hi = _joint_limits(joint_name, -1.0, 1.0)
            joint_sliders.append(
                (
                    joint_name,
                    server.gui.add_slider(
                        joint_name,
                        min=float(lo),
                        max=float(hi),
                        step=0.001,
                        initial_value=_joint_value(joint_name),
                    ),
                )
            )

    with server.gui.add_folder("Diagnostics"):
        status = server.gui.add_text("Status", initial_value="Status: Ready")
        solve_ms = server.gui.add_text("Solve time (ms)", initial_value="--")
        right_err = server.gui.add_text("Right err", initial_value="--")
        left_err = server.gui.add_text("Left err", initial_value="--")

    debug_colors = (
        ((1.0, 0.2, 0.2), (0.2, 0.8, 0.2)),
        ((1.0, 0.5, 0.0), (0.3, 0.7, 1.0)),
        ((0.9, 0.2, 0.9), (0.2, 0.9, 0.9)),
    )
    dbg_points_a = [
        server.scene.add_icosphere(
            f"/collision_debug/point_a_{i}", radius=0.012, color=color_a, visible=False
        )
        for i, (color_a, _color_b) in enumerate(debug_colors)
    ]
    dbg_points_b = [
        server.scene.add_icosphere(
            f"/collision_debug/point_b_{i}", radius=0.012, color=color_b, visible=False
        )
        for i, (_color_a, color_b) in enumerate(debug_colors)
    ]
    dbg_lines = [None for _ in debug_colors]
    com_cfg = None
    support_polygon_cache = np.asarray(support_polygon, dtype=float).copy()
    com_outer_poly = server.scene.add_line_segments(
        "/com_viz/outer_polygon",
        points=_polygon_edge_pts(support_polygon_cache, z=0.002),
        colors=np.repeat(COLOR_OUTER_POLY, max(len(support_polygon_cache), 1), axis=0),
        line_width=4.0,
        visible=bool(show_support_polygon.value),
    )
    com_inner_poly = server.scene.add_line_segments(
        "/com_viz/inner_polygon",
        points=_polygon_edge_pts(support_polygon_cache, z=0.003),
        colors=np.repeat(COLOR_INNER_POLY, max(len(support_polygon_cache), 1), axis=0),
        line_width=3.0,
        visible=bool(show_support_polygon.value),
    )
    com_sphere = server.scene.add_icosphere(
        "/com_viz/sphere",
        radius=0.04,
        color=COLOR_COM_INSIDE,
        position=tuple(float(v) for v in robot.get_com_position()),
        visible=bool(show_com_viz.value),
    )
    com_floor_disk = server.scene.add_icosphere(
        "/com_viz/floor_disk",
        radius=0.025,
        color=COLOR_COM_INSIDE,
        position=(0.0, 0.0, 0.001),
        visible=bool(show_com_viz.value),
    )
    com_drop_line = server.scene.add_line_segments(
        "/com_viz/drop_line",
        points=np.zeros((1, 2, 3), dtype=float),
        colors=COLOR_DROP_LINE,
        line_width=2.0,
        visible=bool(show_com_viz.value and show_drop_line.value),
    )
    contact_point_handles = {
        frame_name: server.scene.add_icosphere(
            f"/com_viz/support_contact/{frame_name}",
            radius=0.018,
            color=COLOR_CONTACT_POINT,
            position=(0.0, 0.0, 0.001),
            visible=bool(show_support_contacts.value),
        )
        for frame_name in WORKER_SUPPORT_CONTACT_FRAMES
    }

    def _current_support_polygon() -> np.ndarray:
        return _compute_support_polygon_from_contacts(robot, WORKER_SUPPORT_CONTACT_FRAMES)

    def _margin_frac() -> float:
        return float(com_margin_pct.value) / 100.0

    def _configure_com_constraint_if_needed(force: bool = False) -> None:
        nonlocal com_cfg, support_polygon_cache
        support_polygon_now = _current_support_polygon()
        enabled = bool(enable_com_constraint.value) and hasattr(solver, "configure_com_constraint")
        next_cfg = (
            enabled,
            round(_margin_frac(), 6),
            bool(com_use_proximity.value),
            round(float(com_vel_max.value), 6),
            round(float(com_acc_max.value), 6),
            bool(com_use_acc_limits.value),
            tuple(np.round(support_polygon_now.reshape(-1), 6)),
        )
        if not force and next_cfg == com_cfg:
            return
        support_polygon_cache = support_polygon_now
        if not enabled:
            if hasattr(solver, "clear_com_constraint"):
                try:
                    solver.clear_com_constraint()
                except Exception:
                    pass
            com_prox_display.value = 0.0
            com_cfg = next_cfg
            return
        try:
            solver.configure_com_constraint(
                support_polygon=support_polygon_now,
                margin=_margin_frac(),
                frame_name="base_link",
                com_vel_max=float(com_vel_max.value),
                com_acc_max=float(com_acc_max.value),
                use_acceleration_limits=bool(com_use_acc_limits.value),
                proximity_fraction=0.05 if bool(com_use_proximity.value) else 0.0,
            )
            if hasattr(solver, "get_com_proximity_threshold"):
                com_prox_display.value = round(float(solver.get_com_proximity_threshold()), 4)
        except Exception:
            if hasattr(solver, "clear_com_constraint"):
                try:
                    solver.clear_com_constraint()
                except Exception:
                    pass
            enable_com_constraint.value = False
            com_prox_display.value = 0.0
        com_cfg = next_cfg

    def _update_com_visualization() -> None:
        support_polygon_now = _current_support_polygon()
        support_contacts = _support_contact_points(robot, WORKER_SUPPORT_CONTACT_FRAMES)
        inner_polygon = _shrink_polygon_2d(support_polygon_now, _margin_frac())
        com_pos = np.asarray(robot.get_com_position(), dtype=float)
        com_xy = com_pos[:2]
        slacks = _polygon_slack(inner_polygon, com_xy)
        min_slack = float(slacks.min()) if slacks.size else 0.0
        com_color = _com_color(min_slack)

        com_outer_poly.points = _polygon_edge_pts(support_polygon_now, z=0.002)
        com_outer_poly.colors = np.repeat(COLOR_OUTER_POLY, max(len(support_polygon_now), 1), axis=0)
        com_outer_poly.visible = bool(show_support_polygon.value)
        com_inner_poly.points = _polygon_edge_pts(inner_polygon, z=0.003)
        com_inner_poly.colors = np.repeat(COLOR_INNER_POLY, max(len(inner_polygon), 1), axis=0)
        com_inner_poly.visible = bool(show_support_polygon.value) and (_margin_frac() > 0.0)

        for frame_name, handle in contact_point_handles.items():
            point = support_contacts[frame_name]
            handle.position = (float(point[0]), float(point[1]), 0.001)
            handle.visible = bool(show_support_contacts.value)

        com_sphere.position = tuple(float(v) for v in com_pos)
        com_sphere.color = com_color
        com_sphere.visible = bool(show_com_viz.value)

        com_floor_disk.position = (float(com_xy[0]), float(com_xy[1]), 0.001)
        com_floor_disk.color = com_color
        com_floor_disk.visible = bool(show_com_viz.value)

        com_drop_line.points = np.array(
            [[[float(com_xy[0]), float(com_xy[1]), float(com_pos[2])], [float(com_xy[0]), float(com_xy[1]), 0.001]]],
            dtype=float,
        )
        com_drop_line.visible = bool(show_com_viz.value) and bool(show_drop_line.value)

    def _clear_collision_debug() -> None:
        nonlocal dbg_lines
        for point in dbg_points_a + dbg_points_b:
            point.visible = False
        for line in dbg_lines:
            if line is not None:
                line.visible = False
        collision_debug_text.value = "Collision: --"

    def _update_collision_debug() -> None:
        nonlocal dbg_lines
        if (
            not bool(enable_collision.value)
            or not show_collision_debug.value
            or not hasattr(solver, "get_last_collision_debug_list")
        ):
            _clear_collision_debug()
            return

        dbg_rows = list(solver.get_last_collision_debug_list())
        if not dbg_rows and hasattr(solver, "get_last_collision_debug"):
            dbg = solver.get_last_collision_debug()
            dbg_rows = [] if dbg is None else [dbg]
        if not dbg_rows:
            _clear_collision_debug()
            return

        debug_summaries: list[str] = []
        for i, row in enumerate(dbg_rows[: len(debug_colors)]):
            p_a = np.asarray(row.point_a_world, dtype=float)
            p_b = np.asarray(row.point_b_world, dtype=float)
            dbg_points_a[i].position = tuple(p_a)
            dbg_points_b[i].position = tuple(p_b)
            dbg_points_a[i].visible = True
            dbg_points_b[i].visible = True
            if dbg_lines[i] is not None:
                dbg_lines[i].remove()
            seg = np.zeros((1, 2, 3), dtype=float)
            seg[0, 0] = p_a
            seg[0, 1] = p_b
            color_a, color_b = debug_colors[i]
            colors = np.array([[color_a, color_b]], dtype=float)
            dbg_lines[i] = server.scene.add_line_segments(
                f"/collision_debug/segment_{i}",
                points=seg,
                colors=colors,
                line_width=3.0,
                visible=True,
            )
            debug_summaries.append(f"{row.object_a} <-> {row.object_b} | d={float(row.distance):.4f} m")

        for i in range(len(dbg_rows), len(debug_colors)):
            dbg_points_a[i].visible = False
            dbg_points_b[i].visible = False
            if dbg_lines[i] is not None:
                dbg_lines[i].visible = False

        collision_debug_text.value = " || ".join(debug_summaries)

    def _sync_joint_sliders_from_q(q_now: np.ndarray) -> None:
        for joint_name, slider in joint_sliders:
            idx = joint_name_to_cfg.get(joint_name)
            if idx is not None and idx < q_now.size:
                slider.value = float(q_now[idx])

    def _sync_posture_sliders_from_q(q_now: np.ndarray) -> None:
        for joint_name, slider in (("lift_joint", lift_slider),):
            idx = joint_name_to_cfg.get(joint_name)
            if idx is not None and idx < q_now.size:
                slider.value = float(q_now[idx])

    @manual_control.on_update
    def _(_evt) -> None:
        if bool(manual_control.value) and bool(auto_ik_solve.value):
            auto_ik_solve.value = False
        if bool(manual_control.value):
            right_ctrl.visible = False
            left_ctrl.visible = False

    @auto_ik_solve.on_update
    def _(_evt) -> None:
        if bool(auto_ik_solve.value) and bool(manual_control.value):
            manual_control.value = False
        if bool(auto_ik_solve.value):
            right_ctrl.visible = bool(enable_right_ee.value)
            left_ctrl.visible = bool(enable_left_ee.value)

    @geometry_view.on_update
    def _(_evt) -> None:
        _set_geometry_view(str(geometry_view.value))

    def _apply_manual_joint_configuration(q_seed: np.ndarray) -> np.ndarray:
        q_manual = np.asarray(q_seed, dtype=float).copy()
        for joint_name, slider in joint_sliders:
            idx = joint_name_to_cfg.get(joint_name)
            if idx is not None and idx < q_manual.size:
                q_manual[idx] = float(slider.value)
        for joint_name, slider in (("lift_joint", lift_slider),):
            idx = joint_name_to_cfg.get(joint_name)
            if idx is not None and idx < q_manual.size:
                q_manual[idx] = float(slider.value)
        q_manual = clip_configuration(robot, q_manual, q_lo, q_hi)
        return _apply_soft_lift_margin(q_manual, joint_name_to_cfg=joint_name_to_cfg, q_lo=q_lo, q_hi=q_hi)

    def _sync_targets_from_robot() -> None:
        right_pose = frame_pose(frame_map["right_tool"])
        left_pose = frame_pose(frame_map["left_tool"])
        _ctrl_from_pose(right_ctrl, right_pose)
        _ctrl_from_pose(left_ctrl, left_pose)
        for task, pose in ((right_task, right_pose), (left_task, left_pose)):
            try:
                task.set_target_pose(pose[:3, 3], pose[:3, :3])
            except Exception:
                pass
        right_err.value = "0.0000 m"
        left_err.value = "0.0000 m"

    def _reset_solver_state(reason: str) -> None:
        nonlocal solver, right_task, left_task, posture, arm_nullspace, collision_cfg, com_cfg
        solver, right_task, left_task, posture, arm_nullspace = _build_solver(posture_target)
        if hasattr(arm_nullspace, "set_controlled_joint_indices"):
            arm_nullspace.set_controlled_joint_indices(list(arm_controlled_indices))
        collision_cfg = None
        com_cfg = None
        _sync_targets_from_robot()
        _configure_com_constraint_if_needed(force=True)
        status.value = f"Status: solver reset after {reason}"

    @snap_targets.on_click
    def _(_evt) -> None:
        _sync_targets_from_robot()
        status.value = "Status: Targets snapped to current EE poses"

    @reset_pose.on_click
    def _(_evt) -> None:
        nonlocal q, posture_target
        q = robot.neutral_configuration()
        q = _apply_named_joint_seed(q, joint_name_to_cfg, q_lo, q_hi, DEFAULT_WORKER_SEED)
        q = clip_configuration(robot, q, q_lo, q_hi)
        q = _apply_soft_lift_margin(q, joint_name_to_cfg=joint_name_to_cfg, q_lo=q_lo, q_hi=q_hi)
        posture_target = q.copy()
        robot.update_configuration(q)
        _update_robot_visuals(q)
        _reset_solver_state("manual reset")
        _sync_joint_sliders_from_q(q)
        _sync_posture_sliders_from_q(q)
        _update_collision_debug()
        _update_com_visualization()
        timing_handle.value = 0.0
        solve_ms.value = "--"
        status.value = "Status: Robot and targets reset"

    opts = embodik.PositionStepOptions()
    _configure_com_constraint_if_needed(force=True)
    _sync_joint_sliders_from_q(q)
    _sync_posture_sliders_from_q(q)
    _sync_targets_from_robot()
    prev_manual_state = False

    while True:
        q_prev = np.asarray(q, dtype=float).copy()
        clear_all_target_velocities_if_available(solver)

        exclusion_pairs = collision_exclusions if exclude_consecutive.value else []
        include_pairs = list(collision_include_pairs)
        total_pairs = len(list(robot.get_collision_pair_names())) if hasattr(robot, "get_collision_pair_names") else 0
        collision_pairs_stats.value = (
            f"total={total_pairs}, included={len(include_pairs)}, excluded={len(exclusion_pairs)}"
        )
        next_collision_cfg = (
            bool(enable_collision.value),
            float(collision_min_dist_mm.value),
            int(collision_max_constraints.value),
            str(collision_tuning.value),
            bool(exclude_consecutive.value),
        )
        if next_collision_cfg != collision_cfg:
            _configure_collision_constraint(
                solver,
                enabled=bool(enable_collision.value),
                min_distance_m=float(collision_min_dist_mm.value) * 1e-3,
                max_constraints=int(collision_max_constraints.value),
                tuning_mode=str(collision_tuning.value),
                include_pairs=include_pairs,
                exclude_pairs=list(exclusion_pairs),
            )
            collision_cfg = next_collision_cfg
            if not bool(enable_collision.value):
                _clear_collision_debug()
        _configure_com_constraint_if_needed()

        if manual_control.value:
            if not prev_manual_state:
                _sync_joint_sliders_from_q(q)
            right_ctrl.visible = False
            left_ctrl.visible = False
            q = _apply_manual_joint_configuration(q)
            posture_target = q.copy()
            posture.weight = float(posture_weight.value)
            posture.set_target_configuration(posture_target)
            robot.update_configuration(q)
            _update_robot_visuals(q)
            _sync_joint_sliders_from_q(q)
            _sync_posture_sliders_from_q(q)
            _update_collision_debug()
            _update_com_visualization()
            right_now = np.asarray(robot.get_frame_pose(frame_map["right_tool"]).translation, dtype=float)
            left_now = np.asarray(robot.get_frame_pose(frame_map["left_tool"]).translation, dtype=float)
            right_err.value = f"{np.linalg.norm(np.asarray(right_ctrl.position, dtype=float) - right_now):.4f} m"
            left_err.value = f"{np.linalg.norm(np.asarray(left_ctrl.position, dtype=float) - left_now):.4f} m"
            status.value = "Status: Manual joint control active"
            timing_handle.value = 0.0
            solve_ms.value = "--"
            prev_manual_state = True
            time.sleep(0.002)
            continue

        active_mode = getattr(
            embodik.TaskSolveMode,
            solve_mode.value,
            embodik.TaskSolveMode.SCALE,
        )
        if not bool(auto_ik_solve.value):
            right_ctrl.visible = False
            left_ctrl.visible = False
        elif not bool(enable_left_ee.value):
            right_ctrl.visible = True
            left_ctrl.visible = False
        elif not bool(enable_right_ee.value):
            right_ctrl.visible = False
            left_ctrl.visible = True
        else:
            right_ctrl.visible = True
            left_ctrl.visible = True
        for task in (right_task, left_task):
            task.solve_mode = active_mode
            task.allow_min_error_fallback = bool(allow_fallback.value)

        posture.weight = float(posture_weight.value)
        for joint_name in ("lift_joint",):
            idx = joint_name_to_cfg.get(joint_name)
            if idx is not None and idx < posture_target.size:
                posture_target[idx] = float(nullspace_bias_q[idx])
        posture.set_target_configuration(posture_target)
        if bool(arm_nullspace_enable.value) and float(arm_nullspace_weight.value) > 0.0:
            arm_nullspace.set_target_configuration(nullspace_bias_q)
            if hasattr(arm_nullspace, "set_controlled_joint_indices"):
                active_arm_indices = []
                if bool(enable_left_ee.value):
                    active_arm_indices.extend(left_arm_velocity_indices)
                if bool(enable_right_ee.value):
                    active_arm_indices.extend(right_arm_velocity_indices)
                arm_nullspace.set_controlled_joint_indices(sorted(set(active_arm_indices)))
            arm_nullspace.weight = float(arm_nullspace_weight.value)
        else:
            arm_nullspace.weight = 0.0
            if hasattr(arm_nullspace, "set_controlled_joint_indices"):
                arm_nullspace.set_controlled_joint_indices([])

        right_pos_err, left_pos_err, right_rot_err, left_rot_err = _current_task_errors()
        posture_err = float(np.linalg.norm(np.asarray(posture_target, dtype=float) - np.asarray(q, dtype=float)))
        right_active = bool(enable_right_ee.value)
        left_active = bool(enable_left_ee.value)

        right_task.weight = 1.0 if right_active else 0.0
        left_task.weight = 1.0 if left_active else 0.0

        right_settled = (not right_active) or (
            right_pos_err <= EE_POSITION_DEADBAND and right_rot_err <= EE_ROTATION_DEADBAND
        )
        left_settled = (not left_active) or (
            left_pos_err <= EE_POSITION_DEADBAND and left_rot_err <= EE_ROTATION_DEADBAND
        )
        settled = right_settled and left_settled and posture_err < POSTURE_SLIDER_DEADBAND
        if settled:
            robot.update_configuration(q)
            _update_robot_visuals(q)
            _sync_joint_sliders_from_q(q)
            _sync_posture_sliders_from_q(q)
            _update_collision_debug()
            _update_com_visualization()
            right_err.value = f"{right_pos_err:.4f} m"
            left_err.value = f"{left_pos_err:.4f} m"
            status.value = "Status: Holding target"
            timing_handle.value = 0.0
            solve_ms.value = "--"
            prev_manual_state = False
            time.sleep(0.002)
            continue

        if not bool(auto_ik_solve.value):
            robot.update_configuration(q)
            _update_robot_visuals(q)
            _update_collision_debug()
            _update_com_visualization()
            _sync_joint_sliders_from_q(q)
            _sync_posture_sliders_from_q(q)
            right_err.value = f"{right_pos_err:.4f} m"
            left_err.value = f"{left_pos_err:.4f} m"
            status.value = "Status: Auto IK solve disabled"
            timing_handle.value = 0.0
            solve_ms.value = "--"
            prev_manual_state = False
            time.sleep(0.002)
            continue

        targets = []
        if right_active:
            targets.append(
                embodik.TaskTarget(
                    "right_tool_pose",
                    _pose_from_ctrl(right_ctrl),
                    float(pos_gain.value),
                    float(ori_gain.value),
                )
            )
        if left_active:
            targets.append(
                embodik.TaskTarget(
                    "left_tool_pose",
                    _pose_from_ctrl(left_ctrl),
                    float(pos_gain.value),
                    float(ori_gain.value),
                )
            )
        opts.max_steps = int(ik_steps.value)
        opts.position_gain = float(pos_gain.value)
        opts.orientation_gain = float(ori_gain.value)
        opts.adaptive_dt = bool(adaptive_dt.value)
        opts.adaptive_dt_max_scale = float(adaptive_dt_max_scale.value)
        opts.adaptive_dt_reference_distance = float(adaptive_dt_ref_dist.value)
        if hasattr(opts, "stall_recovery"):
            opts.stall_recovery = bool(enable_collision.value)
        configure_primary_solve_mode(opts, active_mode, bool(allow_fallback.value))
        dynamic_freeze_indices = []
        right_task_excluded = list(left_arm_velocity_indices)
        left_task_excluded = list(right_arm_velocity_indices)
        if bool(lock_lift_joint.value):
            right_task_excluded.extend(lift_velocity_indices)
            left_task_excluded.extend(lift_velocity_indices)
        if hasattr(right_task, "set_excluded_joint_indices"):
            if right_active and right_task_excluded:
                right_task.set_excluded_joint_indices(sorted(set(right_task_excluded)))
            elif hasattr(right_task, "clear_excluded_joint_indices"):
                right_task.clear_excluded_joint_indices()
        if hasattr(left_task, "set_excluded_joint_indices"):
            if left_active and left_task_excluded:
                left_task.set_excluded_joint_indices(sorted(set(left_task_excluded)))
            elif hasattr(left_task, "clear_excluded_joint_indices"):
                left_task.clear_excluded_joint_indices()
        if right_active and not left_active:
            dynamic_freeze_indices.extend(left_arm_velocity_indices)
        elif left_active and not right_active:
            dynamic_freeze_indices.extend(right_arm_velocity_indices)
        if bool(lock_lift_joint.value):
            dynamic_freeze_indices.extend(lift_velocity_indices)
        if lock_passive.value:
            opts.excluded_joint_indices = sorted(set(list(locked_velocity_indices) + dynamic_freeze_indices))
            opts.integration_zero_velocity_indices = sorted(
                set(list(locked_velocity_indices) + dynamic_freeze_indices)
            )
        else:
            opts.excluded_joint_indices = sorted(set(dynamic_freeze_indices))
            opts.integration_zero_velocity_indices = sorted(set(dynamic_freeze_indices))

        step = robust_solve_position_step(
            robot=robot,
            solver=solver,
            q_current=q,
            targets=targets,
            options=opts,
            q_lo=q_lo,
            q_hi=q_hi,
            zero_velocity_indices=locked_velocity_indices if lock_passive.value else (),
            fallback_status_names=("INVALID_INPUT", "NUMERICAL_ERROR"),
            allow_solver_intervention=True,
            apply_collision_violated_q_solution=True,
        )
        q = step.q_next
        q = _apply_soft_lift_margin(q, joint_name_to_cfg=joint_name_to_cfg, q_lo=q_lo, q_hi=q_hi)
        result = step.solver_result
        if not np.all(np.isfinite(q)):
            q = q_prev

        result_status_name = getattr(getattr(result, "status", None), "name", str(getattr(result, "status", "")))
        if bool(enable_collision.value) and result_status_name == "INFEASIBLE":
            q = q_prev.copy()

        robot.update_configuration(q)
        _update_robot_visuals(q)
        _sync_joint_sliders_from_q(q)
        _sync_posture_sliders_from_q(q)
        _update_collision_debug()
        _update_com_visualization()

        right_now = np.asarray(robot.get_frame_pose(frame_map["right_tool"]).translation, dtype=float)
        left_now = np.asarray(robot.get_frame_pose(frame_map["left_tool"]).translation, dtype=float)
        right_tgt = np.asarray(right_ctrl.position, dtype=float)
        left_tgt = np.asarray(left_ctrl.position, dtype=float)
        right_err.value = f"{np.linalg.norm(right_tgt - right_now):.4f} m"
        left_err.value = f"{np.linalg.norm(left_tgt - left_now):.4f} m"
        collision_min_distance_m = float(collision_min_dist_mm.value) * 1e-3
        current_collision_min = None
        if bool(enable_collision.value):
            try:
                if hasattr(solver, "get_last_collision_debug_list"):
                    debug_rows = list(solver.get_last_collision_debug_list())
                    if debug_rows:
                        current_collision_min = min(float(row.distance) for row in debug_rows)
                if current_collision_min is None and hasattr(solver, "get_last_collision_debug"):
                    dbg = solver.get_last_collision_debug()
                    if dbg is not None:
                        current_collision_min = float(dbg.distance)
            except Exception:
                current_collision_min = None
        deep_penetration_escape_applied = False
        if (
            bool(enable_collision.value)
            and current_collision_min is not None
            and float(current_collision_min) <= 0.0
            and result_status_name in {"INFEASIBLE", "NUMERICAL_ERROR", "NO_PROGRESS"}
            and _joint_velocity_norm(result) <= 1e-8
        ):
            escaped_q = _attempt_deep_penetration_escape_burst(
                robot=robot,
                solver=solver,
                q_current=q,
                targets=targets,
                options=opts,
                q_lo=q_lo,
                q_hi=q_hi,
                zero_velocity_indices=list(getattr(opts, "integration_zero_velocity_indices", []) or []),
                min_distance_m=collision_min_distance_m,
                max_constraints=int(collision_max_constraints.value),
                tuning_mode=str(collision_tuning.value),
                include_pairs=include_pairs,
                exclude_pairs=list(exclusion_pairs),
            )
            if escaped_q is not None:
                q = np.asarray(escaped_q, dtype=float)
                q = _apply_soft_lift_margin(q, joint_name_to_cfg=joint_name_to_cfg, q_lo=q_lo, q_hi=q_hi)
                robot.update_configuration(q)
                deep_penetration_escape_applied = True
                try:
                    if hasattr(solver, "get_last_collision_debug_list"):
                        debug_rows = list(solver.get_last_collision_debug_list())
                        if debug_rows:
                            current_collision_min = min(float(row.distance) for row in debug_rows)
                    if current_collision_min is None and hasattr(solver, "get_last_collision_debug"):
                        dbg = solver.get_last_collision_debug()
                        if dbg is not None:
                            current_collision_min = float(dbg.distance)
                except Exception:
                    current_collision_min = None
        boundary_stalled = (
            _is_collision_boundary_stall(
                result=result,
                collision_enabled=bool(enable_collision.value),
                current_collision_min=current_collision_min,
                collision_min_distance_m=collision_min_distance_m,
            )
        )

        if result_status_name == "NO_PROGRESS" and max(right_pos_err, left_pos_err) <= EE_POSITION_DEADBAND:
            status.value = "Status: Holding target"
        elif deep_penetration_escape_applied:
            status.value = "Status: penetration escape burst accepted"
        elif boundary_stalled:
            _sync_targets_from_robot()
            status.value = (
                "Status: collision limited; targets snapped to current tools"
                + (f" | {result.status_message}" if getattr(result, "status_message", "") else "")
            )
        elif result_status_name == "NO_PROGRESS":
            status.value = (
                "Status: weak progress"
                + (f" | {result.status_message}" if getattr(result, "status_message", "") else "")
            )
        elif bool(enable_collision.value) and result_status_name == "COLLISION_VIOLATED":
            _sync_targets_from_robot()
            status.value = (
                "Status: collision limited; holding last safe configuration"
                + (f" | {result.status_message}" if getattr(result, "status_message", "") else "")
            )
        elif bool(enable_collision.value) and result_status_name == "INFEASIBLE":
            _sync_targets_from_robot()
            status.value = (
                "Status: collision limited; targets snapped to current tools"
                + (f" | {result.status_message}" if getattr(result, "status_message", "") else "")
            )
        else:
            status.value = (
                f"Status: {result.status.name}"
                + (f" | {result.status_message}" if getattr(result, "status_message", "") else "")
            )
        timing_handle.value = float(step.elapsed_ms)
        solve_ms.value = f"{step.elapsed_ms:.2f}"
        prev_manual_state = False
        time.sleep(0.002)


if __name__ == "__main__":
    main()
