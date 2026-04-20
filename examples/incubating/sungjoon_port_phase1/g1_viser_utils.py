#!/usr/bin/env python3
"""Shared helpers for sungjoon Unitree G1 Viser ports.

This module keeps pure logic testable (frame resolution, target generation,
index expansion) and isolates runtime setup (robot + Viser wiring).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np


_RETARGET_PRESETS = {
    "neutral": {
        "right_palm": np.array([0.35, -0.22, 0.05], dtype=float),
        "left_palm": np.array([0.35, 0.22, 0.05], dtype=float),
        "right_ankle": np.array([0.0, -0.10, -0.85], dtype=float),
        "left_ankle": np.array([0.0, 0.10, -0.85], dtype=float),
        "imu_in_torso": np.array([0.0, 0.0, 0.0], dtype=float),
    },
    "reach_forward": {
        "right_palm": np.array([0.55, -0.20, 0.18], dtype=float),
        "left_palm": np.array([0.40, 0.20, 0.12], dtype=float),
        "right_ankle": np.array([0.02, -0.10, -0.85], dtype=float),
        "left_ankle": np.array([-0.02, 0.10, -0.85], dtype=float),
        "imu_in_torso": np.array([0.0, 0.0, 0.0], dtype=float),
    },
    "dual_reach": {
        "right_palm": np.array([0.48, -0.24, 0.18], dtype=float),
        "left_palm": np.array([0.48, 0.24, 0.18], dtype=float),
        "right_ankle": np.array([0.03, -0.10, -0.84], dtype=float),
        "left_ankle": np.array([-0.03, 0.10, -0.84], dtype=float),
        "imu_in_torso": np.array([0.0, 0.0, 0.0], dtype=float),
    },
    "cross_body": {
        "right_palm": np.array([0.40, 0.08, 0.20], dtype=float),
        "left_palm": np.array([0.40, 0.26, 0.10], dtype=float),
        "right_ankle": np.array([0.00, -0.10, -0.85], dtype=float),
        "left_ankle": np.array([0.00, 0.10, -0.85], dtype=float),
        "imu_in_torso": np.array([0.0, 0.0, 0.0], dtype=float),
    },
}


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None


def resolve_g1_urdf_path() -> Path:
    """Resolve a usable G1 URDF path for local examples.

    Priority:
    1) EMBODIK_G1_URDF env var
    2) known local checkout paths used by this workspace
    """
    env_path = os.environ.get("EMBODIK_G1_URDF", "").strip()
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"EMBODIK_G1_URDF does not exist: {p}")

    candidates = [
        Path("/home/andypark/Projects/repos/robot-model-repos/unitree_ros/robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf"),
        Path("/home/andypark/Projects/repos/robot-model-repos/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"),
        Path("/home/andypark/Projects/repos/robot-model-repos/unitree_ros/robots/g1_description/g1_29dof.urdf"),
    ]
    found = _first_existing(candidates)
    if found is not None:
        return found

    raise FileNotFoundError(
        "Could not find a G1 URDF.\n"
        "Set EMBODIK_G1_URDF=/absolute/path/to/g1.urdf"
    )


def create_g1_robot_and_visual(
    *,
    floating_base: bool,
    port: int,
    root_node_name: str = "/robot",
):
    """Create RobotModel + Viser server + ViserUrdf visualizer for G1."""
    import embodik
    import viser
    from viser.extras import ViserUrdf
    from robot_descriptions.loaders.yourdfpy import load_robot_description

    urdf_path = resolve_g1_urdf_path()
    robot = embodik.RobotModel(str(urdf_path), floating_base=floating_base)
    # Validate collision model wiring immediately after URDF import so
    # downstream examples can trust collision constraints/debug outputs.
    report = validate_robot_collision_model(robot)
    status = "OK" if report["ok"] else "WARN"
    print(
        f"[embodiK][collision-model:{status}] urdf={urdf_path.name} "
        f"geometries={report['geometry_count']} pairs={report['pair_count']} "
        f"has_collision_geometry={report['has_collision_geometry']}"
    )
    if report["issues"]:
        for issue in report["issues"]:
            print(f"[embodiK][collision-model] {issue}")

    server = viser.ViserServer(port=port)
    server.scene.add_grid("/ground", width=4, height=4)

    # If robot_descriptions has g1_description installed, prefer it for visuals.
    description_name = "g1_description"
    try:
        urdf = load_robot_description(description_name)
    except Exception:
        urdf = load_robot_description(str(urdf_path))

    urdf_vis = ViserUrdf(server, urdf, root_node_name=root_node_name)
    return robot, server, urdf_vis


def validate_robot_collision_model(robot) -> dict[str, object]:
    """Validate that collision geometry/pairs are available on a loaded model."""
    geometry_names: list[str] = []
    pair_names: list[tuple[str, str]] = []
    issues: list[str] = []
    has_collision_geometry = False

    if hasattr(robot, "has_collision_geometry"):
        try:
            has_collision_geometry = bool(robot.has_collision_geometry())
        except Exception:
            issues.append("has_collision_geometry() failed")
    else:
        issues.append("robot.has_collision_geometry API unavailable")

    if hasattr(robot, "get_collision_geometry_names"):
        try:
            geometry_names = list(robot.get_collision_geometry_names())
        except Exception:
            issues.append("get_collision_geometry_names() failed")
    else:
        issues.append("robot.get_collision_geometry_names API unavailable")

    if hasattr(robot, "get_collision_pair_names"):
        try:
            pair_names = [tuple(p) for p in robot.get_collision_pair_names()]
        except Exception:
            issues.append("get_collision_pair_names() failed")
    else:
        issues.append("robot.get_collision_pair_names API unavailable")

    if not has_collision_geometry:
        issues.append("collision geometry is missing")
    if len(geometry_names) == 0:
        issues.append("no collision geometries found")
    if len(pair_names) == 0:
        issues.append("no collision pairs found")

    return {
        "ok": len(issues) == 0,
        "has_collision_geometry": has_collision_geometry,
        "geometry_count": len(geometry_names),
        "pair_count": len(pair_names),
        "sample_geometries": geometry_names[:8],
        "sample_pairs": pair_names[:8],
        "issues": issues,
    }


def make_visual_config_mapper(robot, urdf_vis):
    """Return q->visual mapping function based on actuated joint names."""
    actuated_names = list(getattr(urdf_vis._urdf, "actuated_joint_names", []))
    name_to_q_idx: dict[str, int] = {}
    if hasattr(robot, "get_joint_config_index"):
        for name in robot.get_joint_names():
            try:
                name_to_q_idx[name] = int(robot.get_joint_config_index(name))
            except Exception:
                # Keep mapper robust across robot wrappers with partial APIs.
                pass
    if not name_to_q_idx:
        # Fallback for simpler wrappers: assume joint-name order matches q layout.
        robot_joint_names = list(robot.get_joint_names())
        name_to_q_idx = {name: idx for idx, name in enumerate(robot_joint_names)}

    def map_q(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if not actuated_names:
            return q
        cfg_vec = np.zeros(len(actuated_names), dtype=float)
        for i, jname in enumerate(actuated_names):
            idx = name_to_q_idx.get(jname)
            if idx is not None and idx < q.size:
                cfg_vec[i] = q[idx]
        return cfg_vec

    return map_q


def _pick(frame_names: set[str], candidates: list[str], label: str) -> str:
    for name in candidates:
        if name in frame_names:
            return name
    raise ValueError(f"Could not resolve frame for {label}. Tried: {candidates}")


def resolve_frames_for_g1_site_mode(frame_names: Iterable[str]) -> list[str]:
    """Resolve 4 frames used in Sungjoon-style site IK mode."""
    names = set(frame_names)
    right_palm = _pick(
        names,
        [
            "right_palm",
            "right_rubber_hand",
            "right_palm_force_sensor",
            "right_wrist_yaw_link",
            "right_wrist_pitch_link",
        ],
        "right_palm",
    )

    # Prefer explicit Sungjoon-style helper frames when available.
    explicit = []
    for label, cands in (
        ("right_palm_top", ["right_palm_top", "rpalm_top", "right_hand_top"]),
        ("right_palm_palmar", ["right_palm_palmar", "rpalm_palmar", "right_hand_palmar"]),
        ("right_palm_front", ["right_palm_front", "rpalm_front", "right_hand_front"]),
    ):
        found = None
        for c in cands:
            if c in names:
                found = c
                break
        explicit.append(found)
    if all(v is not None for v in explicit):
        return [right_palm, explicit[0], explicit[1], explicit[2]]

    # Fallback for Unitree URDF variants without *_top/palmar/front frames:
    # choose three distinct right-hand-related frames.
    preferred_tokens = [
        "right_index_2",
        "right_middle_2",
        "right_thumb_4",
        "right_index_1",
        "right_middle_1",
        "right_thumb_3",
        "right_palm_force_sensor",
        "right_wrist_yaw_link",
    ]
    filtered = [
        n
        for n in names
        if ("right_" in n)
        and ("joint" not in n)
        and any(tok in n for tok in ("index", "middle", "thumb", "palm", "wrist", "hand"))
    ]
    ranked = []
    for tok in preferred_tokens:
        for n in filtered:
            if tok in n and n not in ranked and n != right_palm:
                ranked.append(n)
    for n in sorted(filtered):
        if n not in ranked and n != right_palm:
            ranked.append(n)

    if len(ranked) < 3:
        raise ValueError(
            "Could not resolve enough right-hand auxiliary frames for site mode. "
            f"Resolved base={right_palm}, found_aux={ranked}"
        )
    return [right_palm, ranked[0], ranked[1], ranked[2]]


def resolve_frames_for_g1_base_mode(frame_names: Iterable[str]) -> dict[str, str]:
    """Resolve key full-body frames for G1 base-mode IK demos."""
    names = set(frame_names)
    return {
        "right_palm": _pick(names, ["right_palm", "right_rubber_hand", "right_wrist_yaw_link"], "right_palm"),
        "left_palm": _pick(names, ["left_palm", "left_rubber_hand", "left_wrist_yaw_link"], "left_palm"),
        "imu_in_torso": _pick(names, ["imu_in_torso", "torso_link", "waist_roll_link"], "imu_in_torso"),
        "right_ankle": _pick(names, ["right_ankle", "right_ankle_roll_link", "right_foot_col_front"], "right_ankle"),
        "left_ankle": _pick(names, ["left_ankle", "left_ankle_roll_link", "left_foot_col_front"], "left_ankle"),
    }


def build_embodik_6d_target_pose(p_target: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """Build a 4x4 pose matrix from target translation and rotation."""
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = np.asarray(R_target, dtype=float)
    pose[:3, 3] = np.asarray(p_target, dtype=float)
    return pose


def build_sungjoon_three_point_targets(
    p_target: np.ndarray,
    R_target: np.ndarray,
    local_offsets: list[np.ndarray],
) -> list[np.ndarray]:
    """Build 4 target poses from Sungjoon-style 3-point orientation surrogate."""
    if len(local_offsets) != 4:
        raise ValueError("local_offsets must contain exactly 4 vectors")
    p_target = np.asarray(p_target, dtype=float)
    R_target = np.asarray(R_target, dtype=float)
    poses: list[np.ndarray] = []
    for off in local_offsets:
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = p_target + R_target @ np.asarray(off, dtype=float)
        poses.append(pose)
    return poses


def compute_excluded_velocity_indices(robot, allowed_joint_names: set[str]) -> list[int]:
    """Expand excluded velocity indices, including multi-DoF joint ranges."""
    excluded: list[int] = []
    for jname in robot.get_joint_names():
        if jname in allowed_joint_names:
            continue
        idx_v = int(robot.get_joint_velocity_index(jname))
        if hasattr(robot, "get_joint_velocity_size"):
            nv_j = int(robot.get_joint_velocity_size(jname))
        else:
            nv_j = 1
        for k in range(max(nv_j, 1)):
            excluded.append(idx_v + k)
    # stable dedupe preserving order
    uniq: list[int] = []
    seen = set()
    for i in excluded:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    return uniq


def compute_support_polygon_from_feet(
    right_xy: np.ndarray,
    left_xy: np.ndarray,
    *,
    toe_pad: float,
    side_pad: float,
) -> np.ndarray:
    """Compute axis-aligned support polygon from right/left foot XY targets."""
    both = np.vstack([np.asarray(right_xy, dtype=float), np.asarray(left_xy, dtype=float)])
    x_min = float(np.min(both[:, 0]) - toe_pad)
    x_max = float(np.max(both[:, 0]) + toe_pad)
    y_min = float(np.min(both[:, 1]) - side_pad)
    y_max = float(np.max(both[:, 1]) + side_pad)
    return np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype=float,
    )


def _convex_hull_xy(points_xy: np.ndarray) -> np.ndarray:
    """Compute 2D convex hull with monotone chain (returns CCW vertices)."""
    pts = np.asarray(points_xy, dtype=float)
    if pts.shape[0] <= 1:
        return pts.copy()
    # Lexicographic sort + unique rows
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    uniq = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - uniq[-1]) > 1e-12:
            uniq.append(p)
    pts = np.asarray(uniq, dtype=float)
    if pts.shape[0] <= 2:
        return pts

    def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        oa = a - o
        ob = b - o
        return float(oa[0] * ob[1] - oa[1] * ob[0])

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
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _foot_contact_corners_xy(
    foot_pose: np.ndarray,
    *,
    half_length: float,
    half_width: float,
) -> np.ndarray:
    """Return 4 contact rectangle corners in XY for one foot pose."""
    foot_pose = np.asarray(foot_pose, dtype=float)
    c = foot_pose[:2, 3]
    x_axis = foot_pose[:2, 0]
    y_axis = foot_pose[:2, 1]
    x_n = np.linalg.norm(x_axis)
    y_n = np.linalg.norm(y_axis)
    if x_n < 1e-12 or y_n < 1e-12:
        # Fallback to world-aligned rectangle
        x_axis = np.array([1.0, 0.0], dtype=float)
        y_axis = np.array([0.0, 1.0], dtype=float)
    else:
        x_axis = x_axis / x_n
        y_axis = y_axis / y_n
    return np.array(
        [
            c + half_length * x_axis + half_width * y_axis,
            c + half_length * x_axis - half_width * y_axis,
            c - half_length * x_axis - half_width * y_axis,
            c - half_length * x_axis + half_width * y_axis,
        ],
        dtype=float,
    )


def compute_support_polygon_from_foot_poses(
    right_foot_pose: np.ndarray,
    left_foot_pose: np.ndarray,
    *,
    foot_length: float = 0.22,
    foot_width: float = 0.10,
    toe_pad: float = 0.0,
    side_pad: float = 0.0,
) -> np.ndarray:
    """Compute support polygon from oriented foot-contact rectangles.

    This is more precise than an axis-aligned ankle bounding box because each
    foot contributes a rotated contact patch derived from its local pose.
    """
    hl = 0.5 * float(foot_length) + float(toe_pad)
    hw = 0.5 * float(foot_width) + float(side_pad)
    r_corners = _foot_contact_corners_xy(right_foot_pose, half_length=hl, half_width=hw)
    l_corners = _foot_contact_corners_xy(left_foot_pose, half_length=hl, half_width=hw)
    pts = np.vstack([r_corners, l_corners])
    hull = _convex_hull_xy(pts)
    return hull


def polygon_segments_xy(poly_xy: np.ndarray, z: float = 0.0) -> np.ndarray:
    """Convert closed XY polygon vertices to line-segment array (N, 2, 3)."""
    poly_xy = np.asarray(poly_xy, dtype=float)
    n = poly_xy.shape[0]
    seg = np.zeros((n, 2, 3), dtype=float)
    for i in range(n):
        j = (i + 1) % n
        seg[i, 0, :2] = poly_xy[i]
        seg[i, 1, :2] = poly_xy[j]
        seg[i, :, 2] = z
    return seg


def get_retargeting_presets() -> dict[str, dict[str, np.ndarray]]:
    """Return local-space retargeting presets keyed by frame name."""
    return {k: {kk: vv.copy() for kk, vv in v.items()} for k, v in _RETARGET_PRESETS.items()}


def retarget_position_from_anchor(
    anchor_position: np.ndarray,
    anchor_rotation: np.ndarray,
    local_offset: np.ndarray,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Map local offset into world coordinates using an anchor pose."""
    anchor_position = np.asarray(anchor_position, dtype=float)
    anchor_rotation = np.asarray(anchor_rotation, dtype=float)
    local_offset = np.asarray(local_offset, dtype=float)
    return anchor_position + anchor_rotation @ (float(scale) * local_offset)


def build_site_mode_target_payloads(
    mode: str,
    p_target: np.ndarray,
    R_target: np.ndarray,
    local_offsets: list[np.ndarray],
    *,
    position_gain: float,
    orientation_gain: float,
) -> list[dict]:
    """Build deterministic target payloads for site-IK modes."""
    if mode == "Sungjoon 3-point":
        point_poses = build_sungjoon_three_point_targets(p_target, R_target, local_offsets)
        return [
            {
                "task_name": f"g1_site_{i}",
                "target_pose": pose,
                "position_gain": float(position_gain),
                "orientation_gain": 0.0,
            }
            for i, pose in enumerate(point_poses)
        ]
    if mode == "EmbodiK 6D":
        pose = build_embodik_6d_target_pose(p_target, R_target)
        return [
            {
                "task_name": "g1_pose_pos",
                "target_pose": pose,
                "position_gain": float(position_gain),
                "orientation_gain": 0.0,
            },
            {
                "task_name": "g1_pose_ori",
                "target_pose": pose,
                "position_gain": 0.0,
                "orientation_gain": float(orientation_gain),
            },
        ]
    raise ValueError(f"Unknown site mode: {mode}")


def build_fullbody_target_payloads(
    foot_mode: str,
    target_poses: dict[str, np.ndarray],
    *,
    position_gain: float,
    foot_orientation_gain: float,
) -> list[dict]:
    """Build deterministic full-body target payloads for base IK script."""
    ordered_keys = ["right_palm", "left_palm", "imu_in_torso", "right_ankle", "left_ankle"]
    payloads = [
        {
            "task_name": f"g1_base_{key}_pos",
            "target_pose": target_poses[key],
            "position_gain": float(position_gain),
            "orientation_gain": 0.0,
        }
        for key in ordered_keys
    ]
    if foot_mode == "EmbodiK 6D Feet":
        payloads.append(
            {
                "task_name": "g1_base_right_ankle_ori",
                "target_pose": target_poses["right_ankle"],
                "position_gain": 0.0,
                "orientation_gain": float(foot_orientation_gain),
            }
        )
        payloads.append(
            {
                "task_name": "g1_base_left_ankle_ori",
                "target_pose": target_poses["left_ankle"],
                "position_gain": 0.0,
                "orientation_gain": float(foot_orientation_gain),
            }
        )
    elif foot_mode != "Sungjoon Position-only":
        raise ValueError(f"Unknown foot mode: {foot_mode}")
    return payloads


def collision_arm_only_option_values(
    *,
    arm_only_enabled: bool,
    floating_base: bool,
    excluded_joint_indices: list[int],
) -> tuple[list[int], list[int]]:
    """Return (excluded_joint_indices, integration_zero_velocity_indices)."""
    if not arm_only_enabled:
        return [], []
    if floating_base:
        return list(excluded_joint_indices), list(range(6))
    return list(excluded_joint_indices), []


def compute_feet_center(right_foot_position: np.ndarray, left_foot_position: np.ndarray) -> np.ndarray:
    """Return the world-frame center point between right/left feet."""
    r = np.asarray(right_foot_position, dtype=float)
    l = np.asarray(left_foot_position, dtype=float)
    return 0.5 * (r + l)


def ground_floating_base_from_feet_center(
    q: np.ndarray,
    right_foot_position: np.ndarray,
    left_foot_position: np.ndarray,
) -> np.ndarray:
    """Shift floating-base translation by negative feet-center world position."""
    q_out = np.asarray(q, dtype=float).copy()
    if q_out.size < 3:
        return q_out
    feet_center = compute_feet_center(right_foot_position, left_foot_position)
    q_out[:3] -= feet_center
    return q_out

