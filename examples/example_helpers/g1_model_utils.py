#!/usr/bin/env python3
"""Shared helpers for Unitree G1 examples.

This module keeps pure logic testable (frame resolution, target generation,
index expansion) and isolates runtime setup (robot + Viser wiring).
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
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
    "right_wave_high": {
        "right_palm": np.array([0.38, -0.34, 0.34], dtype=float),
        "left_palm": np.array([0.34, 0.20, 0.08], dtype=float),
        "right_ankle": np.array([0.01, -0.10, -0.85], dtype=float),
        "left_ankle": np.array([-0.01, 0.10, -0.85], dtype=float),
        "imu_in_torso": np.array([0.0, -0.02, 0.02], dtype=float),
    },
    "left_wave_high": {
        "right_palm": np.array([0.34, -0.20, 0.08], dtype=float),
        "left_palm": np.array([0.38, 0.34, 0.34], dtype=float),
        "right_ankle": np.array([-0.01, -0.10, -0.85], dtype=float),
        "left_ankle": np.array([0.01, 0.10, -0.85], dtype=float),
        "imu_in_torso": np.array([0.0, 0.02, 0.02], dtype=float),
    },
    "low_guard": {
        "right_palm": np.array([0.34, -0.30, -0.08], dtype=float),
        "left_palm": np.array([0.34, 0.30, -0.08], dtype=float),
        "right_ankle": np.array([0.03, -0.12, -0.84], dtype=float),
        "left_ankle": np.array([-0.03, 0.12, -0.84], dtype=float),
        "imu_in_torso": np.array([0.02, 0.0, -0.10], dtype=float),
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
    2) bundled generated G1 collision URDF
    """
    env_path = os.environ.get("EMBODIK_G1_URDF", "").strip()
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"EMBODIK_G1_URDF does not exist: {p}")

    candidates = [
        Path(__file__).resolve().parents[1]
        / "assets/g1/generated/g1_29dof_rev_1_0_with_inspire_hand_FTP_box_collision.urdf",
    ]
    found = _first_existing(candidates)
    if found is not None:
        return found

    raise FileNotFoundError(
        "Could not find a G1 URDF.\n" "Set EMBODIK_G1_URDF=/absolute/path/to/g1.urdf"
    )


def resolve_g1_collision_urdf_path() -> Path:
    """Resolve the URDF used for G1 IK/collision models.

    Visuals continue to use ``resolve_g1_urdf_path()``. The generated collision
    URDF replaces high-poly collision meshes with primitives so interactive
    collision-aware IK does not pay detailed mesh distance costs.
    """
    env_path = os.environ.get("EMBODIK_G1_COLLISION_URDF", "").strip()
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"EMBODIK_G1_COLLISION_URDF does not exist: {p}")

    generated = (
        Path(__file__).resolve().parents[1]
        / "assets/g1/generated/g1_29dof_rev_1_0_with_inspire_hand_FTP_box_collision.urdf"
    )
    if generated.is_file():
        return generated
    return resolve_g1_urdf_path()


def prepare_g1_viewer_urdf_path(urdf_path: Path) -> Path:
    """Return a Viser-safe G1 URDF path with unsupported mimic tags removed.

    ``yourdfpy`` logs a warning every time it updates a mimic joint whose source
    joint is not in the reduced IK model. For these examples the Inspire hand
    joints are passive, so removing mimic tags for viewer-only loading preserves
    the visible fixed hand posture and avoids per-frame log spam.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    changed = False
    for joint in root.findall(".//joint"):
        for mimic in list(joint.findall("mimic")):
            joint.remove(mimic)
            changed = True
    if not changed:
        return urdf_path

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{urdf_path.stem}_viewer.urdf",
        prefix="embodik_g1_",
        delete=False,
    )
    tree.write(tmp, encoding="unicode")
    tmp.close()
    return Path(tmp.name)


def g1_reduced_ik_joint_names(joint_names: Iterable[str]) -> list[str]:
    """Keep G1 whole-body IK joints and drop unused Inspire hand/finger DOFs."""
    keep_suffixes = (
        "_hip_pitch_joint",
        "_hip_roll_joint",
        "_hip_yaw_joint",
        "_knee_joint",
        "_ankle_pitch_joint",
        "_ankle_roll_joint",
        "_shoulder_pitch_joint",
        "_shoulder_roll_joint",
        "_shoulder_yaw_joint",
        "_elbow_joint",
        "_wrist_roll_joint",
        "_wrist_pitch_joint",
        "_wrist_yaw_joint",
    )
    keep_exact = {"waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"}
    out: list[str] = []
    for name in joint_names:
        if name == "root_joint":
            continue
        if name in keep_exact or name.endswith(keep_suffixes):
            out.append(name)
    return out


def create_g1_robot_model(*, floating_base: bool, reduced_ik: bool = True):
    """Create a G1 RobotModel, optionally locking unused hand/finger joints."""
    import embodik

    urdf_path = resolve_g1_collision_urdf_path()
    if not reduced_ik:
        return embodik.RobotModel(str(urdf_path), floating_base=floating_base)
    full_robot = embodik.RobotModel(str(urdf_path), floating_base=floating_base)
    ik_joint_names = g1_reduced_ik_joint_names(full_robot.get_joint_names())
    return embodik.RobotModel(
        str(urdf_path),
        actuated_joint_names=ik_joint_names,
        floating_base=floating_base,
    )


def create_g1_robot_and_visual(
    *,
    floating_base: bool,
    port: int,
    root_node_name: str = "/robot",
    reduced_ik: bool = True,
    validate_collision: bool = False,
):
    """Create RobotModel + Viser server + ViserUrdf visualizer for G1."""
    import viser
    from robot_descriptions.loaders.yourdfpy import load_robot_description
    from viser.extras import ViserUrdf

    urdf_path = resolve_g1_urdf_path()
    robot = create_g1_robot_model(floating_base=floating_base, reduced_ik=reduced_ik)
    if validate_collision:
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
        urdf_description = load_robot_description(description_name)
    except Exception:
        urdf_description = load_robot_description(str(urdf_path))

    viewer_source_path = Path(getattr(urdf_description, "path", urdf_path))
    viewer_urdf_path = prepare_g1_viewer_urdf_path(viewer_source_path)
    import yourdfpy

    urdf_vis = ViserUrdf(
        server,
        yourdfpy.URDF.load(str(viewer_urdf_path), mesh_dir=urdf_path.parent),
        root_node_name=root_node_name,
    )
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
    """Resolve 4 frames used in G1-style site IK mode."""
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

    # Prefer explicit G1-style helper frames when available.
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
        "right_palm": _pick(
            names,
            [
                "right_palm",
                "right_rubber_hand",
                "right_palm_force_sensor",
                "right_wrist_yaw_link",
            ],
            "right_palm",
        ),
        "left_palm": _pick(
            names,
            [
                "left_palm",
                "left_rubber_hand",
                "left_palm_force_sensor",
                "left_wrist_yaw_link",
            ],
            "left_palm",
        ),
        "imu_in_torso": _pick(
            names, ["imu_in_torso", "torso_link", "waist_roll_link"], "imu_in_torso"
        ),
        "right_ankle": _pick(
            names, ["right_ankle", "right_ankle_roll_link", "right_foot_col_front"], "right_ankle"
        ),
        "left_ankle": _pick(
            names, ["left_ankle", "left_ankle_roll_link", "left_foot_col_front"], "left_ankle"
        ),
    }


def build_embodik_6d_target_pose(p_target: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """Build a 4x4 pose matrix from target translation and rotation."""
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = np.asarray(R_target, dtype=float)
    pose[:3, 3] = np.asarray(p_target, dtype=float)
    return pose


def build_three_point_orientation_targets(
    p_target: np.ndarray,
    R_target: np.ndarray,
    local_offsets: list[np.ndarray],
) -> list[np.ndarray]:
    """Build 4 target poses from a 3-point orientation surrogate."""
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


def shrink_polygon_xy(poly_xy: np.ndarray, margin_fraction: float) -> np.ndarray:
    """Shrink polygon vertices toward the centroid by margin_fraction * char_size."""
    poly_xy = np.asarray(poly_xy, dtype=float).reshape(-1, 2)
    if margin_fraction <= 0.0 or poly_xy.size == 0:
        return poly_xy.copy()
    centroid = poly_xy.mean(axis=0)
    char_size = float(np.mean(np.linalg.norm(poly_xy - centroid, axis=1)))
    shrink = float(np.clip(margin_fraction, 0.0, 1.0)) * max(0.0, char_size - 1e-9)
    out = []
    for vertex in poly_xy:
        delta = centroid - vertex
        norm = float(np.linalg.norm(delta))
        out.append(vertex if norm < 1e-12 else vertex + (shrink / norm) * delta)
    return np.asarray(out, dtype=float)


def polygon_slack_xy(poly_xy: np.ndarray, point_xy: np.ndarray) -> np.ndarray:
    """Return signed edge slacks for a CCW convex polygon; positive means inside."""
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


def com_min_slack(
    support_polygon_xy: np.ndarray,
    com_xy: np.ndarray,
    *,
    margin_fraction: float = 0.0,
) -> float:
    """Return minimum CoM slack against the active support polygon."""
    inner = shrink_polygon_xy(support_polygon_xy, margin_fraction)
    slacks = polygon_slack_xy(inner, com_xy)
    return float(slacks.min()) if slacks.size else 0.0


def com_slack_color(
    min_slack: float,
    *,
    near_boundary_threshold: float = 0.02,
) -> tuple[float, float, float]:
    """Return green/yellow/red visual color for a CoM slack value."""
    if min_slack < 0.0:
        return (0.92, 0.20, 0.24)
    if min_slack < float(near_boundary_threshold):
        return (1.00, 0.68, 0.10)
    return (0.12, 0.78, 0.32)


def get_retargeting_presets() -> dict[str, dict[str, np.ndarray]]:
    """Return local-space retargeting presets keyed by frame name."""
    return {k: {kk: vv.copy() for kk, vv in v.items()} for k, v in _RETARGET_PRESETS.items()}


_RETARGET_CLIPS = {
    "reach_cycle": [
        {"time": 0.0, "preset": "neutral"},
        {"time": 0.8, "preset": "reach_forward"},
        {"time": 1.6, "preset": "dual_reach"},
        {"time": 2.4, "preset": "neutral"},
    ],
    "cross_body": [
        {"time": 0.0, "preset": "neutral"},
        {"time": 0.7, "preset": "cross_body"},
        {"time": 1.4, "preset": "dual_reach"},
        {"time": 2.1, "preset": "neutral"},
    ],
    "squat_reach": [
        {
            "time": 0.0,
            "targets": {
                "right_palm": [0.35, -0.22, 0.05],
                "left_palm": [0.35, 0.22, 0.05],
                "right_ankle": [0.0, -0.10, -0.85],
                "left_ankle": [0.0, 0.10, -0.85],
                "imu_in_torso": [0.0, 0.0, 0.0],
            },
        },
        {
            "time": 0.8,
            "targets": {
                "right_palm": [0.44, -0.24, -0.04],
                "left_palm": [0.44, 0.24, -0.04],
                "right_ankle": [0.03, -0.11, -0.84],
                "left_ankle": [-0.03, 0.11, -0.84],
                "imu_in_torso": [0.02, 0.0, -0.12],
            },
        },
        {
            "time": 1.6,
            "targets": {
                "right_palm": [0.55, -0.24, 0.14],
                "left_palm": [0.55, 0.24, 0.14],
                "right_ankle": [0.03, -0.11, -0.84],
                "left_ankle": [-0.03, 0.11, -0.84],
                "imu_in_torso": [0.03, 0.0, -0.08],
            },
        },
        {"time": 2.4, "preset": "neutral"},
    ],
    "side_shift": [
        {"time": 0.0, "preset": "neutral"},
        {
            "time": 0.7,
            "targets": {
                "right_palm": [0.42, -0.28, 0.10],
                "left_palm": [0.36, 0.18, 0.08],
                "right_ankle": [0.02, -0.12, -0.85],
                "left_ankle": [0.00, 0.08, -0.85],
                "imu_in_torso": [0.0, -0.05, 0.0],
            },
        },
        {
            "time": 1.4,
            "targets": {
                "right_palm": [0.36, -0.18, 0.08],
                "left_palm": [0.42, 0.28, 0.10],
                "right_ankle": [0.00, -0.08, -0.85],
                "left_ankle": [0.02, 0.12, -0.85],
                "imu_in_torso": [0.0, 0.05, 0.0],
            },
        },
        {"time": 2.1, "preset": "neutral"},
    ],
    "alternating_wave": [
        {"time": 0.0, "preset": "neutral"},
        {"time": 0.5, "preset": "right_wave_high"},
        {"time": 1.0, "preset": "low_guard"},
        {"time": 1.5, "preset": "left_wave_high"},
        {"time": 2.0, "preset": "dual_reach"},
        {"time": 2.6, "preset": "neutral"},
    ],
    "balance_circle": [
        {"time": 0.0, "preset": "neutral"},
        {
            "time": 0.6,
            "targets": {
                "right_palm": [0.44, -0.24, 0.16],
                "left_palm": [0.36, 0.22, 0.06],
                "right_ankle": [0.03, -0.12, -0.85],
                "left_ankle": [0.00, 0.08, -0.85],
                "imu_in_torso": [0.02, -0.04, 0.02],
            },
        },
        {
            "time": 1.2,
            "targets": {
                "right_palm": [0.36, -0.22, 0.06],
                "left_palm": [0.44, 0.24, 0.16],
                "right_ankle": [0.00, -0.08, -0.85],
                "left_ankle": [0.03, 0.12, -0.85],
                "imu_in_torso": [0.02, 0.04, 0.02],
            },
        },
        {
            "time": 1.8,
            "targets": {
                "right_palm": [0.48, -0.24, 0.10],
                "left_palm": [0.48, 0.24, 0.10],
                "right_ankle": [0.02, -0.11, -0.84],
                "left_ankle": [-0.02, 0.11, -0.84],
                "imu_in_torso": [-0.02, 0.00, -0.06],
            },
        },
        {"time": 2.4, "preset": "neutral"},
    ],
}


def _clip_keyframe_targets(keyframe: dict[str, object]) -> dict[str, np.ndarray]:
    preset = keyframe.get("preset")
    if preset is not None:
        return get_retargeting_presets()[str(preset)]
    raw_targets = keyframe.get("targets")
    if not isinstance(raw_targets, dict):
        raise ValueError("Retargeting keyframe must provide either preset or targets")
    return {str(k): np.asarray(v, dtype=float) for k, v in raw_targets.items()}


def get_retargeting_clips() -> dict[str, list[dict[str, object]]]:
    """Return deterministic synthetic retargeting clips.

    Keyframe targets are local offsets from a calibrated anchor pose. The frame
    keys intentionally mirror the G1 whole-body target frames used by the
    examples: hands, ankles, and torso/IMU.
    """
    clips: dict[str, list[dict[str, object]]] = {}
    for name, frames in _RETARGET_CLIPS.items():
        clips[name] = []
        for frame in frames:
            item: dict[str, object] = {"time": float(frame["time"])}
            if "preset" in frame:
                item["preset"] = str(frame["preset"])
            if "targets" in frame:
                item["targets"] = {
                    k: np.asarray(v, dtype=float).copy() for k, v in frame["targets"].items()
                }
            clips[name].append(item)
    return clips


def get_retargeting_clip_duration(clip_name: str) -> float:
    """Return duration in seconds for a synthetic retargeting clip."""
    clips = get_retargeting_clips()
    if clip_name not in clips:
        raise ValueError(f"Unknown retargeting clip: {clip_name}")
    return float(clips[clip_name][-1]["time"])


def sample_retargeting_clip(clip_name: str, t: float) -> dict[str, np.ndarray]:
    """Linearly sample local target offsets from a named retargeting clip."""
    clips = get_retargeting_clips()
    if clip_name not in clips:
        raise ValueError(f"Unknown retargeting clip: {clip_name}")
    frames = clips[clip_name]
    if not frames:
        raise ValueError(f"Retargeting clip has no keyframes: {clip_name}")

    duration = float(frames[-1]["time"])
    if duration <= 0.0:
        return _clip_keyframe_targets(frames[0])
    t = float(np.clip(float(t), 0.0, duration))
    prev = frames[0]
    next_frame = frames[-1]
    for i in range(len(frames) - 1):
        a = frames[i]
        b = frames[i + 1]
        if float(a["time"]) <= t <= float(b["time"]):
            prev = a
            next_frame = b
            break

    t0 = float(prev["time"])
    t1 = float(next_frame["time"])
    alpha = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
    a_targets = _clip_keyframe_targets(prev)
    b_targets = _clip_keyframe_targets(next_frame)
    keys = set(a_targets) | set(b_targets)
    out: dict[str, np.ndarray] = {}
    for key in keys:
        a = a_targets.get(key, b_targets[key])
        b = b_targets.get(key, a_targets[key])
        out[key] = (1.0 - alpha) * np.asarray(a, dtype=float) + alpha * np.asarray(b, dtype=float)
    return out


def build_retargeting_target_poses(
    anchor_pose: np.ndarray,
    local_offsets: dict[str, np.ndarray],
    *,
    scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Map sampled local offsets into world-frame 4x4 target poses."""
    anchor_pose = np.asarray(anchor_pose, dtype=float)
    anchor_p = anchor_pose[:3, 3]
    anchor_R = anchor_pose[:3, :3]
    poses: dict[str, np.ndarray] = {}
    for key, local in local_offsets.items():
        pose = np.eye(4, dtype=float)
        pose[:3, :3] = anchor_R
        pose[:3, 3] = retarget_position_from_anchor(
            anchor_p, anchor_R, np.asarray(local, dtype=float), scale=scale
        )
        poses[key] = pose
    return poses


def build_retargeting_delta_target_poses(
    anchor_pose: np.ndarray,
    reference_poses: dict[str, np.ndarray],
    local_offsets: dict[str, np.ndarray],
    neutral_offsets: dict[str, np.ndarray],
    *,
    scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Apply retargeting clips as deltas from measured initial target poses.

    The synthetic clips are authored as torso-local offsets, but the exact G1
    hand/foot frames vary across URDFs. Using deltas from the neutral keyframe
    keeps the first retarget sample coincident with the current robot pose.
    """
    anchor_pose = np.asarray(anchor_pose, dtype=float)
    anchor_R = anchor_pose[:3, :3]
    poses: dict[str, np.ndarray] = {}
    keys = set(reference_poses) & set(local_offsets)
    for key in keys:
        pose = np.asarray(reference_poses[key], dtype=float).copy()
        neutral = np.asarray(neutral_offsets.get(key, local_offsets[key]), dtype=float)
        local = np.asarray(local_offsets[key], dtype=float)
        pose[:3, 3] = pose[:3, 3] + anchor_R @ (float(scale) * (local - neutral))
        poses[key] = pose
    return poses


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
    if mode == "G1 3-point":
        point_poses = build_three_point_orientation_targets(p_target, R_target, local_offsets)
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
    elif foot_mode != "G1 Position-only":
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


def _g1_side_prefixed(names: set[str], side: str, suffixes: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for suffix in suffixes:
        name = f"{side}_{suffix}"
        if name in names:
            out.append(name)
    return out


def g1_collision_pair_preset_options() -> tuple[str, ...]:
    """Return UI/CLI names for curated G1 collision include-pair presets."""
    return ("core", "core_plus_legs", "all_debug")


def g1_collision_pairs_for_preset(robot, preset: str = "core") -> list[tuple[str, str]]:
    """Return curated G1 collision include pairs for interactive IK.

    The default ``core`` preset intentionally treats the Inspire hand as a
    coarse end-effector envelope. Full finger-vs-finger checks are too expensive
    for interactive use and are not needed to catch the obvious whole-body
    self-collisions: hands/forearms into torso/head/pelvis and left arm into
    right arm.
    """
    if not hasattr(robot, "get_collision_pair_names") or not hasattr(
        robot, "get_collision_geometry_names"
    ):
        return []

    preset = str(preset)
    try:
        available_pairs = [tuple(map(str, p)) for p in robot.get_collision_pair_names()]
        names = set(map(str, robot.get_collision_geometry_names()))
    except Exception:
        return []

    if preset == "all_debug":
        return list(available_pairs)
    if preset not in {"core", "core_plus_legs"}:
        raise ValueError(
            f"Unknown G1 collision pair preset: {preset!r}. "
            f"Expected one of {g1_collision_pair_preset_options()}."
        )

    body = [
        name for name in ("pelvis_contour_link_0", "torso_link_0", "head_link_0") if name in names
    ]
    coarse_arm_suffixes = (
        "shoulder_yaw_link_0",
        "elbow_link_0",
        "wrist_roll_link_0",
        "wrist_pitch_link_0",
        "wrist_yaw_link_0",
        "base_link_0",
        "palm_force_sensor_0",
    )
    left_arm = _g1_side_prefixed(names, "left", coarse_arm_suffixes)
    right_arm = _g1_side_prefixed(names, "right", coarse_arm_suffixes)

    wanted: set[frozenset[str]] = set()
    for arm in (left_arm, right_arm):
        for arm_geom in arm:
            for body_geom in body:
                wanted.add(frozenset((arm_geom, body_geom)))
    for left_geom in left_arm:
        for right_geom in right_arm:
            wanted.add(frozenset((left_geom, right_geom)))

    if preset == "core_plus_legs":
        leg_suffixes = (
            "hip_pitch_link_0",
            "hip_yaw_link_0",
            "knee_link_0",
            "ankle_pitch_link_0",
            "ankle_roll_link_0",
        )
        left_leg = _g1_side_prefixed(names, "left", leg_suffixes)
        right_leg = _g1_side_prefixed(names, "right", leg_suffixes)
        hand_suffixes = (
            "wrist_yaw_link_0",
            "base_link_0",
            "palm_force_sensor_0",
        )
        left_hand = _g1_side_prefixed(names, "left", hand_suffixes)
        right_hand = _g1_side_prefixed(names, "right", hand_suffixes)
        for hand_geom in left_hand:
            for leg_geom in right_leg:
                wanted.add(frozenset((hand_geom, leg_geom)))
        for hand_geom in right_hand:
            for leg_geom in left_leg:
                wanted.add(frozenset((hand_geom, leg_geom)))

    return [(a, b) for a, b in available_pairs if frozenset((a, b)) in wanted]


def compute_feet_center(
    right_foot_position: np.ndarray, left_foot_position: np.ndarray
) -> np.ndarray:
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
