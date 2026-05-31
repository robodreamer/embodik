#!/usr/bin/env python3
"""Bimanual whole-body IK teleop entrypoint.

This entrypoint runs a shared bimanual whole-body constraint teleop demo across
the public ROBOTIS AI Worker assets and the RB-Y1 model from
``robot_descriptions``.

URDF sources:
- explicit ``--urdf`` / ``--collision-urdf``
- a local clone of the public ``ROBOTIS-GIT/ai_worker`` repository via
  ``--ai-worker-root``
- automatic cached download from the public ``ROBOTIS-GIT/ai_worker`` repo

Examples
--------
python 06_bimanual_whole_body_ik.py \
    --variant sg2 \
    --ai-worker-root /path/to/ROBOTIS-GIT/ai_worker

python 06_bimanual_whole_body_ik.py \
    --variant bg2 \
    --urdf /path/to/base.urdf \
    --collision-urdf /path/to/collision.urdf

python 06_bimanual_whole_body_ik.py --robot rby1
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import example_helpers.common_bimanual_teleop_app as common_app
    from example_helpers.ik_common import DEFAULT_VISER_PORT
    from example_helpers.public_ai_worker_paths import resolve_public_ai_worker_urdf_paths
    from example_helpers.seer_teleop import DEFAULT_SEER_CONTROLLER_PORT
except ModuleNotFoundError as exc:
    if exc.name != "example_helpers" and not str(exc.name).startswith("example_helpers."):
        raise
    import examples.example_helpers.common_bimanual_teleop_app as common_app
    from examples.example_helpers.ik_common import DEFAULT_VISER_PORT
    from examples.example_helpers.public_ai_worker_paths import resolve_public_ai_worker_urdf_paths
    from examples.example_helpers.seer_teleop import DEFAULT_SEER_CONTROLLER_PORT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot",
        choices=("ai-worker", "rby1"),
        default="ai-worker",
        help="Robot model family to load. 'ai-worker' uses the ROBOTIS FFW assets; 'rby1' uses robot_descriptions.rby1_description.",
    )
    parser.add_argument("--variant", choices=("sg2", "bg2"), default="sg2")
    parser.add_argument("--port", type=int, default=DEFAULT_VISER_PORT)
    parser.add_argument(
        "--ai-worker-root",
        type=str,
        default=None,
        help="Path to a local clone of https://github.com/ROBOTIS-GIT/ai_worker",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="Explicit follower URDF path. Overrides --ai-worker-root auto-resolution.",
    )
    parser.add_argument(
        "--collision-urdf",
        type=str,
        default=None,
        help="Optional collision-specialized URDF path. Defaults to the base URDF.",
    )
    parser.add_argument(
        "--print-resolved-paths",
        action="store_true",
        help="Print resolved URDF inputs before launching the app.",
    )
    parser.add_argument(
        "--controller-port",
        type=str,
        default=DEFAULT_SEER_CONTROLLER_PORT,
        help="Seer/xvisio serial port (default: /dev/ttyUSB0, same as examples/03_teleop_ik.py).",
    )
    return parser.parse_args()


def _resolve_rby1_urdf_path() -> Path:
    try:
        from robot_descriptions.rby1_description import URDF_PATH
    except ImportError as exc:
        raise RuntimeError(
            "RB-Y1 support requires robot_descriptions with rby1_description. "
            "Install the optional robot_descriptions package or run through the Pixi environment."
        ) from exc
    return Path(URDF_PATH).expanduser().resolve()


def _prepare_rby1_tip_frame_urdf(urdf_path: Path) -> Path:
    """Add fixed palm-tip frames at the center of the two gripper fingertips."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    link_names = {str(link.get("name", "")) for link in root.findall("link")}
    changed = False

    for mesh in root.findall(".//mesh"):
        filename = str(mesh.get("filename", "")).strip()
        if (
            filename
            and not filename.startswith(("package://", "file://"))
            and not Path(filename).is_absolute()
        ):
            candidate = (urdf_path.parent / filename).resolve()
            if candidate.is_file():
                mesh.set("filename", str(candidate))
                changed = True

    for side in ("left", "right"):
        parent = f"ee_{side}"
        tip = f"ee_{side}_tip"
        if parent not in link_names or tip in link_names:
            continue
        ET.SubElement(root, "link", name=tip)
        joint = ET.SubElement(root, "joint", name=f"{tip}_fixed", type="fixed")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=tip)
        ET.SubElement(joint, "origin", xyz="0 0 -0.073", rpy="0 0 0")
        changed = True

    if not changed:
        return urdf_path

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{urdf_path.stem}_tip_frames.urdf",
        prefix="embodik_rby1_",
        delete=False,
    )
    tree.write(tmp, encoding="unicode")
    tmp.close()
    return Path(tmp.name)


def _parse_xyz(text: str | None) -> np.ndarray:
    if not text:
        return np.zeros(3, dtype=float)
    return np.asarray([float(v) for v in text.split()], dtype=float)


def _format_xyz(values: np.ndarray) -> str:
    return " ".join(f"{float(v):.6g}" for v in np.asarray(values, dtype=float))


def _rpy_to_rotation(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _mesh_vertices(mesh_path: Path, scale: np.ndarray) -> np.ndarray | None:
    try:
        import trimesh

        mesh = trimesh.load_mesh(mesh_path, force="mesh")
    except Exception:
        return None
    vertices = np.asarray(getattr(mesh, "vertices", np.empty((0, 3))), dtype=float)
    if vertices.size == 0:
        return None
    return vertices * scale.reshape(1, 3)


def _mesh_split_boxes(
    mesh_path: Path,
    scale: np.ndarray,
    *,
    max_boxes: int = 2,
    split_extent_threshold: float = 0.11,
    min_extent: float = 0.012,
    padding: float = -0.012,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Approximate a visual mesh with a tiny set of local-frame box primitives."""
    vertices = _mesh_vertices(mesh_path, scale)
    if vertices is None:
        return []

    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    extent = hi - lo
    axis = int(np.argmax(extent))
    split_count = max_boxes if float(extent[axis]) >= split_extent_threshold else 1
    if split_count <= 1:
        center = 0.5 * (lo + hi)
        size = np.maximum(extent + 2.0 * padding, min_extent)
        return [(center, size)]

    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    edges = np.linspace(float(lo[axis]), float(hi[axis]), split_count + 1)
    for idx in range(split_count):
        lower = edges[idx] - 1e-9
        upper = edges[idx + 1] + 1e-9
        mask = (vertices[:, axis] >= lower) & (vertices[:, axis] <= upper)
        part = vertices[mask]
        if part.shape[0] < 4:
            continue
        part_lo = part.min(axis=0)
        part_hi = part.max(axis=0)
        part_extent = part_hi - part_lo
        center = 0.5 * (part_lo + part_hi)
        size = np.maximum(part_extent + 2.0 * padding, min_extent)
        boxes.append((center, size))

    if boxes:
        return boxes

    center = 0.5 * (lo + hi)
    size = np.maximum(extent + 2.0 * padding, min_extent)
    return [(center, size)]


def _prepare_rby1_collision_urdf(urdf_path: Path) -> Path:
    """Build a primitive collision URDF for RB-Y1 from fitted visual bounds.

    The public robot_descriptions RB-Y1 URDF currently ships visual meshes only.
    EmbodiK needs collision elements to construct self-collision pairs, so this
    generated URDF adds a small set of local-frame boxes per visual mesh. This
    keeps collision checks bounded while fitting long links more tightly than a
    single coarse box.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    changed = False
    box_padding = -0.012
    min_extent = 0.012

    for link in root.findall("link"):
        for collision in list(link.findall("collision")):
            link.remove(collision)
            changed = True
        for visual_index, visual in enumerate(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            filename = str(mesh.get("filename", "")).strip()
            if filename.startswith("file://"):
                filename = filename[len("file://") :]
            mesh_path = Path(filename)
            if not mesh_path.is_absolute():
                mesh_path = (urdf_path.parent / filename).resolve()
            if not mesh_path.is_file():
                continue

            scale = _parse_xyz(mesh.get("scale") or "1 1 1")
            if scale.size != 3:
                scale = np.ones(3, dtype=float)
            boxes = _mesh_split_boxes(mesh_path, scale, min_extent=min_extent, padding=box_padding)
            if not boxes:
                continue

            origin = visual.find("origin")
            visual_xyz = _parse_xyz(origin.get("xyz") if origin is not None else None)
            visual_rpy = _parse_xyz(origin.get("rpy") if origin is not None else None)
            visual_rotation = _rpy_to_rotation(visual_rpy)
            link_name = str(link.get("name", "link"))
            for box_index, (center, size) in enumerate(boxes):
                box_xyz = visual_xyz + visual_rotation @ center
                collision = ET.SubElement(
                    link,
                    "collision",
                    name=f"{link_name}_{visual_index}_{box_index}_box_collision",
                )
                ET.SubElement(
                    collision,
                    "origin",
                    xyz=_format_xyz(box_xyz),
                    rpy=_format_xyz(visual_rpy),
                )
                geometry = ET.SubElement(collision, "geometry")
                ET.SubElement(geometry, "box", size=_format_xyz(size))
                changed = True

    if not changed:
        return urdf_path

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{urdf_path.stem}_collision_split_boxes.urdf",
        prefix="embodik_rby1_",
        delete=False,
    )
    tree.write(tmp, encoding="unicode")
    tmp.close()
    return Path(tmp.name)


def main() -> None:
    args = parse_args()
    if args.robot == "rby1":
        base_urdf_path = (
            Path(args.urdf).expanduser().resolve() if args.urdf else _resolve_rby1_urdf_path()
        )
        urdf_path = _prepare_rby1_tip_frame_urdf(base_urdf_path)
        collision_urdf_path = (
            _prepare_rby1_tip_frame_urdf(Path(args.collision_urdf).expanduser().resolve())
            if args.collision_urdf
            else _prepare_rby1_collision_urdf(urdf_path)
        )
        variant_for_app = "rby1"
        common_app.COMMON_BIMANUAL_SUPPORT_CONTACT_FRAMES = ("wheel_l", "wheel_r", "base")
    else:
        urdf_path, collision_urdf_path = resolve_public_ai_worker_urdf_paths(
            variant=args.variant,
            ai_worker_root=args.ai_worker_root,
            urdf=args.urdf,
            collision_urdf=args.collision_urdf,
            allow_bundled_base_fallback=False,
        )
        variant_for_app = args.variant

    if args.print_resolved_paths:
        print(f"[bimanual] robot={args.robot}")
        print(f"[bimanual] variant={variant_for_app}")
        print(f"[bimanual] urdf={urdf_path}")
        print(f"[bimanual] collision_urdf={collision_urdf_path or urdf_path}")

    common_app.resolve_ffw_urdf_path = lambda _variant: urdf_path
    common_app.resolve_generated_ffw_collision_urdf_path = lambda _variant: collision_urdf_path
    common_app.parse_args = lambda: argparse.Namespace(variant=variant_for_app, port=args.port)
    common_app.COMMON_BIMANUAL_ENABLE_SEER_TELEOP = True
    common_app.COMMON_BIMANUAL_SEER_CONTROLLER_PORT = args.controller_port
    common_app.main()


if __name__ == "__main__":
    main()
