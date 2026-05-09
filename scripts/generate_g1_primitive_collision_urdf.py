#!/usr/bin/env python3
"""Generate a G1 URDF with primitive box collision geometry.

The source Unitree G1 URDF uses detailed STL meshes for collision. That is
useful for offline geometry inspection but too expensive for an interactive
collision-aware IK loop. This script strips visuals and replaces each collision
mesh with a local AABB box computed from the original mesh vertices. Interactive
Viser examples still load visuals from the original robot description.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True, help="Source G1 URDF.")
    parser.add_argument("--output", type=Path, required=True, help="Derived URDF path.")
    parser.add_argument(
        "--padding",
        type=float,
        default=-0.015,
        help=(
            "Metres added to each side of each AABB dimension. The default "
            "slightly shrinks visual-mesh bounds because IK uses an explicit "
            "collision min-distance shell."
        ),
    )
    parser.add_argument(
        "--min-extent",
        type=float,
        default=0.012,
        help="Minimum box side length in metres after padding to avoid degenerate wire boxes.",
    )
    return parser.parse_args()


def _parse_xyz(value: str | None) -> np.ndarray:
    if not value:
        return np.zeros(3, dtype=float)
    parts = [float(x) for x in value.split()]
    if len(parts) != 3:
        return np.zeros(3, dtype=float)
    return np.asarray(parts, dtype=float)


def _parse_rpy(value: str | None) -> np.ndarray:
    if not value:
        return np.zeros(3, dtype=float)
    parts = [float(x) for x in value.split()]
    if len(parts) != 3:
        return np.zeros(3, dtype=float)
    return np.asarray(parts, dtype=float)


def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def _format_vec(vec: np.ndarray) -> str:
    return " ".join(f"{float(x):.8g}" for x in vec)


def _resolve_mesh_path(source_urdf: Path, filename: str) -> Path | None:
    if not filename:
        return None
    if filename.startswith("package://"):
        rel = filename[len("package://") :]
        for base in (source_urdf.parent, *source_urdf.parents):
            candidate = (base / rel).resolve()
            if candidate.is_file():
                return candidate
        return None
    path = Path(filename)
    if path.is_absolute():
        return path if path.is_file() else None
    candidate = (source_urdf.parent / path).resolve()
    return candidate if candidate.is_file() else None


def _mesh_scale(mesh_elem: ET.Element) -> np.ndarray:
    scale = _parse_xyz(mesh_elem.get("scale"))
    if not np.any(scale):
        return np.ones(3, dtype=float)
    return scale


def main() -> None:
    args = parse_args()
    source_urdf = args.urdf.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(source_urdf)
    root = tree.getroot()

    converted = 0
    skipped = 0
    for link in root.findall(".//link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)

    for joint in root.findall(".//joint"):
        for mimic in list(joint.findall("mimic")):
            joint.remove(mimic)

    for collision in root.findall(".//collision"):
        mesh_elem = collision.find("./geometry/mesh")
        geometry = collision.find("./geometry")
        if mesh_elem is None or geometry is None:
            continue
        filename = str(mesh_elem.get("filename", "")).strip()
        mesh_path = _resolve_mesh_path(source_urdf, filename)
        if mesh_path is None:
            skipped += 1
            continue

        mesh = trimesh.load_mesh(mesh_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
            skipped += 1
            continue
        vertices = np.asarray(mesh.vertices, dtype=float) * _mesh_scale(mesh_elem)
        bounds = np.asarray([vertices.min(axis=0), vertices.max(axis=0)], dtype=float)
        center = 0.5 * (bounds[0] + bounds[1])
        size = np.maximum(bounds[1] - bounds[0] + 2.0 * float(args.padding), float(args.min_extent))

        origin = collision.find("./origin")
        if origin is None:
            origin = ET.Element("origin")
            collision.insert(0, origin)
        old_xyz = _parse_xyz(origin.get("xyz"))
        old_rpy = _parse_rpy(origin.get("rpy"))
        origin.set("xyz", _format_vec(old_xyz + _rotation_from_rpy(old_rpy) @ center))
        origin.set("rpy", _format_vec(old_rpy))

        for child in list(geometry):
            geometry.remove(child)
        ET.SubElement(geometry, "box", {"size": _format_vec(size)})
        converted += 1

    tree.write(output, encoding="unicode")
    print(output)
    print(f"converted_collision_meshes={converted}")
    if skipped:
        print(f"skipped_collision_meshes={skipped}")


if __name__ == "__main__":
    main()
