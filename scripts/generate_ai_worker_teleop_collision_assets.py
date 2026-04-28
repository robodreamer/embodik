#!/usr/bin/env python3
"""Generate lighter collision assets for the ROBOTIS AI worker teleop example.

This leaves visuals untouched and only rewrites collision mesh references in a
derived URDF.  The goal is to reduce convex-collision refinement cost for the
interactive teleop example without mutating the source robot model assets.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import trimesh


@dataclass
class MeshReductionRecord:
    source: str
    output: str
    strategy: str
    original_faces: int
    reduced_faces: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-faces", type=int, default=1200)
    parser.add_argument("--min-face-reduction-ratio", type=float, default=0.85)
    return parser.parse_args()


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh for {path}, got {type(mesh).__name__}")
    return mesh


def _reduce_mesh(mesh: trimesh.Trimesh, max_faces: int, min_face_reduction_ratio: float) -> tuple[trimesh.Trimesh, str]:
    original_faces = int(len(mesh.faces))
    if original_faces <= max_faces:
        return mesh.copy(), "original"

    candidates: list[tuple[str, trimesh.Trimesh]] = []
    try:
        simplified = mesh.simplify_quadric_decimation(face_count=max_faces, aggression=10)
        if len(simplified.faces) < original_faces:
            candidates.append(("quadric", simplified))
    except Exception:
        pass

    try:
        hull = mesh.convex_hull
        if len(hull.faces) < original_faces:
            candidates.append(("convex_hull", hull))
    except Exception:
        pass

    if not candidates:
        return mesh.copy(), "original"

    strategy, reduced = min(candidates, key=lambda item: len(item[1].faces))
    if len(reduced.faces) >= int(original_faces * min_face_reduction_ratio):
        return mesh.copy(), "original"
    return reduced, strategy


def main() -> None:
    args = parse_args()
    source_urdf = args.urdf.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_mesh_root = output_dir / "collision_meshes"
    output_mesh_root.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(source_urdf)
    root = tree.getroot()
    reduced_cache: dict[str, str] = {}
    manifest: list[MeshReductionRecord] = []

    for mesh_elem in root.findall(".//collision/geometry/mesh"):
        filename = str(mesh_elem.get("filename", "")).strip()
        if not filename:
            continue
        source_mesh = (source_urdf.parent / filename).resolve()
        if not source_mesh.is_file():
            continue

        cache_key = str(source_mesh)
        if cache_key in reduced_cache:
            mesh_elem.set("filename", reduced_cache[cache_key])
            continue

        mesh = _load_mesh(source_mesh)
        reduced, strategy = _reduce_mesh(
            mesh,
            max_faces=int(args.max_faces),
            min_face_reduction_ratio=float(args.min_face_reduction_ratio),
        )

        rel_parent = Path(filename).parent
        out_parent = output_mesh_root / rel_parent
        out_parent.mkdir(parents=True, exist_ok=True)
        out_mesh = out_parent / source_mesh.name
        reduced.export(out_mesh)

        rel_out = out_mesh.relative_to(output_dir).as_posix()
        reduced_cache[cache_key] = rel_out
        mesh_elem.set("filename", rel_out)
        manifest.append(
            MeshReductionRecord(
                source=cache_key,
                output=str(out_mesh),
                strategy=strategy,
                original_faces=int(len(mesh.faces)),
                reduced_faces=int(len(reduced.faces)),
            )
        )

    derived_urdf = output_dir / f"{source_urdf.stem}_teleop_collision.urdf"
    tree.write(derived_urdf, encoding="unicode")
    (output_dir / "manifest.json").write_text(
        json.dumps([asdict(item) for item in manifest], indent=2),
        encoding="utf-8",
    )
    print(derived_urdf)
    print(f"reduced_meshes={len(manifest)}")


if __name__ == "__main__":
    main()
