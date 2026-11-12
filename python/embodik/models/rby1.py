"""Rainbow RB-Y1 A/M fixed-base IK descriptions from a pinned official SDK.

Assets are downloaded only by ``resolve_rby1_urdf``. Wheels, head and fingers
are fixed for torso/arm IK; the original SDK files remain untouched.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from embodik import RobotModel

logger = logging.getLogger(__name__)
SDK_RELEASE = "v0.10.1"
SDK_COMMIT = "300a585c297d1bdf8554312c3a1659610a9823f5"
SDK_URL = "https://github.com/RainbowRobotics/rby1-sdk.git"
MODEL_PATHS = {
    "a": "models/rby1a/urdf/model_v1.2.urdf",
    "m": "models/rby1m/urdf/model_v1.3.urdf",
}
READY_JOINTS = {f"torso_{i}": v for i, v in enumerate((0, 0.5236, -1.0472, 0.5236, 0, 0))}
for _side, _shoulder in (("left", 0.5236), ("right", -0.5236)):
    READY_JOINTS.update(
        {
            f"{_side}_arm_{i}": v
            for i, v in enumerate((-0.7854, _shoulder, 0, -1.5708, 0, 0.7854, 0))
        }
    )


def _cache_root() -> Path:
    return Path(os.environ.get("EMBODIK_RBY1_CACHE", "~/.cache/embodik/rby1-sdk")).expanduser()


def _sdk_root() -> Path:
    cache = _cache_root()
    target = cache / SDK_RELEASE
    if not target.exists():
        git = shutil.which("git")
        if git is None:
            raise RuntimeError(
                "RB-Y1 asset download needs git; alternatively set EMBODIK_RBY1_ROOT."
            )
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".rby1-", dir=cache) as tmp:
            staging = Path(tmp) / "sdk"
            logger.info("Downloading official RB-Y1 assets at %s", SDK_RELEASE)
            try:
                subprocess.run(
                    [
                        git,
                        "clone",
                        "--depth",
                        "1",
                        "--branch",
                        SDK_RELEASE,
                        "--filter=blob:none",
                        "--sparse",
                        SDK_URL,
                        str(staging),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                subprocess.run(
                    [
                        git,
                        "-C",
                        str(staging),
                        "sparse-checkout",
                        "set",
                        "models/rby1a/urdf",
                        "models/rby1m/urdf",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                revision = subprocess.check_output(
                    [git, "-C", str(staging), "rev-parse", "HEAD"], text=True
                ).strip()
                if revision != SDK_COMMIT:
                    raise RuntimeError(f"RB-Y1 SDK tag resolved to unexpected commit {revision}")
                try:
                    staging.rename(target)
                except OSError:
                    if not target.is_dir():
                        raise
            except (subprocess.SubprocessError, OSError) as exc:
                raise RuntimeError(
                    "Could not cache official RB-Y1 assets. Set EMBODIK_RBY1_ROOT "
                    "to an SDK checkout for offline use."
                ) from exc
    # This cache is created as a detached checkout, never silently refreshed.
    head = target / ".git" / "HEAD"
    if not head.is_file() or head.read_text().strip() != SDK_COMMIT:
        raise RuntimeError(
            "RB-Y1 cache is not the pinned SDK checkout; set EMBODIK_RBY1_ROOT explicitly."
        )
    return target


def _capsule_collisions(collision: ET.Element, index: int) -> list[ET.Element]:
    """Represent the SDK's Z-axis capsule exactly as three URDF primitives."""
    capsule = collision.find("geometry/capsule")
    assert capsule is not None
    radius, length = float(capsule.attrib["radius"]), float(capsule.attrib["length"])
    if not math.isfinite(radius) or not math.isfinite(length) or radius <= 0 or length <= 0:
        raise ValueError("RB-Y1 capsule dimensions must be finite and positive")
    origin = collision.find("origin")
    xyz = [
        float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()
    ]
    rpy = [
        float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()
    ]
    roll, pitch, yaw = rpy
    axis = (
        math.cos(yaw) * math.sin(pitch) * math.cos(roll) + math.sin(yaw) * math.sin(roll),
        math.sin(yaw) * math.sin(pitch) * math.cos(roll) - math.cos(yaw) * math.sin(roll),
        math.cos(pitch) * math.cos(roll),
    )
    result = []
    name = collision.get("name", f"capsule_{index}")
    for part, offset in (
        ("cylinder", 0),
        ("sphere_minus", -length / 2),
        ("sphere_plus", length / 2),
    ):
        element = ET.Element("collision", name=f"{name}_{part}")
        ET.SubElement(
            element,
            "origin",
            xyz=" ".join(str(x + offset * a) for x, a in zip(xyz, axis)),
            rpy=" ".join(map(str, rpy)),
        )
        geometry = ET.SubElement(element, "geometry")
        if part == "cylinder":
            ET.SubElement(geometry, "cylinder", radius=str(radius), length=str(length))
        else:
            ET.SubElement(geometry, "sphere", radius=str(radius))
        result.append(element)
    return result


def prepare_rby1_ik_urdf(source: Path) -> Path:
    """Cache a standards-compatible torso/arm model without altering source assets.

    Only torso_0..5 and left/right_arm_0..6 remain actuated. Capsule volumes,
    visual meshes, link inertias and the kinematics of those joints are retained.
    SDK collision-type masks and mobile-base control semantics are not imported.
    """
    source = source.expanduser().resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    for mobile in root.findall("mobile"):
        root.remove(mobile)
    for joint in root.findall("joint"):
        if joint.get("type") != "fixed" and joint.get("name") not in READY_JOINTS:
            joint.set("type", "fixed")
            for child in list(joint):
                if child.tag in ("limit", "mimic", "dynamics"):
                    joint.remove(child)
    active_names = {j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"}
    if active_names != set(READY_JOINTS):
        raise ValueError("RB-Y1 description must contain all 20 torso/arm joints")
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith("file://"):
            filename = filename[7:]
        path = Path(filename)
        if not path.is_absolute():
            path = source.parent / path
        if not path.is_file():
            raise FileNotFoundError(f"RB-Y1 mesh is missing: {path}")
        mesh.set("filename", str(path.resolve()))
    for link in root.findall("link"):
        for index, collision in enumerate(list(link.findall("collision"))):
            if collision.find("geometry/capsule") is not None:
                link.remove(collision)
                link.extend(_capsule_collisions(collision, index))
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    digest = hashlib.sha256(payload).hexdigest()
    output = _cache_root() / "generated" / f"{source.stem}-{digest}.urdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.is_file() or output.read_bytes() != payload:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
            tmp = Path(stream.name)
            stream.write(payload)
        tmp.replace(output)
    return output


def resolve_rby1_urdf(variant: str = "a", root: str | Path | None = None) -> Path:
    """Resolve A v1.2 or M v1.3 from SDK v0.10.1, then prepare fixed-base IK.

    ``root`` or EMBODIK_RBY1_ROOT supplies an offline SDK checkout. Otherwise,
    clone the pinned official release into EMBODIK_RBY1_CACHE on first use.
    """
    variant = variant.lower()
    if variant not in MODEL_PATHS:
        raise ValueError(f"Unknown RB-Y1 variant {variant!r}; expected 'a' or 'm'")
    explicit = root if root is not None else os.environ.get("EMBODIK_RBY1_ROOT")
    sdk = Path(explicit).expanduser().resolve() if explicit else _sdk_root()
    source = sdk / MODEL_PATHS[variant]
    if not source.is_file():
        raise FileNotFoundError(f"RB-Y1 {variant.upper()} description is missing: {source}")
    return prepare_rby1_ik_urdf(source)


def ready_configuration(robot: RobotModel) -> np.ndarray:
    """Build the RB-Y1 ready pose by joint name rather than URDF ordering."""
    q = robot.neutral_configuration().copy()
    for name, value in READY_JOINTS.items():
        q[robot.get_joint_config_index(name)] = value
    return q
