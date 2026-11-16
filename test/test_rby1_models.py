"""Deterministic SDK compatibility checks; real assets are explicitly opt-in."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import embodik as eik
from embodik.models.rby1 import (
    MODEL_PATHS,
    READY_JOINTS,
    prepare_rby1_ik_urdf,
    ready_configuration,
    resolve_rby1_urdf,
)


def _sdk_fixture(tmp_path: Path) -> Path:
    root = ET.Element("robot", name="rby1_test")
    ET.SubElement(root, "link", name="base")
    parent = "base"
    for index, name in enumerate([*READY_JOINTS, "right_wheel"]):
        child = f"link_{index}"
        ET.SubElement(root, "link", name=child)
        joint = ET.SubElement(root, "joint", name=name, type="revolute")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)
        ET.SubElement(joint, "axis", xyz="0 0 1")
        ET.SubElement(joint, "limit", lower="-3", upper="3", effort="10", velocity="2")
        parent = child
    capsule = ET.SubElement(root.find("link"), "collision", name="guard")
    ET.SubElement(capsule, "origin", xyz="1 2 3", rpy="0 1.5707963267948966 0")
    geometry = ET.SubElement(capsule, "geometry")
    ET.SubElement(geometry, "capsule", radius="0.1", length="0.4", coltype="1")
    source = tmp_path / "model.urdf"
    ET.ElementTree(root).write(source)
    return source


def test_capsule_conversion_preserves_rotated_volume_and_source(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBODIK_RBY1_CACHE", str(tmp_path / "cache"))
    source = _sdk_fixture(tmp_path)
    original = source.read_bytes()
    generated = prepare_rby1_ik_urdf(source)
    assert source.read_bytes() == original
    root = ET.parse(generated).getroot()
    assert root.find(".//capsule") is None
    cylinder = root.find(".//cylinder")
    assert cylinder.attrib == {"radius": "0.1", "length": "0.4"}
    sphere_collisions = [
        c for c in root.findall(".//collision") if c.find("geometry/sphere") is not None
    ]
    centers = [[float(v) for v in c.find("origin").get("xyz").split()] for c in sphere_collisions]
    np.testing.assert_allclose(centers, [[0.8, 2, 3], [1.2, 2, 3]], atol=1e-14)
    assert root.find("joint[@name='right_wheel']").get("type") == "fixed"
    assert root.find("joint[@name='right_wheel']/limit") is None
    model = eik.RobotModel(str(generated), floating_base=False)
    assert model.nq == model.nv == 20
    assert set(model.get_joint_names()) == set(READY_JOINTS)
    assert model.has_collision_geometry()
    assert prepare_rby1_ik_urdf(source) == generated


def test_missing_mesh_and_missing_torso_are_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBODIK_RBY1_CACHE", str(tmp_path / "cache"))
    source = _sdk_fixture(tmp_path)
    tree = ET.parse(source)
    visual = ET.SubElement(tree.getroot().find("link"), "visual")
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "mesh", filename="missing.stl")
    tree.write(source)
    with pytest.raises(FileNotFoundError, match="mesh"):
        prepare_rby1_ik_urdf(source)
    tree.getroot().remove(tree.getroot().find("joint[@name='torso_0']"))
    tree.write(source)
    with pytest.raises(ValueError, match="20 torso/arm"):
        prepare_rby1_ik_urdf(source)


def test_explicit_missing_checkout_does_not_download(tmp_path, monkeypatch):
    import embodik.models.rby1 as adapter

    monkeypatch.setattr(adapter, "_sdk_root", lambda: pytest.fail("unexpected download"))
    with pytest.raises(FileNotFoundError, match="description is missing"):
        resolve_rby1_urdf(root=tmp_path)
    with pytest.raises(ValueError, match="variant"):
        resolve_rby1_urdf("invalid", root=tmp_path)


def test_wrong_cached_sdk_commit_fails_without_fetch(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBODIK_RBY1_CACHE", str(tmp_path))
    head = tmp_path / "v0.10.1/.git/HEAD"
    head.parent.mkdir(parents=True)
    head.write_text("unqualified checkout")
    with pytest.raises(RuntimeError, match="pinned SDK checkout"):
        resolve_rby1_urdf()


def test_rby1_preset_aliases_official_m():
    from examples.utils.robot_models import get_robot_preset

    alias = get_robot_preset("rby1")
    explicit = get_robot_preset("rby1m")
    assert alias["rby1_variant"] == explicit["rby1_variant"] == "m"
    assert alias["display_name"] == explicit["display_name"]
    assert get_robot_preset("rby1a")["rby1_variant"] == "a"


def test_bimanual_default_resolver_uses_official_m(monkeypatch, tmp_path):
    import importlib.util

    import embodik.models.rby1 as adapter

    script = Path(__file__).resolve().parents[1] / "examples/06_bimanual_whole_body_ik.py"
    spec = importlib.util.spec_from_file_location("rby1_alias_example", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def resolve(variant):
        calls.append(variant)
        return tmp_path / f"{variant}.urdf"

    monkeypatch.setattr(adapter, "resolve_rby1_urdf", resolve)
    assert module._resolve_rby1_urdf_path() == tmp_path / "m.urdf"
    assert module._resolve_rby1_urdf_path("a") == tmp_path / "a.urdf"
    assert calls == ["m", "a"]


@pytest.mark.parametrize("variant", sorted(MODEL_PATHS))
def test_official_assets_load_track_and_keep_collision_geometry(variant, monkeypatch):
    root = os.environ.get("EMBODIK_RBY1_TEST_ROOT")
    if not root:
        pytest.skip("set EMBODIK_RBY1_TEST_ROOT to a v0.10.1 SDK checkout (no CI download)")
    monkeypatch.setenv("EMBODIK_RBY1_ROOT", root)
    path = resolve_rby1_urdf(variant, root=root)
    from examples.utils.robot_models import resolve_robot_configuration

    config = resolve_robot_configuration("rby1" + variant)
    assert config["urdf_path"] == path
    if variant == "m":
        assert resolve_robot_configuration("rby1")["urdf_path"] == path
    model = eik.RobotModel(str(path), floating_base=False)
    assert model.nq == model.nv == 20
    assert model.has_collision_geometry()
    q = ready_configuration(model)
    lower, upper = model.get_joint_limits()
    assert np.all(q >= lower) and np.all(q <= upper)
    model.update_configuration(q)
    solver = eik.KinematicsSolver(model)
    solver.dt = 0.01
    tasks = []
    for side in ("left", "right"):
        frame = f"ee_{side}"
        task = solver.add_frame_task(side, frame)
        task.weight = 5.0  # 5/s pose-error feedback; settle a small goal within 1.6 s.
        pose = model.get_frame_pose(frame)
        goal = np.asarray(pose.translation) + np.array([0.015, 0, -0.01])
        task.set_target_pose(goal, np.asarray(pose.rotation))
        tasks.append((frame, goal))
    for _ in range(160):
        result = solver.solve_velocity(q)
        assert result.status == eik.SolverStatus.SUCCESS
        q = model.integrate(q, np.asarray(result.joint_velocities) * solver.dt)
    model.update_configuration(q)
    for frame, goal in tasks:
        assert np.linalg.norm(np.asarray(model.get_frame_pose(frame).translation) - goal) < 0.001
    # Visual parsing must preserve the same actuated set and resolve actual meshes.
    from yourdfpy import URDF

    visual = URDF.load(str(path), load_meshes=True)
    assert set(visual.actuated_joint_names) == set(model.get_joint_names())
    assert visual.scene.geometry
