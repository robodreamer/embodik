#!/usr/bin/env python3
"""Regression tests for examples copied out of the source tree."""

from __future__ import annotations

import importlib
import runpy
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("embodik.interactive_ik")


def _copy_examples_dir(tmp_path: Path) -> Path:
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    copied_examples_dir = tmp_path / "embodik_examples"
    shutil.copytree(
        examples_dir,
        copied_examples_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return copied_examples_dir


def _clear_example_helper_imports() -> None:
    for name in list(sys.modules):
        if name == "example_helpers" or name.startswith("example_helpers."):
            del sys.modules[name]
        if name == "examples" or name.startswith("examples."):
            del sys.modules[name]


def test_bimanual_example_imports_from_copied_examples_layout(monkeypatch, tmp_path) -> None:
    copied_examples_dir = _copy_examples_dir(tmp_path)
    _clear_example_helper_imports()
    monkeypatch.syspath_prepend(str(copied_examples_dir))

    module_globals = runpy.run_path(
        str(copied_examples_dir / "12_bimanual_whole_body_ik.py"),
        run_name="embodik_bimanual_import_check",
    )

    assert "main" in module_globals
    assert Path(
        module_globals["resolve_public_ai_worker_urdf_paths"].__code__.co_filename
    ).is_relative_to(copied_examples_dir)
    assert Path(module_globals["common_app"].__file__).is_relative_to(copied_examples_dir)

def test_unitree_g1_example_imports_from_copied_examples_layout(monkeypatch, tmp_path) -> None:
    copied_examples_dir = _copy_examples_dir(tmp_path)
    _clear_example_helper_imports()
    monkeypatch.syspath_prepend(str(copied_examples_dir))

    module_globals = runpy.run_path(
        str(copied_examples_dir / "13_unitree_g1_retargeting_ik.py"),
        run_name="embodik_g1_import_check",
    )

    assert "main" in module_globals
    assert Path(module_globals["_clip_q"].__code__.co_filename).is_relative_to(
        copied_examples_dir / "example_helpers"
    )
    assert Path(
        module_globals["resolve_g1_collision_urdf_path"].__code__.co_filename
    ).is_relative_to(copied_examples_dir / "example_helpers")


def test_ai_worker_sg2_resolves_bundled_urdf_before_network(monkeypatch, tmp_path) -> None:
    copied_examples_dir = _copy_examples_dir(tmp_path)
    _clear_example_helper_imports()
    monkeypatch.syspath_prepend(str(copied_examples_dir))
    paths = importlib.import_module("example_helpers.public_ai_worker_paths")

    def fail_download():
        raise AssertionError("network download should not be needed for bundled sg2 assets")

    monkeypatch.setattr(paths, "_download_public_repo_to_cache", fail_download)

    urdf_path, collision_urdf_path = paths.resolve_public_ai_worker_urdf_paths(variant="sg2")

    assert urdf_path.is_file()
    assert collision_urdf_path == urdf_path
    assert urdf_path.is_relative_to(
        copied_examples_dir / "assets" / "ai_worker" / "generated" / "sg2"
    )


def test_ai_worker_visual_entrypoint_can_require_public_visual_urdf(monkeypatch, tmp_path) -> None:
    copied_examples_dir = _copy_examples_dir(tmp_path)
    _clear_example_helper_imports()
    monkeypatch.syspath_prepend(str(copied_examples_dir))
    paths = importlib.import_module("example_helpers.public_ai_worker_paths")

    public_root = tmp_path / "ai_worker-main"
    visual_dir = public_root / "ffw_description" / "urdf" / "ffw_sg2_rev1_follower"
    visual_dir.mkdir(parents=True)
    visual_urdf = visual_dir / "ffw_sg2_rev1_follower.urdf"
    visual_urdf.write_text("<robot name='visual_sg2'/>", encoding="utf-8")

    monkeypatch.setattr(paths, "_download_public_repo_to_cache", lambda: public_root)

    urdf_path, collision_urdf_path = paths.resolve_public_ai_worker_urdf_paths(
        variant="sg2",
        allow_bundled_base_fallback=False,
    )

    assert urdf_path == visual_urdf.resolve()
    assert collision_urdf_path is not None
    assert collision_urdf_path != urdf_path
    assert collision_urdf_path.is_relative_to(
        copied_examples_dir / "assets" / "ai_worker" / "generated" / "sg2"
    )
