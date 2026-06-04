#!/usr/bin/env python3
"""Regression tests for examples copied out of the source tree."""

from __future__ import annotations

import importlib
import re
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
        str(copied_examples_dir / "06_bimanual_whole_body_ik.py"),
        run_name="embodik_bimanual_import_check",
    )

    assert "main" in module_globals
    assert Path(
        module_globals["resolve_public_ai_worker_urdf_paths"].__code__.co_filename
    ).is_relative_to(copied_examples_dir)
    assert Path(module_globals["common_app"].__file__).is_relative_to(copied_examples_dir)


def test_public_viser_examples_use_shared_default_port() -> None:
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    hardcoded_port_patterns = (
        re.compile(r"--port[\s\S]{0,100}default\s*=\s*80[0-9]{2}"),
        re.compile(r"port\s*=\s*80[0-9]{2}"),
        re.compile(r"localhost:80[0-9]{2}"),
    )
    allowed_files = {
        examples_dir / "example_helpers" / "ik_common.py",
    }

    offenders: list[str] = []
    for path in sorted(examples_dir.rglob("*.py")):
        if path in allowed_files or "assets" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        for pattern in hardcoded_port_patterns:
            for match in pattern.finditer(source):
                line_no = source[: match.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(examples_dir)}:{line_no}: {match.group(0)!r}")

    assert offenders == []


def test_unitree_g1_example_imports_from_copied_examples_layout(monkeypatch, tmp_path) -> None:
    copied_examples_dir = _copy_examples_dir(tmp_path)
    _clear_example_helper_imports()
    monkeypatch.syspath_prepend(str(copied_examples_dir))

    module_globals = runpy.run_path(
        str(copied_examples_dir / "07_unitree_g1_retargeting_ik.py"),
        run_name="embodik_g1_import_check",
    )

    assert "main" in module_globals
    assert Path(module_globals["_clip_q"].__code__.co_filename).is_relative_to(
        copied_examples_dir / "example_helpers"
    )
    assert Path(
        module_globals["resolve_g1_collision_urdf_path"].__code__.co_filename
    ).is_relative_to(copied_examples_dir / "example_helpers")

    g1_model_utils = importlib.import_module("example_helpers.g1_model_utils")
    visual_urdf_path = g1_model_utils.resolve_g1_urdf_path()
    collision_urdf_path = g1_model_utils.resolve_g1_collision_urdf_path()

    assert visual_urdf_path.is_file()
    assert visual_urdf_path.is_relative_to(copied_examples_dir / "assets" / "g1" / "visual")
    assert collision_urdf_path.is_file()
    assert collision_urdf_path.is_relative_to(copied_examples_dir / "assets" / "g1" / "generated")


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


def test_examples_copy_hides_internal_harnesses(monkeypatch, tmp_path) -> None:
    copied_examples_dir = _copy_examples_dir(tmp_path)
    from embodik import cli

    dest = tmp_path / "public_examples"
    monkeypatch.setattr(cli, "_find_examples_dir", lambda: copied_examples_dir)

    assert cli.examples_cmd(["--copy", str(dest)]) == 0
    assert (dest / "01_basic_ik_simple.py").is_file()
    assert not (dest / "harnesses").exists()


def test_package_configs_hide_internal_sources_from_public_examples() -> None:
    root = Path(__file__).resolve().parents[1]
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert 'PATTERN "harnesses" EXCLUDE' in cmake
    assert '"examples/harnesses/**"' in pyproject
