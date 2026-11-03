import re
import shlex
import tomllib
from pathlib import Path

import pytest

from scripts.validate_release_metadata import validate_release_metadata


def test_wheel_and_source_builds_use_the_development_eigen_baseline() -> None:
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    requirements = metadata["build-system"]["requires"]
    eigen = next(item for item in requirements if item.startswith("cmeel-eigen=="))
    major_minor = ".".join(eigen.split("==")[1].split(".")[:2])
    required = re.search(
        r"find_package\(Eigen3\s+([\d.]+)\s+REQUIRED", (root / "CMakeLists.txt").read_text()
    )
    assert required is not None and required[1] == major_minor
    pixi = tomllib.loads((root / "pixi.toml").read_text())
    assert pixi["dependencies"]["eigen"].startswith(f">={major_minor}.")
    for platform in ("linux", "macos"):
        before_build = metadata["tool"]["cibuildwheel"][platform]["before-build"]
        assert eigen in shlex.split(before_build)


def _write_release_metadata(
    root: Path,
    *,
    package_version: str = "0.20.17",
    pixi_version: str = "0.20.17",
    cmake_version: str = "0.20.17",
    release_date: str | None = "2026-07-22",
) -> None:
    (root / "pyproject.toml").write_text(f'[project]\nversion = "{package_version}"\n')
    (root / "pixi.toml").write_text(f'[workspace]\nversion = "{pixi_version}"\n')
    (root / "CMakeLists.txt").write_text(f"project(embodik VERSION {cmake_version})\n")
    heading = f"## [{package_version}] - {release_date}\n" if release_date is not None else ""
    (root / "CHANGELOG.md").write_text(f"# Changelog\n\n{heading}")


def test_validate_release_metadata_accepts_synchronized_dated_release(
    tmp_path: Path,
) -> None:
    _write_release_metadata(tmp_path)

    assert validate_release_metadata(tmp_path) == "0.20.17"


def test_validate_release_metadata_rejects_version_mismatch(tmp_path: Path) -> None:
    _write_release_metadata(tmp_path, pixi_version="0.20.16")

    with pytest.raises(ValueError, match="version mismatch"):
        validate_release_metadata(tmp_path)


def test_validate_release_metadata_rejects_cmake_version_mismatch(
    tmp_path: Path,
) -> None:
    _write_release_metadata(tmp_path, cmake_version="0.20.16")

    with pytest.raises(ValueError, match="version mismatch.*cmake=0.20.16"):
        validate_release_metadata(tmp_path)


def test_validate_release_metadata_requires_dated_changelog_entry(
    tmp_path: Path,
) -> None:
    _write_release_metadata(tmp_path, release_date=None)

    with pytest.raises(ValueError, match="exactly one dated 0.20.17"):
        validate_release_metadata(tmp_path)


def test_validate_release_metadata_rejects_invalid_release_date(
    tmp_path: Path,
) -> None:
    _write_release_metadata(tmp_path, release_date="2026-02-30")

    with pytest.raises(ValueError, match="invalid release date 2026-02-30"):
        validate_release_metadata(tmp_path)
