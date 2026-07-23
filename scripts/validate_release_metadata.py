#!/usr/bin/env python3
"""Validate synchronized package versions and the dated changelog entry."""

from __future__ import annotations

import re
import sys
import tomllib
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_toml(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def validate_release_metadata(project_root: Path = PROJECT_ROOT) -> str:
    """Return the release version after validating package metadata."""
    try:
        package_version = _read_toml(project_root / "pyproject.toml")["project"]["version"]
        pixi_version = _read_toml(project_root / "pixi.toml")["workspace"]["version"]
    except KeyError as exc:
        raise ValueError(f"missing release metadata key: {exc}") from exc

    if package_version != pixi_version:
        raise ValueError(f"version mismatch: pyproject={package_version}, pixi={pixi_version}")

    cmake_text = (project_root / "CMakeLists.txt").read_text()
    cmake_match = re.search(
        r"^\s*project\s*\(\s*embodik\s+VERSION\s+(?P<version>\d+\.\d+\.\d+)\s*\)",
        cmake_text,
        re.MULTILINE | re.IGNORECASE,
    )
    if cmake_match is None:
        raise ValueError("CMakeLists.txt must declare project(embodik VERSION ...)")
    cmake_version = cmake_match.group("version")
    if package_version != cmake_version:
        raise ValueError(
            f"version mismatch: pyproject={package_version}, cmake={cmake_version}"
        )

    changelog = (project_root / "CHANGELOG.md").read_text()
    heading_pattern = re.compile(
        rf"^## \[{re.escape(package_version)}\] - " r"(?P<date>\d{4}-\d{2}-\d{2})$",
        re.MULTILINE,
    )
    headings = list(heading_pattern.finditer(changelog))
    if len(headings) != 1:
        raise ValueError(
            f"CHANGELOG.md must contain exactly one dated {package_version} " "release entry"
        )

    release_date = headings[0].group("date")
    try:
        date.fromisoformat(release_date)
    except ValueError as exc:
        raise ValueError(f"CHANGELOG.md contains invalid release date {release_date}") from exc

    return package_version


def main() -> int:
    try:
        print(validate_release_metadata())
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
