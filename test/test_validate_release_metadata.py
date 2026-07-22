from pathlib import Path

import pytest

from scripts.validate_release_metadata import validate_release_metadata


def _write_release_metadata(
    root: Path,
    *,
    package_version: str = "0.20.17",
    pixi_version: str = "0.20.17",
    release_date: str | None = "2026-07-22",
) -> None:
    (root / "pyproject.toml").write_text(f'[project]\nversion = "{package_version}"\n')
    (root / "pixi.toml").write_text(f'[workspace]\nversion = "{pixi_version}"\n')
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
