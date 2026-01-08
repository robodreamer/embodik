#!/usr/bin/env python3
"""Version management script for embodik package.

Usage:
    python scripts/version.py                    # Show current version
    python scripts/version.py 0.2.0              # Set version to 0.2.0
    python scripts/version.py --bump patch       # Bump patch version (0.1.0 -> 0.1.1)
    python scripts/version.py --bump minor       # Bump minor version (0.1.0 -> 0.2.0)
    python scripts/version.py --bump major       # Bump major version (0.1.0 -> 1.0.0)
"""

import argparse
import re
import sys
from pathlib import Path


def get_version():
    """Get current version from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    with open(pyproject_path, "r") as f:
        content = f.read()

    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if match:
        return match.group(1)

    raise ValueError("Could not find version in pyproject.toml")


def set_version(new_version: str):
    """Set version in pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    with open(pyproject_path, "r") as f:
        content = f.read()

    # Replace version line
    content = re.sub(
        r'^version\s*=\s*["\'][^"\']+["\']',
        f'version = "{new_version}"',
        content,
        flags=re.MULTILINE
    )

    with open(pyproject_path, "w") as f:
        f.write(content)

    print(f"✓ Version updated to {new_version}")


def bump_version(bump_type: str):
    """Bump version by type (patch, minor, major)."""
    current_version = get_version()

    # Parse version
    parts = current_version.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {current_version}")

    major, minor, patch = map(int, parts)

    # Bump version
    if bump_type == "patch":
        patch += 1
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    else:
        raise ValueError(f"Invalid bump type: {bump_type}. Use 'patch', 'minor', or 'major'")

    new_version = f"{major}.{minor}.{patch}"
    set_version(new_version)
    return new_version


def main():
    parser = argparse.ArgumentParser(
        description="Manage embodik package version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "version",
        nargs="?",
        help="Set version to this value (e.g., 0.2.0)"
    )

    parser.add_argument(
        "--bump",
        choices=["patch", "minor", "major"],
        help="Bump version by type"
    )

    args = parser.parse_args()

    try:
        if args.version:
            # Set specific version
            set_version(args.version)
        elif args.bump:
            # Bump version
            new_version = bump_version(args.bump)
            print(f"✓ Version bumped to {new_version}")
        else:
            # Show current version
            version = get_version()
            print(f"Current version: {version}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
