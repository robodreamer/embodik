"""Small CLI utilities for embodik.

Currently provides:
- `embodik-sanitize-env`: help users avoid LD_LIBRARY_PATH conflicts between
  pip-installed `pin` wheels and locally-built Pinocchio installs.
- `embodik-examples`: locate or copy example scripts for pip-installed users.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


_DEFAULT_REMOVE_SUBSTRINGS = [
    # Common local pinocchio build install path used during development
    "/pinocchio/install",
    "/pinocchio/install-fcl",
]


@dataclass(frozen=True)
class SanitizeResult:
    original: str
    sanitized: Optional[str]  # None means "unset"
    removed: List[str]


def _split_path_list(value: str) -> List[str]:
    return [p for p in value.split(":") if p]


def _join_path_list(parts: List[str]) -> str:
    return ":".join(parts)


def _sanitize_ld_library_path(
    value: str,
    *,
    remove_substrings: List[str],
    unset_if_empty: bool,
) -> SanitizeResult:
    parts = _split_path_list(value)
    kept: List[str] = []
    removed: List[str] = []

    for p in parts:
        if any(s in p for s in remove_substrings):
            removed.append(p)
        else:
            kept.append(p)

    if not kept and unset_if_empty:
        return SanitizeResult(original=value, sanitized=None, removed=removed)

    return SanitizeResult(original=value, sanitized=_join_path_list(kept), removed=removed)


def sanitize_env(argv: Optional[List[str]] = None) -> int:
    """Entry point for `embodik-sanitize-env`."""
    parser = argparse.ArgumentParser(
        prog="embodik-sanitize-env",
        description=(
            "Sanitize environment variables that commonly break pip-installed Pinocchio (`pin`).\n\n"
            "Typical symptom: `ImportError: libboost_*.so...` unless you `unset LD_LIBRARY_PATH`.\n"
            "Cause: LD_LIBRARY_PATH points at a local Pinocchio install (e.g. pinocchio/install-fcl/lib),\n"
            "which overrides the pip wheel's bundled shared libraries.\n\n"
            "Recommended usage:\n"
            "  eval \"$(embodik-sanitize-env --shell)\"\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Print shell commands to apply the sanitization in the current shell (use with eval).",
    )
    parser.add_argument(
        "--unset-all",
        action="store_true",
        help="Unset LD_LIBRARY_PATH entirely (more aggressive, but simplest).",
    )
    parser.add_argument(
        "--remove",
        action="append",
        default=[],
        help=(
            "Substring to remove from LD_LIBRARY_PATH entries. Can be repeated. "
            f"Default: {', '.join(_DEFAULT_REMOVE_SUBSTRINGS)}"
        ),
    )
    parser.add_argument(
        "--unset-if-empty",
        action="store_true",
        default=True,
        help="If LD_LIBRARY_PATH becomes empty after removal, unset it (default: true).",
    )
    parser.add_argument(
        "--keep-if-empty",
        dest="unset_if_empty",
        action="store_false",
        help="If LD_LIBRARY_PATH becomes empty after removal, keep it set to empty string.",
    )

    args = parser.parse_args(argv)

    current = os.environ.get("LD_LIBRARY_PATH", "")
    remove_substrings = args.remove or _DEFAULT_REMOVE_SUBSTRINGS

    if args.unset_all:
        result = SanitizeResult(original=current, sanitized=None, removed=_split_path_list(current))
    else:
        result = _sanitize_ld_library_path(
            current, remove_substrings=remove_substrings, unset_if_empty=args.unset_if_empty
        )

    if args.shell:
        # Emit shell code only. Keep it minimal and safe to eval.
        if result.sanitized is None:
            print("unset LD_LIBRARY_PATH")
        else:
            # Quote safely for shell
            print(f"export LD_LIBRARY_PATH={shlex.quote(result.sanitized)}")
        return 0

    # Human-readable output
    print("### embodik-sanitize-env")
    print(f"- LD_LIBRARY_PATH (original): {result.original!r}")
    if result.removed:
        print("- Removed entries:")
        for p in result.removed:
            print(f"  - {p}")
    else:
        print("- Removed entries: (none)")

    if result.sanitized is None:
        print("- LD_LIBRARY_PATH (sanitized): (unset)")
    else:
        print(f"- LD_LIBRARY_PATH (sanitized): {result.sanitized!r}")

    print("")
    print("To apply in your current shell:")
    print('  eval "$(embodik-sanitize-env --shell)"')

    return 0


def _find_examples_dir() -> Optional[Path]:
    """Find the examples directory in the installed package."""
    # Try to find examples relative to the package
    try:
        import embodik
        package_dir = Path(embodik.__file__).parent.parent.parent
        examples_dir = package_dir / "examples"
        if examples_dir.exists() and examples_dir.is_dir():
            return examples_dir
    except Exception:
        pass

    # Fallback: check common installation locations
    for site_packages in sys.path:
        try:
            candidate = Path(site_packages) / "embodik" / "examples"
            if candidate.exists() and candidate.is_dir():
                return candidate
        except Exception:
            continue

    return None


def examples_cmd(argv: Optional[List[str]] = None) -> int:
    """Entry point for `embodik-examples`."""
    parser = argparse.ArgumentParser(
        prog="embodik-examples",
        description=(
            "Locate or copy EmbodiK example scripts.\n\n"
            "Examples are included in the pip package and can be run directly.\n"
            "Use --copy to copy them to a local directory for editing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--copy",
        metavar="DEST",
        type=str,
        help="Copy examples to the specified directory (default: ./embodik_examples)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available example scripts.",
    )

    args = parser.parse_args(argv)

    examples_dir = _find_examples_dir()

    if examples_dir is None:
        print("ERROR: Could not find examples directory in installed package.", file=sys.stderr)
        print("Examples may not be included in this installation.", file=sys.stderr)
        print("Try: pip install --force-reinstall embodik", file=sys.stderr)
        return 1

    if args.list:
        print(f"Examples directory: {examples_dir}")
        print("\nAvailable examples:")
        for py_file in sorted(examples_dir.glob("*.py")):
            if py_file.name != "__init__.py":
                print(f"  - {py_file.name}")
        return 0

    if args.copy:
        dest = Path(args.copy).expanduser().resolve()
    else:
        dest = Path.cwd() / "embodik_examples"

    try:
        if dest.exists():
            if not dest.is_dir():
                print(f"ERROR: {dest} exists but is not a directory", file=sys.stderr)
                return 1
            response = input(f"Directory {dest} already exists. Overwrite? [y/N]: ")
            if response.lower() != "y":
                print("Cancelled.")
                return 0
            shutil.rmtree(dest)

        shutil.copytree(examples_dir, dest)
        print(f"✓ Copied examples to {dest}")
        print(f"\nTo run an example:")
        print(f"  cd {dest}")
        print(f"  python 01_basic_ik_simple.py")
        return 0

    except Exception as e:
        print(f"ERROR: Failed to copy examples: {e}", file=sys.stderr)
        return 1

    # Default: just show location
    print(f"Examples directory: {examples_dir}")
    print("\nTo run an example:")
    print(f"  python {examples_dir}/01_basic_ik_simple.py")
    print("\nTo copy examples to a local directory:")
    print("  embodik-examples --copy")
    return 0


def main() -> None:
    # Check if we're being called as examples command
    if len(sys.argv) > 1 and sys.argv[1] == "examples":
        sys.argv = sys.argv[1:]  # Remove 'examples' from argv
        sys.argv[0] = "embodik-examples"  # Update program name
        raise SystemExit(examples_cmd())
    else:
        raise SystemExit(sanitize_env())


if __name__ == "__main__":
    main()

