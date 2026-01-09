"""Runtime dependency helpers.

This module exists to make pip-installed binary dependencies (notably Pinocchio via the `pin`
PyPI wheels) import more robust.

Problem: some wheel layouts (e.g. cmeel-based) ship required `.so` dependencies under a
`site-packages/cmeel.prefix/lib/` directory that is *not* always discoverable by the dynamic
loader when importing extension modules. This can result in errors like:

- ImportError: libboost_filesystem.so.X.Y.Z: cannot open shared object file
- ImportError: libboost_python310.so.X.Y.Z: cannot open shared object file

We work around this by preloading likely required Boost libraries via `ctypes.CDLL(..., RTLD_GLOBAL)`
from the cmeel prefix lib directory before importing `pinocchio`.
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable, Optional

_PINOCCHIO_MODULE: Optional[ModuleType] = None


def ensure_boost_soname_compat(target_versions: tuple[str, ...] = ("1.87.0",)) -> None:
    """Best-effort create compatibility symlinks for common Boost SONAMEs in cmeel.prefix/lib.

    Some wheel stacks end up with binaries linked against an older Boost SONAME
    (e.g. `libboost_filesystem.so.1.87.0`) while only a newer one is present
    (e.g. `libboost_filesystem.so.1.89.0`). Creating a symlink in the same lib directory
    can unblock the dynamic loader.
    """
    bases = [
        "libboost_system.so",
        "libboost_filesystem.so",
        "libboost_serialization.so",
        # Boost.Python varies by python version (310/311/312...), so match broadly.
        "libboost_python",
    ]

    for lib_dir in _iter_cmeel_prefix_lib_dirs():
        for ver in target_versions:
            for base in bases:
                if base == "libboost_python":
                    # Any python variant, e.g. libboost_python310.so.1.87.0
                    for p in lib_dir.glob("libboost_python*.so.*"):
                        # derive a missing soname for this variant only
                        variant = p.name.split(".so.")[0] + ".so"
                        missing = f"{variant}.{ver}"
                        _maybe_create_soname_symlink(lib_dir, missing)
                else:
                    missing = f"{base}.{ver}"
                    _maybe_create_soname_symlink(lib_dir, missing)


def _maybe_create_soname_symlink(lib_dir: Path, missing_soname: str) -> None:
    """Best-effort: create a symlink for a missing Boost SONAME to an available one.

    This is a pragmatic workaround for cases where wheels end up with a Boost minor-version
    mismatch (e.g. linked against 1.87 but only 1.89 is present). It is not guaranteed to be
    ABI-safe in all cases, but it unblocks many environments until upstream wheel metadata
    and binaries are consistent.
    """
    if not missing_soname.startswith("libboost_"):
        return
    dest = lib_dir / missing_soname
    if dest.exists():
        return

    # Determine base prefix, e.g. libboost_filesystem.so or libboost_python310.so
    if ".so." not in missing_soname:
        return
    base = missing_soname.split(".so.")[0] + ".so"

    # Pick the "best" available candidate (highest version-like suffix)
    candidates = sorted(
        (p for p in lib_dir.glob(base + ".*") if p.is_file()),
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        return

    target = candidates[0]
    try:
        os.symlink(target.name, dest)  # relative symlink within the same directory
    except Exception:
        return


def _iter_cmeel_prefix_lib_dirs() -> Iterable[Path]:
    """Yield candidate `cmeel.prefix/lib` directories that may contain shared libs."""
    # Common layout: <site-packages>/cmeel.prefix/lib
    for entry in sys.path:
        try:
            base = Path(entry)
        except Exception:
            continue
        if not base.exists() or not base.is_dir():
            continue
        candidate = base / "cmeel.prefix" / "lib"
        if candidate.is_dir():
            yield candidate

    # Fallback: if cmeel is importable, search relative to its install location
    try:
        import cmeel  # type: ignore

        site_packages = Path(cmeel.__file__).resolve().parent.parent
        candidate = site_packages / "cmeel.prefix" / "lib"
        if candidate.is_dir():
            yield candidate
    except Exception:
        return


def preload_cmeel_pinocchio_stack() -> None:
    """Preload Pinocchio/Boost shared libraries shipped in the cmeel prefix.

    Important: the dynamic loader reads LD_LIBRARY_PATH at process start. If a user shell has
    LD_LIBRARY_PATH pointing at another Pinocchio build, the `pin` wheel can accidentally
    load those libs (and their Boost dependencies). Preloading the wheel's own libs by absolute
    path helps ensure the correct ones are already resident, so later imports reuse them.
    """
    # Order matters: preload Boost first, then core deps, then pinocchio libs.
    patterns = [
        # Boost
        "libboost_atomic.so*",
        "libboost_filesystem.so*",
        "libboost_serialization.so*",
        "libboost_system.so*",
        "libboost_python*.so*",
        # Common deps
        "libconsole_bridge.so*",
        "libtinyxml2.so*",
        "liburdfdom_*.so*",
        "liboctomap.so*",
        "liboctomath.so*",
        "libassimp.so*",
        "libqhull_r.so*",
        "libz.so*",
        # Pinocchio stack
        "libeigenpy.so*",
        "libcoal.so*",
        "libpinocchio_*.so*",
    ]

    for lib_dir in _iter_cmeel_prefix_lib_dirs():
        _preload_from_dir(lib_dir, patterns)


def _dlopen_global(path: Path) -> bool:
    """Best-effort dlopen; returns True if loaded, False otherwise."""
    try:
        mode = ctypes.RTLD_GLOBAL
        # Prefer RTLD_NOW when available.
        mode |= getattr(ctypes, "RTLD_NOW", 0)
        ctypes.CDLL(str(path), mode=mode)
        return True
    except Exception:
        return False


def _preload_from_dir(lib_dir: Path, patterns: list[str]) -> None:
    """Preload shared libs matching the provided glob patterns."""
    for pat in patterns:
        for p in sorted(lib_dir.glob(pat)):
            if p.is_file():
                _dlopen_global(p)


def preload_cmeel_boost_libs() -> None:
    """Preload Boost libs from any discovered cmeel prefix."""
    # Also ensure common compatibility symlinks exist (best-effort).
    ensure_boost_soname_compat()

    patterns = [
        # Core Boost libs frequently needed by Pinocchio / eigenpy stack
        "libboost_system.so*",
        "libboost_filesystem.so*",
        "libboost_serialization.so*",
        # Boost.Python (name varies with python version, e.g. libboost_python310.so*)
        "libboost_python*.so*",
    ]

    for lib_dir in _iter_cmeel_prefix_lib_dirs():
        _preload_from_dir(lib_dir, patterns)


def _try_preload_missing_from_error(err: BaseException) -> None:
    """If ImportError mentions a missing lib, try to preload that specific SONAME from cmeel."""
    msg = str(err)
    m = re.search(r"(lib[^\\s/]+\\.so\\.[0-9][^\\s:]*)", msg)
    if not m:
        return
    missing = m.group(1)
    for lib_dir in _iter_cmeel_prefix_lib_dirs():
        _maybe_create_soname_symlink(lib_dir, missing)
        candidate = lib_dir / missing
        if candidate.exists():
            _dlopen_global(candidate)


def import_pinocchio() -> ModuleType:
    """Import and return `pinocchio`, with best-effort preloading of cmeel-shipped libs."""
    global _PINOCCHIO_MODULE
    if _PINOCCHIO_MODULE is not None:
        return _PINOCCHIO_MODULE

    # Preload from known cmeel prefix dirs (helps the dynamic loader find deps)
    preload_cmeel_boost_libs()

    try:
        import pinocchio as pin  # type: ignore

        _PINOCCHIO_MODULE = pin
        return pin
    except ImportError as e:
        # One more attempt: preload the exact missing lib if it exists on disk
        _try_preload_missing_from_error(e)
        preload_cmeel_boost_libs()
        import pinocchio as pin  # type: ignore

        _PINOCCHIO_MODULE = pin
        return pin

