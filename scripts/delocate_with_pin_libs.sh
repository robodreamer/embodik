#!/usr/bin/env bash
set -euo pipefail

wheel_dir="$1"
wheel="$2"

pin_lib="$(
  python - <<'PY'
import pathlib
import pinocchio

pin_path = pathlib.Path(pinocchio.__file__).resolve()
for parent in pin_path.parents:
    lib_dir = parent / "lib"
    if any(lib_dir.glob("libpinocchio*.dylib")):
        print(lib_dir)
        break
else:
    raise RuntimeError(f"Could not locate Pinocchio dylibs from {pin_path}")
PY
)"

export DYLD_LIBRARY_PATH="${pin_lib}:${DYLD_LIBRARY_PATH:-}"
delocate-wheel --wheel-dir "${wheel_dir}" -v "${wheel}"
