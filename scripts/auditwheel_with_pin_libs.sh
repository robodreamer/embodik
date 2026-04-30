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
    if any(lib_dir.glob("libpinocchio*.so")):
        print(lib_dir)
        break
else:
    raise RuntimeError(f"Could not locate Pinocchio shared libraries from {pin_path}")
PY
)"

export LD_LIBRARY_PATH="${pin_lib}:${LD_LIBRARY_PATH:-}"
auditwheel repair -w "${wheel_dir}" "${wheel}"
