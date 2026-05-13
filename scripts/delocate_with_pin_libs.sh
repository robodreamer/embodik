#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 WHEEL_DIR WHEEL" >&2
  exit 2
fi

wheel_dir="$1"
wheel="$2"

if [[ ! -f "$wheel" ]]; then
  echo "EmbodiK wheel repair error: wheel file not found: $wheel" >&2
  exit 2
fi

pin_lib="$(
  python - <<'PY'
import pathlib

try:
    import pinocchio
except Exception as exc:
    raise SystemExit(
        "EmbodiK wheel repair error: could not import pinocchio. "
        "Install the PyPI build dependency with `python -m pip install 'pin>=3.8.0'` "
        "before running delocate."
    ) from exc

pin_path = pathlib.Path(pinocchio.__file__).resolve()
for parent in pin_path.parents:
    lib_dir = parent / "lib"
    if any(lib_dir.glob("libpinocchio*.dylib")):
        print(lib_dir)
        break
else:
    raise SystemExit(
        f"EmbodiK wheel repair error: could not locate libpinocchio*.dylib from {pin_path}. "
        "Check that the PyPI `pin` package installed its cmeel native libraries."
    )
PY
)"

export DYLD_LIBRARY_PATH="${pin_lib}:${DYLD_LIBRARY_PATH:-}"
echo "EmbodiK wheel repair: using Pinocchio libraries from ${pin_lib}"
delocate-wheel --wheel-dir "${wheel_dir}" -v "${wheel}"
