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
    import importlib.metadata as im
    pin_prefix = pathlib.Path(im.distribution("pin").locate_file("cmeel.prefix")).resolve()
except Exception as exc:
    raise SystemExit(
        "EmbodiK wheel repair error: could not locate the PyPI `pin` prefix. "
        "Install the PyPI build dependency with `python -m pip install 'pin>=3.8.0,<4'` "
        "before running auditwheel repair."
    ) from exc

lib_dir = pin_prefix / "lib"
if any(lib_dir.glob("libpinocchio*.so")):
    print(lib_dir)
else:
    raise SystemExit(
        f"EmbodiK wheel repair error: could not locate libpinocchio*.so from {pin_prefix}. "
        "Check that the PyPI `pin` package installed its cmeel native libraries."
    )
PY
)"

export LD_LIBRARY_PATH="${pin_lib}:${LD_LIBRARY_PATH:-}"
echo "EmbodiK wheel repair: using Pinocchio libraries from ${pin_lib}"
auditwheel repair -w "${wheel_dir}" "${wheel}"
