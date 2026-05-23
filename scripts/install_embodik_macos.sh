#!/usr/bin/env bash
# One-shot macOS setup: Homebrew deps, venv, exports, and pip install embodik (PyPI or editable).
# See docs/installation.md — "If Pip Builds From Source" and "Manual macOS Source Build".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_DIR=""
# Prefer Homebrew 3.12 when present; use 3.11 only when explicitly requested
# or when 3.12 is unavailable.
if [[ -z "${EMBODIK_PYTHON:-}" ]]; then
  if [[ -x /opt/homebrew/bin/python3.12 ]]; then
    PYTHON_CMD="/opt/homebrew/bin/python3.12"
  elif [[ -x /usr/local/bin/python3.12 ]]; then
    PYTHON_CMD="/usr/local/bin/python3.12"
  elif [[ -x /opt/homebrew/bin/python3.11 ]]; then
    PYTHON_CMD="/opt/homebrew/bin/python3.11"
  elif [[ -x /usr/local/bin/python3.11 ]]; then
    PYTHON_CMD="/usr/local/bin/python3.11"
  else
    PYTHON_CMD="python3"
  fi
else
  PYTHON_CMD="${EMBODIK_PYTHON}"
fi
MODE="pypi"
EDITABLE_DIR=""
SKIP_BREW=0
CLEAN_ENV=0

usage() {
  echo "One-shot macOS install: Homebrew deps, venv, env vars, pip install embodik."
  echo ""
  echo "Usage: $0 [options]"
  echo "  --venv PATH       Virtualenv directory (default: ./.venv in current working directory)"
  echo "  --python CMD      Interpreter for venv (default: \$EMBODIK_PYTHON or python3)"
  echo "  --pypi            Install embodik from PyPI (default)"
  echo "  --editable [DIR]  Editable install; DIR = repo root (default: $DEFAULT_REPO_ROOT)"
  echo "  --skip-brew       Do not run brew install (assume deps already present)"
  echo "  --clean-env       Unset LD_LIBRARY_PATH, DYLD_LIBRARY_PATH, CMAKE_PREFIX_PATH, pinocchio_DIR"
  echo "                    before building (use if you have local ROS/Pinocchio on PATH)"
  echo "  -h, --help        Show this help"
  echo ""
  echo "Examples:"
  echo "  cd ~/my_project && bash $SCRIPT_DIR/install_embodik_macos.sh"
  echo "  bash $SCRIPT_DIR/install_embodik_macos.sh --venv .venv --python python3.12"
  echo "  bash $SCRIPT_DIR/install_embodik_macos.sh --venv .venv --python python3.11  # optional downstream fallback"
  echo "  bash $SCRIPT_DIR/install_embodik_macos.sh --editable ~/src/embodik"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_DIR="${2:?}"
      shift 2
      ;;
    --python)
      PYTHON_CMD="${2:?}"
      shift 2
      ;;
    --pypi)
      MODE="pypi"
      shift
      ;;
    --editable)
      MODE="editable"
      if [[ $# -ge 2 && "$2" != -* ]]; then
        EDITABLE_DIR="$2"
        shift 2
      else
        EDITABLE_DIR="$DEFAULT_REPO_ROOT"
        shift
      fi
      ;;
    --skip-brew)
      SKIP_BREW=1
      shift
      ;;
    --clean-env)
      CLEAN_ENV=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is for macOS only. On Linux, use Pixi or follow docs/installation.md." >&2
  exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is not installed or not on PATH. Install from https://brew.sh" >&2
  exit 1
fi

if ! xcode-select -p >/dev/null 2>&1; then
  echo "Xcode command-line tools are required. Run: xcode-select --install" >&2
  exit 1
fi

if [[ -z "$VENV_DIR" ]]; then
  VENV_DIR="$(pwd)/.venv"
fi
VENV_DIR="$(cd "$(dirname "$VENV_DIR")" && pwd)/$(basename "$VENV_DIR")"

PY_MAJOR="$("$PYTHON_CMD" -c 'import sys; print(sys.version_info.major)' 2>/dev/null)" || true
PY_MINOR="$("$PYTHON_CMD" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null)" || true
if [[ -z "${PY_MAJOR:-}" ]]; then
  echo "Cannot run $PYTHON_CMD. Install Python or set --python / EMBODIK_PYTHON." >&2
  exit 1
fi
if [[ "$PY_MAJOR" -gt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -ge 13 ]]; }; then
  echo "Warning: Python ${PY_MAJOR}.${PY_MINOR} is newer than the 3.10–3.12 supported range." >&2
  echo "         If the build fails (e.g. missing headers), retry with: --python python3.12" >&2
fi

if [[ "$SKIP_BREW" -eq 0 ]]; then
  echo "==> Installing Homebrew dependencies (eigen@3, urdfdom_headers, urdfdom)..."
  brew install eigen@3 urdfdom_headers urdfdom
else
  echo "==> Skipping brew install (--skip-brew)"
fi

EIGEN3_DIR="$(brew --prefix eigen@3)/share/eigen3/cmake"
if [[ ! -f "$EIGEN3_DIR/Eigen3Config.cmake" ]]; then
  echo "Eigen3Config.cmake not found at $EIGEN3_DIR" >&2
  echo "Run: brew install eigen@3" >&2
  exit 1
fi

SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
export SDKROOT
# CLT often has a partial libc++ under .../CommandLineTools/usr/include/c++/v1 (no <cmath>).
# Prefer the SDK's libc++ headers so C++ standard library includes resolve.
SDK_LIBCXX="${SDKROOT}/usr/include/c++/v1"
if [[ -d "$SDK_LIBCXX" ]]; then
  export CXXFLAGS="-isystem ${SDK_LIBCXX} ${CXXFLAGS:-}"
  export CPPFLAGS="-isystem ${SDK_LIBCXX} ${CPPFLAGS:-}"
fi
export Eigen3_DIR="$EIGEN3_DIR"

echo "==> Creating virtualenv: $VENV_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install -U pip
"$VENV_DIR/bin/python" -m pip install "pin>=3.8.0,<4" "cmeel-urdfdom>=4,<5" "cmeel-tinyxml2>=10,<11" scikit-build-core nanobind cmake ninja

echo "==> Configuring CMAKE_PREFIX_PATH (PyPI pin + Homebrew)..."
if [[ "$CLEAN_ENV" -eq 1 ]]; then
  echo "    (--clean-env) clearing conflicting library / CMake path variables"
  unset LD_LIBRARY_PATH DYLD_LIBRARY_PATH CMAKE_PREFIX_PATH pinocchio_DIR || true
fi
# shellcheck disable=SC2016
PIN_PREFIX="$("$VENV_DIR/bin/python" -c 'import importlib.metadata as im, pathlib; print(pathlib.Path(im.distribution("pin").locate_file("cmeel.prefix")).resolve())')"
export CMAKE_PREFIX_PATH="${PIN_PREFIX}:$(brew --prefix)"
echo "    Eigen3_DIR=$Eigen3_DIR"
echo "    SDKROOT=$SDKROOT"
echo "    Pinocchio build prefix=$PIN_PREFIX"
echo "    CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH"

if [[ "$MODE" == "pypi" ]]; then
  echo "==> pip install --upgrade embodik (latest from PyPI, no build isolation)..."
  "$VENV_DIR/bin/python" -m pip install --upgrade --no-build-isolation embodik
else
  REPO="$(cd "$EDITABLE_DIR" && pwd)"
  if [[ ! -f "$REPO/CMakeLists.txt" ]] || [[ ! -f "$REPO/pyproject.toml" ]]; then
    echo "Not an embodik repo root: $REPO" >&2
    exit 1
  fi
  echo "==> pip install -e $REPO (no build isolation)..."
  "$VENV_DIR/bin/python" -m pip install --no-build-isolation -e "$REPO"
fi

echo ""
echo "==> Verifying import..."
IMPORT_LOG="$(mktemp)"
if "$VENV_DIR/bin/python" -c 'import embodik; print("embodik", embodik.__version__, embodik.RobotModel)' 2>"$IMPORT_LOG"; then
  rm -f "$IMPORT_LOG"
else
  RPATH_PATCH_NEEDED=1
  echo "    initial import failed"
  sed 's/^/    /' "$IMPORT_LOG" >&2
  rm -f "$IMPORT_LOG"

  if [[ "$MODE" == "pypi" ]]; then
    echo ""
    echo "==> Reinstalling a pristine PyPI wheel and retrying import..."
    "$VENV_DIR/bin/python" -m pip install --upgrade --force-reinstall --no-deps embodik
    RETRY_LOG="$(mktemp)"
    if "$VENV_DIR/bin/python" -c 'import embodik; print("embodik", embodik.__version__, embodik.RobotModel)' 2>"$RETRY_LOG"; then
      RPATH_PATCH_NEEDED=0
      rm -f "$RETRY_LOG"
    else
      echo "    import still failed after reinstall; patching source-build rpaths and retrying"
      sed 's/^/    /' "$RETRY_LOG" >&2
      rm -f "$RETRY_LOG"
    fi
  else
    echo "    patching source-build rpaths and retrying"
  fi

  if [[ "$RPATH_PATCH_NEEDED" -eq 1 ]]; then
    # Source/sdist/editable installs may need an extra rpath for the
    # cmeel.prefix/lib directory where the PyPI `pin` package installs Pinocchio
    # dylibs. Do not patch healthy repaired wheels: install_name_tool mutates the
    # Mach-O file and can invalidate code signatures on newer macOS releases.
    echo ""
    echo "==> Patching rpath for cmeel-installed Pinocchio dylibs..."
    EMBODIK_SO="$("$VENV_DIR/bin/python" -c \
      'import importlib.metadata as im, pathlib, sysconfig
candidates = []
try:
    dist = im.distribution("embodik")
    for file in dist.files or []:
        path = pathlib.PurePosixPath(str(file))
        if (
            path.parent == pathlib.PurePosixPath("embodik")
            and path.name.startswith("_embodik_impl")
            and path.suffix == ".so"
        ):
            candidates.append(pathlib.Path(dist.locate_file(file)))
except Exception:
    pass
platlib = pathlib.Path(sysconfig.get_paths()["platlib"])
candidates.extend((platlib / "embodik").glob("_embodik_impl*.so"))
print(next((str(path) for path in candidates if path.is_file()), ""))' 2>/dev/null || true)"
    EMBODIK_CORE_LIB="$("$VENV_DIR/bin/python" -c \
      'import pathlib, sysconfig
candidates = []
platlib = pathlib.Path(sysconfig.get_paths()["platlib"])
for package_dir in [platlib / "embodik"]:
    candidates.extend((package_dir / "lib").glob("libembodik_core.*"))
    candidates.extend(package_dir.glob("libembodik_core.*"))
print(next((str(path) for path in candidates if path.is_file()), ""))' 2>/dev/null || true)"
    CMEEL_LIB="$("$VENV_DIR/bin/python" -c \
      'import importlib.metadata as im, pathlib; \
       print(pathlib.Path(im.distribution("pin").locate_file("cmeel.prefix")).resolve() / "lib")' 2>/dev/null || true)"
    if [[ -n "$EMBODIK_SO" && -f "$EMBODIK_SO" && -n "$CMEEL_LIB" && -d "$CMEEL_LIB" ]]; then
      # install_name_tool exits non-zero if the rpath already exists; that is fine.
      install_name_tool -add_rpath "$CMEEL_LIB" "$EMBODIK_SO" 2>/dev/null || true
      echo "    rpath -> $CMEEL_LIB"
      if [[ -n "$EMBODIK_CORE_LIB" && -f "$EMBODIK_CORE_LIB" ]]; then
        install_name_tool -add_rpath "$CMEEL_LIB" "$EMBODIK_CORE_LIB" 2>/dev/null || true
        echo "    core rpath -> $CMEEL_LIB"
      fi
    else
      echo "    (skipped: EMBODIK_SO='$EMBODIK_SO'  CMEEL_LIB='$CMEEL_LIB')"
    fi

    echo ""
    echo "==> Verifying import after rpath patch..."
    "$VENV_DIR/bin/python" -c 'import embodik; print("embodik", embodik.__version__, embodik.RobotModel)'
  fi
fi

echo ""
echo "Done. Activate with:"
echo "  source \"$VENV_DIR/bin/activate\""
