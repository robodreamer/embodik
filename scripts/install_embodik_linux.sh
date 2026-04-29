#!/usr/bin/env bash
# One-shot Linux setup (Debian/Ubuntu): apt deps, venv, env vars, and pip install embodik.
# See docs/installation.md — "Linux (Debian/Ubuntu): pip / sdist builds".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_DIR=""
PYTHON_CMD="${EMBODIK_PYTHON:-python3}"
MODE="pypi"
EDITABLE_DIR=""
SKIP_APT=0
CLEAN_ENV=0

usage() {
  echo "One-shot Linux install (Debian/Ubuntu): apt deps, venv, env vars, pip install embodik."
  echo ""
  echo "Usage: $0 [options]"
  echo "  --venv PATH       Virtualenv directory (default: ./.venv in current working directory)"
  echo "  --python CMD      Interpreter for venv (default: \$EMBODIK_PYTHON or python3)"
  echo "  --pypi            Install embodik from PyPI (default)"
  echo "  --editable [DIR]  Editable install; DIR = repo root (default: $DEFAULT_REPO_ROOT)"
  echo "  --skip-apt        Do not run apt-get install (assume deps already present)"
  echo "  --clean-env       Unset LD_LIBRARY_PATH, DYLD_LIBRARY_PATH, CMAKE_PREFIX_PATH, pinocchio_DIR"
  echo "                    before building (use if you have local ROS/Pinocchio on PATH)"
  echo "  -h, --help        Show this help"
  echo ""
  echo "Examples:"
  echo "  cd ~/my_project && bash $SCRIPT_DIR/install_embodik_linux.sh"
  echo "  bash $SCRIPT_DIR/install_embodik_linux.sh --venv .venv --python python3.12"
  echo "  bash $SCRIPT_DIR/install_embodik_linux.sh --editable ~/src/embodik"
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
    --skip-apt)
      SKIP_APT=1
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

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script is for Linux only. On macOS, use scripts/install_embodik_macos.sh." >&2
  exit 1
fi

if [[ -z "$VENV_DIR" ]]; then
  VENV_DIR="$(pwd)/.venv"
fi
VENV_DIR="$(cd "$(dirname "$VENV_DIR")" && pwd)/$(basename "$VENV_DIR")"

if ! "$PYTHON_CMD" -c 'import sys; sys.exit(0)' >/dev/null 2>&1; then
  echo "Cannot run $PYTHON_CMD. Install Python or set --python / EMBODIK_PYTHON." >&2
  exit 1
fi

if [[ "$SKIP_APT" -eq 0 ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    echo "==> Installing apt packages (build-essential, cmake, ninja-build, pkg-config, libeigen3-dev, liburdfdom-dev)..."
    sudo apt-get update
    sudo apt-get install -y \
      build-essential \
      cmake \
      ninja-build \
      pkg-config \
      libeigen3-dev \
      liburdfdom-dev
  else
    echo "apt-get not found. This script currently supports Debian/Ubuntu package install only." >&2
    echo "Use --skip-apt after installing equivalent packages manually." >&2
    exit 1
  fi
else
  echo "==> Skipping apt install (--skip-apt)"
fi

echo "==> Creating virtualenv: $VENV_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install -U pip
"$VENV_DIR/bin/python" -m pip install pin scikit-build-core nanobind cmake ninja

echo "==> Configuring CMAKE_PREFIX_PATH (PyPI pin first)..."
if [[ "$CLEAN_ENV" -eq 1 ]]; then
  echo "    (--clean-env) clearing conflicting library / CMake path variables"
  unset LD_LIBRARY_PATH DYLD_LIBRARY_PATH CMAKE_PREFIX_PATH pinocchio_DIR || true
fi

# shellcheck disable=SC2016
PIN_PREFIX="$("$VENV_DIR/bin/python" -c 'import pinocchio, pathlib; print(pathlib.Path(pinocchio.__file__).resolve().parents[4])')"
export CMAKE_PREFIX_PATH="${PIN_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
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
"$VENV_DIR/bin/python" -c 'import embodik; print("embodik", embodik.__version__, embodik.RobotModel)'

echo ""
echo "Done. Activate with:"
echo "  source \"$VENV_DIR/bin/activate\""
