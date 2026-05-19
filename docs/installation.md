# Installation

EmbodiK supports Python 3.10-3.12 and is distributed through PyPI. Start with
the wheel-only install when a wheel exists for your platform. If pip cannot use
a wheel, use the one-shot installer script instead of debugging a manual source
build first; it creates the virtual environment, installs native dependencies,
builds EmbodiK, and runs an import smoke test.

> Published repaired wheels do not require the Python `pin` package for core
> runtime use. Source builds use `pin` or a system Pinocchio install as the
> native library provider; keep that provider installed in the environment used
> to import EmbodiK. Some optional examples and visualization paths still use
> Python Pinocchio directly.

## Fastest Wheel Path

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install --only-binary=:all: embodik
python -c "import embodik; print(embodik.__version__)"
```

If the import command works, the core package is installed. If pip reports that
no matching binary distribution exists, use the one-shot installer below.

## One-Shot Source Installer

Use this path when a wheel is not available, when pip tries to build from
`embodik-*.tar.gz`, or when you want a local editable install. It is the
recommended source-build path for most users because it handles the venv,
native dependencies, CMake prefix, rpath repair, and final import check.
On Linux, the script targets Debian/Ubuntu and uses `sudo apt-get` unless you
pass `--skip-apt`. On macOS, install Homebrew and Xcode command-line tools
first; use `--skip-brew` only after installing equivalent native packages.

=== "macOS"

    ```bash
    curl -fsSL -O https://gist.githubusercontent.com/robodreamer/adc0b4452d474586c5890877b629005b/raw/install_embodik_macos.sh
    bash install_embodik_macos.sh --python python3.12
    source .venv/bin/activate
    python -c "import embodik; print(embodik.__version__)"
    ```

    Use `--python python3.11` only when a downstream integration explicitly
    needs Python 3.11.

=== "Linux (Debian/Ubuntu)"

    ```bash
    curl -fsSL -O https://gist.githubusercontent.com/robodreamer/adc0b4452d474586c5890877b629005b/raw/install_embodik_linux.sh
    bash install_embodik_linux.sh
    source .venv/bin/activate
    python -c "import embodik; print(embodik.__version__)"
    ```

From a repository checkout, run the scripts directly:

```bash
bash scripts/install_embodik_linux.sh --help
bash scripts/install_embodik_macos.sh --help
```

## Choose An Install Path

| Goal | Use this |
| --- | --- |
| Install EmbodiK as a user when a wheel exists | `python -m pip install --only-binary=:all: embodik` |
| Run copied examples in a venv | `python -m pip install "embodik[examples]"`, then `embodik-examples --copy` |
| Run the Spot mjviser example in a venv | `python -m pip install "embodik[mjviser]"`, then copy examples |
| Recover when pip builds from `embodik-*.tar.gz` | Use the one-shot source installer for your platform |
| Develop from a repository clone | Use Pixi; see [Developer Setup](#developer-setup) |

## Examples

=== "Basic Examples"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    python -m pip install "embodik[examples]"
    embodik-examples --copy
    cd embodik_examples
    python 01_basic_ik_simple.py
    python 14_spot_full_body_ik_viser.py
    ```

=== "Basic + Spot mjviser"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    python -m pip install "embodik[examples,mjviser]"
    embodik-examples --copy
    cd embodik_examples
    python 14_spot_full_body_ik_viser.py
    python 15_spot_locomanip_mjviser.py --policy locomanip
    ```

The examples extra intentionally includes optional packages used by the copied
example scripts, including Python Pinocchio for scripts that import `pinocchio`
directly. Published repaired wheels for the core `embodik` install do not depend
on Python Pinocchio.

The copied examples include the public Spot-with-arm URDF used by
`14_spot_full_body_ik_viser.py`, so a normal pip/venv install no longer needs a
private Spot model path or `EMBODIK_SPOT_IK_URDF` for the default run.

If you already installed only `embodik[examples]`, add the mjviser extra before
running the MuJoCo-backed Spot example:

```bash
python -m pip install "embodik[mjviser]"
python 15_spot_locomanip_mjviser.py --policy locomanip
```

The copied examples include the public MuJoCo Menagerie Spot-with-arm MJCF
scene used by the mjviser example, so the default run does not need a
`robot_descriptions` first-run download or a custom MJCF path.

See the [Examples Guide](examples/index.md) for the full catalog.

## If Pip Builds From Source

Use the one-shot installer scripts above when pip falls back to
`embodik-*.tar.gz`, when no compatible wheel exists for your platform/Python,
or when you need an editable local build. They create a `.venv`, install system
and Python build dependencies, prefer the PyPI `pin` CMake prefix, install
EmbodiK, and run an import smoke test.

The one-shot scripts keep `pin` in the same virtual environment and patch the
installed extension rpath so source-built EmbodiK can find the native Pinocchio,
Coal, and related libraries after the shell exits. If you build manually, do not
remove the native provider used at build time unless you also repair/bundle the
resulting wheel.

Common options:

- `--venv PATH` chooses the virtualenv location.
- `--python CMD` chooses the Python interpreter.
- `--pypi` installs from PyPI.
- `--editable [DIR]` installs a local checkout.
- `--clean-env` clears `LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`,
  `CMAKE_PREFIX_PATH`, and `pinocchio_DIR` before building.
- `--skip-apt` or `--skip-brew` assumes system packages are already installed.

## Manual Linux Source Build

Use the one-shot script above unless you need to debug the build environment.
For Debian/Ubuntu, the equivalent manual setup is:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build pkg-config \
  libeigen3-dev liburdfdom-dev patchelf

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install pin scikit-build-core nanobind cmake ninja

PIN_PREFIX="$(python -c "import pinocchio, pathlib; print(pathlib.Path(pinocchio.__file__).resolve().parents[4])")"
export CMAKE_PREFIX_PATH="$PIN_PREFIX"
export LD_LIBRARY_PATH="${PIN_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

python -m pip install --no-build-isolation embodik
python -c "import embodik; print(embodik.__version__)"
```

On other Linux distributions, install equivalent C++ build, CMake, Ninja,
Eigen3, and URDFDOM packages manually, then run the one-shot script with
`--skip-apt`.

## Manual macOS Source Build

Use Python 3.10-3.12 for source builds. Python 3.12 is the default macOS path.
Use Python 3.11 only when a downstream integration explicitly needs that ABI.
Homebrew's default `eigen` formula can be Eigen 5.x; EmbodiK expects Eigen 3.x,
so use `eigen@3`.

```bash
brew install eigen@3 urdfdom_headers urdfdom

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install pin scikit-build-core nanobind cmake ninja

export Eigen3_DIR="$(brew --prefix eigen@3)/share/eigen3/cmake"
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
export CXXFLAGS="-isystem ${SDKROOT}/usr/include/c++/v1 ${CXXFLAGS:-}"

PIN_PREFIX="$(python -c "import pinocchio, pathlib; print(pathlib.Path(pinocchio.__file__).resolve().parents[4])")"
export CMAKE_PREFIX_PATH="${PIN_PREFIX}:$(brew --prefix)"

python -m pip install --no-build-isolation embodik
python -c "import embodik; print(embodik.__version__)"
```

## Developer Setup

Pixi is the recommended development environment from a repository clone:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
git clone https://github.com/robodreamer/embodik.git
cd embodik
pixi run install
pixi run test
```

Common development commands:

```bash
pixi run install-rebuild
pixi run docs-build
```

Pixi manages Pinocchio, Eigen, Nanobind, CMake, and the Qhull CMake workaround
used by this repository's development builds.

## Optional Extras

| Extra | Install command | Notes |
| --- | --- | --- |
| Examples | `python -m pip install "embodik[examples]"` | Copied examples, robot descriptions, Viser, and Python Pinocchio for scripts that import it directly |
| Direct visualization | `python -m pip install "embodik[visualization]"` | Viser, mesh loading, and URDF parsing without adding Python Pinocchio |
| Pinocchio visualization | `python -m pip install "embodik[visualization-pinocchio]"` | Pinocchio's Python ViserVisualizer |
| GPU solvers | `python -m pip install "embodik[gpu]"` | Experimental; see [GPU Solvers](gpu_solvers.md) |
| GPU collision | `python -m pip install "embodik[gpu-collision]"` | NVIDIA Warp collision experiments |

Teleop controller setup and CUDA/CusADi build steps are intentionally kept out
of the core install path. See the [Examples Guide](examples/index.md) and
[GPU Solvers](gpu_solvers.md) when you need those workflows.

For Seer controller teleop, xvisio stays in optional Pixi environments and is
only imported when `--enable-teleop` is provided:

```bash
pixi install -e teleop
pixi run -e teleop python examples/03_teleop_ik.py --enable-teleop
```

For Spot locomanipulation, `mjviser` is the MuJoCo-backed web viewer
environment for browser GUI policy rollout. Use it when you do not need the
Seer controller:

```bash
pixi install -e mjviser
pixi run -e mjviser spot-locomanip-mjviser
```

For Spot locomanipulation with mjviser and a Seer controller, use
`mjviser-teleop`. It includes the same mjviser stack plus the optional
Seer/xvisio dependencies needed by `--enable-teleop`:

```bash
pixi install -e mjviser-teleop
pixi run -e mjviser-teleop spot-locomanip-mjviser --enable-teleop
```

Use a single Pixi environment per run. `mjviser` and `mjviser-teleop` are
optional environment names, not Python module arguments.

## Troubleshooting Native Builds

### Local Pinocchio, Boost, or ROS paths override the build

If imports fail with missing `libboost_*.so`, `@rpath` errors, or mismatched
Pinocchio libraries, clear local robotics environment paths and rebuild:

```bash
unset LD_LIBRARY_PATH DYLD_LIBRARY_PATH CMAKE_PREFIX_PATH pinocchio_DIR
python -m pip install --force-reinstall --no-cache-dir --no-build-isolation embodik
```

The one-shot installers provide the same cleanup with `--clean-env`.

### `Could not find pinocchio`

Install the PyPI `pin` build dependency and put its prefix first in
`CMAKE_PREFIX_PATH`:

```bash
python -m pip install pin
PIN_PREFIX="$(python -c "import pinocchio, pathlib; print(pathlib.Path(pinocchio.__file__).resolve().parents[4])")"
export CMAKE_PREFIX_PATH="$PIN_PREFIX"                    # Linux
export CMAKE_PREFIX_PATH="${PIN_PREFIX}:$(brew --prefix)" # macOS
```

### `Could not find Eigen3` or incompatible Eigen version

Linux:

```bash
sudo apt-get install libeigen3-dev
```

macOS:

```bash
brew install eigen@3
export Eigen3_DIR="$(brew --prefix eigen@3)/share/eigen3/cmake"
```

### `Could not find urdfdom_headers` on macOS

```bash
brew install urdfdom_headers urdfdom
```

Keep the PyPI `pin` prefix and Homebrew prefix together in `CMAKE_PREFIX_PATH`:

```bash
export CMAKE_PREFIX_PATH="${PIN_PREFIX}:$(brew --prefix)"
```

### `'cstddef' file not found` or `'cmath' file not found` on macOS

```bash
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
export CXXFLAGS="-isystem ${SDKROOT}/usr/include/c++/v1 ${CXXFLAGS:-}"
```

If standard headers still fail, reinstall command-line tools or select full
Xcode when installed:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
```

### `Cannot import scikit_build_core.build`

With `--no-build-isolation`, build dependencies must already be installed:

```bash
python -m pip install scikit-build-core nanobind cmake ninja
```

### Qhull CMake configuration error in Pixi/Conda environments

If CMake reports that imported Qhull executable targets do not exist, run:

```bash
pixi run patch-qhull
```

This patch is already a dependency of the repository's Pixi build tasks.

## Verify Installation

```bash
python -c "import embodik, numpy as np; print(embodik.__version__); print(embodik.log3(np.eye(3)))"
```

## Next Steps

- [Quickstart Guide](quickstart.md) — Get started with EmbodiK
- [API Reference](api/index.md) — Explore the API
- [Examples](examples/index.md) — See example code
