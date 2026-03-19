# Installation

EmbodiK requires Python 3.10+ and is distributed via PyPI as a source distribution (sdist).

> **Note (v0.4.0+)**: EmbodiK no longer requires the Python `pin` package at runtime.
> All Pinocchio functionality is exposed through native C++ bindings. The `pin` package
> is only needed at build time to locate Pinocchio's CMake config. This change resolves
> numpy dependency conflicts when using EmbodiK with other packages.

## Option A: Fresh Environment (No existing Pinocchio)

If you're starting fresh without any local Pinocchio or Boost installations:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# Install build dependencies and Pinocchio
pip install pin scikit-build-core nanobind cmake ninja

# Set CMAKE_PREFIX_PATH so the build can find Pinocchio
export CMAKE_PREFIX_PATH=$(python -c "import pinocchio, pathlib; print(pathlib.Path(pinocchio.__file__).resolve().parents[4])")

# Install embodik
pip install --no-build-isolation embodik

# Verify (no pin import needed at runtime!)
python -c "import embodik; print(embodik.__version__, embodik.RobotModel)"
```

macOS Apple Silicon note:
- Install Xcode command-line tools first: `xcode-select --install`
- Then follow the exact same install flow shown above (`pin` + `CMAKE_PREFIX_PATH`).

## Option B: Robotics Environment (Existing Pinocchio/ROS/Boost)

If you have Pinocchio, Boost, or ROS installed locally (e.g., from source builds, conda, or system packages),
you **must** clear environment variables that point to those installations. Otherwise, embodik may link
against mismatched library versions and fail at runtime with errors like `libboost_*.so.X.Y.Z not found`.

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# CRITICAL: Clear local Pinocchio/Boost paths
unset LD_LIBRARY_PATH DYLD_LIBRARY_PATH CMAKE_PREFIX_PATH pinocchio_DIR

# Install build dependencies and Pinocchio from PyPI
pip install pin scikit-build-core nanobind cmake ninja

# Set CMAKE_PREFIX_PATH to the PyPI pin package
export CMAKE_PREFIX_PATH=$(python -c "import pinocchio, pathlib; print(pathlib.Path(pinocchio.__file__).resolve().parents[4])")

# Install embodik
pip install --no-build-isolation embodik

# Verify (no pin import needed at runtime!)
python -c "import embodik; print(embodik.__version__, embodik.RobotModel)"
```

**Note on package names:** The PyPI package is `pin` (build-time only), but when imported, it's `import pinocchio`.

### With Example Dependencies

```bash
pip install "embodik[examples]"

# Copy and run examples
embodik-examples --copy
cd embodik_examples
python 01_basic_ik_simple.py --robot panda
```

## Troubleshooting

### `ImportError: libboost_*.so...`

This error means `LD_LIBRARY_PATH` points to a locally-built Pinocchio/Boost that conflicts with the `pin` wheel. Fix:

```bash
unset LD_LIBRARY_PATH
```

### `ImportError: Library not loaded: @rpath/...` (macOS)

This usually means `DYLD_LIBRARY_PATH` is pointing to a conflicting local Pinocchio/Boost install. Fix:

```bash
unset DYLD_LIBRARY_PATH
```

### `CMake cannot find pinocchio`

CMake needs to know where the `pin` wheel installed Pinocchio. Fix:

```bash
export CMAKE_PREFIX_PATH=$(python -c "import pinocchio, pathlib; print(pathlib.Path(pinocchio.__file__).resolve().parents[4])")
pip install --no-build-isolation embodik
```

### `Cannot import scikit_build_core.build`

With `--no-build-isolation`, build dependencies must be installed manually:

```bash
pip install scikit-build-core nanobind cmake ninja
```

### For Developers (Building from Source)

We recommend Pixi for development/reproducible builds, but it is optional. If you prefer venv-only development, see the manual section below.

**Step 1: Install Pixi**
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

**Step 2: Clone and Install**
```bash
git clone https://github.com/robodreamer/embodik.git
cd embodik
pixi run install
```

That's it! Pixi automatically:
- ✅ Installs all system dependencies (CMake, Eigen, Pinocchio, nanobind, etc.)
- ✅ Builds the C++ extension
- ✅ Installs the Python package
- ✅ Applies necessary patches (e.g., Qhull CMake workaround)

**For development with auto-rebuild:**
```bash
pixi run install-rebuild
```

**Optional: Seer controller (xvisio) for teleop examples**

The teleop example (`03_teleop_ik.py`) uses the xvisio SDK for Seer wireless controller support. xvisio is optional and requires `libxvsdk.so` on the host. To install it via Pixi:

```bash
# Install the teleop environment (includes xvisio)
pixi install -e teleop

# Run the teleop demo (requires Seer controller + libxvsdk.so)
pixi run -e teleop demo-teleop
```

Without xvisio, the teleop example runs in GUI-only mode with transform controls.

**Activate the environment:**
```bash
pixi shell
```

## Alternative: Manual Installation

<details>
<summary><strong>⚠️ Manual installation (only if Pixi is not available)</strong></summary>

If you cannot use Pixi, you must manually install all system dependencies:

**1. Install system dependencies:**

Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libeigen3-dev python3-dev python3-pip
```

macOS (Homebrew):
```bash
brew install cmake eigen python3
```

**2. Install Pinocchio:**

Option A - via robotpkg (Ubuntu/Debian):
```bash
sudo apt-get install robotpkg-pinocchio
export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH
```

Option B - build from source:
```bash
git clone https://github.com/stack-of-tasks/pinocchio.git
cd pinocchio
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/.local
make -j$(nproc)
make install
export CMAKE_PREFIX_PATH=$HOME/.local:$CMAKE_PREFIX_PATH
```

**3. Install embodiK:**
```bash
git clone https://github.com/robodreamer/embodik.git
cd embodik
python scripts/patch_qhull_cmake.py  # Required for manual installs
pip install -e .
```

</details>


## Optional Dependencies

### GPU Acceleration (CusADi)

> **Experimental:** GPU solvers are under active development and need more validation.

EmbodiK supports GPU-accelerated batched velocity IK solving using CusADi. This is useful for:
- Training RL policies with thousands of parallel environments
- Batch motion planning and optimization
- High-throughput inference

**Requirements:**
- NVIDIA GPU with CUDA support
- PyTorch with CUDA
- CasADi
- CusADi (must be installed manually)

#### Option A: Using Pixi (Recommended)

Pixi manages CUDA-enabled PyTorch automatically:

```bash
cd embodik

# Step 1: Install the CUDA environment
pixi install -e cuda
pixi run -e cuda install        # Install embodik in cuda env

# Step 2: Verify CUDA is available
pixi run -e cuda check-cuda
# Output: PyTorch X.Y.Z, CUDA available: True, CUDA version: 12.4

# Step 3: Install CusADi (one-time, clones to ~/.local/cusadi)
pixi run -e cuda install-cusadi

# Step 4: Verify all GPU components
pixi run -e cuda check-gpu
# Output: CasADi: True, CusADi: True, CUDA: True

# Step 5: Export CasADi function (FI-PeSNS)
pixi run -e cuda export-casadi

# Step 6: Compile CUDA kernel
mv fn_velocity_solve.casadi ~/.local/cusadi/src/casadi_functions/
cd ~/.local/cusadi
python run_codegen.py --fn=fn_velocity_solve

# Optional: Export and compile PPH-SNS (alternative solver)
# pixi run -e cuda export-pph-sns  # Writes directly to cusadi
# cd ~/.local/cusadi && python run_codegen.py --fn=fn_pph_sns_velocity_solve

# Step 7: Run GPU demos
pixi run -e cuda demo-gpu           # Comprehensive benchmark
pixi run -e cuda demo-ik-gpu        # Interactive IK with GPU panel
pixi run -e cuda benchmark-gpu      # Batch IK benchmark
```

#### Available GPU Tasks

| Task | Description |
|------|-------------|
| `check-cuda` | Verify PyTorch CUDA availability |
| `check-gpu` | Verify CasADi + CusADi + CUDA |
| `install-cusadi` | Install CusADi from GitHub to ~/.local/cusadi |
| `export-casadi` | Export FI-PeSNS velocity solve function |
| `export-pph-sns` | Export PPH-SNS velocity solve function |
| `demo-gpu` | Run GPU solver demo/benchmark |
| `demo-ik-gpu` | Interactive IK with GPU benchmark panel |
| `benchmark-gpu` | Batch IK performance benchmark |
| `benchmark-gpu-batched` | GPU batched IK benchmark (100/1000/10000) |
| `benchmark-solver-comparison` | Compare FI-PeSNS vs PPH-SNS (CPU + GPU) |
| `benchmark-solver-batched` | Batched GPU benchmark for both solvers |
| `benchmark-fi-pesns` | FI-PeSNS vs CPU accuracy benchmark |
| `benchmark-collision` | Collision detection benchmark |
| `demo-parallel-tracking` | 100 robots tracking trajectories in parallel |
| `test-gpu` | Run GPU-specific tests |

All GPU tasks should be run with `-e cuda`: `pixi run -e cuda <task>`

### Seer Controller (xvisio)

The teleop example (`03_teleop_ik.py`) supports the Seer wireless controller via the xvisio SDK. This is optional; without it, the example runs in GUI-only mode.

**Requirements:**
- Seer wireless controller hardware
- `libxvsdk.so` on the host (from xvisio SDK setup)

**Install via Pixi:**

```bash
# Install the teleop environment (adds xvisio to default deps)
pixi install -e teleop

# Run the teleop demo
pixi run -e teleop demo-teleop
# Or: pixi run -e teleop python examples/03_teleop_ik.py --robot panda
```

| Task | Description |
|------|-------------|
| `demo-teleop` | Run teleop IK example with panda robot |

<details>
<summary><strong>Technical notes on pixi + PyTorch CUDA setup</strong></summary>

Getting CUDA-enabled PyTorch to work with pixi requires careful configuration:

**Problem:** By default, pixi's dependency solver picks PyTorch from conda-forge, which is CPU-only.

**Solution:** The `cuda` feature in `pixi.toml` uses:
1. **Channel priority**: `channels = ["pytorch", "nvidia", "conda-forge"]` with `channel-priority = "strict"`
2. **Explicit channel specification**: `pytorch = { version = ">=2.0", channel = "pytorch" }`
3. **Platform-specific**: `pytorch-cuda` only exists for `linux-64`
4. **Separate solve group**: Avoids conflicts with the default CPU environment

**Verification:**
```bash
pixi list -e cuda | grep pytorch
# Should show: pytorch from pytorch channel (not conda-forge)

pixi run -e cuda python -c "import torch; print(torch.version.cuda)"
# Should print: 12.4 (not None)
```

</details>

#### Option B: Using pip

```bash
# Install GPU dependencies
pip install "embodik[gpu]"

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install CusADi
git clone https://github.com/se-hwan/cusadi
cd cusadi
pip install -e .

# Export for 7-DOF robot (e.g., Panda)
python -m embodik.gpu.export_casadi_velocity_solve --robot panda --out fn_velocity_solve.casadi

# Move to cusadi and compile
mv fn_velocity_solve.casadi cusadi/src/casadi_functions/
cd cusadi
python run_codegen.py --fn=fn_velocity_solve
```

#### Set environment variable

```bash
export CUSADI_ROOT=/path/to/cusadi
```

#### Verify GPU setup

```python
from embodik.gpu import HAS_CASADI, HAS_CUSADI, HAS_TORCH_CUDA
print(f"CasADi: {HAS_CASADI}, CusADi: {HAS_CUSADI}, CUDA: {HAS_TORCH_CUDA}")
# CasADi: True, CusADi: True, CUDA: True
```

#### Benchmark results (typical, RTX A2000)

| Batch Size | CPU (ms) | GPU (ms) | Speedup |
|------------|----------|----------|---------|
| 100        | 3        | 0.8      | 4x      |
| 1000       | 30       | 1.2      | 25x     |
| 4096       | 120      | 1.5      | 80x     |

#### Run the demos

```bash
# Using pixi (recommended)
pixi run -e cuda demo-gpu              # Comprehensive GPU benchmark
pixi run -e cuda demo-ik-gpu           # Interactive IK with GPU panel
pixi run -e cuda benchmark-gpu         # Batch IK benchmark
pixi run -e cuda benchmark-collision   # Collision detection benchmark

# Or run scripts directly
python examples/06_gpu_solver_demo.py                    # CPU-only benchmark
python examples/06_gpu_solver_demo.py --gpu --casadi_path ~/.local/cusadi/src/casadi_functions/fn_velocity_solve.casadi  # GPU benchmark
python examples/02_collision_aware_IK.py --robot panda --gpu  # Interactive IK with GPU
```

### GPU Collision Detection (Warp)

For GPU-parallel collision detection using NVIDIA Warp:

```bash
pip install "embodik[gpu-collision]"
```

**Example scripts:**

- `examples/05_gpu_collision_batch.py` — Batch collision detection benchmark
- Run with: `pixi run -e cuda benchmark-collision`

See the GPU Collision section in the README for more details.

### Visualization

EmbodiK includes direct Viser visualization that works without the Python `pin` package:

```bash
pip install embodik[visualization]
```

This includes:

- `viser>=0.1.0` — 3D visualization server
- `trimesh>=3.0.0` — Mesh loading for robot visualization

**Default (v0.4.0+):** EmbodiK now defaults to direct Viser visualization with native bindings
for rotation/quaternion math. This eliminates runtime dependency conflicts.

**Optional Pinocchio-based visualization:** If you prefer Pinocchio's ViserVisualizer:
```bash
pip install embodik[visualization-pinocchio]
```
This adds `pin>=3.8.0` which provides Pinocchio's built-in ViserVisualizer.

For legacy visualization (using yourdfpy for URDF parsing):
```bash
pip install embodik[visualization-legacy]
```

### Examples

Install example dependencies:

```bash
pip install embodik[examples]
```

This includes:

- `robot_descriptions` — Robot model descriptions
- `pyyaml` — YAML parsing for robot preset configs
- `viser` — visualization server used by interactive examples
- `yourdfpy` — URDF loader used by some examples (via `robot_descriptions.loaders.yourdfpy`)

## Verify Installation

Test that EmbodiK is installed correctly:

```python
import embodik as eik
print(f"EmbodiK version: {eik.__version__}")

# Test basic functionality
model = eik.RobotModel("path/to/robot.urdf")
print("Installation successful!")

# Test native math utilities (no pin needed)
import numpy as np
R = np.eye(3)
omega = eik.log3(R)  # Native rotation utilities
print(f"log3 works: {omega}")
```

## Troubleshooting

### Import Error: C++ extension not available

If you see an error that `RobotModel` is not available:

**Using Pixi (Recommended):**
```bash
# Rebuild and reinstall
pixi run install

# Or for development with auto-rebuild
pixi run install-rebuild
```

**Using PyPI installation (`pip install embodik`):**
If you installed from PyPI and see this error, it likely means:
1. Only source distribution (sdist) was available (no pre-built wheel for your platform)
2. The build failed because Pinocchio wasn't found

Try installing Pinocchio Python package first, then rebuild:
```bash
pip install pin  # Installs Pinocchio with C++ libraries
pip install --force-reinstall --no-cache-dir embodik
```

**Using manual installation from source:**
1. Ensure all system dependencies are installed (CMake, Eigen, Pinocchio)
2. Rebuild the package: `pip install --force-reinstall --no-cache-dir -e .`
3. Check that CMake found Pinocchio during build

### CMake cannot find Pinocchio

**Using Pixi:** This should not happen - pixi manages Pinocchio automatically. If it does, try:
```bash
pixi run install
```

**Using manual installation:** Set the `CMAKE_PREFIX_PATH`:
```bash
export CMAKE_PREFIX_PATH=/path/to/pinocchio/install:$CMAKE_PREFIX_PATH
pip install -e .
```

### Qhull CMake Configuration Error

If you encounter an error like:
```
CMake Error: The imported target "Qhull::qhull" references the file ".../bin/qhull" but this file does not exist.
```

This is a known issue with conda-forge's qhull package which doesn't include executable binaries. The installation process automatically applies a patch to work around this. If you're installing manually without pixi, run:

```bash
python scripts/patch_qhull_cmake.py
```

before building. This patch is automatically applied when using `pixi run install` or `pixi run build`.

### Build Errors

If you encounter build errors:

1. Ensure you have a C++17 compatible compiler (GCC 7+, Clang 5+)
2. Check that CMake version is 3.16 or higher: `cmake --version`
3. Verify Eigen3 is installed: `pkg-config --modversion eigen3`
4. If using pixi, ensure the Qhull patch was applied: `pixi run patch-qhull`

## Next Steps

- [Quickstart Guide](quickstart.md) — Get started with EmbodiK
- [API Reference](api/index.md) — Explore the API
- [Examples](examples/index.md) — See example code
