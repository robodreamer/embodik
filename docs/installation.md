# Installation

EmbodiK requires Python 3.10+ and several system dependencies.

## Quick Installation

### For End Users (Pre-built Package)

If you just want to use embodiK without building from source:

```bash
pip install embodik
```

This installs wheels from PyPI. The `pin` package (Pinocchio Python bindings with C++ libraries) is installed as a dependency — **no separate Pinocchio installation needed**.

**Note on names:** the PyPI package is `pin`, but the import is `import pinocchio`.

#### Recommended for robotics stacks: Pixi (runtime) + pip (embodik)

If you are in a robotics environment (e.g. you already built Pinocchio from source, or you export `LD_LIBRARY_PATH`),
**pip-only installs can break due to shared-library conflicts**. The most reliable pattern is:

- Use **Pixi/conda-forge** for the native stack (Pinocchio + C++ deps)
- Use **pip** for `embodik`

Example:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
mkdir embodik_env && cd embodik_env
pixi init
# conda/pixi package name is `pinocchio` (PyPI package name is `pin`)
pixi add -c conda-forge python "pinocchio>=3.8,<4" numpy pip

# Install embodik via pip but DON'T let pip install PyPI `pin` on top of conda `pinocchio`
pixi run pip install --no-deps embodik
pixi run python -c "import embodik; import pinocchio as pin; print(embodik.__version__)"
```

#### Troubleshooting pip-only installs

If `import pinocchio` fails with missing Boost / shared libraries and you have `LD_LIBRARY_PATH` set (common in robotics),
sanitize your shell environment:

```bash
eval "$(embodik-sanitize-env --shell)"
# or:
unset LD_LIBRARY_PATH
```

### For Developers (Building from Source)

We **strongly recommend using Pixi** - it automatically manages all dependencies (CMake, Eigen, Pinocchio, nanobind, etc.) with a single command.

**Step 1: Install Pixi**
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

**Step 2: Clone and Install**
```bash
git clone https://github.com/embodik/embodik.git
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
git clone https://github.com/embodik/embodik.git
cd embodik
python scripts/patch_qhull_cmake.py  # Required for manual installs
pip install -e .
```

</details>


## Optional Dependencies

### Visualization

Install optional visualization dependencies:

```bash
pip install embodik[visualization]
```

This includes:
- `pin>=3.8.0` - Pinocchio with built-in ViserVisualizer support
- `viser>=0.1.0` - 3D visualization (required by Pinocchio's ViserVisualizer)
- `trimesh>=3.0.0` - Mesh loading (required by Pinocchio's ViserVisualizer)

**Note:** Pinocchio 3.8.0+ includes native Viser visualization support, eliminating the need for custom URDF parsing libraries like `yourdfpy`. The visualization system automatically uses Pinocchio's built-in visualizer when available.

For legacy visualization (using custom implementation with yourdfpy):
```bash
pip install embodik[visualization-legacy]
```

### Examples

Install example dependencies:

```bash
pip install embodik[examples]
```

This includes:
- `robot_descriptions` - Robot model descriptions
- `scipy` - Scientific computing utilities

## Verify Installation

Test that EmbodiK is installed correctly:

```python
import embodik
print(f"EmbodiK version: {embodik.__version__}")

# Test basic functionality
model = embodik.RobotModel.from_urdf("path/to/robot.urdf")
print("Installation successful!")
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

- [Quickstart Guide](quickstart.md) - Get started with EmbodiK
- [API Reference](api/index.md) - Explore the API
- [Examples](examples/index.md) - See example code
