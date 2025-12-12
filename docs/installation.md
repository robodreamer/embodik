# Installation

EmbodiK requires Python 3.8+ and several system dependencies.


## Prerequisites

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libeigen3-dev \
    python3-dev \
    python3-pip
```

**macOS (Homebrew):**
```bash
brew install cmake eigen python3
```

### Pinocchio

EmbodiK depends on the [Pinocchio](https://github.com/stack-of-tasks/pinocchio) library for robot kinematics.

**Option 1: Install via robotpkg (Ubuntu/Debian)**
```bash
sudo apt-get install robotpkg-pinocchio
```

**Option 2: Build from source**
```bash
git clone https://github.com/stack-of-tasks/pinocchio.git
cd pinocchio
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/.local
make -j$(nproc)
make install
export CMAKE_PREFIX_PATH=$HOME/.local:$CMAKE_PREFIX_PATH
```

## Python Package Installation

### Option 1: Using Pixi (Recommended for Development)

[Pixi](https://pixi.sh/) provides a reproducible development environment with all dependencies managed automatically.

**Install Pixi:**
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

**Clone and install:**
```bash
git clone https://github.com/embodik/embodik.git
cd embodik
pixi run install
```

**For development with auto-rebuild:**
```bash
pixi run install-rebuild
```

All system dependencies (CMake, Eigen, Pinocchio, etc.) are automatically managed by pixi. Activate the environment with `pixi shell`.

### Option 2: From PyPI

```bash
pip install embodik
```

### Option 3: From Source (Manual)

```bash
git clone https://github.com/embodik/embodik.git
cd embodik
pip install -e .
```

**Note:** This requires manual installation of system dependencies (CMake, Eigen, Pinocchio).

### Development Installation

**With Pixi:**
```bash
pixi run install-rebuild
```

**Without Pixi:**
```bash
git clone https://github.com/embodik/embodik.git
cd embodik
pip install -e ".[dev]"
```

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

### CMake cannot find Pinocchio

If CMake cannot find Pinocchio, set the `CMAKE_PREFIX_PATH`:

```bash
export CMAKE_PREFIX_PATH=/path/to/pinocchio/install:$CMAKE_PREFIX_PATH
```

### Import Error: C++ extension not available

If you see an import warning about the C++ extension:

1. Ensure all system dependencies are installed
2. Rebuild the package: `pip install --force-reinstall --no-cache-dir embodik`
3. Check that CMake found Pinocchio during build

### Build Errors

If you encounter build errors:

1. Ensure you have a C++17 compatible compiler (GCC 7+, Clang 5+)
2. Check that CMake version is 3.16 or higher: `cmake --version`
3. Verify Eigen3 is installed: `pkg-config --modversion eigen3`

## Next Steps

- [Quickstart Guide](quickstart.md) - Get started with EmbodiK
- [API Reference](api/index.md) - Explore the API
- [Examples](examples/index.md) - See example code
