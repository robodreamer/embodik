# EmbodiK: Cross-Embodiment Inverse Kinematics with Nanobind

EmbodiK is a high-performance inverse kinematics (IK) library for cross-embodiment VLA/AI applications.

- The core is implemented in C++, with Python bindings created using Nanobind.
- EmbodiK delivers robust and high-performance IK behaviors, particularly optimized for humanoid robots and AI/VLA integrations.
- The name "EmbodiK" highlights its focus on supporting various kinematic structures across different embodiment types.
- The library handles diverse constraint types, supporting both single-task and multi-task velocity IK solvers.
- Advanced inverse methods provide singularity-robustness.
- Features include self-collision avoidance and interactive 3D visualization tools.

**Author:** Andy Park <andypark.purdue@gmail.com>

## Features

- **High Performance**: C++ core with optimized Eigen linear algebra
- **Python Integration**: Seamless numpy array support via Nanobind
- **Multiple Solvers**: Single-step and full multi-task velocity IK
- **Singularity Robust**: Advanced inverse methods for stable solutions
- **Constraint Support**: Joint limits and operational space constraints
- **Collision Avoidance**: Self-collision detection and avoidance
- **Visualization**: Interactive 3D visualization with Viser
- **Robot Models**: Built-in support for common robots (Panda, IIWA)
- **GPU Acceleration**: Batched velocity IK via CusADi for massive parallelism (100-500x speedup)

## Installation

### Option A: Fresh Environment (No existing Pinocchio)

If you don't have Pinocchio/Boost installed locally, installation is straightforward:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# Install build dependencies and Pinocchio
pip install pin scikit-build-core nanobind cmake ninja

# Set CMAKE_PREFIX_PATH and install
export CMAKE_PREFIX_PATH=$(python -c "import pinocchio, pathlib; print(pathlib.Path(pinocchio.__file__).resolve().parents[4])")
pip install --no-build-isolation embodik

# Verify
python -c "import embodik; print(embodik.__version__, embodik.RobotModel)"
```

### Option B: Robotics Environment (Existing Pinocchio/ROS)

If you have local Pinocchio/Boost builds (e.g., from source or ROS), you **must** clear conflicting paths first:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# IMPORTANT: Clear local Pinocchio paths to avoid library conflicts
unset LD_LIBRARY_PATH CMAKE_PREFIX_PATH pinocchio_DIR

# Install build dependencies and Pinocchio from PyPI
pip install pin scikit-build-core nanobind cmake ninja

# Set CMAKE_PREFIX_PATH to the PyPI pin package
export CMAKE_PREFIX_PATH=$(python -c "import pinocchio, pathlib; print(pathlib.Path(pinocchio.__file__).resolve().parents[4])")

# Install embodik
pip install --no-build-isolation embodik

# Verify
python -c "import embodik; print(embodik.__version__, embodik.RobotModel)"
```

### Running Examples

```bash
pip install "embodik[examples]"
embodik-examples --copy
cd embodik_examples
python 01_basic_ik_simple.py --robot panda
```

### Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ImportError: libboost_*.so...` | `LD_LIBRARY_PATH` points to local Pinocchio | `unset LD_LIBRARY_PATH` |
| `CMake cannot find pinocchio` | Build can't find Pinocchio config | Set `CMAKE_PREFIX_PATH` (see above) |
| `Cannot import scikit_build_core` | Missing build deps with `--no-build-isolation` | `pip install scikit-build-core nanobind cmake ninja` |

### For Developers

See [docs/installation.md](docs/installation.md) for development setup with Pixi.

See [PUBLISHING.md](PUBLISHING.md) for wheel building and PyPI publishing.

## Quick Start

```python
import embodik
import numpy as np

# Load robot model from URDF
robot = embodik.RobotModel("path/to/robot.urdf", floating_base=False)

# Create kinematics solver
solver = embodik.KinematicsSolver(robot)

# Add a frame task for end-effector control
frame_task = solver.add_frame_task("ee_task", "end_effector")
frame_task.priority = 0
frame_task.weight = 1.0

# Set target velocity (6D: 3 linear + 3 angular)
target_velocity = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
frame_task.set_target_velocity(target_velocity)

# Solve velocity IK
q = np.zeros(robot.nq)
result = solver.solve_velocity(q, apply_limits=True)

if result.status == embodik.SolverStatus.SUCCESS:
    print(f"Joint velocities: {result.joint_velocities}")
```

## API Overview

### High-Level API (Recommended)

EmbodiK provides a high-level API built on top of Pinocchio for easy robot modeling and IK solving:

```python
import embodik
import numpy as np

# Create robot model
robot = embodik.RobotModel("robot.urdf", floating_base=False)

# Create solver
solver = embodik.KinematicsSolver(robot)

# Add tasks
frame_task = solver.add_frame_task("task1", "end_effector")
posture_task = solver.add_posture_task("posture")

# Configure tasks
frame_task.priority = 0
frame_task.weight = 1.0
posture_task.priority = 1
posture_task.weight = 0.1

# Solve
q = np.zeros(robot.nq)
result = solver.solve_velocity(q, apply_limits=True)
```

### Low-Level API

For advanced users, EmbodiK also provides low-level multi-task velocity IK functions:

```python
import embodik as eik
import numpy as np

# Multiple tasks with constraints
goals = [np.array([0.1, -0.2]), np.array([0.3])]
jacobians = [
    np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    np.array([[0.0, 0.0, 1.0]])
]

# Constraint matrix and limits
C = np.eye(3)
lower = np.array([-1e6, -1e6, -1e6])
upper = np.array([1e6, 1e6, 1e6])

params = {
    "epsilon": 1e-6,
    "regularization_factor": 1e-1,
}

result = eik.solve_velocity_ik_multi_task_np(
    goals, jacobians, C, lower, upper, params
)
```

## Examples

The repository includes several example scripts:

| Script | Description |
|--------|-------------|
| `01_basic_ik_simple.py` | Basic IK solving with interactive visualization |
| `02_collision_aware_IK.py` | Collision-aware IK with self-collision avoidance (supports `--gpu` flag) |
| `04_gpu_batch_ik.py` | GPU-accelerated batched velocity IK benchmark |
| `05_gpu_collision_batch.py` | GPU-accelerated batch collision detection |
| `06_gpu_solver_demo.py` | Comprehensive GPU solver demonstration and benchmark |
| `robot_model_example.py` | Robot model usage and configuration |
| `visualization_example.py` | Interactive 3D visualization examples |

### Running Examples

**For pip-installed users:**
```bash
# Install with example dependencies
pip install embodik[examples]

# Copy examples to a local directory
embodik-examples --copy

# Run examples
cd embodik_examples
python 01_basic_ik_simple.py --robot panda
python 02_collision_aware_IK.py --robot panda
```

**For developers (from repository):**
```bash
# Install example dependencies
pixi run install

# Run basic IK example
pixi run python examples/01_basic_ik_simple.py

# Run collision-aware IK example
pixi run python examples/02_collision_aware_IK.py --robot panda

# Run GPU examples (requires cuda environment)
pixi run -e cuda demo-gpu          # GPU solver benchmark
pixi run -e cuda demo-ik-gpu       # Interactive IK with GPU panel
pixi run -e cuda benchmark-gpu     # Batch IK benchmark
pixi run -e cuda benchmark-collision  # Collision detection benchmark
```

See the [Examples Documentation](docs/examples/index.md) for detailed guides.

## GPU Acceleration

EmbodiK supports GPU-accelerated batched velocity IK solving for massive parallelism, ideal for:

- **RL Training**: 4096+ parallel environments in Isaac Gym/Orbit
- **Motion Planning**: Batch trajectory validation
- **Dataset Generation**: Offline batch processing

### Performance

| Batch Size | CPU Sequential | GPU Batched | Speedup |
|------------|---------------|-------------|---------|
| 100        | 3 ms          | 0.8 ms      | 4x      |
| 1000       | 30 ms         | 1.2 ms      | 25x     |
| 4096       | 120 ms        | 1.5 ms      | 80x     |

*Benchmarks on NVIDIA RTX A2000. Larger batch sizes show greater speedups.*

### Quick Start (GPU)

```python
from embodik import solve_velocity_batched

# Batch of IK problems (e.g., 1000 parallel environments)
result = solve_velocity_batched(
    targets_batch,      # List of (task_dim,) arrays
    jacobians_batch,    # List of (task_dim, n_dof) arrays
    constraints_batch,  # List of (n_dof, n_dof) arrays
    lower_bounds_batch,
    upper_bounds_batch,
    use_gpu=True,
    casadi_path="path/to/fn_velocity_solve.casadi"
)

velocities = result.velocities  # (batch_size, n_dof)
```

### Setup

1. **Install CUDA environment:**
   ```bash
   cd embodik
   pixi install -e cuda
   pixi run -e cuda install        # Install embodik in cuda env
   pixi run -e cuda check-cuda     # Verify PyTorch CUDA
   ```

2. **Install CusADi (one-time):**
   ```bash
   pixi run -e cuda install-cusadi   # Clones to ~/.local/cusadi and installs
   pixi run -e cuda check-gpu        # Verify all GPU components
   # Output: CasADi: True, CusADi: True, CUDA: True
   ```

3. **Export and compile CasADi function:**
   ```bash
   # Export symbolic function
   pixi run -e cuda export-casadi

   # Compile to CUDA kernel
   mv fn_velocity_solve.casadi ~/.local/cusadi/src/casadi_functions/
   cd ~/.local/cusadi
   python run_codegen.py --fn=fn_velocity_solve
   ```

4. **Run GPU demos:**
   ```bash
   pixi run -e cuda demo-gpu           # Comprehensive benchmark
   pixi run -e cuda demo-ik-gpu        # Interactive IK with GPU panel
   pixi run -e cuda benchmark-gpu      # Batch IK benchmark
   pixi run -e cuda benchmark-collision  # Collision benchmark
   ```

### Available GPU Tasks

| Task | Description |
|------|-------------|
| `pixi run -e cuda check-cuda` | Verify PyTorch CUDA availability |
| `pixi run -e cuda check-gpu` | Verify CasADi + CusADi + CUDA |
| `pixi run -e cuda install-cusadi` | Install CusADi from GitHub |
| `pixi run -e cuda export-casadi` | Export CasADi velocity solve function |
| `pixi run -e cuda demo-gpu` | Run GPU solver demo/benchmark |
| `pixi run -e cuda demo-ik-gpu` | Interactive IK with GPU benchmark panel |
| `pixi run -e cuda benchmark-gpu` | Batch IK performance benchmark |
| `pixi run -e cuda benchmark-collision` | Collision detection benchmark |
| `pixi run -e cuda test-gpu` | Run GPU-specific tests |

### GPU Collision Detection (Experimental)

EmbodiK also supports GPU-accelerated collision detection via NVIDIA Warp:

```python
from embodik.gpu.warp_collision import compute_collision_distances_batched

# Batch collision queries
result = compute_collision_distances_batched(
    robot_model,
    q_batch,  # (batch_size, n_dof) configurations
    use_gpu=True
)
distances = result.distances  # (batch_size,) minimum distances
```

See [docs/installation.md](docs/installation.md) for detailed GPU setup instructions.

## Testing

```bash
# Run all tests
pixi run test

# Run tests with verbose output
pixi run test-verbose

# Run tests with coverage
pixi run test-cov
```

## Architecture

```
embodik/
├── cpp_core/              # C++ core implementation
│   ├── include/embodik/  # Header files
│   └── src/              # Implementation files
├── python_bindings/       # Nanobind C++ bindings
│   └── src/              # Binding code
├── python/embodik/        # Python package
│   ├── utils.py          # Utility functions
│   └── visualization.py  # Visualization support
├── examples/              # Example scripts
│   ├── 01_basic_ik_simple.py
│   ├── 02_collision_aware_IK.py
│   └── robot_models/     # Robot URDF files
├── docs/                  # Documentation (MkDocs)
└── test/                  # Test suite
```

## Documentation

Full documentation is available at: **https://embodik.github.io/embodik/**

- [Installation Guide](docs/installation.md) - Detailed installation instructions
- [Quickstart](docs/quickstart.md) - Get started in 5 minutes
- [API Reference](docs/api/index.md) - Complete API documentation
- [Examples](docs/examples/index.md) - Example code and tutorials
- [Development Guide](docs/development.md) - Contributing and development

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key principles:
1. Follow the existing code style
2. Add tests for new functionality
3. Ensure numerical accuracy and stability
4. Update documentation for API changes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Copyright (c) 2025 Andy Park <andypark.purdue@gmail.com>**

The MIT License is a permissive license that allows for:
- Commercial use
- Modification
- Distribution
- Private use

While providing liability protection for the authors. This makes it ideal for open-source projects that want to encourage widespread adoption and contribution.

