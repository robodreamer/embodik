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

## Installation

### Quick Start (Recommended)

```bash
# Create a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# Important: Clear any local Pinocchio paths to avoid conflicts
unset LD_LIBRARY_PATH

# Install embodik with example dependencies
pip install "embodik[examples]"

# Verify installation
python -c "import embodik; print(embodik.__version__)"
```

### Running Examples

```bash
# Copy examples to a local directory
embodik-examples --copy

# Run an example
cd embodik_examples
python 01_basic_ik_simple.py --robot panda
```

### Troubleshooting

If you see `ImportError: libboost_*.so...` errors, your shell has `LD_LIBRARY_PATH` pointing to a local Pinocchio build. Fix with:

```bash
unset LD_LIBRARY_PATH
```

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

- **`01_basic_ik_simple.py`** - Basic IK solving with interactive visualization
- **`02_collision_aware_IK.py`** - Collision-aware IK with self-collision avoidance
- **`robot_model_example.py`** - Robot model usage and configuration
- **`visualization_example.py`** - Interactive 3D visualization examples

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
```

See the [Examples Documentation](docs/examples/index.md) for detailed guides.

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

