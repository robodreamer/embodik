# EmbodiK: Fast Inverse Kinematics with Nanobind

EmbodiK is a high-performance inverse kinematics library built with C++ and Python bindings via Nanobind. It provides both single-task and multi-task velocity IK solvers with singularity-robust inverse methods.

**Author:** Andy Park <andypark.purdue@gmail.com>

## Features

- **High Performance**: C++ core with optimized Eigen linear algebra
- **Python Integration**: Seamless numpy array support via Nanobind
- **Multiple Solvers**: Single-step and full multi-task velocity IK
- **Singularity Robust**: Advanced inverse methods for stable solutions
- **Constraint Support**: Joint limits and operational space constraints

## Installation

### Quick Start

**For end users (installing from PyPI):**
```bash
pip install embodik
```

**For developers (recommended):**
```bash
# Install Pixi (one-time setup)
curl -fsSL https://pixi.sh/install.sh | bash

# Clone and install
git clone https://github.com/embodik/embodik.git
cd embodik
pixi run install
```

> **💡 When to use which?**
> - **Pixi**: Development, automatic dependency management, reproducible builds
> - **pip**: End users, standard Python installation, requires manual system dependencies
>
> See [Installation Guide](docs/INSTALLATION_GUIDE.md) for detailed comparison.

### Prerequisites

**With Pixi:** All dependencies managed automatically ✅

**Without Pixi (manual setup):**
- C++17 compatible compiler
- CMake 3.16+
- Python 3.8+
- Eigen3 development headers (`libeigen3-dev` on Ubuntu)
- Pinocchio library

See [Installation Documentation](docs/installation.md) for detailed instructions.

## API Overview

### Core Types

```python
import embodik as eik

# Basic solver configuration
config = eik.BasicSolverConfig(
    epsilon=1e-6,           # Numerical tolerance
    iteration_limit=20,     # Maximum iterations allowed
    regularization=1e-1     # Tikhonov regularization parameter
)

# Solver result
result = eik.SolverResult(
    solution=[...],           # Joint velocities dq
    status=eik.SolverStatus.SUCCESS,
    computation_time_ms=0.1,
    iterations=1,
    final_error=1e-6,
    task_scales=[...]         # Task scaling factors
)
```

### Multi-Task Velocity IK

embodiK provides a powerful multi-task velocity IK solver that handles task prioritization and constraint satisfaction.

#### Eigen-First API (Recommended)

```python
# Multiple tasks with constraints
goals = [
    np.array([0.1, -0.2]),  # Primary task
    np.array([0.3])          # Secondary task
]

jacobians = [
    np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),  # 2x3
    np.array([[0.0, 0.0, 1.0]])                      # 1x3
]

# Constraint matrix and limits
C = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
lower = np.array([-1e6, -1e6, -1e6])  # Joint limits
upper = np.array([1e6, 1e6, 1e6])

params = {
    "epsilon": 1e-6,
    "precision_threshold": 1e-10,
    "iteration_limit": 20,
    "magnitude_limit": 1e10,
    "stall_detection_count": 2,
    "regularization_epsilon": 1e-6,         # Regularization tolerance
    "regularization_factor": 1e-1,          # Regularization coefficient
}

result = eik.solve_velocity_ik_multi_task_eigen(
    goals, jacobians, C, lower, upper, params
)
```

#### Numpy-First API

```python
# Similar to Eigen-first but with automatic array conversion
result = eik.solve_velocity_ik_multi_task_np(
    goals, jacobians, C, lower, upper, params
)
```

## Parameter Tuning

### Tolerance Settings

- **Epsilon**: `1e-6` - Overall numerical tolerance
- **Precision threshold**: `1e-10` - High-precision constraint satisfaction
- **Regularization epsilon**: `1e-6` - Tolerance for regularized inverse computation
- **Regularization factor**: `1e-1` - Regularization coefficient for stability

### Performance Tuning

- **Iteration limit**: `20` - Maximum solver iterations allowed
- **Magnitude limit**: `1e10` - Maximum allowable solution magnitude
- **Stall detection count**: `2` - Iterations before detecting solver stall

## Performance Characteristics

- **Single-step**: ~0.1ms for 6x7 Jacobian
- **Multi-task**: ~1-5ms for typical 2-3 task problems
- **Memory**: Minimal overhead with Eigen types
- **Scalability**: Linear with problem size

## Testing

```bash
# Run all tests
pixi run test

# Run tests with verbose output
pixi run test-verbose

# Tests should pass successfully
```

## Usage

embodiK provides a clean, modern API for multi-task inverse kinematics:

```python
# Multi-task velocity IK with hierarchical objectives
result = eik.solve_velocity_ik_multi_task_np(
    goals, jacobians, C, lower, upper,
    params={"epsilon": 1e-6, "regularization_factor": 1e-1}
)

# Check if solution was successful
if result.status == eik.SolverStatus.SUCCESS:
    print(f"Solution: {result.solution}")
    print(f"Task scales: {result.task_scales}")
```

## Architecture

```
embodik/
├── cpp_core/           # C++ implementation
│   ├── include/        # Header files
│   └── types.hpp       # Core data structures
├── python_bindings/    # Nanobind bindings
│   ├── src/           # C++ binding code
│   └── python/        # Python package
└── test/              # Test suite
```

## Contributing

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

