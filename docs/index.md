# EmbodiK

**High-performance inverse kinematics solver optimized for cross-embodiment VLA/AI applications**

EmbodiK is a modern C++ library with Python bindings designed for robust, high-performance IK behaviors in cross-embodiment scenarios. The name reflects its focus on **embodied** dynamics and constraint handling, making it ideal for humanoid robots and AI/VLA integrations. Built on top of [Pinocchio](https://github.com/stack-of-tasks/pinocchio) and using [Nanobind](https://github.com/wjakob/nanobind) for seamless Python integration.

## Features

- 🚀 **High Performance**: C++ core with optimized Eigen linear algebra
- 🐍 **Python Integration**: Seamless numpy array support via Nanobind
- 🎯 **Multiple Solvers**: Single-step and full multi-task velocity IK
- 🛡️ **Singularity Robust**: Advanced inverse methods for stable solutions
- 🔒 **Constraint Support**: Joint limits and operational space constraints
- 📊 **Visualization**: Optional Viser-based interactive visualization

## Quick Start

```python
import embodik
import numpy as np

# Create robot model
model = embodik.RobotModel.from_urdf("path/to/robot.urdf")

# Create kinematics solver
solver = embodik.KinematicsSolver(model)

# Solve IK for a target pose
target_pose = np.eye(4)  # 4x4 transformation matrix
result = solver.solve_position_ik(target_pose)

if result.status == embodik.SolverStatus.SUCCESS:
    print(f"Solution: {result.solution}")
```

## Installation

See the [Installation Guide](installation.md) for detailed instructions.

```bash
pip install embodik
```

## Documentation

- [Installation Guide](installation.md) - How to install EmbodiK
- [Quickstart](quickstart.md) - Get started in 5 minutes
- [API Reference](api/index.md) - Complete API documentation
- [Examples](examples/index.md) - Example code and tutorials
- [Development Guide](development.md) - Contributing and development

## License

MIT License - see the [LICENSE](https://github.com/robodreamer/embodik/blob/main/LICENSE) file for details.

**Copyright (c) 2025 Andy Park <andypark.purdue@gmail.com>**
