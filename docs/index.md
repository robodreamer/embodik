# EmbodiK

**High-performance inverse kinematics solver optimized for cross-embodiment VLA/AI applications**

EmbodiK is a modern C++ library with Python bindings designed for robust, high-performance IK behaviors in cross-embodiment scenarios. The name reflects its focus on **embodied** dynamics and constraint handling, making it ideal for humanoid robots and AI/VLA integrations. Built on top of [Pinocchio](https://github.com/stack-of-tasks/pinocchio) and using [Nanobind](https://github.com/wjakob/nanobind) for seamless Python integration.

## Features

- 🚀 **High Performance**: C++ core with optimized Eigen linear algebra
- 🐍 **Python Integration**: Seamless numpy array support via Nanobind
- 🎯 **Multiple Solvers**: Single-step and full multi-task velocity IK
- 🛡️ **Singularity Robust**: Advanced inverse methods for stable solutions
- 🔒 **Constraint Support**: Joint limits and operational space constraints
- 📈 **Solver Diagnostics**: Timing, task scaling, and Jacobian condition-number reporting
- 📊 **Visualization**: Optional Viser-based interactive visualization

## Quick Start

Install EmbodiK, run a maintained example, then adapt the registered-task API
pattern from that script:

```bash
pip install embodik
python -c "import embodik; print(embodik.__version__)"
```

See the [Quickstart](quickstart.md) and [Examples](examples/index.md) pages for
the current `KinematicsSolver`, `add_frame_task()`, and `solve_position_step()`
workflow.

## Installation

See the [Installation Guide](installation.md) for detailed instructions.

```bash
pip install embodik
```

## Documentation

- [Installation Guide](installation.md) — How to install EmbodiK
- [Quickstart](quickstart.md) — Get started in 5 minutes
- [GPU Solvers](gpu_solvers.md) — FI-PeSNS and PPH-SNS GPU-accelerated solvers
- [API Reference](api/index.md) — Complete API documentation
- [Examples](examples/index.md) — Example code and tutorials
- [Development Guide](development.md) — Contributing and development

## Preview

**Franka Panda collision-free IK**

<video autoplay muted loop playsinline controls width="100%" src="assets/media/franka_panda_collision_free_ik.mp4"></video>

**ROBOTIS AI Worker constraint teleop**

<video autoplay muted loop playsinline controls width="100%" src="assets/media/robotis_ai_worker_collision_free_ik.mp4"></video>

**RB-Y1 bimanual whole-body IK**

<video autoplay muted loop playsinline controls width="100%" src="assets/media/rby1_collision_free_ik.mp4"></video>

**Unitree G1 retargeting IK**

<video autoplay muted loop playsinline controls width="100%" src="assets/media/unitree_g1_retargeting_ik.mp4"></video>

**Spot full-body IK**

<video autoplay muted loop playsinline controls width="100%" src="assets/media/spot_fullbody_interactive_ik.mp4"></video>

**Spot locomanipulation mjviser**

<video autoplay muted loop playsinline controls width="100%" src="assets/media/spot_locomanip_interactive_ik_mjviser.mp4"></video>

## License

Apache License 2.0 - see the [LICENSE](https://github.com/robodreamer/embodik/blob/main/LICENSE) file for details. Source: [robodreamer/embodik](https://github.com/robodreamer/embodik)

**Copyright (c) 2026 Andy Park <andypark.purdue@gmail.com>**
