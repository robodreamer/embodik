<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/embodik-wordmark-dark.png">
  <img align="right" src="docs/assets/brand/embodik-wordmark-light.png" alt="EmbodiK logo" width="245">
</picture>

# EmbodiK

![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)
![C++](https://img.shields.io/badge/core-C%2B%2B-00599C?logo=cplusplus&logoColor=white)
![Nanobind](https://img.shields.io/badge/bindings-nanobind-555555)
[![Build](https://github.com/robodreamer/embodik/actions/workflows/ci.yml/badge.svg)](https://github.com/robodreamer/embodik/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/embodik.svg)](https://pypi.org/project/embodik/)
[![Docs](https://img.shields.io/badge/docs-MkDocs-526CFE)](https://robodreamer.github.io/embodik/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/robodreamer/embodik?style=social)](https://github.com/robodreamer/embodik/stargazers)

EmbodiK is a high-performance prioritized numerical inverse kinematics library for cross-embodiment robotics and VLA/AI applications. It pairs a C++ core with Python bindings, exposes robot-model utilities without requiring the Python `pin` package at runtime, and includes interactive examples for collision-aware IK, CoM constraints, teleop, whole-body robots, GPU batch solving, and dual-arm coordination.

## ✨ Overview

EmbodiK is designed for bringing up IK behavior across different robot bodies without rewriting the solver stack for each model. The public examples focus on a practical path:

- 🧭 start with the smallest fixed-base IK loop.
- 🛡️ add collision, joint-limit, and CoM constraints.
- 🎮 connect the same prioritized IK solver path to teleop input.
- 🤖 scale to bimanual, humanoid, and Spot whole-body examples.
- 🧪 use richer clone-only examples for development, stress testing, and policy rollout.

The intent is to keep Python examples lean: visualization and target plumbing
stay in Python, while constraint handling and recovery policy stay in the C++
solver.

The detailed installation notes, API reference, and example walkthroughs live in the official documentation:

https://robodreamer.github.io/embodik/

## 🚀 Quick Start

Fastest path for most users is the wheel-only PyPI install inside a virtual
environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install --only-binary=:all: embodik
python -c "import embodik; print(embodik.__version__)"
```

If that import works, the core package is installed. Copy the examples and run
the fixed-base loop:

```bash
python -m pip install "embodik[examples]"
embodik-examples --copy
cd embodik_examples
python 01_basic_ik_simple.py
```

If pip cannot find a compatible wheel, use the one-shot source installer for
your platform from the [Installation Guide](https://robodreamer.github.io/embodik/installation/).
It creates the venv, installs native dependencies, builds EmbodiK, and runs an
import smoke test.

Published repaired wheels do not require the Python `pin` package at runtime.
Source builds use `pin` or a system Pinocchio install as the native library
provider; keep that provider installed in the environment used to import
EmbodiK.

## ⚡ Experimental GPU whole-body IK

The animation is the default 512-world viewer. The measured reference scale is
1,024 worlds (`--worlds 1024`), which is a separate run from this preview.

<a href="https://robodreamer.github.io/embodik/examples/parallel_trajectory_tracking/">
  <img src="docs/assets/media/gpu_wbc_parallel_showcase_preview.gif?raw=true" alt="Panda, ROBOTIS AI Worker, and Unitree G1 moving through distinct trajectories across 512 fully articulated CUDA worlds" width="960">
</a>

`embodik.gpu.wbc` is experimental and opt-in. The CPU solver remains the stable
default. The GPU path derives joint and task dimensions from the loaded model,
including Panda, AI Worker, and G1. There is no silent CPU fallback.

From a source checkout:

```bash
pixi run setup-gpu-wbc
pixi run -e cuda demo-parallel-tracking
```

The viewer opens at `http://localhost:8080`. Device memory, model switches,
the 1,024-world latency table, and how those samples were measured are in the
[GPU WBC guide](https://robodreamer.github.io/embodik/gpu_solvers/). The
[parallel example guide](https://robodreamer.github.io/embodik/examples/parallel_trajectory_tracking/)
covers headless profiling and RL integration.

## 🎮 Examples

`01_basic_ik_simple.py` is the minimal fixed-base bring-up for a robot preset
or a new URDF. Most scripts default to the Panda preset. Collision, teleop,
CoM, dual-arm coordination, bimanual and humanoid whole-body control, Spot,
and the GPU batch demo are listed in the
[Examples Guide](https://robodreamer.github.io/embodik/examples/), including
Seer controller ports and the mjviser policy flags.

## 🎬 Preview

**Franka Panda collision-free IK**

<a href="https://robodreamer.github.io/embodik/">
  <img src="docs/assets/media/franka_panda_collision_free_ik_preview.gif?raw=true" alt="Franka Panda collision-free IK preview" width="640">
</a>

**ROBOTIS AI Worker constraint teleop**

<a href="https://robodreamer.github.io/embodik/examples/bimanual_whole_body_ik/">
  <img src="docs/assets/media/robotis_ai_worker_collision_free_ik_preview.gif?raw=true" alt="ROBOTIS AI Worker constraint teleop preview" width="640">
</a>

**RB-Y1 bimanual whole-body IK**

<a href="https://robodreamer.github.io/embodik/examples/bimanual_whole_body_ik/">
  <img src="docs/assets/media/rby1_collision_free_ik_preview.gif?raw=true" alt="RB-Y1 bimanual whole-body IK preview" width="640">
</a>

**Unitree G1 retargeting IK**

<a href="https://robodreamer.github.io/embodik/examples/unitree_g1_retargeting_ik/">
  <img src="docs/assets/media/unitree_g1_retargeting_ik_preview.gif?raw=true" alt="Unitree G1 retargeting IK preview" width="640">
</a>

**Spot full-body IK**

<a href="https://robodreamer.github.io/embodik/examples/spot_full_body_ik/">
  <img src="docs/assets/media/spot_fullbody_interactive_ik_preview.gif?raw=true" alt="Spot full-body IK preview" width="640">
</a>

**Spot locomanipulation mjviser**

<a href="https://robodreamer.github.io/embodik/examples/spot_locomanip_mjviser/">
  <img src="docs/assets/media/spot_locomanip_interactive_ik_mjviser_preview.gif?raw=true" alt="Spot locomanipulation mjviser preview" width="640">
</a>

## 🧰 Core Capabilities

- ⚙️ C++ IK core with Nanobind Python bindings.
- 🎯 Hierarchical velocity IK tasks for frames, posture, CoM, and dual-arm coordination.
- 🧮 Fixed-base acceleration eSNS with compatible position, velocity, acceleration, effort, contact-kinematics, and geometry constraints.
- 🛡️ Joint-limit, self-collision, and CoM support-polygon constraints.
- 📈 Solver diagnostics for timing, task scaling, and Jacobian condition-number logging.
- 🧭 Lie-group-aware configuration operations for floating-base, quaternion, and continuous joints.
- 🤖 Native Pinocchio-backed robot model utilities exposed through EmbodiK bindings.
- 👁️ Optional Viser visualization for interactive IK demos.
- ⚡ Model-derived Newton/Warp GPU WBC with device-resident multi-world state, CUDA graph execution, collision constraints, task priority, posture, torso, and centroidal features.

## 📚 Documentation

- [Installation](https://robodreamer.github.io/embodik/installation/) - platform setup, source builds, and troubleshooting.
- [Quickstart](https://robodreamer.github.io/embodik/quickstart/) - first IK calls and solver concepts.
- [Acceleration Solver](https://robodreamer.github.io/embodik/acceleration_solver/) - fixed-base scope, state-box semantics, and collision certification boundaries.
- [Working with Transforms](https://robodreamer.github.io/embodik/transforms/) - transform helpers and SE(3) operations.
- [Examples](https://robodreamer.github.io/embodik/examples/) - public scripts and development-only demos.
- [API Reference](https://robodreamer.github.io/embodik/api/) - Python API generated from docstrings.
- [GPU WBC](https://robodreamer.github.io/embodik/gpu_solvers/) - Newton/Warp setup, model compatibility, constraint coverage, batching, and performance methodology.
- [Development](https://robodreamer.github.io/embodik/development/) - local build, tests, and contributor workflow.

## 🛠️ Development

Use Pixi from a repository clone:

```bash
pixi run build
pixi run test
pixi run docs-build
```

Run examples from the clone:

```bash
pixi run python examples/01_basic_ik_simple.py
pixi run python examples/02_collision_aware_IK.py
pixi run python examples/03_teleop_ik.py
```

## 🗂️ Repository Layout

```text
embodik/
|-- README.md
|-- cpp_core/
|   |-- include/embodik/
|   `-- src/
|-- python_bindings/
|   `-- src/
|-- python/embodik/
|-- examples/
|-- docs/
|-- scripts/
`-- test/
```

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=robodreamer/embodik&type=Date)](https://www.star-history.com/#robodreamer/embodik&Date)

## 📄 License

EmbodiK is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.
Binary wheels may bundle permissively licensed native dependencies; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Cite the project with [CITATION.cff](CITATION.cff).

Developer: Andy Park <andypark.purdue@gmail.com>
