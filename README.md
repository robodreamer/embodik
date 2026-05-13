# EmbodiK

![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)
![C++](https://img.shields.io/badge/core-C%2B%2B-00599C?logo=cplusplus&logoColor=white)
![Nanobind](https://img.shields.io/badge/bindings-nanobind-555555)
[![Build](https://github.com/robodreamer/embodik/actions/workflows/ci.yml/badge.svg)](https://github.com/robodreamer/embodik/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/embodik.svg)](https://pypi.org/project/embodik/)
[![Docs](https://img.shields.io/badge/docs-MkDocs-526CFE)](https://robodreamer.github.io/embodik/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/robodreamer/embodik?style=social)](https://github.com/robodreamer/embodik/stargazers)

EmbodiK is a high-performance inverse kinematics library for cross-embodiment robotics and VLA/AI applications. It pairs a C++ core with Python bindings, exposes robot-model utilities without requiring the Python `pin` package at runtime, and includes interactive examples for collision-aware IK, CoM constraints, teleop, GPU batch solving, and dual-arm coordination.

## Overview

EmbodiK is designed for bringing up IK behavior across different robot bodies without rewriting the solver stack for each model. The public examples focus on a practical path:

- start with the smallest fixed-base IK loop.
- add collision-aware behavior and visualization.
- connect the same stepping IK pattern to teleop input.
- use richer clone-only examples for development and stress testing.

The detailed installation notes, API reference, and example walkthroughs live in the official documentation:

https://robodreamer.github.io/embodik/

## Quick Start

Install from PyPI:

```bash
python -m pip install --only-binary=:all: embodik
python -c "import embodik; print(embodik.__version__)"
```

If that import works, the core package is installed.

If pip cannot find a compatible wheel, follow the
platform-specific source-build setup in the [Installation Guide](https://robodreamer.github.io/embodik/installation/).

Optional: follow the [Installation Guide examples setup](https://robodreamer.github.io/embodik/installation/#examples)
once, then run the basic IK demo from the copied example directory:

```bash
cd embodik_examples
python 01_basic_ik_simple.py
```

Published repaired wheels do not require the Python `pin` package at runtime.
Source builds use `pin` or a system Pinocchio install as the native library
provider; keep that provider installed in the environment used to import
EmbodiK.

## Examples

The pip-facing examples are intentionally split by purpose:

| Script | Purpose |
| --- | --- |
| `01_basic_ik_simple.py` | Minimal fixed-base IK bring-up for a robot preset or new URDF. |
| `02_collision_aware_IK.py` | Collision-aware IK behavior demo and advanced tuning surface. |
| `03_teleop_ik.py` | Small adapter showing how teleop input drives the same IK step. |
| `08_com_constraint_example.py` | CoM support-polygon constraint visualization. |
| `09_dual_arm_ects.py` | Dual-arm ECTS and orthogonal coordination modes. |
| `12_bimanual_whole_body_ik.py` | Bimanual whole-body teleop, defaulting to AI Worker and optionally supporting RB-Y1, with CoM and collision handling. |
| `13_unitree_g1_retargeting_ik.py` | Unitree G1 whole-body retargeting IK with CoM and optional collision handling. |

Run them from a copied example directory:

```bash
python 02_collision_aware_IK.py
python 03_teleop_ik.py
python 12_bimanual_whole_body_ik.py
python 13_unitree_g1_retargeting_ik.py
```

Most examples default to the Panda preset. Use `--robot <key>` when a script supports alternate robot presets. See the [Examples Guide](https://robodreamer.github.io/embodik/examples/) for the full catalog, helper conventions, and clone-only development examples.

## Preview

**Franka Panda collision-free IK**

[![Franka Panda collision-free IK preview](docs/assets/media/franka_panda_collision_free_ik_preview.gif)](https://robodreamer.github.io/embodik/)

**ROBOTIS AI Worker constraint teleop**

[![Bimanual whole-body IK preview](docs/assets/media/robotis_ai_worker_collision_free_ik_preview.gif)](https://robodreamer.github.io/embodik/examples/bimanual_whole_body_ik/)

**RB-Y1 bimanual whole-body IK**

[![RB-Y1 bimanual whole-body IK preview](docs/assets/media/rby1_collision_free_ik_preview.gif)](https://robodreamer.github.io/embodik/examples/bimanual_whole_body_ik/)

**Unitree G1 retargeting IK**

[![Unitree G1 retargeting IK preview](docs/assets/media/unitree_g1_retargeting_ik_preview.gif)](https://robodreamer.github.io/embodik/examples/unitree_g1_retargeting_ik/)

## Core Capabilities

- C++ IK core with Nanobind Python bindings.
- Hierarchical velocity IK tasks for frames, posture, CoM, and dual-arm coordination.
- Joint-limit, self-collision, and CoM support-polygon constraints.
- Solver diagnostics for timing, task scaling, and Jacobian condition-number logging.
- Lie-group-aware configuration operations for floating-base, quaternion, and continuous joints.
- Native Pinocchio-backed robot model utilities exposed through EmbodiK bindings.
- Optional Viser visualization for interactive IK demos.
- Experimental GPU batch IK and collision tooling for high-throughput research workflows.

## Documentation

- [Installation](https://robodreamer.github.io/embodik/installation/) - platform setup, source builds, and troubleshooting.
- [Quickstart](https://robodreamer.github.io/embodik/quickstart/) - first IK calls and solver concepts.
- [Working with Transforms](https://robodreamer.github.io/embodik/transforms/) - transform helpers and SE(3) operations.
- [Examples](https://robodreamer.github.io/embodik/examples/) - public scripts and development-only demos.
- [API Reference](https://robodreamer.github.io/embodik/api/) - Python API generated from docstrings.
- [GPU Solvers](https://robodreamer.github.io/embodik/gpu_solvers/) - FI-PeSNS and PPH-SNS batch solver notes.
- [Development](https://robodreamer.github.io/embodik/development/) - local build, tests, and contributor workflow.

## Development

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

Clone-only advanced surfaces live under `examples/` and are not copied by `embodik-examples --copy`.

## Repository Layout

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
|-- examples/
|-- docs/
|-- scripts/
`-- test/
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=robodreamer/embodik&type=Date)](https://www.star-history.com/#robodreamer/embodik&Date)

## License

EmbodiK is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.
Binary wheels may bundle permissively licensed native dependencies; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Developer: Andy Park <andypark.purdue@gmail.com>
