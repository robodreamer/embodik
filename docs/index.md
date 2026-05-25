<h1 class="embodik-doc-heading">
  <span class="embodik-sr-only">EmbodiK</span>
  <img class="embodik-doc-wordmark embodik-doc-wordmark--light" src="assets/brand/embodik-wordmark-light.png" alt="">
  <img class="embodik-doc-wordmark embodik-doc-wordmark--dark" src="assets/brand/embodik-wordmark-dark.png" alt="">
</h1>

**High-performance prioritized numerical inverse kinematics for cross-embodiment robotics and VLA/AI applications**

EmbodiK is a modern C++ prioritized numerical IK library with Python bindings
for robot bring-up, teleoperation, retargeting, and whole-body IK examples. It
is built on [Pinocchio](https://github.com/stack-of-tasks/pinocchio), exposed to
Python through [Nanobind](https://github.com/wjakob/nanobind), and keeps
task hierarchy, constraints, and recovery policy in the solver so examples can
stay focused on targets and visualization.

## Design Principles

<div class="embodik-principles" markdown>
<div class="embodik-principle">
  <img src="assets/brand/principles/brand-cross-embodiment.png" alt="">
  <div><strong>Cross-Embodiment</strong><span>Apply one IK policy across arms, humanoids, quadrupeds, and mobile bases.</span></div>
</div>
<div class="embodik-principle">
  <img src="assets/brand/principles/brand-solver-intelligence.png" alt="">
  <div><strong>Solver Intelligence</strong><span>Keep priority, recovery, diagnostics, and constraint policy in the solver.</span></div>
</div>
<div class="embodik-principle">
  <img src="assets/brand/principles/brand-reliable-safe.png" alt="">
  <div><strong>Reliable &amp; Safe</strong><span>Respect limits, collision constraints, CoM support, and contact feasibility.</span></div>
</div>
<div class="embodik-principle">
  <img src="assets/brand/principles/brand-scalable.png" alt="">
  <div><strong>Scalable</strong><span>Use the same solver surface for demos, stress tests, and policy rollout.</span></div>
</div>
</div>

## ✨ Features

- **⚙️ Fast C++ core**: Eigen-based IK routines with Python bindings and numpy support.
- **🎯 Prioritized task hierarchy**: Register frame, posture, relative-pose, and CoM objectives with explicit priorities.
- **🛡️ Solver-owned robustness**: Constraint handling and recovery policy live in the C++ solver, not in example-side guard code.
- **🔒 Hard-constraint handling**: Joint limits, collision constraints, CoM support polygons, contact, and relative-pose checks stay in C++.
- **📈 Diagnostics**: Timing, condition numbers, recovery stage, task scaling, and solver status reporting.
- **🎮 Examples and visualization**: Viser and mjviser demos for Panda, AI Worker, RB-Y1, Unitree G1, and Spot workflows.

## 🚀 Quick Start

Install EmbodiK, run a maintained example, then adapt the registered-task API
pattern from that script:

```bash
python -m pip install --only-binary=:all: "embodik[examples]"
python -c "import embodik; print(embodik.__version__)"
embodik-examples --copy
cd embodik_examples
python 01_basic_ik_simple.py
```

See the [Quickstart](quickstart.md) and [Examples](examples/index.md) pages for
the current `KinematicsSolver`, `add_frame_task()`, and `solve_position_step()`
workflow.

## 📦 Installation

See the [Installation Guide](installation.md) for wheel, source-build, and
troubleshooting instructions.

```bash
python -m pip install --only-binary=:all: embodik
```

## 📚 Documentation

- [Installation Guide](installation.md) - Install wheels, source builds, and optional example extras.
- [Quickstart](quickstart.md) - Build a small prioritized IK example with registered tasks.
- [Examples](examples/index.md) - Run maintained public examples and clone-only development demos.
- [KinematicsSolver API](api/kinematics_solver.md) - Configure tasks, constraints, runtime policy, and diagnostics.
- [RobotModel API](api/robot_model.md) - Load models, compute FK/Jacobians, and query collisions or CoM.
- [GPU Solvers](gpu_solvers.md) - FI-PeSNS and PPH-SNS batch solver notes.
- [Development Guide](development.md) - Local builds, tests, and release workflow.

## 🎬 Preview

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

## 📄 License

Apache License 2.0 - see the [LICENSE](https://github.com/robodreamer/embodik/blob/main/LICENSE) file for details. Source: [robodreamer/embodik](https://github.com/robodreamer/embodik)

**Copyright (c) 2026 Andy Park <andypark.purdue@gmail.com>**
