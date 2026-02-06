# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-02-06

### Added
- **Lie-group-aware `integrate()` method on `RobotModel`**: Properly integrates joint
  velocities into configurations using Pinocchio's manifold-aware integration. For standard
  revolute/prismatic joints this is equivalent to `q + v*dt`, but for floating-base (SE3),
  spherical (quaternion), and other non-Euclidean joint types it performs the correct
  exponential-map integration (e.g. quaternion update via `exp`). This replaces naive
  `q += dt * dq` addition which breaks quaternion unit-norm constraints and gives wrong
  results for continuous and floating-base joints.
  - `RobotModel.integrate(q, v, dt=1.0)`: Manifold integration
  - `RobotModel.difference(q0, q1)`: Inverse of integrate; returns tangent vector
  - `RobotModel.neutral_configuration()`: Returns the home/zero configuration (valid
    quaternion for floating-base)
  - `RobotModel.random_configuration()`: Random valid configuration within joint limits
  - `RobotModel.normalize(q)`: Re-normalize quaternion components of configuration

### Fixed
- **Position IK `solve_position()` now uses `pinocchio::integrate()`** instead of naive
  `q += dt * dq` for velocity integration. This fixes incorrect behavior for floating-base
  robots where quaternion components were being updated via simple addition, breaking the
  unit-norm constraint and producing invalid configurations.

### Migration Guide

Users who were manually integrating velocities from `solve_velocity()` using `q += dt * dq`
should switch to `RobotModel.integrate()`:

```python
# Old (v0.4.x) - broken for floating-base / quaternion joints
result = solver.solve_velocity(q)
q = q + dt * np.array(result.solution)  # WRONG for floating-base!

# New (v0.5.0) - correct for all joint types
result = solver.solve_velocity(q)
q = robot.integrate(q, np.array(result.solution), dt)  # Always correct
```

For computing configuration differences:

```python
# Old - broken for floating-base
delta = q1 - q0  # WRONG for floating-base

# New - correct for all joint types
delta = robot.difference(q0, q1)  # Returns tangent vector (size nv)
```

## [0.4.0] - 2026-02-05

### Breaking Changes
- **Removed Python `pin` package from runtime dependencies**: EmbodiK now uses native C++ bindings
  exclusively for all Pinocchio functionality. The `pip pinocchio` (`pin`) package is no longer
  required at runtime, only at build time.
  - This resolves numpy dependency conflicts when using EmbodiK with packages like `validation_robot`
    that have different numpy version requirements
  - All rotation utilities (log3, exp3, quaternion conversions) now use native bindings
  - Collision distance computation now uses native `RobotModel.compute_min_collision_distance()`
  - Visualization defaults to direct Viser (no Pinocchio ViserVisualizer dependency)

### Added
- **Native math utilities via nanobind**: New C++ bindings for rotation/pose math
  - `embodik.log3(R)`: Compute axis-angle from rotation matrix (replaces `pin.log3`)
  - `embodik.exp3(omega)`: Compute rotation matrix from axis-angle (replaces `pin.exp3`)
  - `embodik.matrix_to_quaternion_wxyz(R)`: Convert rotation matrix to (w,x,y,z) quaternion
  - `embodik.quaternion_wxyz_to_matrix(w,x,y,z)`: Convert quaternion to rotation matrix
- **Native collision distance API**: RobotModel now exposes collision distance methods
  - `RobotModel.compute_min_collision_distance()`: Get minimum distance across all collision pairs
  - `RobotModel.compute_collision_distances()`: Get distances for all collision pairs
- **Optional Pinocchio visualization**: `pip install embodik[visualization-pinocchio]` for
  Pinocchio's ViserVisualizer (requires `pin>=3.8.0`)

### Changed
- Visualization now defaults to direct Viser implementation (no `pin` needed)
- `utils.py` functions (`get_pose_error_vector`, `compute_pose_error`, `Rt`) use native bindings
- `visualization.py` quaternion functions use native bindings
- GPU collision fallback uses native `RobotModel.compute_min_collision_distance()`
- Removed `_runtime_deps.py` (lazy Pinocchio import no longer needed)

### Migration Guide
If you were using Pinocchio Python bindings directly through EmbodiK:

```python
# Old (v0.3.x) - required pip pinocchio
import pinocchio as pin
R_error = pose_goal.rotation @ pose_current.rotation.T
error = pin.log3(R_error)
q = pin.Quaternion(R)

# New (v0.4.0) - no pip pinocchio needed
import embodik as eik
R_error = pose_goal.rotation @ pose_current.rotation.T
error = eik.log3(R_error)
w, x, y, z = eik.matrix_to_quaternion_wxyz(R)
```

For collision distance computation:

```python
# Old (v0.3.x) - required pip pinocchio
import pinocchio as pin
pin.updateGeometryPlacements(model, data, collision_model, collision_data)
for i in range(len(collision_model.collisionPairs)):
    pin.computeDistance(collision_model, collision_data, i)
    dist = collision_data.distanceResults[i].min_distance

# New (v0.4.0) - native API
robot_model.update_configuration(q)
dist = robot_model.compute_min_collision_distance()
```

## [0.3.0] - 2026-02-03

### Added
- **PPH-SNS Solver**: Alternative GPU-optimized solver (Parallel Penalized Hierarchical SNS)
  - Soft top-k violation selection using softmax weights
  - Limited rank-1 projector updates (1–2 violators per iteration)
  - Achieves ~632,000 IK solves/second at batch size 10,000
  - Benchmark comparison scripts: `benchmark-solver-comparison`, `benchmark-solver-batched`
- **GPU Acceleration**: Batched velocity IK solving with massive parallelism (100-500x speedup)
  - Achieves ~670,000 IK solves/second at batch size 10,000
  - Ideal for RL training (4096+ parallel environments), motion planning, and dataset generation
  - GPU batched solver via CusADi-compiled CUDA kernels
- **FI-PeSNS Solver**: Fixed-Iteration Penalized eSNS algorithm optimized for GPU
  - Singularity-Robust Inverse (SRINV) for numerical stability
  - Analytical scaling for feasible task scales without iterative saturation
  - Penalty gradient approach for constraint enforcement
  - Fixed iterations for predictable compute time (ideal for real-time RL)
- **GPU Collision Detection**: GPU-accelerated collision detection via NVIDIA Warp
  - Batched collision queries for parallel processing
  - Experimental support for GPU-accelerated self-collision avoidance
- **Parallel Trajectory Tracking Demo**: Visualize 100 robot instances simultaneously tracking different trajectories
  - Interactive demo with Viser visualization
  - Demonstrates GPU parallelization capabilities
  - Achieves ~50,000+ IK solves/second with GPU acceleration
- **Teleoperation IK Example**: Interactive IK control using Seer wireless controller (Xvisio SDK)
- **GPU Benchmarks and Demos**: Comprehensive benchmarking tools for GPU performance
  - Batch IK performance benchmarks
  - FI-PeSNS vs CPU accuracy benchmarks
  - Collision detection benchmarks
  - GPU solver demonstration panels
- **CasADi Integration**: Export and compile velocity IK functions to CUDA kernels
  - Symbolic function export for GPU compilation
  - CusADi integration for CUDA kernel generation
- **GPU Environment Support**: CUDA feature in pixi.toml with GPU-specific tasks
  - `check-cuda`, `check-gpu`, `install-cusadi` tasks
  - GPU demos and benchmarks via pixi tasks
  - CUDA environment configuration

### Changed
- Enhanced examples with GPU acceleration support
- Updated dependencies to include xvisio SDK support for teleoperation examples
- Improved GPU documentation and setup instructions

### Dependencies
- Added `casadi>=3.6.0` and `torch>=2.0.0` to `[gpu]` optional dependencies
- Added `warp-lang>=1.0.0` to `[gpu-collision]` optional dependencies
- Added `xvisio>=0.3.1` to pixi dependencies for teleoperation examples

## [0.2.0] - 2026-01-30

### Added
- Position IK now supports `excluded_joint_indices` to lock joints during solves
- Position IK can enforce collision constraints during iterative solves (when configured)

### Changed
- Velocity solver now groups same-priority tasks to make ordering within a priority level symmetric
- Collision constraint computation now uses cached allow-masks to skip excluded pairs efficiently
- Collision recovery near `min_distance` uses deadband + adaptive push to reduce jitter and stalling
- Collision constraint default `nearest_points_all_pairs` set to true
- Collision debug evaluation is side-effect free (no solver state mutation)

### Fixed
- Position IK now applies collision constraints and excluded joint indices consistently with velocity IK
- Collision constraint pair selection now uses hysteresis across frames for stability

## [0.1.1] - 2025-01-09

### Added
- **PyPI Publishing**: Full wheel building and publishing workflow for TestPyPI and PyPI
- **`embodik-sanitize-env` CLI**: Helper to sanitize `LD_LIBRARY_PATH` for clean pip installs
- **`embodik-examples` CLI**: Tool to list and copy examples for pip-installed users
- **Built-in robot presets**: `panda` and `iiwa` presets work without `robot_presets.yaml`
- **sdist build support**: Source distribution builds now auto-detect PyPI `pin` wheel's Pinocchio

### Changed
- Simplified `_runtime_deps.py` - removed complex preloading logic, now just provides `import_pinocchio()` helper
- Documentation updated to recommend `venv + pip + unset LD_LIBRARY_PATH` as primary user flow
- ViserVisualizer now uses empty GeometryModel instead of None when collisions disabled

### Fixed
- Fixed `KeyError: 'pinocchio/frames/universe'` in ViserVisualizer when `load_collisions=False`
- Fixed RPATH for `_embodik_impl.so` to correctly find `libembodik_core.so`
- Fixed sdist builds failing to find Pinocchio by auto-detecting `pin` wheel's CMake config
- Examples now work correctly for pip-installed users via `embodik-examples --copy`

### Dependencies
- Added `pin>=3.8.0` to build-system requirements for sdist builds
- Added `pyyaml`, `robot_descriptions`, `viser`, `yourdfpy` to `[examples]` optional dependencies

## [0.1.0] - 2025-12-12

### Added
- Initial release of embodiK
- High-performance inverse kinematics solver with hierarchical task resolution
- Python bindings using nanobind
- Support for multiple task types (FrameTask, PostureTask, COMTask, JointTask, MultiJointTask)
- Position and velocity IK solving
- Optional visualization tools using Viser
- Comprehensive test suite
- Example scripts demonstrating various use cases

### Technical Details
- C++17 core library built on Pinocchio and Eigen3
- Python 3.8+ support
- Linux x86_64 support
- CMake-based build system
- scikit-build-core for Python packaging
