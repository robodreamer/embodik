# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.13.0] - 2026-03-10

### Added
- New public solver outcome `INFEASIBLE` across C++/Python status enums and stubs to distinguish clean infeasibility from numerical failures.
- Explicit high-level status-message coverage for infeasible outcomes in velocity and position solving, including position-IK non-convergence classification.
- Regression tests for status exposure, infeasibility hinting, and high-level infeasible/non-convergence behavior.

### Changed
- High-level `KinematicsSolver` now classifies numerically stable but unachievable outcomes as `INFEASIBLE` instead of overloading `NUMERICAL_ERROR`.
- `solve_velocity()` now preserves and surfaces backend `status_message` details to downstream callers.
- CoM rollout tests now accept `INFEASIBLE` as a valid constrained-solve outcome while preserving behavioral assertions.

## [0.12.6] - 2026-03-10

### Added
- **Guided solver debugging helper**: `embodik.get_solver_status_hint(status, status_message=None)` returns actionable troubleshooting guidance per solver status and appends backend details when available.
- **`SolverResult.status_message`** is now exposed to Python bindings for explicit failure diagnostics.
- **Explicit solver status coverage in Python**: new enum values exported (`SHAPE_MISMATCH`, `EMPTY_PROBLEM`, `CONSTRAINT_BOUNDS_MISMATCH`, `NON_FINITE_INPUT`) for clearer handling in downstream code.
- New regression test for status hint helper and updated invalid-input tests to validate explicit status categories.

### Changed
- Constraint robustness logic is streamlined via shared internal helpers for half-space bound shaping and violated-row task projection, and applied consistently across collision, CoM, and relative-pose constraints.
- Solver pre-processing now sanitizes non-finite values and inconsistent per-row bounds before backend solve, reducing NaN-driven ambiguous failures.
- Input-validation failures in the backend now return explicit status categories with descriptive messages instead of lumping all failures into generic `INVALID_INPUT`.

## [0.12.5] - 2026-03-10

### Changed
- CoM support-polygon constraints now apply collision-style violation handling: when outside a half-plane, task Jacobians are projected to remove outward motion components while preserving tangential/inward directions.
- CoM constraint bounds now include active outside recovery behavior with minimum inward recovery speed and proportional scaling, improving escape behavior when starting in violation.
- CoM half-plane construction now includes a centroid-side orientation guard so feasibility is robust to polygon winding/orientation.

### Added
- New CoM regression tests covering outside-violation behavior:
  - single-step suppression of further outward motion while outside, and
  - multi-step constrained-vs-unconstrained rollout comparison.

## [0.12.4] - 2026-03-06

### Changed
- **Saturation exit behavior disabled by default**: Velocity-box softening near joint limits (kMinBoundFraction) is now opt-in. Use `enable_saturation_exit_behavior(True)` to restore previous behavior; requires more testing before enabling broadly.

### Added
- **`enable_saturation_exit_behavior(bool)`** / **`saturation_exit_behavior_enabled()`**: Python API to toggle velocity-box softening near limits.

## [0.12.3] - 2026-03-06

### Added
- **Native Rotation helper** (`embodik.Rotation` / `SO3`): Lightweight SO(3) class replacing `scipy.spatial.transform.Rotation`. Constructors: `from_matrix`, `from_quat`, `from_rotvec`, `from_euler`, `identity`. Spatialmath-style shorthands: `Rx`, `Ry`, `Rz`, `RPY`, `AngVec`, `EulerVec`.
- **SE3 property aliases**: `R`, `t`, `A` (spatialmath-style). `SE3.Rt(R, t)` classmethod.
- **Transform benchmarks** (`pixi run benchmark-transforms`): Latency and throughput benchmarks; native r2q/q2r ~24×/2× faster than SciPy.

### Changed
- **r2q / q2r**: Now use native Pinocchio conversion (no SciPy dependency).
- **API documentation**: Added `docs/api/transforms.md` for Rotation, SO3, SE3. Updated `docs/api/utils.md` with r2q, q2r, Rt, compute_pose_error.

## [0.12.2] - 2026-03-06

### Added
- **SE3 Python API**: Composition operator (`T1 * T2`), equality (`==`, `!=`), and point transforms (`act(p)`, `actInv(p)`). Enables `goal_pose = torso_emb * target_se3` and `p_world = T.act(p_local)` without manual matrix math.

## [0.12.1] - 2026-03-06

### Fixed
- Decoupled velocity-solver tolerances to avoid unintended over-regularization:
  - `set_tolerance()` now maps to regularized pseudoinverse damping threshold
    (`regularization_config.epsilon`), preserving historical behavior.
  - Added `set_constraint_tolerance()` to control constraint violation deadband
    (`VelocitySolverConfig::epsilon`) independently.
  - Updated Python bindings/docstrings to reflect the separated semantics.

## [0.12.0] - 2026-03-05

### Added
- **Top-K simultaneous collision constraints** (`max_constraints` parameter on `configure_collision_constraint()`): instead of protecting only the single globally closest pair, the solver now emits up to K independent QP constraint rows — one per closest pair. Each row has its own Jacobian, velocity-damper bounds, and stuck-detection counter, so multiple tight-clearance regions (e.g. base/leg and arm/torso on a wheeled humanoid) are protected simultaneously.
- **`get_last_collision_debug_list()`**: returns a `list[CollisionDebugInfo]` with one entry per active constraint row (up to `max_constraints`). The existing single-pair `get_last_collision_debug()` remains unchanged for backward compatibility.
- **Per-pair hysteresis and stuck detection**: previous active pair indices are tracked as a vector; each pair has its own stuck counter in a map. Pairs that fall out of the top-K are cleaned up automatically each step.

### Changed
- **Continuous recovery ramp** (collision escape improvement): removed the discrete "slightly-inside deadband" special case that used a `gentle_scale = 0.01` multiplier (producing ~0.005 m/s — too weak to overcome EE task pulls). All non-penetrating violations now use a uniform `kCollisionRecoveryScale = 0.2`, producing a proportional recovery push at all violation depths.
- **Task normal projection** (collision escape improvement): when a collision pair is violated (`lower_bound > 0`), the collision normal is projected out of all EE task Jacobians before the SNS solve. Only the approach component (`proj < 0`) is removed; tangential and escape directions remain fully available. This eliminates the "frozen in all directions" symptom where the entire task velocity was scaled to zero by the SNS solver.
- **`kCollisionStuckRecoverySpeed = 0.10 m/s`** added: stuck condition now boosts recovery to at least 0.10 m/s (previously 0.05 m/s, shared with penetration floor).

## [0.11.0] - 2026-03-04

### Added
- **RobotModel constructor with actuated joint names**: `RobotModel(urdf_path, actuated_joint_names=[...], floating_base=False)` builds a reduced model by locking all joints not in the list at their neutral configuration. The resulting model's `nq`/`nv` match the actuated joint count, eliminating index mapping when integrating with external systems (e.g. validationlib) that use reduced configurations. Visual and collision geometry are reduced in sync with the model.

### Changed
- For floating-base robots, the root freeflyer joint is never locked when using the actuated_joint_names constructor.

## [0.10.0] - 2026-03-03

### Added
- **ECTS (Extended Cooperative Task Space)**: Dual-arm coordination tasks (reference: H. A. Park, IROS 2016, doi:10.1109/IROS.2016.7759161).
  - `add_absolute_frame_task()`: Alpha-blended midpoint frame (Serial L/R, Parallel, Blended).
  - `add_relative_frame_task()`: Grasp configuration (right EE relative to left EE).
  - `add_frame_task()`: Single-frame pose task for Orthogonal mode.
  - `embodik.map_ects_mode()`, `map_ects_mode_blended()`: ECTS config mapping.
- **Relative pose constraint**: `configure_relative_pose_constraint()`, `clear_relative_pose_constraint()` for bounds on stored relative target.
- **Interactive example** (`09_dual_arm_ects.py`): Dual LBR iiwa ECTS demo with Viser.
  - Six coordination modes (Orthogonal, Serial L/R, Blended, Parallel).
  - Orthogonal mode: two independent EE pose controls (blue marker → left EE, green → right EE).
  - Mode-change snap keeps arms at current config; markers snap to effective frames.
  - Collision avoidance with primitive collision shapes; collision debug visualization.
- **Dual iiwa URDF builder** (`utils/dual_iiwa_urdf.py`): Composes two iiwa14 arms with shared base.
  - Strips Drake namespace attributes (`drake:acceleration`) for parser compatibility.
  - Replaces high-poly mesh collision (links 6 & 7) with spheres for ~100× faster collision queries.

### Changed
- Renamed `ects.hpp` / `ects.cpp` to `dual_arm_ects.hpp` / `dual_arm_ects.cpp` for clarity. Added IROS 2016 paper citation in headers.

### Fixed
- Drake iiwa mesh-based collision geometry (links 6 & 7) caused ~100× slowdown; now replaced with sphere primitives in dual_iiwa_urdf.

## [0.9.0] - 2026-03-03

### Added
- **Joint-limit barrier gradient task**: `set_joint_limit_barrier_task(margin, gain)` and `clear_joint_limit_barrier_task()` to proactively drive joints away from limits via an analytical barrier gradient in nullspace. Reduces EE return error and stalls when joints saturate during round-trip motion.
- **Pose metrics module** (`embodik.pose_metrics`): C++ implementations with Python bindings for `joint_limit_distance`, `joint_limit_distance_gradient`, `velocity_manipulability`, and `singularity_joint_limit_metric` for diagnostics and monitoring.
- **Interactive example** (`01_basic_ik_simple.py`): Joint limit scaling slider, barrier task toggle, and real-time pose metrics display.

### Fixed
- **Joint-6 spurious drift bug**: `kMinBoundFraction` in velocity box constraints no longer injects headroom *toward* a limit when a joint is already at that limit. Added `kMarginThreshold` (0.01 rad) so softening applies only in the direction away from limits.
- **Limit scaling in example**: When narrowing joint limits via the slider, the current configuration is now clipped to the new limits to avoid spurious recovery behavior.

### Changed
- Moved `kMinBoundFraction`, `kMarginThreshold`, and pose_metrics constants to file-level for clarity.

## [0.8.1] - 2026-02-27

### Changed
- **CoM margin definition**: The fractional margin for polygon shrink now uses
  the mean distance from centroid to vertices (`char_size`) instead of the
  minimum distance (`min_radius`), producing more
  consistent shrink behavior across polygon shapes.

## [0.8.0] - 2026-02-27

### Added
- **CoM support-polygon inequality constraint API**: Enforce the 2D projection
  of the center of mass to remain inside a convex polygon via per-half-plane QP
  constraints.
  - `KinematicsSolver.configure_com_constraint()`: Configure constraint with
    polygon vertices, fractional margin shrink, velocity/acceleration limits,
    and proximity-based activation.
  - `KinematicsSolver.clear_com_constraint()`: Disable CoM constraint.
  - `KinematicsSolver.get_com_proximity_threshold()`: Read back the auto-computed
    activation distance (proximity_fraction × polygon inradius).
  - Convex hull computation, half-plane extraction, and polygon inradius are
    handled internally in C++ — callers only provide raw vertices.
- **Three-layer velocity bound** per half-plane (matching Spot Flex IK pattern):
  1. `com_vel_max` — always active, caps CoM speed in every direction.
  2. `sqrt(2 * com_acc_max * slack)` — always active, smoothly tapers approach
     speed well before the boundary (bounded tipping energy).
  3. `slack / dt` — active only near the boundary (when slack < proximity
     threshold), prevents overshooting in a single time step.
- **Anti-chattering at boundary**: Slack clamped to non-negative for position
  and acceleration terms; epsilon dead-zone (0.1 mm) prevents sign-flip
  oscillation from numerical noise. Matches the `kMarginEpsilon` pattern used
  in joint position-limit constraints.
- **Interactive Viser example** (`examples/08_com_constraint_example.py`):
  CoM sphere, floor projection disk, vertical drop line, inner/outer polygon
  boundaries, and color-coded status (green/orange/red). GUI sliders for all
  constraint parameters.
- **Unit tests** (`test/test_com_constraint.py`): API usage, invalid input
  rejection, Jacobian shape, CoM containment, velocity/acceleration limits,
  and frame transforms (123 total tests pass).

## [0.7.2] - 2026-02-09

### Fixed
- **Improved position-limit robustness near joint bounds**: Increased the
  position-limit margin used for position-based velocity constraints from
  `1e-4` to `1e-3` radians to better absorb solver tolerance and numerical
  drift during integration.
- **Post-solve velocity bound enforcement for limited solves**:
  `KinematicsSolver.solve_velocity(..., apply_limits=True)` now clamps the
  returned velocity vector to the effective per-joint bounds (including the
  intersection of velocity and position-based bounds when both are active).

## [0.7.1] - 2026-02-09

### Fixed
- **Joint position limit enforcement with mixed joint representations**:
  `KinematicsSolver.solve_velocity(..., apply_limits=True)` now maps velocity-space
  indices (`v`) to the correct configuration-space indices (`q`) when building
  position-based velocity constraints. This fixes cases where models include
  continuous joints (`nq=2, nv=1`) ahead of bounded revolute joints and the
  bounded joint could be constrained using the wrong `q` index.
- **Added regression coverage** in `test_joint_limit_recovery.py` for a chain
  containing a continuous joint followed by a bounded revolute joint to ensure
  integrated solutions remain within revolute position limits.

## [0.7.0] - 2026-02-06

### Added
- **Joint index access API on `RobotModel`**: Expose per-joint configuration-space and
  velocity-space indexing, matching Pinocchio's `model.idx_qs` / `nqs` / `idx_vs` / `nvs`
  arrays through a clean name-based Python API.
  - `RobotModel.has_joint(joint_name)`: Check if a joint exists
  - `RobotModel.get_joint_id(joint_name)`: Get the internal joint index
  - `RobotModel.get_joint_config_index(joint_name)`: Starting index in q (idx_q)
  - `RobotModel.get_joint_config_size(joint_name)`: Number of config variables (nq)
  - `RobotModel.get_joint_velocity_index(joint_name)`: Starting index in v (idx_v)
  - `RobotModel.get_joint_velocity_size(joint_name)`: Number of velocity variables (nv)

### Notes
These APIs let users build per-joint q/v mappings without importing Pinocchio directly,
enabling cleaner joint-to-motor mapping code.
For revolute joints, nq=nv=1. For continuous joints, nq=2 (cos/sin) and nv=1. For
floating-base, nq=7 (xyz + quaternion) and nv=6 (linear + angular velocity).

### Migration Guide

Replace direct Pinocchio joint index access:

```python
# Old - required pip pinocchio
import pinocchio as pin
model = pin.buildModelFromUrdf("robot.urdf")
joint_id = model.getJointId("joint1")
idx_q = model.idx_qs[joint_id]
nq = model.nqs[joint_id]
idx_v = model.idx_vs[joint_id]
nv = model.nvs[joint_id]

# New (v0.7.0) - no pip pinocchio needed
import embodik
robot = embodik.RobotModel("robot.urdf")
idx_q = robot.get_joint_config_index("joint1")
nq = robot.get_joint_config_size("joint1")
idx_v = robot.get_joint_velocity_index("joint1")
nv = robot.get_joint_velocity_size("joint1")
```

## [0.6.0] - 2026-02-06

### Added
- **Inverse dynamics API on `RobotModel`**: Full dynamics computations via Pinocchio's
  RNEA and CRBA algorithms, exposed through native C++ bindings.
  - `RobotModel.compute_generalized_gravity(q)`: Compute gravity torque vector g(q)
  - `RobotModel.rnea(q, v, a)`: Full inverse dynamics via Recursive Newton-Euler Algorithm
    (returns tau = M(q)*a + C(q,v)*v + g(q))
  - `RobotModel.compute_mass_matrix(q)`: Joint-space mass/inertia matrix M(q) via CRBA
  - `RobotModel.compute_coriolis(q, v)`: Coriolis + centrifugal torque vector C(q,v)*v
  - `RobotModel.set_gravity(gravity)` / `get_gravity()`: Configure gravity vector
- **Boolean collision checking on `RobotModel`**:
  - `RobotModel.check_collision(min_distance=0.0)`: Returns True if any collision pair
    has distance below threshold. Simpler API than `compute_min_collision_distance()`
    for pass/fail collision queries.

### Notes
These APIs unify robot model functionality that was previously scattered across
separate Pinocchio imports (e.g. `pin.computeGeneralizedGravity`, `pin.rnea`,
downstream packages collision checks) into EmbodiK's single `RobotModel` class,
eliminating the need for a separate `pin` runtime dependency for dynamics.

### Migration Guide

Replace direct Pinocchio dynamics calls:

```python
# Old - required pip pinocchio
import pinocchio as pin
model = pin.buildModelFromUrdf("robot.urdf")
data = model.createData()
tau_gravity = pin.computeGeneralizedGravity(model, data, q)
tau_total = pin.rnea(model, data, q, v, a)

# New (v0.6.0) - no pip pinocchio needed
import embodik
robot = embodik.RobotModel("robot.urdf")
tau_gravity = robot.compute_generalized_gravity(q)
tau_total = robot.rnea(q, v, a)
```

Replace separate collision checking libraries:

```python
# Old - required direct pinocchio
robot_state.check_collision()

# New (v0.6.0) - built into EmbodiK
robot.update_configuration(q)
in_collision = robot.check_collision()           # contact check
too_close = robot.check_collision(min_distance=0.02)  # proximity check
```

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
  - This resolves numpy dependency conflicts when using EmbodiK with other packages
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
