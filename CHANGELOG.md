# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`scripts/install_embodik_macos.sh`**: one-shot macOS setup (Homebrew `eigen@3` / URDF packages, venv, `SDKROOT` + SDK libc++ include workaround, `CMAKE_PREFIX_PATH` for PyPI `pin` + Homebrew, PyPI or editable `pip install`). Shipped in sdist via `MANIFEST.in`.

### Changed
- **Docs / README**: expanded macOS pip/sdist guidance (Eigen 3.x via `eigen@3`, `urdfdom_headers` / `urdfdom`, combined `CMAKE_PREFIX_PATH`, Command Line Tools libc++ / `CXXFLAGS` workaround, Python 3.10–3.12 recommendation); troubleshooting table and optional `curl` flow for the installer script.
- **CI (`test-macos-arm64`)**: install `eigen@3` and URDF Homebrew packages; set `SDKROOT`, SDK libc++ `CXXFLAGS`, and `CMAKE_PREFIX_PATH="${PIN_PREFIX}:$(brew --prefix)"`.
- **`cibuildwheel` (macOS)**: `brew install eigen@3` plus URDF packages; `Eigen3_DIR` under `opt/eigen@3`; `CMAKE_PREFIX_PATH=/opt/homebrew` to complement CMake’s PyPI `pin` prefix prepended in `CMakeLists.txt`.

### Fixed
- **macOS Python module load after `pip install`**: `INSTALL_RPATH` for `_embodik_impl` now uses separate Mach-O entries (`@loader_path;@loader_path/lib`) instead of one invalid `$ORIGIN:$ORIGIN/lib` string.
- **macOS builds with Xcode CLT**: document and apply `-isystem $SDKROOT/usr/include/c++/v1` so `<cmath>` / standard C++ headers resolve when the CLT-shipped libc++ tree is incomplete.

## [0.17.0] - 2026-03-23

### Added
- **Solver status/options API expansion for position stepping**: introduced no-progress-oriented status and option surface in C++ and Python bindings so position-step callers can control and interpret bounded-step convergence behavior explicitly.
- **Reference corridor support for position-step IK bounds**: added reference-corridor-aware bound handling to improve guidance around prior solutions and reduce drift during repeated teleop-like updates.
- **High-level collision tuning presets**: added `CollisionTuningMode` (`PRECISE`, `BALANCED`, `SPEED`) and `KinematicsSolver.set_collision_tuning_mode(...)` / `get_collision_tuning_mode()` so users can select collision behavior by intent instead of low-level cache/budget knobs.
- **Python enum exposure for collision tuning modes**: `CollisionTuningMode` is available in Python bindings and wired into solver bindings for direct use in scripts and applications.

### Changed
- **Unified IK bound evaluation path**: consolidated position-step bound and stop-condition checks so joint/task limits, corridor checks, and termination semantics run through one consistent status-classification flow.
- **Default no-progress behavior now classifies early exits explicitly**: when incremental motion fails to make meaningful progress, the solver reports a no-progress outcome by default instead of collapsing into less specific statuses.
- **Interactive example controls for live collision tuning**: examples `02_collision_aware_IK.py` and `03_teleop_ik.py` now expose a `speed/balanced/precise` selector with runtime updates while keeping low-level cache/budget knobs hidden.
- **Preset rationale documented in solver implementation**: clarified why `PRECISE` and `BALANCED` disable time-budget truncation (`budget_us=0`) and why only `SPEED` uses a positive refinement budget for bounded latency.

### Fixed
- **No-progress exit semantics consistency**: aligned default no-progress classification across solver internals and bindings so callers observe stable, deterministic status reporting for stalled incremental updates.

## [0.16.0] - 2026-03-21

### Added
- **Collision-query instrumentation in solver results**: `VelocitySolverResult` now reports `collision_pairs_considered`, `collision_exact_distance_queries`, `collision_bound_culled_pairs`, and `collision_budget_exhausted` for per-step profiling.
- **Collision performance controls in public API**: added `enable_collision_pair_cache(...)`, `set_collision_refinement_time_budget_us(...)`, and `get_collision_refinement_time_budget_us()` on `KinematicsSolver` with Python bindings.
- **Benchmark and equivalence tools**: added `scripts/benchmark_teleop_workloads.py` and `scripts/validate_collision_cache_equivalence.py` for teleop-focused performance measurement and cache-vs-baseline correctness checks.

### Changed
- **Default collision path is optimized**: collision pair caching and conservative budgeted refinement are now enabled with tuned defaults to reduce collision overhead in teleop-style loops while preserving conservative behavior.
- **Dual-iiwa URDF utility flexibility**: `build_dual_iiwa_urdf()` now supports optional mesh-collision replacement via `replace_mesh_collision`.
- **Local perf artifact hygiene**: `.editor/reports/perf_runs/*` is now ignored to keep generated profiling outputs out of git tracking by default.

## [0.15.6] - 2026-03-21

### Added
- **Panda joint-limit regression tests**: `test_panda_position_limit_seed.py` (at-limit seed vs validation-style nudge under `solve_position_step`) and `test_panda_joint_limit_barrier_position_step.py` (barrier overhead / incremental teleop-like stepping).
- **`scripts/benchmark_joint_limit_barrier_overhead.py`**: optional local timing comparison for barrier ON vs OFF under repeated `solve_position_step` calls.

### Changed
- **Joint-limit barrier performance**: cache the velocity→configuration index map across `solve_velocity` calls (rebuild when robot model or `nv` changes); inject barrier objectives in one pass without allocating a full `nv`-dimensional gradient vector.
- **Named barrier/limit constants**: `KinematicsSolver` uses `static constexpr` values for barrier clamps, deadband math, numerical guards, and map sentinels instead of scattered literals.
- **Docs**: `docs/joint_limit_saturation_exit_findings.md` table updated to match the barrier implementation.

## [0.15.5] - 2026-03-20

### Changed
- **Simplified stall recovery strategy**: removed MIN_ERROR fallback toggling and iteration-cap coupling from the stall handler so recovery behavior is driven solely by collision-margin relaxation and restoration.
- **Stall-handler API cleanup**: reduced `configure_stall_handler()` to core parameters (`stall_threshold`, `restore_rate`, `floor_fraction`) and removed fallback-specific state/config paths.

### Fixed
- **Collision margin consistency during recovery**: unified restoration into a single ceiling-limited ramp that keeps effective `min_distance` below live collision clearance, preventing abrupt re-entry into infeasible regions.
- **Regression coverage alignment with simplified behavior**: updated stall-handler tests and docs/stubs to validate stable pull-away recovery and no-regression behavior without fallback-dependent assertions.

## [0.15.4] - 2026-03-20

### Added
- **`pull-away` stall regression test**: added dual-EE body-stall coverage that reproduces a stall-then-reverse-target sequence and asserts quick motion recovery when end-effector goals move away from collision.

### Changed
- **Stall detection now treats `kNumericalError` + near-zero `dq` as stuck**: extends handler triggering beyond `kInfeasible` so sustained capped-iteration failures still progress through recovery logic.
- **Iteration-cap probing under fallback**: during fallback cycles, the solver now allows uncapped probe steps at cycle boundaries to detect newly feasible motion sooner when constraints are being relieved.

### Fixed
- **Collision margin clamp removed for runtime tuning**: `set_collision_min_distance()` now accepts negative values, enabling deep-penetration escape strategies that require temporary sub-zero margins.
- **Deep-penetration stall escape path**: on repeated stalls while interpenetrating, the handler drops effective collision margin below current penetration and exits fallback so motion can resume instead of remaining locked with high compute cost.
- **Margin restoration stability after escape**: restoration is bounded by live collision-distance ceilings to avoid immediately reintroducing infeasible collision rows during recovery.

## [0.15.3] - 2026-03-20

### Changed
- **Stall handler tuning is now parameterized**: replaced inline magic constants in fallback logic with explicit config fields (`relax_drop_fraction`, `healthy_motion_multiplier`, `fallback_iteration_limit`, `task_active_weight_eps`) to make behavior easier to reason about and tune.
- **Stall-handler defaults aligned with current behavior**: `healthy_steps_to_clear` default is now 10 in `StallHandlerConfig`, matching the sustained-motion requirement used by the active fallback implementation.

### Fixed
- **Removed ad-hoc floating-point guards in stall cleanup paths**: stall margin restore/relaxed-state checks now use named tolerance constants rather than repeated hardcoded epsilons.
- **Stall regression test robustness**: threshold-reach assertion in dual-EE body-stall coverage now reflects observed counter accumulation under the updated sustained-motion fallback behavior.

## [0.15.2] - 2026-03-20

### Fixed
- **Broadened stall detection under MIN_ERROR fallback**: stall handler now detects stuck states when `dq ≈ 0` under active MIN_ERROR fallback (returns `kSuccess` with near-zero motion). Uses a wider velocity threshold (100× `dq_stall_eps`) when fallback is already active.
- **Progressive margin relaxation**: fallback deactivation now requires collision margin to be fully restored to nominal before clearing, preventing rapid on/off cycling that re-triggered immediate stalls.
- **Jump prevention on task disable**: when all priority-0 tasks have near-zero weight (e.g., user disabling one EE during stall), the handler immediately restores collision margin to nominal and deactivates fallback, preventing configuration jumps into deep penetration.
- **Computation time bounded during deep stalls**: when the stall handler is in active fallback with sustained infeasibility, the inner SNS solver iteration limit is capped to 5, preventing per-step computation spikes from MIN_ERROR constraint-by-constraint saturation.
- **Healthy-step gating**: fallback deactivation requires sustained *meaningful* motion (above 10× the effective stall threshold), not just any non-zero `dq`.

### Added
- **`TestDualEEBodyStall` test class**: 5 new tests covering progressive margin relaxation, repeated threshold hits, computation time budget, no-penetration-jump on task disable, and velocity-loop stall reduction — all using the multi-target `solve_position_step` pattern matching `validation_robot` teleop usage.

## [0.15.1] - 2026-03-20

### Changed
- **`solve_position_step(..., stall_recovery=True)` lifecycle**: stall-handler state now persists across successive single-step calls so stall counters accumulate correctly in outer loops (e.g., teleop ticks with `max_steps=1`).
- **Stall handler enabling semantics**: `enable_stall_handler()` is now idempotent and preserves accumulated recovery state when called repeatedly.

### Fixed
- **Persistent dual-EE stall handling**: fixed a reset path that prevented `consecutive_stall_steps` from reaching threshold in per-tick stepping IK loops, which blocked stall recovery from activating.
- **Regression coverage**: added/updated stall-handler tests for persistent `solve_position_step` behavior and counter accumulation across repeated single-step calls.

### Removed
- **Solver recovery state machine**: the internal velocity-solver recovery state machine has been removed entirely. It caused oscillation and aggressive velocity spikes without measurable benefit. `set_solver_recovery_enabled()` / `solver_recovery_enabled()` are retained as no-ops for backward compatibility.

## [0.15.0] - 2026-03-20

### Added
- **Stall recovery opt-in for `solve_velocity`**: `KinematicsSolver::solve_velocity(..., stall_recovery=False)` and `solve_velocity_dq(..., stall_recovery=False)` now support one-flag activation of the C++ stall handler in user velocity loops.
- **Stall recovery opt-in for position IK**: `PositionIKOptions.stall_recovery` enables the same C++ stall handling path for `solve_position(...)` with sensible defaults.
- **Tests**: expanded stall-handler coverage for `solve_velocity`/`solve_velocity_dq` and `solve_position` flag behavior (default-off, opt-in behavior, and externally-enabled handler semantics).

### Changed
- **`solve_position` stall handling integration**: stall detection now classifies low-level velocity outcomes consistently with `solve_velocity` (including zero-scale primary-task infeasibility) before feeding the stall handler.

### Fixed
- **Stall-handler consistency across call paths**: aligned status classification and option semantics so `stall_recovery` behaves consistently across `solve_velocity`, `solve_position_step`, and `solve_position`.

## [0.14.4] - 2026-03-20

### Fixed
- **`solve_position_step(..., PositionStepOptions.excluded_joint_indices=...)`**: restored consistent behavior with the existing `solve_position` intent by avoiding temporary mutation of registered task exclusion lists during stepping IK.
- **0.14.3 regression note**: `excluded_joint_indices` in stepping IK could cause undesirable behavior by patching registered task exclusions at runtime; this release removes that side effect.

### Changed
- **Stepping IK exclusions implementation**: `excluded_joint_indices` now uses a non-mutating per-step lock path (no scoped merge/restore of registered tasks).

### Changed
- **`clear_collision_constraint()`**: also resets active collision pair indices, stuck-detection counters, per-pair last-distance caches, and recovery homotopy state (`collision_effective_min_distance_`) so disabling collision does not leave stale internal state.
- **Recovery mode exit (collision health)**: when `max_constraints > 1`, recovery now requires **every** active collision QP row (same entries as `get_last_collision_debug_list()`) to be at or above `min_distance`, not inference from the closest pair alone.

### Removed
- Dead locals in `solve_velocity` recovery bookkeeping (`collision_unhealthy` / `joint_unhealthy` / `unhealthy`) that were never read.

### Docs
- Header comments for internal recovery behavior (trigger, `goals[0]` / first merged priority-0 block, diagnostics via `status_message`) and expanded `clear_collision_constraint()` documentation.
- **`PositionStepOptions`**: clarify that `locked_joint_indices` changes the QP (not the same as post-QP masking); recommend `integration_zero_velocity_indices` alone for typical interactive `solve_position_step` loops unless diagnostics require QP-consistent velocities.

## [0.14.3] - 2026-03-19

### Added
- **`PositionStepOptions`** (aligned with **`PositionIKOptions`** naming where applicable):
  - **`excluded_joint_indices`**: same meaning as `PositionIKOptions::excluded_joint_indices` — merged into exclusions on the driven pose task(s) and every `PostureTask` for each inner `solve_velocity`, then restored (parity with how `solve_position` applies exclusions to its frame + nullspace tasks).
  - **`locked_joint_indices`**: `nv`-indices constrained to **v = 0** in the velocity QP (tight bounds on the identity rows + Jacobian column zeroing on objectives and inequality constraints). Cleared automatically at the end of each `solve_velocity`.
  - **`integration_zero_velocity_indices`**: after each inner `solve_velocity`, listed `nv` components are zeroed on the joint velocity **before** `pinocchio::integrate` (legacy post-QP mask; QP may still have assigned non-zero velocity there).
- **Tests**: `test/test_position_step_joint_options.py`.

## [0.14.2] - 2026-03-19

### Added
- **macOS Apple Silicon (Pixi)**: `osx-arm64` in `pixi.toml` workspace platforms; lockfile regenerated for multi-platform solve.
- **CI**: `macos-14` job builds with venv + pip (`pin` + `CMAKE_PREFIX_PATH`) and runs `pytest`.
- **Wheels workflow**: `cibuildwheel` on `ubuntu-latest` (manylinux x86_64/aarch64) and `macos-14` (arm64) with `auditwheel` / `delocate-wheel` repair; sdist artifact; on version tags, publish combined artifacts to PyPI and create a GitHub release (`generate_release_notes`).

### Changed
- **Nanobind stubgen (CMake)**: use `DYLD_LIBRARY_PATH` on Apple when running stub generation; include `DYLD_LIBRARY_PATH` segments in library search paths.
- **`embodik-sanitize-env`**: on macOS, sanitize `DYLD_LIBRARY_PATH` instead of `LD_LIBRARY_PATH`.
- **Docs**: README and `docs/installation.md` — Apple Silicon notes, `DYLD_LIBRARY_PATH` / `@rpath` troubleshooting, unset both loader vars in “existing Pinocchio” flow.
- **Release automation**: tag-based PyPI upload moved to the wheels workflow; `release.yml` is manual (`workflow_dispatch`) for GitHub release only.
- **`scripts/upload_pypi.sh`**: comments updated for sdist-only script vs cibuildwheel artifacts.

### Fixed
- **CI / wheels**: `find_package(Eigen3)` failures — macOS job installs Eigen via Homebrew and sets `Eigen3_DIR`; `cibuildwheel` Linux runs `dnf install -y eigen3-devel`; macOS wheel builds set `Eigen3_DIR` for Apple Silicon Homebrew. Docs updated for local macOS pip installs.

## [0.14.1] - 2026-03-19

### Added
- **`PositionStepOptions`**: optional task-space speed caps `max_linear_speed` and `max_angular_speed` (≤0 = unlimited); applied inside `solve_position_step` after gain × pose error.
- **`KinematicsSolver::clear_all_target_velocities()`** — clears direct `target_velocity` on every registered task; exposed in Python bindings.
- **Tests**: `test/test_position_step_speed_limits.py` — stale-velocity cleanup after stepping IK and linear speed-cap behavior.

### Changed
- **`solve_position_step`**: after each call, clear direct velocities on **all** registered tasks (single-task path uses the same global cleanup as multi-task). Prevents stale `setTargetVelocity` / stepping leftovers from affecting the next `solve_velocity` or frame.

### Docs
- **`Task::getVelocity()`** — note that direct-velocity magnitude is **not** scaled by `weight_`; use `weight = 0`, `active = false`, or `clearTargetVelocity()` to drop contribution.

## [0.14.0] - 2026-03-19

### Added
- **Stepping position IK** — `KinematicsSolver::solve_position_step(...)`: interactive / per-frame position IK that **does not** swap or create temporary tasks. It sets the target on a registered pose task, **computes pose error internally**, maps it to a task-space velocity using **separate linear/angular gains** (`PositionStepOptions`, independent of task **weight**), runs `solve_velocity()` up to `max_steps` with optional per-step integration (`dt` ≤ 0 uses `solver.dt`). Because it goes through `solve_velocity()`, the same collision and limit machinery applies each sub-step.
- **Multi-task stepping IK** — overload `solve_position_step(current_q, std::vector<TaskTarget>, options)` for **one coordinated** velocity solve per sub-step across multiple targets. Supports registered **`FrameTask`**, **`AbsoluteFrameTask`**, and **`RelativeFrameTask`** (by task name); each `TaskTarget` carries its own 4×4 pose and gains.
- **`PositionStepOptions`**: `position_gain`, `orientation_gain`, `max_steps`, `dt`.
- **`TaskTarget`**: `task_name`, `target_pose`, `position_gain`, `orientation_gain`.
- **Python bindings**: `TaskTarget` and `PositionStepOptions`; `TaskTarget.from_se3(...)`; `solve_position_step` overload accepting **`embodik.Rt` / Pinocchio SE3** for the single-task path (in addition to 4×4 `ndarray`).
- **`FrameTask::getFrameName()`** — public accessor for the controlled frame id.
- **Solver recovery state machine** (velocity / stepping paths): ring buffer of recent solve snapshots; **stuck trigger** on sustained `NUMERICAL_ERROR` with near-zero `||dq||`; **rollback** toward the best recent feasible configuration; **recovery mode** that softens primary EE-style objectives and biases toward collision clearance and limit margin; **per-pair collision margin homotopy** (`collision_effective_min_distance_`) to ease out of persistent margin violations; healthy exit counters before returning to normal.
- **Examples**: `examples/utils/pose_utils.py` (`PoseUtils.make_pose_matrix`); Viser examples **01, 02, 03, 08, 09** updated to use `solve_position_step`, aligned default sliders (task weight / gains / iterations), and shared **EE solve mode** + **allow SCALE→MIN_ERROR fallback** controls where applicable.
- **Tests**: `pytest-benchmark` for `solve_position_step` vs a low-level reference loop; stepping API + `Rt`/SE3 overload coverage (`test_adaptive_task_relaxation.py`); default `allow_min_error_fallback` round-trip (`test_tasks.py`); collision recovery / cold-start / deadlock rollouts (`test_embodik.py`).

### Changed
- **Default task behavior**: `Task::allow_min_error_fallback` now defaults to **`false`**. Code that depended on automatic **SCALE → MIN_ERROR** fallback without setting the flag must opt in with `allow_min_error_fallback = True` (matches interactive examples and coordinated IK expectations).
- **`solve_position_step` performance**: O(1) task lookup via `task_map_`, single `setTargetPose` instead of split setters (one cache invalidate), fixed-size 6D velocity buffer in hot loops, `std::move` merge of `VelocitySolverResult` into `PositionIKResult`, multi-task path resolves dynamic task types once per call (not every sub-step).
- **Examples / scripts**: reuse a single `PositionStepOptions` instance in tight loops where practical; `visualization_example.py` uses `PoseUtils`.

### Fixed
- Clearer invalid-input messages for stepping IK (unknown task name, non-`FrameTask` name in single-task API, unsupported task type in multi-task list).

### Docs
- `docs/api/tasks.md` — document default `allow_min_error_fallback` for user-created tasks.

### Note (test suite)
- Full `pixi run test` on this date: **247 passed**, **7 skipped**, **3 failed** (known debt, not introduced by the stepping-IK API itself):
  - `test_joint6_drift_bug`: `test_x_positive_drive_causes_j6_drift`, `test_y_positive_drive_causes_j6_drift` — joint-6 drift exceeds the current tight threshold for pure ±X/±Y EE drives.
  - `test_panda_narrowed_limits`: `TestStrategyComparison.test_strategy_a_vs_baseline[X]` — strategy A reverse-path stall count vs baseline.

## [0.13.2] - 2026-03-12

### Added
- Collision-boundary jitter guardrails for adaptive relaxation: new regression tests compare `SCALE` vs `MIN_ERROR` near active collision-distance limits using both distance-stddev and sign-flip metrics.

### Changed
- Near-boundary collision handling now applies a small violation dead-zone just below `min_distance` to reduce numerical chatter while preserving recovery behavior for real violations.

## [0.13.1] - 2026-03-12

### Added
- Regression coverage for adaptive task relaxation on Panda, including `MIN_ERROR` progress checks with and without nullspace posture bias.
- Adaptive relaxation is now enabled by default for `SCALE` tasks via automatic fallback to `MIN_ERROR` when scale collapses.

### Changed
- Refined clamped `MIN_ERROR` active-set behavior to preserve hierarchical projector updates and improve constrained progress when saturation evolves.
- Kept the release focused by dropping temporary scripted trace/debug tooling from examples.

### Fixed
- Resolved a `MIN_ERROR` loop-control path that could skip objective projector updates and degrade lower-priority task behavior.
- Resolved `MIN_ERROR` termination fallback handling to avoid pathological stagnation/oscillation in constrained scenarios.

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
