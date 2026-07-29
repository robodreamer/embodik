# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.21.1] - 2026-07-28

### Changed

- Modularized the velocity, position-step, and fixed-base acceleration solver
  implementations into responsibility-focused translation units without
  changing their public APIs, supported capabilities, numerical results, or
  exported symbol surfaces.
- Replaced broad acceleration access to velocity-solver collision internals
  with a source-private provider that preserves collision-row ordering,
  validation, and exact-query accounting.
- Preserved the public solver declaration and exported-symbol surfaces while
  keeping modular velocity, position-step, and acceleration dispatch within
  their established latency budgets.

## [0.21.0] - 2026-07-23

### Added

- Added a fixed-base acceleration-level eSNS API with Python bindings for
  compatible joint state boxes, task and affine rows, effort limits,
  fixed-contact kinematics, geometric constraints, and explicit diagnostics.
- Added sampled velocity-collision row compatibility and continuously certified
  native sphere-sphere collision support with distinct result flags.
- Added velocity/acceleration solver-level selection to the basic and
  collision-aware examples, gated by the active model and safety contract.

### Changed

- Kept velocity collision pair filtering authoritative for the sampled
  acceleration compatibility path, including include/exclude pairs, active
  per-pair floors, row ordering, and exact validation accounting.
- Documented the acceleration solver as fixed-base, scalar-joint, and
  non-hard-real-time; floating-base dynamics, wheel constraints, and dynamic
  contact forces remain outside the stable surface.
- Kept the shared task virtual interface and object layout unchanged; source
  consumers must still rebuild the core library and Nanobind extension together.
- Defined the pre-1.0 versioning contract so new public solver capabilities use
  minor releases while compatible fixes and internal improvements use patches.
- Documented acceleration compatibility for every runnable example; velocity
  remains the default and acceleration selection stays limited to examples that
  explicitly own compatible state and constraint policies.

## [0.20.18] - 2026-07-22

### Changed

- Made finite position-step hierarchy validation aggregate task blocks at the
  same priority, preserving lexicographic priority without freezing productive
  bimanual and whole-body tradeoffs.
- Reused validated backtracked directions for unconstrained velocity-level
  steps and skipped redundant shorter-timestep retries when no admissible
  fraction exists.

### Fixed

- Restored continuous G1 single-target oscillation, whole-body torso-bias
  recovery, and split/Auto bimanual progress while retaining hold-last-target
  behavior at genuinely exhausted joint limits.
- Kept stationary-target regression holds responsive to each commanded task
  while limiting continuity decisions to the highest unsatisfied priority.

## [0.20.17] - 2026-07-22

### Added

- Added manipulability and joint-limit avoidance tasks, including C++ and
  Python APIs, for lower-priority conditioning without overriding commanded
  Cartesian motion.
- Added an opt-in active joint-limit non-worsening policy that preserves inward
  and tangent motion near finite scalar limits, including sampled braking when
  acceleration limits are enabled.
- Added position-step controls for whole-step configuration caps, caller-owned
  continuity identity, continuity reference frames, preferred-lock candidates,
  and protected-task position/orientation tolerances.
- Added explicit position-step hold and preferred-lock diagnostics, APIs to
  synchronize the applied acceleration reference, and post-step collision query
  counters for controllers that hold or post-process solver output.
- Added `PostureTask.set_reference_configuration()` so applications can move a
  regularization anchor without declaring a new posture command every tick.

### Changed

- Kept finite position steps inside velocity, acceleration, active joint-limit,
  collision, and protected-task constraints while preserving useful progress
  through nonlinear refinement.
- Made preferred-lock arbitration evaluate the locked candidate and fallback
  from the same entry state, preserving one physical integration step while
  allowing useful arms-first progress before shared-body fallback.
- Improved collision-row retention, tangent-space task projection, conservative
  post-step validation, and broadphase certification so constrained motion can
  slide along a contact boundary without sacrificing the configured recovery
  floor.
- Treated `max_constraints` as a nominal collision-row budget: penetrating and
  floor-critical pairs remain represented when safety requires additional rows.
- Used the configured structural floor as the recovery and retention target for
  pairs first observed inside the global collision margin, instead of pinning
  every positive-clearance pair at its first observed distance.
- Kept task-local joint exclusions local to their tasks so collision recovery
  can still use globally available joints.
- Updated the public bimanual example to cap Auto-mode torso contribution,
  accept measurable arms-only progress before fallback, and use a lower default
  adaptive-dt scale for smoother large target transitions.
- Updated GitHub checkout, Python setup, Pages, and artifact actions used by CI,
  documentation, wheel, and release workflows.
- Automated version-bump releases after merge to `main`: synchronized package
  metadata and the dated changelog entry are validated before the version tag
  is created and the wheel/sdist/PyPI workflow is dispatched.

### Fixed

- Prevented infeasible or stationary targets from producing residual
  self-motion after useful progress is exhausted; intentional holds are now
  reported explicitly to callers.
- Prevented stationary workspace-boundary targets from accumulating regressive
  velocity-level steps after marker motion stops, while retaining productive
  convergence through smaller position/orientation error trades.
- Improved recovery from fully extended, singular, and active joint-limit
  configurations without requiring a joint-space reset.
- Prevented conditioning, acceleration projection, and lower-priority tasks
  from spending active joint-limit slack or returning finite steps outside the
  configured hard limits.
- Preserved primary task and grasp continuity across preferred-lock handoffs,
  acceleration projection, target changes, and collision-pair switches.
- Preserved collision recovery floors across cache refreshes, active-row
  switches, target changes, and diagnostic probes; non-finite collision evidence
  now fails closed without replacing warm solver state.
- Distinguished recoverable finite numerical stalls from invalid non-finite
  input so valid constrained commands can retry without hiding malformed data.

## [0.20.16] - 2026-06-08

### Added

- Added soft per-joint joint metric controls (`set_joint_metric_weights`,
  `clear_joint_metric_weights`) with Python bindings so applications can bias
  torso-vs-arm contribution without hard-excluding fallback DOFs.
- Added per-joint velocity-limit overrides (`set_joint_velocity_limit`,
  `clear_joint_velocity_limit_overrides`) for safety and hard-lock use cases.
- Added bimanual teleop controls for torso contribution, optional torso marker
  control, task-space speed caps, and a `Torso Policy` dropdown with `Free`,
  `Locked`, and `Decoupled` modes.
- Added regression coverage for joint metrics, velocity-limit overrides,
  arm-joint classifiers, torso/arm ownership, and far-target damping in the
  shared bimanual teleop app.

### Changed

- Made production `pixi run upload-pypi` refuse sdist-only uploads unless an
  explicit recovery override is set, so releases do not reach PyPI without the
  repaired wheel artifacts.
- Updated the shared bimanual solver fixture to use the same arm-joint
  classifiers as the interactive helper.

### Fixed

- Kept decoupled torso-marker ownership asymmetric: end-effector tasks exclude
  torso-chain DOFs, while the torso task keeps arm DOFs available so tools can
  be held in place during torso motion.
- Kept marker-off bimanual control in `Free` whole-body mode by default so torso
  DOFs can support end-effector target motion when no explicit torso marker is
  active.

## [0.20.15] - 2026-06-02

### Added

- **Public docs — collision and robustness guides**: `docs/collision_constraints.md`
  (Speed / Balanced / Precise tuning, pairwise bounds, post-step guards, batch IK notes) and
  `docs/solver_robustness.md` (adaptive dt, elastic band, auto task layout, weighted fallback,
  stall handler, diagnostics), plus a `docs/guides/index.md` reading-path hub and tuning-mode demo
  video.
- **Non-worsening floor regression coverage**: expanded `test/test_collision_non_worsening_floor.py`
  for margin-floor recovery and initially penetrating pairs.

### Changed

- **C++ position-step penetration guard**: tightened post-step rejection when a pair is already
  materially penetrating so accepted steps cannot keep drifting deeper; added margin-floor-aware
  worsen tolerances and recovery seeding for pairs first seen in penetration toward the configured
  structural floor (capped by the active collision margin).
- **MkDocs layout**: reordered Guides and Examples navigation, grouped the home-page doc index by
  intent (Learn / Configure / Reference), and expanded cross-links from API and example pages.

### Fixed

- **Incremental penetration on `SUCCESS`**: C++ position-step integration no longer accepts small
  deepenings when an actionable pair is already inside the material penetration band.

## [0.20.14] - 2026-05-31

### Fixed

- Preserve registered `solve_position_step()` task solve modes and min-error
  fallback settings unless the caller supplies an explicit non-default step
  override.
- Apply the shared solver runtime policy in the adaptive gain harness.

## [0.20.13] - 2026-05-31

### Added

- Added an opt-in per-pair **non-worsening collision recovery floor**
  (`set_non_worsening_collision_floor_enabled`, `set_collision_structural_floor`;
  off by default). When enabled, link pairs first seen closer than `min_distance`
  by construction (e.g. an upper arm near the torso) are maintained at a small
  penetration-prevention clearance instead of triggering an infeasible recovery to
  the full `min_distance` — which otherwise over-constrains the QP (arm coupling,
  jamming, or a frozen solve). Off by default because it changes recovery semantics
  for pairs that start in violation; the shared bimanual whole-body teleop app
  enables it.
- Added shared bimanual Seer/xvisio teleop support for
  `examples/06_bimanual_whole_body_ik.py`. The browser transform controls remain
  available when no controller is connected.
- Added adaptive position/orientation gain tuning controls and headless
  regression coverage for the shared bimanual teleop app.
- Added `PositionStepOptions.primary_solve_mode` and
  `primary_allow_min_error_fallback` so `solve_position_step()` callers can keep
  their primary task mode explicit and opt into one constrained MIN_ERROR retry
  when a SCALE/SCALE_ELASTIC step stalls under active constraints.

### Changed

- A freshly constructed `KinematicsSolver` now defaults to the BALANCED collision
  tuning preset (proximity-gated activation with a 5x band, sphere broadphase, and
  the balanced pair-cache cadence) instead of an unconfigured speed-labelled state.
  Collision checking still only runs once `configure_collision_constraint()` is
  called, so solvers without a collision constraint are unaffected. Callers that
  want a different preset can still call `set_collision_tuning_mode(...)`.

### Fixed

- Kept source-only robot fixtures and validation helpers out of installed/copied
  public examples while preserving them in the repository for validation.

## [0.20.12] - 2026-05-28

### Added

- Added paired-SE3 `TaskTarget.from_se3_pair(...)` bindings so
  multi-target `solve_position_step()` can drive an `AbsoluteFrameTask` from
  the two commanded arm targets while preserving calibrated grasp offsets.

### Changed

- Refreshed the docs landing page and README positioning around solver-owned
  robustness behavior, and added theme-aware EmbodiK logo assets.

### Fixed

- Added the `pypi` GitHub Actions environment to the wheel publish job so PyPI
  trusted publishing receives the expected OIDC environment claim.
- Kept elastic-band joint-limit expansion internal by projecting returned
  position-step configurations back to the robot's true scalar joint limits
  after integration.
- Made multi-target position stepping apply paired arm targets to
  `AbsoluteFrameTask` instead of treating the absolute task as a single-frame
  pose target.

## [0.20.11] - 2026-05-23

### Added
- Added solver runtime diagnostics and `SolverRuntimeConfig` controls for
  solver-owned pose-task layout auto-switching and constrained weighted
  fallback recovery.
- Added `PoseTaskGroup` support for switching between merged and split
  same-priority pose task layouts while preserving existing task ownership.

### Changed
- Enabled constrained weighted fallback recovery by default for
  `solve_position_step()` runtime policy when a prioritized candidate fails but
  a weighted same-priority candidate satisfies solver-owned hard constraints.
- Kept PyPI `pin` build dependencies on the tested Pinocchio 3.x line, pinned
  the matching cmeel urdfdom/tinyxml native dependencies, and made release
  repair scripts locate the cmeel prefix without importing Pinocchio first.
- Included the mesh loader used by generated RB-Y1 collision fixtures in the
  examples extra and made CI check out LFS-backed robot assets for collision
  regression tests.
- Made the RB-Y1 auto-layout regression accept platforms where the merged pose
  layout already succeeds while still requiring split/auto to stay productive.
- Kept optional RB-Y1 `robot_descriptions` regressions from failing CI when the
  external model repository cannot be fetched by the runner.
- Renumbered the highlighted public examples to keep the recommended demos
  first, moved narrower examples out of the numbered path, and standardized
  Viser examples on the shared `http://localhost:8080` default endpoint.
- Simplified public examples to call the C++ solver runtime policy directly
  instead of maintaining Python-side constraint clipping, retry, or last-safe
  guard helpers.

### Removed
- Removed the public `TaskLayout.WEIGHTED_FALLBACK` enum value before release;
  weighted fallback is reported through `SolverRecoveryStage.WEIGHTED_FALLBACK`
  while `TaskLayout` remains limited to merged vs. split pose-task layout.
- Removed internal harness files from the installed/copied public examples
  surface while keeping them available in the source tree for release
  validation.

## [0.20.10] - 2026-05-21

### Changed
- Updated the bundled Spot full-body IK collision asset to match the reference
  collision geometry while preserving the previous visual meshes.
- Added a Viser control for switching the Spot full-body IK example between
  visual geometry, collision geometry, and combined inspection.

### Fixed
- Pointed public one-shot installer commands back to the public gist-backed
  scripts so users without repository access can download them.
- Removed the temporary Spot collision workaround now that the packaged asset
  reproduces the intended recovery behavior directly.
- Expanded the curated Spot self-collision set to include auxiliary body-box
  checks against the distal arm, wrist, gripper fingers, and jaw.

## [0.20.9] - 2026-05-18

### Added
- Bundled public Spot-with-arm URDF assets for the Viser Spot full-body IK example.
- Bundled the MuJoCo Menagerie Spot-with-arm MJCF scene and meshes for the Spot locomanipulation mjviser example.

### Changed
- Made the Spot URDF and MJCF resolvers prefer explicit user overrides, then bundled public assets, before falling back to optional cache/download paths.
- Expanded README and installation docs with copy-paste venv, one-shot installer, and mjviser example run paths for pip users.

### Fixed
- Fixed copied pip examples requiring private Spot model paths or first-run `robot_descriptions` downloads for the default Spot URDF/MJCF runs.

## [0.20.8] - 2026-05-17

### Added
- Added Spot full-body IK and Spot loco-manipulation mjviser examples with preview media, docs, and optional Seer teleop support.
- Added reusable Spot policy runtime, mjviser adapter, Seer teleop, and whole-body IK helpers for future policy examples.
- Added bundled Spot locomanipulation ONNX policy assets tracked through Git LFS.

### Changed
- Decoupled policy inference, rendering, and IK scheduling so the Spot loco-manipulation example can keep policy updates at 50 Hz while IK runs asynchronously.
- Tuned the Spot loco-manipulation IK defaults for balanced collision handling, three active collision rows by default, operator-readable controls, and motion-triggered condition protection.
- Updated docs and README example tables to surface the new Spot `mjviser` and `mjviser-teleop` workflows.

## [0.20.7] - 2026-05-13

### Added
- Added Linux and macOS one-shot installer scripts for source-build fallback paths.
- Added native dependency notices for repaired wheels that bundle Pinocchio, Coal/HPP-FCL, and related libraries.

### Changed
- Simplified installation, quickstart, contributing, and example setup docs around a single wheel-first install path.
- Restored Python 3.11 as a supported install and wheel target alongside Python 3.10 and 3.12.
- Updated example docs to describe the current registered-task APIs instead of stale velocity-IK snippets.

### Fixed
- Fixed repaired-wheel loading after upgrades by making Python bindings prefer packaged native libraries before package-root libraries.
- Pointed public source-build installer commands at repository-owned scripts instead of stale external gist copies.

## [0.20.6] - 2026-05-11

### Added
- Added `SolverResult.condition_number` reporting for singularity/conditioning diagnostics without changing the SRINV damping law.
- Added opt-in joint acceleration limits (`enable_acceleration_limits()` / `set_acceleration_limits()`) with Python bindings, tests, and example controls.
- Added the unified `examples/06_bimanual_whole_body_ik.py` entrypoint, defaulting to AI Worker and optionally supporting RB-Y1 with generated bounded primitive collision geometry.

### Changed
- Renamed shared bimanual example helpers and docs away from AI-worker-specific filenames.
- Reduced AI Worker interactive solve-time spikes during fast gizmo motion by avoiding secondary-objective retries after a productive solve step and reusing cached collision debug distances in the live loop.
- Removed the older AI Worker-specific bimanual entrypoints in favor of the common bimanual example.

## [0.20.5] - 2026-05-09

### Added
- Added `examples/07_unitree_g1_retargeting_ik.py`, a Unitree G1 whole-body retargeting IK example with palm, pelvis, and foot target controls.
- Added G1 CoM visualization, optional self-collision handling, collision-geometry inspection, and headless smoke checks for the new example.
- Bundled the generated G1 collision URDF asset so copied and pip-installed examples can run without source-tree asset paths.

### Changed
- Updated the examples documentation and copied-example coverage for the Unitree G1 workflow.

## [0.20.4] - 2026-05-04

### Added
- Added research metric helpers and regression coverage for fluidity, smoothness, sign-flip, and metric-writer outputs.
- Added internal robustness verification tooling and elastic-band investigation notes.

### Changed
- Improved `SCALE_ELASTIC` recovery fluidity and position-step recovery semantics around constrained teleop stalls.
- Avoided full collision scans on ordinary position-step success paths to reduce unnecessary collision-check overhead.

### Fixed
- Fixed the public AI Worker teleop entrypoint so visual URDF resolution can require the public visual mesh while continuing to use bundled reduced collision geometry.
- Aligned AI Worker recovery tests with current solver intervention semantics.

## [0.20.3] - 2026-04-29

### Fixed
- Made copied/pip-installed AI Worker examples resolve bundled SG2 generated URDF assets before attempting network downloads.
- Added copied-example regression coverage so `embodik-examples --copy` imports AI Worker helpers from the copied examples directory, not the source tree.
- Updated AI Worker example docs and installer guidance around copied example usage.

## [0.20.2] - 2026-04-28

### Added
- Public ROBOTIS AI Worker constrained dual-arm teleop example (`examples/06_bimanual_whole_body_ik.py`) with cached public URDF resolution and reduced collision assets.
- Shared `embodik.interactive_ik` runtime helpers for robust interactive IK stepping, constrained last-safe restoration, and boundary stall classification.
- AI Worker robustness harness and headless regression coverage for collision/CoM boundary behavior.

### Changed
- Merged the simplified public example defaults from `main` and kept adaptive dt, nullspace bias, and balanced collision tuning as the default interactive behavior.
- Simplified public-facing example/runtime code by moving detailed constrained-step handling out of the example layer.
- Updated Pixi test/build tasks to use `python -m pytest` and `python -m pip` so release workflow commands run reliably in the Pixi environment.

### Fixed
- Hardened collision and CoM boundary handling around constrained teleop stalls, including target resync and safe-pose restoration.
- Preserved AI Worker visual meshes while using generated reduced collision geometry for IK collision checks.
- Cleaned up stale/order-dependent regression tests and documented the removed expectations in public test history.

## [0.20.0] - 2026-04-19

### Added
- **`SolverStatus.COLLISION_VIOLATED` (= 9)**: returned by `solve_position_step` when the input q was collision-safe but no integration step could maintain `min_distance`. `q_solution` is set to the last safe configuration. Hardware startup recovery (violated seed) never returns this status.
- **Per-link-pair collision distance overrides**: `solver.set_collision_pair_min_distance(link_a, link_b, dist, activate_when_clear=True)` sets a custom minimum clearance for specific link pairs. Uses deferred-latch activation by default — the override activates the first time the pair achieves the desired clearance, preventing immediate stalls when called from inside the threshold.
- **`PositionStepOptions.adaptive_dt`**: scales the integration timestep proportional to current position error for faster large-jump convergence without tuning gains. Configurable via `adaptive_dt_max_scale` and `adaptive_dt_reference_distance`. Includes proximity-aware cap to prevent overshoot stalls near collision boundaries.
- **Collision boundary tuning API**: `set_collision_repulsion_deadband()`, `set_collision_recovery_scale()`, `set_collision_max_separation_speed_nonpenetrating()` for runtime sweep/autoresearch without recompiling.
- **`evaluate_per_pair_override_violations(q)`**: returns the worst margin across active per-pair override pairs; used internally in post-step rejection.
- Example `collision_hardening_demo.py` (Franka Panda interactive demo with collision status, per-pair sliders, nearest-point visualization).
- Benchmarks: `benchmark_position_step_responsiveness.py`, `benchmark_boundary_oscillation.py`.
- Test suite: `test/test_collision_hardening.py` (7 tests covering all new collision guarantees).

### Changed
- **`collision_repulsion_deadband` default changed from 3 mm to 0**: the 3 mm no-braking zone created a 0.145 m/s velocity discontinuity that drove boundary-bounce oscillation. With 0, the lb formula provides a smooth deceleration ramp all the way to `min_distance` with no bounce.
- **Sparse position-limit constraint rows**: `solve_velocity` now only adds QP rows for joints where position limits actually tighten beyond the velocity limit. Reduces constraint matrix size significantly for high-DOF floating-base robots (e.g. nv=47 → ~15 active rows at mid-range configuration).
- **Adaptive dt full-scan for large steps**: when `adaptive_dt` scales the integration step by > 1.01×, the post-step collision check uses a sphere-broadphase-expanded scan to catch pairs that were outside the active set before the jump.
- `set_collision_pair_min_distance` default changed to `activate_when_clear=True`; use `activate_when_clear=False` for the previous immediate-activation behaviour.
- `examples/02_collision_aware_IK.py` updated with adaptive_dt controls (max_scale=10, ref_dist=0.02 defaults) and `COLLISION_VIOLATED` hold handling.

### Fixed
- **Critical**: post-step rejection in both `solve_position_step` overloads previously checked `kCollisionPenetrationDistanceThreshold` (−1e-5) instead of `min_distance`. Solutions violating min_distance were silently returned as `SUCCESS`, requiring client-side collision checks.
- **Critical**: multi-target `solve_position_step` rejection was never updated from the original −1e-5 threshold (only `solve_position` and single-target were fixed initially).
- **Critical**: collision constraint permanently silenced in SPEED/BALANCED mode — the proximity-gated fast-path had no refresh-interval guard, causing `last_constraint_min_distance_` to stay stale indefinitely. Fixed with refresh-interval check and delta-q guard (~0.1 rad motion invalidates stale cache immediately).
- **Critical**: per-pair distance overrides not enforced in post-step rejection. Global `min_distance` was used, allowing custom pairs to penetrate undetected. Now `evaluate_per_pair_override_violations()` is checked at all three rejection sites.
- Desaturation nudge used `kCollisionPenetrationDistanceThreshold` as safe_candidate threshold; could silently place robot inside `min_distance` within the 0.5 mm `not_worse` tolerance window.
- Stall escape (Jacobian/normal) did not account for per-pair override thresholds.
- Removed `kCollisionViolationDeadband` (1 mm dead zone with zero recovery force inside `min_distance`); recovery ramp now activates immediately at the boundary.

## [0.19.0] - 2026-04-04

### Added
- **Sphere broadphase collision culling**: native AABB-derived bounding spheres skip expensive GJK/EPA distance queries for far-apart collision pairs. Achieves 30x collision speedup in isolation, 5790x combined with lazy reuse. New API: `solver.enable_sphere_broadphase(True)`, `result.collision_sphere_culled_pairs`. Enabled by default in `kSpeed` and `kBalanced` tuning modes.
- **Lazy constraint reuse**: skip collision recomputation when joint velocity is near-zero and previous distance is safe. Eliminates 99%+ of collision steps in smooth trajectories.
- **Post-step collision rejection**: targeted safety check after integration prevents penetration without full-scan overhead.
- **Elastic band joint limit expansion**: temporary joint limit widening for overconstrained solver stalls. New `SCALE_ELASTIC` task solve mode auto-enables elastic band.
- **`evaluate_min_collision_distance()` Python binding**: public API for querying collision distance at arbitrary configurations.
- **Release workflow command**: helper workflow for version bump, changelog, and PyPI publishing.

### Changed
- **Default solve mode**: examples 01-09 now default to `SCALE_ELASTIC` instead of `SCALE`.
- **Collision tuning presets**: `kSpeed` and `kBalanced` modes now enable sphere broadphase automatically.
- **Example 02 defaults**: self-collision and collision debug visualization enabled by default.
- **Collision constraint reuse**: coalesced nearest-points queries and added lazy constraint reuse — when joint velocity is near-zero and cached distance is safe, the entire collision constraint computation is skipped. Reuse is bounded by the cache refresh interval to guarantee periodic full scans.
- **Position-step collision hot path**: eliminated redundant full-scan collision overhead in position-step recovery paths; targeted pair evaluation replaces global distance scans during stall-escape and post-step rejection.
- **Collision debug state preservation**: lazy constraint reuse now preserves collision debug info (closest pair, nearest points) from the previous full computation, ensuring debug visualization remains available during reuse steps.

### Removed
- **Collision warm-start**: removed unused `warm_start_collision_margin` feature from elastic band config (stall handler's reactive relaxation is preferred).

## [0.18.9] - 2026-03-29

### Added
- **Focused solver perf gates**: added targeted benchmark scripts for velocity and position-step collision workloads plus a hardened median-of-medians aggregator to make refactor acceptance decisions robust against run-to-run noise.

### Changed
- **Collision-heavy IK loop efficiency**: reduced per-step overhead by removing avoidable allocations in stall/candidate handling, reserving hot-path containers, and simplifying collision debug evaluation in repeated checks.
- **Position-step recovery hot path**: skip redundant post-step collision rejection when integration does not move configuration, and cache collision-object-to-frame lookups during stall normal-escape attempts.
- **Constraint bound readability**: replaced unbounded torso-row magic literals with the shared `kUnboundedConstraintLimit` constant.

## [0.18.8] - 2026-03-30

### Added
- **Position IK collision diagnostics counters**: `PositionIKResult` now exposes `collision_rejection_count` and `stall_escape_count` (C++ + Python bindings) so callers can quantify rejected penetration steps and successful stall-escape nudges.

### Changed
- **Collision candidate-set safety fallback**: collision row selection now forces a full scan when cached/top-K candidate subsets are underfilled or empty, and guarantees any currently penetrating pair is included in the active set before truncation.
- **Collision debug semantics for top-K mode**: `evaluate_collision_debug()` now reports the globally closest pair rather than only the closest active constraint row, improving observability when the nearest pair is outside the current top-K active rows.
- **Penetration/stall constants normalized**: newly introduced penetration-rejection and stall-escape thresholds in position-step paths are now defined as shared file-level constants instead of repeated inline literals.
- **Stall-escape activation widened for clearance violations**: stalled recovery now treats configured clearance violations (`distance < min_distance`) as escape-eligible even when still non-penetrating, reducing prolonged zero-motion plateaus.
- **Debug instrumentation cleanup**: removed session-specific C++ runtime log sinks used during incident debugging while preserving solver behavior changes validated by runtime evidence.

### Fixed
- **Position-step penetration regression guard**: `solve_position` and both `solve_position_step` overloads now reject integration steps that newly create or significantly deepen collision penetration, reverting to the pre-step configuration when unsafe.
- **Stalled penetration escape robustness**: when stalled in penetration, position-step IK now invalidates stale collision-pair caches and applies Jacobian-based escape nudges (active-row first, then global-pair fallback) to recover from deep contact states more reliably.
- **Hard jump protection in step integration**: sudden deep penetration transitions are now rejected via additional hard-jump thresholds in position-step rejection logic.
- **Recovery gating when stall margin goes non-positive**: collision rejection/escape paths now remain active when collision constraints are enabled, even if the temporary stall-adjusted `min_distance` becomes non-positive.

## [0.18.7] - 2026-03-26

### Changed
- **Stall recovery bottleneck classification**: stall handling now evaluates active collision-row distances from `get_last_collision_debug_list()` (with backward-compatible fallback) instead of relying on a single scalar debug pair.
- **Position-step stall path parity**: `solve_position`/`solve_position_step` now propagate saturated-joint information into stall handling, matching `solve_velocity`-path bottleneck context.

### Fixed
- **Limit-dominated deep-collision pull-in**: when joints are saturated near limits, stall recovery no longer relaxes collision margin as if collision were the primary bottleneck, avoiding false-positive margin ratcheting into penetration.
- **Regression coverage for K>1 collision rows**: added stall-handler tests for multi-constraint collision regimes (`max_constraints > 1`) and parity checks for CoM-constrained `solve_position_step` directional behavior.

## [0.18.6] - 2026-03-26

### Changed
- **Stall-handler bottleneck gating tightened**: stall recovery now relaxes collision `min_distance` only when collision constraints are actually binding (`distance <= current min_distance`), preventing false-positive relaxation when joint-limit clamping is the real bottleneck.
- **Stall-handler docs clarified**: API documentation now explicitly describes collision-only relaxation behavior and why joint-limit-driven stalls do not trigger collision-margin relaxation.

### Fixed
- **Deep-collision pull-in regression near joint limits**: resolved a regression where repeated stall-threshold triggers could ratchet down collision margin while Jacobian clamping constrained motion, allowing tasks to pull the arm further into body collision.
- **Regression test coverage for clamping-stall interaction**: added/validated targeted tests ensuring stall recovery does not keep relaxing collision margins once joint limits become the bottleneck.

## [0.18.5] - 2026-03-26

### Changed
- **Joint-limit handling simplification**: removed the optional joint-limit barrier task API from C++/Python solver surfaces and internal solve path, keeping the near-limit Jacobian clamping strategy as the primary behavior near active limits.
- **Examples/docs cleanup**: removed barrier-specific controls and references from examples and recovery documentation to align with current default solver behavior.

### Fixed
- **Two-joint limit trap behavior**: improved near-limit escape behavior by clamping task Jacobian entries that would push joints further into active limits, reducing apparent teleop stalls when one joint must move away while another remains near a bound.
- **Regression stability around recovery fixtures**: updated hardware-seed recovery checks to deterministic non-regression assertions so tests remain robust across environment-sensitive collision fixtures.

## [0.18.4] - 2026-03-26

### Added
- **Explicit proximity-gating toggle API**: added `set_proximity_gated_collision_activation_enabled(...)` / `get_proximity_gated_collision_activation_enabled()` in C++ and Python so gating can be enabled/disabled without mutating threshold multipliers.
- **Regression coverage for gating toggle semantics**: added tests verifying enable/disable behavior flips row activation while preserving configured activation multiplier and effective margin.

### Changed
- **Proximity-gated collision activation controls**: added optional, min-distance-scaled activation threshold controls (`constraint_activation_multiplier`, effective `constraint_activation_margin`) and wired automatic margin updates when `min_distance` changes.
- **Collision tuning preset behavior**: restored `PRECISE` to conservative legacy-equivalent behavior (cache disabled, no budget, gating off), while `BALANCED` and `SPEED` enable proximity gating with distinct activation multipliers.
- **Collision row assembly path**: collision QP rows are now emitted conditionally based on proximity threshold when gating is enabled, reducing far-from-collision overhead.

## [0.18.3] - 2026-03-25

### Changed
- **Constraint assembly internals unified across APIs**: extracted shared helper paths in the solver implementation for excluded-joint Jacobian masking, inequality-row appending, and torso pose-bound row construction so `solve_velocity` and `solve_position` follow one consistent constraint assembly pattern.

### Fixed
- **Reduced drift risk between `solve_velocity` and `solve_position`**: replaced manual row-offset recomputation in position IK with the same append-editor pattern used in velocity IK, reducing error-prone index math during future constraint additions.
- **Safety-net parity across solve APIs**: `solve_position` now runs `sanitize_solver_inputs(...)` before backend solve, matching `solve_velocity` handling for non-finite goal/Jacobian/constraint inputs.
- **Regression coverage for API consistency**: added CoM-constraint parity coverage in `test/test_com_constraint.py` validating that both solve APIs improve or match unconstrained support-polygon violation behavior.

## [0.18.2] - 2026-03-25

### Fixed
- **`solve_position` CoM constraint gap**: position IK now includes configured CoM support-polygon inequality rows in its constraint stack (including excluded-joint handling), matching expected CoM-limit behavior already present in velocity IK.
- **Regression coverage for CoM-constrained position IK**: added a targeted test in `test/test_com_constraint.py` to verify CoM-constraint configuration reduces or matches unconstrained support-polygon violation in `solve_position`.

## [0.18.1] - 2026-03-23

### Added
- **Shared velocity-box headroom policy type**: added `VelocityBoxHeadroomPolicy` (`enabled`, `fraction`, `activation_margin`) and exposed it under `TorsoPoseConstraintOptions.velocity_box_headroom` in C++ and Python bindings.
- **Velocity-box helper API extension**: `KinematicsSolver.calculate_velocity_box_constraint(...)` now accepts optional `min_velocity_headroom` and `headroom_activation_margin` arguments for explicit per-call headroom policy testing.
- **`PositionStepOptions.torso_constraint`**: optional torso orientation task and torso pose bounds on position-step IK; exposed in C++ and Python bindings (including `.pyi` stubs).

### Changed
- **Torso pose-bound headroom path generalized**: torso pose-bound rows now use the same shared `calculate_velocity_box_constraint(...)` headroom path as other velocity-box constraints, replacing torso-specific post-processing.
- **Docs**: Example 10 teleop docs now describe `velocity_box_headroom` as the preferred API and mark `pose_bound_softening_*` as legacy aliases.

### Fixed
- **`torso_constraint` enforced in `solve_position_step`**: `PositionStepOptions.torso_constraint` is now applied in `solve_position_step` (including the multi-target path) via inequality rows consistent with `solve_position`, so teleop-style position stepping respects torso pose limits.
- **Bindings/stubs**: `PositionStepOptions` in Python bindings and type stubs includes `torso_constraint` alongside other position-step fields.
- **Regression tests**: expanded `test_position_step_joint_options.py` coverage for torso-constrained position stepping.
- **Validation coverage for headroom policy**: regression tests for invalid `torso_constraint.velocity_box_headroom.fraction` and preserved softening/backward-compat tests.

## [0.18.0] - 2026-03-24

### Added
- **`TorsoPoseConstraintOptions.pose_bounds_reference_pose`**: optional 4x4 homogeneous anchor so torso pose **box** limits stay fixed relative to a chosen world pose across repeated `solve_position` calls (e.g. teleop ticks), while remaining fixed across inner IK iterations as before.
- **Torso pose-bound softening controls**: added `TorsoPoseConstraintOptions.pose_bound_softening_enabled` and `pose_bound_softening_fraction` (C++ + Python) as an opt-in way to preserve minimum torso-row velocity headroom near active pose-box limits.
- **Example 10 benchmark diagnostics**: `scripts/benchmark_example10_torso_modes.py` now reports per-status timing/error buckets, `iterations_used` distributions/histograms, and a jump-vs-no-jump plus max-iterations matrix sweep for deeper responsiveness analysis.
- **`scripts/install_embodik_macos.sh`**: one-shot macOS setup (Homebrew `eigen@3` / URDF packages, venv, `SDKROOT` + SDK libc++ include workaround, `CMAKE_PREFIX_PATH` for PyPI `pin` + Homebrew, PyPI or editable `pip install`). Shipped in sdist via `MANIFEST.in`.
- **`scripts/install_embodik_linux.sh`**: one-shot Linux setup (Debian/Ubuntu `apt` deps for Eigen/URDF + toolchain, venv, `CMAKE_PREFIX_PATH` from PyPI `pin`, and PyPI or editable `pip install`). Shipped in sdist via `MANIFEST.in`.

### Changed
- **Torso pose-box rotation error**: `solve_position` torso 6D bounds now use `pinocchio::log3` for the relative rotation \(R_\mathrm{ref}^\top R\) instead of a local `acos`/vee implementation, matching Python `embodik.log3` and improving numerical behavior near \(\pi\).
- **Example 10 interactive tuning workflow**: added bounded-mode behavior presets (`Responsive`, `Stable`, `StrictBounds`), torso bound softening controls, and a `Constraint pressure` indicator to make high-constraint stall risk visible during teleop-style target motion.
- **Docs (Pixi)**: Developer setup documents optional `pixi install`, macOS Xcode CLT expectation, and that CMake adds the SDK `libc++` include path on Apple platforms.
- **Docs / README**: expanded pip/sdist guidance with one-shot installer coverage for both macOS and Linux, including optional `curl` flows and distro-specific dependency notes.
- **CI (`test-macos-arm64`)**: install `eigen@3` and URDF Homebrew packages; set `SDKROOT`, SDK libc++ `CXXFLAGS`, and `CMAKE_PREFIX_PATH="${PIN_PREFIX}:$(brew --prefix)"`.
- **`cibuildwheel` (macOS)**: `brew install eigen@3` plus URDF packages; `Eigen3_DIR` under `opt/eigen@3`; `CMAKE_PREFIX_PATH=/opt/homebrew` to complement CMake’s PyPI `pin` prefix prepended in `CMakeLists.txt`.

### Fixed
- **macOS Python module load after `pip install`**: `INSTALL_RPATH` for `_embodik_impl` now uses separate Mach-O entries (`@loader_path;@loader_path/lib`) instead of one invalid `$ORIGIN:$ORIGIN/lib` string.
- **macOS builds with Xcode CLT**: `CMakeLists.txt` adds the active macOS SDK’s `usr/include/c++/v1` on `APPLE` so `pixi run install` and in-tree Ninja builds find `<cmath>` when CLT libc++ is incomplete; manual pip-only installs still use the documented `CXXFLAGS` workaround where needed.

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
- **Panda joint-limit regression tests**: `test_panda_position_limit_seed.py` (at-limit seed vs incremental teleop-style nudge under `solve_position_step`) and `test_panda_joint_limit_barrier_position_step.py` (barrier overhead / incremental teleop-like stepping).
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
- **`TestDualEEBodyStall` test class**: 5 new tests covering progressive margin relaxation, repeated threshold hits, computation time budget, no-penetration-jump on task disable, and velocity-loop stall reduction — all using the multi-target `solve_position_step` pattern typical of dual-arm teleop loops.

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
- **RobotModel constructor with actuated joint names**: `RobotModel(urdf_path, actuated_joint_names=[...], floating_base=False)` builds a reduced model by locking all joints not in the list at their neutral configuration. The resulting model's `nq`/`nv` match the actuated joint count, eliminating index mapping when integrating with external systems that use reduced configurations. Visual and collision geometry are reduced in sync with the model.

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
- **Interactive example** (`05_dual_arm_ects.py`): Dual LBR iiwa ECTS demo with Viser.
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
- **Interactive Viser example** (`examples/04_com_constraint_example.py`):
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
