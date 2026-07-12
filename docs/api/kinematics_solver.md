# KinematicsSolver

`KinematicsSolver` provides high-level inverse kinematics entry points:

- `solve_velocity()` for one-step velocity IK over registered tasks.
- `solve_position()` for iterative position IK with an internal objective stack.
- `solve_position_step()` for marker/teleop loops using registered tasks.

## Result Diagnostics

Solver result objects include `condition_number`, the worst Jacobian condition
number observed by the singularity-robust inverse during that solve. Values near
`1.0` indicate well-conditioned task Jacobians; larger values mean the solve is
more sensitive to target changes, numerical tolerances, or near-singular robot
postures.

Use this field for logging, test assertions, and tuning UI feedback. It is not a
manipulability metric and does not indicate a separate solver mode. A high but
finite value can explain weak or unstable-looking motion even when the solve
returns `SUCCESS`.

## Runtime Policy

`SolverRuntimeConfig` stores runtime defaults for interactive loops. See
[Solver Robustness](../solver_robustness.md) for the full picture (adaptive dt, elastic band,
stall handler, auto layout, weighted fallback). Summary:

```python
cfg = solver.runtime_config()
cfg.enable_auto_task_layout = True
cfg.weighted_fallback_enabled = True
solver.configure_runtime(cfg)
```

`enable_auto_task_layout` applies to `PoseTaskGroup` adapters. It lets the solver
try the merged pose layout first, then switch to split position/orientation
tasks when the merged rows are binding poorly. The switch happens at solve
boundaries, not by mutating the caller's configuration after a rejected attempt.

`weighted_fallback_enabled` keeps the prioritized solver authoritative on
success. If the prioritized path does not find a useful step, the solver may
accept a constrained weighted candidate. The candidate uses the same hard
constraint machinery, so it is still subject to configured joint limits,
collision, CoM, relative-pose, contact projection, and linear constraints.

Disable `weighted_fallback_enabled` only when you are running an A/B benchmark
or need to reproduce historical strict-priority behavior. Disable
`enable_auto_task_layout` when you need a fixed merged or split pose-task layout.

## Adaptive dt, elastic band, stall handler

- **Adaptive dt** — `PositionStepOptions.adaptive_dt` scales integration step with position error;
  capped when collision clearance is tight. Configure defaults via `SolverRuntimeConfig`.
- **Elastic band** — `enable_elastic_band()` or `TaskSolveMode.SCALE_ELASTIC` temporarily widens
  joint limit margins when limit-dominated stalls collapse task scale.
- **Stall handler** — `enable_stall_handler(nominal_min_distance)` + `stall_recovery=True` relaxes
  collision margin only when collision rows bind (not on joint-limit stalls).

See [Solver Robustness](../solver_robustness.md).

## Collision recovery floor

Whole-body robots often include link pairs that rest closer than the configured
collision clearance. Enable the non-worsening floor when those structural pairs
should not trigger an infeasible push to the global `min_distance`:

```python
solver.set_non_worsening_collision_floor_enabled(True)
solver.set_collision_structural_floor(0.005)  # metres; default 5 mm
```

For a pair first seen below the global collision margin, the floor is both the
minimum retained clearance and the recovery target. A pair below the floor
recovers toward it; a pair already above the floor may move down toward it
instead of being pinned at its first observed clearance. The target is capped by
the pair's active collision margin, including per-pair overrides.

The floor is **off by default**. Getter/setter pairs:
`get_non_worsening_collision_floor_enabled()` and
`get_collision_structural_floor()`.

## Collision tuning default

Fresh `KinematicsSolver` instances default to `CollisionTuningMode.BALANCED`
(sphere broadphase + conservative pair cache). Override with
`set_collision_tuning_mode()` when benchmarking or reproducing older behavior.

Three presets trade **latency vs distance fidelity**:

| Mode | Typical use | Character |
| --- | --- | --- |
| `SPEED` | High-rate teleop | Aggressive cache + bounded exact-refinement budget (~300 µs) |
| `BALANCED` | Default interactive IK | Conservative cache, no refinement early-stop |
| `PRECISE` | Debug / regression | Cache off; full exact checks every step |

EmbodiK combines these tuned hot paths with **post-step penetration guards** and optional
**non-worsening floors** so fast modes do not silently accept deepening penetration.
See the [Collision Constraints](../collision_constraints.md) guide for tuning walkthroughs,
batch parallelization notes, and measured Speed vs Precise timings.

## Position-step options (teleop)

`PositionStepOptions` fields used by marker/teleop loops:

- `max_steps` — inner IK iterations per control tick (bimanual teleop default: `2`)
- `primary_solve_mode` — mirrors registered EE task solve mode for the primary band
- `primary_allow_min_error_fallback` — when `True`, retry a stalled SCALE/SCALE_ELASTIC
  primary solve once with MIN_ERROR before accepting freeze
- `continuity_command_revision` — optional caller-owned source-command identity; keep a
  non-negative value stable across derived-frame re-expression and increment it when the source
  command changes (`-1` keeps automatic pose-based detection)

See `docs/examples/collision_aware_ik.md` for collision-floor, adaptive dt, elastic band, and
fallback interaction with `configure_collision_constraint()`.

## Position IK Objective Order

`solve_position()` now supports a three-level stack:

1. Primary end-effector frame objective (priority `0`)
2. Optional torso orientation/pose objective (priority `1`)
3. Optional nullspace posture bias (priority `2` when torso is enabled, else `1`)

## PositionIKOptions (torso + nullspace)

```python
import numpy as np
import embodik as eik

opts = eik.PositionIKOptions()
opts.max_iterations = 20
opts.position_gain = 40.0
opts.orientation_gain = 40.0

# Tertiary nullspace bias
opts.nullspace_bias = q_bias
opts.nullspace_gain = 0.01
opts.nullspace_active_joints = [0, 1, 2, 3, 4, 5, 6]
opts.nullspace_joint_weights = np.ones(len(opts.nullspace_active_joints))

# Secondary torso orientation objective
opts.torso_constraint.enabled = True
opts.torso_constraint.frame_name = "base_link"
opts.torso_constraint.orientation_mask = np.array([1.0, 1.0, 0.0])  # roll/pitch only
opts.torso_constraint.orientation_gain = 0.05

# Optional torso pose box bounds (6D: x,y,z,rx,ry,rz)
# Units: x/y/z in meters, rx/ry/rz in radians.
# Bounds are relative to a fixed world-frame torso reference. Inside solve_position,
# that reference does not move across inner iterations. If you call solve_position
# every control tick with a new seed_q, set pose_bounds_reference_pose once (e.g.
# torso.homogeneous() at session start) so the box does not recentre each tick.
half_range = np.array([0.08, 0.08, 0.08, 0.20, 0.20, 0.20])
torso0 = robot.get_frame_pose("base_link")
opts.torso_constraint.pose_bounds_reference_pose = np.asarray(
    torso0.homogeneous(), dtype=float
)
opts.torso_constraint.pose_lower_bounds = -half_range
opts.torso_constraint.pose_upper_bounds = half_range
opts.torso_constraint.pose_axis_mask = np.ones(6)
# Units: [m/s, m/s, m/s, rad/s, rad/s, rad/s]
opts.torso_constraint.velocity_limits = np.full(6, 0.5)
# Units: [m/s^2, m/s^2, m/s^2, rad/s^2, rad/s^2, rad/s^2]
opts.torso_constraint.acceleration_limits = np.full(6, 1.0)
```

Validation rules:

- `nullspace_bias` must match `robot.nq`.
- `nullspace_active_joints` indices must be unique and in `[0, robot.nv)`.
- `nullspace_joint_weights` length must match `robot.nv` (all joints) or
  `len(nullspace_active_joints)` (selected joints).
- Torso pose bounds require both lower and upper vectors, each length `6`.
- Torso 6D ordering is `[x, y, z, rx, ry, rz]` with units
  `[m, m, m, rad, rad, rad]`.
- Optional `torso_constraint.pose_bounds_reference_pose` (4x4 homogeneous): when
  set, pose bounds are measured vs this fixed transform; when unset, the reference
  is the torso frame at `seed_q` for that `solve_position` call only.

## solve_position Usage

```python
target = np.eye(4)
target[:3, 3] = [0.5, 0.2, 0.3]
result = solver.solve_position(seed_q, target, "end_effector", opts)
```

## API Reference

::: embodik.KinematicsSolver
    options:
      show_root_heading: true
      show_root_toc_entry: true
