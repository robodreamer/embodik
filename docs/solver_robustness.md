# Solver robustness and recovery

EmbodiK keeps **recovery policy in the C++ solver**, not in example-side guard code. The same
`KinematicsSolver` that tracks your end-effector also decides how to unstick near joint limits,
collision margins, and singular layouts — while still respecting hard constraints.

!!! info "Related guides"
    **Collision tuning** (Speed / Balanced / Precise, bounds, post-step guards) →
    [Collision Constraints & Tuning](collision_constraints.md).
    **Guide map and reading order** → [Guides overview](guides/index.md).
    **Runnable teleop** → [Teleop IK](examples/teleop_ik.md), [Collision-Aware IK](examples/collision_aware_ik.md).

## Solver-owned vs app-owned

| Concern | Typical naive stack | EmbodiK |
|---------|---------------------|---------|
| Stuck at joint limit | App lowers gains or nudges joints manually | **Elastic band** expands limit margins temporarily; **SCALE_ELASTIC** task mode bundles this |
| Stuck at collision margin | App disables collision or shrinks targets | **Stall handler** relaxes margin only when collision rows bind; **non-worsening floors** for structural pairs |
| Pose task overconstrained | App splits position/orientation tasks by hand | **Auto task layout** switches merged ↔ split at solve boundaries |
| Priority stack infeasible | App drops tasks or switches to Jacobian transpose | **Weighted fallback** accepts a constrained weighted candidate under the same hard limits |
| Large marker jump | App retunes `dt` or gains per scene | **Adaptive dt** scales integration step with error (capped near collision) |
| Penetration creep | Hope QP constraints are enough | **Post-step acceptance** rejects deepening penetration (see collision guide) |

Hard constraints stay in C++ for all of the above: joint limits, collision, CoM polygon,
contact projection, and linear inequalities.

## Runtime policy (`SolverRuntimeConfig`)

Most interactive examples call `configure_solver_runtime_policy(solver)` to enable the default
robust teleop bundle:

```python
import embodik as eik

solver = eik.KinematicsSolver(robot)
cfg = solver.runtime_config()
cfg.enable_auto_task_layout = True
cfg.weighted_fallback_enabled = True
cfg.adaptive_dt = True  # optional; stamp into PositionStepOptions via make_position_step_options()
solver.configure_runtime(cfg)
```

| Flag | Default | What it does |
|------|---------|--------------|
| `weighted_fallback_enabled` | **on** | After a non-success prioritized solve, may accept a **constrained weighted** MIN_ERROR candidate that still satisfies collision, limits, CoM, etc. |
| `enable_auto_task_layout` | off in raw solver; **on in examples** | Toggles **merged 6D pose** vs **split position + orientation** tasks when binding score says one layout fits better |
| `adaptive_dt` | off | When stamped into `PositionStepOptions`, scales integration `dt` with position error |
| `joint_limit_non_worsening_enabled` | off | Inside `joint_limit_non_worsening_margin`, prevents finite scalar joint-limit slack from decreasing while retaining inward and tangent motion |
| `weighted_advisor_enabled` | off | Computes weighted candidate every step for diagnostics without changing authoritative output |

Prioritized SNS remains authoritative on **success**. Fallback and layout switches happen at
**solve boundaries** — the solver does not silently rewrite your registered task list mid-tick.

### Auto task layout

`PoseTaskGroup` adapters can register both a merged `FRAME_POSE` task and split
position/orientation tasks. With `enable_auto_task_layout`:

1. Start on **merged** layout (fewer rows, faster when it works).
2. Switch to **split** when the previous constrained solve’s **binding score** exceeds
   `auto_layout_binding_threshold_high` (default 0.30).
3. Switch back to merged after `auto_layout_cooldown_ticks` consecutive scores below
   `auto_layout_binding_threshold_low` (default 0.15).

Useful when orientation rows fight position rows near singular or boxy workspaces — the solver
picks the layout, not the teleop script.

### Weighted fallback

When the priority stack cannot make progress (`INFEASIBLE`, `NO_PROGRESS`, etc.), the solver
may evaluate a **weighted stacked** velocity solve under the **same hard constraints**. If that
candidate is feasible, it can replace the failed prioritized step (`recovery_stage` reports
`WEIGHTED_FALLBACK` in diagnostics).

Disable only for strict-priority A/B benchmarks — not for production teleop.

### Active joint-limit non-worsening

Enable `joint_limit_non_worsening_enabled` when a limited-ROM robot must not
spend its remaining joint-limit margin to follow an infeasible Cartesian
direction:

```python
cfg = solver.runtime_config()
cfg.joint_limit_non_worsening_enabled = True
cfg.joint_limit_non_worsening_margin = 0.02  # joint coordinates; default 0.04
solver.configure_runtime(cfg)
```

The policy requires position limits to be enabled. For each finite scalar joint
inside the activation margin, the solver adds an inward/tangent velocity
half-space. An outward `SCALE` objective is first represented by its closest
achievable Cartesian tangent objective; exact task motion through the task
nullspace is retained when feasible. The same hard half-space then constrains
`MIN_ERROR`, posture, manipulability, and other lower-priority objectives, so a
secondary task cannot reintroduce the removed outward motion.

The policy is default-off because the useful activation width depends on robot
range of motion and control rate. It does not replace hard position limits, and
it does not relax collision, CoM, contact, velocity, or acceleration
constraints. Non-finite or non-positive margins are rejected when the policy is
enabled.

## Task solve modes

Per-task `TaskSolveMode` controls how strictly a frame task must be met each velocity step:

| Mode | Behavior | Typical use |
|------|----------|-------------|
| `SCALE` | Task rows scale down under conflict (elastic priority) | Default teleop |
| `MIN_ERROR` | Minimum-error objective under hard constraints | Recovery near body; stall unfreeze |
| `SCALE_ELASTIC` | Like `SCALE` + automatic **elastic band** on joint limits | Whole-body teleop near limit saturation |

`PositionStepOptions.primary_allow_min_error_fallback = True` retries a stalled primary
`SCALE` / `SCALE_ELASTIC` step once with **MIN_ERROR** while keeping collision and CoM active —
see [Collision-Aware IK](examples/collision_aware_ik.md).

## Stationary-target continuity

`solve_position_step()` evaluates nonlinear Cartesian merit over a 20-call window after the same
pose target and solve policy remain unchanged. Productive windows continue even when individual
joint steps reverse direction, as can happen while an automatic layout or redundant whole-body
solve makes useful net progress. A window that moves without enough merit reduction, or repeatedly
reverses without strong net progress, activates a hold at the last accepted configuration. The
minimum absolute benefit scales with the number of calls in the window, so a longer policy dwell
cannot hide the same low-rate drift. Aggregate progress also cannot excuse repeatedly worsening
the currently worst commanded target. This prevents persistent null-space motion, target trading,
and limit cycles after a far or constrained target has exhausted useful progress.

The hold is mode-agnostic: it applies to `SCALE`, `SCALE_ELASTIC`, and `MIN_ERROR`, including the
single-target and multi-target APIs. A held target that is already satisfied remains `SUCCESS`;
an unsatisfied target with no useful nonlinear progress reports `NO_PROGRESS`. In both cases,
`q_solution` and the reported applied velocity describe the unchanged configuration.

The window restarts when an explicit step target, registered task target or control state, task
policy, step option, runtime configuration, or task graph changes. This includes lower-priority
posture and torso objectives, so a new command can make progress immediately even when a
higher-priority target is already held. A hold caused by low progress can reopen when a later
candidate becomes efficient. Explicit collision rejection and stall-escape candidates bypass the
merit hold. Samples produced inside a collision margin remain part of the continuity window,
preserving productive tangential sliding without allowing repeated contact-bound cycling to
masquerade as recovery.

When an adapter re-expresses one source command into changing world-frame poses, set
`PositionStepOptions.continuity_command_revision` to a non-negative caller-owned revision. Keep
the value unchanged while the source command is unchanged, and increment it when target geometry
or ownership changes. With this explicit identity, derived pose changes do not restart the
continuity window. Once strong nonlinear progress is exhausted, the hold remains latched until the
revision, task policy, or task graph changes. This avoids low-rate self-motion from repeatedly
re-derived targets while preserving one-tick command recovery. The default value `-1` retains
automatic pose-based identity and its reversible low-progress hold.

Adapters that add their own outer-loop velocity or acceleration continuity must preserve this
hold contract after the native solve. If the source command is unchanged and the accepted
position-step output is a hold, publish the unchanged configuration and zero applied velocity until
the command identity changes. If an adapter synthesizes an equivalent app-owned hold from repeated
stationary residual or reversal motion, latch that app-owned hold on the same command identity too.
Otherwise an outer-loop filter can reintroduce low-rate self-motion even though the IK target is
stationary.

Adapters that recenter a nullspace posture anchor after each accepted step should use
`PostureTask.set_reference_configuration()`. It updates the regularization reference without
declaring a new command on every tick. Use `set_target_configuration()` or
`set_controlled_joint_targets()` for commanded posture changes; those setters restart the
continuity window so useful motion resumes immediately.

## Adaptive integration timestep

Large marker jumps need larger effective steps; near the target, small steps prevent overshoot.
With `adaptive_dt=True`:

```text
effective_dt = dt × clamp(pos_error / reference_distance, 1.0, max_scale)
```

Defaults: `reference_distance=0.05` m, `max_scale=5.0` (override via `SolverRuntimeConfig` or
per-call `PositionStepOptions`).

**Near collision:** if clearance to the active constraint margin is tight, adaptive scaling is
**capped** so a larger `dt` cannot outrun what the margin allows — avoiding “teleport through”
the safety shell.

Enable in teleop:

```python
opts = solver.make_position_step_options()
opts.adaptive_dt = True
result = solver.solve_position_step(q, target, "ee_task", opts)
```

## Elastic band joint limits

When several joints sit on their limits, the QP can lose degrees of freedom and **collapse task
scale**. Elastic band **temporarily widens position limit margins** on saturated joints:

- Expands after repeated limit-dominated stalls (`expand_rate`, `stall_threshold`).
- **Decays** when solves are healthy again (`decay_rate`).
- Capped per joint (`delta_max`, default 0.05 rad).

Enable explicitly:

```python
solver.enable_elastic_band(0.05)
# or per step:
opts.elastic_band = True
```

Or use `TaskSolveMode.SCALE_ELASTIC` on frame tasks for the same mechanism with tuned defaults.

## Stall handler (collision margin recovery)

Distinct from **solver stall** status and from **eval harness Hold%**:

```python
solver.enable_stall_handler(nominal_min_distance=0.04)
opts.stall_recovery = True  # persists across solve_position_step calls
```

After consecutive near-zero-velocity failed solves, the handler may **ratchet down** effective
`min_distance` — but **only when a collision constraint row is actually binding**. Joint-limit
stalls do **not** open collision margin (prevents driving the arm into the body).

Margin **restores** gradually toward nominal when solves succeed again.

Pair with non-worsening floors for structurally close pairs ([Collision Constraints](collision_constraints.md#non-worsening-recovery-floor)).

## CoM support polygon

`configure_com_constraint()` adds half-plane velocity inequalities so the projected CoM stays
inside a support polygon, with:

- **Margin shrink** via `char_size` (mean centroid-to-vertex distance).
- **Velocity and acceleration caps** near the boundary (smooth approach, reduced overshoot).
- Optional **proximity activation** — rows enter the QP only when CoM slack is within a fraction
  of polygon inradius (cheaper when the CoM is comfortably inside).

See [CoM Constraint Example](examples/com_constraint_ik.md).

## Diagnostics worth logging

`VelocitySolverResult` / `PositionIKResult` expose solver state for UI and tests:

| Field | Meaning |
|-------|---------|
| `condition_number` | Worst task Jacobian condition number this step — high ⇒ sensitive posture |
| `binding_score` | How hard inequality rows (limits, collision, CoM) constrained the step |
| `recovery_stage` | `PRIORITIZED` vs `WEIGHTED_FALLBACK` |
| `task_scales` | Per-task scale under SCALE modes |
| `collision_rejection_count` | Post-step guards rejected integration |
| `stall_escape_count` | Jacobian escape nudges out of penetration |

Use these in Viser panels and regression tests instead of guessing from motion alone.

## Batch and parallel workloads

Real-time control uses **one solver per loop**. Offline reachability and morphology sweeps scale
with **process pools** — each worker owns an EmbodiK instance. GPU batch IK is separate; see
[GPU Solvers](gpu_solvers.md).

## Related API

- [Guides overview](guides/index.md) — reading paths for all topic guides
- [KinematicsSolver](api/kinematics_solver.md) — full API reference
- [Tasks](api/tasks.md) — registering frame, posture, CoM, relative-pose tasks
