# Alpha Wheelbase Stall Repro Report (2026-03-26)

## Context

User-observed issue in `hmnd_robot` alpha wheelbase teleop:
- Arm stalls near torso.
- In some cases, stall recovery appears to pull arm into deeper collision.

Goal:
- Reproduce with an alpha-model harness in `embodik`.
- Identify root cause.
- Implement and validate a fix in `embodik`.

## How To Run The Alpha Harness Locally

Scratch harness path (local debug only):
- `examples/_scratch_alpha_wheelbase_repro.py`

Recommended run:

```bash
pixi run build
pixi run python examples/_scratch_alpha_wheelbase_repro.py --matrix --steps 80
```

Matrix modes:
- A: stall ON, `K=2`, balanced
- B: stall OFF, `K=2`, balanced
- C: stall ON, `K=1`, balanced
- D: stall ON, `K=2`, precise

Notes:
- Uses real alpha URDF and real `collisions.json`.
- Reproduces hmnd-like `solve_position_step` behavior, including success-only update semantics.

## Reproduced Issues

Using the alpha harness before solver-side fix:
- Stall recovery could ratchet `stall_handler_current_min_distance()` negative (penetration-escape path).
- In stall ON (`K=2`) runs, worst penetration became deeper than no-recovery baseline.
- This behavior matched the reported "pulling into collision" symptom pattern.

## Root Cause Analysis

### 1) Stall bottleneck classification lacked joint-limit context

`stall_handler_update(...)` used collision distance but did not account for whether the step was joint-limit-dominated:
- In `solve_velocity`, `result.saturated_joints` is computed before stall update, but the handler did not use it.
- In `solve_position` inner loop, temporary `VelocitySolverResult` passed to stall handling did not populate `saturated_joints` at all.

Impact:
- Limit-dominated stalls could be misinterpreted as collision-dominated.
- This allowed unnecessary collision-margin relaxation.

### 2) Single-scalar collision distance was too narrow for `K>1`

With `max_constraints > 1`, the stall logic used a single scalar distance source, which is weaker than using active collision row set:
- Multi-row states can have mixed pair behavior.
- Single-scalar checks are less robust during transitions.

## What We Changed

### A) Stall handler now uses active collision-row distances

In `cpp_core/src/kinematics_solver.cpp`:
- Gather finite distances from `last_collision_debug_list_`.
- Fallback to `last_collision_debug_` only if list is empty.
- Determine:
  - `collision_dist` as min active distance.
  - `any_collision_binding`.
  - `any_penetration`.

### B) Limit-dominated stalls block collision-margin relaxation

In `stall_handler_update(...)`:
- `limit_dominated_stall = !result.saturated_joints.empty()`.
- Collision-margin relaxation and penetration-escape paths are skipped when limit-dominated.

### C) `solve_position` inner loop now populates saturation before stall update

In `solve_position(...)` inner velocity loop:
- Populate `vsr.saturated_joints` from bounds and solved velocities before calling `stall_handler_update(vsr)`.

### D) Added regression tests

In `test/test_stall_handler.py`:
- `TestStallHandlerMultiConstraintRegression::test_multi_constraint_rows_are_active_with_k2`
- `TestStallHandlerMultiConstraintRegression::test_stall_recovery_k2_does_not_worsen_penetration_vs_off`

## Validation Results

### Targeted tests

```bash
pixi run pytest test/test_stall_handler.py -k "StallHandlerMultiConstraintRegression or ClampingDoesNotTriggerStallRelaxation" -v
```

Result:
- 5 passed.

### Full suite

```bash
pixi run test
```

Result:
- 333 passed, 7 skipped.

### Alpha harness matrix (post-fix)

Observed in representative run:
- A (stall ON, K=2, balanced): no deepening vs baseline; stall margin stayed at nominal.
- B (stall OFF, K=2, balanced): baseline remained stalled/infeasible as expected.
- C (stall ON, K=1, balanced): behavior remained consistent with stricter single-row regime.
- D (stall ON, K=2, precise): consistent with A/B trend, no extra pull-in.

Summary:
- The specific "stall recovery drives deeper collision" behavior was mitigated in the alpha harness.

## Remaining Issues / Open Questions

1. Full in-app `hmnd_robot` end-to-end validation still recommended.
   - Harness reproduces solver-side behavior with real model assets, but does not include all UI/runtime integration behavior from the full wheelbase app.

2. Collision tuning presets do not currently encode `max_constraints`.
   - Presets control cache/budget/proximity gating.
   - `max_constraints` remains separately configured (hmnd UI controls this).
   - Further tuning work may be useful for "recommended K per mode" policy.

3. Proximity gating transitions should continue to be watched.
   - Row activation/deactivation near threshold can still produce mode-dependent behavior, especially under high infeasibility.

## Next Steps

1. Run one full hmnd-wheelbase in-app scenario with the exact problematic marker motion and confirm no pull-in.
2. If any residual issue remains, capture per-step logs from hmnd side:
   - status, dq norm, active collision rows, stall margin, near-limit joints.
3. Consider adding optional policy guidance:
   - recommended `max_constraints` by tuning mode (`precise/balanced/speed`) for teleop.
4. Keep scratch harness uncommitted; keep committed coverage in regression tests only.

## Follow-up Clarifications (2026-03-26)

### Is stall handling enabled by default?

- In EmbodiK APIs, stall recovery is opt-in by default:
  - `solve_velocity(..., stall_recovery=false)` default is false.
  - `PositionIKOptions.stall_recovery` default is false.
  - `PositionStepOptions.stall_recovery` default is false.
- In `hmnd_robot` alpha wheelbase teleop flow, it is explicitly enabled in the reusable step options path, so app-level behavior is "enabled by default" unless changed in that stack.

### How `max_constraints` integrates with proximity-gated collision tuning

- `max_constraints` is configured via `configure_collision_constraint(...)` and used as top-K row cap in collision row assembly.
- Collision tuning presets (`kPrecise`, `kBalanced`, `kSpeed`) currently tune:
  - cache behavior / refresh cadence,
  - refinement time budget,
  - proximity-gated activation (enable + activation multiplier).
- Presets do **not** set `max_constraints`; K remains caller/UI controlled.
- Effective active collision rows are therefore:
  - selected top-K candidates, then
  - possibly pruned by proximity gating threshold (`min_distance + activation_margin`),
  - so active rows can be `< K` or even `0` transiently.

### CoM pre-projection revisit status

- Current CoM projection ("project task Jacobians away from violated CoM half-plane normals") is mathematically a heuristic preconditioning step to avoid whole-task SNS collapse under violated constraints.
- It is directionally consistent with the inequality stack but is not equivalent to solving the original unprojected objective under constraints; this is expected.
- Important consistency note:
  - Projection is clearly applied in the primary `solve_velocity` path.
  - `solve_position` inner-loop constraint stack includes CoM rows, but does not mirror all pre-projection steps from `solve_velocity`.
  - This can create path-dependent behavior between APIs and should be revisited if parity matters for teleop stabilization.
- No immediate hard conflict was observed in full test suite, but parity-focused tests are recommended for CoM-heavy scenarios (especially near boundary and with simultaneous collision/limit pressure).

