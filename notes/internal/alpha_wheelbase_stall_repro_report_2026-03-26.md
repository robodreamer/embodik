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

## Build + Test in `hmnd_robot` Wheelbase Example

Use this runbook to validate the current EmbodiK solver changes in the actual wheelbase app flow.

### 1) Build and test EmbodiK first

From `embodik` repo root:

```bash
pixi run build
pixi run test -k stall -q
```

### 2) Install the local EmbodiK into hmnd Pixi env (no dependency reinstall)

From `hmnd_robot` repo root:

```bash
pixi run python -m pip install --no-deps --force-reinstall "/home/andypark/Projects/repos/embodik"
```

Quick import/path sanity check:

```bash
pixi run python -c "import embodik,sys; print(sys.version); print(embodik.__file__)"
```

Expected:
- import succeeds,
- printed module path resolves to the hmnd Pixi environment site-packages,
- install timestamp matches the current local rebuild window.

### 2b) Actual low-level rebuild path used during debugging (cp311)

During incident debugging we also used a direct CMake target rebuild for rapid iteration on the Python extension:

```bash
cmake --build build-cp311 --target _embodik_impl -j8
```

Notes:
- This rebuilds only the `_embodik_impl` extension from an existing `build-cp311` tree.
- It was used to quickly refresh the cp311 backend while avoiding full dependency reinstall cycles.
- After rebuilding, we still verified that `hmnd_robot` imported the intended freshly built module.

### 3) Run wheelbase viser example and exercise the known scenario

From `hmnd_robot` repo root:

```bash
pixi run alpha_wheelbase_viser
```

Recommended manual validation sequence:
1. Enable interactive IK with collision enabled.
2. Reproduce the near-torso right-arm motion where pull-in/stall previously occurred.
3. Toggle torso lock/unlock and repeat near-contact motion.
4. Switch solve mode between `SCALE` and `MIN_ERROR` and confirm behavior does not regress.

### 4) Optional runtime diagnostics during hmnd validation

If additional telemetry is needed, use existing hmnd-side opt-in logging:

```bash
export HMND_EMBODIK_DEBUG_LOG=/tmp/embodik_debug.jsonl
export HMND_EMBODIK_DEBUG_EVERY=5
pixi run alpha_wheelbase_viser
```

Then inspect:
- solver status transitions (`SUCCESS` / `INFEASIBLE` / `NUMERICAL_ERROR`),
- `dq_norm` near-zero lock windows,
- collision distance trend and stall margin behavior.

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

## Progress Update (2026-03-30)

### Stabilization outcomes

- Removed session-specific runtime debug instrumentation from EmbodiK C++ and wheelbase viser Python after collecting enough evidence.
- Kept solver-side behavioral fixes that were supported by runtime evidence:
  - clearance-aware stall escape activation (not penetration-only),
  - hard-jump rejection guard for sudden deep penetration transitions,
  - recovery/rejection logic now remains active even when stall handler relaxes configured `min_distance` to non-positive values.
- Latest long run showed major reduction in severe deep-penetration events compared with earlier reproductions.

### Current residual behavior

- Residual lock plateaus can still appear as long `INFEASIBLE` / `NUMERICAL_ERROR` windows with near-zero applied motion.
- Dominant residual message remains: `primary task scale collapsed to zero under active constraints`.
- These plateaus are less catastrophic than prior deep pull-in failures, but still a usability issue for fluid teleop.

## Remaining Issues / Next Steps (2026-03-30)

1. Add focused regression tests for *plateau* behavior:
   - prolonged infeasible plateau with active torso secondary task,
   - ensure escape/rejection counters continue to engage while `stall_min_distance <= 0`.
2. Add solver-side guardrail for extended `primary task scale collapsed` streaks:
   - candidate approach: bounded, deterministic fallback step (collision-safe) after N consecutive collapsed ticks.
3. Re-check task-priority interactions in wheelbase app:
   - especially transitions where only `secondary_torso_yaw` remains active.
4. Keep a lightweight debug restore path (see **Debug Logging Mechanism (Restore Notes)** in this document) so runtime telemetry can be re-enabled quickly without reintroducing ad-hoc code.

## Debug Logging Mechanism (Restore Notes)

The session-specific debug instrumentation used during this investigation was intentionally removed from both EmbodiK C++ and wheelbase viser Python once behavior stabilized.

To re-enable quickly in a future incident:

1. Prefer existing opt-in Python telemetry hook in wheelbase runtime:
   - `HMND_EMBODIK_DEBUG_LOG=/tmp/embodik_debug.jsonl`
   - optional: `HMND_EMBODIK_DEBUG_EVERY`, `HMND_EMBODIK_DEBUG_COLLISION_THRESHOLD`.
2. For C++ solver internals, add temporary local logging behind a feature flag/env var only:
   - avoid hardcoded absolute paths and session IDs,
   - emit only under lock/failure predicates (`INFEASIBLE`, `NUMERICAL_ERROR`, near-zero `dq`).
3. Keep instrumentation in clearly marked blocks and remove after verification.
4. Keep fields minimal but actionable:
   - status, dq norm, collision min distance, active rows, saturated joints,
   - reject/escape counters and step acceptance path (full/reduced/reverted).

## Archive Handoff Snapshot (2026-03-30)

This section captures the full state at archive time so a future restart does not
require reconstructing context from chat history.

### Implemented solver-side behavior changes (kept)

- Position-step post-solve rejection for steps that newly create or significantly
  worsen penetration.
- Rejection backoff path (fractional step retries) before full reversion.
- Hard-jump rejection guard for sudden deep-penetration transitions.
- Stall escape activation widened beyond penetration-only:
  - includes configured-clearance violations during stalled states.
- Collision cache/candidate safety improvements:
  - force full scan on underfilled/empty cached candidate subsets,
  - ensure currently penetrating pairs are represented in active rows.
- Limit-dominated stall handling improvements:
  - avoid treating joint-limit-dominated stalls as collision-dominated,
  - preserve/propagate saturated-joint context in position-IK stall paths.
- Recovery/rejection logic remains active even when stall-relaxed
  `min_distance` becomes non-positive.

### Python wheelbase integration behavior (kept)

- Apply `q_solution` when solver reports explicit intervention
  (`collision_rejection_count > 0` or `stall_escape_count > 0`), not only plain success.
- Keep staged no-progress recovery flow used for UX:
  - solver re-sync path first,
  - handle snap only as later-stage fallback.
- Keep torso unlock re-anchoring behavior to reduce stale torso-target locking.

### Debug instrumentation cleanup completed

- Removed session-specific C++ NDJSON logging helpers and probes from
  `cpp_core/src/kinematics_solver.cpp`.
- Removed session-specific Python hypothesis logging blocks from wheelbase files.
- Removed hardcoded debug session path/id usage in cleaned paths.
- Left restore guidance in this document (no dependency on external playbook file).

### Build/test workflow used

EmbodiK:

```bash
pixi run build
pixi run test -k stall -q
```

hmnd_robot install path (preferred):

```bash
pixi run python -m pip install --no-deps --force-reinstall "/home/andypark/Projects/repos/embodik"
pixi run python -c "import embodik,sys; print(sys.version); print(embodik.__file__)"
```

Low-level rebuild path used during incident work:

```bash
cmake --build build-cp311 --target _embodik_impl -j8
```

Wheelbase app validation entrypoint:

```bash
pixi run alpha_wheelbase_viser
```

### Validation status at archive time

- EmbodiK targeted stall suite passed after cleanup:
  - `pixi run test -k stall -q` -> all selected tests passed.
- Wheelbase Python files passed syntax validation after debug cleanup:
  - `python -m py_compile` over edited wheelbase modules succeeded.
- Runtime reproductions showed substantial reduction of severe deep-penetration
  events versus earlier failing revisions, but residual infeasible/zero-motion
  plateaus can still occur in specific constrained postures.

### Commits containing the finalized state

- EmbodiK commit: `b806036`
  - solver behavior + changelog + internal report updates.
- hmnd commit: `bec5caaa90`
  - wheelbase Python debug instrumentation cleanup.

### Remaining known issue to prioritize next

- Long plateau behavior (`INFEASIBLE` / `NUMERICAL_ERROR` with near-zero applied
  motion) under active constraints, often correlated with effective primary-task
  scale collapse.
- Recommended immediate follow-up:
  1. add deterministic plateau regression test(s),
  2. add bounded fallback for prolonged collapsed-scale streaks,
  3. re-validate with unlocked torso + mode transitions (`SCALE` <-> `MIN_ERROR`).

