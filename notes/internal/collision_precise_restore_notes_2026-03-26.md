# Collision Tuning: PRECISE Legacy Restore Rationale (2026-03-26)

## Context

- This note captures why `CollisionTuningMode::kPrecise` was moved back to a
  conservative legacy-equivalent path during the proximity-gating rollout.
- The user concern was potential runtime regression versus older behavior,
  especially for teleop-relevant collision workloads.
- Scope:
  - `cpp_core/src/kinematics_solver.cpp`
  - `cpp_core/include/embodik/kinematics_solver.hpp`
  - `python_bindings/src/kinematics_solver_bindings.cpp`
  - `test/test_embodik.py`

## Reproduction / Benchmark Inputs

- Benchmark script:
  - `pixi run python scripts/benchmark_teleop_workloads.py --warmup 30 --steps 120 ...`
- Key pre-change artifacts used for decision:
  - `.cursor/reports/perf_runs/collision_mode_legacy_equiv.json`
  - `.cursor/reports/perf_runs/collision_mode_precise_like.json`
  - `.cursor/reports/perf_runs/collision_mode_balanced_like.json`
  - `.cursor/reports/perf_runs/collision_mode_speed_like.json`
- Post-change mode validation artifact:
  - `.cursor/reports/perf_runs/collision_tuning_modes_post_change.json`

## Findings

- Legacy-equivalent and precise-like settings were mixed by workload; precise-like
  did not show robust, consistent wins over legacy-equivalent.
- For conservative "accuracy-first" mode semantics, we preferred the path with
  lower regression risk and established behavior.
- Proximity-gated row activation provided clear benefits for far-from-collision
  scenarios in `BALANCED`/`SPEED`, while `PRECISE` remained intentionally
  conservative (gating off).

## Final Decision

- Keep:
  - Proximity-gated activation infrastructure (optional, threshold-based).
  - Explicit enable/disable API for gating:
    - `set_proximity_gated_collision_activation_enabled(...)`
    - `get_proximity_gated_collision_activation_enabled()`
  - Auto-updated activation margin from `min_distance` via multiplier.
- Restore in `kPrecise`:
  - Disabled candidate cache (legacy-equivalent intent).
  - Disabled refinement budget (`budget_us = 0`).
  - Disabled proximity-gated row filtering.
- Rationale:
  - `PRECISE` should prioritize conservative fidelity and predictability over
    aggressive latency optimization.
  - Keep optimization knobs available in `BALANCED`/`SPEED` and via explicit API.

## How To Revisit / Restore More Aggressive PRECISE Later

If we want to re-enable optimized behavior in `PRECISE`, change only one axis at
a time and re-run parity + perf gates.

1. Candidate cache in PRECISE:
   - In `KinematicsSolver::set_collision_tuning_mode(...)`, update
     `enable_collision_pair_cache(...)` under `kPrecise`.
2. Refinement budget in PRECISE:
   - Under `kPrecise`, change `set_collision_refinement_time_budget_us(0)` to a
     bounded positive budget.
3. Proximity gating in PRECISE:
   - Under `kPrecise`, toggle
     `set_proximity_gated_collision_activation_enabled(true)` and choose a
     conservative multiplier.

Required validation gates before keeping any of the above:

- Regression tests:
  - `pixi run pytest test/test_embodik.py -k "collision_constraint or activation"`
  - `pixi run pytest test/test_embodik.py test/test_position_step_joint_options.py`
- Collision equivalence:
  - `pixi run python scripts/validate_collision_cache_equivalence.py`
- Benchmarks:
  - Re-run `scripts/benchmark_teleop_workloads.py` for legacy-equivalent vs new
    precise mapping with p50/p95 comparisons on:
    - `panda_collision`
    - `dual_arm_collision_mesh`
    - `dual_arm_collision_simplified(_masked)`

## Guardrails

- Preserve API stability:
  - Keep explicit gating toggle independent from multiplier.
  - Keep multiplier semantics deterministic (`margin = multiplier * min_distance`).
- Never couple enable/disable toggles to parameter mutation.
- Keep `PRECISE` change-risk low unless benchmark parity is clearly favorable.

## Relevant Commits (for traceability)

- Feature branch commits:
  - `33c9789` optional proximity-gated activation
  - `70d9af0` Python multiplier API
  - `a566cc9` activation regression tests
  - `d78528c` explicit gating enable toggle
- Squashed to main:
  - `266d96e` `feat: add optional proximity-gated collision activation controls`
