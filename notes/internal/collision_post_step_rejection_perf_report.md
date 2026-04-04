# Collision Post-Step Rejection Performance Investigation

**Date:** 2026-04-02  
**Branch:** `fix/collision-post-step-rejection-perf`  
**Worktree:** `~/Projects/git-worktree/fix-collision-perf`

## Problem

Collision-aware IK (`solve_position_step`) regressed from **0.1ms (speed) / 0.4-0.5ms (balanced)** to **~12ms regardless of tuning mode**. The regression was first noticed in the v0.18.9 release.

## Root Cause Analysis

The regression has **three interacting causes**, all introduced between v0.18.0 (tuning mode introduction) and v0.18.9:

### 1. Post-step collision rejection (commit b806036)

`solve_position_step` gained post-step penetration rejection logic that calls `evaluate_min_collision_distance()` — a full brute-force scan of ALL collision pairs — after every integration step. This scan bypasses all tuning mode optimizations (caching, budget, proximity gating).

**Impact:** +7-8ms per step regardless of tuning mode.

### 2. Stall escape full-scan fallback

The stall escape code runs whenever `dq_norm < dq_stall_eps` (1e-5), which triggers frequently when the robot is converged or tracking slowly. The escape path calls `evaluate_collision_debug()` — another full brute-force scan — even when all pairs are well clear of the escape threshold.

**Impact:** +5-7ms per step when near-stalled (most steady-state steps).

### 3. Collision pair cache invalidation on stall

The stall path unconditionally sets `collision_pair_cache_has_full_scan_ = false`, forcing the next `compute_collision_constraint()` to do a full scan instead of using the cached subset. Combined with the underfill safety fallback (which forces a full scan when cached candidates < max_constraints), the cache was effectively disabled whenever the robot wasn't moving rapidly.

**Impact:** `compute_collision_constraint()` itself became ~8ms instead of ~0.7ms.

## Fixes Applied

### Fix 1: Three-tier post-step collision evaluation

Replaced `evaluate_min_collision_distance()` in all 8 post-step rejection call sites with `evaluate_post_step_collision_distance()`:

- **Tier 1 (early-exit):** If `last_constraint_min_distance_` (from `compute_collision_constraint()` that already ran this step) is well above the penetration threshold, skip the scan entirely. Cost: ~0.
- **Tier 2 (targeted scan):** When near collision, only check pairs from the active constraint + cached candidate set (typically 5-20 pairs vs 50-200 total).
- **Tier 3 (full scan fallback):** Only when no constraint data is available.

### Fix 2: Stall escape reuses constraint debug data

Replaced `evaluate_collision_debug()` full-scan calls in stall escape paths with:
- Reuse of `last_collision_debug_` from the constraint computation (for escape direction data)
- `evaluate_post_step_collision_distance()` for escape candidate validation

### Fix 3: Conditional cache invalidation on stall

Cache invalidation in the stall path now only triggers when near actual penetration (`last_constraint_min_distance_ < kCollisionPenetrationDistanceThreshold + margin`), not whenever `dq_norm` is low.

### Fix 4: Conditional underfill safety fallback

The underfill safety fallback (which forces a full scan when cached candidates < max_constraints) now only triggers near penetration, not unconditionally.

### Fix 5: Conditional stall escape activation

The stall escape block now checks whether we're actually near collision before running the expensive escape logic.

### Fix 6: Lazy constraint reuse

When the joint configuration change between steps is tiny (dq² < 1e-6) and the distance margin is safe (>5mm above penetration), reuse the previous constraint result without any geometry update or distance queries. Gated on cache-enabled modes (speed/balanced) — precise mode always recomputes. This respects the original Proxima-inspired tuning mode semantics documented in `collision_precise_restore_notes_2026-03-26.md`.

### Fix 7: Nearest-points coalescing (balanced mode)

When evaluating a small cached subset without time budget (balanced mode), enable nearest points on the initial distance query to eliminate the duplicate re-query from `ensure_nearest_points_for_pair()`. Skipped when budget is active (speed mode) to avoid exhausting the 300μs budget.

## Results

| Scenario | Before Fix | After Fix | Original Target |
|---|---|---|---|
| Speed mode | 12ms | **0.33ms** | 0.1ms |
| Balanced mode | 12ms | **0.34ms** | 0.4-0.5ms |
| Precise mode | 15ms | **9.1ms** | 7-8ms |

**35x improvement** for speed/balanced modes. Balanced mode now meets the original target. Speed mode is within 3x of the original 0.1ms target — the remaining gap is the time budget's 300μs allowance for distance queries that can't be lazily reused when configuration changes are larger.

## Safety Verification

All safety tests pass:
- No deep penetration detected in any mode when driving toward self-collision (200 steps)
- All three modes produce identical safety characteristics
- Collision rejection count and stall escape count remain functional

## Files Modified

- `cpp_core/include/embodik/kinematics_solver.hpp` — added member variables and method declarations
- `cpp_core/src/kinematics_solver.cpp` — all fixes
- `test/test_collision_tuning_perf.py` — new performance regression + safety tests
- `scripts/benchmark_collision_tuning_modes.py` — new benchmark script

## Regression Prevention

The new `test/test_collision_tuning_perf.py` test file includes:
- `TestCollisionTuningPerformance` — timing gates for speed (<3ms) and balanced (<4ms) modes
- `TestCollisionRejectionSafety` — penetration safety checks for all three modes
- Mode differentiation test ensuring speed/balanced are significantly faster than precise
