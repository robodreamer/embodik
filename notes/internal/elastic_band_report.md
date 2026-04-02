# Elastic Band Joint Limit Expansion — Experiment Report

**Date**: 2026-04-02
**Branch**: `elastic-band-joint-limits`
**Status**: Validated on Panda with narrowed limits. Ready for alpha wheelbase testing.

## Problem Statement

The alpha wheelbase robot has very limited joint ranges of motion. During teleop, the SNS solver frequently returns INFEASIBLE (task scale collapses to 0) because joint limit saturation removes too many DOFs from the null space. The existing `min_error` fallback changes the solver's objective entirely, producing jittery and unstable motions.

## Approach A: Velocity Headroom (Hypothesis Validation)

**Hypothesis**: If we keep saturated joints' DOFs active in the SNS solver with slightly wider limits, the *other* joints get better solutions because the null space isn't prematurely collapsed.

**Method**: Python-only test using `set_joint_limits(q_min - 0.001, q_max + 0.001)` before each solve, restoring after. Panda with +/- 0.15 rad narrowed limits.

**Results**:

| Axis | Baseline Stalls | Widened Stalls | Baseline Infeasible | Widened Infeasible |
|------|----------------|----------------|--------------------|--------------------|
| X    | 392            | 102            | 393                | 0                  |
| Y    | 270            | 104            | 168                | 0                  |
| Z    | 190            | 211            | 0                  | 0                  |

**Conclusion**: Just 0.001 rad (0.06 degrees) of extra room **completely eliminated infeasibility** on X and Y axes and reduced stalls by 60-74%. Hypothesis strongly validated.

## Approach B: Elastic Band with Margin Expansion

### Mathematical Formulation

The elastic band operates as a **constraint relaxation** mechanism:

1. **Expansion dynamics** per joint i, per solve step:
   - If stalled at joint limits for `stall_threshold` consecutive steps:
     `δ_i = min(δ_i + expand_rate, delta_max)`
   - If solver is healthy (success + non-zero task error resolved):
     `δ_i = δ_i × (1 - decay_rate)` (exponential snap-back)

2. **Effective limits** used for velocity box computation:
   - `q_min_eff = q_min - δ`
   - `q_max_eff = q_max + δ`

3. **Post-solve clamping**: `q = clip(q + dq·dt, q_min, q_max)` (nominal limits)

4. **Jacobian clamping bypass**: Joints with `δ_i > 5e-4` skip the Jacobian zeroing that normally occurs near limits.

This is equivalent to solving the relaxed problem with expanded feasible region, then projecting onto the nominal feasible set. The SNS solver produces a meaningful descent direction even when the original problem is infeasible, because the expanded box provides enough room to find a non-zero feasible scale.

### Why This Beats min_error

| Property | min_error | Elastic Band |
|----------|-----------|-------------|
| Solver mode | Changes objective (minimize violation) | Keeps scale mode |
| Velocity direction | Arbitrary (minimizes constraint violation) | Task-aligned |
| Stability | Can produce jittery motions | Smooth (exponential decay) |
| Safety | May violate limits | Always clamped to nominal |

### Implementation Details

**C++ changes** (3 files, ~200 lines):
- `kinematics_solver.hpp`: `ElasticBandConfig`/`ElasticBandState` structs, public API
- `kinematics_solver.cpp`:
  - `elastic_band_update()`: stall detection + expansion/decay logic
  - Velocity box margin widening: uses `q_min_eff`/`q_max_eff`
  - Jacobian clamping bypass for expanded joints
  - Hook into `solve_velocity()` after `stall_handler_update()`
- `kinematics_solver_bindings.cpp`: Python bindings

**Integration**:
- Complementary to collision stall handler (collision handler gates on `!limit_dominated_stall`, elastic band gates on `limit_dominated_stall`)
- Both can be enabled simultaneously

### Round-Trip Performance (Panda, +/- 0.15 rad narrowed limits)

| Metric | Baseline | Elastic Band | Improvement |
|--------|----------|-------------|-------------|
| X stalls | 392 | 113 | **71% reduction** |
| X infeasible | 393 | 27 | **93% reduction** |
| X return error | 0.6209 | 0.0000 | **100% improvement** |
| Y stalls | 270 | 125 | **54% reduction** |
| Y infeasible | 168 | 21 | **88% reduction** |
| Y return error | 0.0028 | 0.0000 | **100% improvement** |

### Safety Validation

- **Nominal limit compliance**: All configurations stay within `get_joint_limits()` across all tests
- **Very narrow limits** (+/- 0.05 rad): No NaN, no divergence, delta bounded
- **Extreme parameters** (delta_max=0.2): Bounded behavior, no crashes
- **Regression**: Disabled elastic band produces identical results to baseline

### Test Coverage

20 tests across 5 test classes:

1. **TestApproachAHypothesis** (4 tests): Hypothesis validation with set_joint_limits proxy
2. **TestElasticBandStateDynamics** (4 tests): State grows on stall, decays when healthy, respects max, per-joint gating
3. **TestElasticBandVelocityBox** (3 tests): Feasibility improvement, spring decay, Jacobian bypass
4. **TestElasticBandRoundTrip** (5 tests): Stall count reduction, convergence, oscillation comparison
5. **TestElasticBandSafety** (4 tests): Limit compliance, divergence, regression, coexistence with stall handler

### Default Parameters

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `delta_max` | 0.05 rad (~3°) | Small enough to be safe, large enough to unstick the solver |
| `expand_rate` | 0.01 rad/step | Gradual expansion, ~5 stall triggers to reach max |
| `decay_rate` | 0.2 | ~5 healthy steps to halve, ~15 to reach near-zero |
| `stall_threshold` | 3 | Quick response without being too sensitive |
| `expand_only_saturated` | true | Minimal perturbation to unsaturated joints |

### Smooth Saturation Analysis

The naive "expand + hard clip" concern: when the solver computes velocity into the expanded limit but we clip the position back to nominal, there's a one-step delay discontinuity.

In practice this is negligible because:
1. The expansion is small (max 0.05 rad), so the clipped velocity is tiny (~1e-5 rad/step)
2. The real benefit is for the *other* joints: keeping one DOF barely active prevents the SNS solver from collapsing the entire task scale
3. The exponential decay ensures the expansion is temporary and self-correcting
4. Round-trip tests show no oscillation increase compared to baseline

If the alpha wheelbase shows oscillation in practice, the plan includes **Approach B upgrade**: dual-state tracking (`q_internal` can exceed nominal, `q_command` is clamped) with a restoring spring in the elastic zone. This eliminates the clip discontinuity entirely.

## Next Steps

1. Install into hmnd_robot and test on alpha wheelbase teleop
2. Compare teleop responsiveness vs baseline and vs min_error
3. If oscillation appears, implement dual-state with restoring spring
4. Tune parameters for alpha wheelbase joint ranges
