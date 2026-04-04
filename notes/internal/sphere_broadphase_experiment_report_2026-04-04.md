# Sphere Broadphase for Collision Distance Queries

**Date:** 2026-04-04
**Branches:** `experiment/vamp-broadphase` + `fix/collision-post-step-rejection-perf` (both merged to main)
**Status:** Merged and validated

## Summary

Added a native AABB-derived sphere broadphase to `compute_collision_constraint()` that skips expensive HPP-FCL `computeDistance()` (GJK/EPA) calls for collision pairs whose bounding spheres are far apart. Zero new dependencies — uses only existing Pinocchio, Eigen, and HPP-FCL.

Combined with the lazy constraint reuse from `fix/collision-post-step-rejection-perf`, the total collision pipeline achieves:

| Configuration | Mean collision time | Speedup vs PRECISE |
|---------------|--------------------|--------------------|
| **PRECISE** (full scan, no optimizations) | 4.555 ms | 1x |
| **SPEED** (cache + lazy reuse, no broadphase) | 0.015 ms | 314x |
| **SPEED + sphere broadphase** | 0.001 ms | **5790x** |

The sphere broadphase adds **18.5x** on top of SPEED mode's lazy reuse, and **30x** when measured in isolation (without lazy reuse).

## Background

EmbodIK's collision constraint loop calls `pinocchio::computeDistance()` per collision pair. For Panda with 60 active pairs (after auto-exclusions), PRECISE mode costs ~4.6ms per step. Two complementary optimizations were developed:

1. **Lazy constraint reuse** (fix branch): Skips the entire collision recomputation when `dq` is tiny and previous distance is safe. Eliminates 99%+ of collision steps in smooth trajectories.

2. **Sphere broadphase** (this experiment): When collision IS recomputed, AABB-derived bounding spheres skip 59/60 pairs via cheap sphere-sphere distance, reducing the expensive GJK/EPA calls to ~2/step.

**Insight from VAMP experiment:** The KavrakiLab VAMP library uses sphere approximations to evaluate collision in microseconds. While VAMP lacks distance computation (boolean only), the sphere-distance idea translates natively: if bounding spheres are far apart, the true mesh distance is guaranteed to be even larger.

## How It Works

1. **At model load:** For each collision geometry, compute a bounding sphere from its AABB. Center = `placement * AABB_center` (in parent frame), radius = AABB half-diagonal. Provably conservative: sphere contains AABB contains mesh.

2. **Per solve step:** Before calling `computeDistance()` for a pair, compute `sphere_dist = ||center_a - center_b|| - r_a - r_b` using already-computed frame transforms (`oMf`). If `sphere_dist > min_distance + cache_margin + safety_margin`, skip GJK/EPA.

3. **Safety guarantee:** Since `sphere_dist <= true_dist` always holds, the broadphase can only skip pairs that are truly far apart. A 1cm safety margin provides additional buffer.

## Performance Results

### Panda Robot (60 active pairs, 300-step trajectory, example 02 configuration)

**Sphere broadphase in isolation** (pair cache disabled):

| Metric | Without Broadphase | With Broadphase |
|--------|-------------------|-----------------|
| **Median collision time** | 4.621 ms | 0.153 ms |
| **Speedup** | 1x | **30x** |
| Exact distance queries | 61/step | 2/step |
| Sphere-culled pairs | 0/step | 59/step |

**Combined with all optimizations** (SPEED mode):

| Metric | SPEED (no broadphase) | SPEED + broadphase |
|--------|----------------------|-------------------|
| Mean collision time | 0.015 ms | 0.001 ms |
| Non-zero collision steps | 1/300 | 1/300 |
| Broadphase added speedup | - | **18.5x** over SPEED alone |

### Why median is 0.000 ms in SPEED mode

The lazy constraint reuse short-circuits 299/300 steps entirely (collision time = 0). The sphere broadphase accelerates the 1 remaining step where collision IS recomputed (full refresh). This is why the improvement shows in **mean** but not **median**.

## Safety Validation

| Test | Result |
|------|--------|
| Solver status matches baseline (50 random configs) | PASS |
| Joint velocities identical to baseline (100 configs, norm < 1e-6) | PASS |
| No penetration over 200-step trajectory (3 tuning modes) | PASS |
| Sphere culling ratio > 30% | PASS (98% in isolation) |
| All modes produce similar safety margins | PASS |
| Full test suite (13 sphere + collision tuning tests) | PASS |

## API

```python
solver.enable_sphere_broadphase(True)   # enable (auto-builds from AABBs)
solver.sphere_broadphase_enabled()      # query state

# Instrumentation on VelocitySolverResult:
result.collision_sphere_culled_pairs    # pairs skipped by sphere check
```

Enabled by default in `kSpeed` and `kBalanced` tuning modes. Disabled in `kPrecise`.

## Files Changed (sphere broadphase)

| File | Change |
|------|--------|
| `cpp_core/include/embodik/sphere_broadphase.hpp` | New: SphereBroadphase class |
| `cpp_core/src/sphere_broadphase.cpp` | New: AABB sphere fitting + pair distance |
| `cpp_core/include/embodik/kinematics_solver.hpp` | Added member + public API |
| `cpp_core/include/embodik/types.hpp` | Added instrumentation field |
| `cpp_core/src/kinematics_solver.cpp` | Integration into collision loop + tuning modes |
| `python_bindings/src/kinematics_solver_bindings.cpp` | Python API exposure |
| `python_bindings/src/bindings.cpp` | Result field binding |
| `CMakeLists.txt` | Added source file |
| `test/test_sphere_broadphase.py` | 5 correctness + performance tests |
| `scripts/validate_collision_cache_equivalence.py` | Sphere culling metrics |

## Potential Future Work

- **Alpha wheelbase benchmarks:** More collision pairs = likely even larger broadphase speedup
- **Tighter spheres:** OBB-derived spheres instead of AABB for elongated geometries
- **SIMD batch evaluation:** Vectorized sphere-sphere distance pass over all pairs at once
