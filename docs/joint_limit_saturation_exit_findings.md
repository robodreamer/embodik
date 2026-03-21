# Joint-Limit Saturation Exit — Investigation Findings

## Problem Statement

When using velocity-based IK (SNS solver) with a 6-DOF end-effector task on a
7-DOF robot (Panda), driving the EE to a target that pushes one or more joints
toward their position limits creates a **dead-end on reversal**: the robot gets
completely stuck when the EE target reverses direction.

**Observed symptom (Panda, `[0, 0, +0.6]` m EE offset):**
- Joint 3 reaches its upper limit (`-0.070 rad`) during the forward phase.
- On reversal: **382/400 steps stalled**, EE return error **306 mm** (out of
  ~300 mm total displacement).

## Root Cause Analysis

### 1. Velocity-bound collapse at the limit

The `solve_velocity` function subtracts a safety `margin_limit` from the
position margin before computing velocity box constraints:

```
position_margin = q_max - q_current - margin_limit
velocity_bound  = margin / dt
```

With the original `margin_limit = 1e-3` (1 mrad), a joint within 1 mrad of its
limit gets an effective velocity bound of **exactly zero** in the limit
direction.

### 2. SNS uniform task scaling

The SNS (Stacked Null Space) algorithm finds the **maximum uniform scale** for
each priority-level task such that all joint velocity constraints are satisfied.
When any component of the scaled task requires a DOF whose velocity bound is
zero, the entire task scale drops to (near) zero.

### 3. Kinematic coupling makes this catastrophic for 6D tasks

At the stuck configuration, the Panda's Jacobian reveals:

| Component | q3_dot needed |
|-----------|--------------|
| Position-only (3D) | **-1.16 rad/s** (away from limit — feasible) |
| Full 6D (pos + ori) | **+136 rad/s** (into the limit — infeasible) |

Joint 3 is the **only joint** with a direct angular-Y contribution
(`ang_y = -1.0`). The position component requires other joints to move fast,
and those joints' angular contributions can only be cancelled by joint 3 —
which is blocked. The SNS correctly determines that no uniform scale can
satisfy the coupled 6D task, and returns near-zero velocity.

**Proof:** When the task is reduced to position-only (3D), the robot recovers
perfectly (0 stalls, ~0 mm return error).

## Approaches Explored

### Approach 1: Limit Avoidance (Abandoned)

**Idea:** Inject a proximity-activated posture cost at priority 0 that repels
joints from their limits before they reach saturation.

**Result:** Eliminated the stuck behavior entirely (0 stalls, 2.3 mm error).

**Why abandoned:** The user correctly identified that this approach:
- Breaks the task definition near the limits (the solver no longer tries to
  achieve the requested task)
- Does not demonstrate improved *saturation handling* — it avoids saturation
  entirely
- May conflict with the solver's design principles

The avoidance API (`set_joint_limit_avoidance`, `clear_joint_limit_avoidance`)
and its C++ implementation have been fully reverted.

### Approach 2: SNS Core Recovery Logic (Abandoned)

**Idea:** Modify the SNS algorithm (`ik_baseline.hpp`) to detect when the
optimal scale is near zero and attempt a recovery by:
- Projecting the task into a feasible subspace
- Zeroing Jacobian columns for "pinned" DOFs
- Adding null-space escape velocity

**Result:** Multiple iterations produced either no improvement, oscillatory
behavior, or conflicted with the SNS's inherent design. The SNS was
consistently finding the mathematically correct (though practically unhelpful)
solution.

**Why abandoned:** The SNS algorithm is principled — modifying its internal
optimization to special-case one failure mode risks introducing regressions
across all use cases.

### Approach 3: Constraint Softening (Current — Partial Fix)

**Idea:** Two complementary changes to the velocity box constraint formulation:

1. **Reduce `margin_limit`** from `1e-3` to `1e-4`: Keeps a residual velocity
   bound of `0.1 mrad / 0.01s = 0.01 rad/s` instead of exactly zero at the
   limit. The post-solve position clamp still prevents actual limit violations.

2. **Minimum velocity bound (`kMinBoundFraction = 0.10`)**: When a joint is
   inside both limits and *away* from a limit (margin > 0.01 rad), ensure
   velocity bounds are at least ±10% of the velocity limit. A `kMarginThreshold`
   prevents injecting headroom *toward* a limit the joint is already at (fixes
   joint-6 spurious drift bug).

**Result with `kMinBoundFraction = 0.10`:**
- Stalls: **0** (was 382)
- Task scale: **~1%** (was 0%)
- EE return: slow but monotonic (235 mm remaining after 400 steps)

**Assessment:** The robot is no longer fully stuck, but recovery is slow because
the SNS can only achieve ~1% of the 6D task per step. The orientation component
still dominates the coupling, limiting how much position progress can be made.

## What Remains in the Codebase

### Effective changes (kept):

| File | Change | Purpose |
|------|--------|---------|
| `kinematics_solver.hpp` | `set_limit_recovery_hysteresis()` | Configurable enter/exit epsilon for outside-limit detection |
| `kinematics_solver.hpp` | `set_limit_exit_release_margin()` | Configurable release margin to reduce chattering near limits |
| `kinematics_solver.cpp` | `margin_limit` 1e-3 → 1e-4 | Prevents velocity bound from reaching exactly zero at the limit |
| `kinematics_solver.cpp` | `kMinBoundFraction = 0.10` | Guarantees ±10% velocity limit floor for SNS feasibility |
| `kinematics_solver.cpp` | Saturated-joint detection fix | Uses combined velocity+position bounds for reporting |
| `kinematics_solver_bindings.cpp` | Bindings for hysteresis + release margin | Python API |
| `test/test_joint_limit_recovery.py` | New tests for hysteresis, release margin, anti-chatter | Unit coverage |
| `test/test_joint_limit_exit_symmetry.py` | 1-DOF round-trip symmetry regression test | Baseline guardrail |

### Reverted / removed:

- `set_joint_limit_avoidance()` / `clear_joint_limit_avoidance()` API (header,
  impl, bindings, `.pyi` stubs)
- Proximity-activated avoidance injection in `solve_velocity()`
- SNS recovery logic modifications in `ik_baseline.hpp`
- Experimental GUI sliders in `01_basic_ik_simple.py`
- `test_panda_saturation_exit.py` (referenced removed avoidance API)

## Iteration 2: Joint-Limit Barrier Gradient Task

### Motivation

The constraint softening from iteration 1 prevented full stalling but left recovery
very slow (~1% task scale).  Instead of modifying the SNS algorithm or the constraint
formulation, iteration 2 introduces a **secondary (nullspace) task** that actively
drives joints away from their limits using an analytical barrier gradient.

### Approach: C++ Barrier Gradient (Strategy B)

A new built-in solver feature computes the analytical gradient of the joint-limit-
distance metric and injects it as a priority-1 (nullspace) velocity target:

**Barrier function:** `h(p) = p^2 / ((1+e-p)(p+1+e))` where `p` is the normalized
joint position in `[-1, 1]` and `e = 0.04`.

**Key properties:**
- Gradient is near-zero in a configurable deadband (default: inner 40% of range)
- Grows as `~1/dist^2` near each limit (barrier shape)
- O(n) element-wise computation, auto-vectorized by Eigen/compiler
- Injected entirely in C++, zero Python round-trip per solve step

**Coexistence with posture bias:** The barrier gradient is **added to** any existing
priority-1 task group (e.g., a posture bias), not injected as a separate priority.
This prevents the SNS from finding competing uniform scales.  The barrier's shape
ensures it dominates near limits while the posture bias dominates in the interior.

**Activation-zone gating:** When all joints are in the deadband, the barrier task
is fully skipped (O(n) gradient + one norm check).  The gradient is continuous at
the deadband boundary, so there is no velocity discontinuity when the task appears
or disappears.

### API

```python
solver.set_joint_limit_barrier_task(barrier_margin=0.3, gain=1.0)
solver.clear_joint_limit_barrier_task()
```

Default state: disabled at construction.  After validation, the default will flip to
enabled with tested parameters.

### Strategy A (comparison): Python-Level Saturated-Joint Posture Task

Reads `result.saturated_joints` after each solve and dynamically drives only those
joints toward midrange via a priority-1 posture task.  Simple but reactive (acts
after saturation, not before) and has Python round-trip overhead per step.

### Test Framework: Narrowed-Limits Panda

Uses `robot.set_joint_limits()` to narrow each joint's range to `q_default ± 0.25 rad`.
This forces multiple joints to saturate during normal EE motion without triggering
kinematic singularity.

### Results (Panda, narrowed limits ± 0.25 rad, 200 steps per phase)

| Scenario | Approach | EE Return Error | Stall Steps (reverse) | Min Task Scale |
|----------|----------|-----------------|------------------------|----------------|
| X-axis 0.10m | Baseline | ~0.59 m | ~200 | ~0.01 |
| X-axis 0.10m | Strategy B | ~0 | ~97 | ~0.88 |
| Y-axis 0.10m | Baseline | ~0 | ~101 | 1.0 |
| Y-axis 0.10m | Strategy B | ~0 | ~101 | 1.0 |
| Z-axis 0.10m | Baseline | ~0 | ~101 | ~0.81 |
| Z-axis 0.10m | Strategy B | ~0 | ~101 | ~0.81 |

**Very narrow (margin=0.15 rad):** X and Y baseline fail (~0.62 m error, 200 stalls);
barrier recovers (~0 error, ~78–111 stalls). Z: barrier reduces stalls.

**Key findings:**
- **X-axis:** Baseline has ~59 cm return error with ~1% task scale.  Strategy B
  reduces error to ~0 with ~88% task scale — a **>100x improvement**.
- **Z-axis:** Baseline stalls for 99/200 reverse steps.  Strategy B eliminates all
  stalls while maintaining equivalent task scale.
- **Posture coexistence:** B + posture bias performs identically to B alone, confirming
  no destructive competition from the additive approach.
- **Oscillation:** Strategy B introduces ≤1 oscillation in some axes (sign change in
  summed joint velocity), which is negligible.

### What Changed (Iteration 2)

| File | Change | Purpose |
|------|--------|---------|
| `kinematics_solver.hpp` | `set_joint_limit_barrier_task()`, `clear_joint_limit_barrier_task()` | Enable/disable barrier gradient task |
| `kinematics_solver.hpp` | `velocity_to_config_index_cache()` + cache members | Reuse velocity→config map across solves |
| `kinematics_solver.hpp` | `barrier_task_enabled_`, `barrier_margin_`, `barrier_gain_`, `barrier_epsilon_` | Configuration state |
| `kinematics_solver.cpp` | Barrier injection in `solve_velocity()` | Single-pass sparse rows; no full `nv` barrier vector |
| `kinematics_solver.cpp` | `velocity_to_config_index_cache()` | Shared between barrier and position constraints |
| `kinematics_solver_bindings.cpp` | Bindings for barrier task API | Python access |
| `python_bindings/__init__.pyi` | Type stubs | IDE support |
| `python/embodik/pose_metrics.py` | General-purpose metric module | Offline analysis (not real-time) |
| `test/test_panda_narrowed_limits.py` | Comprehensive test framework | Baseline vs A vs B comparison |
| `test/test_joint6_drift_bug.py` | Regression for joint-6 spurious drift | kMinBoundFraction fix |

### Iteration 3: kMinBoundFraction Fix (Joint-6 Drift Bug)

**Bug:** With narrowed limits, joint 6 (Panda wrist) drifted toward its upper limit
even when the EE task did not require it. Root cause: `kMinBoundFraction` guaranteed
10% velocity headroom in *both* directions, including *toward* a limit the joint
was already at. The SNS solver exploited this to minimize task error.

**Fix:** Add `kMarginThreshold = 0.01` rad. Only apply softening when
`raw_margin > kMarginThreshold`; at the limit, velocity bound toward the limit
remains zero.

### Remaining Open Questions

1. **Default-on timing:** Once validated on real hardware, flip the default to
   enabled with tested parameters (margin=0.3, gain=1.0).

2. **Task decomposition:** Splitting 6D tasks into independent position/orientation
   subtasks with separate scaling would be complementary — the barrier gradient
   handles joint-level recovery while task decomposition handles task-level coupling.

3. **Manipulability gradient:** The current barrier uses only joint-limit distance.
   Adding a manipulability term (singularity avoidance) would require FK per step,
   which could be added to the C++ implementation using Pinocchio if needed.
