# EmbodiK Robust Constraint Handling Context

This file records the domain language for the solver robustness migration. Keep
new solver, binding, test, and example changes aligned with these terms so the
C++ core owns constraint handling and Python examples stay thin.

## Canonical Terms

### Constraint policy

Solver-owned behavior that decides which configuration is accepted for a solve
step when constraints are active. A constraint policy may reject a candidate,
restore a previously safe configuration, select a fallback solve path, or emit
diagnostics. It must not depend on Python example code to clip, repair, or hold
the returned configuration.

### Boundary stall

A state where task error remains outside the deadband, solver motion is near
zero, and an enabled constraint boundary is near or violated. Collision,
center-of-mass, and joint-limit boundaries are all boundary-stall sources. The
solver should classify these states with diagnostics instead of requiring each
example to infer them from status strings and velocity norms.

### Last-safe configuration

The latest configuration observed by the solver to satisfy the enabled
constraints within the configured tolerance. It is a solver-side recovery
anchor, not an example-side cache. Restoring it is valid only when the current
candidate would violate solver-owned constraints.

### Constrained weighted fallback

A single-priority weighted objective solve that still uses the solver's hard
constraint machinery. It is not an unconstrained weighted least-squares step
followed by clipping. The fallback may trade task residuals through weights, but
joint limits, collision limits, CoM support, and other hard bounds must remain
hard constraints.

### PoseTaskGroup layout

The row ownership and state layout for a grouped pose task. Split, merged, and
auto-switching variants should expose the same solver-facing ownership model so
examples configure intent instead of manually reshaping task rows.

### Auto-tuner

A deterministic, bounded update law for solver policy parameters. An auto-tuner
must expose reset and diagnostic state, and it must have regression tests for
its update bounds. Trial-and-error adjustments without a stable rule are not
auto-tuners.

### Target recapture

The UI-level act of moving interactive targets back to the current robot pose.
It may be triggered by solver diagnostics, but it does not decide whether a
configuration is safe. Constraint safety remains solver-owned.

### Solver-owned configuration

The `q_solution` returned by the solver after integration, normalization, and
constraint policy have been applied. Python examples should treat it as the
accepted configuration and must not post-clip it for constraint safety.
