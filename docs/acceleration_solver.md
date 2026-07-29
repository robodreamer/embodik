# Acceleration Solver

`AccelerationSolver` is EmbodiK's native fixed-base acceleration-level eSNS
surface. It solves for physical joint acceleration `ddq` from explicit
configuration `q`, velocity `dq`, and timestep `dt`:

```text
J(q) * ddq + Jdot(q, dq) * dq = xdd_ref
lower <= C(q, dq) * ddq + bias(q, dq) <= upper
```

The solver is intended for controller prototyping and fixed-base constrained
motion experiments. It is not a hard real-time controller and does not manage
command history, integrate a long-horizon trajectory, or replace a downstream
jerk-limited actuator interface.

## Supported Scope

`KinematicsSolver` remains EmbodiK's default velocity-level backend. The
acceleration solver is a separate, opt-in API: constructing or running it does
not replace, reconfigure, or import tasks from an existing velocity solver.

| Area | Current acceleration support |
| --- | --- |
| Model topology | Fixed-base scalar 1-DoF joints |
| State limits | Compatible joint position, velocity, and acceleration box |
| Tasks | Frame, CoM, posture, and joint tasks with explicit velocity and acceleration references |
| Hard rows | Task acceleration bounds, affine acceleration rows, frozen next-velocity rows, and scalar joint locks |
| Dynamics | Fixed-base inverse-dynamics effort limits |
| Allocation | Diagonal generalized acceleration metric and reference |
| Contact | Fixed-base point or rigid contact kinematic acceleration equalities |
| Geometry | Tight point, tight frame pose, relative pose, torso pose bounds, and CoM support polygons |
| Centroidal stability | Momentum-rate objective and bounds, predicted capture point, and physical centroidal-rate ZMP |
| Collision compatibility | Canonical velocity collision rows lifted to next velocity, followed by conservative-or-exact sampled validation |
| Native collision certification | Exact-name sphere-sphere catalogs on fixed-base all-prismatic scalar models |

Unsupported scope fails explicitly. This includes floating bases, dynamic
contact forces, friction cones, wheel rolling/steering constraints,
`SCALE_ELASTIC`, general native collision geometry, and ROS 2 controller or
wheelbase integration.

## Inequality Constraint Parity

The acceleration surface represents velocity inequalities on
`dq_next = dq + dt * ddq` and acceleration inequalities directly. It does not
silently import mutable `KinematicsSolver` configuration, so adapter-owned
rows must be supplied explicitly.

| Current velocity constraint family | Acceleration behavior |
| --- | --- |
| Joint position, velocity, and acceleration limits | Native combined acceleration state box, with accepted-state recheck |
| General linear velocity rows | Supported through `FrozenNextVelocityConstraint` |
| Task-space finite-step bounds | Supported through `TaskAccelerationBounds`; position-step priority-policy rows are not imported automatically |
| Self-collision include/exclude filters and per-pair floors | Supported through `solve_with_velocity_collision()` using the configured velocity solver |
| CoM support polygon | Native `ComSupportPolygonAccelerationConstraint` for fixed-base/root-fixed support |
| Centroidal momentum-rate objective and bounds | Native physical `Ag * ddq + dAg * dq` rows for fixed-base models |
| Predicted capture point | Native `CapturePointAccelerationConstraint` using the solver's constant-acceleration integration rule |
| Physical centroidal-rate ZMP | Native `ZmpAccelerationConstraint` with a positive vertical-force gate and accepted-state recheck |
| Tight point and frame pose bounds | Native acceleration constraints |
| Relative pose bounds | Native acceleration constraint |
| Torso pose bounds | Native acceleration constraint with an explicit caller-owned reference pose |
| Joint locks | Zero acceleration, zero next velocity, and fixed current-position lock rows |
| Fixed-base effort limits | Native inverse-dynamics inequalities |
| Floating-base position/orientation bounds | Rejected because the fixed-base solver constructor rejects floating-base topology |
| Native collision combined with other hard-row families | Rejected until the combined predicted-state certificate is implemented |

`SCALE_ELASTIC`, seed corridors, adaptive timesteps, preferred-lock retries,
stall recovery, task-layout selection, and weighted position-step fallback are
stateful `KinematicsSolver` controller policies rather than missing
acceleration inequalities. A controller adapter must implement or reject those
policies explicitly. Manipulability and joint-limit avoidance are velocity
objective families, not hard inequality rows, and are not currently exposed as
acceleration tasks.

## Compatible Joint Box

Every solve starts from one combined state box. Position, velocity, and
acceleration limits are expressed as compatible bounds on the same `ddq`
decision, so an accepted command cannot satisfy one derivative-order limit by
contradicting another. Position and velocity limits are enabled by default;
callers may supply an acceleration-limit override for models without usable
acceleration metadata.

```python
options = embodik.AccelerationSolveOptions()
options.acceleration_limits_override = acceleration_limits
result = solver.solve(q, dq, dt, options)
```

When the combined box requires braking, zero acceleration is outside the hard
set. The default `allow_state_box_task_fallback = True` makes the resulting
MIN_ERROR task fallback explicit while preserving the mandatory braking
command. Set it to `False` when strict task semantics should fail closed
instead. `state_box_task_fallback_applied` reports which policy was used.

Only `SolverStatus.SUCCESS` carries executable acceleration, next-velocity, and
configuration outputs. Failures clear those outputs.

For latency-sensitive controller loops, set
`options.collect_task_diagnostics = False`. This skips rich physical
per-task records while preserving task scales, task errors, solver status, and
all executable outputs. `computation_time_ms` covers the complete C++ solve;
`preprocessing_time_ms`, `backend_computation_time_ms`, and
`postprocessing_time_ms` expose its measured phases.
For `solve_with_velocity_collision()`, these phase fields describe the inner
acceleration solve; `computation_time_ms` also includes collision-row lifting
and sampled endpoint validation.

## Task References

Acceleration tasks use explicit measured-state feedback and feedforward:

```python
reference = embodik.AccelerationTaskReference()
reference.desired_velocity = desired_task_velocity
reference.desired_acceleration = desired_task_acceleration
reference.proportional_gain = kp
reference.derivative_gain = kd
solver.set_task_reference("ee_position", reference)
```

The solver is stateless across calls. The caller owns the applied `dq` used on
the next tick and must reset it when switching controllers, resetting the
robot, or rejecting a result.

## Centroidal Stability

Centroidal momentum-rate rows use six-axis order
`[linear x, y, z; angular x, y, z]`. The objective and bounds operate on the
physical quantity:

```text
hdot = Ag(q) * ddq + dAg(q, dq) * dq
```

Predicted capture point uses CoM position, velocity, physical acceleration, a
per-solve frozen `omega`, and the same constant-acceleration integration rule
used for `q_solution` and `joint_velocities_next`.

Physical centroidal-rate ZMP uses net force after gravity, enforces
`Fz >= fz_min > 0`, and retains CoM-height and horizontal-force terms in every
cross-multiplied polygon row. Successful results are postvalidated at the
nonlinear predicted state.

These capabilities are fixed-base solver constraints.
`AccelerationSolver.capabilities().supports_dynamic_balance` remains false:
the rows do not allocate contact wrenches, enforce friction cones, or prove
floating-base contact-force feasibility. See
[Centroidal Stability](centroidal_stability.md) for units, frame conventions,
velocity-level semantics, formulas, and diagnostics.

## Collision Modes

### Velocity-row compatibility

`solve_with_velocity_collision()` consumes an already configured
`KinematicsSolver`. It reuses that solver's collision pair filtering,
include/exclude policy, active per-pair floors, row ordering, and row budget.
The lifted rows constrain `dq_next = dq + dt * ddq`. Every allowed pair at
every requested constant-acceleration sample is accounted for in canonical
order. A conservative enclosing-sphere lower bound may prove a pair safe;
ambiguous geometry falls back to the exact collision backend.

This path is fail-closed but sampled. A successful result reports
`collision_endpoint_validated = True` and
`collision_step_certified = False`; increasing validation substeps does not
turn samples into a continuous swept-path proof.

### Native analytic certification

`configure_collision_constraint()` enables continuous constant-acceleration
certification only for complete exact-name sphere-sphere catalogs on fixed-base
all-prismatic scalar models. It proves the current hard row, common reachable
braking authority, the integrated path, and predicted-state viability.

While native certification is configured, only joint state boxes, generalized
allocation, and soft tasks may be combined with it. Other hard families are
rejected until they provide the same predicted-state certificate contract.

## Build And ABI Notes

The acceleration API was introduced in `0.21.0` on top of the `0.20.x`
velocity-solver integration baseline. The shared `Task` virtual interface is
unchanged. The `0.22.0` centroidal additions extend public solver option and
result layouts, so native consumers must rebuild `embodik_core` and the
Nanobind extension together. Pre-`1.0` native binaries are not compatible
across minor releases.

## API Reference

### Solver

::: embodik.AccelerationSolver

::: embodik.AccelerationSolveOptions

::: embodik.AccelerationSolverResult

::: embodik.AccelerationSolverCapabilities

### Tasks And Diagnostics

::: embodik.AccelerationTaskReference

::: embodik.AccelerationTaskDiagnostics

::: embodik.AccelerationAllocationDiagnostics

::: embodik.CentroidalMomentumRateObjective

::: embodik.CentroidalMomentumRateBounds

::: embodik.CentroidalMomentumRateDiagnostics

::: embodik.CapturePointAccelerationDiagnostics

::: embodik.ZmpAccelerationDiagnostics

::: embodik.AccelerationAnalyticCollisionPairDiagnostics

### State, Effort, And Hard Rows

::: embodik.AffineAccelerationConstraint

::: embodik.FrozenNextVelocityConstraint

::: embodik.TaskAccelerationBounds

::: embodik.GeneralizedAccelerationAllocation

::: embodik.EffortConstraintOptions

::: embodik.ContactAccelerationConstraint

### Geometric Constraints

::: embodik.GeometricConstraintAccelerationPolicy

::: embodik.ComSupportPolygonAccelerationPolicy

::: embodik.ComSupportPolygonAccelerationConstraint

::: embodik.CapturePointAccelerationConstraint

::: embodik.ZmpAccelerationConstraint

::: embodik.TightPointAccelerationConstraint

::: embodik.TightFramePoseAccelerationConstraint

::: embodik.RelativePoseAccelerationConstraint

::: embodik.TorsoPoseBoundAccelerationConstraint

### Collision Compatibility And Certification

::: embodik.VelocityCollisionLiftOptions

::: embodik.CollisionConstraintAccelerationPolicy

::: embodik.AccelerationCollisionRegime

::: embodik.CollisionGeometryPair

::: embodik.CollisionPairMinimumDistance

::: embodik.CollisionConstraintDefinition

::: embodik.ComSupportPolygonConstraintDefinition

::: embodik.RelativePoseConstraintDefinition

::: embodik.TightPointConstraintDefinition

::: embodik.TightFramePoseConstraintDefinition

::: embodik.TorsoPoseBoundDefinition

::: embodik.ContactType

::: embodik.ComSupportPolygonOutsidePolicy

::: embodik.CollisionConstraintOutsidePolicy
