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
contact forces, friction cones, wheel rolling/steering constraints, general
native collision geometry, and ROS 2 controller or wheelbase integration.

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

Seed corridors, adaptive timesteps, preferred-lock retries,
stall recovery, task-layout selection, and weighted position-step fallback are
stateful `KinematicsSolver` controller policies rather than missing
acceleration inequalities. A controller adapter must implement or reject those
policies explicitly. Manipulability and joint-limit avoidance are velocity
objective families, not hard inequality rows, and are not currently exposed as
acceleration tasks.

## Scaling and nominal-position elasticity

`SCALE` preserves the acceleration objective direction through the native SNS
backend. `MIN_ERROR` uses its native constrained error objective.
`SCALE_ELASTIC` uses SNS `SCALE` plus a persistent nominal-position band. It
does not rename a `MIN_ERROR` solve. Task diagnostics report the effective
backend mode and explicit fallback use.

When an authorized fallback has already selected `MIN_ERROR`, a numerical
hard-bound rejection can retry that effective policy from the original
problem. This does not authorize fallback for other tasks or soften hard rows;
task diagnostics retain the fallback flag.
An almost-zero Cartesian task error does not suppress this retry when the
candidate violates hard bounds; tracking accuracy alone is not feasibility.

At nonzero joint velocity, the affine term `Jdot * dq` can make zero
acceleration incompatible with the task ray even when a positive scale is
feasible. When directional recovery needs a new seed, the CPU backend searches
joint acceleration and scale together, retaining the original hard bounds and
the achieved higher-priority task images. It audits that seed before refining
the scale; a feasibility witness alone is not an executable command. The
normal predicted-state and command-path acceptance checks still apply.

Use `MIN_ERROR` for a lower-priority posture preference when it should make
the best available use of the remaining freedom. A lower-priority `SCALE`
task still requires its exact scaled direction, including any zero components;
that direction can conflict with an already achieved higher-priority task at
every scale. In that case the solve can return `NO_PROGRESS` even though the
higher-priority task alone has a positive feasible scale. Configuring the
posture as `MIN_ERROR` preserves the higher-priority task image without
requiring the posture's exact direction. This is an explicit task policy;
the solver does not silently relax a requested `SCALE` task.

The model position limits are immutable physical bounds. Optional full-`nq`
`nominal_position_lower_bounds` and `nominal_position_upper_bounds` define a
narrower working envelope inside those bounds for every task mode. Elastic
expansion applies only with an active `SCALE_ELASTIC` registered task or
`CentroidalMomentumRateObjective`, and it is capped by `elastic_delta_max`
and by the physical model bounds. With no separate nominal envelope the
expansion capacity is zero. Measured or caller-owned affine rows, velocity
and acceleration limits, collision floors, CoM polygons, and effort bounds
stay fixed.

Default band parameters are `elastic_delta_max=0.05`,
`elastic_expand_rate=0.01`, `elastic_decay_rate=0.2`, and
`elastic_expand_only_saturated=True`. Delta and expansion use each scalar
joint's coordinate units. Expansion and multiplicative decay apply per solve.
A boundary seed receives a small initial band. Low task scales expand eligible
bands, and unconstrained bands decay. Occupied position and outward braking
reserve are retained up to the configured capacity. States that need a larger
reserve fail closed. Nominal changes reset the band. `reset_elastic_band()`
clears it without replacing caller `q` or `dq`. `get_elastic_band_deltas()`
exposes the persistent state. Results expose the band used for the attempt,
the effective nominal bounds, and each side's expansion capacity. Post-solve
expansion and decay run only after a successful native solve. This band is
solver adaptation, not accepted physical history.

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

Set `options.refine_partial_scale = True` to request one additional bounded
refinement when a directional SCALE objective achieves only part of its target.
This is disabled by default. Refinement preserves earlier task images and
original hard bounds; a larger primary scale can change lower-priority errors
and increase computation time. It does not guarantee nonlinear convergence or
recovery from an infeasible state.

Set `options.prefer_direct_min_error_qp = True` to prefer feasible-primal
quadratic optimization for an all-`MIN_ERROR` hierarchy after a complete hard-set
feasibility witness is available. Each earlier achieved task image becomes an
equality for later objectives. The default is `False`. This can improve minimum
error quality or reduce inverse operations; latency depends on the objective
structure and quadratic iterations. An unsuccessful direct attempt returns to
the established search.
It does not bypass original SCALE/ELASTIC attempts or authorize new fallbacks.
State-box-only solves that skip the full feasibility search retain their
existing route. Physical and nonlinear acceptance checks still apply.

When the combined box requires braking, zero acceleration is outside the hard
set. The default `allow_state_box_task_fallback = True` makes the resulting
MIN_ERROR task fallback explicit while preserving the mandatory braking
command. Set it to `False` when strict task semantics should fail closed
instead. `state_box_task_fallback_applied` reports which policy was used.

For optional posture preferences, `options.allow_best_effort_lower_tasks = True`
can retain an already validated higher-priority command when numerical recovery
cannot optimize a trailing MIN_ERROR task. The default is `False`. This does
not change a failed primary task into success: at least one earlier objective
must have completed, and its achieved image and every hard row are rechecked.
An infinite task condition number can reflect a redundant or masked task row;
it does not invalidate a finite command that passes those checks. Unfinished
lower objectives remain marked as unoptimized, and nonlinear acceptance still
applies.
Lower-task optimality is not certified; the result reports the recovery in its
status message and fallback flags. Physical and nonlinear acceptance checks
still apply, so this option cannot guarantee an executable command.

### Updating option lists

Python acceleration-option lists return owned snapshots of their records.
To update a stored record, edit the snapshot and assign the list back:

```python
saved_rows = options.affine_constraints
options.affine_constraints = []
# saved_rows remains valid after the source list is replaced.
options.affine_constraints = saved_rows

zmp_rows = options.zmp_constraints
zmp_rows[0].fz_min = 10.0  # Assumes a ZMP constraint was already configured.
options.zmp_constraints = zmp_rows
```

The same snapshot semantics apply to task bounds, centroidal objectives/bounds,
contacts, and geometric constraint lists. Editing a snapshot alone does not
change the stored options; reassign it to apply the edit.

Only `SolverStatus.SUCCESS` carries executable acceleration, next-velocity, and
configuration outputs. Failures clear those outputs.

Machine-roundoff excursions outside the compatible acceleration box are
snapped to the original violated face before coupled checks and integration.
This prevents an almost-zero outward acceleration from creating a falsely
infeasible predicted boundary state. Larger violations retain their existing
validation policy. Joint state-box errors include the joint index, position, velocity
and active position bounds to help reproduce a rejected state.

A roundoff-sized boundary or braking-envelope failure can instead select a
single braking command certified within the original physical acceleration caps.
It first tries a scale-free gentler witness using the same local arithmetic
budget, then retains the full-cap witness as a fallback. Each candidate must
pass the same endpoint, rate, in-step stopping, declared-braking, and nominal
next-box checks. This selects one certified command; it does not expose a full
feasible acceleration interval. This
preserves the supplied position and velocity. The command must stop outward
motion within the control interval, respect position endpoints and rate limits,
and leave a usable next-state box. Stopping vertices are checked against a
local position arithmetic budget, including the configured braking policy;
this permits tiny numerical excursions, not exact real-number containment.
All coupled constraints and nonlinear acceptance checks still apply. Larger
outward motion and states without a valid braking command remain rejected.

For successful solves without generalized acceleration allocation, a small
numerical acceleration residual can be projected into the original state box.
The corrected command must retain the original hard-row certificate and achieved
objective images at their existing tolerances, then pass every physical and
nonlinear validator. Already in-box commands are unchanged. This correction does
not alter the supplied state or relax its limits.

If a CoM acceleration bound passes the initial linear model but fails along
the sampled command path, the solver can perform up to eight corrective solves.
Each adds local CoM acceleration rows at violating path samples and re-solves
the original objectives with every original hard constraint retained. The
local rows differentiate the complete constant-acceleration path, including
changes in configuration, velocity, the CoM Jacobian and its acceleration bias.
Central differences share evaluations across polygon edges at the same sample
time. They add work per correction and can reduce the number of retries;
latency depends on the model and command.

Geometric path checks retain fixed quarter-interval samples alongside every
point of the adaptive grid. Increasing the grid density therefore cannot drop
these fixed checks. Existing adaptive sample times keep their original
floating-point calculation. CoM corrective rows use the same combined schedule, and the
validation budget covers all samples. This remains sampled validation; it does
not certify the continuous path between samples.

The corrected command must pass all existing acceptance checks. These local rows
do not establish global nonlinear optimality or continuous collision safety.
Accumulated local rows can exclude a feasible nonlinear solution, so failure
to complete refinement is not a proof of physical infeasibility.
For tight-frame pose constraints, translation and rotation state-bound and endpoint braking
violations can also trigger bounded corrective solves. The local model includes
both the endpoint configuration and velocity derivatives. Each correction
retains the original hard rows and objective policies, adds a local acceleration
trust box, and repeats the complete nonlinear acceptance checks. Rotation rows
differentiate the SO(3) log coordinates and their rates, including the moving
expression frame. Chart and predicted-support failures retain their rejection
behavior. Translation, rotation and CoM corrections share the bounded retry
budget; enabling a corrective direction does not guarantee faster convergence.
At a violating frame sample, the solver groups other violated active scalar
faces at that sample and the endpoint into the same local solve. Endpoint
braking checks apply only at the endpoint, and the group shares one trust box.
These corrections do not establish global recovery or nonlinear optimality.
`result.nonlinear_refinement_iterations` reports the number of corrective
solves; zero means the original candidate needed no correction. Reported
timings include the retries. Exhausted or unsuccessful corrections return
`NUMERICAL_ERROR` with cleared motion outputs.

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

Each call takes explicit `q` and `dq`. Keep them synchronized with the measured
or actually applied state when switching controllers or rejecting a result.
A rejected result does not establish zero physical velocity; do not silently
replace incoming `dq` with zero.

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
Row construction examines the current pair catalog, bypassing historical
candidate-subset and approximate row reuse. Current-state geometry broadphase
culling remains available. This avoids missing a newly active pair during
explicit-state solves or hypothetical stopping rollouts; a successful solve
does not certify that a complete future stopping trajectory exists.
The lifted rows constrain `dq_next = dq + dt * ddq`. Every allowed pair at
every requested constant-acceleration sample is accounted for in canonical
order. A conservative enclosing-sphere lower bound may prove a pair safe;
ambiguous geometry falls back to the exact collision backend.

This path is fail-closed but sampled. A successful result reports
`collision_endpoint_validated = True` and
`collision_step_certified = False`; increasing validation substeps does not
turn samples into a continuous swept-path proof.

### Optional sampled collision correction

`VelocityCollisionLiftOptions.max_refinement_iterations` defaults to zero.
A positive value (at most eight) allows bounded local corrections after a
sampled collision rejection on a fixed-base scalar revolute/prismatic model.
For example:

```python
import embodik

lift = embodik.VelocityCollisionLiftOptions()
lift.max_refinement_iterations = 2
lift.refinement_trust_radius = 0.5  # Joint acceleration coordinate units / s^2.
lift.refinement_clearance_reserve = 1e-6  # Metres; added to local planning rows.
result = solver.solve_with_velocity_collision(collision_solver, q, dq, dt, options, lift)
```

Each correction adds a hard linearized distance row at the failed sample and a
hard acceleration trust box. Earlier correction rows always remain hard.
If a corrective solve fails numerically, makes no progress, or reports
infeasibility, the wrapper
may remove only its internally appended trust boxes and retry over the original
physical domain, provided the original solve succeeded and retry budget remains.
Caller rows (including caller trust boxes) are never removed. An unrestricted
retry consumes the same budget, adds no distance query, and is not repeated
without a new internal trust box. Original tasks, modes, allocation, limits, frozen
collision rows, CoM definitions, and incoming state remain unchanged. All
existing acceptance checks run again. Separated pairs that start below their floor retain the validator's original
non-worsening threshold. Penetrating geometry,
unsupported joints, or unavailable finite gradients retain the existing sampled
rejection; this facility does not supply global recovery or feasibility proofs.
A failed local solve returns `NUMERICAL_ERROR` with cleared motion outputs,
because infeasibility of a local model does not prove physical infeasibility.
Budget exhaustion retains `COLLISION_VIOLATED` with cleared outputs.

`collision_refinement_iterations` counts extra full solves, including a
retry after internal trust removal, and is separate from
`nonlinear_refinement_iterations`, which counts CoM and tight-frame translation corrections across these
attempts. `collision_refinement_exact_queries` and its kinematics/geometry
counters report extra local-model work, separately from sampled-validation
counters. Validation counters accumulate across attempts. Inner phase timings
sum the inner solves; total time also includes row lifting, validation, and local
model construction. Failed opt-in transactions restore elasticity and report
that retained state. The default zero-budget path retains legacy elasticity
behavior. No setting provides swept-path certification:
`collision_step_certified` remains false.

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
across minor releases. The `0.23.2` robustness update also extends public
option and result layouts: native consumers must rebuild against the matching
headers, core library, and Python bindings, including when upgrading from an
earlier `0.23.x` build.

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
