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
| Collision compatibility | Canonical velocity collision rows lifted to next velocity, followed by exact sampled validation |
| Native collision certification | Exact-name sphere-sphere catalogs on fixed-base all-prismatic scalar models |

Unsupported scope fails explicitly. This includes floating bases, dynamic
contact forces, friction cones, wheel rolling/steering constraints,
`SCALE_ELASTIC`, general native collision geometry, and ROS 2 controller or
wheelbase integration.

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

Only `SolverStatus.SUCCESS` carries executable acceleration, next-velocity, and
configuration outputs. Failures clear those outputs.

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

## Collision Modes

### Velocity-row compatibility

`solve_with_velocity_collision()` consumes an already configured
`KinematicsSolver`. It reuses that solver's collision pair filtering,
include/exclude policy, active per-pair floors, row ordering, and row budget.
The lifted rows constrain `dq_next = dq + dt * ddq`, and exact distance checks
validate the requested constant-acceleration samples.

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

The acceleration API is additive to the `0.20.18` integration baseline. The
shared `Task` virtual interface and object layout are unchanged, but source
builds must rebuild `embodik_core` and the Nanobind extension together before
using the new Python types. Do not publish a changed binary under the existing
`0.20.18` package version; the next release must bump package metadata and move
the Unreleased changelog entries into a dated section.

## API Reference

::: embodik.AccelerationSolver

::: embodik.AccelerationSolveOptions

::: embodik.AccelerationSolverResult
