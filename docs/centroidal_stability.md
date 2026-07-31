# Centroidal Stability

EmbodiK exposes centroidal momentum, capture-point, and ZMP controls at both
velocity and acceleration level. These APIs are solver constraints, not a
contact planner or a whole-body dynamics controller.

## Physical Conventions

Centroidal vectors use six rows in this order:

```text
[linear x, y, z; angular x, y, z]
```

The corresponding units are:

| Quantity | Linear rows | Angular rows |
| --- | --- | --- |
| Momentum `h` | kg m/s | kg m^2/s |
| Momentum rate `hdot` | kg m/s^2 | kg m^2/s^2 |

`Ag(q)` maps generalized velocity to momentum, `h = Ag * dq`.
`dAg(q, dq) * dq` is the affine momentum-rate bias, so physical momentum rate
is `hdot = Ag * ddq + dAg * dq`.

Gravity is a world-frame acceleration vector. Examples use
`[0, 0, -9.81] m/s^2`; a support frame must observe gravity pointing toward
its negative vertical axis. Support polygons are planar `(x, y)` vertices in
their named frame, with metres as the unit. A support frame must be world or
structurally root-fixed. Moving support frames fail closed.

## Model Quantities

`RobotModel` provides:

- `get_total_mass()`
- `get_centroidal_momentum_matrix()`
- `compute_centroidal_momentum_matrix(q, dq)`
- `compute_centroidal_momentum(q, dq)`
- `compute_centroidal_momentum_matrix_time_variation(q, dq)`
- `compute_centroidal_momentum_matrix_bias(q, dq)`

The state-taking methods do not depend on a prior solve. Wrong-size or
non-finite state vectors are rejected.

## Velocity Solver

`CentroidalMomentumTask` commands absolute momentum:

```text
Ag(q) * dq_command ~= h_target
```

`configure_centroidal_momentum_bounds()` adds selected-axis hard bounds in the
same momentum coordinates. `configure_capture_point_constraint()` constrains
the point computed directly from commanded CoM velocity:

```text
xi = com_xy + (Jcom_xy * dq_command) / omega
```

Velocity ZMP uses a finite difference from an explicit current `dq` to the
candidate command. Call `solve_velocity_with_state()`, or set
`PositionStepOptions.current_joint_velocity` before `solve_position_step()`,
whenever `configure_velocity_zmp_constraint()` is active:

```text
ddq_fd = (dq_command - current_dq) / dt
hdot = Ag(q) * ddq_fd + dAg(q, current_dq) * current_dq
```

Calling `solve_velocity()` without that explicit state is rejected. The solver
does not seed the next call from a previous command, an EMA, or other hidden
history. `evaluate_capture_point_constraint()` and
`evaluate_velocity_zmp_constraint()` expose accepted-point and slack evidence.

For the primary interactive API:

```python
options = embodik.PositionStepOptions()
options.current_joint_velocity = current_dq
result = solver.solve_position_step(q, target_pose, "tool", options)
current_dq = result.joint_velocities
```

The position-step path propagates the explicit state through inner iterations
and retries, then revalidates commands changed by integration masks, joint-limit
projection, collision backoff, or recovery before accepting them.

## Acceleration Solver

`CentroidalMomentumRateObjective` tracks:

```text
Ag * ddq + dAg * dq =
    hdot_feedforward + proportional_gain * (h_target - h)
```

`CentroidalMomentumRateBounds` applies physical `hdot` bounds. The predicted
capture-point constraint follows the constant-acceleration integration rule:

```text
a = Jcom * ddq + Jdotcom * dq
c_next = c + dt * v + 0.5 * dt^2 * a
v_next = v + dt * a
xi_next = c_next_xy + v_next_xy / omega
```

`CapturePointAccelerationConstraint` freezes `omega` for one solve. If omitted,
the solver derives it from support-frame CoM height and vertical gravity.

`ZmpAccelerationConstraint` uses physical centroidal momentum rate. It forms
the net support force from linear momentum rate and gravity, requires a
positive vertical force `Fz >= fz_min`, and cross-multiplies each polygon
half-plane only after imposing that denominator gate. The ZMP formula includes
the CoM-height and horizontal-force terms; it has no dimensionally unsupported
angular-momentum proxy.

Every successful acceleration result is recomputed at the nonlinear predicted
configuration and velocity. `capture_point_diagnostics` and `zmp_diagnostics`
report the point, half-plane slacks, force, frozen omega, and whether that
postvalidation completed.

## Capability Boundary

The acceleration implementation is fixed-base only. Capability flags advertise
fixed-base centroidal momentum-rate, capture-point, and ZMP support while
`supports_dynamic_balance = False`.

This fixed-base result does not prove contact-force feasibility. The rows do
not allocate contact wrenches, enforce friction cones,
model unilateral contacts, or support floating-base contact dynamics. A caller
that needs those guarantees must use a contact-dynamics solver rather than
interpreting fixed-base ZMP success as a dynamic-balance certificate.

## Examples And Tests

Run the visual Panda support example or the opt-in bimanual integration:

```bash
pixi run python examples/04_com_constraint_example.py --robot panda
pixi run python examples/06_bimanual_whole_body_ik.py
```

The non-visual velocity/acceleration composition checks live in
`test/test_centroidal_stability_integration.py` rather than a public script.
See [CoM And Centroidal Support](examples/com_constraint_ik.md) and
[Bimanual Whole-Body IK](examples/bimanual_whole_body_ik.md) for the visual
controls.
