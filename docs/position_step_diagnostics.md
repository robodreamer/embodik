# Inspect position-step integration

Enable `PositionStepOptions.record_velocity_candidates` to record the native
velocity and nominal integration for each iteration of `solve_position_step()`:

```python
options = embodik.PositionStepOptions()
options.record_velocity_candidates = True
result = solver.solve_position_step(q, targets, options)
for candidate in result.velocity_candidates:
    print(candidate.step_index, candidate.integration_dt)
    print(candidate.solver_velocity, candidate.integrated_velocity)
```

The option defaults to `False`. Recording does not change targets, constraints,
integration, or candidate acceptance. Both the single-target and multi-target
overloads support it.

Each `PositionStepVelocityCandidate` contains:

| Field | Meaning |
| --- | --- |
| `step_index` | Zero-based velocity iteration within this call |
| `integration_dt` | Actual integration interval, including adaptive scaling |
| `reference_q` | Configuration before this nominal integration |
| `solver_velocity` | Native velocity before integration clamps and projections |
| `integrated_velocity` | Manifold difference from reference to candidate divided by the integration interval |
| `candidate_q` | Nominal configuration before priority, collision, and final acceptance guards |
| `task_scales` | Scales from this iteration's native velocity solve |

Velocity fields have dimension `nv`; configurations have dimension `nq`.
They differ for continuous and floating-base joints. Use the robot's manifold
difference rather than subtracting configurations to reconstruct velocity.

These are candidate records. A later guard may backtrack or reject the
configuration, so the last record need not equal `result.q_solution`. Task
scales are local to each velocity solve and do not certify final pose tracking.
The records do not include every retry or continuous swept-path safety.
