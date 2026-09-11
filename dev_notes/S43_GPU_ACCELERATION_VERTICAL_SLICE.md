# S43: GPU acceleration-level vertical slice

## Outcome

This session adds a public, device-resident acceleration-level solver without
claiming parity for constraint families that have not yet been validated. The
public slice is exact for fixed-base models made of scalar one-DoF joints and
one diagonal generalized-acceleration `MIN_ERROR` objective.

The implementation is model-derived. Joint count, coordinate spans, position
limits, and velocity limits come from `RobotModel`; acceleration authority is
explicit. There are no robot-family or fixed-DoF tables.

## CPU semantics preserved

- Explicit `q`, `dq`, and `dt` state.
- Acceleration, next-velocity, position-endpoint, continuous-path, and
  next-step braking-viability bounds use the equations and numerical policy
  from `acceleration_state_box.cpp`.
- `ddq = 0`, `ddq = -dq / dt`, and `ddq = -2 dq / dt` implement the CPU zero
  acceleration, zero next velocity, and fixed current position locks.
- The one-step update is `dq_next = dq + dt ddq` and
  `q_next = q + dt dq + 0.5 dt^2 ddq`.
- The generalized-acceleration reference matches a CPU posture `MIN_ERROR`
  task with feed-forward desired acceleration and zero proportional and
  derivative gains.
- Invalid and infeasible worlds fail independently. Device results contain NaN
  motion outputs for failed worlds; the host wrapper returns empty motion
  arrays.

The CPU solver rebuilds a predicted state box only when geometric or collision
acceptance is active. The joint-only GPU slice follows that behavior and does
not add a stricter second rejection.

## Internal preparation for later task families

The session also adds vectorized acceleration capture-point and ZMP affine-row
builders. An internal Newton frame evaluator computes `Jdot*dq` using a
centered directional Jacobian difference. The evaluator is graph-safe and
useful for prototyping, but its capability explicitly reports that the bias is
not exact; it is therefore not exposed by `GpuAccelerationSolver`.

## Validation

- 25 focused tests pass.
- State-box fixtures match an independent scalar transcription of the CPU C++
  equations to float64 tolerance.
- Public CPU/GPU parity passes on generated 2-, 5-, and unseen 9-DoF chains.
- All three lock policies match CPU results.
- A mixed valid/infeasible batch proves per-world fail-closed publication.
- A 1024-world CUDA graph replays changed inputs with stable output storage.
- Floating-base construction is rejected explicitly.
- Internal frame bias tests cover 2-, 5-, and 8-DoF generated chains and match
  Pinocchio within the documented finite-difference tolerance.

## Measured latency

Measured after warm-up on an NVIDIA RTX PRO 5000 Blackwell Laptop GPU using the
19-DoF Spot model with position, velocity, acceleration, continuous-path, and
braking-viability shaping active:

| Path | Batch | Total time | Per world |
|---|---:|---:|---:|
| Captured GPU device solve | 1 | 0.641 ms | 641 us |
| Captured GPU device solve | 1024 | 0.736 ms | 0.719 us |
| CPU host solve | 1 | 0.031 ms | 31 us |
| GPU NumPy host wrapper | 1 | 5.310 ms | 5.310 ms |

The device path reaches sub-millisecond total time and scales nearly flat from
one to 1024 worlds. CPU remains the correct choice for a single host-driven
joint-only call. The host GPU wrapper is a convenience/debug path; real-time
GPU applications should retain state and outputs on CUDA and replay a captured
graph.

## Deliberate capability boundary

The public acceleration solver does not yet claim frame or CoM tasks, multiple
priorities, `SCALE`, floating bases, centroidal momentum-rate/capture-point/ZMP
constraints, contacts, effort bounds, collision constraints, task-local
exclusions, or nonlinear geometric acceptance. Those features require an exact
analytic task bias plus the general constrained hierarchical solve and their
CPU acceptance certificates. `GPU_ACCELERATION_CAPABILITIES` reports each
unsupported family as false, with no silent CPU fallback.
