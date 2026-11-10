# 0.24.0 release scope

0.24.0 is a numerical-library foundation release: constrained velocity IK
reliability fixes, standalone CPU optimization interfaces, explicit diagnostics,
and low-overhead introspection. It is not a completed hardware whole-body
controller or a universal recovery guarantee.

## Included contracts

| Area | Release scope |
|---|---|
| Default CPU IK | Existing velocity-level API remains the default; documented task and constraint semantics apply |
| Standalone optimization | Experimental dense CPU LS and positive-definite QP interfaces, including their documented unsupported cases and numerical statuses |
| Backend discovery | Native numerical handles and immutable capability metadata only; existing dispatch/defaults are unchanged |
| Collision introspection | Fresh floor metadata without geometric evaluation; exact distance and physical validation APIs remain separate |
| Acceleration API | Experimental, opt-in fixed-base per-step acceleration solving/validation, with the constraints and exclusions in its guide |
| GPU WBC | Experimental, opt-in implementations and model/device scope documented in the GPU guide; separate from standalone CPU math |

Native C++ interfaces/layouts changed from 0.23.2. Rebuild native applications,
core and bindings together using matching headers. Do not mix binaries from
different releases.

## Explicitly unqualified or deferred

- General recovery after sustained saturation, large out-of-reach target returns,
  or arbitrary changes of both end-effector tasks.
- Complete goal-preserving stopping, automatic re-command continuity and natural
  interactive behavior for every acceleration/controller profile.
- Hard 100 Hz control or sub-millisecond complete WBC-loop latency. A nominal dt,
  fast numerical call or aggregate batch throughput is not a wall-clock guarantee.
- Dynamic contact-force/friction-cone control and hardware-ready acceleration WBC
  beyond the documented fixed-base numerical/kinematic constraints.
- ElastiQP/OSQP/Clarabel provider integration, persistent backend workspaces,
  Frax/FRDV reference adapters and new alternative control engines.

These are deferred capabilities, not passed release gates. A fail-closed local
rejection is not proof that no global trajectory exists. A successful numerical
solve is not a complete motion or hardware execution certificate. Applications
retain responsibility for actual-state ownership, stopping/admission policy and
actuator execution within their supported scope.

The [robustness goals](solver_robustness.md) and visual/usability criteria remain
in force when qualifying a profile. This release does not claim a complete
interactive-usability pass for unqualified profiles. Optional experimental paths
remain explicit and do not replace the default CPU API.

## Guides

- [Standalone optimization](optimization.md)
- [Backend discovery](backends.md)
- [Acceleration solver](acceleration_solver.md)
- [GPU solvers](gpu_solvers.md)
- [Collision constraints](collision_constraints.md)
