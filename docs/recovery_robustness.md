# Recovery robustness (hardware seeds)

This note maps how EmbodiK behaves when the initial configuration is slightly outside joint limits or in shallow self-collision—typical when `q` is read from encoders before constraints have “caught up.” It complements [`joint_limit_saturation_exit_findings.md`](joint_limit_saturation_exit_findings.md).

## Code map (where behavior lives)

| Mechanism | Location | Role |
|-----------|----------|------|
| Joint-limit velocity box | [`cpp_core/src/kinematics_solver.cpp`](../cpp_core/src/kinematics_solver.cpp) — `calculate_velocity_box_constraint()`, `solve_velocity()` position rows | Margins from current `q` vs URDF limits (with `margin_limit = 1e-4` in `solve_velocity`). Outside a joint: recovery term pushes velocity back inside; if **both** lower and upper margins are violated, the box widens to `[-vmax, vmax]` (no unique recovery direction). |
| Joint-limit tunables | [`cpp_core/include/embodik/kinematics_solver.hpp`](../cpp_core/include/embodik/kinematics_solver.hpp) | `set_limit_recovery_gain`, `set_limit_recovery_hysteresis`, `set_limit_exit_release_margin`, `enable_saturation_exit_behavior` (off by default). |
| Joint-limit barrier task | Same `.cpp` — `set_joint_limit_barrier_task()` | Optional priority-1 gradient pushing away from limits in the nullspace. |
| Self-collision rows | Same `.cpp` — `compute_collision_constraint()` | Separation velocity lower bound with deadbands, stronger push when penetrating (`signed_distance < 0`). |
| Stall handler | Same `.cpp` — `enable_stall_handler()`, `stall_handler_update()` | After repeated stalls, can **lower** effective `min_distance`; for penetration, sets margin below measured depth so the QP row gains slack. |
| Outcome classification | Same `.cpp` — `classify_velocity_outcome()` | Successful QP can still be reported as `kInfeasible` if the primary task scale collapses to zero under constraints. |
| Position IK inner loop | Same `.cpp` — `solve_position()` | Uses **0.02 rad** joint margin in its velocity box (`lower_margin = q - qmin - 0.02`), **not** `1e-4` like `solve_velocity`. |
| Position step | Same `.cpp` — `solve_position_step()` | Calls `solve_velocity(q, true)` then `pinocchio::integrate`; optional `PositionStepOptions.stall_recovery` enables stall handler across steps. **No** explicit post-step clamp of `q` to limits in this path. |
| Input sanitization | Same `.cpp` — `sanitize_solver_inputs()` | Non-finite constraint bounds are widened; `lower > upper` rows are collapsed to midpoint (avoids hard throw from bad numerics). |

### Failure / stall modes (invalid seeds)

1. **Task vs limits vs collision**: Primary task, limit rows, and collision rows can be jointly unsatisfiable → `INFEASIBLE` or near-zero motion.
2. **Both joint limits violated** (physically inconsistent seed): velocity box allows any velocity in `[-vmax, vmax]` for that joint—recovery direction is undefined; do not expect monotonic return to limits without an external bias (e.g. posture task toward interior).
3. **Margin mismatch**: `solve_velocity` vs `solve_position` use different spatial margins for limits → different strictness between one-shot velocity IK and multi-step position IK.
4. **Collision without stall**: Deep penetration or very tight `min_distance` can yield persistent stalls; stall recovery is **opt-in** (`enable_stall_handler` / `stall_recovery`).
5. **No post-integration projection**: If integration + numerical tolerance leave `q` slightly out of bounds, there is no dedicated `q` projection in `solve_position_step`.

## Regression tests

See [`test/test_hardware_seed_recovery.py`](../test/test_hardware_seed_recovery.py) for:

- Slight joint-limit violations with `solve_velocity` and integration (recovery into `[qmin, qmax]`).
- `solve_position_step` from a violated seed with a frame task toward the interior.
- Two-link collision model: shallow-clearance seed and trend of minimum distance / tolerable statuses.
- Optional `stall_recovery` on position steps in a near-contact regime (when a penetrating sample exists in the search range).

Run: `pixi run pytest test/test_hardware_seed_recovery.py` (from repo root).

### Pytest matrix (implemented)

| Test | What it checks |
|------|----------------|
| `test_solve_velocity_recovers_from_slight_lower_limit_violation` | Integration loop from `q < q_min` with joint task toward interior. |
| `test_solve_velocity_recovers_from_slight_upper_limit_violation` | Same for `q > q_max`. |
| `test_solve_position_step_from_limit_violation_moves_toward_feasible_region` | `solve_position_step` + frame task; tolerates `INFEASIBLE` if `q` moves inward. |
| `test_penetrating_seed_strict_collision_stalls_without_stall_handler` | Penetration + `min_distance` without stall → sustained zero velocity. |
| `test_penetrating_seed_stall_handler_improves_clearance` | Parametric **off vs on**: stall path increases signed distance and successes. |
| `test_solve_position_step_stall_recovery_on_penetrating_seed` | `PositionStepOptions.stall_recovery=True` on penetrating seed. |
| `test_combined_joint_at_limit_and_penetration_with_stall` | Lower limit + penetration; stall + joint task toward zero. |
| `test_panda_solve_velocity_recovers_from_slight_limit_violation` | Real-model check: slight Panda limit violations recover with `SUCCESS` and inward velocity. |
| `test_panda_position_step_recovers_from_slight_upper_limit_violation` | Real-model check: Panda `solve_position_step()` returns `SUCCESS` and re-enters the limit band. |
| `test_dual_iiwa_stall_recovery_improves_status_counts_and_task_progress` | Real-model check: dual-iiwa stall recovery trades margin for fewer stalls / infeasible steps and better task progress. |

## Real-model outcomes

The synthetic hinge / two-link tests are still useful for isolating mechanisms, but the
current evidence for an opt-in recommendation now comes from **Panda** and **dual iiwa**.

### Panda

- Slight violations of `panda_joint1` on either side recover under `solve_velocity()` with
  **`SUCCESS`** on every checked step and velocity pointing back toward the valid range.
- A slightly over-limit Panda seed also recovers under `solve_position_step()` with the
  joint-limit barrier enabled; over a small multi-step budget, `q` returns to the limit
  band while the solver still reports **`SUCCESS`**.

This is the strongest current signal that EmbodiK can already be *forgiving* on realistic
hardware-seeded joint-limit errors without requiring an outer recovery module.

### Dual iiwa

- The cross-arm stall case does **not** show a clean “more clearance is always better”
  narrative. Instead, the stall handler improves behavior by reducing freezes and
  infeasible steps while allowing the controller to keep making task progress.
- In the validated case, enabling stall recovery produces:
  - more `SUCCESS` steps,
  - fewer `INFEASIBLE` steps,
  - fewer near-zero-velocity stall steps,
  - lower end-effector target error,
  - and a **reduced active collision margin** while recovering.

So for collision-heavy startup states, the current EmbodiK story is:

- **without** stall recovery: stricter clearance semantics, but more risk of freezing;
- **with** stall recovery: better escape / task continuation, but only by temporarily
  relaxing the effective minimum distance.

## Strategy comparison (baseline vs HMND-style)

| Approach | Pros | Cons / notes |
|----------|------|----------------|
| **A — Default EmbodiK** (limits + collision + default gains) | No extra moving parts; good for nominal control. | May report `INFEASIBLE` or stall when seed + task + constraints conflict; collision escape may need stall for hard cases. |
| **B — Tune limits** (gain, hysteresis, release margin, optional barrier) | Addresses encoder jitter and boundary chatter; barrier adds nullspace push. | Barrier competes with other P1 tasks; tuning is robot-specific. |
| **C — Stall / `stall_recovery`** | Automatic temporary collision margin relaxation; penetration escape path in C++. | Changes safety envelope while active; must restore to nominal margin when healthy. |
| **D — `enable_saturation_exit_behavior`** | Extra velocity headroom near saturation for SNS feasibility (commented in C++). | Off by default; needs regression coverage before wide use (see `test_joint_limit_exit_symmetry.py`). |
| **E — HMND WBC-style outer recovery** (`hmnd_robot` / `hmnd_robots.wbc_recovery.QPRecovery`: slacks, trust region, gradient fallback, playback) | Strong “find a feasible `q` seed” story before main QP. | Separate optimizer (OSQP); not inside EmbodiK today; integrate at app layer if needed. |
| **F — Post-recovery hysteresis** (WBC interface pattern) | Avoids re-failing on settling encoders after a successful escape. | Policy/state in the controller node, not the IK library. |

**Practical split**: EmbodiK (B–D) handles **continuous** escape during control; HMND-style (E–F) handles **discrete** “re-seed before main solve” and logging gating.

## Recommended hardening sequence

1. **Measure** with `test_hardware_seed_recovery.py` on target URDFs (extend fixtures if needed).
2. **Prefer tuning** (`limit_recovery_*`, optional barrier, collision `min_distance`) before new C++ features.
3. **Unify or document** the `1e-4` vs `0.02` margin split between `solve_velocity` and `solve_position`; add a regression if you change behavior.
4. **Enable stall recovery** on interactive / hardware paths where shallow collision is expected at startup, but treat it as an explicit safety/performance trade-off.
5. **Opt in** only after confirming on the target robot that:
   - slight joint-limit violations recover with `SUCCESS`,
   - stall recovery reduces freezes more than it harms your effective clearance margin,
   - and the controller-level safety policy accepts temporary margin relaxation.
6. **Consider** `enable_saturation_exit_behavior` only after extending saturation symmetry tests to your robot profile.
7. **If still stuck**, add an application-layer recovery (E) that outputs a short `q` path, then run EmbodiK from the last waypoint—mirrors WBC without bloating the core solver.

## Known pitfall: `task.current_orientation` before first update

`FrameTask.current_orientation` returns **uninitialized memory** if read before any
`solve_velocity()` or `task.update()` call. Test code that does
`start_rot = task.current_orientation` as a target rotation will produce
heap-dependent results—passing in isolation but failing when prior tests
allocate differently. Always use `robot.get_frame_pose(frame).rotation` instead.
This was the root cause of an order-dependent failure in
`test_min_error_panda_regression.py` (fixed in the same batch as the recovery tests).

## References

- [`cpp_core/src/kinematics_solver.cpp`](../cpp_core/src/kinematics_solver.cpp)
- [`test/test_joint_limit_recovery.py`](../test/test_joint_limit_recovery.py)
- [`test/test_stall_handler.py`](../test/test_stall_handler.py)
- HMND (separate repo): `hmnd_robots/wbc_recovery.py`, `whole_body_controller/whole_body_controller_interface.py` (recovery probe loop, playback, post-recovery log throttle)
