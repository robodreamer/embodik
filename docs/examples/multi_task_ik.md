# Multi-Task Multi-Constraints IK Example

Example demonstrating hierarchical multi-task inverse kinematics with multiple constraints: frame task, posture task, and CoM support-polygon constraint.

## API Walkthrough

Use the current registered-task API for multi-task examples:

| Step | API calls | Purpose |
| --- | --- | --- |
| Resolve the robot | `resolve_robot_configuration("panda")` | Load the same model, target link, and default posture used by the examples. |
| Add the primary task | `solver.add_frame_task("ee_task", target_link)` | End-effector tracking at priority `0`. |
| Add a posture task | `solver.add_posture_task("posture")`, `set_target_configuration(q_default)` | Nullspace posture objective at a lower priority. |
| Add CoM constraint | `solver.configure_com_constraint(...)` | Enforce the support polygon as a hard inequality constraint. |
| Step or iterate | `solve_position_step(...)` for interactive loops, `solve_position(...)` for bounded solve calls | Use position-level APIs rather than hand-maintaining a partial velocity loop in docs. |
| Read diagnostics | `status`, `q_solution`, `task_scales`, `task_modes_effective`, `task_used_fallback` | Decide whether to accept the update and how to display task behavior. |

## Explanation

Multi-task multi-constraints IK uses a registered solver stack:

1. **Frame task** (priority 0): Drives the end-effector toward the target pose.
2. **Posture task** (priority 1): Maintains preferred joint configuration in null space via `set_target_configuration()`
3. **CoM constraint** (`configure_com_constraint`): Keeps the 2D CoM projection inside the support polygon — a hard inequality constraint, not a task

For a runnable interactive demo with CoM visualization, see
[CoM Constraint Example Overview](com_constraint_ik.md).

## Task Hierarchy

Tasks are solved hierarchically:

- Higher priority tasks are satisfied first
- Lower priority tasks are satisfied in the null space of higher priority tasks
- `configure_com_constraint` adds inequality constraints that the solver enforces regardless of task priorities

## Next Steps

- [Basic IK Example](basic_ik.md) — Simpler example
- [Collision-Aware IK Example](collision_aware_ik.md) — Self-collision avoidance
- [API Reference](../api/index.md) — Detailed API documentation
