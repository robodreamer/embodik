# Quickstart Guide

Get started with EmbodiK by running the maintained examples first, then adapt
their API pattern to your robot.

## Run A Maintained Example

Install and copy the example bundle using the
[Installation Guide](installation.md#examples), then run:

```bash
cd embodik_examples
python 01_basic_ik_simple.py
```

From a repository clone:

```bash
pixi run python examples/01_basic_ik_simple.py
```

## Current IK API Pattern

EmbodiK examples use registered tasks on a `KinematicsSolver`; they do not
instantiate task objects directly.

| Step | API calls | Purpose |
| --- | --- | --- |
| Load or resolve a robot | `RobotModel(...)` or an example `resolve_robot_configuration(...)` helper | Provide the kinematic model, default configuration, and target frame names. |
| Create a solver | `KinematicsSolver(robot)`, `solver.dt` | Configure solver timing for position updates. |
| Enable example runtime policy | `configure_solver_runtime_policy(solver)` | Public examples opt into solver-owned pose-layout auto-switching and constrained weighted fallback. |
| Register a frame task | `solver.add_frame_task("ee_task", target_link)` | Track an end-effector pose target. |
| Add posture bias | `solver.add_posture_task("posture")`, `set_target_configuration(q_default)` | Keep unused freedom near a preferred posture. |
| Configure options | `PositionIKOptions()` or `PositionStepOptions()` | Set gains, iteration counts, timestep behavior, torso constraints, or stall recovery. |
| Solve | `solve_position(...)` or `solve_position_step(...)` | Return a result with `status`, `q_solution`, error metrics, and task diagnostics. |

## Position Solve

Use `solve_position()` when you want EmbodiK to iterate toward one target pose
inside a bounded solve call:

```python
opts = embodik.PositionIKOptions()
opts.max_iterations = 20
opts.position_gain = 40.0
opts.orientation_gain = 40.0

result = solver.solve_position(seed_q, target_pose, "panda_hand", opts)
if result.status == embodik.SolverStatus.SUCCESS:
    q_next = result.q_solution
```

This path creates the internal objective stack for the call. It is the right
starting point for offline checks, retargeting steps, and scripts that do not
need a persistent interactive task stack.

## Interactive Step Solve

Use `solve_position_step()` when a UI, teleop stream, or tracking loop updates a
target every frame. Register tasks once, then call the step solver repeatedly:

```python
ee_task = solver.add_frame_task("ee_task", target_link)
ee_task.priority = 0
ee_task.weight = 1.0

posture = solver.add_posture_task("posture")
posture.priority = 1
posture.set_target_configuration(q_default)

step_opts = embodik.PositionStepOptions()
step_opts.max_steps = 1
step_opts.adaptive_dt = True

result = solver.solve_position_step(q_current, target_pose, "ee_task", step_opts)
q_current = result.q_solution
```

Examples `01_basic_ik_simple.py`, `02_collision_aware_IK.py`, `03_teleop_ik.py`,
and the bimanual demos use this pattern.

## Example Runtime Policy

Maintained interactive examples call
`examples/example_helpers/ik_common.py::configure_solver_runtime_policy()` after
constructing the solver. `weighted_fallback_enabled` is already true by default
in `SolverRuntimeConfig`; the helper keeps that default explicit for examples
and enables pose-task auto layout:

- `SolverRuntimeConfig.enable_auto_task_layout = True`
- `SolverRuntimeConfig.weighted_fallback_enabled = True`

The first option lets solver-owned `PoseTaskGroup` adapters switch between
merged and split pose-task layouts at solve boundaries. The second option allows
the constrained weighted candidate to replace a non-success prioritized solve
only when it satisfies the active hard constraints. It does not replace collision,
CoM, relative-pose, contact projection, or joint-limit constraints with an
unconstrained least-squares solve.

For controlled comparisons, copy the current config, set either flag to `False`,
and call `solver.configure_runtime(cfg)`.

## Adding Constraints

Constraints are configured on the solver and are enforced during the next solve:

| Constraint | API | Used by |
| --- | --- | --- |
| Self-collision avoidance | `solver.configure_collision_constraint(...)` | `02_collision_aware_IK.py`, dual-arm and whole-body examples |
| CoM support polygon | `solver.configure_com_constraint(...)` | `04_com_constraint_example.py`, whole-body examples |
| Torso orientation or pose bounds | `opts.torso_constraint...` | floating-base and whole-body position solves |

## Configuration-Space Operations

Use the robot model's manifold-aware methods when applying velocities or
comparing configurations, especially for floating-base robots:

```python
q_next = model.integrate(q_current, joint_velocity, dt=0.01)
delta = model.difference(q_start, q_goal)
q_home = model.neutral_configuration()
q_valid = model.normalize(q_maybe_drifted)
```

For fixed-base revolute robots this resembles `q + v * dt`; for floating-base
robots it preserves quaternion and SE(3) validity.

## Next Steps

- [Examples](examples/index.md) — Maintained runnable scripts.
- [KinematicsSolver API](api/kinematics_solver.md) — Solver options and result diagnostics.
- [Tasks API](api/tasks.md) — Registered task types, priorities, and solve modes.
- [RobotModel API](api/robot_model.md) — Model loading, FK/Jacobians/CoM, collisions, and configuration-space operations.
