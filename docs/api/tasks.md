# Tasks

## Centroidal Momentum

`CentroidalMomentumTask` is a velocity-level absolute momentum objective:

```text
Ag(q) * dq_command ~= h_target
```

Its six rows are ordered `[linear x, y, z; angular x, y, z]`. Axis masks and
task-local joint exclusions apply to the objective; solver-owned hard
centroidal bounds remain independent. See
[Centroidal Stability](../centroidal_stability.md) for units and support
constraints.

EmbodiK supports various task types for multi-task inverse kinematics.

## Task Types

### FrameTask

Control end-effector pose (position + orientation).

```python
frame_task = solver.add_frame_task("ee_task", "panda_hand")
frame_task.priority = 0
frame_task.weight = 1.0
```

### PostureTask

Maintain desired joint configuration.

```python
posture_task = solver.add_posture_task("posture")
posture_task.priority = 1
posture_task.set_target_configuration(q_default)
```

### ManipulabilityTask

Improve frame-Jacobian conditioning with the analytic gradient of a regularized
log-determinant score:

```python
conditioning = solver.add_manipulability_task(
    "ee_conditioning",
    "panda_hand",
    embodik.TaskType.FRAME_POSITION,
)
conditioning.priority = 1
conditioning.weight = 10.0
conditioning.solve_mode = embodik.TaskSolveMode.MIN_ERROR
conditioning.set_controlled_joint_indices(arm_velocity_indices)
conditioning.set_regularization(0.03)
conditioning.set_joint_limit_penalty(0.002, epsilon=0.04)
```

The regularization keeps the score and gradient finite at singular
configurations. The gradient magnitude is smoothly bounded below `1` before the
task weight is applied. At an exactly symmetric singularity the first-order
gradient can be zero; use a deterministic nominal `PostureTask` at a lower
priority to select a bend direction when the robot has that symmetry.

The frame task type selects the Jacobian whose conditioning is optimized. Use
`FRAME_POSITION` when translational reach and Cartesian position tracking are
the primary concern. Use `FRAME_POSE` only when improving the combined linear
and angular Jacobian is intentional; on a limited-range arm, its rotational
gradient can otherwise consume nullspace motion without improving position
tracking.

The optional joint-limit penalty uses the descent direction of EmbodiK's
normalized joint-limit distance metric. Before adding that inward direction,
EmbodiK projects away any component of the frame-manipulability gradient that
would worsen the limit metric to first order. The remaining tangential
manipulability component can still improve conditioning without trading away
hard-limit recovery. The projection is evaluated only for controlled scalar
joints and does not add another hierarchy level. The penalty defaults to `0`,
preserving the frame-only metric; `epsilon` regularizes the normalized distance
close to either hard limit.

### JointLimitAvoidanceTask

Move selected scalar joints inward before they become pinned at a hard limit:

```python
limit_avoidance = solver.add_joint_limit_avoidance_task(
    "arm_limit_avoidance",
    controlled_joint_indices=[elbow_velocity_index],
)
limit_avoidance.priority = 1
limit_avoidance.weight = 0.012
limit_avoidance.solve_mode = embodik.TaskSolveMode.MIN_ERROR
limit_avoidance.set_activation_margin(0.002)
```

The task uses a signed cubic smoothstep. It is exactly zero outside the
activation margin, rises continuously toward either limit, and never replaces
the solver's hard position or velocity constraints. The margin uses each
joint's configuration units (`rad` for revolute joints, `m` for prismatic
joints).

For limited-range arms, a useful hierarchy is:

1. Position tracking at priority `0`.
2. Joint-limit avoidance at priority `1`.
3. Orientation recovery and manipulability conditioning at priority `2`.

This split permits a continuous bend away from a fully extended limit while
position continues to make progress. For these recovery objectives, near-limit
Jacobian clamping evaluates the requested task direction so safe inward rows
remain available. Ordinary tasks keep the historical row-sign clamp, and all
hard position and velocity constraints remain enforced by the solver.

### COMTask

Control center of mass position. Most current examples use the support-polygon
constraint API instead of a standalone CoM task:

```python
solver.configure_com_constraint(
    support_polygon=polygon_xy,
    margin=0.05,
    frame_name="world",
)
```

### JointTask

Control individual joint position.

```python
joint_task = solver.add_joint_task("joint_bias", "panda_joint4")
joint_task.set_target_value(0.5)
```

### MultiJointTask

Control multiple joints simultaneously.

Use a `PostureTask` with selected controlled joints when you want a compact
multi-joint bias.

## Units Conventions

Use these units consistently across task targets, limits, and tolerances:

- Translation (`x, y, z`): `m`
- Rotation (`rx, ry, rz`, roll/pitch/yaw, angle errors): `rad`
- Linear velocity: `m/s`
- Angular velocity: `rad/s`
- Linear acceleration: `m/s^2`
- Angular acceleration: `rad/s^2`

For 6D torso pose vectors in `PositionIKOptions.torso_constraint`, the ordering is
`[x, y, z, rx, ry, rz]` with units `[m, m, m, rad, rad, rad]`. Pose **bounds** are
measured relative to `torso_constraint.pose_bounds_reference_pose` when set, else
relative to the torso frame at the `solve_position` seed configuration for that call.

## Task Hierarchy

Tasks are solved in priority order (`0` = highest).

```python
ee_task.priority = 0
torso_task.priority = 1
posture_task.priority = 2
```

Typical use:

- Priority `0`: end-effector tracking (`FrameTask`)
- Priority `1`: secondary balance/posture frame objective (for example torso upright)
- Priority `2`: nullspace posture bias (`PostureTask`)

## Task Solve Modes

Each task can be solved in one of two modes:

- `TaskSolveMode.SCALE` (default): classic SNS/eSNS behavior that preserves task
  direction with a scale factor in `[0, 1]`.
- `TaskSolveMode.MIN_ERROR`: clamped minimum-error behavior that computes the
  best feasible residual motion under active constraints.

Automatic fallback from `SCALE` to `MIN_ERROR` when scale collapses is
**disabled by default** for `SCALE` tasks. Enable it explicitly when needed:

```python
task = solver.add_frame_task("ee", "panda_hand", embodik.TaskType.FRAME_POSE)
task.solve_mode = embodik.TaskSolveMode.SCALE
task.allow_min_error_fallback = True
```

After a solve, inspect effective diagnostics:

```python
result = solver.solve_position_step(q, target_pose, "ee_task", step_opts)
print(result.task_modes_effective)
print(result.task_used_fallback)
```

Notes:

- User-created tasks default to `SCALE`.
- User-created tasks default to `allow_min_error_fallback = False`.
- Internal nullspace-bias posture tasks used by `solve_position()` run in
  `MIN_ERROR` mode and are placed after the optional internal torso objective.

## PostureTask Joint Selection and Weights

`PostureTask` supports explicit selection and per-joint weighting:

```python
posture = solver.add_posture_task("posture")
posture.set_controlled_joint_indices([0, 1, 2])
posture.set_controlled_joint_weights(np.array([2.0, 1.0, 0.5]))
```

This behavior is consistent with a diagonal selection/weighting matrix:

- selected joints contribute to the nullspace objective
- unselected joints are not driven by the posture objective

## API Reference

::: embodik.FrameTask
    options:
      show_root_heading: true

::: embodik.PostureTask
    options:
      show_root_heading: true

::: embodik.ManipulabilityTask
    options:
      show_root_heading: true

::: embodik.JointLimitAvoidanceTask
    options:
      show_root_heading: true

::: embodik.COMTask
    options:
      show_root_heading: true

::: embodik.JointTask
    options:
      show_root_heading: true

::: embodik.MultiJointTask
    options:
      show_root_heading: true
