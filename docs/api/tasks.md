# Tasks

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

::: embodik.COMTask
    options:
      show_root_heading: true

::: embodik.JointTask
    options:
      show_root_heading: true

::: embodik.MultiJointTask
    options:
      show_root_heading: true
