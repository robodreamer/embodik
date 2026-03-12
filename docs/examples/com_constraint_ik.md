# CoM Constraint Example Overview

Overview for `examples/08_com_constraint_example.py`.

## What It Demonstrates

- `configure_com_constraint()` with a 2D support polygon
- Safety margin shrinking of the active polygon
- Optional proximity activation near polygon boundary
- Visual feedback for CoM status (inside/near/outside)

## Key API

```python
solver.configure_com_constraint(
    support_polygon=polygon_xy,
    margin=0.05,
    frame_name="world",
    com_vel_max=0.4,
    com_acc_max=0.1,
    use_acceleration_limits=True,
    proximity_fraction=0.05,
)
```

You can test adaptive relaxation behavior in the same demo:

```python
frame_task.solve_mode = embodik.TaskSolveMode.SCALE
frame_task.allow_min_error_fallback = True
posture_task.solve_mode = embodik.TaskSolveMode.MIN_ERROR
```

During runtime, inspect:

```python
result = solver.solve_velocity(q_current, apply_limits=True)
print(result.task_modes_effective[0], result.task_used_fallback[0], result.task_scales[0])
```

## Run

```bash
pixi run python examples/08_com_constraint_example.py --robot panda
```
