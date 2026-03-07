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

## Run

```bash
pixi run python examples/08_com_constraint_example.py --robot panda
```
