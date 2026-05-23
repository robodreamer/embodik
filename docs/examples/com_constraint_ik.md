# CoM Constraint Example Overview

Overview for `examples/04_com_constraint_example.py`.

## What It Demonstrates

- `configure_com_constraint()` with a 2D support polygon
- Safety margin shrinking of the active polygon
- Optional proximity activation near polygon boundary
- Visual feedback for CoM status (inside/near/outside)

## API Walkthrough

The example keeps the same registered-task stepping path and adds a hard CoM
support-polygon constraint:

| Step | API calls | Purpose |
| --- | --- | --- |
| Register IK tasks | `add_frame_task("ee_task", target_link)`, `add_posture_task("posture")` | Track the end-effector marker with a posture bias underneath. |
| Configure support polygon | `configure_com_constraint(...)` | Keep the 2D CoM projection inside the active polygon. |
| Tune relaxation | `frame_task.solve_mode`, `allow_min_error_fallback`, `posture_task.solve_mode` | Explore strict scaling versus minimum-error fallback near constraints. |
| Step IK | `solve_position_step(q_current, target_pose, "ee_task", step_opts)` | Apply one interactive IK update using the registered tasks and CoM constraint. |
| Inspect diagnostics | `result.task_modes_effective`, `result.task_used_fallback`, `result.task_scales` | Display effective mode, fallback use, and scale while the demo runs. |

Use `examples/04_com_constraint_example.py` for the exact slider values,
visualization markers, and support-polygon setup.

## Run

Install and copy the example bundle once using the
[Installation Guide](../installation.md#examples). Then run:

```bash
cd embodik_examples
python 04_com_constraint_example.py
```

For repository development, use Pixi:

```bash
pixi run python examples/04_com_constraint_example.py
```

The example defaults to the Panda preset; pass `--robot <key>` to use another
configured model.
