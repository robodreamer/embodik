# CoM And Centroidal Support Example

Overview for `examples/04_com_constraint_example.py`.

## What It Demonstrates

- `configure_com_constraint()` with a 2D support polygon
- opt-in horizontal `CentroidalMomentumTask` damping
- commanded-velocity `configure_capture_point_constraint()` enforcement
- explicit-state `configure_velocity_zmp_constraint()` enforcement through
  `PositionStepOptions.current_joint_velocity`
- Safety margin shrinking of the active polygon
- Optional proximity activation near polygon boundary
- Visual feedback for CoM, capture point, and finite-difference ZMP

## API Walkthrough

The example keeps the same registered-task IK path and adds a hard CoM
support-polygon constraint:

| Step | API calls | Purpose |
| --- | --- | --- |
| Register IK tasks | `add_frame_task(...)`, `add_posture_task(...)`, `add_centroidal_momentum_task(...)` | Track the end-effector, bias posture, and optionally damp horizontal momentum. |
| Configure support polygon | `configure_com_constraint(...)` | Keep the 2D CoM projection inside the active polygon. |
| Configure dynamic support | `configure_capture_point_constraint(...)`, `configure_velocity_zmp_constraint(...)` | Keep commanded capture point and finite-difference ZMP inside the same active polygon. |
| Tune relaxation | `frame_task.solve_mode`, `allow_min_error_fallback`, `posture_task.solve_mode` | Explore strict scaling versus minimum-error fallback near constraints. |
| Supply state | `step_opts.current_joint_velocity = dq_current` | Provide the measured or caller-integrated velocity required by physical velocity ZMP. |
| Solve update | `solve_position_step(q_current, target_pose, "ee_task", step_opts)` | Apply one IK update with all enabled centroidal constraints. |
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
configured model. It uses ViserUrdf by default; use `--visualizer pinocchio`
to select the alternate viewer.

The capture-point and ZMP disks show accepted-command diagnostics in the
support frame. This remains a fixed-base velocity IK example; successful ZMP
rows do not certify floating-base contact forces or friction feasibility.
