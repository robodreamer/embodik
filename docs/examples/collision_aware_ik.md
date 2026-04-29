# Collision-Aware IK Example

Interactive collision-aware IK behavior demo for development and testing.

## Overview

`examples/02_collision_aware_IK.py` extends the minimal `01` stepping IK pattern
with self-collision constraints, collision-debug visualization, tuning modes, and
an optional GPU benchmark panel.

It is intentionally more detailed than `01_basic_ik_simple.py`: use it when you
want to inspect collision behavior, timing, task modes, and solver tuning.

## Core Pattern

The example uses `solve_position_step()` with a registered task stack:

```python
solver = embodik.KinematicsSolver(robot)
solver.dt = 0.01
solver.set_damping(0.1)
solver.set_tolerance(0.1)

frame_task = solver.add_frame_task("ee_task", target_link)
frame_task.priority = 0
frame_task.weight = 1.0
frame_task.solve_mode = embodik.TaskSolveMode.SCALE_ELASTIC
frame_task.allow_min_error_fallback = False

posture_task = solver.add_posture_task("posture_task")
posture_task.priority = 1
posture_task.weight = 1e-2
posture_task.solve_mode = embodik.TaskSolveMode.MIN_ERROR
posture_task.set_target_configuration(q_default)

solver.configure_collision_constraint(
    min_distance=0.05,
    include_pairs=[],
    exclude_pairs=collision_exclusions,
)

opts = embodik.PositionStepOptions()
opts.position_gain = 10.0
opts.orientation_gain = 10.0
opts.max_steps = 1
opts.adaptive_dt = True
opts.adaptive_dt_max_scale = 10.0
opts.adaptive_dt_reference_distance = 0.02

result = solver.solve_position_step(q_current, target_pose, "ee_task", opts)
q_current = result.q_solution
```

## Collision Configuration

`configure_collision_constraint()` enables self-collision avoidance. The solver
computes active close pairs and adds velocity-damper inequality constraints to
keep them above `min_distance`.

Important options:

- `min_distance`: safety margin in metres
- `include_pairs`: optional allow-list; empty means all configured collision pairs
- `exclude_pairs`: adjacent or known-safe pairs to ignore
- `max_constraints`: optional top-k closest pairs to enforce simultaneously

Example 02 supports auto-generated exclusion lists from robot presets so public
robot demos do not require hand-writing every adjacent pair.

## Collision Debugging

The interactive UI can show:

- closest active pair
- signed distance
- world-space closest points
- a segment connecting those points
- collision timing and broadphase counters

Console collision-pair logs are throttled to avoid noisy output while the visual
debug markers continue updating live.

## Running

```bash
pixi run python examples/02_collision_aware_IK.py
```

The example defaults to the Panda preset; pass `--robot <key>` to use another
configured model.

Optional GPU benchmark panel:

```bash
pixi run -e cuda demo-ik-gpu
# or
pixi run -e cuda python examples/02_collision_aware_IK.py --gpu
```

## Next Steps

- [Basic IK Example](basic_ik.md) — minimal bring-up without collision UI
- [Teleop IK Example](teleop_ik.md) — feed controller targets into the same stepping IK path
- [CoM Constraint Example](com_constraint_ik.md) — support-polygon inequality constraints
