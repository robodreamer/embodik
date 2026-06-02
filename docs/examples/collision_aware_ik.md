# Collision-Aware IK Example

Interactive collision-aware IK behavior demo for development and testing.

## Overview

`examples/02_collision_aware_IK.py` extends the minimal `01` prioritized IK setup
with self-collision constraints, collision-debug visualization, tuning modes, and
an optional GPU benchmark panel.

It is intentionally more detailed than `01_basic_ik_simple.py`: use it when you
want to inspect collision behavior, timing, task modes, and solver tuning.

## API Walkthrough

The example uses the same `solve_position_step()` path as the basic example,
then adds collision constraints and diagnostics:

| Step | API calls | Purpose |
| --- | --- | --- |
| Register tasks | `add_frame_task("ee_task", target_link)`, `add_posture_task("posture_task")` | Track the end effector while keeping a configurable nullspace bias. |
| Configure collision avoidance | `configure_collision_constraint(...)` | Add self-collision velocity-damper constraints to each step. |
| Tune task behavior | `TaskSolveMode`, `allow_min_error_fallback`, posture `weight` | Let the UI switch between strict scaling, fallback, and nullspace settings. |
| Step the target | `solve_position_step(q, target_pose, "ee_task", step_opts)` | Produce the next safe configuration and per-task diagnostics. |
| Visualize/debug | `compute_collision_distances()`, result diagnostics | Show closest pairs, timing, effective task modes, and fallback state. |

Use `examples/02_collision_aware_IK.py` as the runnable source for the full UI
and option wiring.

## Collision Configuration

`configure_collision_constraint()` enables self-collision avoidance. The solver
computes active close pairs and adds velocity-damper inequality constraints to
keep them above `min_distance`.

For the **why** behind fast WBC collision (bounds, pairwise coherence, Speed vs Precise timing),
see [Collision Constraints](../collision_constraints.md#the-wbc-collision-bottleneck).

Important options:

- `min_distance`: safety margin in metres
- `include_pairs`: optional allow-list; empty means all configured collision pairs
- `exclude_pairs`: adjacent or known-safe pairs to ignore
- `max_constraints`: optional top-k closest pairs to enforce simultaneously

Example 02 supports auto-generated exclusion lists from robot presets so public
robot demos do not require hand-writing every adjacent pair.

### Non-worsening recovery floor

Some robots have links that rest closer than the configured `min_distance` (for
example, an upper arm near the torso). Without special handling, the collision
recovery logic can try to push those pairs out to the global clearance and freeze
the solve.

When enabled, the **non-worsening floor** keeps each pair at its achievable
distance instead of forcing recovery to `min_distance`:

```python
solver.set_non_worsening_collision_floor_enabled(True)
```

This is **off by default** on a new `KinematicsSolver`. The bimanual teleop
example opts in because whole-body models commonly include structurally-close
pairs.

`set_collision_structural_floor(metres)` sets the minimum clearance maintained for
those pairs when the floor is active (default `0.005` m). Use it when tuning how
aggressively the solver prevents penetration on resting contacts.

### Collision tuning mode

New solvers default to `CollisionTuningMode.BALANCED`, which enables sphere
broadphase and a conservative pair cache suitable for interactive teleop. Use
`set_collision_tuning_mode()` or the shared helper
`apply_collision_tuning_mode(solver, "balanced")` to switch between `speed`,
`balanced`, and `precise`.

| Mode | When to use | Performance (Panda CI medians) |
| --- | --- | --- |
| `speed` | Viser teleop, tight control loops | `< 3 ms` per `solve_position_step` |
| `balanced` | Default for examples and replay harness | `< 4 ms` |
| `precise` | Distance fidelity checks, regression debug | Slowest; full exact pair checks |

All three modes share the same **post-step penetration guard** — Speed is faster because
pair cache, broadphase, and refinement budgeting skip work in clear space, not because
safety checks are removed.

See [Collision Constraints & Tuning](../collision_constraints.md) for measured timings, batch
parallelization, and collision tuning. See [Solver Robustness & Recovery](../solver_robustness.md) for
adaptive dt, elastic band, and runtime policy.

### Position-step MIN_ERROR fallback

For teleop loops using `solve_position_step()`, set
`PositionStepOptions.primary_allow_min_error_fallback = True` when the primary
EE tasks use `SCALE` or `SCALE_ELASTIC`. If a collision row collapses the primary
task scale, the solver retries once with a MIN_ERROR primary solve while keeping
hard constraints active. This unfreezes motion near the body without permanently
changing registered task modes.

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

Install and copy the example bundle once using the
[Installation Guide](../installation.md#examples). Then run:

```bash
cd embodik_examples
python 02_collision_aware_IK.py
```

For repository development, use Pixi:

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
- [Teleop IK Example](teleop_ik.md) — feed controller targets into the same registered-task IK path
- [CoM Constraint Example](com_constraint_ik.md) — support-polygon inequality constraints
