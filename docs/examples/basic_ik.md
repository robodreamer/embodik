# Basic IK Example

Minimal interactive IK for bringing up a fixed-base robot preset.

## Overview

`examples/01_basic_ik_simple.py` is intentionally small. It shows the shortest
path for trying EmbodiK on a robot model:

- load a robot preset and default posture
- create a `KinematicsSolver`
- add one end-effector frame task and one posture/nullspace task
- drag a Viser target transform
- call `solve_position_step()` and visualize the returned configuration

Detailed tuning panels, joint sliders, diagnostics, and limit-scaling controls
are kept out of this public example. From a git clone, use the clone-only dev
surface instead:

```bash
pixi run demo-advanced-ik
```

## API Walkthrough

The public script uses the current registered-task API:

| Step | API calls | Purpose |
| --- | --- | --- |
| Resolve a robot preset | `resolve_robot_configuration("panda")` | Get the `RobotModel`, target link, and default configuration used by the script. |
| Create the solver | `KinematicsSolver(robot)`, `solver.dt` | Configure the prioritized IK solver. |
| Register the primary task | `solver.add_frame_task("ee_task", target_link)` | Track the draggable end-effector target. |
| Register posture bias | `solver.add_posture_task("posture_bias")`, `set_target_configuration(q_default)` | Keep unused freedom near the default posture. |
| Configure position updates | `PositionStepOptions()` | Set gains, one inner solve, and adaptive timestep behavior. |
| Advance IK | `solver.solve_position_step(q, target_pose, "ee_task", step_opts)` | Return a `PositionIKResult`; the next displayed configuration is `result.q_solution`. |

For the exact imports, visualization wiring, and UI loop, use the script itself:
`examples/01_basic_ik_simple.py`.

## Running

Install and copy the example bundle once using the
[Installation Guide](../installation.md#examples). Then run:

```bash
cd embodik_examples
python 01_basic_ik_simple.py
```

For repository development, use Pixi:

```bash
pixi run python examples/01_basic_ik_simple.py
```

The example defaults to the Panda preset; pass `--robot <key>` to use another
configured model.

## Adding a New Robot

To try a new fixed-base robot, add or override a preset in
`examples/robot_models/robot_presets.yaml` with:

- the URDF source (`urdf_path` or `robot_descriptions` import)
- the target end-effector link
- a default configuration

Then run:

```bash
python 01_basic_ik_simple.py --robot <your-key>
```

## Next Steps

- [Collision-Aware IK Example](collision_aware_ik.md) — add self-collision constraints
- [Teleop IK Example](teleop_ik.md) — connect controller input to the same registered-task IK call
- [Multi-Task Multi-Constraints IK Example](multi_task_ik.md) — hierarchy plus CoM constraints
