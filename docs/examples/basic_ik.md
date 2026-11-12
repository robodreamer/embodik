# Basic IK Example

Minimal interactive IK for bringing up a fixed-base robot preset.

## Overview

`examples/01_basic_ik_simple.py` is intentionally small. It shows the shortest
path for trying EmbodiK on a robot model:

- load a robot preset and default posture
- create velocity and acceleration solver runtimes when the model supports both
- add one end-effector frame task and one posture/nullspace task
- drag a Viser target transform
- switch the **Solver Level** control between velocity and acceleration
- call the selected solver and visualize only an accepted configuration

Detailed tuning panels, joint sliders, diagnostics, and limit-scaling controls
are kept out of this minimal example. Use the other public examples for richer
interactive workflows.

## Rainbow RB-Y1 models

The `rby1` alias and `rby1a` preset select A v1.2; `rby1m` selects M v1.3. They use the official
[Rainbow SDK v0.10.1](https://github.com/RainbowRobotics/rby1-sdk/releases/tag/v0.10.1)
assets. Both expose 20 torso/arm DOFs and start in a bent-elbow ready pose.

```bash
pixi run python examples/01_basic_ik_simple.py --robot rby1
pixi run python examples/01_basic_ik_simple.py --robot rby1m
```

Assets are cached on first use under `~/.cache/embodik/rby1-sdk/` (Git and
network access required; the meshes occupy several hundred MB). For offline use,
set `EMBODIK_RBY1_ROOT` to a local SDK checkout. `EMBODIK_RBY1_CACHE` overrides
the cache location. The resolved release is pinned for reproducibility; it does
not silently follow future SDK changes.

The adapter generates an IK-specific URDF without editing the SDK files:

- Wheels, head and fingers are fixed; torso and both seven-joint arms remain active.
- SDK capsules become equivalent cylinders plus endpoint spheres, retaining their
  collision volume with standard URDF primitives. Visual meshes and inertias remain.
- Missing meshes cause an error. The viewer uses the same generated URDF as IK.

This supports fixed-base torso/arm IK. SDK mobile-base dynamics, collision-type
masks, contact control and hardware execution are outside this adapter's scope.
Existing experimental acceleration/GPU limitations still apply.

For direct loading:

```python
import embodik as eik
from embodik.models.rby1 import ready_configuration, resolve_rby1_urdf

robot = eik.RobotModel(str(resolve_rby1_urdf("a")), floating_base=False)
q = ready_configuration(robot)
robot.update_configuration(q)
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
| Advance acceleration IK | `AccelerationSolver.solve(q, dq, dt, options)` | Return one explicit acceleration, next velocity, and configuration step from the compatible state box. |

Velocity is the default. Acceleration is offered only after the fixed-base
capability and a stationary smoke solve succeed for the selected model. The
example resets its caller-owned `dq` when the solver level changes, when the
robot resets, or when an acceleration result is rejected.

See the [Acceleration Solver guide](../acceleration_solver.md) for the supported
constraint matrix and non-real-time scope.

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
configured model. Visualization defaults to ViserUrdf. The optional
`--visualizer pinocchio` backend remains available for comparison.

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
