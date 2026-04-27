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

## Code Pattern

The public script follows this structure:

```python
config = resolve_robot_configuration("panda")
robot = config["robot"]
target_link = config["target_link"]
q_default = config["default_configuration"]

solver = embodik.KinematicsSolver(robot)
solver.dt = 0.01
solver.set_damping(0.1)
solver.set_tolerance(0.1)

ee_task = solver.add_frame_task("ee_task", target_link)
ee_task.priority = 0
ee_task.weight = 1.0
ee_task.solve_mode = embodik.TaskSolveMode.SCALE_ELASTIC
ee_task.allow_min_error_fallback = False

posture_task = solver.add_posture_task("posture_bias")
posture_task.priority = 1
posture_task.weight = 1e-2
posture_task.solve_mode = embodik.TaskSolveMode.MIN_ERROR
posture_task.set_target_configuration(q_default)

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

## Running

For pip-installed users:

```bash
pip install "embodik[examples]"
embodik-examples --copy
cd embodik_examples
python 01_basic_ik_simple.py --robot panda
```

For repository development:

```bash
pixi run python examples/01_basic_ik_simple.py --robot panda
```

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
- [Teleop IK Example](teleop_ik.md) — connect controller input to the same stepping IK call
- [Multi-Task Multi-Constraints IK Example](multi_task_ik.md) — hierarchy plus CoM constraints
