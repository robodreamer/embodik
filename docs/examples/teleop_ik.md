# Teleop IK Example

Minimal teleop input to EmbodiK IK.

## What It Demonstrates

`examples/03_teleop_ik.py` is a public adapter example. It keeps the IK setup
hidden in `examples/example_helpers/teleop_ik_backend.py` so the script can focus
on the important teleop wiring:

1. Read a relative controller pose.
2. Convert the controller delta into an end-effector target pose.
3. Call `backend.solve_step(goal_pose)`.
4. Visualize or send the returned joint positions.

If no Seer controller is connected, the same IK path runs from the draggable
`/ik_target` transform in the browser.

## Button Mapping

- Trigger hold: stream controller motion into IK.
- Side button hold: update gripper status.
- Button A: reset robot and controller reference.
- Button B: toggle data-collection status.

## Core Pattern

The example intentionally leaves detailed IK tuning outside the public script:

```python
backend = TeleopIKBackend(cfg, enable_collision=True)
controller = SeerController("/dev/ttyUSB0")
controller.connect()

arm_stream_start_pose = backend.get_pose()

while True:
    controller.process_buttons()

    if controller.streaming:
        delta_pos, delta_wxyz = controller.relative_pose()
        goal_pose = apply_controller_delta(
            arm_stream_start_pose,
            delta_pos,
            delta_wxyz,
            scale=1.5,
        )
        result = backend.solve_step(goal_pose)
        q_command = result.joints
```

The backend uses the same stepping IK pattern as the other examples:
`solve_position_step()` with a frame task, posture bias, adaptive dt, and optional
self-collision constraints.

## Running

```bash
pixi run -e teleop demo-teleop
# or
pixi run -e teleop python examples/03_teleop_ik.py
```

The example defaults to the Panda preset; pass `--robot <key>` to use another
configured model.

Useful flags:

```bash
pixi run -e teleop python examples/03_teleop_ik.py --controller-port /dev/ttyUSB1
pixi run -e teleop python examples/03_teleop_ik.py --no-collision
```

## Advanced Teleop Surface

The public teleop example avoids detailed IK controls. From a git clone, use the
clone-only advanced launcher when you need solver tuning/debug panels:

```bash
pixi run python dev_examples/advanced_interactive_ik.py teleop
```

The advanced launcher is not part of the pip-facing `embodik-examples --copy`
workflow.

## Notes

- Requires `xvisio` and host runtime support for Seer controllers.
- Install the teleop Pixi environment once with `pixi install -e teleop`.
- With no controller connected, drag `/ik_target` in the browser to exercise the
  same `backend.solve_step(goal_pose)` path.
- For floating-base torso-oriented validation, use
  `examples/10_floating_base_torso_hierarchy.py`.
