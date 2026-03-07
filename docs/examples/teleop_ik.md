# Teleop IK Example Overview

Overview for `examples/03_teleop_ik.py`.

## What It Demonstrates

- Real-time teleoperation with a Seer wireless controller
- Frame-task target updates from controller pose deltas
- Optional collision-aware IK while teleoperating
- GUI fallback mode when no controller is connected

## Key Controls

- Side button hold: stream on/off
- Trigger hold: grasping on/off
- Button A: reset robot pose
- Button B: toggle data collection

## Run

```bash
pixi run -e teleop demo-teleop
# or
pixi run -e teleop python examples/03_teleop_ik.py --robot panda
```

## Notes

- Requires `xvisio` and host runtime support for Seer controller.
- Use `--no-collision` to disable collision constraints for debugging.
