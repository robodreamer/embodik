# Teleop IK Example Overview

Overview for `examples/03_teleop_ik.py`.

## What It Demonstrates

- Real-time teleoperation with a Seer wireless controller
- Frame-task target updates from controller pose deltas
- Optional collision-aware IK while teleoperating
- Optional torso-upright secondary objective and torso pose bounds
- Nullspace bias as tertiary objective with optional per-joint weights
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

# optional torso controls
pixi run -e teleop python examples/03_teleop_ik.py --robot panda \
  --enable-torso-pose-constraints
```

## Notes

- Requires `xvisio` and host runtime support for Seer controller.
- Use `--no-collision` to disable collision constraints for debugging.
- Use `--nullspace-joint-weights` for explicit per-joint nullspace weighting.
- By default torso pose bounds use symmetric half-range; set both
  `--torso-pose-lower-bounds` and `--torso-pose-upper-bounds` to use asymmetric limits.
- Torso pose bound units are `[x,y,z,rx,ry,rz] = [m,m,m,rad,rad,rad]`.
- Torso pose **box** limits are anchored to the torso world pose when constraints
  become active (startup, after reset, or when re-enabling the GUI checkbox), via
  `pose_bounds_reference_pose`, so they do not recentre on every IK tick.
- For floating-base torso-oriented validation, run `examples/10_floating_base_torso_hierarchy.py`
  (default: **Viser** UI — drag the EE target; sliders adjust torso pose box half-ranges, vel/acc
  limits, torso bound softening, and IK gains; **Re-anchor** updates the fixed
  `pose_bounds_reference_pose`).
- Example 10 includes profile presets (`Responsive`, `Stable`, `StrictBounds`) for quick tuning.
  Start with `Stable` for bounded-mode teleop, then switch to `Responsive` for faster target motion.
- In bounded mode, use the `Constraint pressure` indicator (`Low/Medium/High`) to diagnose when hard
  torso bounds and rapid target changes are likely to trigger infeasible/hold behavior.
- Use `scripts/benchmark_example10_torso_modes.py` for scripted analysis; it reports
  per-status timing/error buckets, iteration distributions, and jump-vs-no-jump / max-iteration
  sweep matrices to isolate responsiveness bottlenecks.
