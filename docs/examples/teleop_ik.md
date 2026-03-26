# Teleop IK Example Overview

Overview for `examples/03_teleop_ik.py`.

## What It Demonstrates

- Real-time teleoperation with a Seer wireless controller
- Frame-task target updates from controller pose deltas
- Optional collision-aware IK while teleoperating
- Fixed-base arm IK via `solve_position_step` (same task stack as `02_collision_aware_IK.py`)
- Nullspace bias toward the default pose with optional per-joint weights
- GUI fallback mode when no controller is connected

Torso / floating-base pose bounds are **not** part of this script; use
`examples/10_floating_base_torso_hierarchy.py` for that.

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
- With self-collision enabled, turn on **Show Collision Debug** in the UI to mirror
  `examples/02_collision_aware_IK.py`: closest pair, distance, and segment between
  `point_a` / `point_b` (updates after each IK step). Console `[embodiK] Collision pair:` lines
  are suppressed unless you pass **`--verbose`** / **`-v`**.
- Use `--nullspace-joint-weights` for explicit per-joint nullspace weighting.
- Install the library into the `teleop` Pixi env once: `pixi run -e teleop install`
  (separate solve-group from `default`).
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

## Example 10 Torso Pose Motion Bounds (Detailed)

`examples/10_floating_base_torso_hierarchy.py` exposes torso bounds as a 6D box in
`[x, y, z, rx, ry, rz]` relative motion space (`m, m, m, rad, rad, rad`).

- **Anchor model**: Bounds are enforced against a fixed reference pose
  (`pose_bounds_reference_pose`). The reference is set when bounds are enabled,
  after reset, or when `Re-anchor bounds to current torso` is clicked.
- **Box limits**: The `±x/±y/±z/±rx/±ry/±rz` sliders define half-ranges around
  the anchor. The solver enforces `pose_lower_bounds`/`pose_upper_bounds` with
  per-axis velocity and acceleration limits.
- **Translation/rotation toggles**: Unchecking `Enable translation bounds` or
  `Enable rotation bounds` does not remove the corresponding rows; it applies an
  epsilon half-range lock (`1e-4 m` translational, `1e-3 rad` rotational) so
  those axes are effectively fixed.
- **Secondary torso task vs hard bounds**: `Enable torso secondary orientation task`
  controls a separate orientation objective (independent from the box rows). This
  lets you test box constraints alone, orientation shaping alone, or both together.
- **Constraint-faithful default**: `Optimize full lock with base joint lock
  (fixed-base emulation)` is opt-in and defaults off. With it off, full 6D locks
  are tested through torso constraint rows; with it on, full lock can map to base
  joint exclusion (`excluded_joint_indices`) for fixed-base-like behavior.
- **Velocity-box headroom policy and numerical robustness**:
  - Preferred API: `torso_constraint.velocity_box_headroom` with:
    - `enabled`
    - `fraction` (minimum headroom as a fraction of per-axis `velocity_limits`)
    - `activation_margin` (minimum slack before headroom is injected)
  - This policy now uses the same shared velocity-box helper path as other
    limit constraints, rather than a torso-specific post-processing block.
  - Legacy aliases (`pose_bound_softening_enabled`,
    `pose_bound_softening_fraction`) are still accepted for backward
    compatibility and map to the shared headroom policy.
  - Solver-side slack dead-zones (`1e-4 m`, `1e-3 rad`) suppress boundary chatter.
- **Operational diagnostics**:
  - `Torso box min slack` reports signed margin to the nearest box face.
  - `Constraint pressure` summarizes runtime stress (`Low/Medium/High`) from slack
    and infeasible streak trends.
  - Near-target infeasible streaks are damped so close-enough constrained states
    do not accumulate misleading stall counters.

### Tuning tips

- Start from `Stable` when validating new torso ranges, then move to `Responsive`
  for faster teleop motions.
- If target jumps are large, keep torso bounds active but reduce jump severity
  (`Limit target step per frame`) before increasing gains aggressively.
- If full 6D lock appears too stiff for your use case, compare constraint-faithful
  mode against fixed-base emulation and benchmark both with
  `scripts/benchmark_example10_torso_modes.py`.
