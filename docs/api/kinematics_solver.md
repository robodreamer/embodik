# KinematicsSolver

`KinematicsSolver` provides high-level inverse kinematics entry points:

- `solve_velocity()` for one-step velocity IK over registered tasks.
- `solve_position()` for iterative position IK with an internal objective stack.
- `solve_position_step()` for marker/teleop loops using registered tasks.

## Position IK Objective Order

`solve_position()` now supports a three-level stack:

1. Primary end-effector frame objective (priority `0`)
2. Optional torso orientation/pose objective (priority `1`)
3. Optional nullspace posture bias (priority `2` when torso is enabled, else `1`)

## PositionIKOptions (torso + nullspace)

```python
import numpy as np
import embodik as eik

opts = eik.PositionIKOptions()
opts.max_iterations = 20
opts.position_gain = 40.0
opts.orientation_gain = 40.0

# Tertiary nullspace bias
opts.nullspace_bias = q_bias
opts.nullspace_gain = 0.01
opts.nullspace_active_joints = [0, 1, 2, 3, 4, 5, 6]
opts.nullspace_joint_weights = np.ones(len(opts.nullspace_active_joints))

# Secondary torso orientation objective
opts.torso_constraint.enabled = True
opts.torso_constraint.frame_name = "base_link"
opts.torso_constraint.orientation_mask = np.array([1.0, 1.0, 0.0])  # roll/pitch only
opts.torso_constraint.orientation_gain = 0.05

# Optional torso pose box bounds (6D: x,y,z,rx,ry,rz)
# Units: x/y/z in meters, rx/ry/rz in radians.
# Bounds are relative to a fixed world-frame torso reference. Inside solve_position,
# that reference does not move across inner iterations. If you call solve_position
# every control tick with a new seed_q, set pose_bounds_reference_pose once (e.g.
# torso.homogeneous() at session start) so the box does not recentre each tick.
half_range = np.array([0.08, 0.08, 0.08, 0.20, 0.20, 0.20])
torso0 = robot.get_frame_pose("base_link")
opts.torso_constraint.pose_bounds_reference_pose = np.asarray(
    torso0.homogeneous(), dtype=float
)
opts.torso_constraint.pose_lower_bounds = -half_range
opts.torso_constraint.pose_upper_bounds = half_range
opts.torso_constraint.pose_axis_mask = np.ones(6)
# Units: [m/s, m/s, m/s, rad/s, rad/s, rad/s]
opts.torso_constraint.velocity_limits = np.full(6, 0.5)
# Units: [m/s^2, m/s^2, m/s^2, rad/s^2, rad/s^2, rad/s^2]
opts.torso_constraint.acceleration_limits = np.full(6, 1.0)
```

Validation rules:

- `nullspace_bias` must match `robot.nq`.
- `nullspace_active_joints` indices must be unique and in `[0, robot.nv)`.
- `nullspace_joint_weights` length must match `robot.nv` (all joints) or
  `len(nullspace_active_joints)` (selected joints).
- Torso pose bounds require both lower and upper vectors, each length `6`.
- Torso 6D ordering is `[x, y, z, rx, ry, rz]` with units
  `[m, m, m, rad, rad, rad]`.
- Optional `torso_constraint.pose_bounds_reference_pose` (4x4 homogeneous): when
  set, pose bounds are measured vs this fixed transform; when unset, the reference
  is the torso frame at `seed_q` for that `solve_position` call only.

## solve_position Usage

```python
target = np.eye(4)
target[:3, 3] = [0.5, 0.2, 0.3]
result = solver.solve_position(seed_q, target, "end_effector", opts)
```

## API Reference

::: embodik.KinematicsSolver
    options:
      show_root_heading: true
      show_root_toc_entry: true
