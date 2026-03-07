# Collision-Aware IK Example

Example demonstrating self-collision avoidance during inverse kinematics using velocity-damper constraints.

## Overview

EmbodiK supports self-collision avoidance via `configure_collision_constraint()`. The solver enforces a minimum distance between collision pairs using velocity-damper inequality constraints. When links approach each other, the solver limits joint velocities to prevent penetration and can apply recovery when already inside the safety margin.

## Code

```python
import embodik
import numpy as np
from embodik.utils import compute_pose_error, limit_task_velocity
from utils.robot_models import resolve_robot_configuration

# 1. Load robot
config = resolve_robot_configuration("panda")
model = config["robot"]
target_link = config["target_link"]
q_default = config["default_configuration"]

# 2. Create solver and add tasks
solver = embodik.KinematicsSolver(model)
solver.dt = 0.01
solver.set_damping(0.1)

frame_task = solver.add_frame_task("ee_task", target_link)
frame_task.priority = 0
frame_task.weight = 1.0

posture_task = solver.add_posture_task("posture")
posture_task.priority = 1
posture_task.weight = 0.01
posture_task.set_target_configuration(q_default)

# 3. Configure collision constraint (self-collision avoidance)
# Exclude adjacent links (e.g. link_i and link_{i+1}) — they are always in contact
exclude_pairs = [
    ("panda_link1", "panda_link2"),
    ("panda_link2", "panda_link3"),
    # ... add more adjacent pairs or use auto-exclusion logic
]
solver.configure_collision_constraint(
    min_distance=0.05,   # 5 cm safety margin
    include_pairs=[],     # empty = use all pairs from model
    exclude_pairs=exclude_pairs,
)

# 4. Velocity IK loop (same as basic/multi-task multi-constraints examples)
q_current = q_default.copy()
target_pose = embodik.Rt(R=np.eye(3), t=np.array([0.5, 0.2, 0.3]))
pos_gain, rot_gain = 60.0, 60.0

for _ in range(500):
    model.update_configuration(q_current)
    current_ee = model.get_frame_pose(target_link)
    pose_error = compute_pose_error(current_ee, target_pose)

    if np.linalg.norm(pose_error) < 5e-4:
        break

    target_velocity = np.concatenate([
        pos_gain * pose_error[:3],
        rot_gain * pose_error[3:],
    ])
    target_velocity = limit_task_velocity(target_velocity, 0.5, 0.5)
    frame_task.set_target_velocity(target_velocity)

    result = solver.solve_velocity(q_current, apply_limits=True)
    if result.status != embodik.SolverStatus.SUCCESS:
        break

    dq = result.joint_velocities * solver.dt
    q_lower, q_upper = model.get_joint_limits()
    q_current = np.clip(q_current + dq, q_lower, q_upper)

# Optional: inspect closest collision pair after solve
debug = solver.get_last_collision_debug()
if debug is not None:
    print(f"Closest pair: {debug.link_a} — {debug.link_b}, distance={debug.distance:.4f}")
```

## Explanation

1. **`configure_collision_constraint()`** — Enables self-collision avoidance. The solver computes the closest collision pair(s) each iteration and adds inequality constraints to keep them above `min_distance`.

2. **`exclude_pairs`** — Pairs that are always in contact (e.g. adjacent links) should be excluded. Example 02 uses auto-exclusion logic based on link indices and end-effector tokens.

3. **`include_pairs`** — If non-empty, only these pairs are checked; otherwise all model collision pairs (minus exclusions) are used.

4. **`get_last_collision_debug()`** — Returns debug info for the closest active pair after each solve (link names, distance, Jacobian, etc.).

## Advanced: Top-K Constraints

For robots with multiple tight-clearance regions (e.g. base/leg and arm/torso), use `max_constraints` to protect several pairs simultaneously:

```python
solver.configure_collision_constraint(
    min_distance=0.05,
    exclude_pairs=exclude_pairs,
    max_constraints=3,  # up to 3 closest pairs enforced at once
)
debug_list = solver.get_last_collision_debug_list()  # list of CollisionDebugInfo
```

## Running the Interactive Example

For a full interactive demo with Viser visualization, collision toggle, and optional GPU acceleration:

```bash
pixi run python examples/02_collision_aware_IK.py --robot panda
# With GPU (CusADi): --gpu
```

## Next Steps

- [Basic IK Example](basic_ik.md) — Simpler example without collision
- [Multi-Task Multi-Constraints IK Example](multi_task_ik.md) — Frame + posture + CoM constraint
- [API Reference](../api/index.md) — Detailed API documentation
