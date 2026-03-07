# Basic IK Example

Simple example demonstrating velocity-based inverse kinematics with EmbodiK.

## Overview

This example follows the same control pattern used in `examples/01_basic_ik_simple.py`:

- Add a high-priority frame task for end-effector motion
- Add a low-priority posture task as nullspace regularization
- Run a velocity IK loop with `solve_velocity()` and integrate joint updates

## Code

```python
import embodik
import numpy as np
from embodik import Rt
from embodik.utils import compute_pose_error, limit_task_velocity
from utils.robot_models import resolve_robot_configuration

# 1. Load robot (run from examples/ directory)
config = resolve_robot_configuration("panda")
model = config["robot"]
target_link = config["target_link"]
q_default = config["default_configuration"]

# 2. Create solver and tasks
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

# 3. Set goal
q_current = q_default.copy()
target_pose = Rt(R=np.eye(3), t=np.array([0.5, 0.2, 0.3]))
pos_gain, rot_gain = 60.0, 60.0

# 4. Velocity IK loop
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

print(f"✓ Basic IK converged: q = {q_current}")
```

## Explanation

1. **Frame task** (priority 0): Drives the end-effector toward the pose target.
2. **Posture task** (priority 1): Keeps motion near a preferred joint configuration in nullspace.
3. **Velocity solve**: `solve_velocity()` computes joint velocities that satisfy tasks and active constraints.
4. **Integration**: Update with `q += dt * dq`, then clamp to joint limits.

## Next Steps

- [Multi-Task Multi-Constraints IK Example](multi_task_ik.md) — Frame + posture + CoM constraint
- [Collision-Aware IK Example](collision_aware_ik.md) — Self-collision avoidance
- [API Reference](../api/index.md) — Detailed API documentation
