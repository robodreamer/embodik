# Multi-Task Multi-Constraints IK Example

Example demonstrating hierarchical multi-task inverse kinematics with multiple constraints: frame task, posture task, and CoM support-polygon constraint.

## Code

```python
import embodik
import numpy as np
from embodik.utils import compute_pose_error, limit_task_velocity
from embodik import Rt

# 1. Load robot (run from examples/ directory)
from utils.robot_models import resolve_robot_configuration
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

# 3. Configure CoM support-polygon constraint (keeps CoM inside polygon)
support_polygon = np.array([
    [-0.08, -0.10],
    [ 0.18, -0.10],
    [ 0.18,  0.10],
    [-0.08,  0.10],
])
solver.configure_com_constraint(
    support_polygon=support_polygon,
    margin=0.05,  # 5% inward margin (fraction of char_size)
    frame_name="world",
    com_vel_max=0.4,
    com_acc_max=0.1,
)

# 4. Velocity IK loop (same lower-level pattern used by the CoM example)
q_current = q_default.copy()
target_pose = Rt(R=np.eye(3), t=np.array([0.5, 0.2, 0.3]))
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

print(f"✓ Multi-task IK converged: q = {q_current}")
```

## Explanation

Multi-task multi-constraints IK uses a velocity-based loop:

1. **Frame task** (priority 0): Drives end-effector toward target pose via `set_target_velocity()`
2. **Posture task** (priority 1): Maintains preferred joint configuration in null space via `set_target_configuration()`
3. **CoM constraint** (`configure_com_constraint`): Keeps the 2D CoM projection inside the support polygon — a hard inequality constraint, not a task

For a full interactive demo with CoM visualization, see [CoM Constraint Example Overview](com_constraint_ik.md).

## Task Hierarchy

Tasks are solved hierarchically:

- Higher priority tasks are satisfied first
- Lower priority tasks are satisfied in the null space of higher priority tasks
- `configure_com_constraint` adds inequality constraints that the solver enforces regardless of task priorities

## Next Steps

- [Basic IK Example](basic_ik.md) — Simpler example
- [Collision-Aware IK Example](collision_aware_ik.md) — Self-collision avoidance
- [API Reference](../api/index.md) — Detailed API documentation
