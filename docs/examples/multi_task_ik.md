# Multi-Task IK Example

Example demonstrating hierarchical multi-task inverse kinematics.

## Code

```python
import embodik
import numpy as np

# 1. Load robot model
model = embodik.RobotModel.from_urdf("panda.urdf")
solver = embodik.KinematicsSolver(model)

# 2. Define multiple tasks with priorities
tasks = [
    # Priority 1: End-effector pose task
    embodik.FrameTask(
        frame_id="panda_link8",
        target_pose=create_target_pose([0.5, 0.2, 0.3])
    ),
    # Priority 2: Posture task (maintain preferred configuration)
    embodik.PostureTask(
        target_q=np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0])
    ),
    # Priority 3: Center of mass task
    embodik.COMTask(
        target_com=np.array([0.0, 0.0, 0.5])
    ),
]

# 3. Solve multi-task IK
result = solver.solve_multi_task_ik(
    tasks=tasks,
    initial_q=np.zeros(model.nq)
)

# 4. Check result
if result.status == embodik.SolverStatus.SUCCESS:
    print(f"✓ Multi-task IK solved!")
    print(f"  Solution: {result.solution}")
    print(f"  Task scales: {result.task_scales}")
else:
    print(f"✗ Multi-task IK failed: {result.status}")

def create_target_pose(position):
    """Create a 4x4 transformation matrix."""
    pose = np.eye(4)
    pose[:3, 3] = position
    return pose
```

## Explanation

Multi-task IK solves tasks in priority order:

1. **Priority 1 (FrameTask)**: End-effector must reach target pose
2. **Priority 2 (PostureTask)**: Maintain preferred joint configuration in the null space of priority 1
3. **Priority 3 (COMTask)**: Control center of mass in the null space of priorities 1-2

## Task Hierarchy

Tasks are solved hierarchically:
- Higher priority tasks are satisfied exactly
- Lower priority tasks are satisfied in the null space of higher priority tasks
- If a lower priority task conflicts with a higher priority task, it may be scaled down

## Output

```
✓ Multi-task IK solved!
  Solution: [ 0.123 -0.456  0.789 -1.234  0.567 -0.890  0.345]
  Task scales: [1.0, 0.95, 0.87]
```

Task scales indicate how well each task was satisfied (1.0 = fully satisfied).

## Next Steps

- [Basic IK Example](basic_ik.md) - Simpler example
- [API Reference](../api/index.md) - Detailed API documentation
