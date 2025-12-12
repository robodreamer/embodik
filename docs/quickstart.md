# Quickstart Guide

Get started with EmbodiK in 5 minutes.

## Basic Usage

### 1. Create a Robot Model

```python
import embodik
import numpy as np

# Load robot model from URDF
model = embodik.RobotModel.from_urdf("path/to/robot.urdf")

# Or create from existing Pinocchio model
# model = embodik.RobotModel(pinocchio_model)
```

### 2. Create a Kinematics Solver

```python
solver = embodik.KinematicsSolver(model)
```

### 3. Define Tasks

```python
# Frame task: control end-effector pose
frame_task = embodik.FrameTask(
    frame_id="end_effector",
    target_pose=np.eye(4)  # 4x4 transformation matrix
)

# Posture task: maintain joint configuration
posture_task = embodik.PostureTask(
    target_q=np.zeros(model.nq)  # Desired joint angles
)
```

### 4. Solve Inverse Kinematics

```python
# Position IK: solve for joint angles to achieve target pose
options = embodik.PositionIKOptions(
    max_iterations=100,
    tolerance=1e-6
)

result = solver.solve_position_ik(
    target_pose=frame_task.target_pose,
    initial_q=np.zeros(model.nq),
    options=options
)

if result.status == embodik.SolverStatus.SUCCESS:
    print(f"Solution found: {result.solution}")
    print(f"Converged in {result.iterations} iterations")
else:
    print(f"Solver failed: {result.status}")
```

## Multi-Task IK

EmbodiK supports hierarchical multi-task inverse kinematics:

```python
# Create multiple tasks with priorities
tasks = [
    embodik.FrameTask("end_effector", target_pose_1),  # Priority 1
    embodik.PostureTask(target_q),                      # Priority 2
    embodik.COMTask(target_com_position)                # Priority 3
]

# Solve with task hierarchy
result = solver.solve_multi_task_ik(
    tasks=tasks,
    initial_q=initial_configuration
)
```

## Velocity IK

For real-time control, use velocity IK:

```python
# Current configuration
q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

# Desired end-effector velocity (6D: 3 linear + 3 angular)
target_velocity = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

# Solve for joint velocities
config = embodik.VelocitySolverConfig(
    epsilon=1e-6,
    regularization=1e-3
)

result = solver.solve_velocity_ik(
    target_velocity=target_velocity,
    q=q,
    config=config
)

if result.status == embodik.SolverStatus.SUCCESS:
    dq = result.solution  # Joint velocities
```

## Complete Example

```python
import embodik
import numpy as np

# 1. Load robot model
model = embodik.RobotModel.from_urdf("robot.urdf")

# 2. Create solver
solver = embodik.KinematicsSolver(model)

# 3. Set target pose (end-effector should be at [0.5, 0.2, 0.3])
target_pose = np.eye(4)
target_pose[:3, 3] = [0.5, 0.2, 0.3]  # Translation
# Rotation can be set via target_pose[:3, :3]

# 4. Solve IK
result = solver.solve_position_ik(
    target_pose=target_pose,
    initial_q=np.zeros(model.nq)
)

# 5. Check result
if result.status == embodik.SolverStatus.SUCCESS:
    print(f"✓ IK solved successfully!")
    print(f"  Joint angles: {result.solution}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final error: {result.final_error}")
else:
    print(f"✗ IK failed: {result.status}")
```

## Next Steps

- [Working with Transforms](transforms.md) - Learn how to create and manipulate 3D transforms
- [API Reference](api/index.md) - Detailed API documentation
- [Examples](examples/index.md) - More complex examples
- [Development Guide](development.md) - Contributing to EmbodiK
