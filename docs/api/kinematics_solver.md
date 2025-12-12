# KinematicsSolver

The `KinematicsSolver` provides inverse kinematics solving capabilities for robot models.

## Overview

`KinematicsSolver` implements efficient IK algorithms including:
- Position IK (iterative optimization)
- Velocity IK (single-step and multi-task)
- Multi-task hierarchical IK

## Creating a Solver

```python
import embodik

model = embodik.RobotModel.from_urdf("robot.urdf")
solver = embodik.KinematicsSolver(model)
```

## Position IK

Solve for joint angles to achieve a target end-effector pose:

```python
target_pose = np.eye(4)  # 4x4 transformation matrix
target_pose[:3, 3] = [0.5, 0.2, 0.3]  # Desired position

options = embodik.PositionIKOptions(
    max_iterations=100,
    tolerance=1e-6
)

result = solver.solve_position_ik(
    target_pose=target_pose,
    initial_q=np.zeros(model.nq),
    options=options
)
```

## Velocity IK

Solve for joint velocities to achieve a target end-effector velocity:

```python
target_velocity = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])  # 6D velocity

config = embodik.VelocitySolverConfig(
    epsilon=1e-6,
    regularization=1e-3
)

result = solver.solve_velocity_ik(
    target_velocity=target_velocity,
    q=current_configuration,
    config=config
)
```

## Multi-Task IK

Solve hierarchical multi-task IK problems:

```python
tasks = [
    embodik.FrameTask("end_effector", target_pose),  # Priority 1
    embodik.PostureTask(target_q),                    # Priority 2
]

result = solver.solve_multi_task_ik(
    tasks=tasks,
    initial_q=initial_configuration
)
```

## API Reference

::: embodik.KinematicsSolver
    options:
      show_root_heading: true
      show_root_toc_entry: true
