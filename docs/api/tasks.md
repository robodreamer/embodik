# Tasks

EmbodiK supports various task types for multi-task inverse kinematics.

## Task Types

### FrameTask

Control end-effector pose (position + orientation).

```python
frame_task = embodik.FrameTask(
    frame_id="end_effector",
    target_pose=np.eye(4)  # 4x4 transformation matrix
)
```

### PostureTask

Maintain desired joint configuration.

```python
posture_task = embodik.PostureTask(
    target_q=np.array([0.0, 0.5, 0.0, -1.0, 0.0, 1.0, 0.0])
)
```

### COMTask

Control center of mass position.

```python
com_task = embodik.COMTask(
    target_com=np.array([0.0, 0.0, 0.8])  # Desired COM position
)
```

### JointTask

Control individual joint position.

```python
joint_task = embodik.JointTask(
    joint_id=3,
    target_q=0.5  # Desired joint angle
)
```

### MultiJointTask

Control multiple joints simultaneously.

```python
multi_joint_task = embodik.MultiJointTask(
    joint_ids=[0, 2, 4],
    target_q=np.array([0.5, 0.3, 0.7])  # Desired angles for selected joints
)
```

## Task Hierarchy

Tasks are solved in priority order. Higher priority tasks (lower index) are satisfied first:

```python
tasks = [
    frame_task,      # Priority 1: Must be satisfied
    posture_task,   # Priority 2: Satisfied in null space of priority 1
    com_task,        # Priority 3: Satisfied in null space of priorities 1-2
]

result = solver.solve_multi_task_ik(tasks=tasks, initial_q=q0)
```

## API Reference

::: embodik.FrameTask
    options:
      show_root_heading: true

::: embodik.PostureTask
    options:
      show_root_heading: true

::: embodik.COMTask
    options:
      show_root_heading: true

::: embodik.JointTask
    options:
      show_root_heading: true

::: embodik.MultiJointTask
    options:
      show_root_heading: true
