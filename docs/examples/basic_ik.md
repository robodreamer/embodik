# Basic IK Example

Simple example demonstrating basic inverse kinematics with EmbodiK.

## Code

```python
import embodik
import numpy as np

# 1. Load robot model
model = embodik.RobotModel.from_urdf("panda.urdf")

# 2. Create solver
solver = embodik.KinematicsSolver(model)

# 3. Define target pose
target_pose = np.eye(4)
target_pose[:3, 3] = [0.5, 0.2, 0.3]  # Position
# Rotation can be set via target_pose[:3, :3]

# 4. Solve IK
result = solver.solve_position_ik(
    target_pose=target_pose,
    initial_q=np.zeros(model.nq)
)

# 5. Check result
if result.status == embodik.SolverStatus.SUCCESS:
    print(f"✓ IK solved!")
    print(f"  Joint angles: {result.solution}")
    print(f"  Iterations: {result.iterations}")
else:
    print(f"✗ IK failed: {result.status}")
```

## Explanation

1. **Load Robot Model**: Create a `RobotModel` from a URDF file
2. **Create Solver**: Instantiate a `KinematicsSolver` with the model
3. **Define Target**: Set the desired end-effector pose as a 4x4 transformation matrix
4. **Solve**: Call `solve_position_ik()` with the target pose and initial configuration
5. **Check Result**: Verify the solver status and access the solution

## Output

```
✓ IK solved!
  Joint angles: [ 0.123 -0.456  0.789 -1.234  0.567 -0.890  0.345]
  Iterations: 12
```

## Next Steps

- [Multi-Task IK Example](multi_task_ik.md) - More complex examples
- [API Reference](../api/index.md) - Detailed API documentation
