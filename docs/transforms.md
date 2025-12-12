# Working with Transforms

EmbodiK uses Pinocchio's `SE3` class for representing 3D rigid body transformations (combining rotation and translation). This guide covers basic transform operations you'll need when working with EmbodiK.

## Quick Reference

For users familiar with spatialmath-python, here are the most common equivalents:

```python
import embodik
import pinocchio as pin
import numpy as np

# Create SE3 from R and t (equivalent to SE3.Rt)
R = np.eye(3)
t = np.array([1, 2, 3])
T = embodik.Rt(R=R, t=t)  # Equivalent to spatialmath's SE3.Rt(R, t)

# Access translation (equivalent to R.t)
t = T.translation  # [1, 2, 3]

# Rotation matrix to quaternion (equivalent to r2q)
R = np.eye(3)
q = embodik.r2q(R)  # [1, 0, 0, 0] (wxyz format)

# Quaternion to rotation matrix (equivalent to q2r)
q = np.array([1, 0, 0, 0])
R = embodik.q2r(q)  # 3x3 identity matrix
```

## Overview

In EmbodiK, transforms are represented using Pinocchio's `SE3` class, which combines:
- **Rotation**: A 3×3 rotation matrix (or quaternion)
- **Translation**: A 3D vector

This is similar to spatialmath-python's `SE3`, but uses Pinocchio's native implementation for better performance and integration.

## Creating Transforms

### Identity Transform

```python
import pinocchio as pin
import numpy as np

# Create an identity transform (no rotation, no translation)
T = pin.SE3.Identity()
```

### From Rotation Matrix and Translation

```python
# Method 1: Using Pinocchio directly
R = np.eye(3)  # Identity rotation
t = np.array([1.0, 2.0, 3.0])
T = pin.SE3(R, t)

# Method 2: Using embodik.Rt() (spatialmath-python compatible)
import embodik
T = embodik.Rt(R=R, t=t)  # Equivalent to spatialmath's SE3.Rt(R, t)
T = embodik.Rt(t=t)  # Identity rotation, translation only
T = embodik.Rt(R=R)  # Rotation only, zero translation
```

### From 4×4 Homogeneous Matrix

```python
# Create 4x4 homogeneous matrix
H = np.eye(4)
H[:3, 3] = [1.0, 2.0, 3.0]  # Translation
H[:3, :3] = np.eye(3)        # Rotation

# Convert to SE3
T = pin.SE3(H[:3, :3], H[:3, 3])
```

## Accessing Transform Components

### Translation

```python
# Get translation vector
translation = T.translation
print(translation)  # [1.0, 2.0, 3.0]

# Set translation
T.translation = np.array([4.0, 5.0, 6.0])
```

### Rotation

```python
# Get rotation matrix (3x3)
rotation = T.rotation
print(rotation.shape)  # (3, 3)

# Set rotation matrix
T.rotation = np.eye(3)
```

### Homogeneous Matrix

```python
# Get 4x4 homogeneous matrix
H = T.homogeneous()
print(H.shape)  # (4, 4)
```

## Basic Operations

### Compose Transforms

Compose two transforms using the multiplication operator (`*`):

```python
T1 = pin.SE3(np.eye(3), np.array([1.0, 0.0, 0.0]))
T2 = pin.SE3(np.eye(3), np.array([0.0, 1.0, 0.0]))

# Compose: T_result = T1 * T2 (apply T2, then T1)
T_result = T1 * T2
print(T_result.translation)  # [1.0, 1.0, 0.0]
```

### Inverse Transform

```python
# Compute inverse transform
T_inv = T.inverse()

# Verify: T * T_inv should be identity
identity_check = T * T_inv
print(identity_check.translation)  # [0.0, 0.0, 0.0]
```

### Transform Points

```python
# Transform a 3D point
point = np.array([1.0, 2.0, 3.0])

# Transform point: p' = T * p
# For points, we need to use homogeneous coordinates
point_homogeneous = np.append(point, 1.0)
transformed_homogeneous = T.homogeneous() @ point_homogeneous
transformed_point = transformed_homogeneous[:3]
```

## Working with Rotations

### Creating Rotation Matrices

For basic rotations, you can use NumPy or SciPy:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

# Create rotation from roll-pitch-yaw (RPY) angles
roll, pitch, yaw = np.pi/2, 0, 0  # 90 degrees roll
R_rpy = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

# Create rotation about X-axis (90 degrees)
R_x = R.from_euler('x', np.pi/2).as_matrix()

# Create rotation about Y-axis
R_y = R.from_euler('y', np.pi/2).as_matrix()

# Create rotation about Z-axis
R_z = R.from_euler('z', np.pi/2).as_matrix()
```

Alternatively, create simple rotations manually:

```python
import numpy as np

# Rotation about Z-axis (yaw)
angle = np.pi / 2
R_z = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle),  np.cos(angle), 0],
    [0,              0,             1]
])
```

### Using Quaternions

```python
from pinocchio import Quaternion

# Create quaternion from rotation matrix
R = np.eye(3)
q = Quaternion(R)

# Access quaternion components (w, x, y, z)
w, x, y, z = q.w, q.x, q.y, q.z

# Create quaternion from components (w, x, y, z)
q2 = Quaternion(1.0, 0.0, 0.0, 0.0)  # Identity quaternion

# Convert quaternion to rotation matrix
R_from_q = q2.matrix()
```

### Quaternion Order Conventions

**Critical**: Different libraries use different quaternion orders. Always specify the `order` parameter to prevent errors:

- **`'sxyz'` or `'wxyz'`** (default): `[w, x, y, z]` - Used by viser, Pinocchio, EmbodiK, most robotics libraries
- **`'xyzs'` or `'xyzw'`**: `[x, y, z, w]` - Used by some libraries (e.g., SciPy in some contexts)

**Best Practice**: Always explicitly specify the `order` parameter when converting quaternions:

```python
from embodik import r2q, q2r

# If your data source uses wxyz format (default)
R = np.eye(3)
q = r2q(R, order='sxyz')  # Explicit: [w, x, y, z]

# If your data source uses xyzw format
q_xyzw = r2q(R, order='xyzs')  # Explicit: [x, y, z, w]

# Converting back - MUST match the order!
R1 = q2r(q, order='sxyz')  # Correct - matches r2q order
R2 = q2r(q_xyzw, order='xyzs')  # Correct - matches r2q order

# Common mistake - wrong order produces incorrect rotation!
R_wrong = q2r(q, order='xyzs')  # WRONG - will produce incorrect result
```

**All quaternion functions support the `order` parameter**:
- `r2q(R, order='sxyz')` - rotation matrix to quaternion
- `q2r(q, order='sxyz')` - quaternion to rotation matrix

**Tip**: When integrating with other libraries, check their quaternion format documentation and always specify the `order` parameter explicitly to prevent convention mismatches.

### Converting Between Representations

```python
from pinocchio import Quaternion
from embodik import r2q, q2r

# Rotation matrix to quaternion (recommended: use r2q with explicit order)
R = np.eye(3)
q_wxyz = r2q(R, order='sxyz')  # Returns [w, x, y, z] - explicit order prevents errors
q_xyzw = r2q(R, order='xyzs')  # Returns [x, y, z, w] - alternative format

# Quaternion to rotation matrix (recommended: use q2r with explicit order)
q = np.array([1, 0, 0, 0])  # wxyz format
R = q2r(q, order='sxyz')  # Explicit order prevents errors

# Using Pinocchio's Quaternion directly
q_pin = Quaternion(R)
wxyz = np.array([q_pin.w, q_pin.x, q_pin.y, q_pin.z])  # For viser format
R_from_q = q_pin.matrix()

# Rotation matrix to axis-angle (using log3)
axis_angle = pin.log3(R)  # Returns 3D vector (axis * angle)
```

## Common Transform Patterns

### Pure Translation

```python
# Create transform with only translation (no rotation)
translation = np.array([1.0, 2.0, 3.0])
T = pin.SE3(np.eye(3), translation)
```

### Pure Rotation

```python
# Create transform with only rotation (no translation)
R = np.eye(3)  # Your rotation matrix
T = pin.SE3(R, np.zeros(3))
```

### Rotation About an Axis

```python
from scipy.spatial.transform import Rotation as R

# Rotate 90 degrees about Z-axis
angle = np.pi / 2
R_z = R.from_euler('z', angle).as_matrix()
T = pin.SE3(R_z, np.zeros(3))
```

### Combining Translation and Rotation

```python
from scipy.spatial.transform import Rotation as R

# First rotate, then translate
R_rot = R.from_euler('z', np.pi/4).as_matrix()  # 45 deg about Z
t = np.array([1.0, 0.0, 0.0])
T = pin.SE3(R_rot, t)
```

## Using Transforms with EmbodiK

### Setting Target Poses

```python
import embodik
import pinocchio as pin
import numpy as np

# Create robot model
model = embodik.RobotModel.from_urdf("robot.urdf")

# Create target pose using SE3
target_pose_se3 = pin.SE3(np.eye(3), np.array([0.5, 0.2, 0.3]))

# Convert to 4x4 matrix for EmbodiK
target_pose_matrix = target_pose_se3.homogeneous()

# Use in frame task
frame_task = embodik.FrameTask(
    frame_id="end_effector",
    target_pose=target_pose_matrix
)
```

### Getting Frame Poses

```python
# Get current frame pose (returns SE3)
pose_se3 = model.get_frame_pose("end_effector")

# Access components
translation = pose_se3.translation
rotation = pose_se3.rotation

# Convert to homogeneous matrix if needed
pose_matrix = pose_se3.homogeneous()
```

## Advanced Operations

### Exponential Coordinates (Axis-Angle)

Pinocchio provides functions for working with exponential coordinates (axis-angle representation):

```python
import pinocchio as pin
import numpy as np

# Convert rotation matrix to axis-angle (exponential coordinates)
R = np.eye(3)  # Your rotation matrix
axis_angle = pin.log3(R)  # Returns 3D vector: axis * angle

# Convert axis-angle back to rotation matrix
R_from_axis_angle = pin.exp3(axis_angle)

# For SE3 transforms (6D twist)
T = pin.SE3(np.eye(3), np.array([1.0, 2.0, 3.0]))
twist = pin.log6(T.homogeneous())  # 6D twist vector
T_from_twist = pin.exp6(twist)  # Back to SE3
```

### Computing Pose Errors

For IK tasks, you often need to compute the error between two poses:

```python
from embodik.utils import compute_pose_error

# Current and goal poses
pose_current = pin.SE3(R1, t1)
pose_goal = pin.SE3(R2, t2)

# Compute 6D error vector [translation_error, rotation_error]
error = compute_pose_error(pose_current, pose_goal)
# error[:3] is translation error
# error[3:] is rotation error (in axis-angle form)
```

## Utility Functions

EmbodiK provides utility functions for common transform operations, including spatialmath-python compatible functions:

### Creating SE3 Transforms

```python
from embodik import Rt
import numpy as np

# Create SE3 from rotation and translation (equivalent to spatialmath's SE3.Rt)
R = np.eye(3)
t = np.array([1.0, 2.0, 3.0])
T = Rt(R=R, t=t)

# Convenience: identity rotation or zero translation
T = Rt(t=t)  # Identity rotation, translation only
T = Rt(R=R)  # Rotation only, zero translation
T = Rt()  # Identity transform
```

### Quaternion Conversions (spatialmath-python compatible)

```python
from embodik import r2q, q2r
import numpy as np

# Convert rotation matrix to quaternion (default: wxyz format)
R = np.eye(3)
q = r2q(R)  # Returns [1, 0, 0, 0] (wxyz format, default)
q_xyzw = r2q(R, order='xyzs')  # Returns [0, 0, 0, 1] (xyzw format)

# Convert quaternion to rotation matrix
q = np.array([1, 0, 0, 0])  # Identity quaternion (wxyz)
R = q2r(q)  # Returns 3x3 identity matrix
R = q2r(q, order='sxyz')  # Same, explicit order

# With xyzw format
q_xyzw = np.array([0, 0, 0, 1])  # Identity quaternion (xyzw)
R = q2r(q_xyzw, order='xyzs')  # Returns 3x3 identity matrix
```

### Viser-Specific Conversions

For viser compatibility, use `r2q` and `q2r` directly:

```python
from embodik import r2q, q2r
import numpy as np

# Convert rotation matrix to viser quaternion format (wxyz)
R = np.eye(3)
wxyz = r2q(R)  # Returns [w, x, y, z] array
# If you need a tuple: tuple(r2q(R))

# Convert viser quaternion to rotation matrix
wxyz = np.array([1, 0, 0, 0])  # [w, x, y, z] format
R = q2r(wxyz)  # Returns 3x3 rotation matrix

# With explicit order parameter (recommended)
wxyz = r2q(R, order='sxyz')  # Explicit: [w, x, y, z]
R = q2r(wxyz, order='sxyz')  # Explicit order prevents errors
```

### Accessing Translation from SE3

```python
import pinocchio as pin

# Create SE3 transform
T = pin.SE3(np.eye(3), np.array([1.0, 2.0, 3.0]))

# Access translation (equivalent to spatialmath-python's R.t)
translation = T.translation  # Returns [1.0, 2.0, 3.0]

# Access rotation
rotation = T.rotation  # Returns 3x3 rotation matrix
```

### Computing Pose Errors

```python
from embodik.utils import compute_pose_error

# Compute pose error between two poses
pose_current = pin.SE3(R1, t1)
pose_goal = pin.SE3(R2, t2)
error = compute_pose_error(pose_current, pose_goal)  # 6D error vector
# error[:3] is translation error
# error[3:] is rotation error (in axis-angle form)
```

## Reference

For more advanced transform operations, see:

- **Pinocchio Documentation**: [Pinocchio Python API](https://stack-of-tasks.github.io/pinocchio/)
- **SE3 Class**: Pinocchio's `SE3` class provides additional methods for Lie group operations
- **Quaternion Operations**: Pinocchio's `Quaternion` class for quaternion-based rotations
- **Rotation Utilities**: Functions like `pin.log3()`, `pin.exp3()` for exponential coordinates
- **SciPy Rotations**: [scipy.spatial.transform.Rotation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html) for creating rotation matrices from various representations

## Comparison with spatialmath-python

If you're familiar with spatialmath-python, here's a quick comparison:

| spatialmath-python | Pinocchio (EmbodiK) |
|-------------------|---------------------|
| `SE3(R, t)` | `pin.SE3(R, t)` or `embodik.Rt(R=R, t=t)` |
| `SE3.Rt(R, t)` | `embodik.Rt(R=R, t=t)` |
| `T1 * T2` | `T1 * T2` |
| `T.inv()` | `T.inverse()` |
| `T.t` | `T.translation` |
| `T.R` | `T.rotation` |
| `T.A` | `T.homogeneous()` |
| `r2q(R)` | `embodik.r2q(R)` |
| `q2r(q)` | `embodik.q2r(q)` |

### Most Common Functions

The most frequently used functions from spatialmath-python are available in EmbodiK:

```python
import embodik
import pinocchio as pin
import numpy as np

# 1. Create SE3 from R and t (equivalent to SE3.Rt(R, t))
R = np.eye(3)
t = np.array([1, 2, 3])
T = embodik.Rt(R=R, t=t)  # Equivalent to spatialmath's SE3.Rt(R, t)

# 2. Accessing translation (equivalent to R.t)
t = T.translation  # [1, 2, 3] - equivalent to spatialmath's R.t

# 3. Rotation matrix to quaternion (equivalent to r2q)
R = np.eye(3)
q = embodik.r2q(R)  # [1, 0, 0, 0] - equivalent to spatialmath's r2q(R)

# 4. Quaternion to rotation matrix (equivalent to q2r)
q = np.array([1, 0, 0, 0])  # wxyz format
R = embodik.q2r(q)  # 3x3 identity - equivalent to spatialmath's q2r(q)

# With explicit order parameter (recommended to prevent errors)
q = np.array([1, 0, 0, 0])  # wxyz format
R = embodik.q2r(q, order='sxyz')  # Explicit order specification
```

The main difference is that Pinocchio's `SE3` is optimized for robotics applications and integrates seamlessly with EmbodiK's kinematics computations.
