# RobotModel

The `RobotModel` class represents a robot kinematic model, typically loaded from a URDF file.

## Overview

`RobotModel` wraps a Pinocchio robot model and provides convenient access to kinematic and
dynamic properties. It supports both fixed-base and floating-base robots, and exposes
Lie-group-aware configuration-space operations for correct handling of quaternion / SE(3) joints.

## Creating a Robot Model

### From URDF

```python
import embodik

# Fixed-base robot
model = embodik.RobotModel("path/to/robot.urdf")

# Floating-base robot (e.g. humanoid)
model = embodik.RobotModel("path/to/robot.urdf", floating_base=True)
```

### From XACRO

```python
model = embodik.RobotModel.from_xacro("path/to/robot.xacro", floating_base=False)
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `nq` | `int` | Number of configuration variables (includes quaternion for floating-base) |
| `nv` | `int` | Number of velocity / tangent-space variables |
| `is_floating_base` | `bool` | Whether the robot has a floating base |
| `urdf_path` | `str` | Path to the loaded URDF file |

## Configuration-Space Operations

These methods use Pinocchio's Lie-group operations to correctly handle floating-base
(SE3 / quaternion), spherical, and other non-Euclidean joint types. **Always use these
instead of naive `q + v*dt` when integrating velocities.**

### `integrate(q, v, dt=1.0)`

Integrate velocity into configuration on the joint manifold.

```python
# After solving for joint velocities
result = solver.solve_velocity(q)
dq = np.array(result.solution)

# Correct integration (works for ALL joint types)
q_new = model.integrate(q, dq, dt=0.01)
```

For revolute/prismatic joints this is `q + v*dt`. For floating-base robots, the quaternion
component is updated via the exponential map, preserving unit-norm.

### `difference(q0, q1)`

Compute the tangent-space difference such that `q1 = integrate(q0, v)`.

```python
v = model.difference(q0, q1)  # Returns vector of size nv
```

### `neutral_configuration()`

Return the home / zero configuration (valid quaternion for floating-base).

```python
q0 = model.neutral_configuration()
```

### `random_configuration()`

Generate a random configuration within joint limits (valid quaternion for floating-base).

```python
q_rand = model.random_configuration()
```

### `normalize(q)`

Re-normalize quaternion components of a configuration (no-op for revolute/prismatic).

```python
q = model.normalize(q)  # Ensures quaternion part has unit norm
```

## Kinematics Methods

| Method | Description |
|--------|-------------|
| `update_configuration(q)` | Update configuration and compute forward kinematics |
| `update_kinematics(q, v)` | Update configuration and velocity |
| `get_frame_pose(frame_name)` | Get SE3 pose of a frame |
| `get_frame_jacobian(frame_name, ref)` | Get 6xN Jacobian of a frame |
| `get_com_position()` | Get center of mass position |
| `get_com_jacobian()` | Get 3xN COM Jacobian |

## Example

```python
import embodik
import numpy as np

# Load robot model
model = embodik.RobotModel("panda.urdf")

print(f"Config dims: nq={model.nq}, nv={model.nv}")

# Start from neutral configuration
q = model.neutral_configuration()
model.update_configuration(q)

# Get end-effector pose
ee_pose = model.get_frame_pose("panda_link8")
print(f"End-effector position: {ee_pose.translation}")

# Integrate a velocity
v = np.zeros(model.nv)
v[0] = 0.1  # Move first joint
q_new = model.integrate(q, v, dt=0.01)
model.update_configuration(q_new)
```

## Floating-Base Example

```python
import embodik
import numpy as np

# Load humanoid with floating base
model = embodik.RobotModel("humanoid.urdf", floating_base=True)
print(f"nq={model.nq} (includes 7 for SE3), nv={model.nv} (includes 6 for SE3)")

q = model.neutral_configuration()

# Velocity: [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z, joint1, ...]
v = np.zeros(model.nv)
v[0] = 0.5   # Move forward
v[5] = 0.1   # Rotate around z

# Correct manifold integration (quaternion stays normalized)
q_new = model.integrate(q, v, dt=0.01)
assert abs(np.linalg.norm(q_new[3:7]) - 1.0) < 1e-12  # Quaternion is valid!
```
