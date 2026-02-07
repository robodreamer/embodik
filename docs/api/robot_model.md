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

## Joint Index Access

These methods expose Pinocchio's per-joint configuration-space and velocity-space indexing
through a name-based API. Useful for building joint-to-motor mappings, extracting per-joint
slices from q/v vectors, and understanding the configuration structure of the robot.

### `has_joint(joint_name)`

Check whether a joint exists in the model.

```python
model.has_joint("joint1")       # True
model.has_joint("nonexistent")  # False
```

### `get_joint_id(joint_name)`

Get the internal joint index (0-based, excluding universe joint).

```python
jid = model.get_joint_id("joint1")  # Raises RuntimeError if not found
```

### `get_joint_config_index(joint_name)` / `get_joint_config_size(joint_name)`

Get the starting index (`idx_q`) and number of variables (`nq`) for a joint in the
configuration vector `q`.

```python
idx_q = model.get_joint_config_index("joint1")  # Where this joint starts in q
nq    = model.get_joint_config_size("joint1")    # 1 for revolute, 2 for continuous, 7 for floating-base
```

### `get_joint_velocity_index(joint_name)` / `get_joint_velocity_size(joint_name)`

Get the starting index (`idx_v`) and number of variables (`nv`) for a joint in the
velocity vector `v`.

```python
idx_v = model.get_joint_velocity_index("joint1")  # Where this joint starts in v
nv    = model.get_joint_velocity_size("joint1")    # 1 for revolute, 1 for continuous, 6 for floating-base
```

### Per-Joint Mapping Example

```python
import embodik
import numpy as np

model = embodik.RobotModel("robot.urdf")

# Build joint-to-index mapping
for name in model.get_joint_names():
    idx_q = model.get_joint_config_index(name)
    nq    = model.get_joint_config_size(name)
    idx_v = model.get_joint_velocity_index(name)
    nv    = model.get_joint_velocity_size(name)
    print(f"{name}: q[{idx_q}:{idx_q+nq}], v[{idx_v}:{idx_v+nv}]")

# Extract a single joint's config value
q = model.neutral_configuration()
joint_q = q[model.get_joint_config_index("joint1")]
```

## Inverse Dynamics

EmbodiK provides full inverse dynamics via Pinocchio's RNEA and CRBA, exposed through
native C++ bindings (no `pip pinocchio` needed at runtime).

### `compute_generalized_gravity(q)`

Compute gravity torque vector g(q) -- the torques needed to hold the robot stationary.

```python
q = np.array([0.5, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
tau_gravity = model.compute_generalized_gravity(q)
```

### `rnea(q, v, a)`

Full inverse dynamics: tau = M(q)*a + C(q,v)*v + g(q).

```python
# Gravity only (v=0, a=0)
tau_g = model.rnea(q, np.zeros(nv), np.zeros(nv))

# Coriolis + gravity (a=0)
tau_cg = model.rnea(q, v, np.zeros(nv))

# Full dynamics
tau = model.rnea(q, v, a)
```

### `compute_mass_matrix(q)`

Joint-space mass/inertia matrix M(q) (symmetric positive-definite, nv x nv).

```python
M = model.compute_mass_matrix(q)
```

### `compute_coriolis(q, v)`

Coriolis + centrifugal torque vector C(q,v)*v.

```python
tau_coriolis = model.compute_coriolis(q, v)
```

### `set_gravity(gravity)` / `get_gravity()`

Configure the gravity vector (default [0, 0, -9.81]).

```python
model.set_gravity(np.array([0.0, 0.0, -9.81]))  # Standard gravity
model.set_gravity(np.array([0.0, 0.0, 0.0]))     # Zero-g environment
```

## Collision Checking

### Distance-Based

```python
model.update_configuration(q)
min_dist = model.compute_min_collision_distance()   # Minimum across all pairs
distances = model.compute_collision_distances()      # All pair distances
```

### Boolean Collision Check

```python
model.update_configuration(q)
in_collision = model.check_collision()               # Contact check (dist <= 0)
too_close = model.check_collision(min_distance=0.02) # Proximity check
```

### Collision Pair Management

```python
model.get_collision_geometry_names()   # List geometry names
model.get_collision_pair_names()       # List active pairs
model.apply_collision_exclusions([("geom_a", "geom_b")])  # Disable pairs
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

# Gravity compensation torques
tau_gravity = model.compute_generalized_gravity(q)
```
