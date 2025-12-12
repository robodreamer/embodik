# RobotModel

The `RobotModel` class represents a robot kinematic model, typically loaded from a URDF file.

## Overview

`RobotModel` wraps a Pinocchio robot model and provides convenient access to kinematic and dynamic properties.

## Creating a Robot Model

### From URDF

```python
model = embodik.RobotModel.from_urdf("path/to/robot.urdf")
```

### From Existing Pinocchio Model

```python
import pinocchio as pin
pin_model = pin.buildModelFromUrdf("robot.urdf")
model = embodik.RobotModel(pin_model)
```

## Properties

::: embodik.RobotModel
    options:
      members:
        - nq
        - nv
        - name
      show_root_heading: false

## Methods

::: embodik.RobotModel
    options:
      members:
        - from_urdf
        - compute_forward_kinematics
        - get_frame_pose
        - get_jacobian
      show_root_heading: false

## Example

```python
import embodik
import numpy as np

# Load robot model
model = embodik.RobotModel.from_urdf("panda.urdf")

print(f"Robot: {model.name}")
print(f"Number of joints: {model.nq}")
print(f"Number of DOFs: {model.nv}")

# Compute forward kinematics for a configuration
q = np.zeros(model.nq)
model.compute_forward_kinematics(q)

# Get end-effector pose
ee_pose = model.get_frame_pose("panda_link8")
print(f"End-effector pose:\n{ee_pose}")
```
