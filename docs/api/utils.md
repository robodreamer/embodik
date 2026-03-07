# Utilities

Utility functions for working with EmbodiK. Includes pose error computation, quaternion conversions, and SE3 creation.

## Pose Error

::: embodik.utils.compute_pose_error
    options:
      show_root_heading: true
      show_root_toc_entry: true

::: embodik.utils.get_pose_error_vector
    options:
      show_root_heading: true
      show_root_toc_entry: true

## Quaternion Conversions (spatialmath-python compatible)

::: embodik.utils.r2q
    options:
      show_root_heading: true
      show_root_toc_entry: true

::: embodik.utils.q2r
    options:
      show_root_heading: true
      show_root_toc_entry: true

## SE3 Creation

::: embodik.utils.Rt
    options:
      show_root_heading: true
      show_root_toc_entry: true

## Example

```python
import embodik
import numpy as np

# Compute pose error between two SE3 transforms
pose_current = embodik.Rt(R=np.eye(3), t=[0, 0, 0])
pose_target = embodik.Rt(R=np.eye(3), t=[0.1, 0.2, 0.3])
error = embodik.compute_pose_error(pose_current, pose_target)
# error[:3] = translation error, error[3:] = rotation error (axis-angle)

# Quaternion conversions (native, no SciPy)
R = np.eye(3)
q_wxyz = embodik.r2q(R, order='sxyz')  # [1, 0, 0, 0]
q_xyzw = embodik.r2q(R, order='xyzs')  # [0, 0, 0, 1]
R_back = embodik.q2r(q_wxyz, order='sxyz')

# Create SE3
T = embodik.Rt(R=np.eye(3), t=[1, 2, 3])
```
