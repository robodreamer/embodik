# Utilities

Utility functions for working with EmbodiK.

## Functions

::: embodik.utils.get_pose_error_vector
    options:
      show_root_heading: true
      show_root_toc_entry: true

## Example

```python
import embodik
import numpy as np

# Compute pose error between two transformations
pose_current = np.eye(4)
pose_target = np.eye(4)
pose_target[:3, 3] = [0.1, 0.2, 0.3]

error = embodik.get_pose_error_vector(pose_current, pose_target)
print(f"Pose error (6D): {error}")
# Output: [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]
```
