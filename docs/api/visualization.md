# Visualization

Optional visualization tools for EmbodiK (requires `embodik[visualization]`).

## EmbodikVisualizer

Interactive 3D visualization using Viser.

```python
import embodik
import numpy as np

model = embodik.RobotModel.from_urdf("robot.urdf")
visualizer = embodik.EmbodikVisualizer(model)

# Update robot configuration
q = np.zeros(model.nq)
visualizer.update_configuration(q)

# Show target pose
target_pose = np.eye(4)
visualizer.show_target_pose(target_pose)
```

## InteractiveVisualizer

Interactive visualization with GUI controls.

```python
visualizer = embodik.InteractiveVisualizer(model)
visualizer.run()  # Opens interactive window
```

## API Reference

::: embodik.visualization.EmbodikVisualizer
    options:
      show_root_heading: true
      show_root_toc_entry: true

::: embodik.visualization.InteractiveVisualizer
    options:
      show_root_heading: true
      show_root_toc_entry: true

## Installation

Install visualization dependencies:

```bash
pip install embodik[visualization]
```

This installs:
- `pin>=3.8.0` - Pinocchio (PyPI package name is `pin`; import is `pinocchio`)
- `viser>=0.1.0` - 3D visualization library
- `trimesh>=3.0.0` - Mesh loading for visualization

**Note:** Pinocchio 3.8.0+ includes native Viser visualization support. For legacy systems, the package falls back to custom visualization using `yourdfpy` if Pinocchio's visualizer is not available.
