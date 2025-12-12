# Visualization

Optional visualization tools for embodiK (requires `embodik[visualization]`).

## embodiKVisualizer

Interactive 3D visualization using Viser.

```python
import embodik
import numpy as np

model = embodik.RobotModel.from_urdf("robot.urdf")
visualizer = embodik.embodiKVisualizer(model)

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

::: embodik.visualization.embodiKVisualizer
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
- `viser` - 3D visualization library
- `yourdfpy` - URDF parsing
- `spatialmath-python` - Spatial math utilities
