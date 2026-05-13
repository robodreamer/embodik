# Visualization

Optional visualization tools for EmbodiK (requires `embodik[visualization]`).

## EmbodikVisualizer

Interactive 3D visualization using Viser.

```python
import embodik
import numpy as np

model = embodik.RobotModel("robot.urdf")
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

- `viser>=0.1.0` — 3D visualization library
- `trimesh>=3.0.0` — Mesh loading for visualization
- `yourdfpy>=0.0.52` — URDF parsing for direct Viser visualization

Use `embodik[visualization-pinocchio]` only when you specifically want
Pinocchio's Python `ViserVisualizer`.
