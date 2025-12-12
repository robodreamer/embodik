# Visualization Update: Using Pinocchio's Built-in ViserVisualizer

## Overview

embodiK now uses Pinocchio's built-in `ViserVisualizer` (available in pin >= 3.8.0) instead of custom URDF parsing with `yourdfpy`. This provides better integration, reduced dependencies, and follows the pattern used in projects like roboplan.

## Changes

### Before (Custom Implementation)
- Used `yourdfpy` for URDF parsing
- Custom Viser integration
- Required `spatialmath-python` for transforms (now removed - using Pinocchio's native transforms)
- More dependencies to manage

### After (Pinocchio Integration)
- Uses Pinocchio's `ViserVisualizer` (pin >= 3.8.0)
- Leverages Pinocchio's geometry model loading
- Automatic fallback to custom implementation if pin < 3.8.0
- Fewer dependencies

## Requirements

### Recommended (Pinocchio 3.8+)
```bash
pip install embodik[visualization]
# Requires: pin >= 3.8.0, viser >= 0.1.0, trimesh >= 3.0.0
```

### Legacy (Custom Implementation)
```bash
pip install embodik[visualization-legacy]
# Requires: viser >= 0.1.0, yourdfpy >= 0.0.52
# Note: spatialmath-python removed - now using Pinocchio's native transforms throughout
```

## Usage

The API remains the same - no code changes needed:

```python
import embodik

# Load robot model
model = embodik.RobotModel.from_urdf("robot.urdf")

# Create visualizer (automatically uses Pinocchio's ViserVisualizer if available)
viz = embodik.EmbodikVisualizer(model)

# Display robot configuration
q = model.get_current_configuration()
viz.display(q)
```

## How It Works

1. **Automatic Detection**: The visualization module automatically detects if Pinocchio's `ViserVisualizer` is available (pin >= 3.8.0)

2. **Pinocchio Path** (Recommended):
   - Uses `pin.visualize.ViserVisualizer`
   - Leverages Pinocchio's geometry models (visual_model, collision_model)
   - Uses Pinocchio's model and data structures directly
   - No URDF parsing needed - Pinocchio handles it

3. **Fallback Path**:
   - If pin < 3.8.0, falls back to custom implementation
   - Uses `yourdfpy` for URDF parsing
   - Maintains backward compatibility

## Benefits

1. **Better Integration**: Uses Pinocchio's native visualization infrastructure
2. **Reduced Dependencies**: No need for `yourdfpy` when using pin >= 3.8.0
3. **Removed spatialmath-python**: All transforms now use Pinocchio's native methods (better performance, fewer dependencies)
4. **Consistency**: Matches the pattern used in roboplan and other Pinocchio-based projects
5. **Performance**: Direct use of Pinocchio's geometry models and transforms is more efficient
6. **Maintenance**: Less custom code to maintain
7. **Numpy Compatibility**: No longer limited to numpy 1.x by spatialmath-python constraints

## Migration Guide

### For Users

**No action required** - the API is unchanged. Just update dependencies:

```bash
# Old way (still works)
pip install embodik[visualization-legacy]

# New way (recommended)
pip install "pin>=3.8.0" viser trimesh
pip install embodik[visualization]
```

### For Developers

If you're extending the visualization:

1. **Prefer Pinocchio's API**: Use `pin.visualize.ViserVisualizer` when available
2. **Check Availability**: Use `hasattr(pin.visualize, 'ViserVisualizer')` to check
3. **Access Models**: Use `robot_model._pinocchio_model` and `robot_model._pinocchio_data` for direct access

## Technical Details

### Pinocchio Integration

The new visualization module (`visualization_pinocchio.py`):
- Accesses Pinocchio model/data via `robot_model._pinocchio_model` and `robot_model._pinocchio_data`
- Uses geometry models from `robot_model.visual_model` and `robot_model.collision_model`
- Builds geometry models from URDF if not already loaded
- Creates `ViserVisualizer` with all required Pinocchio structures

### Backward Compatibility

- Custom visualization (`visualization.py`) is still available
- Automatic fallback ensures compatibility with older Pinocchio versions
- Both implementations provide the same API

## Comparison with roboplan

roboplan uses a similar approach:
- Custom `ViserVisualizer` wrapper (temporary until PR #2718 is merged)
- Extends `pinocchio.visualize.BaseVisualizer`
- Uses Pinocchio's model, collision_model, visual_model

embodiK now follows this pattern:
- Uses Pinocchio's built-in `ViserVisualizer` when available (pin >= 3.8.0)
- Falls back to custom implementation for older versions
- Provides same API regardless of backend

## Future Work

- Once Pinocchio 3.8+ is widely adopted, we can deprecate the custom implementation
- Consider removing `yourdfpy` dependency entirely
- Add more visualization features leveraging Pinocchio's capabilities
