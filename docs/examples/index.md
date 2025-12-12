# Examples

Example code and tutorials for embodiK.

## Basic Examples

### Basic IK

Simple inverse kinematics example:

[Basic IK Example](basic_ik.md)

### Multi-Task IK

Hierarchical multi-task inverse kinematics:

[Multi-Task IK Example](multi_task_ik.md)

## Available Examples

The embodiK repository includes several example scripts:

- `01_basic_ik_simple.py` - Basic IK solving
- `02_collision_aware_IK.py` - Collision-aware IK with self-collision avoidance
- `robot_model_example.py` - Robot model usage
- `visualization_example.py` - Visualization examples

## Running Examples

Examples can be run from the repository root:

```bash
# Install example dependencies
pip install embodik[examples]

# Run an example
python examples/01_basic_ik_simple.py
```

## Example Helpers

The `examples/example_helpers/` directory contains reusable utilities:

- `dual_arm_ik_helper.py` - Dual-arm IK utilities
- `limit_profiles/` - Joint limit profile configurations
