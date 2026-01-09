# Examples

Example code and tutorials for EmbodiK.

## Basic Examples

### Basic IK

Simple inverse kinematics example:

[Basic IK Example](basic_ik.md)

### Multi-Task IK

Hierarchical multi-task inverse kinematics:

[Multi-Task IK Example](multi_task_ik.md)

## Available Examples

The EmbodiK repository includes several example scripts:

- `01_basic_ik_simple.py` - Basic IK solving
- `02_collision_aware_IK.py` - Collision-aware IK with self-collision avoidance
- `robot_model_example.py` - Robot model usage
- `visualization_example.py` - Visualization examples

## Running Examples

### For pip-installed users

Examples are included in the pip package. To access them:

**Option 1: Use the `embodik-examples` command (recommended)**
```bash
# Install with example dependencies
pip install embodik[examples]

# List available examples
embodik-examples --list

# Copy examples to a local directory for editing
embodik-examples --copy

# Then run examples from the copied directory
cd embodik_examples
python 01_basic_ik_simple.py --robot panda
```

**Option 2: Find examples in the package**
```bash
# Find where examples are installed
python -c "import embodik; from pathlib import Path; print(Path(embodik.__file__).parent.parent / 'examples')"

# Run directly (path will vary by installation)
python /path/to/site-packages/embodik/examples/01_basic_ik_simple.py
```

### For developers (from repository)

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
