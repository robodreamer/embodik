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

### For pip-installed users (recommended)

```bash
# Create venv and install
python3 -m venv .venv
source .venv/bin/activate
unset LD_LIBRARY_PATH  # Important: avoid shared-library conflicts

pip install "embodik[examples]"

# Copy examples to local directory
embodik-examples --copy

# Run an example
cd embodik_examples
python 01_basic_ik_simple.py --robot panda
```

**Available CLI commands:**
- `embodik-examples --list` - List available examples
- `embodik-examples --copy` - Copy examples to `./embodik_examples`
- `embodik-examples --copy /path/to/dir` - Copy to custom directory

### For developers (from repository)

```bash
# Using Pixi (recommended for development)
pixi run install
pixi run python examples/01_basic_ik_simple.py --robot panda

# Or manually
pip install -e ".[examples]"
python examples/01_basic_ik_simple.py --robot panda
```

## Example Helpers

The `examples/example_helpers/` directory contains reusable utilities:

- `dual_arm_ik_helper.py` - Dual-arm IK utilities
- `limit_profiles/` - Joint limit profile configurations
