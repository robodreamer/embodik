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

**Basic:**
- `01_basic_ik_simple.py` - Basic IK solving
- `02_collision_aware_IK.py` - Collision-aware IK with self-collision avoidance
- `08_com_constraint_demo.py` - CoM support-polygon constraint with Viser visualization
- `09_dual_arm_ects.py` - Dual-arm ECTS (Orthogonal + ECTS modes), collision avoidance, mode snap
- `robot_model_example.py` - Robot model usage
- `visualization_example.py` - Visualization examples

**GPU Acceleration:**
- `04_gpu_batch_ik.py` - GPU-accelerated batched velocity IK benchmark
- `05_gpu_collision_batch.py` - GPU-accelerated batch collision detection
- `06_gpu_solver_demo.py` - Comprehensive GPU solver demonstration
- `07_parallel_trajectory_tracking.py` - 100 robots tracking trajectories in parallel
- `scripts/benchmark_fi_pesns.py` - FI-PeSNS vs CPU benchmark
- `scripts/benchmark_pph_sns_comparison.py` - FI-PeSNS vs PPH-SNS comparison (CPU + GPU)
- `scripts/benchmark_pph_sns_batched.py` - Batched GPU benchmark for both solvers

## Running Examples

### For pip-installed users (recommended)

```bash
# Create venv and install (see README for full installation steps)
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# Clear environment and set up build
unset LD_LIBRARY_PATH CMAKE_PREFIX_PATH pinocchio_DIR
pip install pin scikit-build-core nanobind cmake ninja
export CMAKE_PREFIX_PATH=$(python -c "import pinocchio, pathlib; print(pathlib.Path(pinocchio.__file__).resolve().parents[4])")

# Install embodik with examples
pip install --no-build-isolation embodik
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

**Teleop example (03_teleop_ik.py) with Seer controller:** Requires xvisio. Install via `pixi install -e teleop`, then run `pixi run -e teleop demo-teleop` (or `pixi run -e teleop python examples/03_teleop_ik.py --robot panda`).

## Example Helpers

The `examples/example_helpers/` directory contains reusable utilities:

- `dual_arm_ik_helper.py` - Dual-arm IK utilities
- `limit_profiles/` - Joint limit profile configurations
