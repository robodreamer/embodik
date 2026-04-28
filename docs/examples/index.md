# Examples

Example code and tutorials for EmbodiK.

## Core Guides

- [Basic IK Example](basic_ik.md) — Minimal `solve_position_step()` IK bring-up
- [Multi-Task Multi-Constraints IK Example](multi_task_ik.md) — Hierarchical tasks with CoM support-polygon constraint
- [Collision-Aware IK Example](collision_aware_ik.md) — Self-collision avoidance with velocity-damper constraints
- [CoM Constraint Example Overview](com_constraint_ik.md) — Support-polygon inequality constraints and visualization
- [Dual-Arm ECTS Example Overview](dual_arm_ects.md) — Coordinated bimanual control modes and constraints

## Script Catalog

### Interactive IK

- [`01_basic_ik_simple.py`](basic_ik.md) — Minimal fixed-base IK loop with interactive target
- [`02_collision_aware_IK.py`](collision_aware_ik.md) — Collision-aware IK with optional GPU mode
- [`03_teleop_ik.py`](teleop_ik.md) — Minimal teleop input adapter into EmbodiK IK
- [`08_com_constraint_example.py`](com_constraint_ik.md) — CoM support-polygon constraint demo in Viser
- [`09_dual_arm_ects.py`](dual_arm_ects.md) — Dual-arm ECTS/Orthogonal coordination with collision handling
- [`12_ai_worker_constraint_teleop.py`](robotis_ai_worker_ik.md) — Dual-arm ROBOTIS AI Worker constraint teleop with CoM and collision handling

### GPU and Batch

- [`04_gpu_batch_ik.py`](gpu_batch_ik.md) — GPU batched velocity IK benchmark
- [`05_gpu_collision_batch.py`](gpu_collision_batch.md) — GPU batch collision detection benchmark
- [`06_gpu_solver_demo.py`](gpu_solver_demo.md) — CPU vs GPU solver scaling demo
- [`07_parallel_trajectory_tracking.py`](parallel_trajectory_tracking.md) — 100+ robots tracking trajectories in parallel

### Utilities and Visualization

- [`robot_model_example.py`](robot_model_usage.md) — RobotModel API walkthrough (FK/Jacobians/CoM)
- [`visualization_example.py`](visualization_examples.md) — Visualization and interactive marker usage

## Running Examples

### For pip-installed users (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

unset LD_LIBRARY_PATH CMAKE_PREFIX_PATH pinocchio_DIR
pip install pin scikit-build-core nanobind cmake ninja
export CMAKE_PREFIX_PATH=$(python3 -c "import pinocchio, pathlib; print(pathlib.Path(pinocchio.__file__).resolve().parents[4])")

pip install --no-build-isolation embodik
pip install "embodik[examples]"
embodik-examples --copy

cd embodik_examples
python3 01_basic_ik_simple.py --robot panda
```

### For developers (from repository)

```bash
pixi run install
pixi run python examples/01_basic_ik_simple.py --robot panda
pixi run python examples/03_teleop_ik.py --robot panda
pixi run demo-advanced-ik  # clone-only advanced/dev IK surface
```

Clone-only advanced surfaces live in `examples/` and are intentionally not
part of the pip-facing `embodik-examples --copy` workflow.

## Example Helpers

The `examples/example_helpers/` directory contains reusable utilities:

- `ik_common.py` — Shared defaults and small IK/collision helper functions
- `teleop_ik_backend.py` — Reusable stepping IK backend for the teleop example
- `dual_arm_ik_helper.py` — Dual-arm IK utilities
- `limit_profiles/` — Joint limit profile configurations
