# Examples

Example code and tutorials for EmbodiK.

## Core Guides

- [Basic IK Example](basic_ik.md) — Velocity-based IK with frame + posture tasks
- [Multi-Task Multi-Constraints IK Example](multi_task_ik.md) — Hierarchical tasks with CoM support-polygon constraint
- [Collision-Aware IK Example](collision_aware_ik.md) — Self-collision avoidance with velocity-damper constraints
- [CoM Constraint Example Overview](com_constraint_ik.md) — Support-polygon inequality constraints and visualization
- [Dual-Arm ECTS Example Overview](dual_arm_ects.md) — Coordinated bimanual control modes and constraints

## Script Catalog

### Interactive IK

- [`01_basic_ik_simple.py`](basic_ik.md) — Basic single-arm IK loop with interactive target
- [`02_collision_aware_IK.py`](collision_aware_ik.md) — Collision-aware IK with optional GPU mode
- [`03_teleop_ik.py`](teleop_ik.md) — Seer-controller teleoperation with collision-aware IK
- [`08_com_constraint_example.py`](com_constraint_ik.md) — CoM support-polygon constraint demo in Viser
- [`09_dual_arm_ects.py`](dual_arm_ects.md) — Dual-arm ECTS/Orthogonal coordination with collision handling
- [`incubating/robotis_ai_worker_ik.py`](../../examples/incubating/robotis_ai_worker_ik.py) — Viser-based dual-arm ROBOTIS AI worker IK demo for local FFW SG2/BG2 URDFs

### GPU and Batch

- [`04_gpu_batch_ik.py`](gpu_batch_ik.md) — GPU batched velocity IK benchmark
- [`05_gpu_collision_batch.py`](gpu_collision_batch.md) — GPU batch collision detection benchmark
- [`06_gpu_solver_demo.py`](gpu_solver_demo.md) — CPU vs GPU solver scaling demo
- [`07_parallel_trajectory_tracking.py`](parallel_trajectory_tracking.md) — 100+ robots tracking trajectories in parallel

### Utilities and Visualization

- [`robot_model_example.py`](robot_model_usage.md) — RobotModel API walkthrough (FK/Jacobians/CoM)
- [`visualization_example.py`](visualization_examples.md) — Visualization and interactive marker usage

### Incubating Ports

- [`incubating/g1_port_phase1/README.md`](../../examples/incubating/g1_port_phase1/README.md) — Behavior-parity matrix and porting intent
- [`incubating/g1_port_phase1/04_g1_ik_site_counterpart_viser.py`](../../examples/incubating/g1_port_phase1/04_g1_ik_site_counterpart_viser.py) — Dual-mode site IK (`G1 3-point` vs `EmbodiK 6D`) with retargeting presets
- [`incubating/g1_port_phase1/05_g1_base_ik_counterpart_viser.py`](../../examples/incubating/g1_port_phase1/05_g1_base_ik_counterpart_viser.py) — Floating-base full-body IK with 6D feet mode, CoM polygon controls, and retargeting presets
- [`incubating/g1_port_phase1/06_g1_collision_constraint_counterpart_viser.py`](../../examples/incubating/g1_port_phase1/06_g1_collision_constraint_counterpart_viser.py) — Collision-constrained IK with arm-only mode and closest-pair diagnostics
- [`incubating/g1_port_phase1/07_g1_dual_hand_grounded_com_viser.py`](../../examples/incubating/g1_port_phase1/07_g1_dual_hand_grounded_com_viser.py) — Reimagined G1 demo: grounded floating base, dual-hand 6D gizmos, CoM support-polygon constraint, and solver-native tight feet epsilon-box constraints

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
```

## Example Helpers

The `examples/example_helpers/` directory contains reusable utilities:

- `dual_arm_ik_helper.py` — Dual-arm IK utilities
- `limit_profiles/` — Joint limit profile configurations
