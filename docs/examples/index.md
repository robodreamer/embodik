# Examples

Example code and tutorials for EmbodiK.

## Core Guides

- [Basic IK Example](basic_ik.md) — Minimal `solve_position_step()` IK bring-up
- [Multi-Task Multi-Constraints IK Example](multi_task_ik.md) — Hierarchical tasks with CoM support-polygon constraint
- [Collision-Aware IK Example](collision_aware_ik.md) — Self-collision avoidance with velocity-damper constraints
- [CoM Constraint Example Overview](com_constraint_ik.md) — Support-polygon inequality constraints and visualization
- [Dual-Arm ECTS Example Overview](dual_arm_ects.md) — Coordinated bimanual control modes and constraints

## Highlighted Scripts

These are the public examples that are kept closest to release quality and are
the best starting points for users.

- [`01_basic_ik_simple.py`](basic_ik.md) — Minimal fixed-base IK loop with interactive target
- [`02_collision_aware_IK.py`](collision_aware_ik.md) — Collision-aware IK with optional GPU mode
- [`03_teleop_ik.py`](teleop_ik.md) — Minimal teleop input adapter into EmbodiK IK
- [`04_com_constraint_example.py`](com_constraint_ik.md) — CoM support-polygon constraint demo in Viser
- [`05_dual_arm_ects.py`](dual_arm_ects.md) — Dual-arm ECTS/Orthogonal coordination with collision handling
- [`06_bimanual_whole_body_ik.py`](bimanual_whole_body_ik.md) — Bimanual whole-body teleop for AI Worker and RB-Y1 with CoM and collision handling
- [`07_unitree_g1_retargeting_ik.py`](unitree_g1_retargeting_ik.md) — Unitree G1 retargeting IK with palm/foot/pelvis targets, CoM visualization, and optional self-collision constraints
- [`08_spot_full_body_ik_viser.py`](spot_full_body_ik.md) — Spot full-body IK in regular Viser with arm+torso, torso-only, full-body, and two-stage modes
- [`09_spot_locomanip_mjviser.py`](spot_locomanip_mjviser.md) — Spot locomanipulation ONNX policy rollout in MuJoCo through mjviser; use `embodik[mjviser]` or `pixi run -e mjviser spot-locomanip-mjviser`; for Seer teleop use the combined environment: `pixi run -e mjviser-teleop spot-locomanip-mjviser --enable-teleop`

Highlighted interactive examples enable the shared runtime policy from
`example_helpers/ik_common.py`: solver-owned pose-layout auto-switching plus the
constrained weighted fallback. This keeps example code focused on tasks,
targets, and visualization while robust constraint handling remains in C++.
Viser examples share the same default browser endpoint,
`http://localhost:8080`; pass `--port` when running multiple viewers at once.

## Specialized References

These scripts remain available in a repository checkout or copied example
bundle, but they are narrower benchmarks or API walkthroughs rather than the
main demo path.

- [`gpu_batch_ik.py`](gpu_batch_ik.md) — GPU batched velocity IK benchmark
- [`gpu_collision_batch.py`](gpu_collision_batch.md) — GPU batch collision detection benchmark
- [`gpu_solver_demo.py`](gpu_solver_demo.md) — CPU vs GPU solver scaling demo
- [`parallel_trajectory_tracking.py`](parallel_trajectory_tracking.md) — 100+ robots tracking trajectories in parallel
- [`robot_model_example.py`](robot_model_usage.md) — RobotModel API walkthrough (FK/Jacobians/CoM)
- [`visualization_example.py`](visualization_examples.md) — Visualization and interactive marker usage

## Running Examples

Install and copy the example bundle once using the
[Installation Guide](../installation.md#examples). Then run scripts from the
copied `embodik_examples` directory:

```bash
cd embodik_examples
python3 01_basic_ik_simple.py
python3 03_teleop_ik.py
```

If pip downloads `embodik-*.tar.gz` or fails while finding native CMake
packages, use the source-build fallback in the [Installation Guide](../installation.md).

Examples default to the Panda preset; use `--robot <key>` to switch models.

For repository development, use Pixi:

```bash
pixi run python examples/01_basic_ik_simple.py
pixi run python examples/03_teleop_ik.py
pixi run demo-advanced-ik  # clone-only advanced/dev IK surface
```

Pass `--robot <key>` when you want a non-default robot preset.

Clone-only advanced surfaces live in `examples/` and are intentionally not
part of the pip-facing `embodik-examples --copy` workflow.

## Example Helpers

The `examples/example_helpers/` directory contains reusable utilities:

- `ik_common.py` — Shared defaults and small IK/collision helper functions
- `teleop_ik_backend.py` — Reusable stepping IK backend for the teleop example
- `common_bimanual_model_utils.py` / `common_bimanual_teleop_app.py` — Shared AI Worker/RB-Y1 bimanual whole-body IK helpers
- `dual_arm_ik_helper.py` — Dual-arm IK utilities
- `g1_model_utils.py` / `g1_ik_runtime.py` — Unitree G1 model, retargeting, and IK runtime helpers
- `limit_profiles/` — Joint limit profile configurations
