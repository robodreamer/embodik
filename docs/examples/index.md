# Examples

Example code and tutorials for EmbodiK.

!!! tip "Not sure where to start?"
    New to the API → [Quickstart](../quickstart.md) and [Basic IK](basic_ik.md).
    Teleop stalls or weak tracking → [Solver Robustness & Recovery](../solver_robustness.md).
    Collision latency or safety → [Collision Constraints & Tuning](../collision_constraints.md).
    Full guide map → [Guides overview](../guides/index.md).

## 🧭 Core Guides

- [Basic IK Example](basic_ik.md) — Minimal `solve_position_step()` IK bring-up
- [Multi-Task Multi-Constraints IK Example](multi_task_ik.md) — Hierarchical tasks with CoM support-polygon constraint
- [Collision-Aware IK Example](collision_aware_ik.md) — Self-collision avoidance with velocity-damper constraints
- [CoM Constraint Example Overview](com_constraint_ik.md) — Support-polygon inequality constraints and visualization
- [Dual-Arm ECTS Example Overview](dual_arm_ects.md) — Coordinated bimanual control modes and constraints

## ⭐ Highlighted Scripts

These are the public examples that are kept closest to release quality and are
the best starting points for users.

- [`01_basic_ik_simple.py`](basic_ik.md) — Minimal fixed-base IK loop with velocity/acceleration solver selection
- [`02_collision_aware_IK.py`](collision_aware_ik.md) — Collision-aware IK with velocity/acceleration solver selection and optional GPU mode
- [`03_teleop_ik.py`](teleop_ik.md) — Minimal teleop input adapter into EmbodiK IK
- [`04_com_constraint_example.py`](com_constraint_ik.md) — Visual Panda CoM, momentum, capture-point, and velocity-ZMP support demo
- [`05_dual_arm_ects.py`](dual_arm_ects.md) — Dual-arm ECTS/Orthogonal coordination with collision handling
- [`06_bimanual_whole_body_ik.py`](bimanual_whole_body_ik.md) — Bimanual whole-body teleop for AI Worker and RB-Y1 with opt-in centroidal support, collision handling, adaptive tuning, and optional Seer input
- [`07_unitree_g1_retargeting_ik.py`](unitree_g1_retargeting_ik.md) — Unitree G1 retargeting IK with palm/foot/pelvis targets, CoM visualization, and optional self-collision constraints
- [`08_spot_full_body_ik_viser.py`](spot_full_body_ik.md) — Spot full-body IK in regular Viser with arm+torso, torso-only, full-body, and two-stage modes
- [`09_spot_locomanip_mjviser.py`](spot_locomanip_mjviser.md) — Spot locomanipulation ONNX policy rollout in MuJoCo through mjviser
- [`10_parallel_trajectory_tracking.py`](parallel_trajectory_tracking.md) — Model-derived Newton/Warp WBC over independently targeted Panda, AI Worker, or G1 worlds

Highlighted interactive examples use a shared solver runtime policy so example
code stays focused on tasks, targets, and visualization while robust constraint
handling remains in C++.
Viser examples share the same default browser endpoint,
`http://localhost:8080`; pass `--port` when running multiple viewers at once.

## Acceleration Compatibility

Velocity remains the default solver for all IK examples. Acceleration is
selectable only where the script owns explicit `q`, `dq`, and `dt` state,
translates every active task and constraint, and can fail closed when the model
or policy is unsupported. A fixed-base model alone does not make a velocity
example acceleration-compatible. The acceleration-limit option on
`KinematicsSolver` only bounds changes in velocity output; it does not switch
the application to `AccelerationSolver`.

| Script | Acceleration status | Reason |
| --- | --- | --- |
| `01_basic_ik_simple.py` | Selectable | Fixed-base scalar joints with explicit acceleration state and task references. |
| `02_collision_aware_IK.py` | Selectable | Adds explicit state ownership and a fail-closed sampled velocity-collision adapter. |
| `03_teleop_ik.py` | Velocity-only | The teleop backend owns velocity position-step and reset policy; no acceleration adapter is implemented. |
| `04_com_constraint_example.py` | Velocity-only | Owns explicit `dq` for position-step momentum, capture-point, and velocity-ZMP constraints; its visual runtime is not an acceleration adapter. |
| `05_dual_arm_ects.py` | Velocity-only | ECTS mode switching and its task semantics are not exposed by the initial acceleration task API. |
| `06_bimanual_whole_body_ik.py` | Velocity-only | Exposes opt-in position-step momentum, capture-point, and velocity-ZMP controls while retaining velocity-specific continuity, ownership, collision, and fallback policies. |
| `07_unitree_g1_retargeting_ik.py` | Unsupported | Uses a floating-base humanoid model; the acceleration solver accepts fixed-base scalar joints only. |
| `08_spot_full_body_ik_viser.py` | Unsupported | Uses a floating-base whole-body model and velocity-specific recovery policy. |
| `09_spot_locomanip_mjviser.py` | Unsupported | Couples floating-base IK to wheel/locomotion policy and MuJoCo runtime state. |
| `collision_hardening_demo.py` | Velocity-only | Demonstrates velocity position-step collision recovery and non-worsening-floor behavior. |
| `example_helpers/common_bimanual_teleop_app.py` | Velocity-only | Internal executable behind the bimanual example; it has the same velocity-specific policy dependencies. |
| `floating_base_torso_hierarchy.py` | Unsupported | Its subject is floating-base hierarchy behavior. |
| `gpu_collision_batch.py` | Not applicable | Benchmarks collision-distance queries, not an IK solver loop. |
| `harnesses/ai_worker_weighted_fallback_harness.py` | Not applicable | Measures the velocity solver's weighted-fallback policy, which the acceleration API does not import. |
| `harnesses/g1_four_gizmo_ik_benchmark.py` | Unsupported | Exercises the same floating-base G1 runtime as example 07. |
| `10_parallel_trajectory_tracking.py` | Not applicable | Exercises the model-derived parallel velocity WBC pipeline. |
| `robot_model_example.py` | Not applicable | Walks through model, FK, Jacobian, and CoM APIs without an IK loop. |
| `visualization_example.py` | Not applicable | Demonstrates visualization and marker APIs without an IK solver loop. |

For the exact supported constraints and failure behavior, see the
[Acceleration Solver](../acceleration_solver.md) guide.

## 🧪 Specialized References

These scripts remain available in a repository checkout or copied example
bundle, but they are narrower benchmarks or API walkthroughs rather than the
main demo path.

- [`gpu_collision_batch.py`](gpu_collision_batch.md) — GPU batch collision detection benchmark
- [`robot_model_example.py`](robot_model_usage.md) — RobotModel API walkthrough (FK/Jacobians/CoM)
- [`visualization_example.py`](visualization_examples.md) — Visualization and interactive marker usage

## 🚀 Running Examples

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
```

Pass `--robot <key>` when you want a non-default robot preset.

## 🧰 Example Helpers

The `examples/example_helpers/` directory contains reusable utilities:

- `ik_common.py` — Shared defaults and small IK/collision helper functions
- `teleop_ik_backend.py` — Reusable registered-task IK backend for the teleop example
- `common_bimanual_model_utils.py` / `common_bimanual_teleop_app.py` — Shared bimanual whole-body IK helpers for public robot examples
- `dual_arm_ik_helper.py` — Dual-arm IK utilities
- `g1_model_utils.py` / `g1_ik_runtime.py` — Unitree G1 model, retargeting, and IK runtime helpers
- `limit_profiles/` — Joint limit profile configurations
