# 2026-04-27 Example Docs Patch Release Note

Patch release context for the example simplification and documentation sync.

## Scope

- Public examples were split by intent:
  - `examples/01_basic_ik_simple.py`: minimal fixed-base IK bring-up.
  - `examples/02_collision_aware_IK.py`: collision-aware interactive IK behavior demo.
  - `examples/03_teleop_ik.py`: minimal teleop input adapter into the shared IK step backend.
- Clone-only advanced surfaces live under `dev_examples/` and are excluded from the pip examples copy path.
- Public docs and README were updated to match the simplified examples.

## Validation

Passed:

- `pixi run python3 -m py_compile examples/01_basic_ik_simple.py examples/02_collision_aware_IK.py examples/03_teleop_ik.py examples/example_helpers/ik_common.py examples/example_helpers/teleop_ik_backend.py dev_examples/advanced_basic_ik.py dev_examples/advanced_interactive_ik.py`
- `pixi run docs-build`

Known release-gate issue:

- `pixi run test` completed with 6 failures, all reproduced in a targeted rerun:
  - `test/test_elastic_band.py::TestElasticBandRoundTrip::test_elastic_converges_back_after_round_trip`
  - `test/test_hardware_seed_recovery.py::test_dual_iiwa_stall_recovery_improves_status_counts_and_task_progress`
  - `test/test_panda_narrowed_limits.py::TestClampingQuantitativeBehavior::test_baseline_not_fully_frozen_on_reverse[X]`
  - `test/test_panda_narrowed_limits.py::TestClampingQuantitativeBehavior::test_baseline_not_fully_frozen_on_reverse[Y]`
  - `test/test_panda_narrowed_limits.py::TestClampingQuantitativeBehavior::test_baseline_not_fully_frozen_on_reverse[Z]`
  - `test/test_stall_handler.py::TestDualEEBodyStall::test_pull_away_unstalls_quickly`

Decision: proceed with the examples/docs patch release with this known solver-regression test debt recorded here.
