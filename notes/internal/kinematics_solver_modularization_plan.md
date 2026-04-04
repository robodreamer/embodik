# Kinematics Solver Modularization Plan

## Context

`cpp_core/src/kinematics_solver.cpp` has grown to include multiple concerns:
- task orchestration and solve pipelines
- collision constraints and cache/state handling
- support-polygon constraints (CoM, ZMP, CP)
- constraint matrix assembly utilities and solver hygiene
- status classification and helper math

The goal is to improve maintainability and testability while preserving public API behavior and binding stability.

## Proposed Phased Approach

### Phase 1 (low risk, mechanical extraction)

Extract pure/stateless helper logic from `kinematics_solver.cpp` into internal detail modules without changing behavior.

Completed in this phase:
- Added internal header: `cpp_core/include/embodik/internal/kinematics_solver_detail.hpp`
- Added implementation: `cpp_core/src/kinematics_solver_detail.cpp`
- Moved helper structs/functions:
  - `HalfspaceBoundResult`
  - `ClassifiedOutcome`
  - `compute_halfspace_velocity_bounds(...)`
  - `sanitize_solver_inputs(...)`
  - `classify_velocity_outcome(...)`
  - `classify_position_outcome(...)`
- Wired `kinematics_solver.cpp` to consume these via `detail::` aliases.
- Updated build target in `CMakeLists.txt`.

Why these first:
- They are pure and broadly reused.
- They do not depend on `KinematicsSolver` private state.
- Behavior can be preserved with minimal code movement risk.

### Phase 2 (subsystem translation unit split)

Split `kinematics_solver.cpp` into feature-aligned files while keeping `KinematicsSolver` as façade:
- `kinematics_solver_collision.cpp`
- `kinematics_solver_support_constraints.cpp` (CoM/ZMP/CP)
- `kinematics_solver_solve.cpp` (velocity/position orchestration)
- keep `kinematics_solver.cpp` for class lifecycle and glue.

### Phase 3 (deeper cleanup and deduplication)

- Reduce duplication between `solve_velocity` and `solve_position` constraint assembly paths.
- Consider explicit internal constraint-builder interfaces once behavior is stable.

## Guardrails

- Keep public API in `kinematics_solver.hpp` stable.
- Keep Python bindings unchanged unless API changes are intentional.
- Prefer behavior-preserving PRs with focused tests over broad rewrites.
- Run `pixi run build` + targeted tests after each phase.

## Notes

- Internal helpers are intentionally placed under `embodik/internal` to avoid expanding the public API contract.
- Collision subsystem and solve orchestration remain coupled and should be split only after active ZMP/CP work stabilizes.
