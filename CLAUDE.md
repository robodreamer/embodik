# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

All commands use Pixi for hermetic environment management:

```bash
pixi run install          # Editable install with auto-rebuild
pixi run build            # Build wheel (required after C++ changes)
pixi run test             # Run pytest (excludes benchmarks)
pixi run test-verbose     # Verbose test output
pixi run test-cov         # Test with coverage
pixi run lint             # Check black/isort
pixi run format           # Apply black/isort formatting
pixi run docs-build       # Build MkDocs site
pixi run verify-agent     # Path-aware agent verification
```

Run a single test file or test:
```bash
pytest test/test_embodik.py
pytest test/test_embodik.py::test_import_and_metadata
pytest -k "multi_task"
```

GPU tests require `EMBODIK_GPU_TESTS=1` and CUDA environment.

## Architecture

**C++ core with Python bindings.** The solver is implemented in C++ (Pinocchio + Eigen), exposed to Python via Nanobind. Only runtime dependency is numpy.

### Layer structure

1. **C++ core** (`cpp_core/`): Headers in `include/embodik/`, implementations in `src/`. The main solver logic lives in `kinematics_solver.cpp` (~6000 lines).

2. **Nanobind bindings** (`python_bindings/src/`): One binding file per C++ class. Must be updated when C++ API changes.

3. **Python package** (`python/embodik/`): Utilities (transforms, stall handler, visualization), GPU solvers (CasADi/Torch/Warp), and CLI.

4. **Tests** (`test/`): Pytest suite with deterministic, targeted tests. Markers: `benchmark` for long-running tests.

### Core classes

- **RobotModel** — Wraps Pinocchio model. Loads URDF, provides FK, Jacobians, COM, collision queries. Exposes `idx_q`/`idx_v`/`nq`/`nv` per joint so Python never imports Pinocchio.
- **KinematicsSolver** — High-level multi-task velocity IK. Priority + weight hierarchy, singularity-robust inverse, joint-limit constraints, task scaling with MIN_ERROR fallback.
- **Tasks**: `FrameTask` (6D pose), `PostureTask` (joint regularization), `COMTask` (center of mass + support polygon), `JointTask` (single joint), `RelativeFrameTask` (relative pose between frames).
- **Result types**: `VelocitySolverResult` (status, joint_velocities, task_scales, task_errors), `PositionIKResult` (status, q). `SolverStatus` enum: SUCCESS, INFEASIBLE, NO_PROGRESS, etc.
- **StallHandler** — Python helper detecting solver stall near joint bounds/constraints.

### Build system

CMake + scikit-build-core + Nanobind. Pinocchio discovered via CMake (`pin>=3.8.0` build-only dep). Style: black (line-length=100), isort (profile=black).

## API Patterns

- `solver.dt = 0.01` (attribute assignment, not `set_dt()`)
- `FrameTask`: use `set_target_pose(position, rotation)`, not separate `set_target_rotation`/`set_target_position`
- Viser `add_line_segments` expects shape `(N, 2, 3)`
- CoM constraint margin uses `char_size` (mean centroid-to-vertex distance)

## File Change Checklist

When modifying features, ensure changes span the full stack:
- C++ changes in `cpp_core/` -> bindings in `python_bindings/src/` if API changes -> tests in `test/` -> examples in `examples/` if user-facing -> CHANGELOG.md on release

## Release Workflow

1. `pixi run test` -> `pixi run build` -> `pixi run version --bump <patch|minor|major>`
2. Sync `pixi.toml` version to match `pyproject.toml` (version script may not update pixi.toml)
3. Edit CHANGELOG.md, commit, push
4. `pixi run build-dist` -> `pixi run upload-pypi`

## Common Gotchas

- `pixi.toml` version can get out of sync with `pyproject.toml` — check both after version bump
- Qhull build errors: `patch-qhull` task runs before build; ensure it succeeds
- Drake iiwa mesh collision is ~100x slower than sphere primitives — use `examples/utils/dual_iiwa_urdf.py`
- Floating-base vs fixed-base mismatch: `viz.display(q)` expects nq to match visualizer model
- `pin` / LD_LIBRARY_PATH conflicts: use `embodik-sanitize-env` or `unset LD_LIBRARY_PATH`
- `RobotVisualizer` has no `visualize_com` — use Viser scene API directly (`add_icosphere`, `add_line_segments`)
- `drake:acceleration` namespace warning: `dual_iiwa_urdf.py` strips `drake:*` attributes

## Git Worktrees

Worktrees live at `~/Projects/git-worktrees/embodik-<feature-name>`:

```bash
git worktree add ~/Projects/git-worktrees/embodik-<feature> -b <branch-name>
cd ~/Projects/git-worktrees/embodik-<feature>
pixi run install   # must build in each worktree separately
```

Existing worktrees are listed in `~/Projects/git-worktrees/`.

## Further Context

See AGENTS.md for agent workflow, coding guardrails, CoM/ECTS best practices, and internal notes policy.
