# EmbodiK — Agent Context

Context for AI agents and contributors to pick up work efficiently. See `README.md` for user-facing docs and `docs/` for detailed guides.

## TL;DR

- **Build/test/release/gotchas:** See `agent guidance` for commands, architecture, and common gotchas
- **Core layout:** C++ in `cpp_core/`, Python bindings in `python_bindings/`, examples in `examples/`
- **CoM constraints:** `configure_com_constraint()` with polygon vertices; margin uses `char_size` (mean centroid→vertex distance)
- **ECTS dual-arm:** `add_absolute_frame_task()`, `add_relative_frame_task()`, `add_frame_task()`; `map_ects_mode()` / `map_ects_mode_blended()`; reference: H. A. Park, IROS 2016 (doi:10.1109/IROS.2016.7759161)

---

## Agent Workflow (Recommended)

1. **Scope first**: confirm task target (`bug`, `feature`, `docs`, `release`) and touched paths.
2. **Load context**: check `AGENTS.md`, then `private maintainer notes/` for prior iteration context, then public docs in `docs/`.
3. **Implement minimally**: prefer localized diffs over broad rewrites; preserve existing APIs unless requested.
4. **Validate**: run the smallest relevant checks first, then broader checks.
5. **Document**: update public docs for user-facing behavior; keep ad-hoc reasoning in `private maintainer notes/`.

Definition of done (default):

- Relevant tests pass (`pixi run test` or targeted subset)
- Build passes (`pixi run build`) for C++/bindings changes
- Docs build passes (`pixi run docs-build`) if docs changed
- Any new behavior is reflected in `docs/` and/or examples

---

## Coding & Documentation Guardrails

Apply these by default unless the user asks otherwise:

- **Design**: prefer DRY/KISS/YAGNI; keep separation of concerns (solver core vs adapters/UI).
- **API safety**: preserve stable public behavior; use additive changes when possible.
- **Readability**: choose descriptive names; keep functions focused; refactor opportunistically when low risk.
- **Typing/logging**: add type hints in Python changes; prefer `logging` over ad-hoc `print` in library code.
- **Dependencies**: reuse existing project/tooling patterns before introducing new frameworks.
- **Performance**: profile/measure before optimizing; avoid speculative complexity.

Docstring and docs quality:

- Follow existing lint/style conventions in repo configs.
- Write concise, audience-aware docs (what/why/how) with runnable examples where relevant.
- Include troubleshooting notes for non-obvious failure modes.
- Keep terminology consistent across API docs, examples, and guides.

Naming/file organization:

- Follow existing naming patterns in the touched area first (do not mass-rename to a new scheme).
- For new module families, keep category/variant naming consistent within that directory.

Testing expectations:

- Prefer deterministic, targeted tests for changed behavior.
- For numerical checks, use tolerances and dtypes that match current test patterns.
- Add regression tests for any fixed bug that was user-visible or release-impacting.

---

## Repo Map

| Path | Purpose |
|------|---------|
| `cpp_core/include/embodik/` | C++ headers (KinematicsSolver, RobotModel, tasks, dual_arm_ects) |
| `cpp_core/src/` | C++ implementation (kinematics_solver.cpp, robot_model.cpp, dual_arm_ects.cpp) |
| `python_bindings/src/` | Nanobind bindings (kinematics_solver_bindings.cpp, tasks_bindings.cpp, etc.) |
| `python/embodik/` | Python package (gpu, examples, utils) |
| `examples/` | Standalone example scripts (01–09; 09 = dual-arm ECTS) |
| `examples/utils/` | dual_iiwa_urdf.py, dual_panda_urdf.py, robot_models.py |
| `test/` | Pytest suite |
| `scripts/` | Build helpers (version.py, patch_qhull_cmake.py, upload_pypi.sh) |
| `docs/` | MkDocs documentation |
| `private maintainer notes/` | Internal iteration notes/runbooks (not published in MkDocs) |

---

## Internal Iteration Notes

When a task references prior experiments, regressions, or release-era debugging context,
check `private maintainer notes/` first before re-investigating from scratch. These files are
maintainer-facing context (not public docs) and should be used to inform follow-up work.

Current high-signal notes:

- `private maintainer notes/joint_limit_saturation_exit_findings.md`
- `private maintainer notes/personal_installer_gist_notes.md`
- `private maintainer notes/TASK_TEMPLATE.md`
- `private maintainer notes/POSTMORTEM_TEMPLATE.md`

Guideline: keep user-facing guidance in `docs/`; keep ad-hoc/internal investigation
artifacts in `private maintainer notes/`.

---

## External/Private Reference Policy

Some useful references may be private or not publicly crawlable (for example, internal
PDFs, private repos, or non-indexed docs). In those cases:

- Do not assume access to non-public repositories.
- Ask for concrete excerpts when precision matters (API signatures, constraints, workflows).
- Record distilled takeaways in `private maintainer notes/` so future iterations keep the context.
- Avoid copying private/internal language into public docs unless explicitly intended.

---

## Quick Commands & Release Workflow

See `agent guidance` for build/test commands and release steps. Always run via `pixi run <task>` so the hermetic environment is used.

---

## Best Practices (Discovered)

### CoM Support-Polygon Constraint

- **Margin definition:** Use `char_size` (mean distance centroid → vertices), not `min_radius`.
- **Three-layer velocity bound** (per half-plane, Spot Flex IK pattern):
  1. `com_vel_max` — always active
  2. `sqrt(2 * com_acc_max * slack)` — always active, smooth saturation
  3. `slack / dt` — only when slack < proximity_threshold
- **Anti-chattering:** Clamp slack to non-negative for position/acceleration terms; use epsilon dead-zone (e.g. 0.1 mm) at boundary to avoid sign-flip oscillation. Same pattern as `kMarginEpsilon` in joint position-limit constraints.
- **Proximity activation:** `proximity_fraction` × inradius; inradius computed in C++ from polygon. Use `get_com_proximity_threshold()` to read back.

### C++ / Bindings

- Geometry helpers (convex hull, half-planes, inradius) live in anonymous namespace in `kinematics_solver.cpp` — no extra deps.
- Python bindings: Nanobind in `python_bindings/src/`; expose with `nb::arg()` and docstrings.

### Testing

- Add unit tests in `test/` for new APIs (e.g. `test_com_constraint.py`).
- Hardware seeds (limits + self-collision + stall): `docs/recovery_robustness.md`, `test/test_hardware_seed_recovery.py`.
- Use `solver.dt = 0.01` (attribute, not `set_dt()`).
- `FrameTask`: use `set_target_pose(position, rotation)`, not `set_target_rotation` / `set_target_position`.

### Visualization (Viser)

- `add_line_segments` expects shape `(N, 2, 3)` — each row `[start, end]`.
- Example 08: CoM sphere, floor disk, drop line, inner/outer polygon boundaries; color by slack (green/orange/red).
- Example 09: Dual-arm ECTS; blue marker = absolute/left EE, green = relative/right EE; collision debug spheres + line segment.

### ECTS / Dual-Arm Coordination

- **Reference:** H. A. Park, "Dual-arm coordinated-motion task specification and performance evaluation," IROS 2016, doi:10.1109/IROS.2016.7759161.
- **Tasks:** `add_absolute_frame_task()` (alpha-blended midpoint), `add_relative_frame_task()` (grasp config), `add_frame_task()` (single EE for Orthogonal mode).
- **Orthogonal mode:** Two independent EE pose controls; use two `FrameTask`s, not absolute+relative ECTS tasks.
- **Mode-change snap:** Skip one IK step when switching modes so arms stay at current config; snap markers to effective frames.
- **dual_iiwa_urdf.py:** Replaces Drake mesh collision (links 6 & 7) with spheres for ~100× faster collision; strips `drake:acceleration` namespace attributes.

---

## Common Gotchas

See `agent guidance` for the full gotchas list.

---

## External References

- **ECTS (IROS 2016):** H. A. Park, "Dual-arm coordinated-motion task specification and performance evaluation," 2016 IEEE/RSJ IROS, Daejeon, pp. 516-522, doi:10.1109/IROS.2016.7759161.
- **Spot Flex IK:** `flex-ik-new/flex_ik_v1/spot_flex_ik/` — CoM constraint pattern (`check_proximity_to_com_constraints`, `_compute_com_constraints`), `calculate_box_constraints` for margin/vel/accel bounds.

---

## File Change Checklist

See `agent guidance` for the full-stack change checklist.
