# Explicit Task Stacks

EmbodiK supports two registered-task hierarchy styles. Existing code keeps the
legacy behavior: integer `task.priority` values define the hierarchy, tasks at
the same priority are assembled jointly when their solve settings are
compatible, and no new configuration is required.

An explicit task stack makes level names, membership, order, and solve policy
part of one inspectable value. It is opt-in and uses the existing SNS numerical
backend; this phase does not add a QP dependency or a second lexicographic
solver.

## Configure named levels

Register tasks through the existing factory API, then identify them by name:

```python
import embodik as eik

ee = solver.add_frame_task("ee", "tool", eik.TaskType.FRAME_POSE)
posture = solver.add_posture_task("posture")

stack = eik.TaskStackConfig(
    [
        eik.TaskLevelSpec(
            "tracking",
            ["ee"],
            eik.TaskSolveMode.SCALE,
        ),
        eik.TaskLevelSpec(
            "regularization",
            ["posture"],
            eik.TaskSolveMode.MIN_ERROR,
        ),
    ]
)
solver.configure_task_stack(stack)
```

The first level is highest priority. `configure_task_stack()` copies and
normalizes the value, so callers can inspect the active configuration without
retaining task handles:

```python
assert solver.has_explicit_task_stack()
print(solver.task_stack_config.levels)
```

Call `solver.clear_task_stack()` to return to legacy integer-priority behavior.
Task priorities and per-task solve settings are never changed by stack
configuration.

The corresponding C++ value types are `embodik::TaskLevelSpec` and
`embodik::TaskStackConfig`; the solver methods have the same names.

## Level semantics

Each explicit level becomes exactly one SNS backend objective:

- Member tasks are canonicalized lexicographically by registered task name.
- Their target-velocity and Jacobian rows are stacked into one joint objective.
- Input order within a level therefore cannot turn peers into sequential
  priorities.
- Configured level order is authoritative; member `Task::priority` values are
  ignored but not mutated.
- Active registered tasks not named by the explicit stack are omitted from the
  solve.

A level also has exactly one `solve_mode` and one
`allow_min_error_fallback` value. These level fields explicitly normalize mixed
member task settings while the stack is active; member `solve_mode` and
`allow_min_error_fallback` fields are ignored and remain unchanged.

`MIN_ERROR` with `allow_min_error_fallback=True` is rejected because a level
already in `MIN_ERROR` cannot fall back to the same policy.
`SCALE_ELASTIC` continues to use the SNS SCALE objective together with the
existing elastic-band mechanism.

## Validation and task lifetime

Configuration rejects:

- empty stacks or levels;
- empty or duplicate level names;
- missing or inactive registered tasks;
- duplicate task membership within or across levels; and
- unsupported or incompatible level policies.

The stack stores names rather than owning extra task handles. If a member task
is later removed or made inactive, the configuration remains inspectable but
`solve_velocity()` and registered-task `solve_position_step()` return
`INVALID_INPUT` with the stale member named in `status_message`. Reconfigure the
stack or call `clear_task_stack()` to continue. `solve_position()` builds its
own internal objective sequence and is not controlled by this registered-task
configuration.

## Per-level diagnostics

When the SNS result maps one-to-one to the configured levels,
`result.task_level_diagnostics` contains one entry per level:

```python
result = solver.solve_velocity(q)
for level in result.task_level_diagnostics:
    print(
        level.name,
        level.task_names,
        level.effective_solve_mode,
        level.scale,
        level.residual_norm,
    )
```

The scale and residual describe the jointly stacked level objective. EmbodiK
does not split that residual into per-task values because the SNS backend does
not report such precision.

## Hard constraints and weighted recovery

Explicit hierarchy changes only objective assembly. Joint limits, collision,
CoM, relative-pose and user linear constraints, contact projection,
finite-step guards, and other global hard-constraint behavior use the same
solver path as legacy priority mode.

Constrained weighted fallback remains a separate runtime recovery policy:

```python
runtime = solver.runtime_config()
runtime.weighted_fallback_enabled = True
solver.configure_runtime(runtime)
```

It is not a hierarchy level and is not configured through `TaskStackConfig`.
If weighted fallback replaces the prioritized SNS result,
`weighted_fallback_used` is true and `task_level_diagnostics` is empty rather
than attributing the weighted result to individual hierarchy levels.
