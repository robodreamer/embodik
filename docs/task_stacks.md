# Explicit task stacks

EmbodiK's explicit task stack is a named, declarative way to assemble registered
tasks into priority levels. It uses the existing constrained hierarchical
velocity solver, including its singularity robust inverse (SRI), saturation,
and `SCALE` / `MIN_ERROR` objective handling. It does not introduce a second
inverse-kinematics algorithm.

The stack can select the existing SNS policy per level, or use
`LEXICOGRAPHIC_LEAST_SQUARES` as a convenience preset that selects
`MIN_ERROR` for every level. The preset changes the task objective policy; the
underlying hierarchy and hard-constraint path are shared.

## Configure named levels

```python
import embodik as eik

ee = solver.add_frame_task("ee", "tool", eik.TaskType.FRAME_POSE)
posture = solver.add_posture_task("posture")

stack = eik.TaskStackConfig([
    eik.TaskLevelSpec("tracking", ["ee"], eik.TaskSolveMode.SCALE),
    eik.TaskLevelSpec("regularization", ["posture"], eik.TaskSolveMode.MIN_ERROR),
])
solver.configure_task_stack(stack)
```

Level order is highest to lowest priority. Tasks within one level are peers:
their rows are assembled into one objective, with member names canonicalized
for deterministic assembly. The level's mode and fallback policy govern all
its members; task priorities and task solve settings are not changed. Active
registered tasks omitted from the stack do not participate in that solve.

Configuration is copied and normalized. Inspect it with
`solver.task_stack_config`; `solver.clear_task_stack()` restores legacy integer
priority grouping. The stack applies to `solve_velocity()` and registered-task
`solve_position_step()`. `solve_position()` constructs its own objective stack.

## Relation to eSNS and minimum error

The original eSNS formulation already combines prioritized objectives with
inequality constraints and provides variants for direction-preserving task
scaling and minimum error. EmbodiK's existing multi-objective solver has the
same relevant division: objectives are passed by level, while the global hard
constraint matrix and bounds are passed separately. The explicit stack adds
names, level membership, explicit ordering, normalized configuration, and
per-level diagnostics around that solver.

For a level with assembled Jacobian \(J_i\), target \(b_i\), and joint
velocity \(v\), `SCALE` seeks a feasible scalar \(s_i\in[0,1]\) so the solver
can pursue \(J_i v=s_i b_i\). `MIN_ERROR` minimizes the level's residual
\(\|J_i v-b_i\|\) subject to the active constraints and the hierarchy already
established above it. If a target is unreachable, its residual can remain
nonzero; lower levels use remaining freedom without intentionally degrading
the higher level's achieved objective. `MIN_ERROR` is already available on
individual tasks. The stack-wide preset is equivalent to configuring every
level for that mode, and adds no new inverse or optimization backend.

This differs from the direction-preserving minimum-scaling policy in eSNS:
`SCALE` retains a level target's direction and reduces its magnitude when
needed; `MIN_ERROR` can use reachable components even when the full target
direction is infeasible. Both policies still share the existing SRI and
constraint-aware hierarchical solve.

## Hard constraints are separate

Hard constraints are not task levels and are not converted into objectives.
Joint velocity and position limits, collision, CoM, relative-pose, user linear
bounds, and active contact constraints continue through the solver's global
constraint and safety path. At the mathematical interface, the solve has
objective levels \((J_i,b_i)\) and a separate feasible set
\(\mathcal{F}=\{v: l\le Cv\le u\}\). Hierarchy chooses the task solution
inside \(\mathcal{F}\); a lower task cannot relax a hard bound.

`TaskStackBackend.LEXICOGRAPHIC_LEAST_SQUARES` maps each level to the existing
`MIN_ERROR` objective mode. It rejects `SCALE_ELASTIC` and explicit per-level
`allow_min_error_fallback`, since those flags describe the other SNS policy.
Global weighted recovery remains a separate runtime option and is reported as
`WEIGHTED_FALLBACK`, not as a successful prioritized hierarchy.

## Results and expectations

Velocity and registered-task position-step results report
`task_level_diagnostics` when their prioritized result maps to the configured
levels. Each record contains the level name and task names, configured and
effective solve modes, scale, residual norm, target norm, and normalized
residual. A `MIN_ERROR` level reports effective mode `MIN_ERROR` and scale
`-1`. Diagnostics describe the stacked level; they do not split values by
member task or report intermediate active sets.

Results also expose:

- `hierarchy_backend`: configured explicit stack policy.
- `hierarchy_solve_path`: legacy priority, explicit SNS, the MIN_ERROR stack
  preset, or weighted fallback.
- `higher_level_preservation_active`: whether the returned velocity came from
  a successful prioritized solve, rather than weighted fallback or a failed
  hierarchy.
- `prioritized_status` and `prioritized_status_message`: outcome before any
  weighted fallback replaces the candidate.

An infeasible global constraint set is a solver infeasibility, not a task
residual. An unreachable but solvable `MIN_ERROR` task can return success with
a nonzero residual. Callers should inspect both status and per-level residuals.

Configuration rejects empty or duplicate level names, empty stacks or levels,
duplicate membership, missing/inactive task names, and unsupported policies.
If a task is removed or made inactive after configuration, the next registered
task solve reports `INVALID_INPUT` and names the stale member.

## Runtime push/pop and agent requests

The current branch does not yet expose `push_task`, `push_level`, or matching
`pop` methods. Callers can replace the full configuration with
`configure_task_stack()`, but that is not a scoped incremental editing API.
Push/pop can be added without changing the solver algorithm: treat a stack as
an immutable, validated configuration snapshot, then atomically publish a new
snapshot between solves.

For agentic applications, the eventual interface should accept a constrained
intent or patch, not arbitrary Jacobians, constraints, or executable code. A
policy layer should map the intent to already registered task names and
permitted priority bands. Global hard constraints remain owned by the solver
and unavailable for agent reprioritization.

A useful API shape is:

```python
patch = TaskStackPatch(
    base_revision=solver.task_stack_revision,
    source="planner-session-42",
    push_levels=[TaskLevelSpec("look_at_object", ["camera_gaze"])],
)
receipt = solver.apply_task_stack_patch(patch)
solver.remove_task_stack_patch(receipt.patch_id)  # scoped pop
```

This is a design proposal, not an API implemented by this branch. It should
have these semantics:

- Validate the complete candidate stack first; publish all changes together
  or leave the active snapshot untouched.
- Require the caller's base revision and return a new monotonically increasing
  revision. Reject stale edits so concurrent agents cannot silently overwrite
  one another.
- Give every patch an opaque ID and source. Removing a patch removes only its
  own additions; it must not pop an unrelated edit that arrived later.
- Apply snapshot changes at a solve boundary. Each control tick reads one
  stable revision for its whole solve; a mid-solve update takes effect on the
  next tick.
- Permit idempotency keys and explicit receipts (`accepted`, `rejected`,
  `active_revision`, and validation reason) for retries and agent feedback.
- Support optional expiry/lease for temporary intents, with a deterministic
  fallback stack when the lease expires or its source disconnects.
- Keep safety policy, global hard constraints, and authority to activate
  physical motion outside this edit API. The agent may propose task intent;
  a trusted application layer decides whether to commit it.

For a small local application, `push_level(spec) -> token` and
`pop(token)` can be convenience wrappers over the same patch mechanism. A
token-scoped pop is safer than a blind `pop()` because multiple planner,
operator, and recovery sources can update the stack concurrently.

## Whole-body APIs for VLA, WAM, and agent callers

The task-stack API is a solver configuration interface. A VLA or general
agent usually needs a higher-level whole-body controller interface so it can
request a motion outcome without constructing task Jacobians or choosing
priority details. Keep that interface provider-neutral so the same contract
works with different VLA/WAM policies and frontier models.

Recommended layers:

1. **Typed intent:** submit an end-effector or body goal with named frame,
   target pose, tolerances, optional approach/contact mode, time window, and
   source/expiry metadata. Use typed units and explicit frames. Agents can
   request registered capabilities such as `reach`, `look_at`, `hold`, or
   `bimanual_grasp`; a trusted mapper selects EmbodiK tasks and a priority
   template.
2. **Preflight:** validate frame freshness, target bounds, robot capability,
   and current hard constraints. Return a structured accepted/rejected result
   with a reason and any reachable or limiting information. Preflight is an
   estimate for the current state, not a promise that the scene will remain
   unchanged.
3. **Execution handle:** return an opaque goal ID and revision. Support
   `get_status`, `cancel`, and `revise` operations. Bound each submitted action
   chunk by duration or horizon and re-check conditions as execution proceeds.
4. **Feedback/event stream:** report state timestamp, active stack revision,
   per-level status/residuals, active hard constraints, progress, and terminal
   reason (`reached`, `blocked`, `infeasible`, `stale_target`, `cancelled`,
   etc.). VLA/WAM policies need observations of what happened, not only an
   opaque success boolean.
5. **Audit record:** record the intent, validated configuration revision,
   solver status, applied command interval, and verification result as a
   structured trace. Keep model identity and prompt/application provenance in
   the calling orchestration layer.

Keep this request loop asynchronous and lower-rate than the deterministic
whole-body control loop. VLA action chunks or WAM-predicted trajectories can
be treated as bounded proposals: validate them, execute a short horizon, then
observe and replan. They should not replace global hard constraints or directly
write joint commands around the controller. The policy layer should also
separate permission to propose an intent from permission to commit motion.

This direction is consistent with recent interfaces that connect skill
selection, bounded low-level VLA execution, precondition checks, outcome
verification, and recovery traces, and with WAM work that composes predictors
and action generators through explicit video/action interfaces. Those are
useful design patterns, not dependencies for EmbodiK. See the references in
the [math and literature note](task_stacks_math.tex).

## Literature-informed scope and open points

The literature treats task hierarchies and inequality constraints in several
ways. eSNS and hierarchical quadratic programming can assign inequality
constraints to priority levels. This API intentionally chooses a simpler
contract: global hard constraints are independent of task priority levels.
That matches the current solver architecture and keeps safety bounds from
being mistaken for soft task objectives.

Before broadening the API, the main design questions to resolve are:

1. **Mixed units within peer levels.** A level minimizes a residual over its
   stacked rows. Frame position, orientation, momentum, and posture may use
   different units or scales. Users need explicit row/task weights or a clear
   normalization policy before combining such peers.
2. **Constraint priority.** There is no API for a constraint that is hard but
   only applies at a selected task priority. Keep constraints global unless a
   concrete use case justifies hierarchical constraints and their feasibility
   semantics.
3. **Numerical contract.** SRI damping, rank thresholds, feasibility
   tolerances, and near-singular behavior affect the practical meaning of
   strict priority. They should be documented from the existing solver and
   covered by targeted numerical checks before claiming exact mathematical
   lexicographic optimality.
4. **Priority changes over time.** Changing level membership or order between
   control ticks can change the command discontinuously. Applications that
   switch stacks may need transition or continuity policies.
5. **Evidence and performance.** The new API inherits the existing solver's
   performance and numerical limits. It has no independent timing or
   optimality claim.
6. **Runtime editing.** Define conflict resolution, ownership, expiry, and
   control-tick activation before adding push/pop convenience calls. Dynamic
   priority changes can alter commands discontinuously, so transition behavior
   should be explicit and observable.

See the [math and literature note](task_stacks_math.tex) for the formulations,
comparison table, references, and detailed scope analysis.
