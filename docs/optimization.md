# Standalone optimization

`solve_ls` and `solve_qp` solve dense matrix problems without constructing a robot
model. These experimental interfaces use the CPU numerical core shared with
MIN_ERROR.

The shared feasibility and constrained quadratic implementation lives in
`embodik/control_solver_kernel.hpp`. Standalone optimization includes it
directly; `embodik/ik_baseline.hpp` continues to expose the same internal helpers
for existing hierarchy callers. The kernel retains `VelocitySolverConfig` for
compatibility and does not own robot state, task policy or geometric admission.

```python
import numpy as np
import embodik

result = embodik.solve_ls(
    np.eye(2), np.array([2., -3.]),
    A=np.eye(2), l=-np.ones(2), u=np.ones(2),
)
assert result.success, result.message
print(result.x)  # [1., -1.]
```

Run the complete matrix-only example without a robot model or viewer:

```bash
pixi run python examples/control_optimization.py
pixi run python examples/control_optimization.py --case fit
```

The example includes bounded actuator allocation, a QP projection with a hard
sum equality and nonnegative commands, and weighted affine fitting with explicit
ridge. It prints each result as JSON. For the fit, the reported achieved/target
rows include the square-root weights. The source shows how to translate
`M @ x + bias - desired` into `J @ x - target` without changing its meaning.

`--backend gpu` reports `UNSUPPORTED` and exits with a nonzero status; the example
does not silently move a requested GPU solve to the CPU.

## Formulations

`solve_ls(J, target, A=None, l=None, u=None, *, ridge=0)` minimizes
`0.5 * ||J @ x - target||² + 0.5 * ridge * ||x||²`, subject to
`l <= A @ x <= u`. Rank deficient residual matrices are supported without an
implicit ridge. Nonunique minimizers may differ in their nullspace coordinates.
When numerical rank drops below both matrix dimensions and the residual is
nonzero, this initial implementation conservatively returns `NO_PROGRESS`.
It cannot distinguish an exact dependency from a weak independent direction
at working precision. This also limits some genuinely inconsistent dependent
least squares problems; it does not establish infeasibility.

`solve_qp(P, q, A=None, l=None, u=None)` minimizes
`0.5 * x.T @ P @ x + q.T @ x` with the same bounds. The linear term has a **plus**
sign. The initial backend requires symmetric positive definite `P`.
Semidefinite, indefinite and linear programs return `UNSUPPORTED`; express a
singular residual objective with `solve_ls` instead.

Omit `A` for an unconstrained problem. Omitted lower or upper bounds are infinite.
Equal bounds represent equalities. Matrices and objective vectors must be finite
and dimensions must agree. Bounds may be infinite but cannot be NaN.

The CPU adapter normalizes the entire objective by one positive constant before
optimization. Relative residual weights, explicit ridge, hard bounds and warm
starts retain their original meaning. Changing LS residual units by a factor
`s` requires changing ridge by `s**2` to describe the same problem. Results report
the objective in the supplied units. If normalization cannot preserve a finite
nonzero coefficient or ridge, the adapter returns `NO_PROGRESS` instead of
silently dropping it.

## Soft affine bounds

`solve_ls` also accepts keyword arguments `S`, `sl`, `su`, and `weights` to add

```
0.5 * sum(weights * dist(S @ x, [sl, su])**2)
```

to its residual objective. Distance is measured separately for each row: it is
zero inside the interval, `sl - S @ x` below it, and `S @ x - su` above it.
One-sided bounds give squared hinge penalties. Equal bounds give weighted
residuals. This expresses preferences that may be violated to improve the
objective; `A`, `l`, and `u` continue to express hard constraints.

```python
result = embodik.solve_ls(
    np.eye(1), np.array([-2.]),
    S=np.eye(1), sl=np.array([0.]), weights=np.array([3.]),
)
assert result.success, result.message
print(result.x)  # [-0.5]: the lower soft bound is traded against the target
```

`S` has one column per original variable. Omitted `sl` and `su` default to
negative and positive infinity; omitted `weights` default to ones. Weights
must be finite and nonnegative. Zero-weight rows and intervals
`[-inf, +inf]` have no effect. Bounds cannot be NaN, reversed, or intervals
containing no finite value. Invalid soft intervals return `INVALID_INPUT`;
conflicting valid preferences do not establish hard infeasibility. Supplying
soft bounds or weights requires `S`. Omitting `S` preserves the original API
and objective.

For an affine offset `S @ x + bias`, supply bounds `sl - bias` and `su - bias`.
To normalize distance by a positive row scale, use its inverse square as the
weight. Penalty rows keep their supplied units and relative weights. Multiplying
all residuals by `s` also requires multiplying both ridge and soft weights by
`s**2` to scale the entire objective uniformly.

The initial CPU implementation solves an exact lifted least-squares problem:
one auxiliary `z_i` is bounded by each effective soft interval, with residual
`sqrt(weights_i) * (S_i @ x - z_i)`. Minimizing over `z_i` gives precisely the
interval-distance penalty. This uses the existing bounded numerical kernel
once and adds one internal variable per effective row. Dense memory and solve
cost can therefore grow substantially with the number of soft rows. The
existing conservative numerical-rank rules also apply to the lifted residual
matrix; no implicit ridge is added.

Ridge penalizes only the original `x`. Returned `x` contains only the original
variables, and `objective` is recomputed from their true interval distances.
`max_violation` and `normalized_violation` refer only to the original hard
`A`, `l`, and `u`; a positive soft residual is not a hard violation. The same
optimization iteration limit and separately bounded feasibility search apply.

## Results and budgets

Check `result.success` before using `result.x`. Failed results have no solution.
The returned solution is independent of the input arrays.

| Status | Meaning |
| --- | --- |
| `SOLVED` | Optimization succeeded and normalized hard rows passed. |
| `INFEASIBLE` | An inconsistent hard interval or the feasibility procedure established infeasibility. |
| `NO_PROGRESS` | The procedure could not certify success; this does not prove infeasibility. |
| `INVALID_INPUT` | Invalid dimensions, coefficients, options or preparation overflow. |
| `UNSUPPORTED` | Requested backend or problem family is unavailable. |

Both functions accept `x0=None`, `tol=1e-8`, `max_iter=200`, and `backend="auto"`
as keyword options. A finite `x0` is a seed; an infeasible seed is replaced by the
feasibility procedure. `tol` checks rows normalized by their Euclidean norms;
it is not an objective accuracy guarantee. `max_iter` bounds optimization
iterations. Preliminary feasibility has a separate bounded search.

Results include optimization `iterations`, original `max_violation`,
`normalized_violation`, `objective`, and native `elapsed_ms`. Timing includes
feasibility and validation, excluding Python conversion. The LS objective
includes its constant term; include that constant when comparing QP expansions.

A numerical result is not a robot execution certificate. Robot integration still
needs its physical and geometric admission checks.

## Backend availability

`auto` and `cpu` currently select the dense CPU implementation. An explicit
`gpu` request returns `UNSUPPORTED`. Batched device resident GPU support is
planned but is not supplied yet. Current bindings accept CPU arrays and do not
provide a CUDA tensor interface or automatic device dispatch.
