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

## Results and budgets

Check `result.success` before using `result.x`. Failed results have no solution.
The returned solution is independent of the input arrays.

| Status | Meaning |
| --- | --- |
| `SOLVED` | Optimization succeeded and normalized hard rows passed. |
| `INFEASIBLE` | An inconsistent interval or the feasibility procedure established infeasibility. |
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
