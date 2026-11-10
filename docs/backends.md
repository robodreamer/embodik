# Backend discovery

`embodik.backends` describes usable numerical implementations without changing
solver defaults. The initial catalog contains the native CPU backend only.
Robot/control engines and reference generators are separate roles; they do not
inherit a numerical `solve_qp` interface merely because they generate motion.

## Native handle

```python
import numpy as np
import embodik as eik

assert eik.backends.available() == ("native",)
backend = eik.backends.get("native")
result = backend.solve_ls(np.eye(2), np.array([1.0, 2.0]))
assert result.success
np.testing.assert_allclose(result.x, [1.0, 2.0])
```

Resolve a handle once before a hot loop. Its `solve_ls` and `solve_qp` attributes
are the exact existing native callable objects, without an extra Python solve
wrapper. Existing top-level functions and defaults remain unchanged:

```python
assert backend.solve_ls is eik.solve_ls
assert backend.solve_qp is eik.solve_qp
```

`native`, `cpu`, and `auto` are catalog aliases for the same provider. Native
results keep their existing `result.backend == "cpu"` label; provider identity
and execution-device labels are distinct. Handle calls normally omit the
native functions' legacy `backend=` keyword. That keyword is still exposed
with its original semantics; the handle does not rebind their signature.

See [Standalone Optimization](optimization.md) for supported mathematics,
objective conventions, normalized-row tolerances and result statuses.

## Capabilities and availability

```python
info = eik.backends.info("native")
assert info.id == "native"
assert info.role == "numerical"
assert info.devices == ("cpu",)
assert info.dtypes == ("float64",)
```

`info.version` comes from the loaded native extension, not installed distribution
metadata. Dtypes describe arithmetic; existing native array conversions are
unchanged. Capabilities cover least squares (including rank-deficient problems),
positive-definite QP, linear hard bounds, squared soft bounds and primal initial
guesses. They do not promise that every numerical instance converges.

Metadata records are frozen and contain immutable tuples. `info.available`
means the native math entry points are loaded, not that a robot profile is
qualified. `info.qualifications` holds version/device/dtype/problem/profile-scoped
evidence records; it is empty in this initial catalog. A successful numerical
result is not a physical robot execution certificate.

The catalog is loaded only when requested. Discovery does not import optional
numerical engines or select one merely because it is installed. It does not
change existing standalone GPU requests, GPU WBC APIs or their limitations.

## Missing implementations and errors

`info("native")` is still available when the native math functions cannot be
loaded; its `available` field is false and `reason` explains why. In that state,
`available()` returns an empty tuple and `get("native")` raises `RuntimeError`.
There is no fallback.

Unknown provider IDs raise `ValueError`. `available(role=...)` accepts
`numerical`, `control`, or `reference`; unknown roles raise `ValueError`.
Control/reference roles currently return empty tuples because those handles are
not registered in this catalog. Existing robot classes remain accessible through
their normal APIs.

ElastiQP, Frax, FRDV and additional CPU/GPU providers are not implemented by this
initial catalog. They are not advertised as interchangeable or implicitly used.
