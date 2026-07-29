# Centroidal Stability Example

[`10_centroidal_stability.py`](https://github.com/humanoid-path-planner/embodik/blob/main/examples/10_centroidal_stability.py)
is a deterministic headless example for the velocity and acceleration
centroidal APIs. It generates a small fixed-base model locally, so it needs no
robot download or visualizer.

The velocity pass demonstrates:

- absolute `CentroidalMomentumTask`
- selected-axis hard momentum bounds
- explicit-state `solve_velocity_with_state()`
- commanded-velocity capture point
- finite-difference physical ZMP

The acceleration pass demonstrates:

- `CentroidalMomentumRateObjective`
- physical `CentroidalMomentumRateBounds`
- predicted `CapturePointAccelerationConstraint`
- physical `ZmpAccelerationConstraint` with a positive `Fz` gate
- accepted-state diagnostics

Run it with:

```bash
pixi run python examples/10_centroidal_stability.py
```

For automated checks, request one JSON object:

```bash
pixi run python examples/10_centroidal_stability.py --json
```

The report includes solver status, minimum capture-point and ZMP half-plane
slack, vertical force, command magnitude, and the truthful
`supports_dynamic_balance` capability flag.
