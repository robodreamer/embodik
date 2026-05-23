# Parallel Trajectory Tracking Overview

Overview for `examples/parallel_trajectory_tracking.py`.

## What It Demonstrates

- 100+ robots tracking different trajectories in parallel
- Batched FK/Jacobians (pytorch_kinematics) with GPU solve (CusADi)
- Large-scale real-time trajectory tracking architecture

## Run

```bash
pixi run -e cuda demo-parallel-tracking
```
