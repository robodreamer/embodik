# GPU Solver Demo Overview

Overview for `examples/gpu_solver_demo.py`.

## What It Demonstrates

- End-to-end CPU vs GPU velocity IK benchmark flow
- Batch scaling and effective throughput
- Optional result plotting and error checks

## Run

```bash
python3 examples/gpu_solver_demo.py
# GPU mode
python3 examples/gpu_solver_demo.py --gpu --casadi_path path/to/fn_velocity_solve.casadi
```
