# GPU Solvers

> **Experimental:** GPU solvers are under active development and need more validation. Use with caution in production systems.

EmbodiK provides two GPU-optimized velocity IK solvers for massive parallelism via CusADi:

- **FI-PeSNS** (Fixed-Iteration Penalized eSNS) — primary solver
- **PPH-SNS** (Parallel Penalized Hierarchical SNS) — alternative formulation

Both compile to CUDA kernels and achieve **100% constraint satisfaction** with zero violations.

## Comparison

| Solver | Description | Throughput (10K batch) |
|--------|-------------|-------------------------|
| **FI-PeSNS** | Penalty-based eSNS with analytical scaling | ~675,000 solves/sec |
| **PPH-SNS** | Soft top-k violation selection, limited rank-1 updates | ~632,000 solves/sec |

## FI-PeSNS

Fixed-Iteration Penalized eSNS trades exact constraint saturation for simpler, parallelizable penalty-based enforcement.

**Key features:**
- SRINV (Singularity-Robust Inverse) for numerical stability
- Analytical feasible task scales without iterative saturation
- Penalty gradient nudge toward feasibility each iteration
- Fixed iterations (k_max=12) for predictable compute time

```python
from embodik.gpu.casadi_fi_pesns import build_fi_pesns_single_task

fn = build_fi_pesns_single_task(
    n_dof=7, task_dim=6, n_constraints=7,
    k_max=12, mu0=1e-3, gamma=2.5, eta=0.1,
)
velocity, scales = fn(target, jacobian.flatten(), C, lower, upper)
```

## PPH-SNS

Parallel Penalized Hierarchical SNS is a GPU-native redesign with:

- **Soft top-k violation selection** using softmax weights
- **Limited rank-1 projector updates** (1–2 violators per iteration)
- **Aggressive penalty ramping** (γ=3.0)
- **Fixed-depth unrolling** for CusADi compilation

```python
from embodik.gpu.casadi_pph_sns import build_pph_sns_single_task

fn = build_pph_sns_single_task(
    n_dof=7, task_dim=6, n_constraints=7,
    k_max=14, m_max=2,  # Outer iterations, max saturations per iteration
)
velocity, scales = fn(target, jacobian.flatten(), C, lower, upper)
```

## Export and Compile

### FI-PeSNS

```bash
pixi run -e cuda export-casadi
mkdir -p ~/.local/cusadi/src/casadi_functions
cp build/casadi/fn_velocity_solve.casadi ~/.local/cusadi/src/casadi_functions/
cd ~/.local/cusadi && python run_codegen.py --fn=fn_velocity_solve
```

### PPH-SNS

```bash
pixi run -e cuda export-pph-sns  # Writes to ~/.local/cusadi/src/casadi_functions/
cd ~/.local/cusadi && python run_codegen.py --fn=fn_pph_sns_velocity_solve
```

## Benchmarking

```bash
# Compare both solvers (CPU + GPU)
pixi run -e cuda benchmark-solver-comparison

# Batched GPU benchmark at various batch sizes
pixi run -e cuda benchmark-solver-batched
```
