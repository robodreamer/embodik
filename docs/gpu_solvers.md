# GPU Solvers

> **Experimental:** GPU solvers are under active development and need more validation. Use with caution in production systems.

!!! tip "When to read this guide"
    Use the model-derived WBC API for interactive GPU IK or many independent
    device-resident worlds. The CPU [KinematicsSolver](api/kinematics_solver.md)
    remains the stable production API.

## Model-derived whole-body GPU IK

`embodik.gpu.wbc` contains the Newton/Warp runtime used by the public Panda,
bimanual, G1, and Spot examples. Native Torch, Warp, and cuSOLVER backends do
not require a pre-generated CusADi manifest.

```python
from pathlib import Path

import torch
import embodik
from embodik.gpu.wbc import GpuWbcMultiFrameSolver

urdf = Path("robot.urdf")
robot = embodik.RobotModel(str(urdf), floating_base=False)
q0 = robot.neutral_configuration()

solver = GpuWbcMultiFrameSolver.from_robot(
    urdf,
    Path("build/gpu-wbc-cache"),
    robot=robot,
    robot_name="my_robot",       # cache/diagnostic label, not dispatch
    frames=("tool_frame",),
    frame_task_dimensions=(6,),
    default_configuration=q0,
    solver_backend="warp_srinv",
    batch_size=1024,
)

# Keep simulation state and targets resident on CUDA.
q_cuda = torch.as_tensor(q_batch, dtype=torch.float32, device="cuda")
target_cuda = torch.as_tensor(target_batch_wxyz, dtype=torch.float32, device="cuda")
result = solver.solve_device_batch(q_cuda, target_cuda)
q_next_cuda = result.q_solution
```

The factory includes every supported movable joint by default so collision,
posture, CoM, torso, and secondary tasks do not silently lose authority. Pass
`active_joint_names` or `active_velocity_indices` explicitly to request a
smaller specialization.

Current model envelope:

- fixed-base models with scalar one-DoF joints;
- floating-base models with one standard 7-coordinate/6-velocity free root
  plus scalar joints;
- body/link frame position or pose tasks, including overdetermined layouts;
- arbitrary batch sizes with `warp_srinv`; `cusolver_srinv` is currently B=1.

Unsupported joint manifolds and backend capacities fail explicitly during
construction. Internal kernels are fixed-shape for CUDA graph performance, but
their dimensions are derived from the model and task specification rather than
from a robot-family table.

### Feature parity

Applications can inspect `GPU_WBC_CAPABILITIES` before exposing a control. A
false capability must be disabled or rejected; GPU mode never silently invokes
the CPU implementation.

| Feature | GPU status |
|---|---|
| Pose tasks and two-level primary/secondary priority | Supported |
| Joint position/velocity bounds and contact-frame constraints | Supported |
| Self-collision constraints and lazy collision debug | Supported |
| Posture/nullspace, torso bounds, and torso staging | Supported |
| Adaptive dt and velocity-solver acceleration-history limits | Supported |
| CoM support-polygon constraints | Supported |
| Device-resident multi-world solve | Supported with Warp |
| Capture-point and velocity-ZMP constraints | Supported in the world support frame |
| Centroidal momentum tasks and hard momentum bounds | Supported |
| Acceleration-level task solver | Planned |
| General task axis masks and joint metrics | Planned |
| Exact CPU collision tuning/certification policy | Planned |

The acceleration limit above bounds changes in the velocity command using
device-resident history. It is not the acceleration-level eSNS API exposed by
the CPU solver.

Velocity-ZMP uses measured generalized velocity, not the solver's previous
command. Pass `current_velocity` to `solve_step()` or `solve_device_batch()`
whenever that constraint is enabled; a missing or non-finite state fails
closed. Capture-point and ZMP polygons, margins, and physical settings can be
updated with `configure_runtime()` without changing their construction-time
row capacities. Momentum priority and excluded velocity columns are
construction-time layout choices; changing either requires rebuilding the
solver.

Install the optional Python dependencies with `embodik[gpu-wbc]`. The validated
Newton 1.6 development build must currently be installed from the
`newton-physics/newton` source repository.

## Legacy CusADi velocity solvers

EmbodiK also retains two earlier GPU-optimized velocity IK experiments via CusADi:

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
