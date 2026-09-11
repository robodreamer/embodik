# GPU Whole-Body IK

!!! warning "Experimental API"
    The GPU WBC API is suitable for evaluation and high-throughput research,
    but it is not yet the stable production default. Unsupported model shapes
    and features fail explicitly; GPU mode never silently calls the CPU solver.

EmbodiK's current GPU path combines:

- **Newton** for model-derived, batched forward kinematics and Jacobians;
- **Warp directional SRINV** for parallel prioritized velocity IK;
- **Torch CUDA tensors** for device-resident application state and targets;
- **CUDA graphs** for fixed-shape warm execution.

The solver derives dimensions, joint limits, velocity limits, frame ancestry,
and specialization metadata from `RobotModel`. Robot names are diagnostic and
cache labels, not dispatch keys. Panda, AI Worker, RB-Y1, G1, and Spot examples
exercise the same public adapters without a robot-family DoF table.

## When to use CPU or GPU

| Workload | Recommended path |
| --- | --- |
| One interactive robot and minimum latency | CPU `KinematicsSolver` remains the baseline |
| Hundreds or thousands of independent worlds | GPU `warp_srinv` with device-resident state |
| One GPU world with supported constraints | GPU is available, but benchmark the complete application loop |
| A CPU-only feature or unsupported joint manifold | Stay on CPU; construction fails rather than falling back |

GPU throughput improves when launch and kinematics work are amortized across
many worlds. A fast solve kernel does not make Viser rendering, host tensor
copies, target generation, or physics stepping free.

## Requirements

- Python 3.10–3.12 and a working EmbodiK source build;
- an NVIDIA GPU supported by the installed Torch, Warp, and Newton versions;
- `torch`, `warp-lang`, and a compatible Newton installation;
- Viser, yourdfpy, and robot descriptions for the visual examples.

With pip or a virtual environment:

```bash
python -m pip install -e ".[examples,gpu-wbc]"
git clone --depth 1 https://github.com/newton-physics/newton.git ../newton
python -m pip install -e ../newton
```

For a repository checkout managed by Pixi, create the CUDA environment and
install both editable source trees once:

```bash
git clone --depth 1 https://github.com/newton-physics/newton.git ../newton
pixi run -e cuda install
pixi run -e cuda python -m pip install -e ../newton
pixi run -e cuda check-cuda
```

Skip the clone command when the sibling `../newton` checkout already exists.
`check-cuda` verifies both CUDA visibility and execution of a real kernel for
the device architecture. If it reports that Torch lacks `sm_120`, repair that
Pixi environment and check again:

```bash
pixi run -e cuda setup-cuda-sm120
pixi run -e cuda check-cuda
```

The repair task installs the CUDA 12.9 Torch wheel used for `sm_120`; it does
not modify the system Python environment.

The current integration was validated against the Newton 1.6 development
line. The first run builds and caches Newton/Warp kernels for the selected
model and shape; benchmark only after warm-up.

## Run the parallel showcase

The same script supports three materially different model and task shapes:

```bash
# The interactive viewer defaults to 512 solved and rendered CUDA worlds.
pixi run -e cuda demo-parallel-tracking
pixi run -e cuda python examples/10_parallel_trajectory_tracking.py --robot ai-worker
pixi run -e cuda python examples/10_parallel_trajectory_tracking.py --robot g1

# Reproduce the 1,024-world solve-only reference profile.
pixi run -e cuda demo-parallel-tracking-benchmark
```

The viewer listens on `http://localhost:8080` by default and runs until
`Ctrl+C`. Pass `--port` when another process already uses that address.

Worlds rotate through circle, figure-eight, helix, and sweep targets with
independent phases and speeds. The viewer renders all 512 default worlds through
shared per-link mesh instances. Four colored world bands make the motion
families easy to distinguish at field scale. `--show` controls only browser
visualization; every run still solves the full `--worlds` batch. When omitted,
`--show` follows `--worlds`, so `--worlds 1024` solves and renders all 1,024.
To solve 1,024 worlds while publishing only 512 robots to the browser, pass
`--worlds 1024 --show 512`.

## Model-derived API

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
    robot_name="my_robot",  # label only
    frames=("tool_frame",),
    frame_task_dimensions=(6,),
    default_configuration=q0,
    solver_backend="warp_srinv",
    batch_size=1024,
)

q_cuda = torch.as_tensor(q_batch, dtype=torch.float32, device="cuda")
target_cuda = torch.as_tensor(target_batch_wxyz, dtype=torch.float32, device="cuda")
result = solver.solve_device_batch(q_cuda, target_cuda)
q_next_cuda = result.q_solution
```

`q_cuda` has shape `[batch_size, configuration_dim]`. Targets have shape
`[batch_size, frame_count, 7]` in position plus WXYZ quaternion order. Keep the
returned solution and optional measured velocity on the device between calls.
The public adapter rejects non-CUDA execution or backend fallback.

The factory includes every supported movable joint by default. Pass
`active_joint_names` or `active_velocity_indices` only when deliberately
building a reduced specialization.

## RL and simulator integration

Treat GPU WBC as a device-resident control layer between policy outputs and
simulator actuator commands. The scalable path keeps simulator state, targets,
solver results, and reset masks on CUDA for the entire control tick:

```python
q_index = torch.tensor(
    solver.active_configuration_indices, device=q_sim.device
)
dq_index = torch.tensor(solver.active_velocity_indices, device=dq_sim.device)
q_active = q_sim.index_select(1, q_index)
dq_active = dq_sim.index_select(1, dq_index)

# Shape: [num_envs, num_frames, 7], encoded as xyz + WXYZ quaternion.
result = solver.solve_device_batch(
    q_active,
    policy_targets,
    current_velocity=dq_active,
)
sim.set_joint_position_targets(result.q_solution)
```

Build one solver at the training batch size, warm it before collecting timing
or rollouts, and reuse it across episode resets. Reset state and enable masks
in place; rebuilding the solver or changing tensor shapes forces compilation
and CUDA-graph setup back onto the critical path. For velocity-controlled
actuators, consume `result.accepted_velocity` instead of `q_solution`.

The exact transport depends on the simulator:

- A GPU-native simulator can gather active joints, solve, and apply commands
  without a host synchronization.
- CPU MuJoCo with mjviser requires a host/device boundary. Batch one state
  upload and one command download per control tick rather than copying each
  world independently.
- Run WBC at the control cadence, which may be decimated from physics, and hold
  or interpolate commands between control ticks. Measure policy inference,
  WBC, transfers, and physics together when setting the environment rate.
- Publish only selected environments at a decimated rate. mjviser is a viewer
  and MuJoCo integration surface, not the transport for a 512-world training
  batch; rendering every environment every physics step can dominate runtime.

Example 10 demonstrates the batched control and selective-publication pattern.
The [Spot locomanipulation example](examples/spot_locomanip_mjviser.md) shows a
MuJoCo/mjviser application loop; combine the two patterns when bringing GPU WBC
into an RL environment.

## Model envelope

Supported today:

- fixed-base models with scalar one-DoF joints;
- floating-base models with one standard 7-coordinate/6-velocity free root
  plus scalar joints;
- link-frame position and pose tasks, including overdetermined layouts;
- model-derived fixed shapes and arbitrary Warp batch sizes;
- `cusolver_srinv` for batch size one and `warp_srinv` for parallel worlds.

Unsupported joint manifolds, inconsistent frame layouts, insufficient row
capacity, and non-CUDA runtime attribution fail during construction or solve.
Adding a new robot normally means supplying its URDF, frames, and policy—not
editing a kernel with that robot's joint count.

## Constraint and task coverage

Applications can inspect `GPU_WBC_CAPABILITIES` before exposing a control. A
false capability must be disabled or rejected.

| Feature | GPU status |
| --- | --- |
| Pose tasks and two-level primary/secondary priority | Supported |
| Joint position/velocity bounds and contact-frame constraints | Supported |
| Self-collision constraints and lazy collision debug | Supported |
| Posture/nullspace, torso bounds, and torso staging | Supported |
| Adaptive dt and command acceleration-history limits | Supported |
| CoM support polygon | Supported |
| Capture point and velocity ZMP in the world support frame | Supported |
| Centroidal momentum task and hard momentum bounds | Supported |
| Runtime shape-stable constraint enable/disable and tuning | Supported where advertised by capabilities |
| Device-resident multi-world solve | Supported with Warp |
| General task axis masks and joint metrics | Planned |
| Exact CPU collision tuning/certification policy | Planned |

Velocity ZMP requires measured generalized velocity. Pass `current_velocity`
to `solve_step()` or `solve_device_batch()` when it is enabled; missing or
non-finite state fails closed. Collision contact and row capacities are fixed at
construction so CUDA graph shapes remain stable. Collision debug publication is
lazy because decoding witnesses requires host-visible data.

## Public interactive examples

GPU-enabled examples retain their CPU default. Passing `--gpu-wbc` initializes
the GPU adapter and exposes the supported backend controls:

```bash
python examples/02_collision_aware_IK.py --gpu-wbc
python examples/06_bimanual_whole_body_ik.py \
  --gpu-wbc --gpu-wbc-cache-dir build/gpu-wbc-cache
python examples/07_unitree_g1_retargeting_ik.py \
  --gpu-wbc --gpu-wbc-cache-dir build/gpu-wbc-cache
python examples/08_spot_full_body_ik_viser.py --gpu-wbc
```

Use each script's `--help` for model-specific collision and backend flags.
First use of a newly selected model, task layout, or fixed capacity includes
kernel warm-up and should not be treated as steady-state solve latency.

## Measured scale-out profile

The following profile was captured on a CUDA-capable NVIDIA GPU with
approximately 24 GB of device memory, 1,024 CUDA-resident worlds, two solver
iterations, 20 warm-up steps, and 50 measured steps:

| Model | Active DoF | 6D tasks / world | p50 | p95 | Mean throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| Franka Panda | 7 | 1 | 0.906 ms | 0.924 ms | 1.13M worlds/s |
| ROBOTIS AI Worker SG2 | 15 | 2 | 4.854 ms | 4.875 ms | 211k worlds/s |
| Unitree G1 | 29 | 4 | 20.295 ms | 20.408 ms | 50.4k worlds/s |

This is a warm **CUDA solve-only** profile. Target generation, Viser, collision,
host publication, and physics stepping were disabled or excluded. The table is
evidence of batch scaling, not a claim that every constrained model runs below
one millisecond or that a 1,024-world simulation has the same end-to-end rate.

## Acceleration-level slice

`GpuAccelerationSolver` is a separate API for fixed-base models whose movable
joints each have `nq == nv == 1`. It consumes an already assembled generalized
acceleration reference and applies the CPU-compatible acceleration, one-step
velocity, position endpoint, continuous-path, and braking-viability state-box
equations. It supports CUDA graph capture and independent per-world failure.

```python
import numpy as np
import torch
from embodik.gpu.wbc import GpuAccelerationSolver

solver = GpuAccelerationSolver(robot, acceleration_limits=np.full(robot.nv, 8.0))
q = torch.zeros((1024, robot.nq), dtype=torch.float64, device="cuda")
dq = torch.zeros((1024, robot.nv), dtype=torch.float64, device="cuda")
reference = torch.zeros_like(dq)
result = solver.solve_device_batch(q, dq, 0.01, reference)
```

Frame and CoM tasks, multiple acceleration priorities, `SCALE`, floating bases,
centroidal, contact, effort, and geometric constraints are not yet part of this
acceleration slice. Inspect `GPU_ACCELERATION_CAPABILITIES` before exposing
them.
