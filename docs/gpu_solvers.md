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
- `torch`, `warp-lang>=1.17.0`, and a compatible Newton installation;
- Viser, yourdfpy, and robot descriptions for the visual examples.

With pip or a virtual environment:

```bash
python -m pip install -e ".[examples,gpu-wbc]"
git clone --depth 1 https://github.com/newton-physics/newton.git ../newton
python -m pip install -e ../newton
```

For a Linux x86-64 repository checkout managed by Pixi, one command creates the
CUDA environment, installs EmbodiK, selects a Torch wheel that can execute on
the current GPU, and installs Newton:

```bash
pixi run setup-gpu-wbc
```

The script clones `../newton` when that checkout is absent. Set `NEWTON_DIR` to
use a different Newton source tree. It runs `check-cuda` before and after the
optional `sm_120` Torch repair. Newton is installed after that repair, because
replacing the Torch wheel removes packages that were installed into the same
environment. `check-cuda` verifies both CUDA visibility and execution of a real
kernel for the device architecture. The repair stays inside `.pixi/envs/cuda`.

The current integration was validated against the Newton 1.6 development
line with Warp 1.17.0. Older Warp builds can fail during Newton import before
an EmbodiK solve begins. The first run builds and caches Newton/Warp kernels for the selected
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

Build a solver with `from_robot()`. That factory selects `warp_srinv` and
derives the joint layout from the loaded `RobotModel`. The class constructor
is the legacy artifact path and is not the integration entry point.

Use `GpuWbcMultiFrameSolver` when `robot.is_floating_base` is false. Use
`GpuWbcFloatingMultiFrameSolver` for one standard free root. Passing the other
class raises before any CUDA work. A fixed-base call looks like this:

```python
from pathlib import Path

import numpy as np
import torch
import embodik
from embodik.gpu.wbc import GpuWbcMultiFrameSolver

urdf = Path("robot.urdf")
robot = embodik.RobotModel(str(urdf), floating_base=False)
q0 = np.asarray(robot.neutral_configuration(), dtype=np.float64)
batch_size = 1024

solver = GpuWbcMultiFrameSolver.from_robot(
    urdf,
    Path("build/gpu-wbc-cache"),
    robot=robot,
    robot_name="my_robot",  # label only
    frames=("tool_frame",),
    frame_task_dimensions=(6,),
    default_configuration=q0,
    batch_size=batch_size,
    dt=0.01,
)

q_active = np.asarray(q0, dtype=np.float32)[list(solver.active_configuration_indices)]
q_cuda = torch.as_tensor(q_active, dtype=torch.float32, device="cuda").expand(
    batch_size, -1
).contiguous()
target_cuda = torch.zeros(
    (batch_size, len(solver.frames), 7), dtype=torch.float32, device="cuda"
)
target_cuda[..., 3] = 1.0  # identity quaternion, WXYZ
result = solver.solve_device_batch(q_cuda, target_cuda)
tracked = result.world_status == solver.WORLD_STATUS_SUCCESS
q_next_cuda = result.q_solution
```

`default_configuration` is the full model vector from `RobotModel`. The tensor
passed to `solve_device_batch()` is narrower: `[batch_size, configuration_dim]`,
with one column per name in `active_joint_names`. Gather it with
`active_configuration_indices`. `q_solution` has that same width, so scatter it
back with those indices. The factory includes every supported movable joint
unless you pass `active_joint_names`.

Finite joint positions that fall outside their URDF limits are projected onto
the nearest limit before the solve. The Franka Panda description places joint 4
outside its upper limit at `neutral_configuration()`, and that home is repaired
automatically. Non-finite values and a zero quaternion still return
`WORLD_STATUS_INVALID_INPUT` and do not move. `get_joint_limits()` reports the
limits when an application wants to inspect them.

The fixed-base factory defaults to `dt=0.1` and `iterations=2`. Pass `dt`
explicitly when the control period is different. The floating-base factory
defaults to `dt=0.01` and also requires `frame_position_gains`,
`frame_orientation_gains`, and six positive `base_velocity_limits`. Its `q`
is the full configuration, shaped `[batch_size, robot.nq]`, not an active-joint
slice. The first solve for a new model or shape compiles Newton and Warp
kernels, so exclude it from latency measurements.

### Quaternion layout

These three tensors do not share a quaternion convention:

| Value | Shape | Quaternion |
| --- | --- | --- |
| Fixed-base `q` and `q_solution` | `[batch, configuration_dim]` | none; scalar joints only |
| Floating-base `q` and `q_solution` | `[batch, robot.nq]` | root quaternion is **XYZW**, after the root translation |
| `target` | `[batch, frame_count, 7]` | position, then **WXYZ** |
| `evaluate_body_poses_device(q)` | `[batch, body_count, 7]` | position, then **XYZW** |

A body pose cannot be copied into a target without swapping the quaternion.
A floating-base root packed as WXYZ is still finite, so the solve runs and the
base orientation is wrong. `current_velocity` and `previous_velocity`, when
supplied, are float32 tensors of shape `[batch, velocity_dim]` in
`active_velocity_indices` order.

### Per-world status

`result.status` is always `"solved_or_held_needs_verification"`. The outcome
of each row is `result.world_status`, an `int8` tensor of shape `[batch_size]`.
Both adapters publish the same codes:

| Code | Constant | Meaning |
| --- | --- | --- |
| 0 | `WORLD_STATUS_SUCCESS` | The world produced an accepted step |
| 1 | `WORLD_STATUS_INVALID_INPUT` | Non-finite input, a zero quaternion, or a position-limit violation |
| 2 | `WORLD_STATUS_HELD` | No accepted step; `q_solution` stays at the input |
| 3 | `WORLD_STATUS_NUMERICAL_FAILURE` | The numerical solve failed for that world |
| 4 | `WORLD_STATUS_INACTIVE` | `valid_mask` excluded the world |

Invalid, held, and inactive rows keep a configuration that is safe to write
back. Read `world_status` when the application needs to know which worlds
tracked the target. One bad row does not reject the batch.

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
    reset_mask=done_mask,
    valid_mask=active_mask,
)
q_command = q_sim.clone()
q_command.index_copy_(1, q_index, result.q_solution)
sim.set_joint_position_targets(q_command)
```

That gather is the fixed-base shape. A floating-base solver takes the full
`[num_envs, robot.nq]` configuration and does not expose
`active_configuration_indices`. Its root quaternion remains XYZW.

`reset_mask` and `valid_mask` are boolean tensors of shape `[batch_size]`.
Resetting selected worlds zeros command-acceleration history without rebuilding
the solver. Inactive worlds hold their configuration and do not advance stored
history. One non-finite or out-of-limit world sets
`WORLD_STATUS_INVALID_INPUT` on that row and does not reject the batch.

Provided tensors for a participating world are contemporaneous this tick.
Omitted `previous_velocity` retains the accepted command. There is no separate
per-field freshness schedule.

Build one solver at the training batch size, warm it before collecting timing
or rollouts, and reuse it across episode resets. Rebuilding the solver or
changing tensor shapes forces compilation and CUDA-graph setup back onto the
critical path. For velocity-controlled actuators, consume
`result.accepted_velocity` instead of `q_solution`. Use
`measure_device_batch()` when you need host-dispatch versus synchronize time
and a CUDA allocator snapshot. That helper is a solver-path measurement, not a
physics + observation + policy + rendering profile.

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
| Per-world reset/validity masks and categorical status | Supported |
| Solver-path dispatch, sync, and memory snapshot | Supported via `measure_device_batch()` |
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
Use `measure_device_batch()` to split host dispatch from the blocking
synchronize and to read allocated/reserved CUDA bytes for the same path. Full
RL-loop budgets still have to be measured in the training application.

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
