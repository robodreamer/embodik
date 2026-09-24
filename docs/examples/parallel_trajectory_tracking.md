# Example 10: Parallel GPU WBC Showcase

`examples/10_parallel_trajectory_tracking.py` solves many independently targeted
robot worlds through the public model-derived GPU WBC API. It replaces the
earlier Panda-only synthetic-Jacobian/CusADi demo.

<video autoplay muted loop playsinline controls width="100%" aria-label="Panda, ROBOTIS AI Worker, and Unitree G1 moving through distinct trajectories across 512 fully articulated CUDA worlds" src="../../assets/media/gpu_wbc_parallel_showcase.mp4"></video>

## What it demonstrates

- real Newton forward kinematics and Jacobians for every world;
- Warp directional SRINV with joint position and velocity bounds;
- one generalized solver surface for Panda, AI Worker, and G1;
- dimensions and active joints derived from `RobotModel`, not hard-coded DoF;
- circle, figure-eight, helix, and sweep targets with independent per-world
  phases and speeds;
- 512 live articulated robots by default, with a tested 1,024-world scale mode,
  rendered through shared per-link mesh instances;
- CUDA-event timing with explicit no-fallback attribution checks.

The profiles intentionally differ: Panda has one moving 6D hand task, AI
Worker has two moving 6D tool tasks, and G1 has two moving hand tasks plus two
anchored 6D foot tasks.

## Run the viewer

From a repository checkout, prepare the CUDA environment once:

```bash
pixi run setup-gpu-wbc
```

That installs EmbodiK, a Torch wheel that can run on this GPU, and Newton.
After it prints that the kernel probe passed, launch the default 512-world
Panda viewer:

```bash
pixi run -e cuda demo-parallel-tracking
```

Open `http://localhost:8080` if a browser does not open automatically, and use
`Ctrl+C` to stop the server. Select the other robot profiles with:

```bash
pixi run -e cuda python examples/10_parallel_trajectory_tracking.py --robot ai-worker
pixi run -e cuda python examples/10_parallel_trajectory_tracking.py --robot g1
```

The default viewer solves and renders all 512 live robots. It reuses each link
mesh in a single batched Viser object, then updates the per-world link transforms
from the device-resident Newton body poses. This keeps the browser scene
proportional to the number of unique link meshes rather than robots times links.
Four colored world bands identify the motion families, while phase and speed
still vary per world.

Use the full tested scale with:

```bash
pixi run -e cuda python examples/10_parallel_trajectory_tracking.py \
  --robot panda --worlds 1024
```

`--show` follows `--worlds` when omitted, so that command also renders all 1,024
robots. Add `--show 512` to solve all 1,024 while reducing browser publication.

The AI Worker viewer resolves its visual URDF from an explicit
`--ai-worker-root`, `--urdf`, or the cached public ROBOTIS repository. If none
is available, the helper downloads the public repository once. The headless
profile uses EmbodiK's bundled reduced model and needs no visual assets.

## Run a headless profile

```bash
pixi run -e cuda demo-parallel-tracking-benchmark
```

The JSON result records the source revision, model and task shape, actual CUDA
device, warm-up count, p50/p95/mean solve time, throughput, and measurement
scope. It excludes target generation, visualization, collision, host
publication, and physics stepping.

The first launch for a model/layout compiles and caches Newton and Warp kernels,
so startup can be substantially slower than later runs. The AI Worker profile
may also fetch its public visual assets on first use. Judge steady-state motion
and timing only after warm-up.

## Reference profile

CUDA-capable NVIDIA GPU with approximately 24 GB of device memory, 1,024
CUDA-resident worlds, two solver iterations, 20 warm-up steps, 50 measured
steps:

| Model | Active DoF | 6D tasks / world | p50 | p95 | Mean throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| Franka Panda | 7 | 1 | 0.906 ms | 0.924 ms | 1.13M worlds/s |
| ROBOTIS AI Worker SG2 | 15 | 2 | 4.854 ms | 4.875 ms | 211k worlds/s |
| Unitree G1 | 29 | 4 | 20.295 ms | 20.408 ms | 50.4k worlds/s |

These are warm CUDA solve-only results, not end-to-end simulation timings.
Task count and active dimension materially affect latency, so Panda's sub-ms
result should not be generalized to the larger whole-body profiles.

## Use the pattern in RL training

Keep the training loop headless and call `solve_device_batch()` with the full
environment batch. State, measured velocity, target poses, and returned commands
should remain CUDA tensors; avoid converting each environment through NumPy or
calling `.cpu()` inside the control loop. Create and warm one fixed-shape solver
before rollout, then pass `reset_mask` and `valid_mask` into
`solve_device_batch()` so finished or invalid worlds reset or hold independently.

Viser in this example is intentionally optional publication. During training,
publish a small rotating sample of environments at a lower rate—or disable the
viewer entirely. The same rule applies to mjviser: use it to inspect selected
MuJoCo worlds, while batching any unavoidable MuJoCo-to-CUDA state transfer once
per control tick. See [GPU WBC: RL and simulator integration](../gpu_solvers.md#rl-and-simulator-integration)
for the device-resident API pattern and actuator choices.
