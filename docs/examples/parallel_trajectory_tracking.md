# Parallel GPU WBC Showcase

`examples/parallel_trajectory_tracking.py` solves many independently targeted
robot worlds through the public model-derived GPU WBC API. It replaces the
earlier Panda-only synthetic-Jacobian/CusADi demo.

<video autoplay muted loop playsinline controls width="100%" aria-label="Panda, ROBOTIS AI Worker, and Unitree G1 moving in parallel Viser worlds while 1,024 independent worlds are solved on CUDA" src="../../assets/media/gpu_wbc_parallel_showcase.mp4"></video>

## What it demonstrates

- real Newton forward kinematics and Jacobians for every world;
- Warp directional SRINV with joint position and velocity bounds;
- one generalized solver surface for Panda, AI Worker, and G1;
- dimensions and active joints derived from `RobotModel`, not hard-coded DoF;
- circle, figure-eight, helix, and sweep targets with independent per-world
  phases and speeds;
- a live 32×32 point map for all 1,024 worlds plus detailed robots sampled
  across the full solve batch;
- CUDA-event timing with explicit no-fallback attribution checks.

The profiles intentionally differ: Panda has one moving 6D hand task, AI
Worker has two moving 6D tool tasks, and G1 has two moving hand tasks plus two
anchored 6D foot tasks.

## Run the viewer

Install the [GPU WBC requirements](../gpu_solvers.md#requirements), then select
a model:

```bash
python examples/parallel_trajectory_tracking.py --robot panda --worlds 1024
python examples/parallel_trajectory_tracking.py --robot ai-worker --worlds 1024
python examples/parallel_trajectory_tracking.py --robot g1 --worlds 1024
```

The default viewer renders a colored point for every world and nine detailed
URDFs sampled across the full batch. Labels and trajectory splines identify the
motion family assigned to each detailed world. Change the detailed count with
`--show`; reducing it improves browser responsiveness without changing the
solver batch or the all-world map.

The AI Worker viewer resolves its visual URDF from an explicit
`--ai-worker-root`, `--urdf`, or the cached public ROBOTIS repository. If none
is available, the helper downloads the public repository once. The headless
profile uses EmbodiK's bundled reduced model and needs no visual assets.

## Run a headless profile

```bash
python examples/parallel_trajectory_tracking.py \
  --robot panda \
  --worlds 1024 \
  --headless \
  --warmup-steps 20 \
  --steps 50 \
  --output-json build/panda-gpu-wbc.json
```

The JSON result records the source revision, model and task shape, actual CUDA
device, warm-up count, p50/p95/mean solve time, throughput, and measurement
scope. It excludes target generation, visualization, collision, host
publication, and physics stepping.

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
