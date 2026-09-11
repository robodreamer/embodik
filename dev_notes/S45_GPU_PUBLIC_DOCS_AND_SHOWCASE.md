# S45: GPU public documentation and scale-out showcase

## Outcome

The public GPU story now centers on the model-derived Newton/Warp WBC runtime
instead of the older Panda-only synthetic-Jacobian/CusADi demonstration.

`examples/parallel_trajectory_tracking.py` uses the same solver factory for
Panda, ROBOTIS AI Worker, and Unitree G1. Active coordinates and task rows are
derived from each loaded `RobotModel`; the common solve and visualization loop
contains no robot-specific DoF constants.

## Reference profile

Hardware: NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU. Each run used
1,024 device-resident worlds, two solver iterations, 20 warm-up steps, and 50
measured CUDA-event samples.

| Model | Active DoF | 6D tasks / world | p50 | p95 | Mean throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| Franka Panda | 7 | 1 | 0.906 ms | 0.924 ms | 1.13M worlds/s |
| ROBOTIS AI Worker SG2 | 15 | 2 | 4.854 ms | 4.875 ms | 211k worlds/s |
| Unitree G1 | 29 | 4 | 20.295 ms | 20.408 ms | 50.4k worlds/s |

The measurement is warm CUDA solve-only latency. It excludes target
generation, collision, host publication, visualization, and physics. The
result demonstrates throughput scaling; it does not imply every constrained
whole-body profile is sub-millisecond.

## Public documentation changes

- README and docs home add the three-model hero capture and qualified metrics.
- `docs/gpu_solvers.md` is the canonical Newton/Warp setup, API, capability,
  failure, and performance guide.
- installation and guide navigation now distinguish model-derived GPU WBC from
  legacy CusADi experiments.
- the parallel example page documents viewer and machine-readable headless use.
- media provenance records source revision, assets, hardware, scope, and edits.

## Remaining boundaries

- CPU remains the recommended minimum-latency path for one interactive robot.
- Collision, centroidal constraints, and browser rendering were not enabled in
  the reference scale-out profile and need separate, named benchmarks.
- The acceleration-level GPU API remains a smaller fixed-base state-box slice;
  it is not feature-equivalent to velocity WBC.
