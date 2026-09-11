# S42: GPU centroidal parity

## Outcome

The public model-derived GPU WBC adapters now expose capture-point constraints,
explicit-state velocity-ZMP constraints, and centroidal momentum tasks and
bounds for fixed- and floating-base robots. The implementation derives its
dimensions and active-coordinate mapping from the loaded robot model.

The public capability contract advertises these features only because both
constructors route their fixed capacities and initial settings into the GPU
core. Shape-stable values can be changed at runtime. Momentum priority and
excluded source-velocity indices remain construction-time layout choices and
fail explicitly if a caller tries to change them in place.

## State and safety contract

- `previous_velocity` remains accepted-command history for command-change
  limits.
- `current_velocity` is a separate measured tangent state used by velocity-ZMP.
- Enabling velocity-ZMP without a finite measured state fails closed; command
  history is never substituted for measurement.
- Capture-point, ZMP, and momentum diagnostics are included in compact host
  publication, and infeasible centroidal constraints publish a safe-hold
  status.
- GPU examples do not invoke the CPU solver as a fallback after a GPU fault.

## Public examples

Example 04 reserves fixed row capacity for all three centroidal features, keeps
them disabled during warm-up, and then routes the live support polygon, margin,
momentum weight, and measured joint velocity on each GPU step.

The shared bimanual application follows the same pattern. Its robot-specific
joint count is not encoded in the example: measured velocity is selected using
the solver's model-derived active velocity order.

## Validation

- Pure tensor row tests cover capture point, exact affine velocity-ZMP,
  positive normal-force floor, momentum bounds, arbitrary leading batch
  dimensions, and varying velocity dimensions.
- Public API tests cover feature discovery, separate measured-state routing,
  runtime centroidal controls, and explicit rejection of layout-changing
  momentum updates.
- CUDA integration tests compare model-derived centroidal quantities against
  the CPU implementation and exercise fixed and floating coordinate mappings.
- An end-to-end unseen five-joint model test exercises a four-joint active
  subset against full-model centroidal state. It covers explicit velocity
  mapping, CUDA graph capture/replay, feasibility publication, and the compact
  float32 readback contract.

On the RTX PRO 5000 Blackwell laptop GPU, a graph-replayed Panda solve with one
pose task plus CoM, capture point, velocity-ZMP, and horizontal momentum
damping measured 1.59 ms total at batch 1 and 2.78 ms total at batch 1024
(2.71 microseconds per world). These are device-batch timings after warm-up;
interactive host conversion and readback are intentionally excluded. The
analytic centroidal bias component alone measured about 0.13 ms at batch 1 and
0.44 ms total at batch 1024.

## Remaining boundary

The centroidal support frame is currently world-aligned in the GPU API. General
structurally root-fixed support-frame transforms remain a follow-up parity item.
The acceleration-history limiter is also distinct from the acceleration-level
WBC solver planned for S43.
