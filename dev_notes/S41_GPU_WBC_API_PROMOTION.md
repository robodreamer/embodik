# S41 — GPU WBC API promotion and stack migration

Date: 2026-09-10

## Objective

Expose the reusable GPU WBC implementation as an EmbodiK package API. Preserve
model-derived specialization, make the supported model envelope explicit, and
prepare the work as the next layer above the existing remote PR stack.

## Repository and PR layout

The existing open stack is, bottom to top:

1. `release/0.21.0-prep` — PR #9
2. `refactor/solver-modularization-0.21.1` — PR #10
3. `integration/centroidal-stability-0.22.0` — PR #11
4. `feat/gpu-wbc-api` — this GPU WBC layer

The first three branches were created with Graphite. The repository has
`github/gh-stack` v0.1.0 installed, so S41 adopts the existing branches with
`gh stack init` and places this branch above PR #11. No PR is submitted during
this session.

## Promoted API

The source of truth is now `python/embodik/gpu/wbc`:

- `GpuWbcMultiFrameSolver` for fixed-base models;
- `GpuWbcFloatingMultiFrameSolver` for one-free-root models;
- `from_robot(...)` constructors that need no generated manifest for native
  Torch, Warp, or cuSOLVER SRINV;
- model-derived `RobotSolveSpec`, joint spans, frame names, collision geometry,
  limits, and active-coordinate mappings;
- `solve_device_batch(...)` for a device-resident batch with persistent
  accepted-velocity history;
- batch-size-independent Warp specialization and an explicit B=1 cuSOLVER
  capability error.

The public examples import `embodik.gpu.wbc`. The former 2,136-line example
adapter and the Panda/CusADi horizon smoke mode were removed. A manifest remains
an optional compatibility validation input; only legacy FI-PeSNS requires its
compiled artifact.

## Generality contract

There is no dispatch on Panda, Spot, G1, RB-Y1, IIWA, or any known DoF count in
the generalized multi-frame path. Fixed shapes are compiled from the loaded
model and requested task/constraint layout.

The currently supported model envelope is intentionally narrower than “any
Pinocchio model”:

- fixed-base robots with scalar one-DoF movable joints;
- one standard 7-coordinate/6-velocity free root plus scalar joints;
- body/link frame position and pose tasks;
- overdetermined task matrices through directional SRINV;
- arbitrary Warp batch capacity, subject to configured backend row/column and
  collision-query capacities;
- cuSOLVER remains B=1 and limited by its small-matrix specialization.

Unsupported multi-coordinate non-root joints must fail at construction. The
factory defaults to all supported movable coordinates rather than only joints
visible in the primary Jacobian, because collision recovery, posture, CoM,
torso, and secondary tasks may require otherwise invisible joints.

## Regressions added

- An unseen five-DoF branched fixed model mock with deliberately unfamiliar
  joint/frame names validates order, dimensions, limits, and factory behavior.
- An unseen floating model mock validates a 7/6 root, non-contiguous task
  influence, and full-coordinate factory selection.
- A rectangular 6-by-5 directional-SRINV regression covers overdetermined
  tasks.
- A device-resident B=8 regression checks accepted-velocity history without a
  host round trip.
- Existing public example feature tests remain the cross-model regression set
  for Panda, dual IIWA, bimanual RB-Y1/AI Worker, Unitree G1, and Spot.

On the RTX PRO 5000 Blackwell test GPU, a warmed B=1024 Panda Warp solve with
distinct tool-X targets completed in 1.084 ms. All 1024 outputs were finite and
the maximum joint spread was 0.026661 rad, confirming that worlds were not
collapsed to one trajectory. This is a pose-only throughput check, not an
all-constraints latency claim.

## Remaining work after S41

- Add generated URDF + real Newton CUDA regressions for unseen fixed and
  floating models; current unseen-model tests isolate the API/model contract.
- Derive Newton body/frame correspondence for every EmbodiK frame type rather
  than accepting a frame that Newton cannot resolve.
- Remove the private legacy Panda branches still retained inside the generic
  collision compatibility module.
- Lift cuSOLVER beyond B=1 or select Warp automatically for larger batches.
- Add checked-in B=1024 feature-composition regressions beyond the pose-only
  validation performed in S41.

## Extended feature-parity goals

`GPU_WBC_CAPABILITIES` is the executable source of truth for controls exposed
by applications. Examples must disable an unsupported control; they must not
ignore it or invoke the CPU implementation while reporting GPU attribution.

The follow-on sessions should close the remaining CPU/GPU semantic gaps in
dependency order:

1. Add GPU capture-point halfspaces, explicit-state velocity ZMP halfspaces,
   and centroidal momentum objectives using the model-derived centroidal terms
   introduced by the 0.22.0 stack layer.
2. Add acceleration-level task composition and bounds. Distinguish it clearly
   from the currently supported velocity solver's acceleration-history limit.
3. Generalize task axis masks and joint metrics, then exercise mixed position,
   orientation, contact, and secondary-task layouts on unseen models.
4. Match CPU collision policy semantics: tuning presets, structural
   non-worsening/recovery rules, continuous certification, pair updates, and
   overflow diagnostics.
5. Run a fixed feature-by-model matrix for Panda, IIWA, RB-Y1/AI Worker,
   Unitree G1, Spot, one generated fixed-base URDF, and one generated
   floating-base URDF at B=1 and B=1024 where applicable.

Completion requires correctness, fail-closed behavior, CPU-reference parity,
and measured warm steady-state latency. A checkbox appearing in an example is
not evidence of support by itself.
