# Demo media provenance

Public demo media is captured from working EmbodiK examples. New media should
record its executable source, capture state, and transformations here so public
claims can be reproduced.

## `gpu_wbc_parallel_showcase.mp4`

- **Source:** `examples/parallel_trajectory_tracking.py` at commit `4917bf7`
- **Capture date:** 2026-09-11
- **Capture owner and permission:** captured locally by the EmbodiK maintainer
  from repository-owned example code and redistributable public robot assets;
  approved for this repository's documentation
- **Runtime:** Newton 1.6 development build, Warp 1.17.0, Torch 2.12.1 CUDA
- **Hardware:** NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU
- **Profile:** 1,024 CUDA-resident worlds, two solver iterations, nine visible
  worlds, Panda then ROBOTIS AI Worker SG2 then Unitree G1
- **Solver scope:** Newton batched kinematics plus Warp directional SRINV;
  collision disabled; target generation, Viser rendering, host publication,
  and physics stepping excluded from displayed solve time
- **Capture method:** authenticated browser screenshots of the live Viser
  examples, 24 frames per model
- **Edit history:** top-cropped from 1308×957, resized to 1280×720, concatenated
  in model order, encoded as 8 fps H.264/yuv420p with no audio
- **Output:** 9.0 seconds, 1280×720, approximately 347 KiB

The companion `gpu_wbc_parallel_showcase_preview.gif` is derived from the MP4
at 8 fps and 960×540 with a 128-color palette for GitHub README playback.
