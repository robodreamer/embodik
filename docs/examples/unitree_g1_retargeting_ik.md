# Unitree G1 Retargeting IK Example

`examples/13_unitree_g1_retargeting_ik.py` is a Viser-based whole-body IK demo
for Unitree G1. It exposes palm, foot, and pelvis target frames, synthetic
retargeting playback, CoM support-polygon visualization, and optional
self-collision constraints.

<video autoplay muted loop playsinline controls width="100%" src="../../assets/media/unitree_g1_retargeting_ik.mp4"></video>

## What It Covers

- Five interactive transform controls: right palm, left palm, right foot, left
  foot, and pelvis.
- Per-target enable checkboxes so a subset of target frames can drive IK.
- Synthetic retargeting clips that reuse the same IK path as interactive gizmo
  control.
- Optional CoM support-polygon constraint with combined CoM marker, drop line,
  and support polygon visualization.
- Optional self-collision constraints with debug closest-pair visualization.
- Reduced IK model by default so hand/finger joints do not dominate solve cost.

## Run It

Install and copy the example bundle once using the
[Installation Guide](../installation.md#examples). Then run:

```bash
cd embodik_examples
python 13_unitree_g1_retargeting_ik.py
```

From a repository checkout, use Pixi:

```bash
pixi run python examples/13_unitree_g1_retargeting_ik.py
```

## Assets

The example resolves the visual G1 URDF from one of these locations:

- `EMBODIK_G1_URDF`
- known local Unitree ROS checkout paths

The reduced box-collision URDF used for collision-aware IK is bundled under
`examples/assets/g1/generated/` and is used by default. Override it with
`EMBODIK_G1_COLLISION_URDF` when testing a different collision model.

## Headless Checks

The public example keeps smoke-test flags for regression coverage:

```bash
python 13_unitree_g1_retargeting_ik.py --headless-smoke-steps 4
python 13_unitree_g1_retargeting_ik.py --headless-com-smoke
python 13_unitree_g1_retargeting_ik.py --headless-ik-smoke-steps 20 --headless-reset-interval 8
```

The performance harness lives separately:

```bash
python harnesses/g1_four_gizmo_ik_benchmark.py --max-steps 2 --reset-interval 8 --quiet
```
