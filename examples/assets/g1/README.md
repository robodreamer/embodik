# Unitree G1 Assets

This directory contains the G1 assets used by
`examples/07_unitree_g1_retargeting_ik.py`.

- `visual/` is a minimal subset copied from
  [`unitreerobotics/unitree_ros`](https://github.com/unitreerobotics/unitree_ros),
  containing only `g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf` and the mesh
  files directly referenced by that URDF.
- `generated/` contains EmbodiK's generated primitive collision URDF for the
  same model. The example uses this for IK/collision by default so interactive
  collision checks avoid high-poly mesh distance costs.

The upstream Unitree assets are distributed under BSD-3-Clause. The license
text is included at `visual/LICENSE`.
