# Unitree G1 Phase1 Ports (G1 -> EmbodiK)

These scripts are **behavioral counterparts** of the original G1 notebooks,
adapted to EmbodiK-style Viser examples (not literal notebook ports).

## Parity Matrix

| G1 source notebook | EmbodiK phase1 script | Parity focus |
|---|---|---|
| `ri_motion_v5_notebook/03_forward_kinematics/01_fk_g1_inspire.ipynb` | `04_g1_ik_site_counterpart_viser.py` | end-effector/site target manipulation and pose response loop |
| `ri_motion_v5_notebook/04_inverse_kinematics/02_ik_site_g1_inspire.ipynb` | `04_g1_ik_site_counterpart_viser.py` | 3-point surrogate orientation behavior (`IK_R`) and interactive target updates |
| `ri_motion_v5_notebook/04_inverse_kinematics/05_ik_base_g1.ipynb` | `05_g1_base_ik_counterpart_viser.py` | full-body multi-target IK, feet constraints, torso/hand target handling |
| collision/contact exploration flows in G1 G1 notebooks | `06_g1_collision_constraint_counterpart_viser.py` | collision distance constraint tuning + closest-pair diagnostics |
| reimagined EmbodiK-style G1 flow (02 + 08 patterns) | `07_g1_dual_hand_grounded_com_viser.py` | dual hand 6D gizmos + grounded floating base + CoM support polygon |

## Intentional EmbodiK-Style Upgrades

- Replace slider-only notebook interaction with Viser transform controls + grouped GUI folders.
- Keep both interaction styles available:
  - G1-style 3-point goal mode
  - EmbodiK native 6D mode
- Use floating-base full-body formulation for base IK counterpart.
- Add CoM support-polygon controls and visualization in full-body example.
- Use manifold-safe integration and robust fallback behavior in collision-constrained example.

## Retargeting Examples

The phase1 scripts include preset-driven retargeting examples via
`g1_viser_utils.get_retargeting_presets()` and
`g1_viser_utils.retarget_position_from_anchor(...)`.

## New Grounded Dual-Hand Demo

`07_g1_dual_hand_grounded_com_viser.py` grounds the floating base by shifting it
by negative feet-center, anchors both feet with solver-native tight 6D
epsilon-box constraints, and lets you control both hands with independent 6D
gizmos while monitoring CoM constraints.

Default strict profile used in tests:
- position residual target `<= 1e-5 m`
- orientation residual target `<= 1e-4 rad`

