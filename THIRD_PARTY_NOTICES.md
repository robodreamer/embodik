# Third-Party Notices

EmbodiK is licensed under Apache-2.0. EmbodiK links against and, in binary
wheel distributions, may bundle native libraries from third-party projects.
Those projects remain under their own licenses.

This file is an attribution and redistribution checklist for packaged native
dependencies. The license texts for the explicitly listed native projects are
included under `third_party_licenses/`.

## Pinocchio

- Project: Pinocchio
- Upstream: https://github.com/stack-of-tasks/pinocchio
- License: BSD-2-Clause
- Role in EmbodiK: rigid-body model, kinematics, dynamics, geometry placement,
  Lie-group configuration operations, and CMake build dependency discovery.

The upstream license text is bundled as
`third_party_licenses/PINOCCHIO_LICENSE.txt`.

## Coal / HPP-FCL

- Project: Coal, formerly HPP-FCL
- Upstream: https://github.com/coal-library/coal
- License: BSD-style permissive license with non-endorsement terms
- Role in EmbodiK: collision geometry and distance queries through Pinocchio's
  geometry/collision interfaces.

The upstream license text is bundled as
`third_party_licenses/COAL_LICENSE.txt`.

## Other Native Dependencies

Depending on the platform and wheel repair output, EmbodiK wheels may also
include native libraries from Pinocchio's dependency stack, such as Boost,
Assimp, Qhull, OctoMap, TinyXML-2, console_bridge, or URDFDOM libraries. The
current bundled-native dependency inventory is packaged as
`third_party_licenses/BUNDLED_NATIVE_DEPENDENCIES_NOTICE.txt`.

Before publishing a wheel, inspect the repaired artifact with `auditwheel show`
on Linux or `delocate-listdeps`/`otool -L` on macOS. Treat publication as
blocked until every bundled library is covered by this notice and by the
required license text in `third_party_licenses/` or in the repaired wheel's
metadata.

## Example Spot Models

- Project: MuJoCo Menagerie Boston Dynamics Spot MJCF asset
- Upstream: https://github.com/google-deepmind/mujoco_menagerie/tree/main/boston_dynamics_spot
- License: BSD-3-Clause
- Role in EmbodiK: bundled MuJoCo/mjviser Spot-with-arm model for the
  locomanipulation example.

The upstream license text is included at
`examples/assets/spot_mjcf/LICENSE`.

- Project: RAI Open Source `spot_description`
- Upstream: https://github.com/rai-opensource/spot_description
- License: MIT and BSD-3-Clause notices in the upstream package
- Role in EmbodiK: bundled Spot-with-arm URDF and meshes for the regular Viser
  Spot full-body IK example and optional EmbodiK IK overlay.

The upstream license text is included at
`examples/assets/spot_description/LICENSE`.
