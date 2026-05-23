# Bundled Spot URDF Asset

This directory packages the Spot-with-arm URDF/mesh set used by the Spot IK
examples. `urdf/spot_with_arm.urdf` is the stable packaged path consumed by the
examples.

It is used by `examples/08_spot_full_body_ik_viser.py` and by the optional
EmbodiK IK overlay in `examples/09_spot_locomanip_mjviser.py`.

The controller keeps using the short policy/MJCF joint names internally, which
match this packaged URDF.
