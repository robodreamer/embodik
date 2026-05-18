# Bundled Spot URDF Asset

This directory is copied from the public `spot_description` package:

```text
https://github.com/rai-opensource/spot_description
```

`urdf/spot_with_arm.urdf` was generated from `spot.urdf.xacro` with `arm:=true`
and `feet:=true`, then rewritten so mesh paths are relative to this packaged
directory. It is used by `examples/14_spot_full_body_ik_viser.py` and by the
optional EmbodiK IK overlay in `examples/15_spot_locomanip_mjviser.py`.

The controller keeps using the short policy/MJCF joint names internally; the
resolver maps those names to this URDF's public `spot_description` joint names
where they differ.
