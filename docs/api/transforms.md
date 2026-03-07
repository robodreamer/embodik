# Transforms

Native spatial transform helpers for SO(3) rotations and SE(3) rigid-body transforms. Replaces `scipy.spatial.transform` with Pinocchio-backed operations (no SciPy dependency).

## Rotation (SO3)

Lightweight SO(3) rotation helper. Canonical methods are primary; spatialmath-style shorthands are convenience aliases.

::: embodik.Rotation
    options:
      show_root_heading: true
      show_root_toc_entry: true
      members:
        - from_matrix
        - from_quat
        - from_rotvec
        - from_euler
        - identity
        - Rx
        - Ry
        - Rz
        - RPY
        - AngVec
        - EulerVec
        - as_matrix
        - as_quat
        - as_rotvec
        - inv
        - apply
        - R

## SO3 Alias

`SO3` is an alias for `Rotation` (spatialmath-style):

```python
from embodik import SO3
R = SO3.Rx(np.pi/2)
R = SO3.RPY([roll, pitch, yaw], order='xyz')
```

## SE3

Rigid-body transform (rotation + translation). Supports composition, point transforms, and spatialmath-style property aliases.

::: embodik.SE3
    options:
      show_root_heading: true
      show_root_toc_entry: true
      members:
        - Rt
        - rotation
        - translation
        - homogeneous
        - R
        - t
        - A
        - inverse
        - act
        - actInv

### SE3 Composition

```python
from embodik import Rt, SE3
T1 = Rt(R=np.eye(3), t=[1, 0, 0])
T2 = Rt(R=np.eye(3), t=[0, 1, 0])
T_composed = T1 * T2  # Pinocchio: ^A M_B * ^B M_C = ^A M_C
```

### SE3 Point Transforms

```python
p_world = T.act(p_local)   # p_world = R @ p_local + t
p_local = T.actInv(p_world)  # p_local = R.T @ (p_world - t)
```

## Low-Level Utilities

Native C++ functions exposed for advanced use:

| Function | Description |
|----------|-------------|
| `log3(R)` | Axis-angle from 3×3 rotation matrix |
| `exp3(omega)` | 3×3 rotation matrix from axis-angle |
| `matrix_to_quaternion_wxyz(R)` | Matrix → [w,x,y,z] quaternion |
| `matrix_to_quaternion_xyzw(R)` | Matrix → [x,y,z,w] quaternion |
| `quaternion_wxyz_to_matrix(w,x,y,z)` | [w,x,y,z] → 3×3 matrix |
| `quaternion_xyzw_to_matrix(x,y,z,w)` | [x,y,z,w] → 3×3 matrix |
| `rotation_from_rpy(r,p,y)` | Roll-pitch-yaw → 3×3 matrix |
