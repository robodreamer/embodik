"""Transform correctness tests for embodiK Rotation, r2q, q2r, and SE3."""

import numpy as np
import pytest

import embodik as eik

TOL = 1e-10


# =============================================================================
# r2q / q2r round-trips and quaternion order
# =============================================================================


def test_r2q_q2r_roundtrip_wxyz():
    """Round-trip: R -> r2q (wxyz) -> q2r -> R."""
    R = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    q = eik.r2q(R, order="sxyz")
    R2 = eik.q2r(q, order="sxyz")
    np.testing.assert_allclose(R, R2, atol=TOL)


def test_r2q_q2r_roundtrip_xyzw():
    """Round-trip: R -> r2q (xyzw) -> q2r -> R."""
    R = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    q = eik.r2q(R, order="xyzs")
    R2 = eik.q2r(q, order="xyzs")
    np.testing.assert_allclose(R, R2, atol=TOL)


def test_r2q_identity_wxyz():
    """Identity matrix -> [1,0,0,0] in wxyz."""
    R = np.eye(3)
    q = eik.r2q(R, order="sxyz")
    np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=TOL)


def test_r2q_identity_xyzw():
    """Identity matrix -> [0,0,0,1] in xyzw."""
    R = np.eye(3)
    q = eik.r2q(R, order="xyzs")
    np.testing.assert_allclose(q, [0.0, 0.0, 0.0, 1.0], atol=TOL)


def test_q2r_identity_wxyz():
    """[1,0,0,0] (wxyz) -> identity matrix."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    R = eik.q2r(q, order="sxyz")
    np.testing.assert_allclose(R, np.eye(3), atol=TOL)


def test_q2r_identity_xyzw():
    """[0,0,0,1] (xyzw) -> identity matrix."""
    q = np.array([0.0, 0.0, 0.0, 1.0])
    R = eik.q2r(q, order="xyzs")
    np.testing.assert_allclose(R, np.eye(3), atol=TOL)


def test_quaternion_order_consistency():
    """r2q and q2r produce consistent results for both orders."""
    R = eik.SO3.Rz(np.pi / 4).as_matrix()
    q_wxyz = eik.r2q(R, order="sxyz")
    q_xyzw = eik.r2q(R, order="xyzs")
    R_wxyz = eik.q2r(q_wxyz, order="sxyz")
    R_xyzw = eik.q2r(q_xyzw, order="xyzs")
    np.testing.assert_allclose(R, R_wxyz, atol=TOL)
    np.testing.assert_allclose(R, R_xyzw, atol=TOL)
    # Check quat formats: wxyz[0]=w, xyzw[3]=w
    assert abs(q_wxyz[0] - q_xyzw[3]) < TOL
    assert abs(q_wxyz[1] - q_xyzw[0]) < TOL
    assert abs(q_wxyz[2] - q_xyzw[1]) < TOL
    assert abs(q_wxyz[3] - q_xyzw[2]) < TOL


# =============================================================================
# Rotation class
# =============================================================================


def test_rotation_identity():
    """Rotation.identity() is identity."""
    R = eik.Rotation.identity()
    np.testing.assert_allclose(R.as_matrix(), np.eye(3), atol=TOL)


def test_rotation_from_matrix_roundtrip():
    """from_matrix -> as_matrix round-trip."""
    R_orig = eik.SO3.Rx(np.pi / 3).as_matrix()
    R = eik.Rotation.from_matrix(R_orig)
    np.testing.assert_allclose(R.as_matrix(), R_orig, atol=TOL)


def test_rotation_from_quat_roundtrip():
    """from_quat -> as_quat round-trip (xyzw)."""
    q_orig = np.array([0.5, 0.5, 0.5, 0.5])
    q_orig = q_orig / np.linalg.norm(q_orig)
    R = eik.Rotation.from_quat(q_orig)
    q = R.as_quat()
    np.testing.assert_allclose(np.abs(q), np.abs(q_orig), atol=TOL)


def test_rotation_from_rotvec_roundtrip():
    """from_rotvec -> as_rotvec round-trip."""
    omega = np.array([0.1, 0.2, 0.3])
    R = eik.Rotation.from_rotvec(omega)
    omega2 = R.as_rotvec()
    # Rotvec may differ by sign for 2*pi equivalence; check rotation matrix
    R2 = eik.Rotation.from_rotvec(omega2)
    np.testing.assert_allclose(R.as_matrix(), R2.as_matrix(), atol=TOL)


def test_rotation_inv():
    """R * R.inv() = identity."""
    R = eik.SO3.Ry(np.pi / 4)
    R_inv = R.inv()
    composed = R * R_inv
    np.testing.assert_allclose(composed.as_matrix(), np.eye(3), atol=TOL)


def test_rotation_composition():
    """(R1 * R2).apply(v) == R1.apply(R2.apply(v))."""
    R1 = eik.SO3.Rx(np.pi / 4)
    R2 = eik.SO3.Rz(np.pi / 6)
    v = np.array([1.0, 0.0, 0.0])
    composed = R1 * R2
    v1 = composed.apply(v)
    v2 = R1.apply(R2.apply(v))
    np.testing.assert_allclose(v1, v2, atol=TOL)


def test_rotation_apply():
    """apply rotates vector correctly (Rz(pi/2) maps x to y)."""
    R = eik.SO3.Rz(np.pi / 2)
    v = np.array([1.0, 0.0, 0.0])
    v_rot = R.apply(v)
    np.testing.assert_allclose(v_rot, [0.0, 1.0, 0.0], atol=TOL)


def test_rotation_from_euler_xyz():
    """from_euler('xyz', [r,p,y]) matches RPY."""
    rpy = np.array([0.1, 0.2, 0.3])
    R1 = eik.Rotation.from_euler("xyz", rpy)
    R2 = eik.SO3.RPY(rpy, order="xyz")
    np.testing.assert_allclose(R1.as_matrix(), R2.as_matrix(), atol=TOL)


def test_rotation_Rx_Ry_Rz():
    """Rx, Ry, Rz produce correct single-axis rotations."""
    theta = np.pi / 2
    Rx = eik.SO3.Rx(theta)
    Ry = eik.SO3.Ry(theta)
    Rz = eik.SO3.Rz(theta)
    np.testing.assert_allclose(Rx.apply([1, 0, 0]), [1, 0, 0], atol=TOL)
    np.testing.assert_allclose(Ry.apply([1, 0, 0]), [0, 0, -1], atol=TOL)
    np.testing.assert_allclose(Rz.apply([1, 0, 0]), [0, 1, 0], atol=TOL)


def test_rotation_AngVec():
    """AngVec(theta, axis) matches from_rotvec(theta * axis)."""
    theta = 0.5
    axis = np.array([1.0, 0.0, 0.0])
    R1 = eik.SO3.AngVec(theta, axis)
    R2 = eik.Rotation.from_rotvec(theta * axis)
    np.testing.assert_allclose(R1.as_matrix(), R2.as_matrix(), atol=TOL)


def test_rotation_EulerVec():
    """EulerVec is alias for from_rotvec."""
    omega = np.array([0.1, 0.2, 0.3])
    R1 = eik.SO3.EulerVec(omega)
    R2 = eik.Rotation.from_rotvec(omega)
    np.testing.assert_allclose(R1.as_matrix(), R2.as_matrix(), atol=TOL)


def test_SO3_alias():
    """SO3 is alias for Rotation."""
    assert eik.SO3 is eik.Rotation
    R = eik.SO3.identity()
    assert isinstance(R, eik.Rotation)


def test_rotation_R_property():
    """R property returns rotation matrix."""
    R = eik.SO3.Rz(np.pi / 4)
    np.testing.assert_allclose(R.R, R.as_matrix(), atol=TOL)


# =============================================================================
# SE3 aliases (R, t, A)
# =============================================================================


def test_se3_Rt_classmethod():
    """SE3.Rt(R, t) creates correct transform."""
    R = np.eye(3)
    t = np.array([1.0, 2.0, 3.0])
    T = eik.SE3.Rt(R, t)
    np.testing.assert_allclose(T.rotation, R, atol=TOL)
    np.testing.assert_allclose(T.translation, t, atol=TOL)


def test_se3_R_property():
    """SE3.R returns rotation matrix."""
    R = np.eye(3)
    t = np.array([1.0, 2.0, 3.0])
    T = eik.Rt(R=R, t=t)
    np.testing.assert_allclose(T.R, R, atol=TOL)


def test_se3_t_property():
    """SE3.t returns translation."""
    R = np.eye(3)
    t = np.array([1.0, 2.0, 3.0])
    T = eik.Rt(R=R, t=t)
    np.testing.assert_allclose(T.t, t, atol=TOL)


def test_se3_A_property():
    """SE3.A returns 4x4 homogeneous matrix."""
    R = np.eye(3)
    t = np.array([1.0, 2.0, 3.0])
    T = eik.Rt(R=R, t=t)
    A = T.A
    assert A.shape == (4, 4)
    np.testing.assert_allclose(A[:3, :3], R, atol=TOL)
    np.testing.assert_allclose(A[:3, 3], t, atol=TOL)
    np.testing.assert_allclose(A[3, :], [0, 0, 0, 1], atol=TOL)
    np.testing.assert_allclose(A, T.homogeneous(), atol=TOL)


# =============================================================================
# r2q/q2r vs Rotation consistency
# =============================================================================


def test_r2q_matches_rotation_as_quat_xyzw():
    """r2q(R, order='xyzs') matches Rotation.from_matrix(R).as_quat()."""
    R = eik.SO3.Ry(np.pi / 6).as_matrix()
    q_r2q = eik.r2q(R, order="xyzs")
    q_rot = eik.Rotation.from_matrix(R).as_quat()
    np.testing.assert_allclose(np.abs(q_r2q), np.abs(q_rot), atol=TOL)


def test_q2r_matches_rotation_as_matrix():
    """q2r(q, order='xyzs') matches Rotation.from_quat(q).as_matrix()."""
    q = np.array([0.5, 0.5, 0.5, 0.5])
    q = q / np.linalg.norm(q)
    R_q2r = eik.q2r(q, order="xyzs")
    R_rot = eik.Rotation.from_quat(q).as_matrix()
    np.testing.assert_allclose(R_q2r, R_rot, atol=TOL)
